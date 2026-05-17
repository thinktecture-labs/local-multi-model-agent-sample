"""
Voice pipeline routes — local STT (whisper.cpp) + TTS (Piper).

Routes:
  POST /voice/chat       — full voice round-trip (STT → agent → TTS) with SSE
  GET  /voice/audio/{id} — serve generated TTS audio (WAV)
  POST /voice/synthesize — standalone text-to-speech
"""

import asyncio
import collections
import json as _json
import logging
import os
import re as _re
import struct as _struct
import time as _time
import uuid as _uuid
from typing import Any

import regex as _regex  # supports Unicode properties (\p{...})
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response, StreamingResponse

from src.engine.inference.config import (
    PIPELINE_TIMEOUT,
    PIPER_VOICE_EN,
    PIPER_VOICE_DE,
    PIPER_VOICES_DIR,
    AUDIO_CACHE_TTL,
    AUDIO_CACHE_MAX_ENTRIES,
    SCENARIO_CONFIG,
)
from .state import state


# ─── Canned voice responses ──────────────────────────────────────────────────
# Pinned client-facing answers for known triggers. Used for the keynote climax
# where we want the punchline locked instead of trusting Gemma3's variance.
# After STT, if the transcript matches a trigger regex, we emit fake LogReg +
# Gemma3 agent_step events (so the model strip lights up like the normal
# direct-chat path) plus the canned response + TTS audio, and skip the agent
# pipeline entirely.
#
# Disabled by default for forks — set VOICE_CANNED_RESPONSES=1 in .env to
# re-enable the stage-demo punchline path.
_CANNED_VOICE_ENABLED = os.getenv("VOICE_CANNED_RESPONSES", "0").lower() in ("1", "true", "yes", "on")

CANNED_VOICE_RESPONSES: list[tuple[Any, str]] = (
    [
        (
            _re.compile(r"what would i do without", _re.IGNORECASE),
            "Sir, you would not be having this conversation. "
            "You would be paying someone in the cloud to have it for you.",
        ),
    ]
    if _CANNED_VOICE_ENABLED
    else []
)


def _match_canned_response(transcript: str) -> str | None:
    for pattern, response in CANNED_VOICE_RESPONSES:
        if pattern.search(transcript):
            return response
    return None

logger = logging.getLogger(__name__)

router = APIRouter()


class _AudioCache:
    """Bounded LRU cache with TTL for TTS audio bytes.

    - Entries expire after ``ttl`` seconds.
    - When ``maxsize`` is reached, the least-recently-used entry is evicted.
    - ``get()`` promotes the entry to most-recent (true LRU) and lazily
      purges expired items.
    """

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._data: collections.OrderedDict[str, tuple[bytes, float]] = (
            collections.OrderedDict()
        )
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> bytes | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        wav, ts = entry
        if _time.time() - ts >= self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)  # promote to most-recent
        return wav

    def put(self, key: str, wav: bytes) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = (wav, _time.time())
        else:
            self._purge_expired()
            while len(self._data) >= self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)  # evict LRU
                logger.debug("Audio cache evicted LRU entry %s", evicted_key)
            self._data[key] = (wav, _time.time())

    def _purge_expired(self) -> None:
        now = _time.time()
        expired = [k for k, (_, ts) in self._data.items() if now - ts >= self._ttl]
        for k in expired:
            del self._data[k]

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


_audio_cache = _AudioCache(maxsize=AUDIO_CACHE_MAX_ENTRIES, ttl=AUDIO_CACHE_TTL)

# Piper voice models — loaded lazily on first TTS request
_piper_voices: dict[str, Any] = {}  # lang -> PiperVoice instance


def _get_piper_voice(language: str):
    """Get or lazily load a Piper voice model for the given language."""
    if language in _piper_voices:
        return _piper_voices[language]
    from piper import PiperVoice
    voice_name = PIPER_VOICE_DE if language == "de" else PIPER_VOICE_EN
    model_path = os.path.join(PIPER_VOICES_DIR, f"{voice_name}.onnx")
    if not os.path.exists(model_path):
        return None
    _piper_voices[language] = PiperVoice.load(model_path)
    return _piper_voices[language]


def _build_wav_header(pcm_length: int, sample_rate: int = 22050,
                      channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Build a 44-byte RIFF WAV header for raw PCM data."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return _struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + pcm_length, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", pcm_length,
    )


async def _convert_to_wav(audio_bytes: bytes) -> bytes:
    """Convert browser audio (WebM/Opus) to WAV 16kHz 16-bit mono for whisper."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
        "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate(audio_bytes)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail="Audio conversion failed (ffmpeg)")
    return stdout


async def _synthesize_speech(text: str, language: str) -> bytes:
    """Generate WAV speech using Piper TTS. Runs in thread pool (CPU-bound)."""
    voice = _get_piper_voice(language)
    if voice is None:
        raise HTTPException(status_code=503, detail=f"Piper voice not available for '{language}'")

    def _run():
        # Strip markdown + emojis — keep in display text, remove from speech
        clean = text
        try:
            clean = _regex.sub(r'\p{Emoji_Presentation}|\p{Emoji}\uFE0F', '', clean)
        except Exception:
            pass
        clean = _re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', clean)
        clean = _re.sub(r'^#{1,6}\s+', '', clean, flags=_re.MULTILINE)
        clean = _re.sub(r'^\s*[\*\-]\s+', '', clean, flags=_re.MULTILINE)
        clean = _re.sub(r'^\s*\d+\.\s+', '', clean, flags=_re.MULTILINE)
        clean = _re.sub(r'`([^`]+)`', r'\1', clean)
        clean = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
        clean = _re.sub(r'\n{2,}', '. ', clean)
        clean = _re.sub(r'\s+', ' ', clean).strip()
        # Normalize smart quotes/apostrophes to ASCII (Piper espeak needs straight quotes)
        clean = clean.replace('\u2018', "'").replace('\u2019', "'")
        clean = clean.replace('\u201c', '"').replace('\u201d', '"')
        # Strip backslash-escaped quotes from model output (e.g. I\'m → I'm)
        clean = clean.replace("\\'", "'").replace('\\"', '"')
        # Expand currency codes — Piper otherwise reads "EUR" as "ur".
        clean = _re.sub(r'\bEUR\b', 'Euro', clean)
        clean = _re.sub(r'\bUSD\b', 'US dollars', clean)
        clean = _re.sub(r'\bGBP\b', 'pounds', clean)
        audio_chunks = [chunk.audio_int16_bytes for chunk in voice.synthesize(clean)]
        pcm_data = b"".join(audio_chunks)
        header = _build_wav_header(len(pcm_data), sample_rate=voice.config.sample_rate)
        return header + pcm_data

    return await asyncio.to_thread(_run)


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"


@router.post("/voice/chat", tags=["Voice"])
async def voice_chat(file: UploadFile = File(...)):
    """
    Full voice round-trip with SSE streaming at each pipeline stage.

    Flow: transcribe → agent query → TTS synthesis → audio URL
    Each stage emits an SSE event so the browser can show progressive feedback.
    """

    async def event_stream():
        # Stage 1: Transcribe (whisper-server)
        audio_bytes = await file.read()
        wav_bytes = await _convert_to_wav(audio_bytes)
        t0 = _time.perf_counter()
        try:
            result = await state.client.transcribe(wav_bytes)
        except Exception as exc:
            yield _sse_event("error", {"message": f"Transcription failed: {exc}"})
            return
        stt_ms = (_time.perf_counter() - t0) * 1000
        transcript = result.get("text", "").strip()
        language = result.get("language") or SCENARIO_CONFIG.language
        yield _sse_event("transcription", {
            "text": transcript, "language": language, "duration_ms": round(stt_ms, 1),
        })

        if not transcript or transcript.startswith("[BLANK_AUDIO]") or transcript.startswith("(blank"):
            yield _sse_event("error", {"message": "No speech detected"})
            return

        # Stage 2a: Canned-response intercept. If the transcript matches a
        # pinned trigger (e.g. the Demo-9 climax line), skip the agent and
        # emit the locked answer + TTS instead. Fakes a LogReg + Gemma3
        # direct-chat path so the model strip lights up the same way it
        # would for a real direct-chat query.
        canned = _match_canned_response(transcript)
        if canned:
            yield _sse_event("agent_step", {
                "action": "classify_intent", "model": "logreg",
                "duration_ms": 8.0, "tokens_used": 0, "details": {"canned": True},
            })
            yield _sse_event("agent_step", {
                "action": "direct_response", "model": "gemma3",
                "duration_ms": 0.0, "tokens_used": 0, "details": {"canned": True},
            })
            yield _sse_event("response", {
                "text": canned, "intent": "casual_chat", "duration_ms": 8.0,
            })
            try:
                tts_wav = await _synthesize_speech(canned, language)
            except Exception as exc:
                yield _sse_event("audio", {"url": None, "duration_ms": 0, "error": str(exc)})
                return
            audio_id = str(_uuid.uuid4())
            _audio_cache.put(audio_id, tts_wav)
            yield _sse_event("audio", {
                "url": f"/voice/audio/{audio_id}", "duration_ms": 0,
            })
            return

        # Stage 2: Agent query (reuse existing pipeline)
        t1 = _time.perf_counter()
        try:
            agent_resp = await asyncio.wait_for(
                state.agent.process(transcript),
                timeout=PIPELINE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            yield _sse_event("error", {"message": f"Pipeline timed out after {PIPELINE_TIMEOUT}s"})
            return
        except Exception as exc:
            yield _sse_event("error", {"message": f"Agent error: {exc}"})
            return
        agent_ms = (_time.perf_counter() - t1) * 1000
        for step in agent_resp.steps:
            yield _sse_event("agent_step", {
                "action": step.action, "model": step.model,
                "duration_ms": round(step.duration_ms, 1),
                "tokens_used": step.tokens_used,
                "details": step.details,
            })
        yield _sse_event("response", {
            "text": agent_resp.response,
            "intent": agent_resp.intent.value if hasattr(agent_resp.intent, 'value') else str(agent_resp.intent),
            "duration_ms": round(agent_ms, 1),
        })

        # Stage 3: TTS synthesis (Piper, CPU)
        t2 = _time.perf_counter()
        try:
            tts_wav = await _synthesize_speech(agent_resp.response, language)
        except HTTPException:
            yield _sse_event("audio", {"url": None, "duration_ms": 0, "error": "TTS not available"})
            return
        except Exception as exc:
            yield _sse_event("audio", {"url": None, "duration_ms": 0, "error": str(exc)})
            return
        tts_ms = (_time.perf_counter() - t2) * 1000
        audio_id = str(_uuid.uuid4())
        _audio_cache.put(audio_id, tts_wav)
        yield _sse_event("audio", {
            "url": f"/voice/audio/{audio_id}",
            "duration_ms": round(tts_ms, 1),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/voice/audio/{audio_id}", tags=["Voice"])
async def get_voice_audio(audio_id: str):
    """Serve generated TTS audio (WAV). Auto-expires after AUDIO_CACHE_TTL seconds."""
    wav_bytes = _audio_cache.get(audio_id)
    if wav_bytes is None:
        raise HTTPException(status_code=404, detail="Audio expired or not found")
    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/voice/synthesize", tags=["Voice"])
async def voice_synthesize(text: str = "", language: str = SCENARIO_CONFIG.language):
    """Synthesize text to speech (standalone TTS endpoint)."""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    wav_bytes = await _synthesize_speech(text, language)
    return Response(content=wav_bytes, media_type="audio/wav")
