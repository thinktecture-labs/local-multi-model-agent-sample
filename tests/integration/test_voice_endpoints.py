"""
Integration tests for voice endpoints.

Tests HTTP request/response structure for /voice/chat, /voice/audio,
and /voice/synthesize. Uses mocked whisper transcription and Piper TTS
so no external services are needed.
"""

import io
import json

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.engine.agent import AgentResponse, Intent, ExecutionStep
from src.engine.inference.client import SmallLanguageModelRole


# ---------------------------------------------------------------------------
# Build a fully-mocked app for voice tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with mocked lifespan + mocked whisper transcribe."""
    import src.server as server_module

    mock_client = MagicMock()
    mock_client.models = {
        SmallLanguageModelRole.INFERENCE: "gemma3:1b-it",
        SmallLanguageModelRole.FUNCTION:  "qwen",
        SmallLanguageModelRole.EMBEDDING: "embeddinggemma",
        SmallLanguageModelRole.VISION:    "gemma3-4b-vision",
    }
    mock_client.check_health = AsyncMock(return_value={
        "INFERENCE": True, "FUNCTION": True, "EMBEDDING": True, "VISION": True,
        "WHISPER": True,
    })
    # Mock transcribe to return English text
    mock_client.transcribe = AsyncMock(return_value={
        "text": "What were our Q3 results?",
        "language": "en",
    })

    mock_vs = MagicMock()
    mock_vs.count = AsyncMock(return_value=0)
    mock_vs.set_client = MagicMock()

    mock_tools = MagicMock()
    mock_tools.list_tools = MagicMock(return_value=[])
    mock_tools.get_all_schemas = MagicMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.process = AsyncMock(return_value=AgentResponse(
        query="What were our Q3 results?",
        intent=Intent.DIRECT_ANSWER,
        response="Q3 revenue was $2.1M, up 15% year-over-year.",
        steps=[ExecutionStep(action="direct_response", model="gemma3:1b-it",
                             duration_ms=150.0, tokens_used=25)],
        execution_time_ms=150.0,
    ))
    mock_agent.interaction_count = 0
    mock_agent.total_tokens_generated = 0

    @asynccontextmanager
    async def mock_lifespan(app):
        server_module._state.client = mock_client
        server_module._state.vector_store = mock_vs
        server_module._state.tools = mock_tools
        server_module._state.agent = mock_agent
        yield

    original_lifespan = server_module.app.router.lifespan_context
    server_module.app.router.lifespan_context = mock_lifespan

    with TestClient(server_module.app, raise_server_exceptions=False) as tc:
        tc._mock_agent = mock_agent
        tc._mock_client = mock_client
        yield tc

    server_module.app.router.lifespan_context = original_lifespan
    from src.server.voice_routes import _audio_cache
    _audio_cache.clear()


def _fake_wav_bytes():
    """Generate a minimal valid WAV file (44-byte header + 100 bytes PCM)."""
    import struct
    pcm = b"\x00\x01" * 50
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE",
        b"fmt ", 16, 1, 1, 16000, 32000, 2, 16,
        b"data", len(pcm),
    )
    return header + pcm


# ---------------------------------------------------------------------------
# POST /voice/chat — SSE streaming voice round-trip
# ---------------------------------------------------------------------------

class TestVoiceChat:
    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_voice_chat_streams_sse_events(self, mock_tts, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        mock_tts.return_value = _fake_wav_bytes()

        # Simulate uploading a WebM audio file
        audio_blob = b"\x1a\x45\xdf\xa3" + b"\x00" * 100  # fake WebM header
        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(audio_blob), "audio/webm")},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events
        events = []
        for line in resp.text.split("\n"):
            if line.startswith("event: "):
                events.append(line[7:])

        assert "transcription" in events
        assert "response" in events
        assert "audio" in events

    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_voice_chat_transcription_event_has_text(self, mock_tts, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        mock_tts.return_value = _fake_wav_bytes()

        audio_blob = b"\x00" * 100
        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(audio_blob), "audio/webm")},
        )

        # Find the transcription data
        lines = resp.text.split("\n")
        for i, line in enumerate(lines):
            if line == "event: transcription":
                data = json.loads(lines[i + 1][6:])  # "data: {...}"
                assert data["text"] == "What were our Q3 results?"
                assert data["language"] == "en"
                assert "duration_ms" in data
                break
        else:
            pytest.fail("No transcription event found")

    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_voice_chat_response_event_has_agent_output(self, mock_tts, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        mock_tts.return_value = _fake_wav_bytes()

        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(b"\x00" * 50), "audio/webm")},
        )

        lines = resp.text.split("\n")
        for i, line in enumerate(lines):
            if line == "event: response":
                data = json.loads(lines[i + 1][6:])
                assert "Q3 revenue" in data["text"]
                assert "duration_ms" in data
                break
        else:
            pytest.fail("No response event found")

    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_voice_chat_audio_event_has_url(self, mock_tts, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        mock_tts.return_value = _fake_wav_bytes()

        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(b"\x00" * 50), "audio/webm")},
        )

        lines = resp.text.split("\n")
        for i, line in enumerate(lines):
            if line == "event: audio":
                data = json.loads(lines[i + 1][6:])
                assert data["url"].startswith("/voice/audio/")
                assert "duration_ms" in data
                break
        else:
            pytest.fail("No audio event found")

    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    def test_voice_chat_empty_transcript_returns_error(self, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        client._mock_client.transcribe = AsyncMock(return_value={"text": "", "language": "en"})

        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(b"\x00" * 50), "audio/webm")},
        )

        assert "error" in resp.text

    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_voice_chat_german_detection(self, mock_tts, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        mock_tts.return_value = _fake_wav_bytes()
        client._mock_client.transcribe = AsyncMock(return_value={
            "text": "Was waren unsere Q3 Ergebnisse?",
            "language": "de",
        })

        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(b"\x00" * 50), "audio/webm")},
        )

        lines = resp.text.split("\n")
        for i, line in enumerate(lines):
            if line == "event: transcription":
                data = json.loads(lines[i + 1][6:])
                assert data["language"] == "de"
                break
        else:
            pytest.fail("No transcription event found")


# ---------------------------------------------------------------------------
# GET /voice/audio/{id}
# ---------------------------------------------------------------------------

class TestVoiceAudio:
    @patch("src.server.voice_routes._convert_to_wav", new_callable=AsyncMock)
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_audio_endpoint_returns_wav(self, mock_tts, mock_convert, client):
        mock_convert.return_value = _fake_wav_bytes()
        wav_data = _fake_wav_bytes()
        mock_tts.return_value = wav_data

        # First do a voice chat to populate the cache
        resp = client.post(
            "/voice/chat",
            files={"file": ("recording.webm", io.BytesIO(b"\x00" * 50), "audio/webm")},
        )

        # Extract audio URL from events
        audio_url = None
        for line in resp.text.split("\n"):
            if '"url":' in line and "/voice/audio/" in line:
                data = json.loads(line[6:]) if line.startswith("data: ") else None
                if data:
                    audio_url = data["url"]
                    break

        assert audio_url is not None, "No audio URL found in SSE events"
        audio_resp = client.get(audio_url)
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"] == "audio/wav"
        assert audio_resp.content == wav_data

    def test_audio_endpoint_404_for_missing_id(self, client):
        resp = client.get("/voice/audio/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /voice/synthesize — standalone TTS
# ---------------------------------------------------------------------------

class TestVoiceSynthesize:
    @patch("src.server.voice_routes._synthesize_speech", new_callable=AsyncMock)
    def test_synthesize_returns_wav(self, mock_tts, client):
        wav = _fake_wav_bytes()
        mock_tts.return_value = wav

        resp = client.post("/voice/synthesize?text=Hello+world&language=en")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_synthesize_empty_text_returns_400(self, client):
        resp = client.post("/voice/synthesize?text=&language=en")
        assert resp.status_code == 400
