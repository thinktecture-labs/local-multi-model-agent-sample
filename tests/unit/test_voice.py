"""
Unit tests for voice pipeline helpers.

Tests WAV header construction, emoji stripping, language-based voice selection,
audio cache expiration, and SSE event formatting — all without external services.
"""

import struct
import time

import pytest


# ---------------------------------------------------------------------------
# WAV header construction
# ---------------------------------------------------------------------------

class TestBuildWavHeader:
    """Tests for _build_wav_header() in server.py."""

    def _build_wav_header(self, pcm_length, sample_rate=22050, channels=1, bits_per_sample=16):
        """Import and call the helper from voice_routes."""
        from src.server.voice_routes import _build_wav_header
        return _build_wav_header(pcm_length, sample_rate, channels, bits_per_sample)

    def test_header_length_is_44_bytes(self):
        header = self._build_wav_header(0)
        assert len(header) == 44

    def test_riff_magic(self):
        header = self._build_wav_header(100)
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"

    def test_data_chunk_size_matches_pcm(self):
        pcm_len = 44100
        header = self._build_wav_header(pcm_len)
        # "data" chunk size is at bytes 40-44
        data_size = struct.unpack_from("<I", header, 40)[0]
        assert data_size == pcm_len

    def test_riff_size_is_36_plus_pcm(self):
        pcm_len = 22050
        header = self._build_wav_header(pcm_len)
        riff_size = struct.unpack_from("<I", header, 4)[0]
        assert riff_size == 36 + pcm_len

    def test_sample_rate_encoded(self):
        header = self._build_wav_header(0, sample_rate=22050)
        sr = struct.unpack_from("<I", header, 24)[0]
        assert sr == 22050

    def test_16khz_for_whisper(self):
        header = self._build_wav_header(0, sample_rate=16000)
        sr = struct.unpack_from("<I", header, 24)[0]
        assert sr == 16000

    def test_mono_channel(self):
        header = self._build_wav_header(0, channels=1)
        ch = struct.unpack_from("<H", header, 22)[0]
        assert ch == 1

    def test_valid_wav_with_pcm_data(self):
        """A WAV built from header + PCM should be parseable."""
        pcm = b"\x00\x01" * 100  # 200 bytes of fake PCM
        header = self._build_wav_header(len(pcm))
        wav = header + pcm
        # Verify total size
        assert len(wav) == 44 + 200
        # Verify RIFF header reports correct total size
        riff_size = struct.unpack_from("<I", wav, 4)[0]
        assert riff_size == len(wav) - 8


# ---------------------------------------------------------------------------
# SSE event formatting
# ---------------------------------------------------------------------------

class TestSseEvent:
    def _sse_event(self, event_type, data):
        from src.server.voice_routes import _sse_event
        return _sse_event(event_type, data)

    def test_event_format(self):
        result = self._sse_event("transcription", {"text": "hello"})
        assert result.startswith("event: transcription\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_data_is_json(self):
        import json
        result = self._sse_event("response", {"text": "world", "duration_ms": 42.5})
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        parsed = json.loads(data_line[6:])
        assert parsed["text"] == "world"
        assert parsed["duration_ms"] == 42.5


# ---------------------------------------------------------------------------
# Audio cache expiration
# ---------------------------------------------------------------------------

class TestAudioCache:
    def test_get_returns_none_for_missing(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=10, ttl=120)
        assert cache.get("nonexistent") is None

    def test_put_and_get(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=10, ttl=120)
        cache.put("id-1", b"wav-data")
        assert cache.get("id-1") == b"wav-data"

    def test_expired_entry_returns_none(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=10, ttl=0.0)  # immediate expiry
        cache.put("old-id", b"wav-data")
        # Entry is expired immediately (ttl=0)
        assert cache.get("old-id") is None

    def test_fresh_entry_survives(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=10, ttl=120)
        cache.put("fresh", b"data")
        assert cache.get("fresh") == b"data"

    def test_evicts_lru_when_full(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=3, ttl=120)
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.put("c", b"3")
        # Cache is full (3/3). Adding "d" should evict "a" (LRU).
        cache.put("d", b"4")
        assert len(cache) == 3
        assert cache.get("a") is None  # evicted
        assert cache.get("b") == b"2"
        assert cache.get("d") == b"4"

    def test_get_promotes_to_mru(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=3, ttl=120)
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.put("c", b"3")
        # Access "a" to promote it (no longer LRU)
        cache.get("a")
        # Now "b" is LRU — adding "d" should evict "b"
        cache.put("d", b"4")
        assert cache.get("a") == b"1"  # survived (was promoted)
        assert cache.get("b") is None  # evicted (was LRU)

    def test_maxsize_enforced(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=5, ttl=120)
        for i in range(20):
            cache.put(f"id-{i}", b"wav")
        assert len(cache) <= 5

    def test_put_updates_existing_entry(self):
        from src.server.voice_routes import _AudioCache
        cache = _AudioCache(maxsize=10, ttl=120)
        cache.put("id-1", b"old")
        cache.put("id-1", b"new")
        assert cache.get("id-1") == b"new"
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# Language → voice mapping
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    def test_english_voice_default(self):
        from src.engine.inference.config import PIPER_VOICE_EN
        assert "en_" in PIPER_VOICE_EN

    def test_german_voice_configured(self):
        from src.engine.inference.config import PIPER_VOICE_DE
        assert "de_DE" in PIPER_VOICE_DE

    def test_whisper_url_format(self):
        from src.engine.inference.config import WHISPER_URL, WHISPER_PORT
        assert str(WHISPER_PORT) in WHISPER_URL
        assert WHISPER_URL.startswith("http://")


# ---------------------------------------------------------------------------
# Client transcribe method exists
# ---------------------------------------------------------------------------

class TestClientTranscribe:
    def test_transcribe_method_exists(self):
        from src.engine.inference.client import SmallLanguageModelClient
        client = SmallLanguageModelClient.__new__(SmallLanguageModelClient)
        assert hasattr(client, "transcribe")
        assert callable(client.transcribe)

    async def test_health_check_includes_whisper(self):
        """check_health should include a WHISPER key."""
        from src.engine.inference.client import SmallLanguageModelClient, SmallLanguageModelRole
        client = SmallLanguageModelClient.__new__(SmallLanguageModelClient)
        # Provide dummy URLs for all model roles so iteration doesn't KeyError
        client._urls = {role: "http://localhost:99999/v1" for role in SmallLanguageModelRole}
        client._whisper_url = "http://localhost:99999"  # won't connect
        result = await client.check_health()
        assert "WHISPER" in result
        assert result["WHISPER"] is False  # server not running
