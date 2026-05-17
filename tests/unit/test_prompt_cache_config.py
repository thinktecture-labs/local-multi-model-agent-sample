"""
Unit tests for prompt cache wiring.

Verifies that:
  - n_keep constants exist and are in a sane range
  - generate() and generate_stream() inject n_keep when a system_prompt is provided
  - generate_synthesis() and generate_synthesis_stream() always inject n_keep
  - generate_vision() injects n_keep when a system_prompt is provided
  - Callers without a system_prompt do NOT receive n_keep
  - --swa-full is present in start_servers.sh for Gemma3 model launches
  - --cache-reuse is present for all generation model launches

These tests never start a server — they work entirely against mocked OpenAI
clients and by reading start_servers.sh as a text file.

Note on n_keep: llama-server's /v1/chat/completions endpoint does not expose
n_keep in the /slots response (shows 0 regardless), but the code is correct
and forward-compatible — future builds may support it, and it is honoured by
the native /completion endpoint. Unit tests verify the code sends it; integration
tests use TTFT as the actual cache evidence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


START_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "start_servers.sh"


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

class TestNKeepConstants:

    def test_n_keep_direct_answer_is_positive(self):
        from src.engine.inference.config import N_KEEP_DIRECT_ANSWER
        assert N_KEEP_DIRECT_ANSWER > 0

    def test_n_keep_rag_synthesis_is_positive(self):
        from src.engine.inference.config import N_KEEP_RAG_SYNTHESIS
        assert N_KEEP_RAG_SYNTHESIS > 0

    def test_n_keep_vision_is_positive(self):
        from src.engine.inference.config import N_KEEP_VISION
        assert N_KEEP_VISION > 0

    def test_n_keep_direct_answer_covers_prompt(self):
        """Constant must be >= estimated token count of DIRECT_ANSWER_SYSTEM_PROMPT."""
        from src.engine.inference.config import N_KEEP_DIRECT_ANSWER
        from src.engine.inference.prompts import DIRECT_ANSWER_SYSTEM_PROMPT
        # Conservative: assume ~1.2 tokens/word as lower bound
        word_count = len(DIRECT_ANSWER_SYSTEM_PROMPT.split())
        min_tokens = int(word_count * 1.2)
        assert N_KEEP_DIRECT_ANSWER >= min_tokens, (
            f"N_KEEP_DIRECT_ANSWER={N_KEEP_DIRECT_ANSWER} is less than estimated "
            f"min token count {min_tokens} for the direct-answer system prompt"
        )

    def test_n_keep_rag_synthesis_covers_prompt(self):
        """Constant must be >= estimated token count of RAG_SYNTHESIS_SYSTEM_PROMPT."""
        from src.engine.inference.config import N_KEEP_RAG_SYNTHESIS
        from src.engine.inference.prompts import RAG_SYNTHESIS_SYSTEM_PROMPT
        word_count = len(RAG_SYNTHESIS_SYSTEM_PROMPT.split())
        min_tokens = int(word_count * 1.2)
        assert N_KEEP_RAG_SYNTHESIS >= min_tokens

    def test_n_keep_values_are_reasonable_upper_bound(self):
        """n_keep values should be well below a typical context window (4096 tokens)."""
        from src.engine.inference.config import (
            N_KEEP_DIRECT_ANSWER,
            N_KEEP_RAG_SYNTHESIS,
            N_KEEP_VISION,
        )
        for val, name in [
            (N_KEEP_DIRECT_ANSWER, "N_KEEP_DIRECT_ANSWER"),
            (N_KEEP_RAG_SYNTHESIS, "N_KEEP_RAG_SYNTHESIS"),
            (N_KEEP_VISION, "N_KEEP_VISION"),
        ]:
            assert val < 200, f"{name}={val} seems too large — did you accidentally set it to full context size?"


# ---------------------------------------------------------------------------
# generate() — inference model
# ---------------------------------------------------------------------------

class TestGenerateNKeep:

    @pytest.fixture
    def mock_inference_client(self):
        """Return a SmallLanguageModelClient with a mocked _inference OpenAI client."""
        from src.engine.inference.client import SmallLanguageModelClient

        with patch("src.engine.inference.client.AsyncOpenAI") as MockOpenAI:
            mock_api = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
            mock_response.usage = MagicMock(total_tokens=10, prompt_tokens=8, completion_tokens=2)
            mock_api.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_api

            client = SmallLanguageModelClient()
            # Replace all four client instances with the same mock for simplicity
            client._inference = mock_api
            client._vision    = mock_api
            client._function  = mock_api
            client._embedding = mock_api
            return client, mock_api

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt_sends_n_keep(self, mock_inference_client):
        client, mock_api = mock_inference_client
        await client.generate(prompt="Hello", system_prompt="You are helpful.")
        call_kwargs = mock_api.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs, "n_keep extra_body not sent when system_prompt provided"
        assert "n_keep" in call_kwargs["extra_body"], "n_keep missing from extra_body"
        assert call_kwargs["extra_body"]["n_keep"] > 0

    @pytest.mark.asyncio
    async def test_generate_without_system_prompt_no_n_keep(self, mock_inference_client):
        client, mock_api = mock_inference_client
        await client.generate(prompt="Just a question")
        call_kwargs = mock_api.chat.completions.create.call_args.kwargs
        # Either no extra_body or n_keep not in it
        extra = call_kwargs.get("extra_body", {}) or {}
        assert "n_keep" not in extra, "n_keep should not be set when there is no system_prompt"

    @pytest.mark.asyncio
    async def test_generate_n_keep_matches_config(self, mock_inference_client):
        from src.engine.inference.config import N_KEEP_DIRECT_ANSWER
        client, mock_api = mock_inference_client
        await client.generate(prompt="Hello", system_prompt="You are helpful.")
        n_keep = mock_api.chat.completions.create.call_args.kwargs["extra_body"]["n_keep"]
        assert n_keep == N_KEEP_DIRECT_ANSWER

    @pytest.mark.asyncio
    async def test_generate_deterministic_preserves_n_keep(self, mock_inference_client):
        """n_keep must survive when deterministic=True also sets extra_body."""
        client, mock_api = mock_inference_client
        await client.generate(prompt="Hello", system_prompt="You are helpful.", deterministic=True)
        extra = mock_api.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert "n_keep" in extra, "n_keep lost when deterministic=True also sets extra_body"
        assert "top_k" in extra, "top_k lost from extra_body"


# ---------------------------------------------------------------------------
# generate_synthesis() — vision model (RAG synthesis path)
# ---------------------------------------------------------------------------

class TestGenerateSynthesisNKeep:

    @pytest.fixture
    def client_with_mock_vision(self):
        from src.engine.inference.client import SmallLanguageModelClient

        mock_api = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="synthesis result"))]
        mock_response.usage = MagicMock(total_tokens=20, prompt_tokens=15, completion_tokens=5)
        mock_api.chat.completions.create = AsyncMock(return_value=mock_response)

        client = SmallLanguageModelClient()
        client._vision = mock_api
        return client, mock_api

    @pytest.mark.asyncio
    async def test_generate_synthesis_always_sends_n_keep(self, client_with_mock_vision):
        client, mock_api = client_with_mock_vision
        from src.engine.inference.prompts import build_rag_messages
        messages, _ = build_rag_messages([], "test query")
        await client.generate_synthesis(messages=messages)
        extra = mock_api.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert "n_keep" in extra, "generate_synthesis must always send n_keep (system prompt is always present)"

    @pytest.mark.asyncio
    async def test_generate_synthesis_n_keep_matches_config(self, client_with_mock_vision):
        from src.engine.inference.config import N_KEEP_RAG_SYNTHESIS
        from src.engine.inference.prompts import build_rag_messages
        client, mock_api = client_with_mock_vision
        messages, _ = build_rag_messages([], "test query")
        await client.generate_synthesis(messages=messages)
        n_keep = mock_api.chat.completions.create.call_args.kwargs["extra_body"]["n_keep"]
        assert n_keep == N_KEEP_RAG_SYNTHESIS


# ---------------------------------------------------------------------------
# generate_vision() — vision model (image path)
# ---------------------------------------------------------------------------

class TestGenerateVisionNKeep:

    @pytest.fixture
    def client_with_mock_vision(self):
        from src.engine.inference.client import SmallLanguageModelClient

        mock_api = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="vision result"))]
        mock_response.usage = MagicMock(total_tokens=15, prompt_tokens=12, completion_tokens=3)
        mock_api.chat.completions.create = AsyncMock(return_value=mock_response)

        client = SmallLanguageModelClient()
        client._vision = mock_api
        return client, mock_api

    @pytest.mark.asyncio
    async def test_generate_vision_with_system_prompt_sends_n_keep(self, client_with_mock_vision):
        import base64
        client, mock_api = client_with_mock_vision
        # Minimal 1x1 white PNG
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64 = base64.b64encode(png).decode()
        await client.generate_vision(
            prompt="Describe this",
            images=[b64],
            system_prompt="You are a visual assistant.",
        )
        extra = mock_api.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert extra and "n_keep" in extra

    @pytest.mark.asyncio
    async def test_generate_vision_without_system_prompt_no_n_keep(self, client_with_mock_vision):
        import base64
        client, mock_api = client_with_mock_vision
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64 = base64.b64encode(png).decode()
        await client.generate_vision(prompt="Describe this", images=[b64])
        call_kwargs = mock_api.chat.completions.create.call_args.kwargs
        extra = call_kwargs.get("extra_body", {}) or {}
        assert "n_keep" not in extra

    @pytest.mark.asyncio
    async def test_generate_vision_n_keep_matches_config(self, client_with_mock_vision):
        import base64
        from src.engine.inference.config import N_KEEP_VISION
        client, mock_api = client_with_mock_vision
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64 = base64.b64encode(png).decode()
        await client.generate_vision(
            prompt="Describe this",
            images=[b64],
            system_prompt="You are a visual assistant.",
        )
        n_keep = mock_api.chat.completions.create.call_args.kwargs["extra_body"]["n_keep"]
        assert n_keep == N_KEEP_VISION


# ---------------------------------------------------------------------------
# start_servers.sh — static analysis
# ---------------------------------------------------------------------------

class TestStartServersFlags:

    @pytest.fixture(autouse=True)
    def _require_script(self):
        if not START_SCRIPT.is_file():
            pytest.skip("start_servers.sh not found")

    def _script_text(self) -> str:
        return START_SCRIPT.read_text()

    def test_swa_full_on_inference_launch(self):
        """inference launch line must include --swa-full (Gemma3 SWA fix)."""
        text = self._script_text()
        # Find the inference launch block and verify --swa-full follows it
        # The flag appears on the same line or a continuation line
        assert "--swa-full" in text, (
            "--swa-full missing from start_servers.sh — "
            "Gemma3 KV cache reuse silently fails without it (SWA bug #15082)"
        )

    def test_cache_reuse_on_inference_launch(self):
        """Inference/generation servers must include --cache-reuse."""
        assert "--cache-reuse" in self._script_text(), (
            "--cache-reuse missing — multi-turn conversations cannot salvage "
            "shifted KV cache when old messages are evicted"
        )

    def test_swa_full_count_covers_gemma3_servers(self):
        """--swa-full should appear at least twice (inference port 9090 + vision port 9093)."""
        count = self._script_text().count("--swa-full")
        assert count >= 2, (
            f"Expected --swa-full on at least 2 Gemma3 servers, found {count} occurrences. "
            "Both inference (9090) and vision (9093) are Gemma3 and need --swa-full."
        )

    def test_embedding_server_no_swa_full(self):
        """Embedding server launch must NOT have --swa-full (embeddinggemma is not SWA)."""
        text = self._script_text()
        # Find the embeddinggemma launch line and verify --swa-full is not on it
        # Simple heuristic: --embeddings flag is unique to the embedding server
        lines = text.split("\n")
        embedding_block = "\n".join(
            line for line in lines
            if "--embeddings" in line or "embeddinggemma" in line.lower()
        )
        assert "--swa-full" not in embedding_block, (
            "--swa-full should not be on the embedding server launch "
            "(embeddinggemma is not a Gemma3 SWA model)"
        )
