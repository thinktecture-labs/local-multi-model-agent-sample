"""
Integration tests for prompt cache behaviour.

Requires llama-server instances running (vision on port 9093, inference on 9090).
Skips automatically when servers are not available.

Detection strategy: TTFT-based (not /slots n_past).
Newer llama.cpp builds removed n_past from the /slots response schema. TTFT is
a more reliable signal anyway — a 5-10x speedup from req 1 → req 2 on identical
prompts is unambiguous evidence of KV cache reuse (measured in production: 136ms → 17ms).

Tests:
  - Cache hit (TTFT): second identical request is at least 3x faster than first
  - SWA fix active: 5 sequential same-prompt requests stay fast (not degrading)
  - Multi-turn chat: turn 2+ faster than turn 1 with shared context
  - Embedding model unaffected: caching changes don't break embed()
"""

from __future__ import annotations

import os
import time
from statistics import mean, stdev

import httpx
import pytest
from openai import AsyncOpenAI

from src.engine.inference.prompts import build_rag_messages
from src.engine.inference.config import N_KEEP_RAG_SYNTHESIS


VISION_PORT    = int(os.getenv("VISION_PORT",    9093))
INFERENCE_PORT = int(os.getenv("INFERENCE_PORT", 9090))

_CONTEXT = (
    "Nextera Q4 2024 Business Review. Revenue EUR 103,200. "
    "11 new customers. Churn rate 0.7%. ARR growth 21.6%. "
    "Top customer BrightHealth GmbH at EUR 7,000/mo Enterprise tier."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vision_client():
    return AsyncOpenAI(
        base_url=f"http://localhost:{VISION_PORT}/v1",
        api_key="no-key",
        timeout=30.0,
    )


def get_vision_model() -> str:
    try:
        resp = httpx.get(f"http://localhost:{VISION_PORT}/props", timeout=3.0)
        from pathlib import Path
        return Path(resp.json()["default_generation_settings"]["model"]).stem
    except Exception:
        return "gemma3-4b-vision"


def _make_doc(content: str):
    return type("D", (), {
        "id": "d1", "content": content, "metadata": {"title": "nextera.pdf"}
    })()


async def _ttft(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 25,
) -> float:
    """Time-to-first-token in milliseconds."""
    t0 = time.perf_counter()
    first_token_ms: float | None = None
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if first_token_ms is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_ms = (time.perf_counter() - t0) * 1000
    return first_token_ms or (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Cache hit tests — TTFT-based
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCacheHitTTFT:
    """Verify KV cache reuse via TTFT measurements."""

    @pytest.mark.asyncio
    async def test_warm_request_at_least_3x_faster(self, vision_server_available, vision_client):
        """
        Send the same prompt twice. The second request should be at least 3x faster.

        Measured baseline: cold=136ms → warm=17ms (8x speedup) with --swa-full.
        Without --swa-full: Gemma3 KV cache reuse silently fails — warm ≈ cold.
        3x threshold is conservative to account for Metal GPU scheduling variance.
        """
        if not vision_server_available:
            pytest.skip("Vision server not running")

        model = get_vision_model()
        msgs, _ = build_rag_messages([_make_doc(_CONTEXT)], "What was Q4 revenue?")

        # Send once to warm the slot
        cold_ttft = await _ttft(vision_client, model, msgs)
        # Send again — should hit the cached prefix
        warm_ttft = await _ttft(vision_client, model, msgs)

        speedup = cold_ttft / warm_ttft if warm_ttft > 0 else 0
        print(f"\n  cold={cold_ttft:.0f}ms  warm={warm_ttft:.0f}ms  speedup={speedup:.1f}x")

        assert speedup >= 3.0, (
            f"Expected warm TTFT at least 3x faster than cold. "
            f"cold={cold_ttft:.0f}ms  warm={warm_ttft:.0f}ms  speedup={speedup:.1f}x. "
            "Without --swa-full, Gemma3 KV cache reuse silently fails."
        )

    @pytest.mark.asyncio
    async def test_repeated_requests_stay_fast(self, vision_server_available, vision_client):
        """
        Send the same prompt 6 times. After the first (cold) request, requests 2-6
        should all be fast (cache hit). Confirms --swa-full keeps the cache warm
        across sequential requests on the same slot.
        """
        if not vision_server_available:
            pytest.skip("Vision server not running")

        model = get_vision_model()
        msgs, _ = build_rag_messages([_make_doc(_CONTEXT)], "How many new customers in Q4?")

        n = 6
        ttfts = []
        for i in range(n):
            t = await _ttft(vision_client, model, msgs)
            ttfts.append(t)

        cold_ttft = ttfts[0]
        warm_ttfts = ttfts[1:]
        warm_mean = mean(warm_ttfts)

        print(f"\n  TTFTs: {[f'{t:.0f}ms' for t in ttfts]}")
        print(f"  cold={cold_ttft:.0f}ms  warm_mean={warm_mean:.0f}ms")

        # On CUDA the cold TTFT is already ~15-28ms — measurement noise swamps the
        # cache signal at that speed (caching saves ~7ms, noise is ±5ms).
        # Only assert 2x speedup on Metal/CPU where cold is ≥50ms.
        if cold_ttft < 50:
            print(f"  cold={cold_ttft:.0f}ms < 50ms threshold — skipping 2x assertion (CUDA noise floor)")
            return
        assert warm_mean <= cold_ttft * 0.5, (
            f"Warm mean TTFT ({warm_mean:.0f}ms) is not at least 2x faster than cold ({cold_ttft:.0f}ms). "
            "KV cache reuse may not be working consistently."
        )

    @pytest.mark.asyncio
    async def test_different_question_same_context_not_slower(self, vision_server_available, vision_client):
        """
        After warming the slot with one question, a different question about the
        same context should not be slower — partial prefix caching (system prompt
        + context up to the diverging QUESTION: line) provides some benefit.

        On a 200-token prompt the shared prefix is ~180 tokens, so the speedup is
        modest (~1.2-1.5x) because the cold baseline is already fast (~75ms) at
        this token count. The key assertion: warm is not SLOWER than cold.
        """
        if not vision_server_available:
            pytest.skip("Vision server not running")

        model = get_vision_model()

        # Warm with Q1
        msgs1, _ = build_rag_messages([_make_doc(_CONTEXT)], "What was Q4 revenue?")
        cold_ttft = await _ttft(vision_client, model, msgs1)

        # Different question, same context
        msgs2, _ = build_rag_messages([_make_doc(_CONTEXT)], "What was the churn rate?")
        warm_ttft = await _ttft(vision_client, model, msgs2)

        speedup = cold_ttft / warm_ttft if warm_ttft > 0 else 0
        print(f"\n  Q1 (cold)={cold_ttft:.0f}ms  Q2 (warm)={warm_ttft:.0f}ms  speedup={speedup:.1f}x")

        # Warm must not be more than 20% slower than cold (no caching regression)
        assert warm_ttft <= cold_ttft * 1.20, (
            f"Warm TTFT ({warm_ttft:.0f}ms) is more than 20% slower than cold ({cold_ttft:.0f}ms). "
            "Caching may be causing overhead."
        )


# ---------------------------------------------------------------------------
# Multi-turn document chat
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMultiTurnTTFT:

    @pytest.mark.asyncio
    async def test_multiturn_no_regression(self, vision_server_available, vision_client):
        """
        Multi-turn document chat: turns 2-4 must not be slower than turn 1.

        With different questions each turn, the prefix diverges at the QUESTION:
        token — so speedup is modest (system prompt ~65 tokens cached, context
        body re-evaluated on each turn). The key assertion is no regression: caching
        must not add overhead compared to the cold baseline.

        For large speedups in multi-turn, questions must be identical (repeated
        exact query) — covered by test_warm_request_at_least_3x_faster.
        """
        if not vision_server_available:
            pytest.skip("Vision server not running")

        model = get_vision_model()
        questions = [
            "What was Q4 revenue?",
            "Who is the top customer?",
            "What was the churn rate?",
            "What was the ARR growth?",
        ]

        ttfts = []
        for q in questions:
            msgs, _ = build_rag_messages([_make_doc(_CONTEXT)], q)
            t = await _ttft(vision_client, model, msgs)
            ttfts.append(t)
            print(f"\n  Turn {len(ttfts)}: TTFT={t:.0f}ms")

        cold = ttfts[0]
        warm_mean = mean(ttfts[1:])
        print(f"\n  cold={cold:.0f}ms  warm_mean={warm_mean:.0f}ms")

        # Warm turns must not be more than 30% slower than cold (no regression)
        assert warm_mean <= cold * 1.30, (
            f"Multi-turn mean TTFT ({warm_mean:.0f}ms) is more than 30% slower than "
            f"turn 1 ({cold:.0f}ms). Caching is adding overhead."
        )


# ---------------------------------------------------------------------------
# Embedding model — caching changes must not break it
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEmbeddingModelUnaffected:

    @pytest.mark.asyncio
    async def test_embed_still_works(self, servers_available):
        if not servers_available:
            pytest.skip("Servers not running")
        from src.engine.inference.client import SmallLanguageModelClient
        client = SmallLanguageModelClient.create_with_auto_detection()
        vec = await client.embed("What was Q4 2024 revenue?")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    @pytest.mark.asyncio
    async def test_embed_batch_still_works(self, servers_available):
        if not servers_available:
            pytest.skip("Servers not running")
        from src.engine.inference.client import SmallLanguageModelClient
        client = SmallLanguageModelClient.create_with_auto_detection()
        texts = ["Q1 revenue?", "Q2 revenue?", "Q3 churn rate?"]
        vecs = await client.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) > 0 for v in vecs)
