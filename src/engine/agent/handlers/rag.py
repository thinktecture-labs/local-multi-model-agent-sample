"""
RAG handler — query rewrite + vector search + 4B synthesis.
"""

import logging
import time
from collections.abc import AsyncIterator

from ..types import ExecutionStep
from ...inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from ...inference.config import (
    RAG_CONTEXT_DOCS,
    RAG_SYNTHESIS_MAX_TOKENS,
    RAG_SYNTHESIS_TEMPERATURE,
    RAG_TOP_K,
    REWRITE_MAX_TOKENS,
    REWRITE_TEMPERATURE,
)
from ...inference.prompts import (
    RAG_REWRITE_PROMPT_TEMPLATE,
    build_rag_messages,
)
from ...tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class RAGHandler:
    """
    Answer questions from the knowledge base.

    Three-step flow:
      1. Rewrite the query for better retrieval  (gemma3)
      2. Semantic search  (embeddinggemma via VectorSearchTool)
      3. Synthesize an answer from the retrieved context  (gemma3-4B)

    Query rewriting alone improves retrieval quality by 20-30%.
    """

    def __init__(
        self,
        client: SmallLanguageModelClient,
        tools: ToolRegistry,
    ) -> None:
        self._client = client
        self._tools = tools

    async def handle(
        self, query: str, *, deadline: float = 0, **kwargs,
    ) -> tuple[str, list[ExecutionStep]]:
        from .protocol import collect_stream
        return await collect_stream(self.handle_stream(query, deadline=deadline, **kwargs))

    async def handle_stream(
        self, query: str, *, deadline: float = 0, **kwargs,
    ) -> AsyncIterator[ExecutionStep | str]:
        """Stream RAG response. Steps 1-2 yield ExecutionSteps, step 3 streams tokens."""

        # --- Step 1: Query rewriting (non-streaming, fast) ---
        t0 = time.perf_counter()
        rewrite_response = await self._client.generate(
            prompt=RAG_REWRITE_PROMPT_TEMPLATE.format(query=query),
            temperature=REWRITE_TEMPERATURE,
            max_tokens=REWRITE_MAX_TOKENS,
            deterministic=True,
        )
        search_query = rewrite_response.content.strip()
        yield ExecutionStep(
            action="rewrite_query",
            model=self._client.models[SmallLanguageModelRole.INFERENCE],
            details={"original": query, "rewritten": search_query},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            tokens_used=rewrite_response.tokens_used,
            prompt_tokens=rewrite_response.prompt_tokens,
            completion_tokens=rewrite_response.completion_tokens,
        )

        # --- Step 2: Vector search (non-streaming, fast) ---
        # Search with both the rewritten query AND the original query, then
        # deduplicate by doc ID and keep the highest-scoring results.
        # The rewrite helps for some queries (adds keywords) but can hurt
        # for precise technical terms by dropping important words.
        t0 = time.perf_counter()
        rewrite_result = await self._tools.execute("vector_search", query=search_query, top_k=RAG_TOP_K)
        original_result = await self._tools.execute("vector_search", query=query, top_k=RAG_TOP_K)

        # Merge: deduplicate by doc ID, keep highest score
        seen: dict[str, object] = {}
        for doc in (rewrite_result.data or []) + (original_result.data or []):
            if doc.id not in seen or (doc.score or 0) > (seen[doc.id].score or 0):
                seen[doc.id] = doc
        merged = sorted(seen.values(), key=lambda d: d.score if d.score is not None else 0, reverse=True)[:RAG_TOP_K]
        search_result = rewrite_result  # preserve success/error state
        search_result.data = merged if merged else search_result.data

        retrieved_docs = [
            {"id": doc.id, "content": doc.content, "metadata": doc.metadata, "score": round(doc.score, 4) if doc.score is not None else None}
            for doc in (search_result.data or [])
        ]
        yield ExecutionStep(
            action="vector_search",
            model=self._client.models[SmallLanguageModelRole.EMBEDDING],
            details={"query": search_query, "documents": retrieved_docs},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

        if not search_result.success or not retrieved_docs:
            yield "I couldn't find relevant information in the knowledge base for that query."
            return

        # --- Step 3: Synthesis (streaming) ---
        t0 = time.perf_counter()
        top_docs = search_result.data[:RAG_CONTEXT_DOCS]

        remaining = (deadline - time.perf_counter()) if deadline else None
        if remaining is not None and remaining <= 0:
            yield "The pipeline deadline was exceeded before RAG synthesis could start."
            return

        messages, _ = build_rag_messages(top_docs, query)

        # Determinism note: RAG synthesis runs at temperature=RAG_SYNTHESIS_TEMPERATURE
        # (currently 0.1) without `deterministic=True`. The routing/classification/
        # tool-selection path IS byte-deterministic; this synthesis call is the one
        # documented gap — gemma3-4B at temp=0 produces noticeably more terse and
        # less natural prose, which hurts a streamed stage demo. Same retrieved
        # context will almost always produce the same answer, but byte-identity
        # is NOT guaranteed. See README "Determinism" section.
        stream = self._client.generate_synthesis_stream(
            messages=messages,
            temperature=RAG_SYNTHESIS_TEMPERATURE,
            max_tokens=RAG_SYNTHESIS_MAX_TOKENS,
        )

        synth_chunks: list[str] = []
        tokens_used = prompt_tokens = completion_tokens = 0
        async for chunk in stream:
            if chunk.done:
                tokens_used = chunk.tokens_used
                prompt_tokens = chunk.prompt_tokens
                completion_tokens = chunk.completion_tokens
            elif chunk.text:
                synth_chunks.append(chunk.text)
                yield chunk.text

        # gemma3-4B serves both vision (multimodal) and RAG synthesis on the same port.
        # Strip the "-vision" suffix so the logged model name is "gemma3-4b", not "gemma3-4b-vision".
        synth_model = self._client.models[SmallLanguageModelRole.VISION].replace("-vision", "")
        yield ExecutionStep(
            action="synthesize_response",
            model=synth_model,
            details={"context_docs": len(top_docs), "response": "".join(synth_chunks)},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
