"""
SmallLanguageModelAgentOrchestrator — thin router that coordinates all local SLM models.

Flow for every incoming query:
  1. If images are present -> route directly to VisionHandler
  2. Otherwise: inference model classifies the intent (constrained text generation)
  3. Router dispatches to the right handler
  4. Handler uses embedding model (RAG) or function model (tools)
  5. Everything is logged for future fine-tuning

The orchestrator itself holds no business logic — each intent is handled by
a dedicated handler class in the handlers/ package.
"""

import asyncio
import time
from collections.abc import AsyncIterator

from openai import APIConnectionError

from .types import (  # noqa: F401 — re-exported for backward compat
    AgentResponse,
    ExecutionStep,
    Intent,
)
from .intent_classifier import IntentClassifier, looks_like_adversarial
from .interaction_logger import InteractionLogger
from ..inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from ..inference.config import PIPELINE_TIMEOUT
from .query_decomposer import QueryDecomposer
from ..tools.tool_registry import ToolRegistry
from .handlers import (
    DirectAnswerHandler,
    RAGHandler,
    ToolUseHandler,
    VisionHandler,
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

from ..inference.config import SCENARIO_CONFIG

_ADVERSARIAL_REFUSAL = SCENARIO_CONFIG.adversarial_refusal


class SmallLanguageModelAgentOrchestrator:
    """
    Coordinates multiple specialized SLMs to answer queries.

    The orchestrator is a thin router — it classifies intent and dispatches
    to the appropriate handler. All business logic lives in the handlers.
    """

    def __init__(
        self,
        client: SmallLanguageModelClient,
        tools: ToolRegistry,
    ) -> None:
        self.client = client
        self.tools  = tools
        self._classifier = IntentClassifier(client)
        self._logger = InteractionLogger()

        self._decomposer = QueryDecomposer(
            client, client.models.get(SmallLanguageModelRole.INFERENCE, "unknown"),
        )

        # Handlers
        self._direct_answer = DirectAnswerHandler(client)
        self._rag = RAGHandler(client, tools)
        self._tool_use = ToolUseHandler(client, tools, self._decomposer)
        self._vision = VisionHandler(client)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process(
        self,
        query: str,
        images: list[str] | None = None,
    ) -> AgentResponse:
        """
        Process a user query end-to-end.

        This is the method to call from your API endpoint or CLI.
        Returns a fully populated AgentResponse including execution trace.

        If images are provided (base64-encoded), the query is routed directly
        to the vision model — no intent classification needed.
        """
        if not query or not query.strip():
            return AgentResponse(
                query=query or "",
                intent=Intent.DIRECT_ANSWER,
                response="Please provide a question or request.",
                success=False,
            )

        _MAX_QUERY_CHARS = 2000
        if len(query) > _MAX_QUERY_CHARS:
            return AgentResponse(
                query=query[:100] + "...",
                intent=Intent.DIRECT_ANSWER,
                response=f"Query too long ({len(query)} chars). Please keep it under {_MAX_QUERY_CHARS} characters.",
                success=False,
            )

        start = time.perf_counter()
        deadline = start + PIPELINE_TIMEOUT

        # 0. Deterministic routing: images present -> vision model
        if images:
            intent = Intent.IMAGE_QUERY
            try:
                async with asyncio.timeout(deadline - time.perf_counter()):
                    response_text, steps = await self._vision.handle(query, images=images)
            except (APIConnectionError, asyncio.TimeoutError) as exc:
                response_text = (
                    "The vision model server is not responding. "
                    "Please verify that llama-server is running on port 9093."
                ) if isinstance(exc, APIConnectionError) else (
                    f"Vision pipeline timed out after {PIPELINE_TIMEOUT}s."
                )
                steps = []

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._logger.log(query, intent, response_text, steps)

            return AgentResponse(
                query=query,
                intent=intent,
                response=response_text,
                steps=steps,
                execution_time_ms=round(elapsed_ms, 1),
            )

        # 1. Classify intent (LogReg primary on embeddinggemma vectors;
        # gemma3-ft generative fallback only when LogReg model is absent)
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(deadline - time.perf_counter()):
                intent, classify_resp = await self._classifier.classify(query)
        except asyncio.TimeoutError:
            return AgentResponse(
                query=query, intent=Intent.DIRECT_ANSWER,
                response=f"Pipeline timed out after {PIPELINE_TIMEOUT}s.",
                success=False,
            )
        classify_ms = (time.perf_counter() - t0) * 1000

        # 2. Route to the appropriate handler
        #    Adversarial check runs FIRST — before any handler dispatch.
        #    The classifier already catches most adversarial inputs, but queries
        #    that slip through (e.g. classified as rag_query/tool_use) are
        #    caught here before they reach tool execution or RAG retrieval.
        try:
            async with asyncio.timeout(deadline - time.perf_counter()):
                if looks_like_adversarial(query):
                    # Canned refusal — don't waste LLM inference on adversarial input
                    response_text, steps = _ADVERSARIAL_REFUSAL, []
                elif intent == Intent.RAG_QUERY:
                    response_text, steps = await self._rag.handle(query, deadline=deadline)
                elif intent == Intent.TOOL_USE:
                    response_text, steps = await self._tool_use.handle(query, deadline=deadline)
                else:
                    response_text, steps = await self._direct_answer.handle(query, deadline=deadline)
        except asyncio.TimeoutError:
            response_text = f"Pipeline timed out after {PIPELINE_TIMEOUT}s."
            steps = []
        except APIConnectionError:
            response_text = (
                "The model server is not responding. "
                "Please verify that llama-server is running on all required ports."
            )
            steps = []

        # Prepend the classification step so it appears first in the trace
        if classify_resp is None:
            # LogReg or adversarial filter — no LLM call
            classify_model = "logreg" if self._classifier.using_logreg else "filter"
        else:
            classify_model = self.client.models[SmallLanguageModelRole.INFERENCE]
        steps.insert(0, ExecutionStep(
            action="classify_intent",
            model=classify_model,
            details={"intent": intent.value},
            duration_ms=round(classify_ms, 1),
            tokens_used=classify_resp.tokens_used if classify_resp else 0,
            prompt_tokens=classify_resp.prompt_tokens if classify_resp else 0,
            completion_tokens=classify_resp.completion_tokens if classify_resp else 0,
        ))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 3. Log interaction for future fine-tuning
        self._logger.log(query, intent, response_text, steps)

        return AgentResponse(
            query=query,
            intent=intent,
            response=response_text,
            steps=steps,
            execution_time_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Streaming entry point
    # ------------------------------------------------------------------

    async def process_stream(
        self,
        query: str,
        images: list[str] | None = None,
    ) -> AsyncIterator[ExecutionStep | str]:
        """
        Stream a query response. Yields ExecutionStep objects for trace and
        str tokens for the response text.

        Classification is non-streaming (fast). Handler dispatch uses
        handle_stream() for incremental output.
        """
        if not query or not query.strip():
            yield "Please provide a question or request."
            return

        _MAX_QUERY_CHARS = 2000
        if len(query) > _MAX_QUERY_CHARS:
            yield f"Query too long ({len(query)} chars). Please keep it under {_MAX_QUERY_CHARS} characters."
            return

        start = time.perf_counter()
        deadline = start + PIPELINE_TIMEOUT

        # Image queries: non-streaming (vision responses are typically short)
        if images:
            try:
                async with asyncio.timeout(deadline - time.perf_counter()):
                    response_text, steps = await self._vision.handle(query, images=images)
            except asyncio.TimeoutError:
                yield f"Vision pipeline timed out after {PIPELINE_TIMEOUT}s."
                return
            except APIConnectionError:
                yield "The vision model server is not responding."
                return
            for step in steps:
                yield step
            yield response_text
            self._logger.log(query, Intent.IMAGE_QUERY, response_text, steps)
            return

        # 1. Classify intent (fast, non-streaming)
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(deadline - time.perf_counter()):
                intent, classify_resp = await self._classifier.classify(query)
        except asyncio.TimeoutError:
            yield f"Pipeline timed out after {PIPELINE_TIMEOUT}s."
            return
        classify_ms = (time.perf_counter() - t0) * 1000

        if classify_resp is None:
            classify_model = "logreg" if self._classifier.using_logreg else "filter"
        else:
            classify_model = self.client.models[SmallLanguageModelRole.INFERENCE]

        yield ExecutionStep(
            action="classify_intent",
            model=classify_model,
            details={"intent": intent.value},
            duration_ms=round(classify_ms, 1),
            tokens_used=classify_resp.tokens_used if classify_resp else 0,
            prompt_tokens=classify_resp.prompt_tokens if classify_resp else 0,
            completion_tokens=classify_resp.completion_tokens if classify_resp else 0,
        )

        # 2. Dispatch to streaming handler
        accumulated_text = ""
        all_steps: list[ExecutionStep] = []

        try:
            async with asyncio.timeout(deadline - time.perf_counter()):
                if looks_like_adversarial(query):
                    yield _ADVERSARIAL_REFUSAL
                    accumulated_text = _ADVERSARIAL_REFUSAL
                elif intent == Intent.RAG_QUERY:
                    async for item in self._rag.handle_stream(query, deadline=deadline):
                        if isinstance(item, ExecutionStep):
                            all_steps.append(item)
                        else:
                            accumulated_text += item
                        yield item
                elif intent == Intent.TOOL_USE:
                    async for item in self._tool_use.handle_stream(query, deadline=deadline):
                        if isinstance(item, ExecutionStep):
                            all_steps.append(item)
                        else:
                            accumulated_text += item
                        yield item
                else:
                    async for item in self._direct_answer.handle_stream(query, deadline=deadline):
                        if isinstance(item, ExecutionStep):
                            all_steps.append(item)
                        else:
                            accumulated_text += item
                        yield item
        except asyncio.TimeoutError:
            yield f"Pipeline timed out after {PIPELINE_TIMEOUT}s."
            return
        except APIConnectionError:
            yield "The model server is not responding."
            return

        # 3. Log interaction
        self._logger.log(query, intent, accumulated_text, all_steps)

    # ------------------------------------------------------------------
    # Logging & fine-tuning data export — delegated to InteractionLogger
    # ------------------------------------------------------------------

    def export_training_data(self, filepath: str) -> int:
        """Export interaction logs as JSON. Returns the number exported."""
        return self._logger.export(filepath)

    @property
    def interaction_count(self) -> int:
        """Number of interactions logged so far."""
        return self._logger.interaction_count

    @property
    def total_tokens_generated(self) -> int:
        """Total tokens consumed across all interactions."""
        return self._logger.total_tokens_generated

    @property
    def eviction_count(self) -> int:
        """Number of interaction log entries evicted due to buffer overflow."""
        return self._logger.eviction_count
