"""
Cloud orchestrator — wraps a cloud LLM API call as an AgentResponse with trace.

Used by the three-path comparison mode to produce a pipeline trace for
GPT-5.4 (or any OpenAI-compatible model) that is structurally identical
to the local pipeline traces, enabling side-by-side UI rendering.

The cloud path:
  1. Pre-fetches RAG context via local vector search (same as local pipeline)
  2. Injects context into the system prompt
  3. Exposes sql_query + calculator as function-calling tools
     (mirrors the local pipeline where the tool-calling model only sees these two)
  4. GPT decides whether to use tools or answer from RAG context directly
"""

import json as _json
import logging
import time
from collections.abc import AsyncIterator

from .types import AgentResponse, ExecutionStep, Intent

from ..inference.config import (
    CLOUD_COMPARISON_ENABLED,
    CLOUD_INPUT_COST_PER_1M,
    CLOUD_OUTPUT_COST_PER_1M,
    OPENAI_API_KEY,
    OPENAI_COMPARE_MODEL,
    SCENARIO_CONFIG,
)

logger = logging.getLogger(__name__)

# Maximum tool-call round-trips before forcing a final answer
_MAX_TOOL_ROUNDS = 5

# Tools exposed to GPT — matches what the tool-calling model sees in the local pipeline.
# vector_search is NOT a tool; RAG context is pre-injected into the prompt.
_CLOUD_TOOL_NAMES = {"sql_query", "calculator"}


class CloudOrchestrator:
    """Process queries via a cloud LLM with local RAG context and tool access."""

    def __init__(self, *, tools=None) -> None:
        self._available = CLOUD_COMPARISON_ENABLED
        self._tools = tools  # ToolRegistry

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(context_block: str) -> str:
        """Build a scenario-aware system prompt for the cloud model."""
        lang = "German" if SCENARIO_CONFIG.language == "de" else "English"
        return (
            f"{SCENARIO_CONFIG.rag_synthesis_system_prompt}\n\n"
            f"CRITICAL TOOL ROUTING RULES:\n"
            f"1. You have a SQL database with live data. {SCENARIO_CONFIG.sql_tool_description}\n"
            f"2. When the user asks for specific numbers, percentages, counts, lists of records, "
            f"or anything that could be answered by querying a database — you MUST use the sql_query tool. "
            f"Do NOT answer from the knowledge base context for data questions.\n"
            f"3. The knowledge base context below contains reference documents (policies, handbooks, SOPs). "
            f"Use it ONLY for conceptual questions (\"what is X?\", \"how does Y work?\", \"what are the rules for Z?\").\n"
            f"4. For calculations with specific numbers, use the calculator tool.\n"
            f"5. When presenting query results, summarize and aggregate — do not list every raw record. "
            f"Group by type, show counts and totals.\n"
            f"6. Answer in {lang}."
            f"{context_block}"
        )

    def _openai_tools(self) -> list[dict]:
        """Return OpenAI function-calling schemas for sql_query + calculator only."""
        if not self._tools:
            return []
        return [
            schema for schema in self._tools.get_all_schemas()
            if schema.get("function", {}).get("name") in _CLOUD_TOOL_NAMES
        ]

    async def _execute_tool_call(self, name: str, arguments: dict, steps: list[ExecutionStep]) -> str:
        """Execute a local tool and record an ExecutionStep. Returns JSON string."""
        t0 = time.perf_counter()
        result = await self._tools.execute(name, **arguments)
        elapsed = (time.perf_counter() - t0) * 1000

        steps.append(ExecutionStep(
            action="execute_tool",
            model="local",
            details={
                "tool": name,
                "arguments": arguments,
                "success": result.success,
                "error": result.error,
                "result": result.data,
            },
            duration_ms=round(elapsed, 1),
        ))

        if result.success:
            return _json.dumps(result.data, default=str)
        return _json.dumps({"error": result.error})

    # ------------------------------------------------------------------

    async def process(
        self, query: str, *, images: list[str] | None = None,
    ) -> AgentResponse:
        if not self._available:
            return AgentResponse(
                query=query,
                intent=Intent.DIRECT_ANSWER,
                response="Cloud comparison unavailable (no API key configured).",
                success=False,
            )

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_COMPARE_MODEL
        steps: list[ExecutionStep] = []
        total_start = time.perf_counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_bytes_sent = 0

        # --- Step 1: Local vector search (pre-fetched, same as local pipeline) ---
        context_block = ""
        if self._tools:
            try:
                t0 = time.perf_counter()
                search_result = await self._tools.execute("vector_search", query=query, top_k=5)
                retrieved_docs = [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "score": round(doc.score, 4) if doc.score is not None else None,
                    }
                    for doc in (search_result.data or [])
                ]
                search_ms = (time.perf_counter() - t0) * 1000

                steps.append(ExecutionStep(
                    action="vector_search",
                    model="embeddinggemma",
                    details={
                        "query": query,
                        "results": len(retrieved_docs),
                        "documents": retrieved_docs,
                    },
                    duration_ms=round(search_ms, 1),
                ))

                if retrieved_docs:
                    docs_text = "\n\n".join(
                        f"[{d['id']}] {d['content']}" for d in retrieved_docs
                    )
                    context_block = (
                        f"\n\nRelevant context from the knowledge base:\n{docs_text}\n\n"
                        "Use this context to answer knowledge base questions. "
                        "Only use sources whose title is relevant to the question. "
                        "If the context does not contain the answer, use the available tools."
                    )
            except Exception as exc:
                logger.warning("Cloud path vector search failed: %s", exc)

        # --- Step 2: Cloud inference with tools ---
        system_prompt = self._build_system_prompt(context_block)

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        openai_tools = self._openai_tools()

        # --- Tool-calling loop ---
        for _round in range(_MAX_TOOL_ROUNDS):
            payload_bytes = len(_json.dumps(messages).encode("utf-8"))
            total_bytes_sent += payload_bytes

            t0 = time.perf_counter()
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": 2000,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools
                else:
                    # reasoning_effort not supported alongside tools on GPT-5.4
                    kwargs["reasoning_effort"] = "none"
                resp = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                total_elapsed = (time.perf_counter() - total_start) * 1000
                logger.error("Cloud API call failed: %s", exc)
                steps.append(ExecutionStep(
                    action="cloud_inference",
                    model=model,
                    details={"error": str(exc)},
                    duration_ms=round(elapsed, 1),
                ))
                return AgentResponse(
                    query=query,
                    intent=Intent.DIRECT_ANSWER,
                    response=f"Cloud error: {exc}",
                    success=False,
                    steps=steps,
                    execution_time_ms=round(total_elapsed, 1),
                )

            elapsed = (time.perf_counter() - t0) * 1000

            # Accumulate token usage
            if resp.usage:
                total_prompt_tokens += resp.usage.prompt_tokens
                total_completion_tokens += resp.usage.completion_tokens

            choice = resp.choices[0]

            # If the model wants to call tools, execute them and loop
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Record the cloud inference step for this round
                steps.append(ExecutionStep(
                    action="cloud_inference",
                    model=model,
                    details={
                        "round": _round + 1,
                        "tool_calls": [tc.function.name for tc in choice.message.tool_calls],
                    },
                    duration_ms=round(elapsed, 1),
                    tokens_used=(resp.usage.prompt_tokens + resp.usage.completion_tokens) if resp.usage else 0,
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                ))

                # Append assistant message with tool calls
                messages.append(choice.message.model_dump())

                # Execute each tool call locally
                for tc in choice.message.tool_calls:
                    try:
                        args = _json.loads(tc.function.arguments)
                    except _json.JSONDecodeError:
                        args = {}

                    tool_result_str = await self._execute_tool_call(
                        tc.function.name, args, steps,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_str,
                    })

                continue  # Next round — GPT will see tool results

            # No tool calls — model produced a final answer
            break

        # --- Final response ---
        total_elapsed = (time.perf_counter() - total_start) * 1000
        response_text = choice.message.content or ""
        total_tokens = total_prompt_tokens + total_completion_tokens

        cost = (
            (total_prompt_tokens / 1_000_000) * CLOUD_INPUT_COST_PER_1M
            + (total_completion_tokens / 1_000_000) * CLOUD_OUTPUT_COST_PER_1M
        )

        # Record the final inference step
        steps.append(ExecutionStep(
            action="cloud_inference",
            model=model,
            details={
                "cost": round(cost, 6),
                "bytes_sent": total_bytes_sent,
                "tool_rounds": _round + 1,
            },
            duration_ms=round(elapsed, 1),
            tokens_used=(resp.usage.prompt_tokens + resp.usage.completion_tokens) if resp.usage else 0,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        ))

        return AgentResponse(
            query=query,
            intent=Intent.DIRECT_ANSWER,
            response=response_text,
            steps=steps,
            execution_time_ms=round(total_elapsed, 1),
        )

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    async def process_stream(
        self, query: str, *, images: list[str] | None = None,
    ) -> AsyncIterator[ExecutionStep | str]:
        """Stream cloud query response via SSE.

        Tool-calling rounds are non-streaming (need full response to parse
        tool calls). The final answer round uses stream=True for incremental
        token delivery — dropping GPT-5.4 TTFT from 10-15s to ~200-500ms.
        """
        if not self._available:
            yield "Cloud comparison unavailable (no API key configured)."
            return

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_COMPARE_MODEL
        steps: list[ExecutionStep] = []
        total_start = time.perf_counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_bytes_sent = 0

        # --- Step 1: Local vector search ---
        context_block = ""
        if self._tools:
            try:
                t0 = time.perf_counter()
                search_result = await self._tools.execute("vector_search", query=query, top_k=5)
                retrieved_docs = [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "score": round(doc.score, 4) if doc.score is not None else None,
                    }
                    for doc in (search_result.data or [])
                ]
                search_ms = (time.perf_counter() - t0) * 1000

                step = ExecutionStep(
                    action="vector_search",
                    model="embeddinggemma",
                    details={
                        "query": query,
                        "results": len(retrieved_docs),
                        "documents": retrieved_docs,
                    },
                    duration_ms=round(search_ms, 1),
                )
                steps.append(step)
                yield step

                if retrieved_docs:
                    docs_text = "\n\n".join(
                        f"[{d['id']}] {d['content']}" for d in retrieved_docs
                    )
                    context_block = (
                        f"\n\nRelevant context from the knowledge base:\n{docs_text}\n\n"
                        "Use this context to answer knowledge base questions. "
                        "Only use sources whose title is relevant to the question. "
                        "If the context does not contain the answer, use the available tools."
                    )
            except Exception as exc:
                logger.warning("Cloud path vector search failed: %s", exc)

        # --- Step 2: Cloud inference with tools ---
        system_prompt = self._build_system_prompt(context_block)

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        openai_tools = self._openai_tools()

        # --- Streaming loop: always stream, parse tool calls from deltas ---
        elapsed = 0.0
        for _round in range(_MAX_TOOL_ROUNDS):
            payload_bytes = len(_json.dumps(messages).encode("utf-8"))
            total_bytes_sent += payload_bytes

            t0 = time.perf_counter()
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": 4000,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools
                else:
                    kwargs["reasoning_effort"] = "none"

                stream = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("Cloud API call failed: %s", exc)
                yield ExecutionStep(
                    action="cloud_inference", model=model,
                    details={"error": str(exc)}, duration_ms=round(elapsed, 1),
                )
                yield f"Cloud error: {exc}"
                return

            # Consume the stream, collecting tool calls OR yielding content tokens
            tool_calls_by_idx: dict[int, dict] = {}  # idx -> {id, name, arguments}
            finish_reason = None
            round_prompt = 0
            round_completion = 0

            async for chunk in stream:
                if chunk.usage:
                    round_prompt = chunk.usage.prompt_tokens
                    round_completion = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason

                # Accumulate tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_by_idx:
                            tool_calls_by_idx[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            tool_calls_by_idx[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_by_idx[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_by_idx[idx]["arguments"] += tc_delta.function.arguments

                # Yield content tokens immediately
                if delta.content:
                    yield delta.content

            elapsed = (time.perf_counter() - t0) * 1000
            total_prompt_tokens += round_prompt
            total_completion_tokens += round_completion

            # If tool calls were collected, execute them and loop
            if tool_calls_by_idx:
                # Parse arguments for trace display
                parsed_calls = []
                for tc in tool_calls_by_idx.values():
                    try:
                        tc_args = _json.loads(tc["arguments"])
                    except _json.JSONDecodeError:
                        tc_args = {}
                    parsed_calls.append({"tool": tc["name"], "arguments": tc_args})
                step = ExecutionStep(
                    action="cloud_inference", model=model,
                    details={
                        "round": _round + 1,
                        "tool_calls": [tc["name"] for tc in tool_calls_by_idx.values()],
                        "calls": parsed_calls,
                    },
                    duration_ms=round(elapsed, 1),
                    tokens_used=round_prompt + round_completion,
                    prompt_tokens=round_prompt,
                    completion_tokens=round_completion,
                )
                steps.append(step)
                yield step

                # Build assistant message with tool calls for the conversation
                assistant_tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls_by_idx.values()
                ]
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                })

                for tc in tool_calls_by_idx.values():
                    try:
                        args = _json.loads(tc["arguments"])
                    except _json.JSONDecodeError:
                        args = {}

                    tool_result_str = await self._execute_tool_call(
                        tc["name"], args, steps,
                    )
                    yield steps[-1]

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result_str,
                    })

                continue

            # No tool calls — content tokens already yielded above
            if finish_reason == "length":
                yield "\n\n⚠️ *[Response truncated — token limit reached]*"
            break

        # --- Final metadata step ---
        truncated = finish_reason == "length"
        cost = (
            (total_prompt_tokens / 1_000_000) * CLOUD_INPUT_COST_PER_1M
            + (total_completion_tokens / 1_000_000) * CLOUD_OUTPUT_COST_PER_1M
        )
        yield ExecutionStep(
            action="cloud_inference", model=model,
            details={"cost": round(cost, 6), "bytes_sent": total_bytes_sent, "truncated": truncated},
            duration_ms=round(elapsed, 1),
            tokens_used=total_prompt_tokens + total_completion_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )
