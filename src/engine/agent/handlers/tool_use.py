"""
Tool-use handler — function model handles all tool selection and argument generation.

The function model (Qwen3.5-4B FT) selects the tool and generates all arguments
directly from the natural-language query. Earlier iterations shipped
ExpressionResolver/SQLResolver pre-routers as a deterministic compensator
for weaker base models; they were retired once Qwen FT reached 99.4% single-step
routing. See src/engine/scaffolding/README.md for the history.
"""

import json
import logging
import time
from collections.abc import AsyncIterator

from ..types import ExecutionStep
from ..query_decomposer import QueryDecomposer
from ..tool_argument_resolver import patch_calculator_expression
from ...inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from ...inference.config import (
    MULTI_STEP_DEADLINE_BUFFER,
    MULTI_STEP_SYNTHESIS_MAX_TOKENS,
    MULTI_STEP_SYNTHESIS_TEMPERATURE,
    TOOL_FORMAT_MAX_RESULT_CHARS,
    TOOL_FORMAT_MAX_TOKENS,
    TOOL_FORMAT_TEMPERATURE,
)
from ...inference.prompts import (
    MULTI_STEP_SYNTHESIS_PROMPT_TEMPLATE,
    TOOL_FORMAT_PROMPT_TEMPLATE,
)
from ...tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolUseHandler:
    """
    Handle queries that require executing tools (calculator, sql_query).

    First decomposes the query using gemma3. If multi-step (2+ sub-tasks),
    uses the chained execution path. Otherwise uses the fast single-step path.
    """

    def __init__(
        self,
        client: SmallLanguageModelClient,
        tools: ToolRegistry,
        decomposer: QueryDecomposer,
    ) -> None:
        self._client = client
        self._tools = tools
        self._decomposer = decomposer

    @property
    def _tool_schemas(self) -> list[dict]:
        """Tool schemas excluding vector_search (not used in tool-use path)."""
        return [
            s for s in self._tools.get_all_schemas()
            if s["function"]["name"] != "vector_search"
        ]

    def _resolve_tool_arguments(
        self,
        tool_name: str,
        tool_args: dict,
        normalized: str,
    ) -> tuple[str, dict]:
        """Apply lightweight post-processing fixes to tool arguments.

        Currently only patches incomplete calculator expressions (e.g. a bare
        "15%" reconstructed from the normalised query). Qwen FT handles
        everything else natively.
        """
        if tool_name == "calculator":
            tool_args = patch_calculator_expression(tool_args, normalized)
        return tool_name, tool_args

    async def handle(
        self, query: str, *, deadline: float, **kwargs,
    ) -> tuple[str, list[ExecutionStep]]:
        """Route to single-step or multi-step based on decomposition."""
        from .protocol import collect_stream
        return await collect_stream(self.handle_stream(query, deadline=deadline, **kwargs))

    async def handle_stream(
        self, query: str, *, deadline: float, **kwargs,
    ) -> AsyncIterator[ExecutionStep | str]:
        """Stream tool use response. Steps arrive incrementally, format_response streams tokens.

        Uses _select_and_execute() for the fast steps (select + execute),
        then streams the format LLM call directly — no double LLM call.
        """
        plan, decompose_step = await self._decomposer.decompose(query)

        if len(plan) > 1:
            # Multi-step: fall back to non-streaming (complex chain)
            response_text, steps = await self._handle_multi_step(
                query, plan, decompose_step, deadline=deadline,
            )
            for step in steps:
                yield step
            yield response_text
            return

        # Single-step: select + execute, then stream format
        result = await self._select_and_execute(query)
        if result is None:
            # No tool selected — stream direct answer
            from .direct_answer import DirectAnswerHandler
            handler = DirectAnswerHandler(self._client)
            async for item in handler.handle_stream(query, deadline=deadline):
                yield item
            return

        tool_name, tool_args, tool_result, steps = result

        # Yield select + execute steps incrementally
        for step in steps:
            yield step

        if not tool_result.success:
            yield f"The tool '{tool_name}' encountered an error: {tool_result.error}"
            return

        # Format — stream for LLM, static for calculator
        t0 = time.perf_counter()
        if tool_name == "calculator":
            result_val = tool_result.data.get("result")
            expr = tool_result.data.get("expression", "")
            if isinstance(result_val, float) and result_val == int(result_val):
                result_val = int(result_val)
            formatted = f"{result_val:,}" if isinstance(result_val, (int, float)) else str(result_val)
            response_text = f"{expr} = {formatted}"
            yield response_text
            yield ExecutionStep(
                action="format_response", model="local_execution",
                details={"response": response_text},
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        else:
            result_str = json.dumps(tool_result.data, indent=2, default=str)
            if len(result_str) > TOOL_FORMAT_MAX_RESULT_CHARS:
                result_str = result_str[:TOOL_FORMAT_MAX_RESULT_CHARS] + "\n... (truncated)"
            stream = self._client.generate_stream(
                prompt=TOOL_FORMAT_PROMPT_TEMPLATE.format(
                    query=query, tool_name=tool_name, result_str=result_str,
                ),
                temperature=TOOL_FORMAT_TEMPERATURE,
                max_tokens=TOOL_FORMAT_MAX_TOKENS,
                deterministic=True,
            )
            format_chunks: list[str] = []
            tokens_used = prompt_tokens = completion_tokens = 0
            async for chunk in stream:
                if chunk.done:
                    tokens_used = chunk.tokens_used
                    prompt_tokens = chunk.prompt_tokens
                    completion_tokens = chunk.completion_tokens
                elif chunk.text:
                    format_chunks.append(chunk.text)
                    yield chunk.text

            yield ExecutionStep(
                action="format_response",
                model=self._client.models[SmallLanguageModelRole.INFERENCE],
                details={"response": "".join(format_chunks)},
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    # ------------------------------------------------------------------
    # Single-step tool use
    # ------------------------------------------------------------------

    async def _select_and_execute(
        self, query: str,
    ) -> tuple[str, dict, object, list[ExecutionStep]] | None:
        """Select tool and execute it. Returns (tool_name, tool_args, tool_result, steps) or None if no tool."""
        steps: list[ExecutionStep] = []

        t0 = time.perf_counter()
        tool_response = await self._client.call_function(
            messages=[{"role": "user", "content": query}],
            tools=self._tool_schemas,
            deterministic=True,
        )
        select_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not tool_response.function_call:
            return None

        tool_name = tool_response.function_call["name"]
        tool_args = tool_response.function_call["arguments"]
        tool_name, tool_args = self._resolve_tool_arguments(
            tool_name, tool_args, query,
        )
        steps.append(ExecutionStep(
            action="select_tool",
            model=self._client.models[SmallLanguageModelRole.FUNCTION],
            details={"tool": tool_name, "arguments": tool_args},
            duration_ms=select_ms,
            tokens_used=tool_response.tokens_used,
            prompt_tokens=tool_response.prompt_tokens,
            completion_tokens=tool_response.completion_tokens,
        ))

        t0 = time.perf_counter()
        tool_result = await self._tools.execute(tool_name, **tool_args)
        steps.append(ExecutionStep(
            action="execute_tool", model="local_execution",
            details={
                "success": tool_result.success, "error": tool_result.error,
                "tool": tool_name, "result": tool_result.data,
            },
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        ))

        return tool_name, tool_args, tool_result, steps

    # ------------------------------------------------------------------
    # Multi-step tool use
    # ------------------------------------------------------------------

    async def _handle_multi_step(
        self,
        query: str,
        plan: list[str],
        decompose_step: ExecutionStep,
        *,
        deadline: float,
    ) -> tuple[str, list[ExecutionStep]]:
        """Handle multi-step tool_use queries with chained tool execution."""
        steps: list[ExecutionStep] = [decompose_step]
        accumulated_results: list[dict] = []
        tool_schemas = self._tool_schemas

        conversation: list[dict] = []

        for i, sub_task in enumerate(plan):
            remaining = deadline - time.perf_counter()
            if remaining < MULTI_STEP_DEADLINE_BUFFER:
                logger.warning(
                    "Multi-step budget exhausted (%.1fs remaining), "
                    "skipping step %d/%d to allow synthesis",
                    remaining, i + 1, len(plan),
                )
                break

            current_query = sub_task

            # ----- Concretize step 2+ via Qwen FT -----
            # Qwen FUNCTION-role concretization rewrites the abstract step
            # using accumulated_results so the next tool call has concrete
            # numbers/IDs to work with. Routes through FUNCTION role because
            # the smaller gemma3-1B plagiarised fewshot example numbers.
            if i > 0 and accumulated_results:
                current_query, concretize_step = await self._decomposer.concretize_step(
                    sub_task, accumulated_results, original_query=query,
                )
                steps.append(concretize_step)

            is_multi_step_first = i == 0 and len(plan) > 1

            conversation.append({"role": "user", "content": current_query})

            step_schemas = (
                [s for s in tool_schemas if s["function"]["name"] == "sql_query"]
                if is_multi_step_first
                else tool_schemas
            )

            # For step 2+, pass the full conversation so the model sees intermediate
            # tool results and can decide calculator vs sql_query with context.
            call_messages = conversation if i > 0 else [{"role": "user", "content": current_query}]

            t0 = time.perf_counter()
            tool_response = await self._client.call_function(
                messages=call_messages,
                tools=step_schemas,
                deterministic=True,
            )
            select_ms = round((time.perf_counter() - t0) * 1000, 1)

            if not tool_response.function_call:
                logger.warning(
                    "Multi-step %d/%d: no tool selected, aborting chain",
                    i + 1, len(plan),
                )
                break

            tool_name = tool_response.function_call["name"]
            tool_args = tool_response.function_call["arguments"]

            step_query = plan[i] if i < len(plan) else query
            tool_name, tool_args = self._resolve_tool_arguments(
                tool_name, tool_args, step_query,
            )

            conversation.append({
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                }}],
            })

            steps.append(ExecutionStep(
                action="select_tool",
                model=self._client.models[SmallLanguageModelRole.FUNCTION],
                details={
                    "tool": tool_name,
                    "arguments": tool_args,
                    "step": f"{i + 1}/{len(plan)}",
                },
                duration_ms=select_ms,
                tokens_used=tool_response.tokens_used,
                prompt_tokens=tool_response.prompt_tokens,
                completion_tokens=tool_response.completion_tokens,
            ))

            t0 = time.perf_counter()
            tool_result = await self._tools.execute(tool_name, **tool_args)

            tool_content = json.dumps(tool_result.data, default=str)
            if len(tool_content) > TOOL_FORMAT_MAX_RESULT_CHARS:
                tool_content = tool_content[:TOOL_FORMAT_MAX_RESULT_CHARS] + "... (truncated)"
            conversation.append({
                "role": "tool",
                "name": tool_name,
                "content": tool_content,
            })

            steps.append(ExecutionStep(
                action="execute_tool",
                model="local_execution",
                details={
                    "success": tool_result.success,
                    "error": tool_result.error,
                    "tool": tool_name,
                    "result": tool_result.data,
                    "step": f"{i + 1}/{len(plan)}",
                },
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            ))

            if tool_result.success:
                accumulated_results.append({
                    "tool": tool_name,
                    "sub_task": sub_task,
                    "result": tool_result.data,
                })
            else:
                # Short-circuit: downstream steps depend on this result.
                # Continuing would produce garbage (e.g. calculation on missing data).
                logger.warning(
                    "Multi-step %d/%d failed (%s: %s), aborting chain",
                    i + 1, len(plan), tool_name, tool_result.error,
                )
                break

        # Final synthesis
        if not accumulated_results:
            return "No tool results were obtained.", steps

        t0 = time.perf_counter()
        results_str = "\n\n".join(
            f"Step {i + 1} ({r['tool']}): {json.dumps(r['result'], default=str)}"
            for i, r in enumerate(accumulated_results)
        )
        if len(results_str) > TOOL_FORMAT_MAX_RESULT_CHARS:
            results_str = results_str[:TOOL_FORMAT_MAX_RESULT_CHARS] + "\n... (truncated)"
        # Route multi-step synthesis through Qwen3.5-4B FT (FUNCTION role).
        # gemma3-1B fails on this in four distinct ways depending on prompt
        # strategy: plagiarises example numbers when given examples; mislabels
        # units when not given examples; drops digits or invents math when
        # given strict unit-preservation rules. Qwen-4B reads structured
        # tool-result context (column names, calculator expressions) far more
        # faithfully — same reasoning as the concretize_step routing in
        # 3db64e4. Cost: ~+200ms per multi-step synthesis (Qwen p50 vs
        # gemma3-ft p50), fires once per query, negligible vs the correctness
        # recovery.
        synth_response = await self._client.generate(
            prompt=MULTI_STEP_SYNTHESIS_PROMPT_TEMPLATE.format(
                query=query, results_str=results_str,
            ),
            temperature=MULTI_STEP_SYNTHESIS_TEMPERATURE,
            max_tokens=MULTI_STEP_SYNTHESIS_MAX_TOKENS,
            deterministic=True,
            role=SmallLanguageModelRole.FUNCTION,
        )
        response_text = synth_response.content.strip()
        steps.append(ExecutionStep(
            action="synthesize_response",
            model=synth_response.model,
            details={"response": response_text},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            tokens_used=synth_response.tokens_used,
            prompt_tokens=synth_response.prompt_tokens,
            completion_tokens=synth_response.completion_tokens,
        ))

        return response_text, steps
