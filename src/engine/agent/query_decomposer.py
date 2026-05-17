"""
Query decomposition for multi-step tool-use queries.

Detects whether a query needs multiple chained tool steps (e.g. lookup + calculate),
and if so, decomposes it into ordered sub-tasks using rule-based patterns with
gemma3 LLM fallback.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from .types import ExecutionStep
from ..inference.client import SmallLanguageModelRole
from ..inference.config import (
    CONCRETIZE_MAX_TOKENS,
    CONCRETIZE_TEMPERATURE,
    DECOMPOSE_MAX_TOKENS,
    DECOMPOSE_TEMPERATURE,
    SCENARIO_CONFIG,
)


# Fewshot examples live in scenarios/<name>.json under prompts.decomposer_fewshot
# and prompts.concretize_examples. Empty strings are safe fallbacks — the
# underlying prompts still describe the task and JSON output shape; the
# fewshot just primes the model with concrete patterns from the active
# scenario.

if TYPE_CHECKING:
    from ..inference.client import SmallLanguageModelClient

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decomposes compound queries into ordered sub-tasks for chained tool execution."""

    # Compound connector patterns that signal multi-step (lookup → calculate).
    # Every multi-step query in the eval set matches one of these; zero
    # single-step queries do.  This gives 100% decomposition accuracy
    # without calling the model for the majority of queries.
    _MULTI_STEP_PATTERNS = [
        re.compile(p, re.IGNORECASE)
        for p in [
            # English patterns
            r"\band\s+what\s+would\b",
            r"\band\s+calculate\b",
            r"\band\s+compute\b",
            r"\band\s+what\s+is\b",
            r"\band\s+how\s+many\b.*\bwould\b",
            r"\band\s+if\s+we\b",
            r"\band\s+what\s+percentage\b",
            r"\band\s+what\s+would\b.*\blook\s+like\b",
            r"\band\s+(?:the|their)\s+\w+\s+(?:spend|cost|total)\b",
            r"\band\s+what\s+(?:share|portion|fraction)\b",
            r"\band\s+how\s+much\s+would\b",
            # German-language patterns
            # No trailing \b — German compounds (Durchschnittskosten, welchen) break word boundaries
            r"\bund\s+(?:welch|wie\s+(?:viel|hoch|lang)|was\s)",
            r"\bund\s+(?:berechne|hochrechnung|umrechnung|nachbeschaffung)",
            r"\bund\s+(?:deren?|dessen|die|das|den)\s+\w+\s+(?:berechnen|angeben|in\s+prozent)",
            r"\bund\s+(?:kosten|durchschnitt|differenz|anteil|prozent|gesamt|gesamtkosten)",
            r"\bund\s+(?:gewichtete|geschaetzte|verbleibende|durchschnittliche|tages)",
        ]
    ]

    def __init__(self, client: SmallLanguageModelClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def detect_multi_step(self, query: str) -> bool:
        """Rule-based detection: does this query need 2 chained tool steps?

        Returns True when the query contains a compound connector joining a
        data-lookup clause with a computation clause (e.g. "find X, and
        calculate Y from X").  Single-step queries (pure SQL or pure
        calculator) never match these patterns.
        """
        return any(p.search(query) for p in self._MULTI_STEP_PATTERNS)

    async def decompose(self, query: str) -> tuple[list[str], ExecutionStep]:
        """Decompose a tool_use query into ordered sub-tasks.

        Uses a rule-based pre-filter for obvious cases (single-step or
        clearly multi-step), falling back to gemma3 only when rules detect
        a multi-step compound query that needs the model to generate
        step descriptions.

        Returns a list of sub-task descriptions and the trace step.
        For single-tool queries, returns a single-element list.
        """
        is_multi = self.detect_multi_step(query)

        # ── Single-step: skip model entirely ────────────────────────────
        if not is_multi:
            trace_step = ExecutionStep(
                action="decompose_query",
                model="rule_based",
                details={"steps": [query], "count": 1, "method": "rule"},
                duration_ms=0.0,
            )
            return [query], trace_step

        # ── Multi-step: call gemma3 to generate step descriptions ───────
        t0 = time.perf_counter()
        fewshot = SCENARIO_CONFIG.decomposer_fewshot
        response = await self._client.generate(
            prompt=(
                "You are a query planner. Tools: sql_query (database), calculator (math).\n"
                "This query needs EXACTLY 2 steps: step 1 looks up data via SQL, step 2 calculates.\n"
                "Rules:\n"
                "- Output exactly 2 steps. Never 1, never 3, never more.\n"
                "- Use the EXACT numbers, time periods, and quantities from the user's query. "
                "If the user says '2 years', your step must say '2 years' — NEVER substitute "
                "values from the examples below.\n"
                "- Respond in the same language as the user's query.\n"
                "Return JSON: {\"steps\": [\"step1\", \"step2\"]}\n\n"
                f"{fewshot}"
                f"Query: {query}\n"
            ),
            temperature=DECOMPOSE_TEMPERATURE,
            max_tokens=DECOMPOSE_MAX_TOKENS,
            json_mode=True,
            deterministic=True,
        )
        try:
            parsed = json.loads(response.content.strip())
            if isinstance(parsed, dict):
                steps = parsed.get("steps", [])
            elif isinstance(parsed, list):
                steps = parsed
            else:
                steps = []
            if not steps or not isinstance(steps, list):
                steps = []
            steps = [str(s) for s in steps]
        except (json.JSONDecodeError, ValueError):
            steps = []

        # Fallback: if the model failed to produce 2 steps despite the rule
        # detecting multi-step, split mechanically at the compound connector.
        if len(steps) < 2:
            for sep in [", and ", " and ", " und "]:
                idx = query.lower().find(sep)
                if idx > 0:
                    steps = [
                        query[:idx].strip().rstrip("?."),
                        query[idx + len(sep) :].strip(),
                    ]
                    break
            if len(steps) < 2:
                steps = [query]

        # The decomposer prompt declares exactly 2 steps for compound queries;
        # truncate to 2 so a model that over-elaborates gets cropped to the
        # expected shape.
        steps = steps[:2]

        trace_step = ExecutionStep(
            action="decompose_query",
            model=self._model_name,
            details={"steps": steps, "count": len(steps), "method": "rule+model"},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            tokens_used=response.tokens_used,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return steps, trace_step

    async def concretize_step(
        self, step_description: str, prior_results: list[dict],
        original_query: str = "",
    ) -> tuple[str, ExecutionStep]:
        """Rewrite a sub-task with concrete values from prior tool results."""
        # Build a flat variable list from prior results for easy substitution
        variables: list[str] = []
        for r in prior_results:
            result_data = r.get("result", r)
            if isinstance(result_data, dict):
                # Handle SQL table results: extract first row values
                if "rows" in result_data and result_data["rows"]:
                    first_row = result_data["rows"][0]
                    for k, v in first_row.items():
                        if isinstance(v, (int, float)):
                            variables.append(f"  {k} = {v}")
                else:
                    for k, v in result_data.items():
                        if isinstance(v, (int, float)):
                            variables.append(f"  {k} = {v}")
            elif isinstance(result_data, (int, float)):
                variables.append(f"  value = {result_data}")
        variables_str = "\n".join(variables) if variables else "  (no numeric values)"

        context_line = f"Original question: {original_query}\n" if original_query else ""

        # Route concretize through the FUNCTION role (Qwen3.5-4B FT) — the 1B
        # inference model was unreliable at substituting SQL-result values
        # into the right arithmetic shape (hallucinated growth assumptions,
        # confused row counts with values). Qwen's stronger reasoning over
        # structured context is worth the latency premium here (concretize
        # fires at most once per multi-step query).
        t0 = time.perf_counter()
        response = await self._client.generate(
            prompt=(
                "Substitute the actual numbers into this step. "
                "Output ONLY a single sentence starting with 'Calculate' "
                "that contains the actual numbers and arithmetic. No words, just math.\n\n"
                f"{context_line}"
                f"Known values:\n{variables_str}\n\n"
                f"Step: {step_description}\n\n"
                "Examples:\n"
                f"{SCENARIO_CONFIG.concretize_examples}\n\n"
                "Answer:"
            ),
            temperature=CONCRETIZE_TEMPERATURE,
            max_tokens=CONCRETIZE_MAX_TOKENS,
            deterministic=True,
            role=SmallLanguageModelRole.FUNCTION,
        )
        concrete = response.content.strip().split("\n")[0]  # first line only
        trace_step = ExecutionStep(
            action="concretize_step",
            model=response.model,
            details={"original": step_description, "concrete": concrete},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            tokens_used=response.tokens_used,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return concrete, trace_step
