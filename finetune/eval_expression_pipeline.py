"""
Pipeline-level Calculator Expression Evaluator — end-to-end expression accuracy.

Measures how often the full agent pipeline (intent classification → tool
selection → expression building → calculator execution) produces the correct
numeric result.  This complements eval_tool_routing which tests the raw model
in isolation.

Historical note: an earlier version of the pipeline used a deterministic
build_single_step_expression() pre-router that boosted expression correctness
from ~39% to ~96%+ on top of a weak base model. Qwen3.5-4B FT v9 now reaches
99.4% routing natively, so the pre-router has been retired (see
src/engine/scaffolding/README.md). This eval still measures end-to-end
expression correctness — the path is now Intent → Qwen FT → calculator.

Usage:
  INFERENCE_PORT=9094 FUNCTION_PORT=9095 EMBEDDING_PORT=9096 \\
    python -m finetune.eval_expression_pipeline
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from datetime import datetime

from finetune.eval_base import compute_latency_stats, fmt_latency, fmt_pct_with_ci, wilson_ci
from finetune.eval_tool_routing import CALCULATOR_EXPECTED

from src.engine.agent import SmallLanguageModelAgentOrchestrator, Intent
from src.engine.inference.client import SmallLanguageModelClient
from src.engine.inference.config import SCENARIO_CONFIG
from src.engine.tools.tool_registry import create_default_registry
from src.engine.knowledge.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _extract_result(response_text: str) -> float | None:
    """Extract the numeric result from a pipeline response like '15% of 84900 = 12,735'."""
    if "=" in response_text:
        rhs = response_text.rsplit("=", 1)[1].strip()
        rhs = rhs.replace(",", "")
        try:
            return float(rhs)
        except ValueError:
            pass
    matches = _NUMBER_RE.findall(response_text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_eval() -> list[dict]:
    """Run all CALCULATOR_EXPECTED queries through the full agent pipeline."""
    client = SmallLanguageModelClient.create_with_auto_detection()
    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    vector_store.set_client(client)
    tools = create_default_registry(vector_store=vector_store)
    agent = SmallLanguageModelAgentOrchestrator(client=client, tools=tools)

    queries = list(CALCULATOR_EXPECTED.items())
    results = []

    print(f"\nRunning pipeline expression eval ({len(queries)} queries)…\n")

    for i, (query, expected) in enumerate(queries, 1):
        try:
            t0 = time.perf_counter()
            agent_result = await agent.process(query)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            response = agent_result.response
            intent = agent_result.intent
            actual = _extract_result(response)

            correct = False
            if actual is not None and expected != 0:
                correct = math.isclose(actual, expected, rel_tol=0.01)
            elif actual is not None and expected == 0:
                correct = abs(actual) < 0.01

            routed_to_calc = intent == Intent.TOOL_USE and any(
                s.details.get("tool") == "calculator"
                for s in agent_result.steps
                if s.action == "select_tool"
            )

            results.append({
                "query": query,
                "expected": expected,
                "actual": actual,
                "response": response[:200],
                "correct": correct,
                "intent": intent.value,
                "routed_to_calculator": routed_to_calc,
                "latency_ms": latency_ms,
            })
        except Exception as e:
            results.append({
                "query": query,
                "expected": expected,
                "actual": None,
                "response": str(e)[:200],
                "correct": False,
                "intent": "error",
                "routed_to_calculator": False,
                "latency_ms": 0,
            })

        if i % 10 == 0:
            print(f"  [{i}/{len(queries)}]")

    return results


def print_report(results: list[dict]) -> None:
    """Print a formatted report of pipeline expression accuracy."""
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_routed = sum(1 for r in results if r["routed_to_calculator"])
    n_routed_correct = sum(1 for r in results if r["routed_to_calculator"] and r["correct"])

    print("=" * 60)
    print("  Pipeline Expression Evaluation")
    print(f"  Run   : {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)
    print()
    print(f"  Total queries:              {n}")
    print(f"  Routed to calculator:       {n_routed}/{n} ({n_routed/n*100:.1f}%)")
    print(f"  Overall correct result:     {fmt_pct_with_ci(n_correct, n)}")

    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    if latencies:
        print(f"  Latency:                    {fmt_latency(compute_latency_stats(latencies))}")
    print()
    if n_routed > 0:
        print(f"  Calculator-routed correct:  {fmt_pct_with_ci(n_routed_correct, n_routed)}")
    print()

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"  Incorrect ({len(failures)}):")
        for r in failures:
            actual_str = f"{r['actual']}" if r["actual"] is not None else "None"
            print(f"    X  {r['query'][:70]}")
            print(f"       expected={r['expected']}  got={actual_str}  intent={r['intent']}")
    else:
        print("  All queries produced correct results!")
    print()


async def main():
    results = await run_eval()
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
