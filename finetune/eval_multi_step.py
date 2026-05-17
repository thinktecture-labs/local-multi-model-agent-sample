"""
Multi-Step Agentic Reasoning Evaluator — measure decomposition + tool chaining accuracy.

Uses a fixed test set of 60 queries (20 multi-step, 20 single-step SQL, 20 single-step
calculator) to evaluate whether the agent correctly decomposes compound queries and
chains the right tools in sequence.

Usage:
  python -m finetune.eval_multi_step                                      # run + print report
  python -m finetune.eval_multi_step --save results/multi_step.json       # save raw results

Metrics:
  1. Decomposition accuracy: Did the model correctly identify multi-step vs single-step?
  2. Tool chain accuracy: Did the plan result in the correct tool sequence?
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from finetune.eval_base import (
    compute_latency_stats,
    fmt_latency,
    fmt_pct as _fmt_pct,
    save_results,
)


# ---------------------------------------------------------------------------
# Load test set from JSONL — scenario-aware
# ---------------------------------------------------------------------------

from finetune.eval_base import load_eval_jsonl
from finetune._scenario import SCENARIO_NAME as _SCENARIO

_EVAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_test_set() -> list[dict]:
    """Load the multi-step eval dataset for the active scenario."""
    path = _EVAL_DATA_DIR / "eval-data" / f"eval_multi_step_{_SCENARIO}.jsonl"
    entries: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


CATEGORIES = ["multi_step", "single_sql", "single_calc"]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_eval(agent) -> dict:
    """
    Run every test query through the full agent pipeline and check results.

    Returns a results dict with per-query predictions and overall scores.
    """
    test_set = load_test_set()

    try:
        from src.engine.inference.client import SmallLanguageModelRole
        model_name = agent.client.models.get(SmallLanguageModelRole.INFERENCE, "unknown")
    except Exception:
        model_name = "unknown"

    predictions: list[dict] = []

    for item in test_set:
        t0 = time.perf_counter()
        result = await agent.process(item["query"])
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract tools used from execution trace
        tools_used = [
            s.details.get("tool", "")
            for s in result.steps
            if s.action == "select_tool"
        ]

        # Check if decompose step was present
        has_decompose = any(s.action == "decompose_query" for s in result.steps)
        decompose_count = 0
        if has_decompose:
            for s in result.steps:
                if s.action == "decompose_query":
                    decompose_count = s.details.get("count", 1)
                    break

        # Determine category
        expected_multi = item["multi_step"]
        predicted_multi = len(tools_used) > 1

        # Tool chain match
        expected_tools = item["expected_tools"]
        tools_match = tools_used == expected_tools

        # Decomposition correctness: multi-step query → 2+ steps, single → 1
        decompose_correct = (expected_multi == predicted_multi)

        predictions.append({
            "query": item["query"],
            "expected_tools": expected_tools,
            "predicted_tools": tools_used,
            "expected_multi_step": expected_multi,
            "predicted_multi_step": predicted_multi,
            "decompose_count": decompose_count,
            "tools_match": tools_match,
            "decompose_correct": decompose_correct,
            "intent": result.intent.value,
            "latency_ms": round(latency_ms, 1),
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "n": len(predictions),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    """Compute decomposition accuracy, tool chain accuracy, and per-category breakdown."""
    preds = results["predictions"]
    n = len(preds)

    # Overall
    decompose_correct = sum(1 for p in preds if p["decompose_correct"])
    tools_correct = sum(1 for p in preds if p["tools_match"])

    # Per category
    per_category: dict[str, dict] = {}
    for cat_key, filter_fn in [
        ("multi_step", lambda p: p["expected_multi_step"]),
        ("single_sql", lambda p: not p["expected_multi_step"] and p["expected_tools"] == ["sql_query"]),
        ("single_calc", lambda p: not p["expected_multi_step"] and p["expected_tools"] == ["calculator"]),
    ]:
        cat_preds = [p for p in preds if filter_fn(p)]
        n_cat = len(cat_preds)
        n_decompose = sum(1 for p in cat_preds if p["decompose_correct"])
        n_tools = sum(1 for p in cat_preds if p["tools_match"])
        per_category[cat_key] = {
            "n": n_cat,
            "decompose_correct": n_decompose,
            "decompose_accuracy": n_decompose / n_cat if n_cat else 0.0,
            "tools_correct": n_tools,
            "tools_accuracy": n_tools / n_cat if n_cat else 0.0,
        }

    return {
        "decompose_accuracy": decompose_correct / n if n else 0.0,
        "decompose_correct": decompose_correct,
        "tools_accuracy": tools_correct / n if n else 0.0,
        "tools_correct": tools_correct,
        "n": n,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "Multi-Step Eval Results") -> None:
    """Print a formatted evaluation report to stdout."""
    s = score(results)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  Model : {results.get('model', '?')}")
    print(f"  Run   : {results.get('timestamp', '?')[:19]}")
    print(f"{'=' * 60}")

    print(f"\n  Decomposition accuracy: {_fmt_pct(s['decompose_accuracy'])}"
          f"  ({s['decompose_correct']}/{s['n']})")
    print(f"  Tool chain accuracy:    {_fmt_pct(s['tools_accuracy'])}"
          f"  ({s['tools_correct']}/{s['n']})")

    latencies = [p["latency_ms"] for p in results["predictions"] if "latency_ms" in p]
    if latencies:
        print(f"  Latency:                {fmt_latency(compute_latency_stats(latencies))}")
    print()

    print("  Per-category breakdown:")
    for cat in CATEGORIES:
        cs = s["per_category"][cat]
        bar = "\u2588" * int(cs["tools_accuracy"] * 20)
        print(f"    {cat:<14s}  decompose {_fmt_pct(cs['decompose_accuracy'])}"
              f"  tools {_fmt_pct(cs['tools_accuracy'])}"
              f"  ({cs['tools_correct']}/{cs['n']})  {bar}")

    wrong = [p for p in results["predictions"] if not p["tools_match"]]
    if wrong:
        print(f"\n  Incorrect tool chains ({len(wrong)}):")
        for p in wrong:
            print(f"    \u2717  expected={p['expected_tools']}"
                  f"  got={p['predicted_tools']}")
            print(f"       \"{p['query'][:70]}\"")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate multi-step agentic reasoning accuracy",
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save raw results to this JSON file",
    )
    parser.add_argument("--function-port", type=int, metavar="PORT",
                        help="Override function-model port (e.g. 9100 for Qwen MoE).")
    parser.add_argument("--function-model", metavar="NAME",
                        help="Model name label to record in results.")
    args = parser.parse_args()

    # Build agent from live servers (register tools like server.py does)
    from src.engine.inference.client import SmallLanguageModelClient
    from src.engine.tools.tool_registry import create_default_registry
    from src.engine.agent import SmallLanguageModelAgentOrchestrator

    if args.function_port:
        function_url = f"http://localhost:{args.function_port}/v1"
        function_model = args.function_model or f"port-{args.function_port}"
        print(f"\nUsing function model: {function_model} ({function_url})")
        client = SmallLanguageModelClient(
            function_url=function_url,
            function_model=function_model,
        )
    else:
        client = SmallLanguageModelClient.create_with_auto_detection()
    tools = create_default_registry()
    agent = SmallLanguageModelAgentOrchestrator(client, tools)

    results = asyncio.run(run_eval(agent))
    print_report(results)

    if args.save:
        save_results(results, args.save)
