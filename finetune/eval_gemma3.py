"""
Intent Classification Evaluator — measure gemma3 accuracy before/after fine-tuning.

Uses a fixed labelled test set of 180 queries (60 per intent class) to produce
a reproducible accuracy score. Run once before fine-tuning to establish a
baseline, then again after to quantify the improvement.

Usage:
  python -m finetune.eval_gemma3                                        # run + print report
  python -m finetune.eval_gemma3 --save results/baseline_gemma3.json    # save raw results
  python -m finetune.eval_gemma3 --compare before.json after.json       # show delta

Demo talk workflow:
  1. python -m finetune.eval_gemma3 --save results/baseline_gemma3.json
  2. python -m finetune.pipeline --train-all
  3. python -m finetune.eval_gemma3 --save results/finetuned_gemma3.json
  4. python -m finetune.eval_gemma3 --compare results/baseline_gemma3.json results/finetuned_gemma3.json
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from finetune.eval_base import (
    compute_latency_stats,
    fmt_ci as _fmt_ci,
    fmt_latency,
    fmt_pct as _fmt_pct,
    fmt_pct_with_ci as _fmt_pct_ci,
    load_eval_jsonl,
    load_results,
    mcnemar_test,
    save_results,
    wilson_ci,
)
from finetune._scenario import SCENARIO_NAME as _SCENARIO


# ---------------------------------------------------------------------------
# Fixed labelled test set — 180 queries, 60 per intent class
# Loaded from data/eval-data/ JSONL files (not inline).
# ---------------------------------------------------------------------------
TEST_SET = load_eval_jsonl(f"eval_gemma3_{_SCENARIO}.jsonl")

CLASSES = ["rag_query", "tool_use", "direct_answer"]

# Shared with SmallLanguageModelAgentOrchestrator — single source of truth
from src.engine.agent.types import CLASSIFY_PROMPT as _CLASSIFY_PROMPT


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def classify_query(client, query: str) -> str:
    """Classify a single query using the live model. Returns an intent label."""
    response = await client.generate(
        prompt=_CLASSIFY_PROMPT.format(query=query),
        temperature=0.0,
        max_tokens=50,
    )
    return _extract_intent(response.content)


def _extract_intent(text: str) -> str:
    """Extract intent label from model output — handles both terse and verbose formats.

    Gemma-FT returns a bare label like "rag_query".
    Qwen/other models may return verbose text like "**Classification:** `rag_query`".
    This parser handles both by scanning for known labels anywhere in the response.
    """
    raw = text.strip().lower()

    # Fast path: first line is exactly a known class (Gemma-FT style)
    first_line = raw.split("\n")[0].strip().replace(" ", "_").rstrip(".")
    if first_line in CLASSES:
        return first_line

    # Slow path: scan full text for any known class label (verbose models)
    # Priority order matters — check longer labels first to avoid partial matches
    for label in sorted(CLASSES, key=len, reverse=True):
        if label in raw:
            return label

    return "unknown"


async def run_eval(
    client,
    test_set: list[dict[str, str]] | None = None,
) -> dict:
    """
    Classify every query in test_set against the live model.

    Returns a results dict suitable for score(), print_report(), and save_results().
    """
    if test_set is None:
        test_set = TEST_SET

    # Resolve the inference model name for the results record
    try:
        from src.engine.inference.client import SmallLanguageModelRole
        model_name = client.models.get(SmallLanguageModelRole.INFERENCE, "unknown")
    except Exception:
        model_name = "unknown"

    predictions: list[dict] = []
    for item in test_set:
        t0 = time.perf_counter()
        predicted = await classify_query(client, item["query"])
        latency_ms = (time.perf_counter() - t0) * 1000
        predictions.append({
            "query":      item["query"],
            "expected":   item["intent"],
            "predicted":  predicted,
            "correct":    predicted == item["intent"],
            "latency_ms": round(latency_ms, 1),
        })

    return {
        "timestamp":   datetime.now().isoformat(),
        "model":       model_name,
        "n":           len(predictions),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    """
    Compute per-class and overall accuracy from a results dict.

    Returns a scoring dict with keys:
      overall_accuracy, overall_correct, overall_ci, n, per_class
    """
    preds = results["predictions"]
    n = len(preds)
    overall_correct = sum(1 for p in preds if p["correct"])
    overall_accuracy = overall_correct / n if n else 0.0

    per_class: dict[str, dict] = {}
    for cls in CLASSES:
        cls_preds = [p for p in preds if p["expected"] == cls]
        n_cls = len(cls_preds)
        n_correct = sum(1 for p in cls_preds if p["correct"])
        per_class[cls] = {
            "n":        n_cls,
            "correct":  n_correct,
            "accuracy": n_correct / n_cls if n_cls else 0.0,
            "ci":       wilson_ci(n_correct, n_cls),
        }

    return {
        "overall_accuracy": overall_accuracy,
        "overall_correct":  overall_correct,
        "overall_ci":       wilson_ci(overall_correct, n),
        "n":                n,
        "per_class":        per_class,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(before: dict, after: dict) -> dict:
    """
    Compute accuracy deltas between two result snapshots.

    Returns a comparison dict with overall_delta, per-class deltas, and
    McNemar's significance test (paired comparison on the same queries).
    Positive values = improvement after fine-tuning.
    """
    s_before = score(before)
    s_after  = score(after)

    per_class_delta: dict[str, float] = {
        cls: (s_after["per_class"][cls]["accuracy"]
              - s_before["per_class"][cls]["accuracy"])
        for cls in CLASSES
    }

    # Paired significance test — match predictions by query text
    before_by_query = {p["query"]: p["correct"] for p in before["predictions"]}
    after_by_query = {p["query"]: p["correct"] for p in after["predictions"]}
    shared = sorted(set(before_by_query) & set(after_by_query))
    before_correct = [before_by_query[q] for q in shared]
    after_correct = [after_by_query[q] for q in shared]
    mcnemar = mcnemar_test(before_correct, after_correct) if shared else None

    return {
        "overall_delta":   s_after["overall_accuracy"] - s_before["overall_accuracy"],
        "per_class_delta": per_class_delta,
        "before":          s_before,
        "after":           s_after,
        "mcnemar":         mcnemar,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "Evaluation Results") -> None:
    """Print a formatted accuracy report to stdout."""
    s = score(results)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Model : {results.get('model', '?')}")
    print(f"  Run   : {results.get('timestamp', '?')[:19]}")
    print(f"{'='*60}")
    lo, hi = s["overall_ci"]
    print(f"\n  Overall accuracy: {_fmt_pct(s['overall_accuracy'])}"
          f"  {_fmt_ci(lo, hi)}  ({s['overall_correct']}/{s['n']})")

    latencies = [p["latency_ms"] for p in results["predictions"] if "latency_ms" in p]
    if latencies:
        print(f"  Latency:          {fmt_latency(compute_latency_stats(latencies))}")
    print()

    print("  Per-class accuracy:")
    for cls in CLASSES:
        cs  = s["per_class"][cls]
        bar = "█" * int(cs["accuracy"] * 20)
        clo, chi = cs["ci"]
        print(f"    {cls:<16s}  {_fmt_pct(cs['accuracy'])}"
              f"  {_fmt_ci(clo, chi)}  ({cs['correct']}/{cs['n']})  {bar}")

    wrong = [p for p in results["predictions"] if not p["correct"]]
    if wrong:
        print(f"\n  Misclassified ({len(wrong)}):")
        for p in wrong:
            print(f"    ✗  expected={p['expected']:<14s}  "
                  f"got={p['predicted']!r}")
            print(f"       \"{p['query'][:70]}\"")
    print()


def print_comparison(
    comparison: dict,
    labels: tuple[str, str] = ("Before", "After"),
) -> None:
    """Print a before/after delta table to stdout."""
    before_label, after_label = labels
    c = comparison

    print(f"\n{'='*60}")
    print(f"  Accuracy Comparison:  {before_label}  →  {after_label}")
    print(f"{'='*60}")

    delta = c["overall_delta"]
    sign  = "+" if delta >= 0 else ""
    print(f"\n  Overall:  {_fmt_pct(c['before']['overall_accuracy'])}  →  "
          f"{_fmt_pct(c['after']['overall_accuracy'])}"
          f"   ({sign}{_fmt_pct(delta)})", end="")
    if c.get("mcnemar"):
        p = c["mcnemar"]["p_value"]
        sig = " *" if c["mcnemar"]["significant_at_05"] else " n.s."
        print(f"  p={p:.3f}{sig}")
    else:
        print()
    print()

    print("  Per-class:")
    for cls in CLASSES:
        b_acc = c["before"]["per_class"][cls]["accuracy"]
        a_acc = c["after"]["per_class"][cls]["accuracy"]
        d     = c["per_class_delta"][cls]
        sign  = "+" if d >= 0 else ""
        arrow = "▲" if d > 0.0001 else ("▼" if d < -0.0001 else "═")
        print(f"    {cls:<16s}  {_fmt_pct(b_acc)}  →  {_fmt_pct(a_acc)}"
              f"   {arrow} {sign}{_fmt_pct(d)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate gemma3 intent classification accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save raw results to this JSON file (e.g. results/baseline.json)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two saved result files — no servers required",
    )
    args = parser.parse_args()

    if args.compare:
        before_data = load_results(args.compare[0])
        after_data  = load_results(args.compare[1])
        print_report(before_data, title=f"Before  ({Path(args.compare[0]).name})")
        print_report(after_data,  title=f"After   ({Path(args.compare[1]).name})")
        print_comparison(
            compare(before_data, after_data),
            labels=(Path(args.compare[0]).stem, Path(args.compare[1]).stem),
        )
    else:
        # Live evaluation — requires llama-server instances to be running
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.engine.inference.client import SmallLanguageModelClient  # noqa: E402

        async def _main() -> None:
            client = SmallLanguageModelClient.create_with_auto_detection()
            print(f"\nRunning intent classification eval "
                  f"({len(TEST_SET)} queries)…")
            results = await run_eval(client)
            print_report(results)
            if args.save:
                save_results(results, args.save)

        asyncio.run(_main())
