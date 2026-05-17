"""
Adversarial & Out-of-Distribution Evaluator — measure robustness of intent classification.

All queries in this set are adversarial or out-of-domain inputs that should be
classified as ``direct_answer`` (the safest fallback: no tool execution, no RAG
search).  "Accuracy" here means the model correctly avoided routing adversarial
input to tools or knowledge-base search.

Categories:
  off_topic       — non-Nextera domain questions
  injection       — prompt injection / jailbreak attempts
  multilang       — non-English queries (German, French, Spanish, etc.)
  gibberish       — random text, symbols, emoji noise
  sql_injection   — raw SQL / code as queries
  adversarial     — intent-confusion attacks designed to trick the classifier

Usage:
  python -m finetune.eval_adversarial                                     # generative only
  python -m finetune.eval_adversarial --pipeline                          # full pipeline (LogReg + filters)
  python -m finetune.eval_adversarial --compare                           # both + head-to-head
  python -m finetune.eval_adversarial --save results/adversarial.json     # save results
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
    load_eval_jsonl,
    save_results,
    wilson_ci,
)
from finetune._scenario import SCENARIO_NAME as _SCENARIO


# ---------------------------------------------------------------------------
# Fixed labelled test set — 60 adversarial / OOD queries
# Loaded from data/eval-data/ JSONL files (not inline).
# All expected intent: direct_answer (safest fallback)
# ---------------------------------------------------------------------------

CATEGORIES = [
    "off_topic",
    "injection",
    "multilang",
    "gibberish",
    "sql_injection",
    "adversarial",
]
TEST_SET = load_eval_jsonl(f"eval_adversarial_{_SCENARIO}.jsonl")


# ---------------------------------------------------------------------------
# Classify prompt — shared single source of truth
# ---------------------------------------------------------------------------

from src.engine.agent.types import CLASSIFY_PROMPT as _CLASSIFY_PROMPT, INTENT_LABELS as INTENT_CLASSES


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def classify_query_generative(client, query: str) -> str:
    """Classify a single query using the generative model only. Returns an intent label."""
    response = await client.generate(
        prompt=_CLASSIFY_PROMPT.format(query=query),
        temperature=0.0,
        max_tokens=50,
    )
    return _extract_intent(response.content)


def _extract_intent(text: str) -> str:
    """Extract intent label from model output — handles both terse and verbose formats."""
    raw = text.strip().lower()

    # Fast path: first line is exactly a known class (Gemma-FT style)
    first_line = raw.split("\n")[0].strip().replace(" ", "_").rstrip(".")
    if first_line in INTENT_CLASSES:
        return first_line

    # Slow path: scan full text for any known class label (verbose models)
    for label in sorted(INTENT_CLASSES, key=len, reverse=True):
        if label in raw:
            return label

    return "unknown"


async def classify_query_pipeline(classifier, query: str) -> str:
    """Classify using the full IntentClassifier pipeline (filters + LogReg + fallback)."""
    intent, _ = await classifier.classify(query)
    return intent.value


async def run_eval(
    client,
    test_set: list[dict[str, str]] | None = None,
    *,
    use_pipeline: bool = False,
) -> dict:
    """
    Classify every query in test_set.

    Args:
        client: SmallLanguageModelClient instance.
        test_set: List of {query, intent, category} dicts.
        use_pipeline: If True, use the full IntentClassifier (LogReg + filters + fallback)
                      instead of raw generative classification.

    Returns a results dict suitable for score() and print_report().
    """
    if test_set is None:
        test_set = TEST_SET

    classifier = None
    if use_pipeline:
        from src.engine.agent.intent_classifier import IntentClassifier
        classifier = IntentClassifier(client)

    try:
        from src.engine.inference.client import SmallLanguageModelRole
        model_name = client.models.get(SmallLanguageModelRole.INFERENCE, "unknown")
    except Exception:
        model_name = "unknown"

    mode = "pipeline" if use_pipeline else "generative"

    predictions: list[dict] = []
    for item in test_set:
        t0 = time.perf_counter()
        if use_pipeline:
            predicted = await classify_query_pipeline(classifier, item["query"])
        else:
            predicted = await classify_query_generative(client, item["query"])
        latency_ms = (time.perf_counter() - t0) * 1000
        predictions.append({
            "query":      item["query"],
            "expected":   item["intent"],
            "predicted":  predicted,
            "correct":    predicted == item["intent"],
            "category":   item["category"],
            "latency_ms": round(latency_ms, 1),
        })

    return {
        "timestamp":   datetime.now().isoformat(),
        "model":       model_name,
        "mode":        mode,
        "n":           len(predictions),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    """
    Compute overall and per-category robustness accuracy.

    "Accuracy" = fraction of adversarial queries correctly classified as
    direct_answer (i.e., not misrouted to tools or RAG).
    """
    preds = results["predictions"]
    n = len(preds)
    overall_correct = sum(1 for p in preds if p["correct"])
    overall_accuracy = overall_correct / n if n else 0.0

    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_preds = [p for p in preds if p["category"] == cat]
        n_cat = len(cat_preds)
        n_correct = sum(1 for p in cat_preds if p["correct"])
        per_category[cat] = {
            "n":        n_cat,
            "correct":  n_correct,
            "accuracy": n_correct / n_cat if n_cat else 0.0,
            "ci":       wilson_ci(n_correct, n_cat),
        }

    return {
        "overall_accuracy": overall_accuracy,
        "overall_correct":  overall_correct,
        "overall_ci":       wilson_ci(overall_correct, n),
        "n":                n,
        "per_category":     per_category,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "Adversarial / OOD Robustness") -> None:
    """Print a formatted robustness report to stdout."""
    s = score(results)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Model : {results.get('model', '?')}")
    print(f"  Run   : {results.get('timestamp', '?')[:19]}")
    print(f"{'='*60}")
    lo, hi = s["overall_ci"]
    print(f"\n  Overall robustness: {_fmt_pct(s['overall_accuracy'])}"
          f"  {_fmt_ci(lo, hi)}  ({s['overall_correct']}/{s['n']})")

    latencies = [p["latency_ms"] for p in results["predictions"] if "latency_ms" in p]
    if latencies:
        print(f"  Latency:            {fmt_latency(compute_latency_stats(latencies))}")
    print()

    print("  Per-category robustness:")
    for cat in CATEGORIES:
        cs  = s["per_category"][cat]
        bar = "+" * int(cs["accuracy"] * 20)
        clo, chi = cs["ci"]
        print(f"    {cat:<16s}  {_fmt_pct(cs['accuracy'])}"
              f"  {_fmt_ci(clo, chi)}  ({cs['correct']}/{cs['n']})  {bar}")

    wrong = [p for p in results["predictions"] if not p["correct"]]
    if wrong:
        print(f"\n  Misrouted ({len(wrong)}) — should have been direct_answer:")
        for p in wrong:
            print(f"    !  got={p['predicted']!r:<16s}  [{p['category']}]")
            print(f"       \"{p['query'][:70]}\"")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial / OOD evaluation")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--pipeline", action="store_true",
                        help="Use full IntentClassifier pipeline (LogReg + filters + fallback)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both generative and pipeline, then compare head-to-head")
    args = parser.parse_args()

    from src.engine.inference.client import SmallLanguageModelClient
    client = SmallLanguageModelClient.create_with_auto_detection()

    if args.compare:
        print("Running adversarial eval — generative ...")
        gen_results = await run_eval(client, use_pipeline=False)
        print_report(gen_results, title="Adversarial — Generative Only")

        print("Running adversarial eval — full pipeline ...")
        pipe_results = await run_eval(client, use_pipeline=True)
        print_report(pipe_results, title="Adversarial — Full Pipeline (LogReg + filters)")

        # Head-to-head comparison
        gen_correct = [p["correct"] for p in gen_results["predictions"]]
        pipe_correct = [p["correct"] for p in pipe_results["predictions"]]
        from finetune.eval_base import mcnemar_test
        mc = mcnemar_test(gen_correct, pipe_correct)
        gen_score = score(gen_results)
        pipe_score = score(pipe_results)

        print(f"{'='*60}")
        print(f"  Head-to-Head Comparison")
        print(f"{'='*60}")
        print(f"  Generative:  {_fmt_pct(gen_score['overall_accuracy'])} ({gen_score['overall_correct']}/{gen_score['n']})")
        print(f"  Pipeline:    {_fmt_pct(pipe_score['overall_accuracy'])} ({pipe_score['overall_correct']}/{pipe_score['n']})")
        print(f"  McNemar p:   {mc['p_value']:.4f} ({'significant' if mc['significant_at_05'] else 'not significant'})")
        print(f"  Improved:    {mc['c']} queries")
        print(f"  Regressed:   {mc['b']} queries")
        print()

        if args.save:
            save_results(pipe_results, args.save)
    else:
        use_pipeline = args.pipeline
        mode = "pipeline" if use_pipeline else "generative"
        print(f"Running adversarial / OOD evaluation ({mode}) ...")
        results = await run_eval(client, use_pipeline=use_pipeline)
        print_report(results)

        if args.save:
            save_results(results, args.save)


if __name__ == "__main__":
    asyncio.run(_main())
