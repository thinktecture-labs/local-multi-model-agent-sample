"""
Vision Evaluator — measure gemma3-4b-vision accuracy on image understanding tasks.

Uses a fixed labelled test set of 10 image+query pairs across 3 sample images
to produce a reproducible keyword-hit accuracy score. Run once before
fine-tuning to establish a baseline, then again after to quantify improvement.

Usage:
  python -m finetune.eval_vision                                        # run + print report
  python -m finetune.eval_vision --save results/baseline_vision.json    # save raw results
  python -m finetune.eval_vision --compare before.json after.json       # show delta

Demo talk workflow:
  1. python -m finetune.eval_vision --save results/baseline_vision.json
  2. python -m finetune.pipeline --train-all
  3. python -m finetune.eval_vision --save results/finetuned_vision.json
  4. python -m finetune.eval_vision --compare results/baseline_vision.json results/finetuned_vision.json
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime
from pathlib import Path

from finetune._scenario import SCENARIO_NAME as _SCENARIO
from finetune.eval_base import (
    compute_latency_stats,
    fmt_latency,
    fmt_pct as _fmt_pct,
    fmt_pct_with_ci,
    load_results,
    save_results,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# Fixed labelled test sets — 10 image+query pairs each, across 3 sample images.
# Each item specifies the image filename, a natural-language query, and
# expected keywords that should appear in a correct response.
# ---------------------------------------------------------------------------

from src.engine.inference.config import SCENARIO_CONFIG as _SC

# Images live under the scenario-configured demo_images_dir. The Nextera test set
# is the only one shipped in this repo; add per-scenario test sets as needed.
IMAGES_DIR = Path(__file__).parent.parent / _SC.demo_images_dir.lstrip("./")

_VALID_IMAGES_NEXTERA = {"revenue_chart.png", "pricing_table.png", "architecture_diagram.png"}

_TEST_SET_NEXTERA: list[dict] = [
    # --- revenue_chart.png (4 items) ---
    {
        "query": "What is the highest revenue quarter?",
        "image": "revenue_chart.png",
        "expected_keywords": ["Q4", "2024", "103"],
    },
    {
        "query": "Is revenue growing or declining?",
        "image": "revenue_chart.png",
        "expected_keywords": ["growing", "growth", "increasing", "upward"],
    },
    {
        "query": "What was the Q1 2023 revenue?",
        "image": "revenue_chart.png",
        "expected_keywords": ["18", "Q1"],
    },
    {
        "query": "Describe the chart",
        "image": "revenue_chart.png",
        "expected_keywords": ["revenue", "quarter", "bar", "chart"],
    },
    # --- pricing_table.png (3 items) ---
    {
        "query": "What is the monthly price of Enterprise?",
        "image": "pricing_table.png",
        "expected_keywords": ["3500", "enterprise"],
    },
    {
        "query": "How many pricing tiers are shown?",
        "image": "pricing_table.png",
        "expected_keywords": ["3", "three"],
    },
    {
        "query": "What is included in the Professional plan?",
        "image": "pricing_table.png",
        "expected_keywords": ["professional"],
    },
    # --- architecture_diagram.png (3 items) ---
    {
        "query": "How many models are shown?",
        "image": "architecture_diagram.png",
        "expected_keywords": ["3", "three", "4", "four"],
    },
    {
        "query": "What is the Seer model?",
        "image": "architecture_diagram.png",
        "expected_keywords": ["seer", "vision", "image", "multimodal"],
    },
    {
        "query": "Describe the architecture",
        "image": "architecture_diagram.png",
        "expected_keywords": ["model", "architecture", "agent", "gemma"],
    },
]

VALID_IMAGES = _VALID_IMAGES_NEXTERA
TEST_SET = _TEST_SET_NEXTERA

IMAGE_NAMES = sorted(VALID_IMAGES)


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------

def _load_image_b64(filename: str) -> str:
    """Load an image from data/demo-images/ and return as base64 string."""
    path = IMAGES_DIR / filename
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def evaluate_query(client, query: str, image_b64: str) -> str:
    """Send an image + query to the vision model and return the response text."""
    response = await client.generate_vision(
        prompt=query,
        images=[image_b64],
        temperature=0.0,
        max_tokens=200,
    )
    return response.content.strip()


def check_keywords(response: str, expected_keywords: list[str]) -> bool:
    """Check if ANY expected keyword appears in the response (case-insensitive)."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in expected_keywords)


async def run_eval(
    client,
    test_set: list[dict] | None = None,
) -> dict:
    """
    Evaluate every image+query pair in test_set against the live vision model.

    Returns a results dict suitable for score(), print_report(), and save_results().
    """
    if test_set is None:
        test_set = TEST_SET

    # Resolve the vision model name for the results record
    try:
        from src.engine.inference.client import SmallLanguageModelRole
        model_name = client.models.get(SmallLanguageModelRole.VISION, "unknown")
    except Exception:
        model_name = "unknown"

    # Pre-load images to avoid repeated disk reads
    image_cache: dict[str, str] = {}

    predictions: list[dict] = []
    for item in test_set:
        filename = item["image"]
        if filename not in image_cache:
            image_cache[filename] = _load_image_b64(filename)

        t0 = time.perf_counter()
        response_text = await evaluate_query(client, item["query"], image_cache[filename])
        latency_ms = (time.perf_counter() - t0) * 1000
        correct = check_keywords(response_text, item["expected_keywords"])
        predictions.append({
            "query":             item["query"],
            "image":             filename,
            "expected_keywords": item["expected_keywords"],
            "response":          response_text,
            "correct":           correct,
            "latency_ms":        round(latency_ms, 1),
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
    Compute per-image and overall accuracy from a results dict.

    Returns a scoring dict with keys:
      overall_accuracy, overall_correct, n, per_image
    """
    preds = results["predictions"]
    n = len(preds)
    overall_correct = sum(1 for p in preds if p["correct"])
    overall_accuracy = overall_correct / n if n else 0.0

    per_image: dict[str, dict] = {}
    for img in IMAGE_NAMES:
        img_preds = [p for p in preds if p["image"] == img]
        n_img = len(img_preds)
        n_correct = sum(1 for p in img_preds if p["correct"])
        per_image[img] = {
            "n":        n_img,
            "correct":  n_correct,
            "accuracy": n_correct / n_img if n_img else 0.0,
        }

    return {
        "overall_accuracy": overall_accuracy,
        "overall_correct":  overall_correct,
        "n":                n,
        "per_image":        per_image,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(before: dict, after: dict) -> dict:
    """
    Compute accuracy deltas between two result snapshots.

    Returns a comparison dict with overall_delta and per-image deltas.
    Positive values = improvement after fine-tuning.
    """
    s_before = score(before)
    s_after  = score(after)

    per_image_delta: dict[str, float] = {
        img: (s_after["per_image"][img]["accuracy"]
              - s_before["per_image"][img]["accuracy"])
        for img in IMAGE_NAMES
    }

    return {
        "overall_delta":   s_after["overall_accuracy"] - s_before["overall_accuracy"],
        "per_image_delta": per_image_delta,
        "before":          s_before,
        "after":           s_after,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "Vision Evaluation Results") -> None:
    """Print a formatted accuracy report to stdout."""
    s = score(results)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Model : {results.get('model', '?')}")
    print(f"  Run   : {results.get('timestamp', '?')[:19]}")
    print(f"{'='*60}")
    print(f"\n  Overall accuracy: {fmt_pct_with_ci(s['overall_correct'], s['n'])}"
          f"  ({s['overall_correct']}/{s['n']} correct)")
    if s['n'] < 30:
        print(f"  ⚠️  Small N (n={s['n']}); a single failure shifts the headline ~{round(100 / s['n'])}pp. CI is wide.")

    latencies = [p["latency_ms"] for p in results["predictions"] if "latency_ms" in p]
    if latencies:
        print(f"  Latency:          {fmt_latency(compute_latency_stats(latencies))}")
    print()

    print("  Per-image accuracy:")
    for img in IMAGE_NAMES:
        cs  = s["per_image"][img]
        bar = "█" * int(cs["accuracy"] * 20)
        print(f"    {img:<30s}  {_fmt_pct(cs['accuracy'])}"
              f"  ({cs['correct']}/{cs['n']})  {bar}")

    wrong = [p for p in results["predictions"] if not p["correct"]]
    if wrong:
        print(f"\n  Incorrect ({len(wrong)}):")
        for p in wrong:
            print(f"    ✗  image={p['image']:<30s}  "
                  f"keywords={p['expected_keywords']}")
            print(f"       \"{p['query'][:70]}\"")
            print(f"       response: \"{p['response'][:80]}\"")
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
          f"   ({sign}{_fmt_pct(delta)})\n")

    print("  Per-image:")
    for img in IMAGE_NAMES:
        b_acc = c["before"]["per_image"][img]["accuracy"]
        a_acc = c["after"]["per_image"][img]["accuracy"]
        d     = c["per_image_delta"][img]
        sign  = "+" if d >= 0 else ""
        arrow = "▲" if d > 0.0001 else ("▼" if d < -0.0001 else "═")
        print(f"    {img:<30s}  {_fmt_pct(b_acc)}  →  {_fmt_pct(a_acc)}"
              f"   {arrow} {sign}{_fmt_pct(d)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate gemma3-4b vision model accuracy on image understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save raw results to this JSON file (e.g. results/baseline_vision.json)",
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
            print(f"\nRunning vision eval "
                  f"({len(TEST_SET)} image+query pairs)...")
            results = await run_eval(client)
            print_report(results)
            if args.save:
                save_results(results, args.save)

        asyncio.run(_main())
