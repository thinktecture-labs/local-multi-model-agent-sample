"""
Intent LogReg Classifier Evaluator — compare LogReg vs generative accuracy.

Uses the same 180-query test set as eval_gemma3.py for direct comparison.

Usage:
  python -m finetune.eval_intent_logreg                                  # run + print report
  python -m finetune.eval_intent_logreg --save results/logreg_intent.json  # save results
  python -m finetune.eval_intent_logreg --compare results/finetuned_gemma3.json results/logreg_intent.json

The --compare flag works with any eval_gemma3.py saved results, so you can
directly compare generative vs LogReg on the same queries.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

from finetune.eval_base import (
    compute_latency_stats,
    fmt_latency,
    fmt_pct as _fmt_pct,
    fmt_ci as _fmt_ci,
    load_results,
    save_results,
    wilson_ci,
)
from finetune.eval_gemma3 import (
    CLASSES,
    TEST_SET,
    compare,
    print_comparison,
    print_report,
    score,
)


# ---------------------------------------------------------------------------
# LogReg evaluation runner
# ---------------------------------------------------------------------------

async def classify_query_logreg(classifier, query: str) -> str:
    """Classify a single query using the LogReg classifier."""
    intent, _ = await classifier.classify(query)
    return intent.value


async def run_eval(
    client,
    test_set: list[dict[str, str]] | None = None,
) -> dict:
    """Classify every query using LogReg and return results."""
    if test_set is None:
        test_set = TEST_SET

    from src.engine.agent.intent_classifier_logreg import LogRegIntentClassifier

    classifier = LogRegIntentClassifier(client)
    if not classifier.available:
        print("ERROR: LogReg model not found. Run: python -m training.train_intent_logreg")
        sys.exit(1)

    predictions: list[dict] = []
    for item in test_set:
        t0 = time.perf_counter()
        predicted = await classify_query_logreg(classifier, item["query"])
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
        "model":       "logreg_on_embeddinggemma",
        "n":           len(predictions),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LogReg intent classification accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save raw results to JSON file",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"),
        help="Compare two saved result files (works with eval_gemma3 results too)",
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
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.engine.inference.client import SmallLanguageModelClient

        async def _main() -> None:
            client = SmallLanguageModelClient.create_with_auto_detection()
            print(f"\nRunning LogReg intent classification eval "
                  f"({len(TEST_SET)} queries)…")
            results = await run_eval(client)
            print_report(results, title="LogReg Intent Evaluation Results")
            if args.save:
                save_results(results, args.save)

        asyncio.run(_main())
