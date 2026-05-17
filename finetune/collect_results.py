"""
Benchmark Matrix Collector — aggregate per-machine eval JSON files into a single matrix.

Scans results/ for JSON files with a _meta block (written by run_all_evals.sh),
calls the appropriate score() function for each eval type, and produces a
consolidated benchmark_matrix.json that can be loaded by visualisation scripts.

Usage:
  python -m finetune.collect_results                                 # read results/, print table
  python -m finetune.collect_results --dir path/to/results          # alternate results dir
  python -m finetune.collect_results --matrix results/matrix.json   # also write matrix JSON
  python -m finetune.collect_results --matrix matrix.json --quiet   # write only, no table

Output schema (schema_version=1):
  {
    "schema_version": 1,
    "generated": "<ISO timestamp>",
    "runs": [
      {
        "machine":             str,   # e.g. "gpu-host-1"
        "gpu":                 str,   # e.g. "NVIDIA-RTX-PRO-6000"
        "platform":            str,   # "Darwin" | "Linux"
        "eval_type":           str,   # "tool_selection" | "multi_step" | "intent" | ...
        "model_tag":           str,   # e.g. "qwen3.5-4b-ft-v8"
        "port":                int,
        "run_ts":              str,   # "20260320T123456"
        "n":                   int,   # number of test queries
        "primary_metric_name": str,   # canonical accuracy field for this eval type
        "primary_metric":      float, # value of that field (0.0–1.0)
        "metrics":             dict,  # full score() output
        "latency":             dict   # {p50_ms, p95_ms, p99_ms, mean_ms, min_ms, max_ms}
      },
      ...
    ]
  }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from finetune.eval_base import compute_latency_stats


# ---------------------------------------------------------------------------
# Eval-type registry: maps _meta.eval_type → (score_fn, primary_key, latency_path)
# latency_path: dot-path into the raw result dict to find latency_ms list
# ---------------------------------------------------------------------------

def _score_tool_selection(raw: dict) -> dict:
    from finetune.eval_tool_routing import score
    return score(raw)


def _score_multi_step(raw: dict) -> dict:
    from finetune.eval_multi_step import score
    return score(raw)


def _score_intent(raw: dict) -> dict:
    from finetune.eval_gemma3 import score
    return score(raw)


def _score_logreg_intent(raw: dict) -> dict:
    from finetune.eval_gemma3 import score  # same schema
    return score(raw)


def _score_embedding(raw: dict) -> dict:
    from finetune.eval_embeddinggemma import score
    return score(raw)


def _score_adversarial(raw: dict) -> dict:
    from finetune.eval_adversarial import score
    return score(raw)


def _score_vision(raw: dict) -> dict:
    from finetune.eval_vision import score
    return score(raw)


# (score_fn, primary_metric_name)
_EVAL_REGISTRY: dict[str, tuple] = {
    "tool_selection": (_score_tool_selection, "tool_accuracy"),
    "multi_step":     (_score_multi_step,     "tools_accuracy"),
    "intent":         (_score_intent,         "overall_accuracy"),
    "logreg_intent":  (_score_logreg_intent,  "overall_accuracy"),
    "embedding":      (_score_embedding,      "mrr_at_10"),
    "adversarial":    (_score_adversarial,    "overall_accuracy"),
    "vision":         (_score_vision,         "overall_accuracy"),
}


def _extract_latencies(raw: dict, eval_type: str) -> list[float]:
    """Pull per-prediction latency_ms values from any result dict."""
    preds = raw.get("predictions", [])
    return [p["latency_ms"] for p in preds if "latency_ms" in p]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(path: Path) -> dict | None:
    """
    Load one result JSON, score it, and return a normalised run dict.

    Returns None if the file has no _meta block (not produced by run_all_evals.sh).
    """
    with path.open() as f:
        raw = json.load(f)

    meta = raw.get("_meta")
    if not meta:
        return None

    eval_type = meta.get("eval_type", "unknown")
    entry = _EVAL_REGISTRY.get(eval_type)
    if entry is None:
        # Preserve the run in the matrix but without derived metrics
        metrics: dict = {}
        primary_metric_name = "n/a"
        primary_metric = 0.0
    else:
        score_fn, primary_metric_name = entry
        try:
            metrics = score_fn(raw)
        except Exception as exc:
            metrics = {"error": str(exc)}
        primary_metric = metrics.get(primary_metric_name, 0.0)

    latencies = _extract_latencies(raw, eval_type)
    latency_stats = compute_latency_stats(latencies) if latencies else {}

    return {
        "machine":             meta.get("machine", "unknown"),
        "gpu":                 meta.get("gpu", "unknown"),
        "platform":            meta.get("platform", "unknown"),
        "eval_type":           eval_type,
        "model_tag":           meta.get("model_tag", "unknown"),
        "port":                meta.get("port", 0),
        "run_ts":              meta.get("run_ts", ""),
        "source_file":         path.name,
        "n":                   metrics.get("n", raw.get("n", 0)),
        "primary_metric_name": primary_metric_name,
        "primary_metric":      primary_metric,
        "metrics":             metrics,
        "latency":             latency_stats,
    }


def collect(results_dir: str = "results") -> list[dict]:
    """Load and score all JSON files in results_dir that have a _meta block."""
    p = Path(results_dir)
    if not p.is_dir():
        return []

    runs: list[dict] = []
    for f in sorted(p.glob("*.json")):
        if f.name == "benchmark_matrix.json":
            continue
        run = process_file(f)
        if run is not None:
            runs.append(run)

    return runs


def build_matrix(runs: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "generated":      datetime.now().isoformat(),
        "runs":           runs,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_EVAL_ORDER = [
    "intent",
    "logreg_intent",
    "embedding",
    "tool_selection",
    "multi_step",
    "adversarial",
    "vision",
]

_EVAL_LABEL = {
    "intent":         "Intent (generative)",
    "logreg_intent":  "Intent (LogReg)",
    "embedding":      "Embedding",
    "tool_selection": "Tool selection",
    "multi_step":     "Multi-step reasoning",
    "adversarial":    "Adversarial robustness",
    "vision":         "Vision",
}


def print_matrix(runs: list[dict]) -> None:
    if not runs:
        print("  (no runs found — ensure results/*.json files have _meta blocks)")
        return

    # Group by eval_type
    by_type: dict[str, list[dict]] = {}
    for r in runs:
        by_type.setdefault(r["eval_type"], []).append(r)

    ordered_types = _EVAL_ORDER + [t for t in by_type if t not in _EVAL_ORDER]

    for et in ordered_types:
        if et not in by_type:
            continue
        group = by_type[et]
        label = _EVAL_LABEL.get(et, et)

        print(f"\n  ┌─ {label} ({'primary: ' + group[0]['primary_metric_name']})")

        # Sort: machine, then model_tag
        group_sorted = sorted(group, key=lambda r: (r["machine"], r["model_tag"]))

        col_w = {"machine": 20, "gpu": 28, "model": 22, "n": 5, "acc": 8, "p50": 7}
        header = (
            f"  │  {'Machine':<{col_w['machine']}}  "
            f"{'GPU':<{col_w['gpu']}}  "
            f"{'Model':<{col_w['model']}}  "
            f"{'N':>{col_w['n']}}  "
            f"{'Accuracy':>{col_w['acc']}}  "
            f"{'p50ms':>{col_w['p50']}}  "
            f"Run"
        )
        print(header)
        print("  │  " + "─" * (len(header) - 5))

        for r in group_sorted:
            acc = r["primary_metric"]
            acc_str = f"{acc * 100:.1f}%"
            p50 = r["latency"].get("p50_ms", 0)
            p50_str = f"{p50:.0f}" if p50 else "  —"
            ts = r["run_ts"][:13].replace("T", " ") if r["run_ts"] else "—"
            gpu = r["gpu"][:col_w["gpu"]]
            model = r["model_tag"][:col_w["model"]]
            machine = r["machine"][:col_w["machine"]]

            bar_len = int(acc * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(
                f"  │  {machine:<{col_w['machine']}}  "
                f"{gpu:<{col_w['gpu']}}  "
                f"{model:<{col_w['model']}}  "
                f"{r['n']:>{col_w['n']}}  "
                f"{acc_str:>{col_w['acc']}}  "
                f"{p50_str:>{col_w['p50']}}  "
                f"{ts}  {bar}"
            )

        print("  └" + "─" * (len(header) - 4))

    total_machines = len({r["machine"] for r in runs})
    total_models   = len({r["model_tag"] for r in runs})
    print(f"\n  {len(runs)} run(s) across {total_machines} machine(s), {total_models} model variant(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results into a comparison matrix",
    )
    parser.add_argument(
        "--dir", metavar="DIR", default="results",
        help="Directory containing result JSON files (default: results/)",
    )
    parser.add_argument(
        "--matrix", metavar="PATH",
        help="Write the aggregated benchmark_matrix.json to this path",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress table output (useful when only writing --matrix)",
    )
    args = parser.parse_args()

    runs = collect(args.dir)
    matrix = build_matrix(runs)

    if not args.quiet:
        print(f"\nBenchmark matrix — {len(runs)} result file(s) from '{args.dir}/'")
        print_matrix(runs)

    if args.matrix:
        out = Path(args.matrix)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(matrix, fh, indent=2)
        print(f"\n  Matrix written → {args.matrix}  ({len(runs)} runs)")
    elif not args.quiet:
        print(
            "\n  Tip: rerun with --matrix results/benchmark_matrix.json "
            "to persist the matrix for visualisation"
        )
