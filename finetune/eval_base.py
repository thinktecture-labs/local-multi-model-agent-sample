"""
Shared utilities for all model evaluation scripts.

Provides common I/O, formatting, statistical testing, and CLI helpers so that
eval_gemma3.py, eval_tool_routing.py, eval_embeddinggemma.py, and eval_vision.py
stay focused on their model-specific logic.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_pct(v: float) -> str:
    """Format a float (0–1) as a percentage string like '75.0%'."""
    return f"{v * 100:.1f}%"


def fmt_latency(stats: dict) -> str:
    """Format latency stats as 'p50=123ms  p95=456ms  mean=234ms'."""
    return (
        f"p50={stats['p50_ms']:.0f}ms  "
        f"p95={stats['p95_ms']:.0f}ms  "
        f"mean={stats['mean_ms']:.0f}ms"
    )


def fmt_ci(lower: float, upper: float) -> str:
    """Format a confidence interval as '[85.2%, 94.8%]'."""
    return f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"


def fmt_pct_with_ci(k: int, n: int) -> str:
    """Format accuracy with Wilson CI: '95.0% [89.3%, 98.1%]'."""
    if n == 0:
        return "N/A"
    lo, hi = wilson_ci(k, n)
    return f"{k / n * 100:.1f}% {fmt_ci(lo, hi)}"


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------

def compute_latency_stats(latencies_ms: list[float]) -> dict:
    """Compute p50, p95, p99, mean, min, max from a list of latencies in ms."""
    if not latencies_ms:
        return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "mean_ms": 0, "min_ms": 0, "max_ms": 0}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "p50_ms":  s[int(n * 0.50)],
        "p95_ms":  s[min(int(n * 0.95), n - 1)],
        "p99_ms":  s[min(int(n * 0.99), n - 1)],
        "mean_ms": sum(s) / n,
        "min_ms":  s[0],
        "max_ms":  s[-1],
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    """Persist evaluation results to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {path}")


def load_results(path: str) -> dict:
    """Load evaluation results from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Statistical testing — no external dependencies (stdlib only)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation for small n or extreme p.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Z-score for the confidence level (1.96 = 95% CI).

    Returns:
        (lower, upper) bounds as floats in [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    correct: list[bool],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap percentile confidence interval for accuracy.

    Args:
        correct: List of True/False per query.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (0.95 = 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds as floats in [0, 1].
    """
    n = len(correct)
    if n == 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(correct, k=n)
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1.0 - ci
    lo_idx = int(math.floor(alpha / 2.0 * n_boot))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    return (means[max(0, lo_idx)], means[min(n_boot - 1, hi_idx)])


def mcnemar_test(
    before_correct: list[bool],
    after_correct: list[bool],
) -> dict:
    """
    McNemar's test for paired nominal data (same queries, two models).

    Uses the chi-squared approximation with continuity correction.
    P-value computed via math.erfc for chi-squared with df=1.

    Args:
        before_correct: List of True/False for each query (model A).
        after_correct: List of True/False for each query (model B).

    Returns:
        Dict with chi2, p_value, n_discordant, b (regressed), c (improved),
        and significant_at_05 flag.
    """
    assert len(before_correct) == len(after_correct), "Paired lists must be same length"

    # b = correct before, wrong after (regressions)
    # c = wrong before, correct after (improvements)
    b = sum(1 for bv, av in zip(before_correct, after_correct) if bv and not av)
    c = sum(1 for bv, av in zip(before_correct, after_correct) if not bv and av)
    n_disc = b + c

    if n_disc == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "n_discordant": 0,
            "b": b,
            "c": c,
            "significant_at_05": False,
        }

    # Chi-squared with continuity correction (Edwards)
    chi2 = (abs(b - c) - 1) ** 2 / n_disc if n_disc > 0 else 0.0

    # P-value for chi-squared df=1: p = erfc(sqrt(chi2 / 2))
    p_value = math.erfc(math.sqrt(chi2 / 2.0))

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n_discordant": n_disc,
        "b": b,
        "c": c,
        "significant_at_05": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Train/eval overlap detection
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: str, b: str) -> float:
    """Word-set Jaccard similarity between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def check_eval_training_overlap(
    eval_queries: list[str],
    training_path: str,
    query_key: str = "input",
    threshold: float = 0.7,
) -> list[dict]:
    """
    Check for eval/training data leakage via Jaccard word-set similarity.

    Loads training data from a JSONL file, extracts the query_key field,
    and compares each eval query against all training queries.

    Returns:
        List of {eval_query, train_query, similarity} for any pair above threshold.
    """
    train_path = Path(training_path)
    if not train_path.exists():
        return []

    train_queries: list[str] = []
    with train_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            q = entry.get(query_key, "")
            if q:
                train_queries.append(q)

    overlaps: list[dict] = []
    for eq in eval_queries:
        for tq in train_queries:
            sim = _jaccard_similarity(eq, tq)
            if sim >= threshold:
                overlaps.append({
                    "eval_query": eq,
                    "train_query": tq,
                    "similarity": round(sim, 3),
                })
    return overlaps


# ---------------------------------------------------------------------------
# Eval data loaders
# ---------------------------------------------------------------------------

_EVAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval-data"


def load_eval_jsonl(filename: str) -> list[dict]:
    """Load a JSONL eval data file from data/eval-data/."""
    path = _EVAL_DATA_DIR / filename
    items: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_eval_json(filename: str):
    """Load a JSON eval data file from data/eval-data/."""
    path = _EVAL_DATA_DIR / filename
    with path.open() as f:
        return json.load(f)
