"""
RAG Ground-Truth Evaluator — measure factual *faithfulness* of RAG responses.

A response is "correct" only when **both** hold:
  1. coverage: every expected keyword from the ground-truth set appears in
     the response (case-insensitive substring). This is the standard fact-
     coverage check — has the model included the canonical facts at all?
  2. value_grounded: every numeric value in the response is either expected
     OR present in the retrieved vector_search context. Catches hallucinated
     numbers — the failure mode where the synthesis says "Enterprise costs
     €3,500/month and includes the new Quantum Module" alongside an
     accurate price (coverage passes) by introducing fabricated detail
     (value_grounded fails).

The earlier "ANY keyword present" substring check graded faithful and
unfaithful syntheses identically and was the single weakest link in the
project's eval rigor. See REVIEW_2026-05-15_202209.md H1.

Per-query failures (llama-server compute errors, timeouts) are now caught
and recorded with intent="error" instead of aborting the run — the eval
is meant to survive bad luck on individual queries.

Usage:
  python -m finetune.eval_rag_groundtruth
  python -m finetune.eval_rag_groundtruth --save results/rag_gt.json
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from datetime import datetime

from finetune.eval_base import (
    compute_latency_stats,
    fmt_latency,
    fmt_pct as _fmt_pct,
    load_eval_jsonl,
    save_results,
)
from finetune._scenario import SCENARIO_NAME as _SCENARIO


# ---------------------------------------------------------------------------
# Test set — loaded from data/eval-data/
# ---------------------------------------------------------------------------

_EVAL_FILE = f"eval_rag_groundtruth_{_SCENARIO}.jsonl"

TEST_SET: list[dict] = load_eval_jsonl(_EVAL_FILE)


# ---------------------------------------------------------------------------
# Faithfulness scoring primitives
# ---------------------------------------------------------------------------

# Numbers as they typically appear in responses and source text. Captures
# integers, decimals, comma-grouped thousands, and percentages. Trailing
# `%` is kept so "12%" and "12" are different tokens.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

# Numbers we don't care about — common boilerplate that's not a real claim.
# Years, single-digit list indices, and zero are noisy and produce false
# positives in the hallucination check.
_TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}


def _normalize_number(token: str) -> str:
    """Strip commas; keep trailing '%' marker; return canonical form.

    Whole-number floats collapse to their integer form so "3000.0" and
    "3,000" both normalize to "3000".
    """
    is_pct = token.endswith("%")
    bare = token.rstrip("%").replace(",", "")
    try:
        val = float(bare)
        if val == int(val):
            bare = str(int(val))
        else:
            bare = str(val)
    except ValueError:
        return token
    return bare + ("%" if is_pct else "")


def _extract_numbers(text: str) -> set[str]:
    """Pull all non-trivial numbers from `text` as normalized tokens."""
    found: set[str] = set()
    for match in _NUMBER_RE.findall(text or ""):
        norm = _normalize_number(match)
        if norm in _TRIVIAL_NUMBERS:
            continue
        # Skip trailing-period artefacts and dangling commas
        if norm in {"", ".", "-"}:
            continue
        found.add(norm)
    return found


def _coverage(response: str, expected: list[str]) -> tuple[float, list[str], list[str]]:
    """Fraction of expected keywords found (case-insensitive substring)."""
    if not expected:
        return 0.0, [], []
    resp_lower = response.lower()
    found = [kw for kw in expected if kw.lower() in resp_lower]
    missing = [kw for kw in expected if kw.lower() not in resp_lower]
    return len(found) / len(expected), found, missing


def _retrieved_context(steps) -> str:
    """Concatenate every retrieved document's content from `vector_search` steps."""
    chunks: list[str] = []
    for step in steps or []:
        if getattr(step, "action", None) != "vector_search":
            continue
        docs = (getattr(step, "details", {}) or {}).get("documents") or []
        for d in docs:
            content = d.get("content") if isinstance(d, dict) else None
            if content:
                chunks.append(content)
    return "\n".join(chunks)


def _value_grounded(
    response: str, expected_keywords: list[str], retrieved_context: str,
) -> tuple[bool, list[str]]:
    """Every numeric value in the response must be sourced from expected or context.

    Returns (grounded, hallucinated_numbers). When there are no numbers in
    the response at all, returns (True, []) — nothing to hallucinate.
    """
    resp_numbers = _extract_numbers(response)
    if not resp_numbers:
        return True, []
    expected_text = " ".join(expected_keywords or [])
    allowed = _extract_numbers(expected_text) | _extract_numbers(retrieved_context)
    hallucinated = sorted(n for n in resp_numbers if n not in allowed)
    return (len(hallucinated) == 0), hallucinated


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_eval(agent, strict_coverage: bool = True) -> dict:
    """Run every test query through the agent and grade faithfulness.

    strict_coverage=True (default): a query is correct only when **every**
        expected keyword is in the response AND no hallucinated numbers.
    strict_coverage=False: legacy mode — at least one keyword present
        (kept for back-compat comparison plots only; not the headline number).
    """
    predictions: list[dict] = []

    for item in TEST_SET:
        t0 = time.perf_counter()
        try:
            result = await agent.process(item["query"])
            latency_ms = (time.perf_counter() - t0) * 1000

            coverage_pct, found_kw, missing_kw = _coverage(result.response, item["expected_keywords"])
            ctx = _retrieved_context(result.steps)
            grounded, hallucinated = _value_grounded(result.response, item["expected_keywords"], ctx)

            if strict_coverage:
                correct = (coverage_pct >= 1.0) and grounded
            else:
                # Legacy semantics: ANY keyword + don't enforce grounding
                correct = coverage_pct > 0.0

            predictions.append({
                "query": item["query"],
                "source_doc": item.get("source_doc", ""),
                "expected_keywords": item["expected_keywords"],
                "found_keywords": found_kw,
                "missing_keywords": missing_kw,
                "coverage_pct": round(coverage_pct, 3),
                "value_grounded": grounded,
                "hallucinated_numbers": hallucinated,
                "response": result.response[:500],
                "correct": correct,
                "intent": result.intent.value,
                "latency_ms": round(latency_ms, 1),
                "error": None,
            })
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            predictions.append({
                "query": item["query"],
                "source_doc": item.get("source_doc", ""),
                "expected_keywords": item["expected_keywords"],
                "found_keywords": [],
                "missing_keywords": item["expected_keywords"],
                "coverage_pct": 0.0,
                "value_grounded": False,
                "hallucinated_numbers": [],
                "response": "",
                "correct": False,
                "intent": "error",
                "latency_ms": round(latency_ms, 1),
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "timestamp": datetime.now().isoformat(),
        "scenario": _SCENARIO,
        "n": len(predictions),
        "strict_coverage": strict_coverage,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    """Compute accuracy metrics from eval results."""
    preds = results["predictions"]
    if not preds:
        return {"overall": 0.0, "correct": 0, "total": 0, "per_doc": {}}

    correct = sum(1 for p in preds if p["correct"])
    overall = correct / len(preds)

    # Faithfulness sub-metrics — visible separately so it's clear *which*
    # axis is failing on a regression.
    full_coverage = sum(1 for p in preds if p.get("coverage_pct", 0.0) >= 1.0)
    any_coverage  = sum(1 for p in preds if p.get("coverage_pct", 0.0) > 0.0)
    grounded      = sum(1 for p in preds if p.get("value_grounded"))
    errors        = sum(1 for p in preds if p.get("error"))

    # Per source document
    per_doc: dict[str, dict] = {}
    docs = sorted(set(p.get("source_doc", "") for p in preds))
    for doc in docs:
        if not doc:
            continue
        doc_preds = [p for p in preds if p.get("source_doc") == doc]
        doc_correct = sum(1 for p in doc_preds if p["correct"])
        per_doc[doc] = {
            "accuracy": doc_correct / len(doc_preds),
            "correct": doc_correct,
            "total": len(doc_preds),
        }

    latencies = [p["latency_ms"] for p in preds if not p.get("error")]
    return {
        "overall": overall,
        "correct": correct,
        "total": len(preds),
        "errors": errors,
        "full_coverage": full_coverage,
        "any_coverage": any_coverage,
        "value_grounded": grounded,
        "per_doc": per_doc,
        "latency": compute_latency_stats(latencies) if latencies else None,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    """Print a formatted evaluation report."""
    s = score(results)

    print("\n" + "=" * 60)
    print(f"  RAG Ground-Truth Evaluation — {results.get('scenario', '?')}")
    mode = "STRICT (ALL keywords + value-grounded)" if results.get("strict_coverage", True) else "LEGACY (ANY keyword)"
    print(f"  Mode  : {mode}")
    print("=" * 60)
    print(f"\n  Overall:        {_fmt_pct(s['overall'])} ({s['correct']}/{s['total']})")
    print(f"  Full coverage:  {_fmt_pct(s['full_coverage']/s['total'])} ({s['full_coverage']}/{s['total']})  — every expected keyword present")
    print(f"  Any  coverage:  {_fmt_pct(s['any_coverage']/s['total'])} ({s['any_coverage']}/{s['total']})  — at least one expected keyword (legacy bar)")
    print(f"  Value grounded: {_fmt_pct(s['value_grounded']/s['total'])} ({s['value_grounded']}/{s['total']})  — no hallucinated numbers")
    if s["errors"]:
        print(f"  Errors:         {s['errors']}  — llama-server compute errors or timeouts")

    if s.get("per_doc"):
        print("\n  Per document:")
        for doc, data in sorted(s["per_doc"].items()):
            bar = "█" * int(data["accuracy"] * 20) + "░" * (20 - int(data["accuracy"] * 20))
            print(f"    {doc:40s} {bar} {_fmt_pct(data['accuracy'])} ({data['correct']}/{data['total']})")

    if s.get("latency"):
        print(f"\n  Latency: {fmt_latency(s['latency'])}")

    # Show incorrect predictions
    incorrect = [p for p in results["predictions"] if not p["correct"]]
    if incorrect:
        print(f"\n  Incorrect ({len(incorrect)}):")
        for p in incorrect:
            print(f"    ✗ {p['query'][:65]}")
            if p.get("error"):
                print(f"      error: {p['error'][:120]}")
                continue
            if p.get("missing_keywords"):
                print(f"      missing keywords: {p['missing_keywords']}")
            if p.get("hallucinated_numbers"):
                print(f"      hallucinated numbers: {p['hallucinated_numbers']}")
            print(f"      response: {p['response'][:100]}…")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ground-truth evaluation")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy 'ANY keyword present' scoring (for back-compat comparison).",
    )
    args = parser.parse_args()

    if not TEST_SET:
        print(f"No RAG ground-truth test set for scenario '{_SCENARIO}'.")
        sys.exit(1)

    from src.engine.inference.client import SmallLanguageModelClient
    from src.engine.inference.config import SCENARIO_CONFIG
    from src.engine.knowledge.vector_store import VectorStore
    from src.engine.tools import create_default_registry
    from src.engine.agent import SmallLanguageModelAgentOrchestrator

    print("Setting up agent...")
    client = SmallLanguageModelClient.create_with_auto_detection()
    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    vector_store.set_client(client)
    tools = create_default_registry(vector_store=vector_store)
    agent = SmallLanguageModelAgentOrchestrator(client, tools)

    strict = not args.legacy
    print(f"\nRunning RAG ground-truth eval ({len(TEST_SET)} queries, scenario={_SCENARIO}, strict={strict})...")
    results = await run_eval(agent, strict_coverage=strict)
    print_report(results)

    if args.save:
        save_results(results, args.save)
        print(f"  Results saved to {args.save}")


# Back-compat alias — kept so other modules importing the old name don't break.
check_keywords = lambda response, expected: _coverage(response, expected)[0] > 0.0


if __name__ == "__main__":
    asyncio.run(main())
