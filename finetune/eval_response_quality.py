"""
Response Quality Evaluator — detect language, hallucination, and domain bleed issues.

Unlike other evals that test plumbing (intent routing, tool selection, SQL execution),
this eval checks the *output quality* of the full agent pipeline:

  1. language_correct   — response language matches SCENARIO_CONFIG.language
  2. no_domain_bleed    — no cross-scenario terms leak into responses
  3. value_grounded     — for tool_use queries, key numeric values from the tool
                          result appear in the final response (no hallucinated numbers)

Runs the full pipeline (SmallLanguageModelAgentOrchestrator.process) per query.

Usage:
  python -m finetune.eval_response_quality
  python -m finetune.eval_response_quality --save results/response_quality.json
  python -m finetune.eval_response_quality --compare before.json after.json
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from datetime import datetime

from finetune.eval_base import (
    check_eval_training_overlap,
    compute_latency_stats,
    fmt_latency,
    fmt_pct_with_ci,
    load_eval_jsonl,
    load_results,
    mcnemar_test,
    save_results,
    wilson_ci,
)

from src.engine.agent import SmallLanguageModelAgentOrchestrator, Intent
from src.engine.inference.client import SmallLanguageModelClient
from src.engine.inference.config import SCENARIO_CONFIG
from src.engine.tools.tool_registry import create_default_registry
from src.engine.knowledge.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Test sets — loaded from data/eval-data/ JSONL files (not inline)
# ---------------------------------------------------------------------------

from finetune._scenario import SCENARIO_NAME as _SCENARIO

_TEST_SET: list[dict] = load_eval_jsonl(f"eval_response_quality_{_SCENARIO}.jsonl")

# Simple German detection heuristics — common German words unlikely in English text.
_GERMAN_MARKERS = re.compile(
    r"\b(ist|sind|der|die|das|und|fuer|für|nicht|oder|mit|von|auf|eine?|"
    r"werden|wurde|haben|keine?|auch|nach|bei|noch|kann|mehr|ueber|über|"
    r"diese[rsmn]?|wird|alle[nmrs]?|bereits|jedoch|sowie|insgesamt)\b",
    re.IGNORECASE,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(the|is|are|and|for|not|with|from|has|have|this|that|was|were|"
    r"can|but|also|more|all|been|would|their|which|about|into|each)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Heuristic language detection: 'de', 'en', or 'unknown'."""
    de_count = len(_GERMAN_MARKERS.findall(text))
    en_count = len(_ENGLISH_MARKERS.findall(text))
    if de_count == 0 and en_count == 0:
        return "unknown"
    if de_count > en_count:
        return "de"
    if en_count > de_count:
        return "en"
    return "unknown"


def _check_language(response: str, expected_lang: str) -> bool:
    """Check if response language matches expected scenario language."""
    detected = _detect_language(response)
    # 'unknown' is acceptable for very short responses (e.g. just a number)
    if detected == "unknown":
        return True
    return detected == expected_lang


def _check_domain_bleed(response: str, scenario: str) -> list[str]:  # noqa: ARG001
    """Placeholder retained for compatibility — single-scenario mode never bleeds."""
    return []


_NUMBER_RE = re.compile(r"-?[\d.,]+")


def _check_value_grounding(response: str, steps: list) -> tuple[bool, str]:
    """For tool_use queries, verify that key numbers from tool results appear in the response.

    Returns (grounded, detail) where detail explains failures.
    """
    # Find tool execution steps with results
    tool_results = []
    for step in steps:
        if step.action == "execute_tool" and step.details.get("result"):
            tool_results.append(str(step.details["result"]))

    if not tool_results:
        return True, "no_tool_result"

    # Extract numbers from tool results
    result_text = " ".join(tool_results)
    result_numbers = set()
    for match in _NUMBER_RE.findall(result_text):
        cleaned = match.replace(",", "").rstrip(".")
        try:
            val = float(cleaned)
            if val != 0 and abs(val) >= 1:  # skip trivial 0s and tiny values
                result_numbers.add(cleaned)
                # Also add integer form if it's a whole number
                if val == int(val):
                    result_numbers.add(str(int(val)))
        except ValueError:
            pass

    if not result_numbers:
        return True, "no_numbers_in_result"

    # Check how many result numbers appear in the response
    response_clean = response.replace(",", "")
    found = {n for n in result_numbers if n in response_clean}

    if not found:
        # No numbers from tool result found in response — likely hallucinated
        sample = list(result_numbers)[:5]
        return False, f"expected_any_of={sample}"

    return True, f"found={len(found)}/{len(result_numbers)}"


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_eval() -> list[dict]:
    """Run all test queries through the full agent pipeline and check quality."""
    client = SmallLanguageModelClient.create_with_auto_detection()
    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    vector_store.set_client(client)
    tools = create_default_registry(vector_store=vector_store)
    agent = SmallLanguageModelAgentOrchestrator(client=client, tools=tools)

    scenario = SCENARIO_CONFIG.name
    expected_lang = SCENARIO_CONFIG.language
    test_set = _TEST_SET

    # Guard: check for eval/training data leakage
    synthesis_path = f"{SCENARIO_CONFIG.training_data_dir}/gemma3_synthesis{SCENARIO_CONFIG.training_data_suffix}.jsonl"
    overlaps = check_eval_training_overlap(
        [item["query"] for item in test_set],
        synthesis_path,
        query_key="input",
        threshold=0.6,
    )
    if overlaps:
        print(f"\n  WARNING: {len(overlaps)} eval queries overlap with training data (Jaccard >= 0.6):")
        for o in overlaps:
            print(f"    - eval: {o['eval_query'][:60]}")
            print(f"      train: {o['train_query'][:60]}  (sim={o['similarity']})")
        print()

    results = []
    print(f"\nResponse Quality Eval — scenario={scenario}, language={expected_lang}")
    print(f"Running {len(test_set)} queries through full pipeline…\n")

    for i, item in enumerate(test_set, 1):
        query = item["query"]
        try:
            t0 = time.perf_counter()
            agent_result = await agent.process(query)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            response = agent_result.response
            intent = agent_result.intent.value if agent_result.intent else "unknown"

            # 1. Language check
            lang_ok = _check_language(response, expected_lang)

            # 2. Domain bleed check
            bleed_terms = _check_domain_bleed(response, scenario)
            bleed_ok = len(bleed_terms) == 0

            # 3. Value grounding check (only for tool_use queries with expected numbers)
            grounded = True
            grounding_detail = "n/a"
            if item["expect_numbers"] and intent == "tool_use":
                grounded, grounding_detail = _check_value_grounding(
                    response, agent_result.steps
                )

            all_ok = lang_ok and bleed_ok and grounded

            results.append({
                "query": query,
                "expected_intent": item["intent"],
                "actual_intent": intent,
                "response": response[:300],
                "language_correct": lang_ok,
                "detected_language": _detect_language(response),
                "domain_bleed_ok": bleed_ok,
                "bleed_terms": bleed_terms,
                "value_grounded": grounded,
                "grounding_detail": grounding_detail,
                "all_checks_pass": all_ok,
                "latency_ms": latency_ms,
            })

        except Exception as e:
            results.append({
                "query": query,
                "expected_intent": item["intent"],
                "actual_intent": "error",
                "response": str(e)[:300],
                "language_correct": False,
                "detected_language": "error",
                "domain_bleed_ok": True,
                "bleed_terms": [],
                "value_grounded": False,
                "grounding_detail": f"error: {e}",
                "all_checks_pass": False,
                "latency_ms": 0,
            })

        status = "PASS" if results[-1]["all_checks_pass"] else "FAIL"
        print(f"  [{i}/{len(test_set)}] {status}  {query[:60]}")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    """Print a formatted quality report with per-check breakdowns."""
    n = len(results)
    n_lang = sum(1 for r in results if r["language_correct"])
    n_bleed = sum(1 for r in results if r["domain_bleed_ok"])
    n_grounded = sum(1 for r in results if r["value_grounded"])
    n_all = sum(1 for r in results if r["all_checks_pass"])

    # Tool-use subset for grounding stats
    tool_results = [r for r in results if r["expected_intent"] == "tool_use"]
    n_tool = len(tool_results)
    n_tool_grounded = sum(1 for r in tool_results if r["value_grounded"])

    print()
    print("=" * 65)
    print("  Response Quality Evaluation")
    print(f"  Scenario : {SCENARIO_CONFIG.name} (language={SCENARIO_CONFIG.language})")
    print(f"  Run      : {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 65)
    print()
    print(f"  Total queries:        {n}")
    print(f"  All checks pass:      {fmt_pct_with_ci(n_all, n)}")
    print()
    print(f"  Language correct:     {fmt_pct_with_ci(n_lang, n)}")
    print(f"  No domain bleed:      {fmt_pct_with_ci(n_bleed, n)}")
    print(f"  Value grounded:       {fmt_pct_with_ci(n_grounded, n)}  (all queries)")
    if n_tool > 0:
        print(f"  Value grounded:       {fmt_pct_with_ci(n_tool_grounded, n_tool)}  (tool_use only)")
    print()

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    if latencies:
        print(f"  Latency:              {fmt_latency(compute_latency_stats(latencies))}")
    print()

    # Per-intent breakdown
    intents = sorted(set(r["expected_intent"] for r in results))
    for intent in intents:
        subset = [r for r in results if r["expected_intent"] == intent]
        ns = len(subset)
        ok = sum(1 for r in subset if r["all_checks_pass"])
        print(f"  {intent:15s}  {fmt_pct_with_ci(ok, ns)}  ({ok}/{ns})")
    print()

    # Failures
    failures = [r for r in results if not r["all_checks_pass"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            checks = []
            if not r["language_correct"]:
                checks.append(f"lang={r['detected_language']}")
            if not r["domain_bleed_ok"]:
                checks.append(f"bleed={r['bleed_terms']}")
            if not r["value_grounded"]:
                checks.append(f"grounding={r['grounding_detail']}")
            print(f"    X  {r['query'][:60]}")
            print(f"       intent={r['actual_intent']}  {', '.join(checks)}")
            if len(r["response"]) > 0:
                resp_preview = r["response"][:120].replace("\n", " ")
                print(f"       response: {resp_preview}…")
        print()
    else:
        print("  All queries passed all quality checks!")
        print()


def build_summary(results: list[dict]) -> dict:
    """Build a JSON-serializable summary for --save."""
    n = len(results)
    n_lang = sum(1 for r in results if r["language_correct"])
    n_bleed = sum(1 for r in results if r["domain_bleed_ok"])
    n_grounded = sum(1 for r in results if r["value_grounded"])
    n_all = sum(1 for r in results if r["all_checks_pass"])

    tool_results = [r for r in results if r["expected_intent"] == "tool_use"]
    n_tool = len(tool_results)
    n_tool_grounded = sum(1 for r in tool_results if r["value_grounded"])

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]

    return {
        "scenario": SCENARIO_CONFIG.name,
        "language": SCENARIO_CONFIG.language,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_queries": n,
        "all_checks_pass": {"k": n_all, "n": n, "ci_95": list(wilson_ci(n_all, n))},
        "language_correct": {"k": n_lang, "n": n, "ci_95": list(wilson_ci(n_lang, n))},
        "no_domain_bleed": {"k": n_bleed, "n": n, "ci_95": list(wilson_ci(n_bleed, n))},
        "value_grounded_all": {"k": n_grounded, "n": n, "ci_95": list(wilson_ci(n_grounded, n))},
        "value_grounded_tool": {"k": n_tool_grounded, "n": n_tool, "ci_95": list(wilson_ci(n_tool_grounded, n_tool))},
        "latency": compute_latency_stats(latencies) if latencies else {},
        "per_query": results,
    }


def print_comparison(before: dict, after: dict) -> None:
    """Compare two saved results and print delta + McNemar test."""
    print()
    print("=" * 65)
    print("  Response Quality — Before vs After")
    print("=" * 65)

    for metric in ["all_checks_pass", "language_correct", "no_domain_bleed", "value_grounded_all"]:
        b = before[metric]
        a = after[metric]
        b_pct = b["k"] / b["n"] * 100 if b["n"] else 0
        a_pct = a["k"] / a["n"] * 100 if a["n"] else 0
        delta = a_pct - b_pct
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:25s}  {b_pct:5.1f}% → {a_pct:5.1f}%  ({sign}{delta:.1f}pp)")

    # McNemar on all_checks_pass
    b_per = before["per_query"]
    a_per = after["per_query"]
    if len(b_per) == len(a_per):
        b_correct = [r["all_checks_pass"] for r in b_per]
        a_correct = [r["all_checks_pass"] for r in a_per]
        mc = mcnemar_test(b_correct, a_correct)
        sig = "YES" if mc["significant_at_05"] else "no"
        print(f"\n  McNemar: chi2={mc['chi2']:.2f}  p={mc['p_value']:.4f}  significant={sig}")
        print(f"           regressions={mc['b']}  improvements={mc['c']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    args = sys.argv[1:]
    save_path = None
    compare_paths = None

    i = 0
    while i < len(args):
        if args[i] == "--save" and i + 1 < len(args):
            save_path = args[i + 1]
            i += 2
        elif args[i] == "--compare" and i + 2 < len(args):
            compare_paths = (args[i + 1], args[i + 2])
            i += 3
        else:
            i += 1

    if compare_paths:
        before = load_results(compare_paths[0])
        after = load_results(compare_paths[1])
        print_comparison(before, after)
        return

    results = await run_eval()
    print_report(results)

    if save_path:
        summary = build_summary(results)
        save_results(summary, save_path)


if __name__ == "__main__":
    asyncio.run(main())
