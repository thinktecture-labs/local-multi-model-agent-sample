"""
Tool Routing Evaluator — measure tool selection accuracy before/after fine-tuning.

Uses a fixed labelled test set of 160 queries (80 per tool: sql_query, calculator)
to produce a reproducible accuracy score.  Run once before fine-tuning to establish a
baseline, then again after to quantify the improvement.

Metrics:
  tool_accuracy        — % of queries where the correct tool name was selected
  arg_key_present      — % of queries where the expected argument key exists in the response
  expression_accuracy  — % of calculator queries where the expression evaluates to the
                         correct numeric result (within 1% relative tolerance)
  sql_exec_success     — % of sql_query queries where the generated SQL executes without error
  sql_returns_rows     — % of sql_query queries where the SQL returns at least one row
  overall              — harmonic mean of tool_accuracy and arg_key_present

Usage:
  python -m finetune.eval_tool_routing                          # run + print report
  python -m finetune.eval_tool_routing --save results/baseline_tool_routing.json
  python -m finetune.eval_tool_routing --compare before.json after.json

Demo talk workflow:
  1. python -m finetune.eval_tool_routing --save results/baseline_tool_routing.json
  2. python -m finetune.train_qwen35_toolcalling
  3. bash finetune/convert_qwen35_to_gguf.sh
  4. bash scripts/start_servers.sh --bg --ft
  5. python -m finetune.eval_tool_routing --save results/finetuned_tool_routing.json
  6. python -m finetune.eval_tool_routing --compare results/baseline_tool_routing.json results/finetuned_tool_routing.json
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path

from finetune.eval_base import (
    compute_latency_stats,
    fmt_ci as _fmt_ci,
    fmt_latency,
    fmt_pct as _fmt_pct,
    load_eval_json,
    load_eval_jsonl,
    load_results,
    mcnemar_test,
    save_results,
    wilson_ci,
)
from finetune._scenario import SCENARIO_NAME as _SCENARIO


# ---------------------------------------------------------------------------
# Fixed labelled test set — 160 queries (80 per tool)
# Loaded from data/eval-data/ JSONL files (not inline).
# ---------------------------------------------------------------------------
TEST_SET = load_eval_jsonl(f"eval_tool_routing_{_SCENARIO}.jsonl")
CALCULATOR_EXPECTED = load_eval_json(f"eval_calculator_{_SCENARIO}.json")
TOOLS = ["sql_query", "calculator"]


# ---------------------------------------------------------------------------
# Expression evaluation helper
# ---------------------------------------------------------------------------

def _try_eval_expression(expr: str) -> float | None:
    """Normalise and evaluate a calculator expression, returning the result or None."""
    try:
        from src.engine.tools.calculator import CalculatorTool, _make_evaluator
        sanitized = CalculatorTool._normalize_expression(expr)
        evaluator = _make_evaluator()
        result = evaluator.eval(sanitized)
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# SQL execution validation
# ---------------------------------------------------------------------------

from src.engine.inference.config import SCENARIO_CONFIG as _SC
_DEFAULT_DB_PATH = _SC.db_path


async def _try_exec_sql(
    sql: str, db_path: str = _DEFAULT_DB_PATH
) -> dict:
    """Execute a SQL query against the business database and return validation info.

    Returns dict with keys:
        sql_exec_success: bool — did the query execute without error?
        sql_returns_rows: bool — did it return at least one row?
        sql_row_count:    int  — number of rows returned (0 if error)
        sql_error:        str | None — error message if execution failed
        sql_columns:      list[str] — column names returned (empty if error)
    """
    import aiosqlite
    import re

    if not sql or not sql.strip():
        return {
            "sql_exec_success": False,
            "sql_returns_rows": False,
            "sql_row_count": 0,
            "sql_error": "Empty query",
            "sql_columns": [],
        }

    query = sql.strip()

    # Strip wrapping parens (same logic as SQLQueryTool)
    bare = query.rstrip(";").rstrip()
    if bare.startswith("(") and bare.endswith(")"):
        inner = bare[1:-1]
        depth = 0
        balanced = True
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    balanced = False
                    break
                depth -= 1
        if balanced and depth == 0:
            query = inner.strip()

    # Safety: only SELECT
    normalized = query.lstrip("(").lower()
    if not normalized.startswith("select"):
        return {
            "sql_exec_success": False,
            "sql_returns_rows": False,
            "sql_row_count": 0,
            "sql_error": f"Not a SELECT: {query[:60]}",
            "sql_columns": [],
        }

    # Add LIMIT if missing
    if not re.search(r'\bLIMIT\s+\d+', query, re.IGNORECASE):
        query = f"{query.rstrip(';')} LIMIT 50"

    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return {
            "sql_exec_success": True,
            "sql_returns_rows": len(rows) > 0,
            "sql_row_count": len(rows),
            "sql_error": None,
            "sql_columns": columns,
        }
    except Exception as exc:
        return {
            "sql_exec_success": False,
            "sql_returns_rows": False,
            "sql_row_count": 0,
            "sql_error": str(exc),
            "sql_columns": [],
        }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _build_tool_schemas() -> list[dict]:
    """
    Build OpenAI-compatible tool schemas for all three tools.

    Uses get_schema() directly on tool instances. VectorSearchTool requires
    a vector_store argument for execution, but get_schema() only reads the
    static name/description/parameters — a None placeholder is safe here.
    """
    from src.engine.tools.vector_search import VectorSearchTool
    from src.engine.tools.sql_query import SQLQueryTool
    from src.engine.tools.calculator import CalculatorTool

    # Only calculator and sql_query — vector_search is handled by gemma3's
    # rag_query intent, not by the tool-calling model.
    return [
        SQLQueryTool().get_schema(),
        CalculatorTool().get_schema(),
    ]


async def evaluate_query(
    client, item: dict, schemas: list[dict],
    *, recommended_sampling: bool = False,
) -> dict:
    """Run a single query through the tool-calling model and score the result."""
    messages = [{"role": "user", "content": item["query"]}]
    response = await client.call_function(
        messages=messages, tools=schemas,
        recommended_sampling=recommended_sampling,
    )

    tool_selected = None
    args = {}
    if response.function_call:
        tool_selected = response.function_call.get("name")
        args = response.function_call.get("arguments", {})

    tool_correct   = tool_selected == item["expected_tool"]
    arg_key_present = item["expected_arg_key"] in args

    # Expression correctness: evaluate calculator expressions against expected results
    expression_correct = None
    expression_result = None
    if item["expected_tool"] == "calculator" and "expression" in args:
        expected = CALCULATOR_EXPECTED.get(item["query"])
        if expected is not None:
            expression_result = _try_eval_expression(args["expression"])
            if expression_result is not None:
                expression_correct = math.isclose(
                    expression_result, expected, rel_tol=0.01
                )
            else:
                expression_correct = False

    # SQL execution validation: run generated SQL against real database
    sql_validation = None
    if tool_selected == "sql_query" and "query" in args:
        sql_validation = await _try_exec_sql(args["query"])

    result = {
        "query":              item["query"],
        "expected_tool":      item["expected_tool"],
        "selected_tool":      tool_selected,
        "tool_correct":       tool_correct,
        "arg_key_present":    arg_key_present,
        "arguments":          args,
        "expression_correct": expression_correct,
        "expression_result":  expression_result,
    }
    if sql_validation is not None:
        result.update(sql_validation)

    return result


async def run_eval(
    client, test_set: list[dict] | None = None,
    *, recommended_sampling: bool = False,
) -> dict:
    """Run the full evaluation against the live tool-calling server.

    sampling_mode is recorded in the result JSON so two eval runs (default
    config-greedy vs Qwen-recommended) can be compared cleanly via --compare.
    """
    if test_set is None:
        test_set = TEST_SET

    try:
        from src.engine.inference.client import SmallLanguageModelRole
        model_name = client.models.get(SmallLanguageModelRole.FUNCTION, "unknown")
    except Exception:
        model_name = "unknown"

    schemas = _build_tool_schemas()
    predictions = []
    for item in test_set:
        t0 = time.perf_counter()
        result = await evaluate_query(
            client, item, schemas,
            recommended_sampling=recommended_sampling,
        )
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        predictions.append(result)

    return {
        "timestamp":     datetime.now().isoformat(),
        "model":         model_name,
        "n":             len(predictions),
        "sampling_mode": "qwen_recommended" if recommended_sampling else "default_greedy",
        "predictions":   predictions,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    """Compute tool accuracy, argument key presence, overall score, and CIs."""
    preds = results["predictions"]
    n = len(preds)
    if n == 0:
        return {"tool_accuracy": 0.0, "arg_key_accuracy": 0.0, "overall": 0.0, "n": 0, "per_tool": {}}

    n_tool_correct = sum(1 for p in preds if p["tool_correct"])
    n_arg_correct  = sum(1 for p in preds if p["arg_key_present"])
    tool_acc = n_tool_correct / n
    arg_acc  = n_arg_correct  / n
    # Harmonic mean: penalises cases where tool is right but args are empty
    overall  = 2 * tool_acc * arg_acc / (tool_acc + arg_acc) if (tool_acc + arg_acc) > 0 else 0.0

    per_tool: dict[str, dict] = {}
    for tool in TOOLS:
        tool_preds = [p for p in preds if p["expected_tool"] == tool]
        n_t = len(tool_preds)
        n_tc = sum(1 for p in tool_preds if p["tool_correct"])
        n_ac = sum(1 for p in tool_preds if p["arg_key_present"])
        tool_entry: dict = {
            "n":               n_t,
            "tool_correct":    n_tc,
            "arg_key_present": n_ac,
            "tool_accuracy":   n_tc / n_t if n_t else 0.0,
            "arg_key_accuracy": n_ac / n_t if n_t else 0.0,
            "tool_accuracy_ci": wilson_ci(n_tc, n_t),
        }
        # Expression correctness — calculator only
        if tool == "calculator":
            expr_preds = [p for p in tool_preds if p.get("expression_correct") is not None]
            n_expr = len(expr_preds)
            n_expr_correct = sum(1 for p in expr_preds if p["expression_correct"])
            tool_entry["expression_evaluated"] = n_expr
            tool_entry["expression_correct"] = n_expr_correct
            tool_entry["expression_accuracy"] = n_expr_correct / n_expr if n_expr else 0.0
            tool_entry["expression_accuracy_ci"] = wilson_ci(n_expr_correct, n_expr)

        # SQL execution validation — sql_query only
        if tool == "sql_query":
            sql_preds = [p for p in tool_preds if "sql_exec_success" in p]
            n_sql = len(sql_preds)
            n_exec_ok = sum(1 for p in sql_preds if p["sql_exec_success"])
            n_has_rows = sum(1 for p in sql_preds if p["sql_returns_rows"])
            tool_entry["sql_evaluated"] = n_sql
            tool_entry["sql_exec_success"] = n_exec_ok
            tool_entry["sql_exec_accuracy"] = n_exec_ok / n_sql if n_sql else 0.0
            tool_entry["sql_exec_accuracy_ci"] = wilson_ci(n_exec_ok, n_sql)
            tool_entry["sql_returns_rows"] = n_has_rows
            tool_entry["sql_returns_rows_accuracy"] = n_has_rows / n_sql if n_sql else 0.0
            tool_entry["sql_returns_rows_ci"] = wilson_ci(n_has_rows, n_sql)

        per_tool[tool] = tool_entry

    return {
        "tool_accuracy":    tool_acc,
        "tool_accuracy_ci": wilson_ci(n_tool_correct, n),
        "arg_key_accuracy": arg_acc,
        "overall":          overall,
        "n":                n,
        "per_tool":         per_tool,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(before: dict, after: dict) -> dict:
    """Compute tool accuracy deltas with McNemar's significance test."""
    s_before = score(before)
    s_after  = score(after)
    per_tool_delta: dict[str, float] = {
        tool: s_after["per_tool"][tool]["tool_accuracy"]
              - s_before["per_tool"][tool]["tool_accuracy"]
        for tool in TOOLS
    }

    # Paired significance test — match predictions by query text
    before_by_query = {p["query"]: p["tool_correct"] for p in before["predictions"]}
    after_by_query = {p["query"]: p["tool_correct"] for p in after["predictions"]}
    shared = sorted(set(before_by_query) & set(after_by_query))
    before_correct = [before_by_query[q] for q in shared]
    after_correct = [after_by_query[q] for q in shared]
    mcnemar = mcnemar_test(before_correct, after_correct) if shared else None

    return {
        "overall_delta":    s_after["overall"] - s_before["overall"],
        "tool_acc_delta":   s_after["tool_accuracy"] - s_before["tool_accuracy"],
        "arg_acc_delta":    s_after["arg_key_accuracy"] - s_before["arg_key_accuracy"],
        "per_tool_delta":   per_tool_delta,
        "before":           s_before,
        "after":            s_after,
        "mcnemar":          mcnemar,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "Tool Routing Evaluation") -> None:
    s = score(results)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Model : {results.get('model', '?')}")
    print(f"  Run   : {results.get('timestamp', '?')[:19]}")
    print(f"{'='*60}")
    tlo, thi = s["tool_accuracy_ci"]
    print(f"\n  Tool selection accuracy: {_fmt_pct(s['tool_accuracy'])}  {_fmt_ci(tlo, thi)}")
    print(f"  Argument key presence:   {_fmt_pct(s['arg_key_accuracy'])}")
    print(f"  Overall (harmonic mean): {_fmt_pct(s['overall'])}  ({s['n']} queries)")

    latencies = [p["latency_ms"] for p in results["predictions"] if "latency_ms" in p]
    if latencies:
        print(f"  Latency:                 {fmt_latency(compute_latency_stats(latencies))}")
    print()

    print("  Per-tool tool accuracy:")
    for tool in TOOLS:
        ts = s["per_tool"][tool]
        bar = "\u2588" * int(ts["tool_accuracy"] * 20)
        clo, chi = ts["tool_accuracy_ci"]
        print(f"    {tool:<14s}  {_fmt_pct(ts['tool_accuracy'])}"
              f"  {_fmt_ci(clo, chi)}  ({ts['tool_correct']}/{ts['n']})  {bar}")

    # Expression correctness for calculator
    calc = s["per_tool"].get("calculator", {})
    if calc.get("expression_evaluated", 0) > 0:
        elo, ehi = calc["expression_accuracy_ci"]
        print(f"\n  Calculator expression correctness: {_fmt_pct(calc['expression_accuracy'])}"
              f"  {_fmt_ci(elo, ehi)}"
              f"  ({calc['expression_correct']}/{calc['expression_evaluated']} evaluated)")

    # SQL execution validation for sql_query
    sql = s["per_tool"].get("sql_query", {})
    if sql.get("sql_evaluated", 0) > 0:
        slo, shi = sql["sql_exec_accuracy_ci"]
        rlo, rhi = sql["sql_returns_rows_ci"]
        print(f"\n  SQL execution validation ({sql['sql_evaluated']} queries with sql_query selected):")
        print(f"    Executes without error:  {_fmt_pct(sql['sql_exec_accuracy'])}"
              f"  {_fmt_ci(slo, shi)}"
              f"  ({sql['sql_exec_success']}/{sql['sql_evaluated']})")
        print(f"    Returns at least 1 row:  {_fmt_pct(sql['sql_returns_rows_accuracy'])}"
              f"  {_fmt_ci(rlo, rhi)}"
              f"  ({sql['sql_returns_rows']}/{sql['sql_evaluated']})")

        # Show SQL errors
        sql_errors = [p for p in results["predictions"]
                      if p.get("sql_exec_success") is False and p.get("sql_error")]
        if sql_errors:
            print(f"\n  SQL errors ({len(sql_errors)}):")
            for p in sql_errors:
                print(f"    \u2717  \"{p['query'][:65]}\"")
                generated_sql = p.get("arguments", {}).get("query", "???")
                print(f"       SQL: {generated_sql[:80]}")
                print(f"       Err: {p['sql_error'][:80]}")

        # Show queries that execute but return 0 rows
        sql_empty = [p for p in results["predictions"]
                     if p.get("sql_exec_success") is True and p.get("sql_returns_rows") is False]
        if sql_empty:
            print(f"\n  SQL returns 0 rows ({len(sql_empty)}):")
            for p in sql_empty:
                generated_sql = p.get("arguments", {}).get("query", "???")
                print(f"    \u25cb  \"{p['query'][:65]}\"")
                print(f"       SQL: {generated_sql[:80]}")

    wrong = [p for p in results["predictions"] if not p["tool_correct"]]
    if wrong:
        print(f"\n  Wrong tool ({len(wrong)}):")
        for p in wrong:
            print(f"    \u2717  expected={p['expected_tool']:<14s}  "
                  f"got={p['selected_tool']!r}")
            print(f"       \"{p['query'][:70]}\"")
    print()


def print_comparison(comparison: dict, labels: tuple[str, str] = ("Before", "After")) -> None:
    before_label, after_label = labels
    c = comparison
    print(f"\n{'='*60}")
    print(f"  Tool Routing:  {before_label}  \u2192  {after_label}")
    print(f"{'='*60}")
    delta = c["overall_delta"]
    sign  = "+" if delta >= 0 else ""
    print(f"\n  Overall:        {_fmt_pct(c['before']['overall'])}  \u2192  "
          f"{_fmt_pct(c['after']['overall'])}   ({sign}{_fmt_pct(delta)})", end="")
    if c.get("mcnemar"):
        p = c["mcnemar"]["p_value"]
        sig = " *" if c["mcnemar"]["significant_at_05"] else " n.s."
        print(f"  p={p:.3f}{sig}")
    else:
        print()
    print(f"  Tool accuracy:  {_fmt_pct(c['before']['tool_accuracy'])}  \u2192  "
          f"{_fmt_pct(c['after']['tool_accuracy'])}")
    print(f"\n  Per-tool tool accuracy:")
    for tool in TOOLS:
        b = c["before"]["per_tool"][tool]["tool_accuracy"]
        a = c["after"]["per_tool"][tool]["tool_accuracy"]
        d = c["per_tool_delta"][tool]
        sign = "+" if d >= 0 else ""
        arrow = "\u25b2" if d > 0.0001 else ("\u25bc" if d < -0.0001 else "\u2550")
        print(f"    {tool:<14s}  {_fmt_pct(b)}  \u2192  {_fmt_pct(a)}"
              f"   {arrow} {sign}{_fmt_pct(d)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate tool routing accuracy (Qwen tool caller)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--save",    metavar="PATH", help="Save raw results to JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved result files — no servers required")
    parser.add_argument("--function-port", type=int, metavar="PORT",
                        help="Override function-model port (e.g. 9100 for Qwen MoE). "
                             "Default: auto-detect FT (9095) or base (9091).")
    parser.add_argument("--function-model", metavar="NAME",
                        help="Model name label to record in results (e.g. qwen3.5-35b-a3b). "
                             "Inferred from server if omitted.")
    parser.add_argument("--recommended-sampling", action="store_true",
                        help="Use Qwen's published recommended-sampling params "
                             "(temp=0.7, top_p=0.95, top_k=20) instead of the "
                             "shipping config-default greedy mode. Quantifies the "
                             "accuracy cost of greedy decoding on the FT model.")
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
            if args.function_port:
                function_url = f"http://localhost:{args.function_port}/v1"
                function_model = args.function_model or f"port-{args.function_port}"
                client = SmallLanguageModelClient(
                    function_url=function_url,
                    function_model=function_model,
                )
                print(f"\nUsing function model: {function_model} ({function_url})")
            else:
                client = SmallLanguageModelClient.create_with_auto_detection()
            mode_label = "Qwen-recommended sampling (temp=0.7)" if args.recommended_sampling else "default greedy"
            print(f"\nRunning tool selection eval ({len(TEST_SET)} queries, mode: {mode_label})\u2026")
            results = await run_eval(client, recommended_sampling=args.recommended_sampling)
            print_report(results)
            if args.save:
                save_results(results, args.save)

        asyncio.run(_main())
