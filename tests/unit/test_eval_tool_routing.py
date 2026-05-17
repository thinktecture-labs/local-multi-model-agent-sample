"""
Unit tests for finetune/eval_tool_routing.py.

Tests the scoring, comparison, and reporting logic without touching any server.
All tests are pure-Python — no async, no external services.
"""

import json
import os

import pytest

from finetune.eval_tool_routing import (
    CALCULATOR_EXPECTED,
    TEST_SET,
    TOOLS,
    _try_eval_expression,
    _try_exec_sql,
    compare,
    load_results,
    print_comparison,
    print_report,
    save_results,
    score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(predictions: list[dict], model: str = "qwen") -> dict:
    return {
        "timestamp":   "2025-01-01T00:00:00",
        "model":       model,
        "n":           len(predictions),
        "predictions": predictions,
    }


def _perfect_results() -> dict:
    """All queries correctly routed AND argument key present."""
    preds = [
        {
            "query":           item["query"],
            "expected_tool":   item["expected_tool"],
            "selected_tool":   item["expected_tool"],
            "tool_correct":    True,
            "arg_key_present": True,
            "arguments":       {item["expected_arg_key"]: "some_value"},
        }
        for item in TEST_SET
    ]
    return _make_results(preds)


def _zero_results() -> dict:
    """All queries routed to wrong tool."""
    preds = [
        {
            "query":           item["query"],
            "expected_tool":   item["expected_tool"],
            "selected_tool":   "calculator" if item["expected_tool"] != "calculator" else "sql_query",
            "tool_correct":    False,
            "arg_key_present": False,
            "arguments":       {},
        }
        for item in TEST_SET
    ]
    return _make_results(preds)


def _partial_results(correct_tools: set[str]) -> dict:
    """Correct only for specified tools."""
    preds = []
    for item in TEST_SET:
        correct = item["expected_tool"] in correct_tools
        preds.append({
            "query":           item["query"],
            "expected_tool":   item["expected_tool"],
            "selected_tool":   item["expected_tool"] if correct else "vector_search",
            "tool_correct":    correct,
            "arg_key_present": correct,
            "arguments":       {item["expected_arg_key"]: "val"} if correct else {},
        })
    return _make_results(preds)


# ---------------------------------------------------------------------------
# Test set integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTestSet:
    def test_total_query_count(self):
        assert len(TEST_SET) == 160

    def test_all_tools_represented(self):
        tools_found = {item["expected_tool"] for item in TEST_SET}
        assert tools_found == set(TOOLS)

    def test_all_tools_have_valid_names(self):
        for item in TEST_SET:
            assert item["expected_tool"] in TOOLS

    def test_all_have_expected_arg_key(self):
        for item in TEST_SET:
            assert "expected_arg_key" in item
            assert item["expected_arg_key"]

    def test_no_duplicate_queries(self):
        queries = [item["query"] for item in TEST_SET]
        assert len(queries) == len(set(queries))

    def test_sql_query_count(self):
        count = sum(1 for item in TEST_SET if item["expected_tool"] == "sql_query")
        assert count == 80

    def test_calculator_count(self):
        count = sum(1 for item in TEST_SET if item["expected_tool"] == "calculator")
        assert count == 80

    def test_no_vector_search(self):
        count = sum(1 for item in TEST_SET if item["expected_tool"] == "vector_search")
        assert count == 0  # vector_search handled by gemma3's rag_query intent


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScore:
    def test_perfect_accuracy(self):
        s = score(_perfect_results())
        assert s["tool_accuracy"]    == pytest.approx(1.0)
        assert s["arg_key_accuracy"] == pytest.approx(1.0)
        assert s["overall"]          == pytest.approx(1.0)
        assert s["n"] == 160

    def test_zero_accuracy(self):
        s = score(_zero_results())
        assert s["tool_accuracy"]    == pytest.approx(0.0)
        assert s["arg_key_accuracy"] == pytest.approx(0.0)
        assert s["overall"]          == pytest.approx(0.0)

    def test_score_has_tool_accuracy_ci(self):
        s = score(_perfect_results())
        assert "tool_accuracy_ci" in s
        lo, hi = s["tool_accuracy_ci"]
        assert 0.0 <= lo <= hi <= 1.0

    def test_per_tool_has_ci(self):
        s = score(_perfect_results())
        for tool in TOOLS:
            assert "tool_accuracy_ci" in s["per_tool"][tool]
            lo, hi = s["per_tool"][tool]["tool_accuracy_ci"]
            assert 0.0 <= lo <= hi <= 1.0

    def test_ci_narrows_vs_small_sample(self):
        small = score(_make_results([
            {"query": "x", "expected_tool": "sql_query", "selected_tool": "sql_query",
             "tool_correct": True, "arg_key_present": True, "arguments": {"expression": "v"}}
        ] * 10))
        large = score(_perfect_results())
        small_width = small["tool_accuracy_ci"][1] - small["tool_accuracy_ci"][0]
        large_width = large["tool_accuracy_ci"][1] - large["tool_accuracy_ci"][0]
        assert large_width < small_width

    def test_per_tool_keys_present(self):
        s = score(_perfect_results())
        for tool in TOOLS:
            assert tool in s["per_tool"]
            assert "tool_accuracy" in s["per_tool"][tool]
            assert "n" in s["per_tool"][tool]

    def test_per_tool_perfect(self):
        s = score(_perfect_results())
        for tool in TOOLS:
            assert s["per_tool"][tool]["tool_accuracy"] == pytest.approx(1.0)

    def test_per_tool_partial(self):
        s = score(_partial_results({"sql_query"}))
        assert s["per_tool"]["sql_query"]["tool_accuracy"] == pytest.approx(1.0)
        assert s["per_tool"]["calculator"]["tool_accuracy"] == pytest.approx(0.0)

    def test_empty_predictions(self):
        s = score(_make_results([]))
        assert s["overall"] == pytest.approx(0.0)
        assert s["n"] == 0

    def test_overall_is_harmonic_mean(self):
        # If tool=1.0 and arg=0.5 → hmean = 2*1.0*0.5 / 1.5 = 2/3 ≈ 0.667
        preds = []
        for item in TEST_SET:
            preds.append({
                "query": item["query"],
                "expected_tool": item["expected_tool"],
                "selected_tool": item["expected_tool"],
                "tool_correct": True,
                "arg_key_present": False,  # tool right, arg missing
                "arguments": {},
            })
        s = score(_make_results(preds))
        assert s["tool_accuracy"] == pytest.approx(1.0)
        assert s["arg_key_accuracy"] == pytest.approx(0.0)
        assert s["overall"] == pytest.approx(0.0)  # hmean with 0 = 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompare:
    def test_improvement(self):
        c = compare(_zero_results(), _perfect_results())
        assert c["overall_delta"] == pytest.approx(1.0)

    def test_regression(self):
        c = compare(_perfect_results(), _zero_results())
        assert c["overall_delta"] == pytest.approx(-1.0)

    def test_no_change(self):
        r = _perfect_results()
        c = compare(r, r)
        assert c["overall_delta"] == pytest.approx(0.0)

    def test_per_tool_delta_keys(self):
        c = compare(_zero_results(), _perfect_results())
        for tool in TOOLS:
            assert tool in c["per_tool_delta"]

    def test_before_and_after_in_comparison(self):
        c = compare(_zero_results(), _perfect_results())
        assert "before" in c
        assert "after" in c

    def test_compare_has_mcnemar(self):
        c = compare(_zero_results(), _perfect_results())
        assert "mcnemar" in c
        assert c["mcnemar"] is not None
        assert "p_value" in c["mcnemar"]
        assert "significant_at_05" in c["mcnemar"]

    def test_compare_significant_for_large_improvement(self):
        c = compare(_zero_results(), _perfect_results())
        assert c["mcnemar"]["significant_at_05"] is True

    def test_compare_not_significant_for_no_change(self):
        r = _perfect_results()
        c = compare(r, r)
        assert c["mcnemar"]["significant_at_05"] is False


# ---------------------------------------------------------------------------
# Printing (smoke tests — verify no exceptions)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrinting:
    def test_print_report_perfect(self, capsys):
        print_report(_perfect_results(), title="Test")
        captured = capsys.readouterr()
        assert "100.0%" in captured.out

    def test_print_report_with_wrong(self, capsys):
        print_report(_zero_results())
        captured = capsys.readouterr()
        assert "Wrong tool" in captured.out

    def test_print_report_shows_ci(self, capsys):
        print_report(_perfect_results(), title="CI Test")
        captured = capsys.readouterr()
        assert "[" in captured.out and "]" in captured.out

    def test_print_comparison(self, capsys):
        c = compare(_zero_results(), _perfect_results())
        print_comparison(c, labels=("Baseline", "Fine-tuned"))
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Fine-tuned" in captured.out
        assert "▲" in captured.out

    def test_print_comparison_shows_p_value(self, capsys):
        c = compare(_zero_results(), _perfect_results())
        print_comparison(c, labels=("Base", "FT"))
        captured = capsys.readouterr()
        assert "p=" in captured.out


# ---------------------------------------------------------------------------
# I/O roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIO:
    def test_save_and_load(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "results.json")
        save_results(results, path)
        loaded = load_results(path)
        assert loaded["n"] == results["n"]
        assert loaded["model"] == results["model"]

    def test_save_creates_parent_dirs(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "nested" / "deep" / "results.json")
        save_results(results, path)
        loaded = load_results(path)
        assert loaded["n"] == 160

    def test_loaded_results_scoreable(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "r.json")
        save_results(results, path)
        loaded = load_results(path)
        s = score(loaded)
        assert s["tool_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Train/eval overlap
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOverlap:
    def test_no_overlap_with_training_data(self):
        """Verify zero overlap between TEST_SET and training JSONL."""
        from finetune.eval_base import check_eval_training_overlap
        eval_queries = [item["query"] for item in TEST_SET]
        overlaps = check_eval_training_overlap(
            eval_queries,
            "data/training-data/tool_routing_2tool.jsonl",
            query_key="input",
            threshold=0.7,
        )
        assert len(overlaps) == 0, (
            f"Found {len(overlaps)} overlapping queries:\n"
            + "\n".join(f"  eval: {o['eval_query']!r}\n  train: {o['train_query']!r}\n  sim: {o['similarity']}"
                        for o in overlaps[:5])
        )


# ---------------------------------------------------------------------------
# Expression correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExpressionCorrectness:
    def test_expected_results_cover_all_calculator_queries(self):
        calc_queries = [item["query"] for item in TEST_SET if item["expected_tool"] == "calculator"]
        assert len(CALCULATOR_EXPECTED) == len(calc_queries)
        for q in calc_queries:
            assert q in CALCULATOR_EXPECTED, f"Missing expected result for: {q!r}"

    def test_expected_results_are_finite_numbers(self):
        import math
        for query, val in CALCULATOR_EXPECTED.items():
            assert isinstance(val, (int, float)), f"Not a number: {query!r} → {val!r}"
            assert math.isfinite(val), f"Non-finite: {query!r} → {val}"

    def test_try_eval_basic_arithmetic(self):
        assert _try_eval_expression("2 + 3") == 5.0
        assert _try_eval_expression("100 * 12") == 1200.0
        assert _try_eval_expression("999 / 3") == 333.0

    def test_try_eval_percentage_normalisation(self):
        import math
        result = _try_eval_expression("15% of 120000")
        assert result is not None
        assert math.isclose(result, 18000.0, rel_tol=1e-9)

    def test_try_eval_caret_alias(self):
        result = _try_eval_expression("2 ^ 10")
        assert result == 1024.0

    def test_try_eval_returns_none_on_bad_input(self):
        assert _try_eval_expression("not a math expression") is None
        assert _try_eval_expression("") is None

    def test_try_eval_trailing_result_stripped(self):
        import math
        result = _try_eval_expression("15 * 100 = 1500")
        assert result is not None
        assert math.isclose(result, 1500.0, rel_tol=1e-9)

    def test_score_includes_expression_accuracy_for_calculator(self):
        """Score with expression_correct fields should report expression_accuracy."""
        preds = []
        for item in TEST_SET:
            correct = True
            expr_correct = None
            expr_result = None
            if item["expected_tool"] == "calculator":
                expr_correct = True
                expr_result = CALCULATOR_EXPECTED.get(item["query"], 0.0)
            preds.append({
                "query":              item["query"],
                "expected_tool":      item["expected_tool"],
                "selected_tool":      item["expected_tool"],
                "tool_correct":       True,
                "arg_key_present":    True,
                "arguments":          {item["expected_arg_key"]: "val"},
                "expression_correct": expr_correct,
                "expression_result":  expr_result,
            })
        s = score(_make_results(preds))
        calc = s["per_tool"]["calculator"]
        assert calc["expression_accuracy"] == pytest.approx(1.0)
        assert calc["expression_correct"] == 80
        assert calc["expression_evaluated"] == 80

    def test_score_expression_accuracy_partial(self):
        """Half correct, half wrong expressions."""
        preds = []
        for i, item in enumerate(TEST_SET):
            expr_correct = None
            if item["expected_tool"] == "calculator":
                expr_correct = (i % 2 == 0)
            preds.append({
                "query":              item["query"],
                "expected_tool":      item["expected_tool"],
                "selected_tool":      item["expected_tool"],
                "tool_correct":       True,
                "arg_key_present":    True,
                "arguments":          {item["expected_arg_key"]: "val"},
                "expression_correct": expr_correct,
                "expression_result":  None,
            })
        s = score(_make_results(preds))
        calc = s["per_tool"]["calculator"]
        assert calc["expression_evaluated"] == 80
        assert 0.4 <= calc["expression_accuracy"] <= 0.6

    def test_score_no_expression_data_graceful(self):
        """Old-format predictions without expression_correct should not crash."""
        preds = [
            {
                "query": item["query"],
                "expected_tool": item["expected_tool"],
                "selected_tool": item["expected_tool"],
                "tool_correct": True,
                "arg_key_present": True,
                "arguments": {item["expected_arg_key"]: "val"},
            }
            for item in TEST_SET
        ]
        s = score(_make_results(preds))
        calc = s["per_tool"]["calculator"]
        assert calc["expression_evaluated"] == 0
        assert calc["expression_accuracy"] == pytest.approx(0.0)

    def test_print_report_shows_expression_correctness(self, capsys):
        """Report should include expression correctness line when data present."""
        preds = []
        for item in TEST_SET:
            expr_correct = True if item["expected_tool"] == "calculator" else None
            preds.append({
                "query":              item["query"],
                "expected_tool":      item["expected_tool"],
                "selected_tool":      item["expected_tool"],
                "tool_correct":       True,
                "arg_key_present":    True,
                "arguments":          {item["expected_arg_key"]: "val"},
                "expression_correct": expr_correct,
                "expression_result":  None,
            })
        print_report(_make_results(preds), title="Expr Test")
        captured = capsys.readouterr()
        assert "expression correctness" in captured.out.lower()

    def test_no_expected_results_overlap_with_sql_queries(self):
        """CALCULATOR_EXPECTED should only contain calculator query strings."""
        sql_queries = {item["query"] for item in TEST_SET if item["expected_tool"] == "sql_query"}
        overlap = sql_queries & set(CALCULATOR_EXPECTED.keys())
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# SQL execution validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSQLExecValidation:
    """Tests for _try_exec_sql and SQL metrics in score/report."""

    @pytest.mark.asyncio
    async def test_valid_sql_executes(self):
        q = "SELECT name FROM customers LIMIT 3"
        col = "name"
        result = await _try_exec_sql(q)
        assert result["sql_exec_success"] is True
        assert result["sql_returns_rows"] is True
        assert result["sql_row_count"] > 0
        assert result["sql_error"] is None
        assert col in result["sql_columns"]

    @pytest.mark.asyncio
    async def test_invalid_sql_fails(self):
        result = await _try_exec_sql("SELECT nonexistent_col FROM customers")
        assert result["sql_exec_success"] is False
        assert result["sql_error"] is not None

    @pytest.mark.asyncio
    async def test_empty_query_fails(self):
        result = await _try_exec_sql("")
        assert result["sql_exec_success"] is False
        assert result["sql_error"] == "Empty query"

    @pytest.mark.asyncio
    async def test_non_select_rejected(self):
        result = await _try_exec_sql("DROP TABLE customers")
        assert result["sql_exec_success"] is False
        assert "Not a SELECT" in result["sql_error"]

    @pytest.mark.asyncio
    async def test_valid_sql_no_rows(self):
        q = "SELECT name FROM customers WHERE industry = 'Nonexistent'"
        result = await _try_exec_sql(q)
        assert result["sql_exec_success"] is True
        assert result["sql_returns_rows"] is False
        assert result["sql_row_count"] == 0

    @pytest.mark.asyncio
    async def test_wrapped_parens_stripped(self):
        result = await _try_exec_sql("(SELECT COUNT(*) FROM products)")
        assert result["sql_exec_success"] is True
        assert result["sql_returns_rows"] is True

    @pytest.mark.asyncio
    async def test_aggregate_query(self):
        q = "SELECT SUM(revenue) FROM sales WHERE year = 2024"
        result = await _try_exec_sql(q)
        assert result["sql_exec_success"] is True
        assert result["sql_returns_rows"] is True

    def test_score_includes_sql_metrics_when_present(self):
        """score() should report sql_exec_accuracy when predictions have sql fields."""
        preds = []
        for item in TEST_SET:
            pred = {
                "query":           item["query"],
                "expected_tool":   item["expected_tool"],
                "selected_tool":   item["expected_tool"],
                "tool_correct":    True,
                "arg_key_present": True,
                "arguments":       {item["expected_arg_key"]: "val"},
            }
            if item["expected_tool"] == "sql_query":
                pred["sql_exec_success"] = True
                pred["sql_returns_rows"] = True
                pred["sql_row_count"] = 5
                pred["sql_error"] = None
                pred["sql_columns"] = ["name"]
            preds.append(pred)
        s = score(_make_results(preds))
        sql = s["per_tool"]["sql_query"]
        assert sql["sql_evaluated"] == 80
        assert sql["sql_exec_success"] == 80
        assert sql["sql_exec_accuracy"] == pytest.approx(1.0)
        assert sql["sql_returns_rows"] == 80
        assert sql["sql_returns_rows_accuracy"] == pytest.approx(1.0)

    def test_score_sql_partial_failures(self):
        """Score with mix of success/failure SQL results."""
        preds = []
        for i, item in enumerate(TEST_SET):
            pred = {
                "query":           item["query"],
                "expected_tool":   item["expected_tool"],
                "selected_tool":   item["expected_tool"],
                "tool_correct":    True,
                "arg_key_present": True,
                "arguments":       {item["expected_arg_key"]: "val"},
            }
            if item["expected_tool"] == "sql_query":
                success = (i % 2 == 0)
                pred["sql_exec_success"] = success
                pred["sql_returns_rows"] = success
                pred["sql_row_count"] = 3 if success else 0
                pred["sql_error"] = None if success else "SQL error: no such column"
                pred["sql_columns"] = ["name"] if success else []
            preds.append(pred)
        s = score(_make_results(preds))
        sql = s["per_tool"]["sql_query"]
        assert sql["sql_evaluated"] == 80
        assert 0.4 <= sql["sql_exec_accuracy"] <= 0.6

    def test_score_no_sql_data_graceful(self):
        """Old predictions without sql fields should not crash."""
        s = score(_perfect_results())
        sql = s["per_tool"]["sql_query"]
        assert sql["sql_evaluated"] == 0
        assert sql["sql_exec_accuracy"] == pytest.approx(0.0)

    def test_score_sql_has_ci(self):
        """SQL metrics should include Wilson CIs."""
        preds = []
        for item in TEST_SET:
            pred = {
                "query":           item["query"],
                "expected_tool":   item["expected_tool"],
                "selected_tool":   item["expected_tool"],
                "tool_correct":    True,
                "arg_key_present": True,
                "arguments":       {item["expected_arg_key"]: "val"},
            }
            if item["expected_tool"] == "sql_query":
                pred["sql_exec_success"] = True
                pred["sql_returns_rows"] = True
                pred["sql_row_count"] = 1
                pred["sql_error"] = None
                pred["sql_columns"] = ["x"]
            preds.append(pred)
        s = score(_make_results(preds))
        sql = s["per_tool"]["sql_query"]
        lo, hi = sql["sql_exec_accuracy_ci"]
        assert 0.0 <= lo <= hi <= 1.0
        rlo, rhi = sql["sql_returns_rows_ci"]
        assert 0.0 <= rlo <= rhi <= 1.0

    def test_print_report_shows_sql_validation(self, capsys):
        """Report should include SQL validation section when data present."""
        preds = []
        for item in TEST_SET:
            pred = {
                "query":           item["query"],
                "expected_tool":   item["expected_tool"],
                "selected_tool":   item["expected_tool"],
                "tool_correct":    True,
                "arg_key_present": True,
                "arguments":       {item["expected_arg_key"]: "val"},
            }
            if item["expected_tool"] == "sql_query":
                pred["sql_exec_success"] = True
                pred["sql_returns_rows"] = True
                pred["sql_row_count"] = 1
                pred["sql_error"] = None
                pred["sql_columns"] = ["x"]
            preds.append(pred)
        print_report(_make_results(preds), title="SQL Test")
        captured = capsys.readouterr()
        assert "SQL execution validation" in captured.out
        assert "Executes without error" in captured.out
        assert "Returns at least 1 row" in captured.out

    def test_print_report_shows_sql_errors(self, capsys):
        """Report should list SQL error details."""
        preds = [{
            "query":           "What was total revenue?",
            "expected_tool":   "sql_query",
            "selected_tool":   "sql_query",
            "tool_correct":    True,
            "arg_key_present": True,
            "arguments":       {"query": "SELECT bad_col FROM sales"},
            "sql_exec_success": False,
            "sql_returns_rows": False,
            "sql_row_count":   0,
            "sql_error":       "no such column: bad_col",
            "sql_columns":     [],
        }]
        print_report(_make_results(preds), title="Error Test")
        captured = capsys.readouterr()
        assert "SQL errors" in captured.out
        assert "bad_col" in captured.out

    def test_print_report_shows_empty_results(self, capsys):
        """Report should list queries returning 0 rows."""
        preds = [{
            "query":           "List customers in Nonexistent industry.",
            "expected_tool":   "sql_query",
            "selected_tool":   "sql_query",
            "tool_correct":    True,
            "arg_key_present": True,
            "arguments":       {"query": "SELECT * FROM customers WHERE industry='Nonexistent'"},
            "sql_exec_success": True,
            "sql_returns_rows": False,
            "sql_row_count":   0,
            "sql_error":       None,
            "sql_columns":     ["id", "name"],
        }]
        print_report(_make_results(preds), title="Empty Test")
        captured = capsys.readouterr()
        assert "SQL returns 0 rows" in captured.out
