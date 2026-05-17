"""
Unit tests for finetune/eval_gemma3.py.

Tests the scoring, comparison, and reporting logic without any external services.
All tests are pure-Python — no async, no external services.
"""

import json
import pytest

from finetune.eval_gemma3 import (
    CLASSES,
    TEST_SET,
    compare,
    load_results,
    print_comparison,
    print_report,
    save_results,
    score,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic results dicts
# ---------------------------------------------------------------------------

def _make_results(predictions: list[dict], model: str = "gemma3:1b-it") -> dict:
    """Build a results dict in the same format as run_eval()."""
    return {
        "timestamp":   "2025-01-01T00:00:00",
        "model":       model,
        "n":           len(predictions),
        "predictions": predictions,
    }


def _perfect_results() -> dict:
    """All 20 queries correctly classified."""
    preds = [
        {"query": item["query"], "expected": item["intent"],
         "predicted": item["intent"], "correct": True}
        for item in TEST_SET
    ]
    return _make_results(preds)


def _zero_results() -> dict:
    """All queries misclassified as 'direct_answer'."""
    wrong_class = "direct_answer"
    preds = [
        {
            "query":     item["query"],
            "expected":  item["intent"],
            "predicted": wrong_class if item["intent"] != wrong_class else "rag_query",
            "correct":   False,
        }
        for item in TEST_SET
    ]
    return _make_results(preds)


def _partial_results(correct_classes: set[str]) -> dict:
    """Correct only for the specified intent classes."""
    preds = []
    for item in TEST_SET:
        if item["intent"] in correct_classes:
            preds.append({
                "query": item["query"], "expected": item["intent"],
                "predicted": item["intent"], "correct": True,
            })
        else:
            other = next(c for c in CLASSES if c != item["intent"])
            preds.append({
                "query": item["query"], "expected": item["intent"],
                "predicted": other, "correct": False,
            })
    return _make_results(preds)


# ---------------------------------------------------------------------------
# Test set integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTestSet:
    def test_total_query_count(self):
        assert len(TEST_SET) == 180

    def test_queries_per_class(self):
        for cls in CLASSES:
            count = sum(1 for item in TEST_SET if item["intent"] == cls)
            assert count == 60, f"Expected 60 {cls} queries, got {count}"

    def test_all_intents_are_valid(self):
        for item in TEST_SET:
            assert item["intent"] in CLASSES

    def test_no_duplicate_queries(self):
        queries = [item["query"] for item in TEST_SET]
        assert len(queries) == len(set(queries))

    def test_all_queries_are_nonempty(self):
        for item in TEST_SET:
            assert item["query"].strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScore:
    def test_perfect_accuracy(self):
        s = score(_perfect_results())
        assert s["overall_accuracy"] == pytest.approx(1.0)
        assert s["overall_correct"] == 180

    def test_zero_accuracy(self):
        s = score(_zero_results())
        assert s["overall_accuracy"] == pytest.approx(0.0)
        assert s["overall_correct"] == 0

    def test_partial_accuracy(self):
        # Correct on rag_query (60/180) and tool_use (60/180) = 120/180 = 0.667
        s = score(_partial_results({"rag_query", "tool_use"}))
        assert s["overall_accuracy"] == pytest.approx(2/3, abs=0.01)
        assert s["overall_correct"] == 120

    def test_per_class_keys_present(self):
        s = score(_perfect_results())
        for cls in CLASSES:
            assert cls in s["per_class"]
            assert "n" in s["per_class"][cls]
            assert "correct" in s["per_class"][cls]
            assert "accuracy" in s["per_class"][cls]
            assert "ci" in s["per_class"][cls]

    def test_score_has_overall_ci(self):
        s = score(_perfect_results())
        assert "overall_ci" in s
        lo, hi = s["overall_ci"]
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_narrows_vs_small_sample(self):
        small = score(_make_results([
            {"query": "x", "expected": "rag_query", "predicted": "rag_query", "correct": True}
        ] * 10))
        large = score(_perfect_results())
        small_width = small["overall_ci"][1] - small["overall_ci"][0]
        large_width = large["overall_ci"][1] - large["overall_ci"][0]
        assert large_width < small_width

    def test_per_class_perfect(self):
        s = score(_perfect_results())
        for cls in CLASSES:
            assert s["per_class"][cls]["accuracy"] == pytest.approx(1.0)
            assert s["per_class"][cls]["n"] == 60
            assert s["per_class"][cls]["correct"] == 60

    def test_per_class_zero(self):
        s = score(_zero_results())
        for cls in CLASSES:
            assert s["per_class"][cls]["accuracy"] == pytest.approx(0.0)

    def test_per_class_selective(self):
        s = score(_partial_results({"rag_query"}))
        assert s["per_class"]["rag_query"]["accuracy"] == pytest.approx(1.0)
        assert s["per_class"]["tool_use"]["accuracy"] == pytest.approx(0.0)

    def test_unknown_prediction_counts_as_wrong(self):
        preds = [
            {"query": "x", "expected": "rag_query",
             "predicted": "unknown", "correct": False}
        ]
        s = score(_make_results(preds))
        assert s["overall_correct"] == 0

    def test_empty_predictions(self):
        s = score(_make_results([]))
        assert s["overall_accuracy"] == pytest.approx(0.0)
        assert s["n"] == 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompare:
    def test_improvement(self):
        before = _zero_results()
        after  = _perfect_results()
        c = compare(before, after)
        assert c["overall_delta"] == pytest.approx(1.0)

    def test_regression(self):
        before = _perfect_results()
        after  = _zero_results()
        c = compare(before, after)
        assert c["overall_delta"] == pytest.approx(-1.0)

    def test_no_change(self):
        results = _perfect_results()
        c = compare(results, results)
        assert c["overall_delta"] == pytest.approx(0.0)

    def test_per_class_delta_keys(self):
        c = compare(_zero_results(), _perfect_results())
        for cls in CLASSES:
            assert cls in c["per_class_delta"]

    def test_per_class_delta_values(self):
        before = _partial_results({"rag_query"})
        after  = _perfect_results()
        c = compare(before, after)
        # rag_query was already 1.0 → delta = 0
        assert c["per_class_delta"]["rag_query"] == pytest.approx(0.0)
        # others were 0.0 → delta = 1.0
        assert c["per_class_delta"]["tool_use"] == pytest.approx(1.0)

    def test_before_and_after_scores_included(self):
        c = compare(_zero_results(), _perfect_results())
        assert "before" in c
        assert "after" in c
        assert c["before"]["overall_accuracy"] == pytest.approx(0.0)
        assert c["after"]["overall_accuracy"] == pytest.approx(1.0)

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
        results = _perfect_results()
        c = compare(results, results)
        assert c["mcnemar"]["significant_at_05"] is False


# ---------------------------------------------------------------------------
# Print functions (smoke tests — just verify they don't raise)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrinting:
    def test_print_report_perfect(self, capsys):
        print_report(_perfect_results(), title="Test")
        captured = capsys.readouterr()
        assert "100.0%" in captured.out
        assert "180/180" in captured.out

    def test_print_report_with_misclassified(self, capsys):
        print_report(_zero_results(), title="Zero")
        captured = capsys.readouterr()
        assert "Misclassified" in captured.out

    def test_print_report_no_misclassified_section_when_perfect(self, capsys):
        print_report(_perfect_results())
        captured = capsys.readouterr()
        assert "Misclassified" not in captured.out

    def test_print_report_shows_ci(self, capsys):
        print_report(_perfect_results(), title="CI Test")
        captured = capsys.readouterr()
        assert "[" in captured.out and "]" in captured.out

    def test_print_comparison_shows_delta(self, capsys):
        c = compare(_zero_results(), _perfect_results())
        print_comparison(c, labels=("Baseline", "Fine-tuned"))
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Fine-tuned" in captured.out
        assert "▲" in captured.out  # improvement arrow

    def test_print_comparison_shows_p_value(self, capsys):
        c = compare(_zero_results(), _perfect_results())
        print_comparison(c, labels=("Base", "FT"))
        captured = capsys.readouterr()
        assert "p=" in captured.out


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "results.json")
        save_results(results, path)
        loaded = load_results(path)
        assert loaded["n"] == results["n"]
        assert loaded["model"] == results["model"]
        assert len(loaded["predictions"]) == len(results["predictions"])

    def test_save_creates_parent_dirs(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "nested" / "deep" / "results.json")
        save_results(results, path)
        loaded = load_results(path)
        assert loaded["n"] == 180

    def test_loaded_results_scoreable(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "r.json")
        save_results(results, path)
        loaded = load_results(path)
        s = score(loaded)
        assert s["overall_accuracy"] == pytest.approx(1.0)


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
            "data/training-data/gemma3_intent.jsonl",
            query_key="input",
            threshold=0.7,
        )
        assert len(overlaps) == 0, (
            f"Found {len(overlaps)} overlapping queries:\n"
            + "\n".join(f"  eval: {o['eval_query']!r}\n  train: {o['train_query']!r}\n  sim: {o['similarity']}"
                        for o in overlaps[:5])
        )
