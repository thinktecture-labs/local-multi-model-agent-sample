"""
Unit tests for finetune/eval_adversarial.py.

Tests the scoring, reporting logic, test set integrity, and training overlap.
All tests are pure-Python — no async, no external services.
"""

import json
import pytest

from finetune.eval_adversarial import (
    CATEGORIES,
    TEST_SET,
    print_report,
    score,
)
from finetune.eval_base import save_results


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
    """All 60 queries correctly classified as direct_answer."""
    preds = [
        {"query": item["query"], "expected": item["intent"],
         "predicted": item["intent"], "correct": True, "category": item["category"]}
        for item in TEST_SET
    ]
    return _make_results(preds)


def _zero_results() -> dict:
    """All queries misclassified (not direct_answer)."""
    preds = [
        {"query": item["query"], "expected": item["intent"],
         "predicted": "rag_query", "correct": False, "category": item["category"]}
        for item in TEST_SET
    ]
    return _make_results(preds)


def _partial_results(correct_categories: set[str]) -> dict:
    """Correct only for the specified categories."""
    preds = []
    for item in TEST_SET:
        if item["category"] in correct_categories:
            preds.append({
                "query": item["query"], "expected": item["intent"],
                "predicted": item["intent"], "correct": True,
                "category": item["category"],
            })
        else:
            preds.append({
                "query": item["query"], "expected": item["intent"],
                "predicted": "tool_use", "correct": False,
                "category": item["category"],
            })
    return _make_results(preds)


# ---------------------------------------------------------------------------
# Test set integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTestSet:
    def test_total_query_count(self):
        assert len(TEST_SET) == 60

    def test_all_intents_are_direct_answer(self):
        for item in TEST_SET:
            assert item["intent"] == "direct_answer", (
                f"Expected direct_answer, got {item['intent']!r} for: {item['query']!r}"
            )

    def test_all_categories_are_valid(self):
        for item in TEST_SET:
            assert item["category"] in CATEGORIES, (
                f"Invalid category {item['category']!r} for: {item['query']!r}"
            )

    def test_each_category_has_10_queries(self):
        for cat in CATEGORIES:
            count = sum(1 for item in TEST_SET if item["category"] == cat)
            assert count == 10, f"Expected 10 {cat} queries, got {count}"

    def test_no_duplicate_queries(self):
        queries = [item["query"] for item in TEST_SET]
        assert len(queries) == len(set(queries))

    def test_all_queries_are_nonempty(self):
        for item in TEST_SET:
            assert item["query"].strip()

    def test_all_items_have_required_keys(self):
        for item in TEST_SET:
            assert "query" in item
            assert "intent" in item
            assert "category" in item


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScore:
    def test_perfect_accuracy(self):
        s = score(_perfect_results())
        assert s["overall_accuracy"] == pytest.approx(1.0)
        assert s["overall_correct"] == 60

    def test_zero_accuracy(self):
        s = score(_zero_results())
        assert s["overall_accuracy"] == pytest.approx(0.0)
        assert s["overall_correct"] == 0

    def test_partial_accuracy(self):
        # Correct on off_topic (10) and injection (10) = 20/60 = 1/3
        s = score(_partial_results({"off_topic", "injection"}))
        assert s["overall_accuracy"] == pytest.approx(20 / 60, abs=0.01)
        assert s["overall_correct"] == 20

    def test_per_category_keys_present(self):
        s = score(_perfect_results())
        for cat in CATEGORIES:
            assert cat in s["per_category"]
            assert "n" in s["per_category"][cat]
            assert "correct" in s["per_category"][cat]
            assert "accuracy" in s["per_category"][cat]
            assert "ci" in s["per_category"][cat]

    def test_score_has_overall_ci(self):
        s = score(_perfect_results())
        assert "overall_ci" in s
        lo, hi = s["overall_ci"]
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_narrows_vs_small_sample(self):
        small = score(_make_results([
            {"query": "x", "expected": "direct_answer",
             "predicted": "direct_answer", "correct": True, "category": "off_topic"}
        ] * 10))
        large = score(_perfect_results())
        small_width = small["overall_ci"][1] - small["overall_ci"][0]
        large_width = large["overall_ci"][1] - large["overall_ci"][0]
        assert large_width < small_width

    def test_per_category_perfect(self):
        s = score(_perfect_results())
        for cat in CATEGORIES:
            assert s["per_category"][cat]["accuracy"] == pytest.approx(1.0)
            assert s["per_category"][cat]["n"] == 10
            assert s["per_category"][cat]["correct"] == 10

    def test_per_category_zero(self):
        s = score(_zero_results())
        for cat in CATEGORIES:
            assert s["per_category"][cat]["accuracy"] == pytest.approx(0.0)

    def test_per_category_selective(self):
        s = score(_partial_results({"off_topic"}))
        assert s["per_category"]["off_topic"]["accuracy"] == pytest.approx(1.0)
        assert s["per_category"]["injection"]["accuracy"] == pytest.approx(0.0)

    def test_unknown_prediction_counts_as_wrong(self):
        preds = [
            {"query": "x", "expected": "direct_answer",
             "predicted": "unknown", "correct": False, "category": "off_topic"}
        ]
        s = score(_make_results(preds))
        assert s["overall_correct"] == 0

    def test_empty_predictions(self):
        s = score(_make_results([]))
        assert s["overall_accuracy"] == pytest.approx(0.0)
        assert s["n"] == 0


# ---------------------------------------------------------------------------
# Print functions (smoke tests — just verify they don't raise)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrinting:
    def test_print_report_perfect(self, capsys):
        print_report(_perfect_results(), title="Test")
        captured = capsys.readouterr()
        assert "100.0%" in captured.out
        assert "60/60" in captured.out

    def test_print_report_with_misrouted(self, capsys):
        print_report(_zero_results(), title="Zero")
        captured = capsys.readouterr()
        assert "Misrouted" in captured.out

    def test_print_report_no_misrouted_section_when_perfect(self, capsys):
        print_report(_perfect_results())
        captured = capsys.readouterr()
        assert "Misrouted" not in captured.out

    def test_print_report_shows_ci(self, capsys):
        print_report(_perfect_results(), title="CI Test")
        captured = capsys.readouterr()
        assert "[" in captured.out and "]" in captured.out

    def test_print_report_shows_categories(self, capsys):
        print_report(_perfect_results())
        captured = capsys.readouterr()
        for cat in CATEGORIES:
            assert cat in captured.out


# ---------------------------------------------------------------------------
# Train/eval overlap
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOverlap:
    def test_no_overlap_with_training_data(self):
        """Verify zero overlap between adversarial TEST_SET and training JSONL."""
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
