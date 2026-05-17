"""
Unit tests for finetune/eval_vision.py.

Tests the scoring, comparison, and reporting logic without any external services.
All tests are pure-Python — no async, no external services.
"""

import json
import pytest

from finetune.eval_vision import (
    IMAGE_NAMES,
    VALID_IMAGES,
    TEST_SET,
    check_keywords,
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

def _make_results(predictions: list[dict], model: str = "gemma3-4b-vision") -> dict:
    """Build a results dict in the same format as run_eval()."""
    return {
        "timestamp":   "2025-01-01T00:00:00",
        "model":       model,
        "n":           len(predictions),
        "predictions": predictions,
    }


def _perfect_results() -> dict:
    """All 10 queries answered correctly (keyword hit for every item)."""
    preds = [
        {
            "query":             item["query"],
            "image":             item["image"],
            "expected_keywords": item["expected_keywords"],
            "response":          " ".join(item["expected_keywords"]),
            "correct":           True,
        }
        for item in TEST_SET
    ]
    return _make_results(preds)


def _zero_results() -> dict:
    """All queries answered incorrectly (no keyword hit)."""
    preds = [
        {
            "query":             item["query"],
            "image":             item["image"],
            "expected_keywords": item["expected_keywords"],
            "response":          "completely irrelevant answer with no matches",
            "correct":           False,
        }
        for item in TEST_SET
    ]
    return _make_results(preds)


_FIRST_IMAGE = IMAGE_NAMES[0]
_FIRST_IMAGE_COUNT = sum(1 for item in TEST_SET if item["image"] == _FIRST_IMAGE)


def _partial_results() -> dict:
    """Correct only for the first image's items."""
    preds = []
    for item in TEST_SET:
        if item["image"] == _FIRST_IMAGE:
            preds.append({
                "query":             item["query"],
                "image":             item["image"],
                "expected_keywords": item["expected_keywords"],
                "response":          " ".join(item["expected_keywords"]),
                "correct":           True,
            })
        else:
            preds.append({
                "query":             item["query"],
                "image":             item["image"],
                "expected_keywords": item["expected_keywords"],
                "response":          "completely irrelevant answer with no matches",
                "correct":           False,
            })
    return _make_results(preds)


# ---------------------------------------------------------------------------
# Test set integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTestSet:
    def test_ten_items(self):
        assert len(TEST_SET) == 10

    def test_all_have_required_keys(self):
        required = {"query", "image", "expected_keywords"}
        for item in TEST_SET:
            assert required.issubset(item.keys()), (
                f"Missing keys in item: {required - item.keys()}"
            )

    def test_no_duplicate_queries(self):
        queries = [item["query"] for item in TEST_SET]
        assert len(queries) == len(set(queries))

    def test_all_queries_nonempty(self):
        for item in TEST_SET:
            assert item["query"].strip()

    def test_all_images_are_known(self):
        for item in TEST_SET:
            assert item["image"] in VALID_IMAGES, (
                f"Unknown image: {item['image']}. "
                f"Valid images: {VALID_IMAGES}"
            )


# ---------------------------------------------------------------------------
# check_keywords
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckKeywords:
    def test_case_insensitive_match(self):
        assert check_keywords("The answer is Q4", ["q4"]) is True

    def test_any_keyword_suffices(self):
        assert check_keywords("Revenue was 103M", ["Q4", "2024", "103"]) is True

    def test_no_match_returns_false(self):
        assert check_keywords("No relevant content here", ["Q4", "2024"]) is False

    def test_empty_keywords_returns_false(self):
        assert check_keywords("Some response", []) is False

    def test_empty_response(self):
        assert check_keywords("", ["keyword"]) is False

    def test_partial_word_match(self):
        # "growing" should match in "it is growing fast"
        assert check_keywords("it is growing fast", ["growing"]) is True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScore:
    def test_perfect_accuracy(self):
        s = score(_perfect_results())
        assert s["overall_accuracy"] == pytest.approx(1.0)
        assert s["overall_correct"] == 10

    def test_zero_accuracy(self):
        s = score(_zero_results())
        assert s["overall_accuracy"] == pytest.approx(0.0)
        assert s["overall_correct"] == 0

    def test_partial_accuracy(self):
        # First image's items correct, rest wrong
        s = score(_partial_results())
        assert s["overall_accuracy"] == pytest.approx(_FIRST_IMAGE_COUNT / 10)
        assert s["overall_correct"] == _FIRST_IMAGE_COUNT

    def test_per_image_keys(self):
        s = score(_perfect_results())
        for img in IMAGE_NAMES:
            assert img in s["per_image"]
            assert "n" in s["per_image"][img]
            assert "correct" in s["per_image"][img]
            assert "accuracy" in s["per_image"][img]

    def test_per_image_perfect(self):
        s = score(_perfect_results())
        for img in IMAGE_NAMES:
            assert s["per_image"][img]["accuracy"] == pytest.approx(1.0)

    def test_per_image_zero(self):
        s = score(_zero_results())
        for img in IMAGE_NAMES:
            assert s["per_image"][img]["accuracy"] == pytest.approx(0.0)

    def test_per_image_partial(self):
        s = score(_partial_results())
        assert s["per_image"][_FIRST_IMAGE]["accuracy"] == pytest.approx(1.0)
        # All other images should be 0.0
        for img in IMAGE_NAMES:
            if img != _FIRST_IMAGE:
                assert s["per_image"][img]["accuracy"] == pytest.approx(0.0)

    def test_per_image_counts(self):
        s = score(_perfect_results())
        total = sum(s["per_image"][img]["n"] for img in IMAGE_NAMES)
        assert total == 10

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

    def test_per_image_delta_keys(self):
        c = compare(_zero_results(), _perfect_results())
        for img in IMAGE_NAMES:
            assert img in c["per_image_delta"]

    def test_per_image_delta_values(self):
        before = _partial_results()  # first image correct, others wrong
        after  = _perfect_results()
        c = compare(before, after)
        # first image was already 1.0 → delta = 0
        assert c["per_image_delta"][_FIRST_IMAGE] == pytest.approx(0.0)
        # others were 0.0 → delta = 1.0
        for img in IMAGE_NAMES:
            if img != _FIRST_IMAGE:
                assert c["per_image_delta"][img] == pytest.approx(1.0)

    def test_before_and_after_scores_included(self):
        c = compare(_zero_results(), _perfect_results())
        assert "before" in c
        assert "after" in c
        assert c["before"]["overall_accuracy"] == pytest.approx(0.0)
        assert c["after"]["overall_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Print functions (smoke tests — just verify they don't raise)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrinting:
    def test_print_report_perfect(self, capsys):
        print_report(_perfect_results(), title="Test")
        captured = capsys.readouterr()
        assert "100.0%" in captured.out
        assert "10/10" in captured.out

    def test_print_report_with_incorrect(self, capsys):
        print_report(_zero_results(), title="Zero")
        captured = capsys.readouterr()
        assert "Incorrect" in captured.out

    def test_print_report_no_incorrect_section_when_perfect(self, capsys):
        print_report(_perfect_results())
        captured = capsys.readouterr()
        assert "Incorrect" not in captured.out

    def test_print_comparison_shows_delta(self, capsys):
        c = compare(_zero_results(), _perfect_results())
        print_comparison(c, labels=("Baseline", "Fine-tuned"))
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Fine-tuned" in captured.out
        assert "▲" in captured.out  # improvement arrow


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
        assert loaded["n"] == 10

    def test_loaded_results_scoreable(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "r.json")
        save_results(results, path)
        loaded = load_results(path)
        s = score(loaded)
        assert s["overall_accuracy"] == pytest.approx(1.0)
