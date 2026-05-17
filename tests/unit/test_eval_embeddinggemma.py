"""
Unit tests for finetune/eval_embeddinggemma.py.

Tests scoring, comparison, and cosine similarity logic without any server calls.
All tests are pure-Python — no async, no external services.
"""

import json
import math
import pytest

from finetune.eval_embeddinggemma import (
    CORPUS,
    TEST_PAIRS,
    _cosine,
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

def _make_results(predictions: list[dict], model: str = "embeddinggemma") -> dict:
    return {
        "timestamp":   "2025-01-01T00:00:00",
        "model":       model,
        "n":           len(predictions),
        "predictions": predictions,
    }


def _perfect_results() -> dict:
    """Correct doc always ranked #1."""
    preds = [
        {
            "query":       item["query"],
            "correct_idx": item["correct_idx"],
            "rank":        1,
            "in_top_5":    True,
            "in_top_10":   True,
            "mrr":         1.0,
        }
        for item in TEST_PAIRS
    ]
    return _make_results(preds)


def _worst_results() -> dict:
    """Correct doc always ranked last (outside top 10)."""
    n_corpus = len(CORPUS)
    preds = [
        {
            "query":       item["query"],
            "correct_idx": item["correct_idx"],
            "rank":        n_corpus,
            "in_top_5":    False,
            "in_top_10":   False,
            "mrr":         0.0,
        }
        for item in TEST_PAIRS
    ]
    return _make_results(preds)


def _rank_2_results() -> dict:
    """Correct doc always ranked #2."""
    preds = [
        {
            "query":       item["query"],
            "correct_idx": item["correct_idx"],
            "rank":        2,
            "in_top_5":    True,
            "in_top_10":   True,
            "mrr":         0.5,
        }
        for item in TEST_PAIRS
    ]
    return _make_results(preds)


# ---------------------------------------------------------------------------
# Corpus and test pair integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCorpusAndPairs:
    def test_corpus_nonempty(self):
        assert len(CORPUS) > 0

    def test_all_pairs_have_valid_correct_idx(self):
        for pair in TEST_PAIRS:
            assert 0 <= pair["correct_idx"] < len(CORPUS)

    def test_all_pairs_have_query(self):
        for pair in TEST_PAIRS:
            assert pair["query"].strip()

    def test_no_duplicate_queries(self):
        queries = [pair["query"] for pair in TEST_PAIRS]
        assert len(queries) == len(set(queries))

    def test_corpus_passages_nonempty(self):
        for passage in CORPUS:
            assert passage.strip()

    def test_test_pairs_cover_all_corpus_indices(self):
        covered = {p["correct_idx"] for p in TEST_PAIRS}
        # All passages should be referenced at least once
        assert len(covered) == len(CORPUS)


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_normalized_vectors(self):
        # For unit vectors, dot product = cosine similarity
        a = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        b = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        assert _cosine(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScore:
    def test_perfect_scores(self):
        s = score(_perfect_results())
        assert s["mrr_at_10"]   == pytest.approx(1.0)
        assert s["recall_at_5"] == pytest.approx(1.0)
        assert s["n"] == len(TEST_PAIRS)

    def test_worst_scores(self):
        s = score(_worst_results())
        assert s["mrr_at_10"]   == pytest.approx(0.0)
        assert s["recall_at_5"] == pytest.approx(0.0)

    def test_rank_2_mrr(self):
        s = score(_rank_2_results())
        assert s["mrr_at_10"]   == pytest.approx(0.5)
        assert s["recall_at_5"] == pytest.approx(1.0)  # rank 2 is in top 5

    def test_empty_predictions(self):
        s = score(_make_results([]))
        assert s["mrr_at_10"]   == pytest.approx(0.0)
        assert s["recall_at_5"] == pytest.approx(0.0)
        assert s["n"] == 0

    def test_partial_recall(self):
        # Mix of rank 1 and rank 6 (outside top 5)
        preds = []
        for i, item in enumerate(TEST_PAIRS):
            rank = 1 if i % 2 == 0 else 6
            preds.append({
                "query":       item["query"],
                "correct_idx": item["correct_idx"],
                "rank":        rank,
                "in_top_5":    rank <= 5,
                "in_top_10":   rank <= 10,
                "mrr":         1.0 / rank if rank <= 10 else 0.0,
            })
        s = score(_make_results(preds))
        # Roughly half in top 5
        assert 0.3 <= s["recall_at_5"] <= 0.7


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompare:
    def test_improvement(self):
        c = compare(_worst_results(), _perfect_results())
        assert c["mrr_delta"]    == pytest.approx(1.0)
        assert c["recall_delta"] == pytest.approx(1.0)

    def test_regression(self):
        c = compare(_perfect_results(), _worst_results())
        assert c["mrr_delta"]    == pytest.approx(-1.0)
        assert c["recall_delta"] == pytest.approx(-1.0)

    def test_no_change(self):
        r = _perfect_results()
        c = compare(r, r)
        assert c["mrr_delta"]    == pytest.approx(0.0)
        assert c["recall_delta"] == pytest.approx(0.0)

    def test_before_after_in_output(self):
        c = compare(_worst_results(), _perfect_results())
        assert "before" in c
        assert "after" in c
        assert c["before"]["mrr_at_10"] == pytest.approx(0.0)
        assert c["after"]["mrr_at_10"]  == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Printing (smoke tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrinting:
    def test_print_report_perfect(self, capsys):
        print_report(_perfect_results(), title="Test")
        captured = capsys.readouterr()
        assert "MRR@10" in captured.out
        assert "1.0000" in captured.out

    def test_print_report_shows_hard_queries(self, capsys):
        print_report(_worst_results())
        captured = capsys.readouterr()
        assert "Not in top-5" in captured.out

    def test_print_comparison(self, capsys):
        c = compare(_worst_results(), _perfect_results())
        print_comparison(c, labels=("Baseline", "Fine-tuned"))
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Fine-tuned" in captured.out
        assert "▲" in captured.out


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
        assert loaded["n"] == len(TEST_PAIRS)

    def test_loaded_results_scoreable(self, tmp_path):
        results = _perfect_results()
        path = str(tmp_path / "r.json")
        save_results(results, path)
        loaded = load_results(path)
        s = score(loaded)
        assert s["mrr_at_10"] == pytest.approx(1.0)
