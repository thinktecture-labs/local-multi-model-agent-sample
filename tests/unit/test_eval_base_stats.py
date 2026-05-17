"""
Unit tests for finetune/eval_base.py statistical functions.

Tests Wilson CI, bootstrap CI, McNemar's test, and overlap checker.
All tests are pure-Python — no async, no external services.
"""

import json
import math
import tempfile

import pytest

from finetune.eval_base import (
    bootstrap_ci,
    check_eval_training_overlap,
    fmt_ci,
    fmt_pct,
    fmt_pct_with_ci,
    mcnemar_test,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWilsonCI:
    def test_perfect_accuracy(self):
        lo, hi = wilson_ci(20, 20)
        assert hi == pytest.approx(1.0, abs=0.01)
        assert lo > 0.80

    def test_zero_accuracy(self):
        lo, hi = wilson_ci(0, 20)
        assert lo == pytest.approx(0.0, abs=0.01)
        assert hi < 0.20

    def test_empty_sample(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_known_value_19_of_20(self):
        # 19/20 = 95%, Wilson 95% CI should be approximately [0.754, 0.996]
        lo, hi = wilson_ci(19, 20)
        assert 0.72 < lo < 0.78
        assert 0.98 < hi <= 1.0

    def test_known_value_50_of_100(self):
        # 50/100 = 50%, Wilson 95% CI should be approximately [0.402, 0.598]
        lo, hi = wilson_ci(50, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_ci_narrows_with_more_samples(self):
        lo1, hi1 = wilson_ci(19, 20)
        lo2, hi2 = wilson_ci(190, 200)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 < width1

    def test_lower_bound_nonnegative(self):
        lo, hi = wilson_ci(1, 100)
        assert lo >= 0.0

    def test_upper_bound_at_most_one(self):
        lo, hi = wilson_ci(99, 100)
        assert hi <= 1.0

    def test_bounds_order(self):
        lo, hi = wilson_ci(57, 60)
        assert lo < hi

    def test_custom_z_score(self):
        # 99% CI (z=2.576) should be wider than 95% CI (z=1.96)
        lo95, hi95 = wilson_ci(50, 100, z=1.96)
        lo99, hi99 = wilson_ci(50, 100, z=2.576)
        assert (hi99 - lo99) > (hi95 - lo95)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBootstrapCI:
    def test_perfect_accuracy(self):
        lo, hi = bootstrap_ci([True] * 100)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_zero_accuracy(self):
        lo, hi = bootstrap_ci([False] * 100)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_empty_list(self):
        lo, hi = bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_reproducible_with_seed(self):
        data = [True] * 80 + [False] * 20
        lo1, hi1 = bootstrap_ci(data, seed=42)
        lo2, hi2 = bootstrap_ci(data, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_narrows_with_more_samples(self):
        small = [True] * 8 + [False] * 2
        large = [True] * 80 + [False] * 20
        lo_s, hi_s = bootstrap_ci(small)
        lo_l, hi_l = bootstrap_ci(large)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_bounds_contain_point_estimate(self):
        data = [True] * 75 + [False] * 25
        lo, hi = bootstrap_ci(data)
        point = sum(data) / len(data)
        assert lo <= point <= hi

    def test_bounds_order(self):
        data = [True] * 57 + [False] * 3
        lo, hi = bootstrap_ci(data)
        assert lo <= hi


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMcNemarTest:
    def test_all_concordant(self):
        """When both models agree on everything, test should be non-significant."""
        before = [True, True, False, False]
        after = [True, True, False, False]
        result = mcnemar_test(before, after)
        assert result["n_discordant"] == 0
        assert result["p_value"] == pytest.approx(1.0)
        assert result["significant_at_05"] is False

    def test_clear_improvement(self):
        """When after model fixes many errors, test should detect it."""
        before = [False] * 20 + [True] * 10
        after = [True] * 20 + [True] * 10
        result = mcnemar_test(before, after)
        assert result["c"] == 20  # 20 improved
        assert result["b"] == 0   # 0 regressed
        assert result["significant_at_05"] is True

    def test_clear_regression(self):
        """When after model introduces many errors, should detect it."""
        before = [True] * 20 + [False] * 10
        after = [False] * 20 + [False] * 10
        result = mcnemar_test(before, after)
        assert result["b"] == 20  # 20 regressed
        assert result["c"] == 0   # 0 improved

    def test_symmetric_discordant(self):
        """Equal improvements and regressions should not be significant."""
        before = [True, False, True, False, True, False, True, False]
        after = [False, True, False, True, False, True, False, True]
        result = mcnemar_test(before, after)
        assert result["b"] == result["c"]
        assert result["significant_at_05"] is False

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            mcnemar_test([True, False], [True])

    def test_result_keys(self):
        result = mcnemar_test([True], [True])
        assert "chi2" in result
        assert "p_value" in result
        assert "n_discordant" in result
        assert "b" in result
        assert "c" in result
        assert "significant_at_05" in result

    def test_p_value_range(self):
        before = [True] * 10 + [False] * 10
        after = [False] * 5 + [True] * 15
        result = mcnemar_test(before, after)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Overlap checker
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOverlapChecker:
    def test_exact_match_detected(self, tmp_path):
        training_file = tmp_path / "train.jsonl"
        training_file.write_text(
            json.dumps({"input": "What is the total revenue?"}) + "\n"
            + json.dumps({"input": "Show all customers."}) + "\n"
        )
        overlaps = check_eval_training_overlap(
            ["What is the total revenue?"],
            str(training_file),
            query_key="input",
            threshold=0.7,
        )
        assert len(overlaps) == 1
        assert overlaps[0]["similarity"] == pytest.approx(1.0)

    def test_no_overlap(self, tmp_path):
        training_file = tmp_path / "train.jsonl"
        training_file.write_text(
            json.dumps({"input": "Alpha beta gamma delta."}) + "\n"
        )
        overlaps = check_eval_training_overlap(
            ["Completely different query about Nextera pricing."],
            str(training_file),
            query_key="input",
            threshold=0.7,
        )
        assert len(overlaps) == 0

    def test_partial_overlap_below_threshold(self, tmp_path):
        training_file = tmp_path / "train.jsonl"
        training_file.write_text(
            json.dumps({"input": "What is the total revenue in Q1 2024?"}) + "\n"
        )
        overlaps = check_eval_training_overlap(
            ["What is the total number of customers?"],
            str(training_file),
            query_key="input",
            threshold=0.7,
        )
        assert len(overlaps) == 0

    def test_missing_file_returns_empty(self):
        overlaps = check_eval_training_overlap(
            ["Any query"],
            "/nonexistent/path.jsonl",
            threshold=0.7,
        )
        assert overlaps == []

    def test_custom_query_key(self, tmp_path):
        training_file = tmp_path / "train.jsonl"
        training_file.write_text(
            json.dumps({"query": "What is the total revenue?"}) + "\n"
        )
        overlaps = check_eval_training_overlap(
            ["What is the total revenue?"],
            str(training_file),
            query_key="query",
            threshold=0.7,
        )
        assert len(overlaps) == 1

    def test_empty_eval_queries(self, tmp_path):
        training_file = tmp_path / "train.jsonl"
        training_file.write_text(
            json.dumps({"input": "Something."}) + "\n"
        )
        overlaps = check_eval_training_overlap(
            [],
            str(training_file),
            threshold=0.7,
        )
        assert overlaps == []


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatHelpers:
    def test_fmt_pct(self):
        assert fmt_pct(0.95) == "95.0%"
        assert fmt_pct(0.0) == "0.0%"
        assert fmt_pct(1.0) == "100.0%"

    def test_fmt_ci(self):
        assert fmt_ci(0.852, 0.948) == "[85.2%, 94.8%]"

    def test_fmt_pct_with_ci(self):
        result = fmt_pct_with_ci(19, 20)
        assert "95.0%" in result
        assert "[" in result and "]" in result

    def test_fmt_pct_with_ci_empty(self):
        assert fmt_pct_with_ci(0, 0) == "N/A"
