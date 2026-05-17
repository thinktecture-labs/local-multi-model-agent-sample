"""Unit tests for tool_argument_resolver helpers."""

import pytest

from src.engine.agent.tool_argument_resolver import (
    normalize_tool_query,
    patch_calculator_expression,
    rephrase_for_sql,
)


@pytest.mark.unit
class TestNormalizeToolQuery:
    def test_strips_currency_symbols(self):
        assert "$" not in normalize_tool_query("$500")
        assert "€" not in normalize_tool_query("€200")
        assert "£" not in normalize_tool_query("£100")

    def test_replaces_unicode_operators(self):
        assert "*" in normalize_tool_query("5 × 10")
        assert "/" in normalize_tool_query("10 ÷ 2")

    def test_removes_digit_grouping_commas(self):
        # The regex replaces one comma at a time (left to right)
        result = normalize_tool_query("45,000")
        assert result == "45000"

    def test_expands_whats(self):
        assert "What is" in normalize_tool_query("What's the total?")


@pytest.mark.unit
class TestRephraseForSql:
    def test_count_pattern(self):
        assert rephrase_for_sql("Count total customers") == "How many total customers are there?"

    def test_find_pattern(self):
        assert rephrase_for_sql("Find the best product") == "Show the best product."

    def test_look_up_pattern(self):
        assert rephrase_for_sql("Look up revenue data") == "Show revenue data."

    def test_get_pattern(self):
        assert rephrase_for_sql("Get customer names") == "Show customer names."

    def test_passthrough_unmatched(self):
        q = "Show me the revenue"
        assert rephrase_for_sql(q) == q

    def test_strips_trailing_dot(self):
        assert rephrase_for_sql("Find revenue.") == "Show revenue."


@pytest.mark.unit
class TestPatchCalculatorExpression:
    def test_patches_bare_percentage(self):
        args = {"expression": "15%"}
        result = patch_calculator_expression(args, "15% of 45000")
        assert result["expression"] == "15% of 45000"

    def test_no_patch_when_not_bare_percentage(self):
        args = {"expression": "100 * 1.15"}
        result = patch_calculator_expression(args, "15% of something")
        assert result["expression"] == "100 * 1.15"

    def test_no_patch_without_expression_key(self):
        args = {"query": "SELECT 1"}
        result = patch_calculator_expression(args, "test")
        assert "expression" not in result
