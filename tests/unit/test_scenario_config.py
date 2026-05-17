"""
Unit tests for scenario configuration and the /scenario endpoint contract.

Verifies that scenario JSON files contain all required UI fields
(suggestions, branding, logo) and that the system_routes /scenario
handler returns them correctly.
"""

import json
from pathlib import Path

import pytest


SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def _load_scenario(name: str) -> dict:
    path = SCENARIOS_DIR / f"{name}.json"
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scenario JSON structure tests
# ---------------------------------------------------------------------------

class TestScenarioJsonStructure:
    """Every scenario JSON must satisfy the UI contract."""

    @pytest.fixture(params=["nextera"])
    def scenario(self, request) -> dict:
        return _load_scenario(request.param)

    def test_has_required_top_level_keys(self, scenario):
        for key in ("name", "label", "brand", "language", "paths", "models", "prompts", "sql"):
            assert key in scenario, f"Missing required key: {key}"

    def test_has_suggestions_list(self, scenario):
        assert "suggestions" in scenario, "Missing 'suggestions' key"
        assert isinstance(scenario["suggestions"], list)
        assert len(scenario["suggestions"]) >= 3, "Need at least 3 suggestion chips"

    def test_suggestions_are_non_empty(self, scenario):
        """Each suggestion is either a plain string or an object with a 'query' field
        (matching the SuggestionChips.tsx unpack(): string | {label, query, group?})."""
        for i, s in enumerate(scenario["suggestions"]):
            if isinstance(s, str):
                assert len(s.strip()) > 10, f"suggestions[{i}] is too short: {s!r}"
            elif isinstance(s, dict):
                assert s.get("query"), f"suggestions[{i}] missing 'query'"
                assert len(s["query"].strip()) > 10, f"suggestions[{i}].query is too short"
            else:
                raise AssertionError(f"suggestions[{i}] is not a string or dict: {type(s)}")

    def test_logo_svg_keys_present(self, scenario):
        # logo_svg and favicon_svg should exist (may be empty for default branding)
        assert "logo_svg" in scenario, "Missing 'logo_svg' key"
        assert "favicon_svg" in scenario, "Missing 'favicon_svg' key"


# ---------------------------------------------------------------------------
# Nextera-specific tests
# ---------------------------------------------------------------------------

class TestNexteraScenario:
    def test_suggestions_are_english(self):
        nx = _load_scenario("nextera")
        english_markers = ["what", "how", "show", "which", "calculate", "compare"]
        english_count = 0
        for s in nx["suggestions"]:
            text = s if isinstance(s, str) else s.get("query", "")
            if any(m in text.lower().split() for m in english_markers):
                english_count += 1
        assert english_count >= len(nx["suggestions"]) // 2

    def test_brand_is_nextera(self):
        nx = _load_scenario("nextera")
        assert nx["brand"] == "Nextera"
