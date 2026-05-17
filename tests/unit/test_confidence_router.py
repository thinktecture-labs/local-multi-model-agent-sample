"""
Unit tests for ConfidenceRouter.

Tests the 7-factor confidence scoring and escalation decision logic.
"""

import pytest

from src.engine.scaffolding.confidence_router import score_confidence, ConfidenceResult


@pytest.mark.unit
class TestConfidenceScoring:
    def test_confident_response_high_score(self):
        result = score_confidence(
            "PostgreSQL 16 was chosen for its open-source license and JSONB support.",
            "Which database did they choose?",
        )
        assert result.score >= 0.7
        assert not result.should_escalate

    def test_uncertain_response_low_score(self):
        result = score_confidence(
            "I don't know the answer to that question.",
            "What is the company's revenue?",
        )
        assert result.score < 0.6
        assert result.should_escalate

    def test_hedging_reduces_score(self):
        result = score_confidence(
            "It might be PostgreSQL, but it could be MongoDB. Perhaps they use both.",
            "Which database?",
        )
        hedging_penalty = result.factors.get("hedging", 0)
        assert hedging_penalty < 0

    def test_apology_reduces_score(self):
        result = score_confidence(
            "I apologize, but I cannot provide that information accurately.",
            "What is the pricing?",
        )
        assert result.factors.get("apology", 0) < 0

    def test_clarification_question_reduces_score(self):
        result = score_confidence(
            "Could you clarify what you mean by data residency?",
            "What about data residency?",
        )
        assert result.factors.get("clarification", 0) < 0

    def test_short_response_to_complex_query(self):
        result = score_confidence(
            "Yes.",
            "Can you explain the full architecture including database choices, API design, and security requirements?",
        )
        assert result.factors.get("response_length", 0) < 0

    def test_confident_language_bonus(self):
        result = score_confidence(
            "The answer is definitely PostgreSQL 16 for the primary database.",
            "Which database?",
        )
        assert result.factors.get("confident_language", 0) > 0

    def test_structured_response_bonus(self):
        result = score_confidence(
            "The key points are:\n- PostgreSQL 16\n- AES-256 encryption\n- EU data residency",
            "Summarize the architecture.",
        )
        assert result.factors.get("structured", 0) > 0

    def test_score_clamped_to_range(self):
        # Even with many negative factors, score stays >= 0.1
        result = score_confidence(
            "I don't know. I'm not sure. I apologize. Could you clarify?",
            "Very complex question about many things that requires deep analysis",
        )
        assert 0.1 <= result.score <= 1.0

    def test_empty_response(self):
        result = score_confidence("", "What is this?")
        assert result.score < 0.8  # Short response penalty
        assert isinstance(result.factors, dict)

    def test_result_has_score_pct(self):
        result = score_confidence("Good answer.", "Question?")
        assert isinstance(result.score_pct, int)
        assert 0 <= result.score_pct <= 100


@pytest.mark.unit
class TestEscalationDecision:
    def test_below_threshold_escalates(self):
        result = score_confidence(
            "I don't know.",
            "Complex question?",
            threshold=0.6,
        )
        assert result.should_escalate

    def test_above_threshold_no_escalation(self):
        result = score_confidence(
            "The database is PostgreSQL 16 with row-level security for multi-tenant isolation.",
            "Which database?",
            threshold=0.6,
        )
        assert not result.should_escalate

    def test_custom_threshold(self):
        result = score_confidence(
            "PostgreSQL was chosen.",
            "Which database?",
            threshold=0.9,  # very high threshold
        )
        # With threshold at 0.9, even a decent response should escalate
        assert result.should_escalate

    def test_factors_dict_populated(self):
        result = score_confidence("Some answer.", "Some question?")
        assert "base" in result.factors
        assert "strong_uncertainty" in result.factors
        assert "hedging" in result.factors
