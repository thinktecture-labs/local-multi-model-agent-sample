"""Unit tests for QueryDecomposer (extracted from agent.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.engine.agent.query_decomposer import QueryDecomposer


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def decomposer(mock_client):
    return QueryDecomposer(mock_client, "gemma3-1b-ft")


@pytest.mark.unit
class TestMultiStepDetection:
    def test_single_step_queries(self, decomposer):
        assert not decomposer.detect_multi_step("What is our total revenue?")
        assert not decomposer.detect_multi_step("How many customers do we have?")
        assert not decomposer.detect_multi_step("Show me sales data")
        assert not decomposer.detect_multi_step("What is 5 * 10?")

    def test_multi_step_queries(self, decomposer):
        assert decomposer.detect_multi_step(
            "What is revenue, and what would it be with 15% growth?"
        )
        assert decomposer.detect_multi_step(
            "Show total sales, and calculate 10% discount"
        )
        assert decomposer.detect_multi_step(
            "How many customers do we have, and if we charge 999/month what is ARR?"
        )

    def test_and_calculate_pattern(self, decomposer):
        assert decomposer.detect_multi_step("Find revenue and calculate growth")

    def test_and_compute_pattern(self, decomposer):
        assert decomposer.detect_multi_step("Get total sales and compute margin")

    def test_and_what_percentage_pattern(self, decomposer):
        assert decomposer.detect_multi_step(
            "Show revenue and what percentage is from Enterprise?"
        )


@pytest.mark.unit
class TestDecompose:
    async def test_single_step_no_model_call(self, decomposer, mock_client):
        steps, trace = await decomposer.decompose("What is total revenue?")
        assert steps == ["What is total revenue?"]
        assert trace.action == "decompose_query"
        assert trace.details["method"] == "rule"
        mock_client.generate.assert_not_called()

    async def test_multi_step_calls_model(self, decomposer, mock_client):
        mock_client.generate.return_value = MagicMock(
            content='{"steps": ["Find revenue", "Calculate growth"]}',
            tokens_used=20, prompt_tokens=50, completion_tokens=20,
        )
        steps, trace = await decomposer.decompose(
            "What was revenue, and calculate 15% growth?"
        )
        assert len(steps) == 2
        assert trace.details["method"] == "rule+model"
        mock_client.generate.assert_called_once()

    async def test_fallback_on_bad_json(self, decomposer, mock_client):
        mock_client.generate.return_value = MagicMock(
            content="not valid json at all",
            tokens_used=10, prompt_tokens=30, completion_tokens=10,
        )
        steps, _ = await decomposer.decompose(
            "What was revenue, and calculate growth?"
        )
        # Should fall back to mechanical split
        assert len(steps) == 2

    async def test_fallback_splits_at_and(self, decomposer, mock_client):
        mock_client.generate.return_value = MagicMock(
            content='{"steps": []}',
            tokens_used=10, prompt_tokens=30, completion_tokens=10,
        )
        steps, _ = await decomposer.decompose(
            "Find total revenue, and calculate 10% growth"
        )
        assert len(steps) == 2
        assert "revenue" in steps[0].lower()


@pytest.mark.unit
class TestConcretizeStep:
    async def test_concretize_with_prior_results(self, decomposer, mock_client):
        mock_client.generate.return_value = MagicMock(
            content="Calculate 103200 * 1.15",
            tokens_used=10, prompt_tokens=40, completion_tokens=10,
        )
        concrete, trace = await decomposer.concretize_step(
            "Calculate 15% growth",
            [{"result": {"rows": [{"revenue": 103200}]}}],
            original_query="What was revenue and calculate 15% growth?",
        )
        assert concrete == "Calculate 103200 * 1.15"
        assert trace.action == "concretize_step"
