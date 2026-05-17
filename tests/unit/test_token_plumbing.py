"""
Tests for token count propagation through the agent pipeline.

Verifies that LLMResponse prompt/completion breakdown flows through
ExecutionStep into AgentResponse and QueryResponse.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.conftest import make_stream

from src.engine.agent import AgentResponse, ExecutionStep, SmallLanguageModelAgentOrchestrator, Intent
from src.engine.inference.client import SmallLanguageModelRole, LLMResponse
from src.engine.tools.tool_result import ToolResult


# ---------------------------------------------------------------------------
# LLMResponse token breakdown
# ---------------------------------------------------------------------------

class TestLLMResponseTokens:
    def test_default_token_fields(self):
        r = LLMResponse(content="ok", model="test", tokens_used=10)
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0

    def test_explicit_token_fields(self):
        r = LLMResponse(
            content="ok", model="test", tokens_used=15,
            prompt_tokens=10, completion_tokens=5,
        )
        assert r.tokens_used == 15
        assert r.prompt_tokens == 10
        assert r.completion_tokens == 5


# ---------------------------------------------------------------------------
# ExecutionStep token fields
# ---------------------------------------------------------------------------

class TestExecutionStepTokens:
    def test_default_zero_tokens(self):
        step = ExecutionStep(action="test", model="test")
        assert step.tokens_used == 0
        assert step.prompt_tokens == 0
        assert step.completion_tokens == 0

    def test_explicit_tokens(self):
        step = ExecutionStep(
            action="test", model="test",
            tokens_used=20, prompt_tokens=12, completion_tokens=8,
        )
        assert step.tokens_used == 20
        assert step.prompt_tokens == 12
        assert step.completion_tokens == 8


# ---------------------------------------------------------------------------
# AgentResponse aggregation
# ---------------------------------------------------------------------------

class TestAgentResponseTokenAggregation:
    def test_empty_steps_zero_tokens(self):
        resp = AgentResponse(query="q", intent=Intent.DIRECT_ANSWER, response="r")
        assert resp.total_tokens == 0
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0

    def test_aggregates_across_steps(self):
        steps = [
            ExecutionStep(action="a", model="m", tokens_used=10, prompt_tokens=7, completion_tokens=3),
            ExecutionStep(action="b", model="m", tokens_used=20, prompt_tokens=15, completion_tokens=5),
        ]
        resp = AgentResponse(query="q", intent=Intent.RAG_QUERY, response="r", steps=steps)
        assert resp.total_tokens == 30
        assert resp.prompt_tokens == 22
        assert resp.completion_tokens == 8


# ---------------------------------------------------------------------------
# End-to-end: tokens flow from LLMResponse through handlers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.models = {
        SmallLanguageModelRole.INFERENCE: "gemma3",
        SmallLanguageModelRole.FUNCTION: "qwen",
        SmallLanguageModelRole.EMBEDDING: "embeddinggemma",
        SmallLanguageModelRole.VISION: "gemma3-4b",
    }
    return client


@pytest.fixture
def mock_tools():
    tools = MagicMock()
    tools.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "calculator", "parameters": {}}},
    ])
    return tools


class TestTokenPropagation:
    async def test_direct_answer_propagates_tokens(self, mock_client, mock_tools):
        mock_client.generate_stream = MagicMock(side_effect=make_stream(
            "response", tokens_used=5, prompt_tokens=3, completion_tokens=2,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        text, steps = await agent._direct_answer.handle("Hello")
        assert steps[0].tokens_used == 5
        assert steps[0].prompt_tokens == 3
        assert steps[0].completion_tokens == 2

    async def test_classify_returns_tokens(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="rag_query", model="gemma3", tokens_used=8,
            prompt_tokens=6, completion_tokens=2,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, resp = await agent._classifier.classify("What is X?")
        assert intent == Intent.RAG_QUERY
        assert resp is not None
        assert resp.tokens_used == 8

    async def test_session_token_counter(self, mock_client, mock_tools):
        """total_tokens_generated accumulates across queries."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3", tokens_used=10,
            prompt_tokens=7, completion_tokens=3,
        ))
        mock_client.generate_stream = MagicMock(side_effect=make_stream(
            "response", tokens_used=10, prompt_tokens=7, completion_tokens=3,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        await agent.process("q1")
        await agent.process("q2")

        # Each query: classify (10) + direct_response (10) = 20 tokens
        assert agent.total_tokens_generated == 40
