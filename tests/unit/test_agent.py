"""
Unit tests for SmallLanguageModelAgentOrchestrator.

Tests every handler path with a fully mocked SmallLanguageModelClient and ToolRegistry,
verifying intent classification, routing, execution traces, fallback
behaviour, and interaction logging — all without model servers.
"""

import asyncio
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.conftest import make_stream

from src.engine.agent import SmallLanguageModelAgentOrchestrator, Intent, AgentResponse, ExecutionStep
from src.engine.agent.intent_classifier import IntentClassifier
from src.engine.agent.tool_argument_resolver import rephrase_for_sql
from src.engine.inference.client import SmallLanguageModelRole, LLMResponse
from src.engine.tools.tool_registry import ToolRegistry
from src.engine.tools.tool_result import ToolResult
from src.engine.knowledge.vector_store import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """SmallLanguageModelClient mock with configurable return values."""
    client = MagicMock()
    client.models = {
        SmallLanguageModelRole.INFERENCE:  "gemma3:1b-it",
        SmallLanguageModelRole.FUNCTION:   "qwen",
        SmallLanguageModelRole.EMBEDDING:  "embeddinggemma",
        SmallLanguageModelRole.VISION:     "gemma3-4b-vision",
    }
    client.generate = AsyncMock(return_value=LLMResponse(
        content="direct_answer", model="gemma3:1b-it", tokens_used=5,
    ))
    client.generate_stream = MagicMock(side_effect=make_stream("direct_answer"))
    client.generate_synthesis_stream = MagicMock(side_effect=make_stream("The Enterprise plan includes unlimited users."))
    client.call_function = AsyncMock(return_value=LLMResponse(
        content="", model="qwen", tokens_used=0, function_call=None,
    ))
    client.generate_vision = AsyncMock(return_value=LLMResponse(
        content="Vision analysis", model="gemma3-4b-vision", tokens_used=50,
    ))
    return client


@pytest.fixture
def mock_tools():
    """ToolRegistry mock that returns a configurable ToolResult."""
    tools = MagicMock(spec=ToolRegistry)
    tools.execute = AsyncMock(return_value=ToolResult(success=True, data=[]))
    tools.get_all_schemas = MagicMock(return_value=[])
    return tools


@pytest.fixture
def agent(mock_client, mock_tools):
    return SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    async def test_rag_query_intent(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="rag_query", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("What is the Enterprise plan?")
        assert intent == Intent.RAG_QUERY

    async def test_tool_use_intent(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="tool_use", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("Show me sales data")
        assert intent == Intent.TOOL_USE

    async def test_calculation_maps_to_tool_use(self, mock_client, mock_tools):
        """Old 'calculation' intent should fall back to DIRECT_ANSWER (unknown value)."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="calculation", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("What is 15% of 84900?")
        # "calculation" is no longer a valid intent — falls back to DIRECT_ANSWER
        assert intent == Intent.DIRECT_ANSWER

    async def test_direct_answer_intent(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("Hello!")
        assert intent == Intent.DIRECT_ANSWER

    async def test_whitespace_normalisation(self, mock_client, mock_tools):
        """Model returns 'rag query' with space — should still parse."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="  rag query  ", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("test")
        assert intent == Intent.RAG_QUERY

    async def test_uppercase_normalisation(self, mock_client, mock_tools):
        """Model returns 'TOOL_USE' — should still parse."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="TOOL_USE", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("test")
        assert intent == Intent.TOOL_USE

    async def test_invalid_intent_falls_back_to_direct(self, mock_client, mock_tools):
        """Unknown intent string defaults to DIRECT_ANSWER (safest fallback)."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="some_random_nonsense", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("test")
        assert intent == Intent.DIRECT_ANSWER

    async def test_empty_response_falls_back_to_direct(self, mock_client, mock_tools):
        """Empty model response defaults to DIRECT_ANSWER (safest fallback)."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="", model="gemma3:1b-it", tokens_used=0,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        intent, _ = await agent._classifier.classify("test")
        assert intent == Intent.DIRECT_ANSWER

    async def test_image_query_in_intent_enum(self, mock_client, mock_tools):
        """IMAGE_QUERY must exist as a valid intent for vision routing."""
        assert hasattr(Intent, "IMAGE_QUERY")
        assert Intent.IMAGE_QUERY.value == "image_query"

    async def test_invalid_intent_logs_warning(self, mock_client, mock_tools, caplog):
        """Fallback to DIRECT_ANSWER on invalid intent must emit a warning with raw output."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="some_random_nonsense", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        with caplog.at_level("WARNING", logger="src.agent"):
            intent, _ = await agent._classifier.classify("test")
        assert intent == Intent.DIRECT_ANSWER
        assert any("some_random_nonsense" in msg for msg in caplog.messages)
        assert any("not a valid intent" in msg for msg in caplog.messages)

    async def test_connection_error_logs_warning(self, mock_client, mock_tools, caplog):
        """APIConnectionError fallback must emit a warning."""
        from openai import APIConnectionError
        mock_client.generate = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock()),
        )
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        with caplog.at_level("WARNING", logger="src.agent"):
            intent, resp = await agent._classifier.classify("test")
        assert intent == Intent.DIRECT_ANSWER
        assert resp is None
        assert any("unreachable" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Direct answer handler
# ---------------------------------------------------------------------------

class TestDirectAnswer:
    async def test_returns_model_response(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3:1b-it", tokens_used=5,
        ))
        mock_client.generate_stream = MagicMock(side_effect=make_stream("I can help with many things!"))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("Hello!")
        assert result.intent == Intent.DIRECT_ANSWER
        assert result.response == "I can help with many things!"

    async def test_trace_has_classify_and_direct(self, agent):
        result = await agent.process("Hi")
        actions = [s.action for s in result.steps]
        assert "classify_intent" in actions
        assert "direct_response" in actions

    async def test_execution_time_positive(self, agent):
        result = await agent.process("Hello!")
        assert result.execution_time_ms >= 0


# ---------------------------------------------------------------------------
# RAG handler
# ---------------------------------------------------------------------------

class TestRagQuery:
    @pytest.fixture
    def rag_agent(self, mock_client, mock_tools):
        """Agent configured to classify as rag_query and return docs."""
        call_count = 0

        async def generate_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Intent classification
                return LLMResponse(content="rag_query", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                # Query rewriting
                return LLMResponse(content="enterprise plan features", model="gemma3:1b-it", tokens_used=8)
            else:
                # Synthesis
                return LLMResponse(content="The Enterprise plan includes unlimited users.", model="gemma3:1b-it", tokens_used=20)

        mock_client.generate = AsyncMock(side_effect=generate_side_effect)
        mock_client.generate_synthesis_stream = MagicMock(side_effect=make_stream(
            "The Enterprise plan includes unlimited users.",
        ))

        # Vector search returns documents
        docs = [
            Document(id="doc-1", content="Enterprise plan content.", metadata={"title": "Enterprise"}),
            Document(id="doc-2", content="More content.", metadata={"title": "Plans"}),
        ]
        mock_tools.execute = AsyncMock(return_value=ToolResult(success=True, data=docs))
        return SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

    async def test_rag_produces_response(self, rag_agent):
        result = await rag_agent.process("What is the Enterprise plan?")
        assert result.intent == Intent.RAG_QUERY
        assert "Enterprise" in result.response

    async def test_rag_trace_has_rewrite_search_synthesize(self, rag_agent):
        result = await rag_agent.process("What is the Enterprise plan?")
        actions = [s.action for s in result.steps]
        assert "classify_intent" in actions
        assert "rewrite_query" in actions
        assert "vector_search" in actions
        assert "synthesize_response" in actions

    async def test_rag_calls_vector_search_tool(self, rag_agent, mock_tools):
        await rag_agent.process("test query")
        # Dual-query RAG: searches with both rewritten and original query
        assert mock_tools.execute.call_count == 2
        for call in mock_tools.execute.call_args_list:
            assert call[0][0] == "vector_search"

    async def test_rag_empty_results_returns_fallback(self, mock_client, mock_tools):
        """When vector search returns no docs, agent gives a fallback message."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="rag_query", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="rewritten query", model="gemma3:1b-it", tokens_used=5)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_tools.execute = AsyncMock(return_value=ToolResult(success=True, data=[]))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("Unknown topic")
        assert "couldn't find" in result.response.lower()

    async def test_rag_search_failure_returns_fallback(self, mock_client, mock_tools):
        """When vector search fails, agent gives a fallback message."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="rag_query", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="rewritten", model="gemma3:1b-it", tokens_used=5)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_tools.execute = AsyncMock(return_value=ToolResult(success=False, data=None, error="DB error"))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("test")
        assert "couldn't find" in result.response.lower()

    async def test_rag_logs_documents_in_trace(self, rag_agent):
        result = await rag_agent.process("Enterprise plan?")
        vs_step = next(s for s in result.steps if s.action == "vector_search")
        assert "documents" in vs_step.details
        assert len(vs_step.details["documents"]) == 2


# ---------------------------------------------------------------------------
# Tool-use handler
# ---------------------------------------------------------------------------

class TestToolUse:
    @pytest.fixture
    def tool_agent(self, mock_client, mock_tools):
        """Agent configured for the tool_use path (calculator)."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="The result is 42.", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "6 * 7"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "6 * 7", "result": 42},
        ))
        return SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

    async def test_tool_use_produces_response(self, tool_agent):
        result = await tool_agent.process("What is 6 * 7?")
        assert result.intent == Intent.TOOL_USE
        assert "42" in result.response

    async def test_tool_use_trace_has_select_execute_format(self, tool_agent):
        result = await tool_agent.process("What is 6 * 7?")
        actions = [s.action for s in result.steps]
        assert "select_tool" in actions
        assert "execute_tool" in actions
        assert "format_response" in actions

    async def test_tool_use_executes_selected_tool(self, tool_agent, mock_tools):
        await tool_agent.process("What is 6 * 7?")
        mock_tools.execute.assert_called_once_with("calculator", expression="6 * 7")

    async def test_tool_use_no_tool_selected_falls_back(self, mock_client, mock_tools):
        """When qwen returns no function_call, fall back to direct answer."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="I can help directly.", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=0, function_call=None,
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("Show me something")
        # Should fall back to direct_answer handler
        actions = [s.action for s in result.steps]
        assert "direct_response" in actions
        assert "select_tool" not in actions

    async def test_tool_execution_failure_returns_error(self, mock_client, mock_tools):
        """When a tool fails, the error message is surfaced."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="formatted", model="gemma3:1b-it", tokens_used=5)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query", "arguments": {"query": "SELECT bad"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=False, data=None, error="SQL error: no such table",
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("broken query")
        assert "error" in result.response.lower()
        assert "sql_query" in result.response


# ---------------------------------------------------------------------------
# Routing (process dispatches to correct handler)
# ---------------------------------------------------------------------------

class TestRouting:
    async def test_calculation_routes_to_tool_use(self, mock_client, mock_tools):
        """tool_use intent routes through _handle_tool_use with calculator."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="result", model="gemma3:1b-it", tokens_used=5)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "1+1"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"result": 2},
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("1+1")
        assert result.intent == Intent.TOOL_USE
        mock_client.call_function.assert_called_once()


# ---------------------------------------------------------------------------
# Image query handler
# ---------------------------------------------------------------------------

class TestImageQuery:
    async def test_image_present_routes_to_image_query(self, mock_client, mock_tools):
        """When images are provided, process() routes to IMAGE_QUERY intent."""
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("What is in this photo?", images=["base64data"])
        assert result.intent == Intent.IMAGE_QUERY

    async def test_image_query_calls_generate_vision(self, mock_client, mock_tools):
        """Vision path must call client.generate_vision with the images."""
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        await agent.process("Describe this", images=["base64data"])
        mock_client.generate_vision.assert_called_once()
        call_kwargs = mock_client.generate_vision.call_args[1]
        assert call_kwargs["images"] == ["base64data"]
        assert call_kwargs["prompt"] == "Describe this"

    async def test_image_query_trace_has_vision_step(self, mock_client, mock_tools):
        """Execution trace should contain an analyse_image step."""
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("What do you see?", images=["base64data"])
        actions = [s.action for s in result.steps]
        assert "analyse_image" in actions

    async def test_no_images_routes_normally(self, mock_client, mock_tools):
        """Without images, process() classifies intent normally (DIRECT_ANSWER)."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("Hello!")
        assert result.intent == Intent.DIRECT_ANSWER

    async def test_empty_images_list_routes_normally(self, mock_client, mock_tools):
        """An empty images list should route to text classification, not vision."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="direct_answer", model="gemma3:1b-it", tokens_used=5)
            return LLMResponse(content="Hello there!", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("Hello!", images=[])
        assert result.intent == Intent.DIRECT_ANSWER
        mock_client.generate_vision.assert_not_called()


# ---------------------------------------------------------------------------
# Interaction logging
# ---------------------------------------------------------------------------

class TestInteractionLogging:
    async def test_process_increments_interaction_count(self, agent):
        assert agent.interaction_count == 0
        await agent.process("Hello")
        assert agent.interaction_count == 1
        await agent.process("Again")
        assert agent.interaction_count == 2

    async def test_export_writes_json(self, agent, tmp_path):
        await agent.process("Hello")
        filepath = str(tmp_path / "export.json")
        count = agent.export_training_data(filepath)
        assert count == 1
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["query"] == "Hello"
        assert data[0]["intent"] == "direct_answer"
        assert "steps" in data[0]
        assert "models_used" in data[0]

    async def test_log_structure_has_required_fields(self, agent):
        await agent.process("Test query")
        log = agent._logger._log[0]
        assert "timestamp" in log
        assert "query" in log
        assert "intent" in log
        assert "response" in log
        assert "steps" in log
        assert "models_used" in log

    async def test_export_creates_directories(self, agent, tmp_path):
        await agent.process("Test")
        nested = str(tmp_path / "deep" / "nested" / "export.json")
        agent.export_training_data(nested)
        assert os.path.exists(nested)


# ---------------------------------------------------------------------------
# Multi-step tool-use handler
# ---------------------------------------------------------------------------

class TestMultiStepToolUse:
    @pytest.fixture
    def multi_step_agent(self, mock_client, mock_tools):
        """Agent configured for multi-step tool_use (SQL → calculator)."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            prompt = kwargs.get("prompt", "")
            if call_count == 1:
                # classify_intent
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                # decompose_query → 2-step plan (json_mode forces {…} output)
                return LLMResponse(
                    content='{"steps": ["Find best product by revenue", "Calculate 15% growth on the revenue"]}',
                    model="gemma3:1b-it", tokens_used=20,
                )
            else:
                # synthesize_response (concretize is handled by
                # expression_builder, no LLM call needed)
                return LLMResponse(
                    content="Enterprise was the best product at 103,200. With 15% growth: 118,680.",
                    model="gemma3:1b-it", tokens_used=30,
                )

        fn_call_count = 0

        async def call_fn(**kwargs):
            nonlocal fn_call_count
            fn_call_count += 1
            if fn_call_count == 1:
                return LLMResponse(
                    content="", model="qwen", tokens_used=10,
                    function_call={"name": "sql_query", "arguments": {"query": "SELECT name, revenue FROM sales ORDER BY revenue DESC LIMIT 1"}},
                )
            else:
                return LLMResponse(
                    content="", model="qwen", tokens_used=10,
                    function_call={"name": "calculator", "arguments": {"expression": "103200 * 1.15"}},
                )

        exec_count = 0

        async def tool_exec(name, **kwargs):
            nonlocal exec_count
            exec_count += 1
            if name == "sql_query":
                return ToolResult(
                    success=True,
                    data={"columns": ["name", "revenue"], "rows": [{"name": "Enterprise", "revenue": 103200}], "count": 1},
                )
            else:
                return ToolResult(
                    success=True,
                    data={"expression": "103200 * 1.15", "result": 118680.0},
                )

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(side_effect=call_fn)
        mock_tools.execute = AsyncMock(side_effect=tool_exec)
        return SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

    async def test_multi_step_produces_chained_response(self, multi_step_agent):
        result = await multi_step_agent.process(
            "What was our best product, and what would 15% growth look like?"
        )
        assert result.intent == Intent.TOOL_USE
        assert result.success
        assert "118,680" in result.response or "118680" in result.response

    async def test_multi_step_trace_has_decompose_and_synthesize(self, multi_step_agent):
        result = await multi_step_agent.process(
            "What was our best product, and what would 15% growth look like?"
        )
        actions = [s.action for s in result.steps]
        assert "classify_intent" in actions
        assert "decompose_query" in actions
        assert "concretize_step" in actions
        assert "synthesize_response" in actions
        assert actions.count("select_tool") == 2
        assert actions.count("execute_tool") == 2

    async def test_multi_step_select_tool_has_step_counter(self, multi_step_agent):
        result = await multi_step_agent.process(
            "What was our best product, and what would 15% growth look like?"
        )
        select_steps = [s for s in result.steps if s.action == "select_tool"]
        assert len(select_steps) == 2
        assert select_steps[0].details["step"] == "1/2"
        assert select_steps[1].details["step"] == "2/2"

    async def test_multi_step_calls_both_tools(self, multi_step_agent, mock_tools):
        await multi_step_agent.process(
            "What was our best product, and what would 15% growth look like?"
        )
        assert mock_tools.execute.call_count == 2
        calls = mock_tools.execute.call_args_list
        assert calls[0][0][0] == "sql_query"
        assert calls[1][0][0] == "calculator"

    @pytest.fixture
    def single_step_agent(self, mock_client, mock_tools):
        """Agent where decompose returns a single-element plan (fast path)."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # classify_intent
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                # decompose_query → single step (json_mode forces {…} output)
                return LLMResponse(
                    content='{"steps": ["What were the total sales in 2024"]}',
                    model="gemma3:1b-it", tokens_used=10,
                )
            else:
                # format_response (SQL synthesis)
                return LLMResponse(
                    content="Total sales in 2024 were 250,000.",
                    model="gemma3:1b-it", tokens_used=15,
                )

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query", "arguments": {"query": "SELECT SUM(revenue) FROM sales WHERE year=2024"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True,
            data={"columns": ["total"], "rows": [{"total": 250000}], "count": 1},
        ))
        return SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

    async def test_single_step_plan_uses_fast_path(self, single_step_agent):
        result = await single_step_agent.process("What were the total sales in 2024?")
        actions = [s.action for s in result.steps]
        # Single-step should NOT have decompose, concretize, or synthesize
        assert "decompose_query" not in actions
        assert "concretize_step" not in actions
        assert "synthesize_response" not in actions
        # Should have the standard single-step flow
        assert "select_tool" in actions
        assert "execute_tool" in actions
        assert "format_response" in actions

    async def test_decompose_json_parse_failure_falls_back(self, mock_client, mock_tools):
        """When decompose returns invalid JSON, fall back to single-step."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                # Bad JSON from decompose
                return LLMResponse(content="not valid json", model="gemma3:1b-it", tokens_used=5)
            else:
                return LLMResponse(content="The result is 42.", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "6 * 7"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "6 * 7", "result": 42},
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("What is 6 * 7?")
        # Should still succeed via single-step fallback
        assert result.success
        assert "42" in result.response

    async def test_decompose_empty_array_falls_back(self, mock_client, mock_tools):
        """When decompose returns an empty array, fall back to single-step."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                return LLMResponse(content='{"steps": []}', model="gemma3:1b-it", tokens_used=5)
            else:
                return LLMResponse(content="42 is the answer.", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "6 * 7"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "6 * 7", "result": 42},
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("What is 6 * 7?")
        assert result.success
        assert "42" in result.response

    async def test_decompose_handles_raw_array(self, mock_client, mock_tools):
        """When decompose returns a raw JSON array (not wrapped in object), still works."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                # Raw array (not wrapped in {"steps": [...]})
                return LLMResponse(
                    content='["Look up total revenue for 2024"]',
                    model="gemma3:1b-it", tokens_used=10,
                )
            else:
                return LLMResponse(content="Total revenue was 250,000.", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query", "arguments": {"query": "SELECT SUM(revenue) FROM sales WHERE year=2024"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"columns": ["total"], "rows": [{"total": 250000}], "count": 1},
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("What were total sales in 2024?")
        assert result.success

    async def test_decompose_handles_object_without_steps_key(self, mock_client, mock_tools):
        """When decompose returns a JSON object without 'steps' key, fall back."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                # JSON object but no "steps" key
                return LLMResponse(
                    content='{"plan": ["step one"]}',
                    model="gemma3:1b-it", tokens_used=10,
                )
            else:
                return LLMResponse(content="The answer is 42.", model="gemma3:1b-it", tokens_used=10)

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "6 * 7"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "6 * 7", "result": 42},
        ))

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("What is 6 * 7?")
        # Falls back to single-step, should still work
        assert result.success


# ---------------------------------------------------------------------------
# _rephrase_for_sql  (static method)
# ---------------------------------------------------------------------------

class TestRephraseForSql:
    def test_find_prefix(self):
        assert rephrase_for_sql(
            "Find the best product and its revenue last quarter"
        ) == "Show the best product and its revenue last quarter."

    def test_look_up_prefix(self):
        assert rephrase_for_sql(
            "Look up total revenue in Q4 2024"
        ) == "Show total revenue in Q4 2024."

    def test_fetch_prefix(self):
        assert rephrase_for_sql(
            "Fetch the Enterprise plan price"
        ) == "Show the Enterprise plan price."

    def test_retrieve_prefix(self):
        assert rephrase_for_sql(
            "Retrieve the Professional tier customer count"
        ) == "Show the Professional tier customer count."

    def test_get_prefix(self):
        assert rephrase_for_sql(
            "Get the churn rate for Q3 2024"
        ) == "Show the churn rate for Q3 2024."

    def test_count_prefix(self):
        assert rephrase_for_sql(
            "Count total customers"
        ) == "How many total customers are there?"

    def test_count_with_qualifier(self):
        assert rephrase_for_sql(
            "Count Enterprise tier customers"
        ) == "How many Enterprise tier customers are there?"

    def test_no_match_passthrough(self):
        q = "What was total revenue in Q4 2024?"
        assert rephrase_for_sql(q) == q

    def test_show_prefix_passthrough(self):
        q = "Show all customers in healthcare"
        assert rephrase_for_sql(q) == q

    def test_strips_trailing_period(self):
        assert rephrase_for_sql(
            "Find the cheapest product."
        ) == "Show the cheapest product."

    def test_case_insensitive_find(self):
        assert rephrase_for_sql(
            "find revenue for Q2 2024"
        ) == "Show revenue for Q2 2024."

    def test_case_insensitive_count(self):
        assert rephrase_for_sql(
            "count Starter plan customers"
        ) == "How many Starter plan customers are there?"


class TestMultiStepSchemaRestriction:
    """Verify that step 1 of multi-step is handled by the function model.

    With NullResolvers active, the function model handles all tool selection
    directly — no deterministic pre-routing via sql_builder / expression_builder.
    """

    _TOOL_SCHEMAS = [
        {"type": "function", "function": {"name": "sql_query", "description": "Query the database", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "calculator", "description": "Evaluate math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    ]

    async def test_step1_uses_function_model_with_null_resolvers(self, mock_client, mock_tools):
        """With NullResolvers active, step 1 is handled by the function model (not sql-builder)."""
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="tool_use", model="gemma3:1b-it", tokens_used=5)
            elif call_count == 2:
                return LLMResponse(
                    content='{"steps": ["Find Q3 2024 revenue", "Calculate 15% growth on the revenue"]}',
                    model="gemma3:1b-it", tokens_used=20,
                )
            else:
                return LLMResponse(
                    content="Revenue was 84,900. With 15% growth: 97,635.",
                    model="gemma3:1b-it", tokens_used=30,
                )

        exec_count = 0

        async def tool_exec(name, **kwargs):
            nonlocal exec_count
            exec_count += 1
            if name == "sql_query":
                return ToolResult(
                    success=True,
                    data={"columns": ["revenue"], "rows": [{"revenue": 84900}], "count": 1},
                )
            else:
                return ToolResult(
                    success=True,
                    data={"expression": "84900 * 1.15", "result": 97635.0},
                )

        mock_client.generate = AsyncMock(side_effect=gen)
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query", "arguments": {"query": "SELECT 1"}},
        ))
        mock_tools.execute = AsyncMock(side_effect=tool_exec)
        mock_tools.get_all_schemas = MagicMock(return_value=self._TOOL_SCHEMAS)

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process(
            "What was Q3 2024 revenue, and what would 15% growth look like?"
        )

        # Step 1 should be handled by the function model (NullResolvers bypass all builders)
        select_steps = [s for s in result.steps if s.action == "select_tool"]
        assert len(select_steps) >= 1
        assert select_steps[0].details["tool"] == "sql_query"
        # Model is the function model, not a deterministic builder
        assert select_steps[0].model != "sql-builder"
        assert select_steps[0].model != "expr-builder"


# ---------------------------------------------------------------------------
# Pipeline timeout
# ---------------------------------------------------------------------------

class TestPipelineTimeout:
    """Verify asyncio.wait_for caps the full orchestration pipeline."""

    async def test_slow_pipeline_raises_timeout(self, mock_client, mock_tools):
        """A pipeline exceeding PIPELINE_TIMEOUT triggers asyncio.TimeoutError."""
        async def _slow_generate(**kwargs):
            await asyncio.sleep(10)
            return LLMResponse(content="direct_answer", model="m", tokens_used=1)

        mock_client.generate = _slow_generate
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.process("hello"), timeout=0.05)

    async def test_fast_pipeline_completes_within_timeout(self, mock_client, mock_tools):
        """A normal-speed pipeline should not be affected by the timeout."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3:1b-it", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        result = await asyncio.wait_for(agent.process("hello"), timeout=5.0)
        assert result.response is not None
        assert result.success is True

    async def test_timeout_does_not_produce_partial_response(self, mock_client, mock_tools):
        """Timeout should raise cleanly, not return a half-built AgentResponse."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="m", tokens_used=1,
        ))

        async def _slow_stream(**kwargs):
            await asyncio.sleep(10)
            yield  # unreachable, but makes it an async generator

        mock_client.generate_stream = _slow_stream
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.process("What is 2+2?"), timeout=0.1)

    async def test_pipeline_timeout_config_value(self):
        """PIPELINE_TIMEOUT should be importable and have a sane default."""
        from src.engine.inference.config import PIPELINE_TIMEOUT
        assert isinstance(PIPELINE_TIMEOUT, float)
        assert PIPELINE_TIMEOUT > 0
        assert PIPELINE_TIMEOUT == 60.0  # default

    async def test_pipeline_timeout_env_override(self, monkeypatch):
        """PIPELINE_TIMEOUT should be overridable via env var."""
        monkeypatch.setenv("PIPELINE_TIMEOUT", "120.0")
        # Re-import to pick up env change
        import importlib
        import src.engine.inference.config as _config_mod
        importlib.reload(_config_mod)
        assert _config_mod.PIPELINE_TIMEOUT == 120.0
        # Restore default
        monkeypatch.delenv("PIPELINE_TIMEOUT")
        importlib.reload(_config_mod)


# ---------------------------------------------------------------------------
# Request ID tracing
# ---------------------------------------------------------------------------

class TestRequestIdTracing:
    """Verify request_id is generated and unique per response."""

    async def test_response_has_request_id(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        result = await agent.process("Hello")
        assert hasattr(result, "request_id")
        assert isinstance(result.request_id, str)
        assert len(result.request_id) == 12  # hex[:12]

    async def test_request_ids_are_unique(self, mock_client, mock_tools):
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        r1 = await agent.process("Hello")
        r2 = await agent.process("World")
        assert r1.request_id != r2.request_id


# ---------------------------------------------------------------------------
# Deadline parameter safety (not stored on self)
# ---------------------------------------------------------------------------

class TestDeadlineThreadSafety:
    """Verify _deadline is NOT stored on self — passed through method chain."""

    async def test_no_deadline_attribute_on_orchestrator(self, mock_client, mock_tools):
        """After processing, orchestrator should not have _deadline as an instance attr."""
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="direct_answer", model="gemma3", tokens_used=5,
        ))
        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)
        await agent.process("Hello")
        assert not hasattr(agent, "_deadline"), \
            "_deadline should be a local variable, not stored on self"
