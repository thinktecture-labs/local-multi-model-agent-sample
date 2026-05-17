"""
Unit tests for ToolUseHandler — direct tests of all code paths.

Tests the handler in isolation (not through the orchestrator), covering:
- _resolve_tool_arguments (calculator expression patching only)
- Single-step happy paths (calculator, sql_query) and edge cases
- Multi-step happy path with Qwen-routed concretize_step
- Multi-step deadline exhaustion
- Multi-step with no tool selected
- _tool_schemas filtering
"""

import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.engine.agent.handlers.tool_use import ToolUseHandler
from tests.conftest import make_stream
from src.engine.agent.types import ExecutionStep
from src.engine.agent.query_decomposer import QueryDecomposer
from src.engine.inference.client import SmallLanguageModelRole, LLMResponse
from src.engine.tools.tool_registry import ToolRegistry
from src.engine.tools.tool_result import ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.models = {
        SmallLanguageModelRole.INFERENCE: "gemma3:1b-it",
        SmallLanguageModelRole.FUNCTION: "qwen",
        SmallLanguageModelRole.EMBEDDING: "embeddinggemma",
        SmallLanguageModelRole.VISION: "gemma3-4b-vision",
    }
    client.generate = AsyncMock(return_value=LLMResponse(
        content="formatted answer", model="gemma3:1b-it", tokens_used=10,
    ))
    client.generate_stream = MagicMock(side_effect=make_stream("formatted answer"))
    client.call_function = AsyncMock(return_value=LLMResponse(
        content="", model="qwen", tokens_used=10,
        function_call={"name": "calculator", "arguments": {"expression": "1+1"}},
    ))
    return client


@pytest.fixture
def mock_tools():
    tools = MagicMock(spec=ToolRegistry)
    tools.execute = AsyncMock(return_value=ToolResult(
        success=True, data={"expression": "1+1", "result": 2},
    ))
    tools.get_all_schemas = MagicMock(return_value=[
        {"function": {"name": "vector_search", "parameters": {}}},
        {"function": {"name": "sql_query", "parameters": {}}},
        {"function": {"name": "calculator", "parameters": {}}},
    ])
    return tools


@pytest.fixture
def mock_decomposer():
    decomposer = MagicMock(spec=QueryDecomposer)
    # Default: single-step (1-item plan)
    decomposer.decompose = AsyncMock(return_value=(
        ["single step"],
        ExecutionStep(action="decompose", model="gemma3:1b-it", details={}, duration_ms=0.0),
    ))
    return decomposer


@pytest.fixture
def handler(mock_client, mock_tools, mock_decomposer):
    return ToolUseHandler(
        client=mock_client,
        tools=mock_tools,
        decomposer=mock_decomposer,
    )


# ---------------------------------------------------------------------------
# _tool_schemas property
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_excludes_vector_search(self, handler):
        schemas = handler._tool_schemas
        names = [s["function"]["name"] for s in schemas]
        assert "vector_search" not in names
        assert "sql_query" in names
        assert "calculator" in names

    def test_returns_fresh_list_each_call(self, handler):
        """Property returns a new list (no stale cache issues)."""
        a = handler._tool_schemas
        b = handler._tool_schemas
        assert a == b
        assert a is not b


# ---------------------------------------------------------------------------
# _resolve_tool_arguments — only patches incomplete calculator expressions
# ---------------------------------------------------------------------------

class TestResolveToolArguments:
    def test_calculator_bare_percent_reconstructs_from_normalized(self, handler):
        """patch_calculator_expression rebuilds '15%' → '15% of 100'."""
        name, args = handler._resolve_tool_arguments(
            "calculator", {"expression": "15%"}, "15% of 100",
        )
        assert name == "calculator"
        assert args["expression"] == "15% of 100"

    def test_calculator_complete_expression_unchanged(self, handler):
        """Calculator expressions that don't match the bare-percent pattern pass through."""
        name, args = handler._resolve_tool_arguments(
            "calculator", {"expression": "6 * 7"}, "What is 6 * 7?",
        )
        assert name == "calculator"
        assert args["expression"] == "6 * 7"

    def test_sql_query_unchanged(self, handler):
        """SQL queries are never rewritten by the handler — Qwen FT handles them."""
        original_query = "SELECT * FROM products"
        name, args = handler._resolve_tool_arguments(
            "sql_query", {"query": original_query}, "Show products",
        )
        assert name == "sql_query"
        assert args["query"] == original_query

    def test_unknown_tool_passes_through(self, handler):
        """Non-calculator tools are returned unchanged."""
        name, args = handler._resolve_tool_arguments(
            "web_search", {"url": "http://example.com"}, "search",
        )
        assert name == "web_search"
        assert args == {"url": "http://example.com"}


# ---------------------------------------------------------------------------
# Single-step path
# ---------------------------------------------------------------------------

class TestSingleStep:
    async def test_calculator_happy_path(self, handler, mock_client, mock_tools):
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "6 * 7"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "6 * 7", "result": 42},
        ))

        text, steps = await handler.handle("What is 6 * 7?", deadline=time.perf_counter() + 60)
        assert "42" in text
        actions = [s.action for s in steps]
        assert actions == ["select_tool", "execute_tool", "format_response"]

    async def test_sql_query_happy_path(self, handler, mock_client, mock_tools):
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query", "arguments": {"query": "SELECT COUNT(*) FROM sales"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"columns": ["count"], "rows": [[42]]},
        ))

        text, steps = await handler.handle("How many sales?", deadline=time.perf_counter() + 60)
        assert isinstance(text, str)
        # Non-calculator results use streaming LLM formatting
        mock_client.generate_stream.assert_called_once()

    async def test_no_tool_selected_falls_back_to_direct(
        self, handler, mock_client,
    ):
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=0, function_call=None,
        ))
        mock_client.generate_stream = MagicMock(side_effect=make_stream("I can help directly."))

        text, steps = await handler.handle("What is the meaning of life?", deadline=time.perf_counter() + 60)
        assert text == "I can help directly."
        actions = [s.action for s in steps]
        assert "direct_response" in actions

    async def test_tool_execution_failure(self, handler, mock_client, mock_tools):
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query", "arguments": {"query": "SELECT bad"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=False, data=None, error="no such table: bad",
        ))

        text, steps = await handler.handle("bad query", deadline=time.perf_counter() + 60)
        assert "error" in text.lower()
        assert "sql_query" in text

    async def test_calculator_formats_locally(self, handler, mock_client, mock_tools):
        """Calculator results use local formatting, not LLM."""
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "1000 + 2000"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "1000 + 2000", "result": 3000},
        ))

        text, steps = await handler.handle("1000 + 2000", deadline=time.perf_counter() + 60)
        assert text == "1000 + 2000 = 3,000"
        # LLM generate should NOT have been called for formatting
        mock_client.generate.assert_not_called()

    async def test_calculator_float_to_int_display(self, handler, mock_client, mock_tools):
        """Float results that are whole numbers display as ints."""
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "10 / 2"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "10 / 2", "result": 5.0},
        ))

        text, steps = await handler.handle("10 / 2", deadline=time.perf_counter() + 60)
        assert text == "10 / 2 = 5"  # Not "5.0"


# ---------------------------------------------------------------------------
# Multi-step path
# ---------------------------------------------------------------------------

class TestMultiStep:
    @pytest.fixture
    def multi_decomposer(self, mock_decomposer):
        """Decomposer that returns a 2-step plan with Qwen-routed concretize."""
        mock_decomposer.decompose = AsyncMock(return_value=(
            ["Find the price of DataFlow Pro", "Calculate 15% of the price"],
            ExecutionStep(action="decompose", model="gemma3:1b-it",
                         details={"plan": ["step1", "step2"]}, duration_ms=5.0),
        ))
        mock_decomposer.concretize_step = AsyncMock(return_value=(
            "Calculate 15% of 899",
            ExecutionStep(action="concretize_step", model="qwen",
                         details={}, duration_ms=1.0),
        ))
        return mock_decomposer

    async def test_multi_step_happy_path(
        self, handler, mock_client, mock_tools, multi_decomposer,
    ):
        """Two-step: SQL fetch → Qwen concretize → calculator compute → synthesis."""
        handler._decomposer = multi_decomposer

        call_count = 0

        async def mock_call_function(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="", model="qwen", tokens_used=10,
                    function_call={"name": "sql_query",
                                   "arguments": {"query": "SELECT price FROM products"}},
                )
            return LLMResponse(
                content="", model="qwen", tokens_used=10,
                function_call={"name": "calculator",
                               "arguments": {"expression": "899 * 0.15"}},
            )

        mock_client.call_function = AsyncMock(side_effect=mock_call_function)

        exec_count = 0

        async def mock_execute(name, **kwargs):
            nonlocal exec_count
            exec_count += 1
            if name == "sql_query":
                return ToolResult(success=True, data={"columns": ["price"], "rows": [[899]]})
            return ToolResult(success=True, data={"expression": "899 * 0.15", "result": 134.85})

        mock_tools.execute = AsyncMock(side_effect=mock_execute)
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="The 15% discount on DataFlow Pro is $134.85.",
            model="gemma3:1b-it", tokens_used=20,
        ))

        text, steps = await handler.handle(
            "What is 15% of the DataFlow Pro price?",
            deadline=time.perf_counter() + 60,
        )
        assert "134.85" in text
        actions = [s.action for s in steps]
        assert "decompose" in actions
        assert "concretize_step" in actions
        assert "synthesize_response" in actions
        assert actions.count("execute_tool") == 2
        # Both tool calls go through Qwen (no more deterministic pre-routers)
        assert mock_client.call_function.call_count == 2

    async def test_multi_step_deadline_exhaustion(
        self, handler, mock_client, mock_tools, multi_decomposer,
    ):
        """When deadline is nearly expired, skips remaining steps and synthesizes."""
        handler._decomposer = multi_decomposer

        # Give only 3 seconds — less than the 5s threshold
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "sql_query",
                           "arguments": {"query": "SELECT 1"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"result": 1},
        ))
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="partial result", model="gemma3:1b-it", tokens_used=10,
        ))

        text, steps = await handler.handle(
            "complex multi-step query",
            deadline=time.perf_counter() + 3,
        )
        # Should still produce a response (synthesis of whatever was gathered)
        assert isinstance(text, str)

    async def test_multi_step_no_results(
        self, handler, mock_client, mock_tools, multi_decomposer,
    ):
        """When no tool returns results, return fallback message."""
        handler._decomposer = multi_decomposer

        # All function calls return no tool selection
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=0, function_call=None,
        ))

        text, steps = await handler.handle(
            "impossible query",
            deadline=time.perf_counter() + 60,
        )
        assert text == "No tool results were obtained."


# ---------------------------------------------------------------------------
# handle() routing
# ---------------------------------------------------------------------------

class TestHandleRouting:
    async def test_single_step_plan_routes_to_single(
        self, handler, mock_client, mock_tools, mock_decomposer,
    ):
        """1-item plan routes to _handle_single_step."""
        mock_decomposer.decompose = AsyncMock(return_value=(
            ["one step"],
            ExecutionStep(action="decompose", model="m", details={}, duration_ms=0),
        ))
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "1+1"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"expression": "1+1", "result": 2},
        ))

        text, steps = await handler.handle("1+1", deadline=time.perf_counter() + 60)
        # Single-step path does NOT include a "decompose" step in its output
        actions = [s.action for s in steps]
        assert "decompose" not in actions
        assert "select_tool" in actions

    async def test_multi_step_plan_routes_to_multi(
        self, handler, mock_client, mock_tools, mock_decomposer,
    ):
        """2-item plan routes to _handle_multi_step."""
        mock_decomposer.decompose = AsyncMock(return_value=(
            ["step one", "step two"],
            ExecutionStep(action="decompose", model="m", details={}, duration_ms=0),
        ))
        mock_decomposer.concretize_step = AsyncMock(return_value=(
            "step two concretized",
            ExecutionStep(action="concretize_step", model="m", details={}, duration_ms=0),
        ))
        mock_client.call_function = AsyncMock(return_value=LLMResponse(
            content="", model="qwen", tokens_used=10,
            function_call={"name": "calculator", "arguments": {"expression": "1"}},
        ))
        mock_tools.execute = AsyncMock(return_value=ToolResult(
            success=True, data={"result": 1},
        ))
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="synthesized", model="gemma3:1b-it", tokens_used=10,
        ))

        text, steps = await handler.handle("multi", deadline=time.perf_counter() + 60)
        actions = [s.action for s in steps]
        assert "decompose" in actions
        assert "synthesize_response" in actions
