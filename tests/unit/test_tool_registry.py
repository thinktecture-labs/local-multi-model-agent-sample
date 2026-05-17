"""
Unit tests for ToolRegistry.

Verifies registration, schema generation, execution routing,
unknown-tool handling, and execution logging — all without touching
any external services.
"""

import pytest

from src.engine.tools.base_tool import BaseTool
from src.engine.tools.calculator import CalculatorTool
from src.engine.tools.tool_registry import ToolRegistry
from src.engine.tools.tool_result import ToolResult


# ---------------------------------------------------------------------------
# Minimal stub tool for isolation
# ---------------------------------------------------------------------------

class _EchoTool(BaseTool):
    name        = "echo"
    description = "Returns its input unchanged."

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, text: str) -> ToolResult:
        return ToolResult(success=True, data={"echo": text})


class _FailingTool(BaseTool):
    name        = "fail"
    description = "Always raises an exception."

    def _get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Intentional failure")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegistration:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = _EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("no_such_tool") is None

    def test_list_tools_empty(self):
        assert ToolRegistry().list_tools() == []

    def test_list_tools_after_registration(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        registry.register(CalculatorTool())
        names = registry.list_tools()
        assert "echo" in names
        assert "calculator" in names
        assert len(names) == 2

    def test_register_overwrites_existing(self):
        registry = ToolRegistry()
        tool_a = _EchoTool()
        tool_b = _EchoTool()  # same name, different instance
        registry.register(tool_a)
        registry.register(tool_b)
        assert registry.get("echo") is tool_b
        assert len(registry.list_tools()) == 1


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchemaGeneration:
    def test_get_all_schemas_empty(self):
        assert ToolRegistry().get_all_schemas() == []

    def test_schema_structure(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_schema_count_matches_registered(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        registry.register(CalculatorTool())
        assert len(registry.get_all_schemas()) == 2


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecution:
    async def test_execute_known_tool(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        result = await registry.execute("echo", text="hello")
        assert result.success
        assert result.data["echo"] == "hello"

    async def test_execute_unknown_tool_returns_failure(self):
        registry = ToolRegistry()
        result = await registry.execute("unknown_tool")
        assert not result.success
        assert "unknown_tool" in result.error.lower() or "unknown" in result.error.lower()

    async def test_execute_exception_returns_failure(self):
        """Registry must catch unexpected exceptions from tools."""
        registry = ToolRegistry()
        registry.register(_FailingTool())
        result = await registry.execute("fail")
        assert not result.success
        assert result.error is not None

    async def test_calculator_via_registry(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = await registry.execute("calculator", expression="2 + 2")
        assert result.success
        assert result.data["result"] == 4


# ---------------------------------------------------------------------------
# Execution logging
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecutionLogging:
    async def test_successful_execution_logged(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        await registry.execute("echo", text="test")
        log = registry.export_execution_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["tool"] == "echo"
        assert entry["success"] is True
        assert "duration_ms" in entry

    async def test_unknown_tool_execution_logged(self):
        # Unknown tool calls are now logged for fine-tuning data collection.
        registry = ToolRegistry()
        await registry.execute("unknown")
        log = registry.export_execution_log()
        assert len(log) == 1
        assert log[0]["tool"] == "unknown"
        assert log[0]["success"] is False
        assert "Unknown tool" in log[0]["error"]

    async def test_failed_execution_logged_for_known_tool(self):
        # A registered tool that raises an exception should still be logged.
        registry = ToolRegistry()
        registry.register(_FailingTool())
        await registry.execute("fail")
        log = registry.export_execution_log()
        assert len(log) == 1
        assert log[0]["success"] is False

    async def test_multiple_executions_all_logged(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        await registry.execute("echo", text="a")
        await registry.execute("echo", text="b")
        log = registry.export_execution_log()
        assert len(log) == 2

    async def test_export_returns_copy(self):
        """Mutating the exported log must not affect the registry's internal log."""
        registry = ToolRegistry()
        registry.register(_EchoTool())
        await registry.execute("echo", text="x")
        log = registry.export_execution_log()
        log.clear()
        assert len(registry.export_execution_log()) == 1


# ---------------------------------------------------------------------------
# Schema drift guard — training schemas must stay in sync with production
# ---------------------------------------------------------------------------

class TestSchemaDriftGuard:
    """Detect silent drift between the schemas used in eval/training and production.

    finetune/eval_tool_routing.py._build_tool_schemas() defines the schemas
    the fine-tuning eval uses. If someone changes a production tool schema
    (e.g. renames an argument) but forgets to update the eval, the model will
    be trained on stale schemas and silently degrade.

    Finding (H) from the code review: schema drift is invisible without an
    explicit comparison test.
    """

    def test_eval_schemas_match_production(self):
        """Training-time tool schemas must be byte-for-byte identical to production."""
        from src.engine.tools.calculator import CalculatorTool
        from src.engine.tools.sql_query import SQLQueryTool
        from finetune.eval_tool_routing import _build_tool_schemas

        # Production schemas (the exact same source of truth used at inference time)
        prod = sorted(
            [SQLQueryTool().get_schema(), CalculatorTool().get_schema()],
            key=lambda s: s["function"]["name"],
        )

        # Training/eval schemas
        training = sorted(
            _build_tool_schemas(),
            key=lambda s: s["function"]["name"],
        )

        assert prod == training, (
            "Tool schemas used in finetune/eval_tool_routing.py diverge from "
            "production tool definitions. Update _build_tool_schemas() to match "
            "or the model will be evaluated against stale schemas.\n"
            f"Production: {prod}\nTraining:   {training}"
        )
