"""ToolRegistry — central store that manages all available agent tools."""

import inspect
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from typing import TYPE_CHECKING

from .base_tool import BaseTool
from .tool_result import ToolResult

if TYPE_CHECKING:
    from ..knowledge.vector_store import VectorStore


def create_default_registry(
    *,
    vector_store: "Optional[VectorStore]" = None,
    upload_store: "Optional[VectorStore]" = None,
    db_path: str | None = None,
) -> "ToolRegistry":
    """Create a ToolRegistry with the standard agent tools.

    Registration order matters: small tool-calling models have a positional bias
    toward earlier tools for ambiguous queries. sql_query is registered
    before calculator so data-lookup queries win over arithmetic by default.

    When upload_store is provided, VectorSearchTool searches both the curated
    knowledge base and user-uploaded documents, merging results by score.
    """
    from .calculator import CalculatorTool
    from .sql_query import SQLQueryTool
    from ..inference.config import DB_PATH

    if db_path is None:
        db_path = DB_PATH

    registry = ToolRegistry()
    if vector_store is not None:
        from .vector_search import VectorSearchTool
        registry.register(VectorSearchTool(
            vector_store=vector_store,
            upload_store=upload_store,
        ))
    registry.register(SQLQueryTool(db_path=db_path))
    registry.register(CalculatorTool())
    return registry


class ToolRegistry:
    """
    Central registry that manages all available tools.

    Responsibilities:
    - Store and retrieve tools by name
    - Generate OpenAI-compatible schemas for the tool-calling model
    - Execute tools safely, catching unexpected exceptions
    - Log every execution for fine-tuning data collection

    The execution log is a gold mine: real tool-call pairs from live traffic
    can directly improve the tool-calling model's accuracy after fine-tuning.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._execution_log: list[dict] = []
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Overwrites any existing tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Schema generation  (fed to the tool-calling model)
    # ------------------------------------------------------------------

    def get_all_schemas(self) -> list[dict]:
        """
        Return OpenAI-compatible function schemas for all registered tools.

        This list is passed to the tool-calling model so it understands what actions
        are available and can select the right one.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name, with error handling and logging.

        Always returns a ToolResult — never raises. Failed tool calls are
        surfaced through ToolResult.success=False and ToolResult.error.
        """
        tool = self.get(tool_name)
        if not tool:
            with self._log_lock:
                self._execution_log.append({
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "tool":        tool_name,
                    "parameters":  kwargs,
                    "success":     False,
                    "duration_ms": 0.0,
                    "error":       f"Unknown tool: '{tool_name}'",
                })
            return ToolResult(
                success=False,
                data=None,
                error=(
                    f"Unknown tool: '{tool_name}'. "
                    f"Available tools: {self.list_tools()}"
                ),
            )

        # Filter kwargs to only parameters the tool's execute() accepts.
        # LLMs (especially small ones) sometimes hallucinate extra arguments
        # like "params" that would cause a TypeError.
        sig = inspect.signature(tool.execute)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            filtered = kwargs  # tool accepts **kwargs, pass everything
        else:
            accepted = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in accepted}

        t0 = time.perf_counter()
        try:
            result = await tool.execute(**filtered)
        except Exception as exc:
            result = ToolResult(success=False, data=None, error=str(exc))

        duration_ms = (time.perf_counter() - t0) * 1000

        with self._log_lock:
            self._execution_log.append({
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "tool":        tool_name,
                "parameters":  kwargs,
                "success":     result.success,
                "duration_ms": round(duration_ms, 2),
                "error":       result.error,
            })

        return result

    # ------------------------------------------------------------------
    # Fine-tuning data export
    # ------------------------------------------------------------------

    def export_execution_log(self) -> list[dict]:
        """
        Return a copy of all logged tool executions.

        Use this data to fine-tune the tool-calling model for your specific tool set.
        After a few hundred real interactions, tool selection accuracy can
        improve from ~65% to ~90%.
        """
        with self._log_lock:
            return list(self._execution_log)
