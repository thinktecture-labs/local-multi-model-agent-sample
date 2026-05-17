"""
Tool system — pluggable capabilities for the Gemma agent.

Each tool is a self-contained module. Add new tools by:
  1. Subclassing BaseTool in a new file
  2. Implementing execute() and _get_parameters()
  3. Registering an instance with ToolRegistry
"""

from .tool_result import ToolResult
from .base_tool import BaseTool
from .tool_registry import ToolRegistry, create_default_registry
from .vector_search import VectorSearchTool
from .calculator import CalculatorTool
from .sql_query import SQLQueryTool

__all__ = [
    "ToolResult",
    "BaseTool",
    "ToolRegistry",
    "create_default_registry",
    "VectorSearchTool",
    "CalculatorTool",
    "SQLQueryTool",
]
