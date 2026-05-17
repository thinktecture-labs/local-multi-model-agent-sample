"""ToolResult — the standardized return type from every tool execution."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolResult:
    """
    Standardized result from any tool execution.

    Always explicit about success/failure — tools must never fail silently.
    The agent uses `success` to decide whether to surface an error or
    pass `data` to the response synthesis step.
    """
    success: bool
    data: Any
    error: Optional[str] = None
