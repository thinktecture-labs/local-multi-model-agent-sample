"""
Intent handlers — each handler implements one branch of the agent pipeline.

The orchestrator dispatches to these handlers based on classified intent.
"""

from .protocol import Handler
from .direct_answer import DirectAnswerHandler
from .rag import RAGHandler
from .tool_use import ToolUseHandler
from .vision import VisionHandler

__all__ = [
    "Handler",
    "DirectAnswerHandler",
    "RAGHandler",
    "ToolUseHandler",
    "VisionHandler",
]
