"""
Handler protocol — the contract every intent handler must satisfy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, Union

from ..types import ExecutionStep


class Handler(Protocol):
    """
    Strategy for one intent type — processes a query and returns (text, steps).

    Three concrete implementations:
      - DirectAnswerHandler  — intent: direct_answer
      - RAGHandler           — intent: rag_query
      - ToolUseHandler       — intent: tool_use
    """

    async def handle(
        self, query: str, **kwargs,
    ) -> tuple[str, list[ExecutionStep]]: ...

    def handle_stream(
        self, query: str, **kwargs,
    ) -> AsyncIterator[Union[str, ExecutionStep]]: ...


async def collect_stream(
    stream: AsyncIterator[Union[str, ExecutionStep]],
) -> tuple[str, list[ExecutionStep]]:
    """Collect a handle_stream() into (text, steps). Shared by all handlers."""
    chunks: list[str] = []
    steps: list[ExecutionStep] = []
    async for item in stream:
        if isinstance(item, str):
            chunks.append(item)
        else:
            steps.append(item)
    return "".join(chunks), steps


# Alias: communicates the intent-strategy pattern at call sites in the orchestrator.
IntentStrategy = Handler
