"""
Direct answer handler — gemma3 responds without tools.
"""

import logging
import time
from collections.abc import AsyncIterator

from ..types import ExecutionStep
from ...inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from ...inference.config import DIRECT_ANSWER_MAX_TOKENS, DIRECT_ANSWER_TEMPERATURE
from ...inference.prompts import DIRECT_ANSWER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DirectAnswerHandler:
    """Answer simple conversational queries directly — no tools needed."""

    def __init__(self, client: SmallLanguageModelClient) -> None:
        self._client = client

    async def handle(
        self, query: str, *, deadline: float = 0, **kwargs,
    ) -> tuple[str, list[ExecutionStep]]:
        from .protocol import collect_stream
        return await collect_stream(self.handle_stream(query, deadline=deadline, **kwargs))

    async def handle_stream(
        self, query: str, *, deadline: float = 0, **kwargs,
    ) -> AsyncIterator[ExecutionStep | str]:
        """Stream direct answer tokens. Yields str tokens then a final ExecutionStep."""
        t0 = time.perf_counter()

        remaining = (deadline - time.perf_counter()) if deadline else None
        if remaining is not None and remaining <= 0:
            yield "The pipeline deadline was exceeded before a response could be generated."
            return

        stream = self._client.generate_stream(
            prompt=query,
            system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
            temperature=DIRECT_ANSWER_TEMPERATURE,
            max_tokens=DIRECT_ANSWER_MAX_TOKENS,
            deterministic=True,
        )

        chunks: list[str] = []
        tokens_used = prompt_tokens = completion_tokens = 0
        async for chunk in stream:
            if chunk.done:
                tokens_used = chunk.tokens_used
                prompt_tokens = chunk.prompt_tokens
                completion_tokens = chunk.completion_tokens
            elif chunk.text:
                chunks.append(chunk.text)
                yield chunk.text

        yield ExecutionStep(
            action="direct_response",
            model=self._client.models[SmallLanguageModelRole.INFERENCE],
            details={"response": "".join(chunks)},
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
