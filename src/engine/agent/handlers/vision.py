"""
Vision handler — image analysis via gemma3-4B.
"""

import time

from ..types import ExecutionStep
from ...inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from ...inference.config import VISION_MAX_TOKENS, VISION_TEMPERATURE


class VisionHandler:
    """
    Analyse images using the vision model (gemma3-4B + mmproj).

    Single-step flow: send image(s) + user prompt to generate_vision().
    No classification needed — the presence of images is deterministic routing.
    """

    def __init__(self, client: SmallLanguageModelClient) -> None:
        self._client = client

    async def handle(
        self, query: str, *, images: list[str], **kwargs,
    ) -> tuple[str, list[ExecutionStep]]:
        t0 = time.perf_counter()
        response = await self._client.generate_vision(
            prompt=query,
            images=images,
            system_prompt=(
                "You are a helpful visual assistant. Describe what you see accurately "
                "and answer the user's question based on the image content."
            ),
            temperature=VISION_TEMPERATURE,
            max_tokens=VISION_MAX_TOKENS,
        )
        return response.content.strip(), [
            ExecutionStep(
                action="analyse_image",
                model=self._client.models[SmallLanguageModelRole.VISION],
                details={
                    "n_images": len(images),
                    "response": response.content.strip(),
                },
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                tokens_used=response.tokens_used,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        ]
