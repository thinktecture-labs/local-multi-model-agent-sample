"""Chonkie embeddings adapter for llama-server OpenAI-compatible API.

Bridges chonkie's SemanticChunker to the local embeddinggemma model
running on llama-server. Uses the same embedding endpoint that powers
retrieval, ensuring alignment between chunking and search.

Zero additional dependencies — uses httpx (already in the project)
and tiktoken (installed with chonkie[openai]).
"""

import logging
import os

import httpx
import numpy as np
import tiktoken
from chonkie.embeddings.base import BaseEmbeddings

logger = logging.getLogger(__name__)

# Default ports match .env / state.py conventions
_DEFAULT_BASE_PORT = int(os.getenv("EMBEDDING_PORT", "9092"))
_DEFAULT_FT_PORT = int(os.getenv("EMBEDDING_PORT_FT", "9096"))


class LlamaServerEmbeddings(BaseEmbeddings):
    """Chonkie-compatible embeddings via llama-server OpenAI API.

    Calls the same embeddinggemma model used for vector retrieval,
    ensuring that semantic chunking boundaries align with how the
    retrieval engine "thinks" about similarity.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "embeddinggemma",
        timeout: float = 30.0,
    ):
        super().__init__()
        if base_url is None:
            base_url = f"http://localhost:{_DEFAULT_BASE_PORT}"
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=timeout)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

        # Probe embedding dimension
        try:
            test_vec = self._call_api(["dimension probe"])
            self._dim = len(test_vec[0])
            logger.info(
                "LlamaServerEmbeddings: connected to %s, dimension=%d",
                self._base_url, self._dim,
            )
        except Exception:
            logger.warning(
                "LlamaServerEmbeddings: could not connect to %s — "
                "semantic chunking will be unavailable",
                self._base_url,
            )
            self._dim = 0

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self._base_url}/v1/embeddings",
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [d["embedding"] for d in data]

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return np.array(self._call_api([text])[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts in one API call."""
        vecs = self._call_api(texts)
        return [np.array(v, dtype=np.float32) for v in vecs]

    @property
    def dimension(self) -> int:
        return self._dim

    def get_tokenizer(self):
        """Return tiktoken tokenizer for chonkie's token counting."""
        return self._tokenizer

    @classmethod
    def is_available(cls) -> bool:
        return True

    def __repr__(self) -> str:
        return f"LlamaServerEmbeddings({self._base_url!r}, dim={self._dim})"
