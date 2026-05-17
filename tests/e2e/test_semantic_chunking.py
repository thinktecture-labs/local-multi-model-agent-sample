"""
E2E tests for semantic chunking with live embeddinggemma server.

Requires llama-server running on the embedding port (9092).
Tests the full pipeline: upload document → semantic chunk → embed → query → retrieve.
"""

import os
import pytest
import httpx

EMBEDDING_PORT = int(os.getenv("EMBEDDING_PORT", "9092"))
APP_PORT = 8000
APP_URL = f"http://localhost:{APP_PORT}"


@pytest.fixture(scope="module")
def app_available():
    """Check if the FastAPI app is running."""
    try:
        resp = httpx.get(f"{APP_URL}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def embedding_available():
    """Check if the embedding server is running."""
    try:
        resp = httpx.get(f"http://localhost:{EMBEDDING_PORT}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LlamaServerEmbeddings live tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestLlamaServerEmbeddingsLive:
    def test_connect_and_probe_dimension(self, embedding_available):
        if not embedding_available:
            pytest.skip("Embedding server not running")

        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        emb = LlamaServerEmbeddings(base_url=f"http://localhost:{EMBEDDING_PORT}")
        assert emb.dimension == 768

    def test_embed_single_text(self, embedding_available):
        if not embedding_available:
            pytest.skip("Embedding server not running")

        import numpy as np
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        emb = LlamaServerEmbeddings(base_url=f"http://localhost:{EMBEDDING_PORT}")
        vec = emb.embed("Local LLM Inference Masterclass")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (768,)

    def test_embed_batch(self, embedding_available):
        if not embedding_available:
            pytest.skip("Embedding server not running")

        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        emb = LlamaServerEmbeddings(base_url=f"http://localhost:{EMBEDDING_PORT}")
        vecs = emb.embed_batch(["text one", "text two", "text three"])
        assert len(vecs) == 3


# ---------------------------------------------------------------------------
# Semantic chunking live tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestSemanticChunkingLive:
    def test_chonkie_chunker_with_live_embeddings(self, embedding_available):
        """SemanticChunker produces chunks using live embeddinggemma."""
        if not embedding_available:
            pytest.skip("Embedding server not running")

        from chonkie import SemanticChunker
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings

        emb = LlamaServerEmbeddings(base_url=f"http://localhost:{EMBEDDING_PORT}")
        chunker = SemanticChunker(
            embedding_model=emb,
            threshold=0.7,
            chunk_size=256,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=12,
            delim=["\n\n", ". ", "! ", "? "],
        )

        text = (
            "Local LLM Inference Masterclass\nSpeaker A\n"
            "Running LLMs on consumer hardware is finally practical.\n\n"
            "Vector Search at Scale\nSpeaker B\n"
            "Modern embedding models unlock retrieval-augmented generation.\n\n"
            "Fine-tuning Small Models\nSpeaker A\n"
            "LoRA and QLoRA let you specialise 1-4B models on a single GPU."
        )

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # All content should be preserved across chunks
        all_text = " ".join(c.text for c in chunks)
        assert "Speaker A" in all_text
        assert "Speaker B" in all_text
        assert "Local LLM Inference Masterclass" in all_text

