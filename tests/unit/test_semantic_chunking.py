"""
Unit tests for semantic chunking integration.

Tests the LlamaServerEmbeddings adapter, SemanticChunker config wiring,
and DocumentProcessor fallback behavior. Does NOT require llama-server —
all embedding calls are mocked.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.engine.knowledge.document_processor import DocumentProcessor, TextChunk
from src.engine.knowledge.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROSE_TEXT = (
    "Acme Analytics is building a next-generation customer data platform "
    "to serve enterprise SaaS customers. The platform integrates with "
    "existing CRM systems and provides real-time analytics. Data residency "
    "is configurable per tenant — processing happens either on customer "
    "infrastructure or in a dedicated cloud region. The system uses fine-tuned "
    "language models for structured extraction from contracts and reports."
)


def _make_mock_semantic_embeddings(dimension=768):
    """Create a mock LlamaServerEmbeddings that returns random vectors."""
    mock = MagicMock()
    mock.dimension = dimension
    return mock


def _make_processor_with_semantic(chroma_dir, mock_client, semantic_emb=None):
    store = VectorStore(persist_dir=chroma_dir)
    store.set_client(mock_client)
    return DocumentProcessor(store, semantic_embeddings=semantic_emb)


# ---------------------------------------------------------------------------
# LlamaServerEmbeddings adapter
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLlamaServerEmbeddings:
    def test_import(self):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        assert LlamaServerEmbeddings is not None

    def test_is_available_returns_true(self):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        assert LlamaServerEmbeddings.is_available() is True

    @patch("src.engine.knowledge.semantic_embeddings.httpx.Client")
    def test_dimension_probed_on_init(self, mock_client_cls):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1] * 768}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
        assert emb.dimension == 768

    @patch("src.engine.knowledge.semantic_embeddings.httpx.Client")
    def test_dimension_zero_on_connection_failure(self, mock_client_cls):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        mock_instance = MagicMock()
        mock_instance.post.side_effect = Exception("Connection refused")
        mock_client_cls.return_value = mock_instance

        emb = LlamaServerEmbeddings(base_url="http://localhost:9999")
        assert emb.dimension == 0

    @patch("src.engine.knowledge.semantic_embeddings.httpx.Client")
    def test_embed_returns_numpy_array(self, mock_client_cls):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        mock_instance = MagicMock()
        mock_response = MagicMock()
        vec = [0.5] * 768
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": vec}]}
        mock_response.raise_for_status = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
        result = emb.embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)
        assert result.dtype == np.float32

    @patch("src.engine.knowledge.semantic_embeddings.httpx.Client")
    def test_embed_batch_returns_list_of_arrays(self, mock_client_cls):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1] * 768},
                {"index": 1, "embedding": [0.2] * 768},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
        results = emb.embed_batch(["text 1", "text 2"])
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)

    @patch("src.engine.knowledge.semantic_embeddings.httpx.Client")
    def test_get_tokenizer_returns_tiktoken(self, mock_client_cls):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 768}]}
        mock_response.raise_for_status = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
        tok = emb.get_tokenizer()
        # tiktoken tokenizers have encode/decode
        assert hasattr(tok, "encode")
        assert hasattr(tok, "decode")

    @patch("src.engine.knowledge.semantic_embeddings.httpx.Client")
    def test_repr(self, mock_client_cls):
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 768}]}
        mock_response.raise_for_status = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
        assert "LlamaServerEmbeddings" in repr(emb)
        assert "9092" in repr(emb)


# ---------------------------------------------------------------------------
# DocumentProcessor semantic chunking init
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSemanticChunkingInit:
    def test_no_semantic_embeddings_means_no_chunker(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=None)
        assert proc._semantic_chunker is None

    def test_zero_dimension_means_no_chunker(self, temp_chroma_dir, mock_small_language_model_client):
        mock_emb = _make_mock_semantic_embeddings(dimension=0)
        proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=mock_emb)
        assert proc._semantic_chunker is None

    def test_valid_embeddings_creates_chunker(self, temp_chroma_dir, mock_small_language_model_client):
        """With valid embeddings and chonkie installed, _semantic_chunker should be set."""
        try:
            from chonkie import SemanticChunker
        except ImportError:
            pytest.skip("chonkie not installed")

        # Use a mock embeddings that satisfies chonkie's BaseEmbeddings interface
        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        with patch("src.engine.knowledge.semantic_embeddings.httpx.Client") as mock_cls:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 768}]}
            mock_response.raise_for_status = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_cls.return_value = mock_instance

            emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
            proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=emb)
            assert proc._semantic_chunker is not None


# ---------------------------------------------------------------------------
# DocumentProcessor._chunk_semantic fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSemanticChunkingFallback:
    def test_chunk_semantic_returns_none_when_no_chunker(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=None)
        result = proc._chunk_semantic("Some text", "test.txt", 1)
        assert result is None

    def test_fallback_to_fixed_size_when_semantic_unavailable(self, temp_chroma_dir, mock_small_language_model_client):
        """When semantic chunking is None, _chunk_with_tables should be used."""
        proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=None)
        # _chunk_semantic returns None, so the pipeline falls through to _chunk_with_tables
        chunks = proc._chunk_with_tables(PROSE_TEXT, "test.txt", 1)
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_semantic_returns_none_on_exception(self, temp_chroma_dir, mock_small_language_model_client):
        """If the semantic chunker raises, gracefully return None."""
        proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=None)
        proc._semantic_chunker = MagicMock()
        proc._semantic_chunker.chunk.side_effect = RuntimeError("Model crashed")
        result = proc._chunk_semantic("Some text", "test.txt", 1)
        assert result is None


# ---------------------------------------------------------------------------
# Semantic chunking config
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSemanticChunkingConfig:
    def test_config_defaults(self):
        from src.engine.inference.config import (
            SEMANTIC_CHUNKING_ENABLED,
            SEMANTIC_CHUNKING_THRESHOLD,
            SEMANTIC_CHUNKING_MAX_TOKENS,
            SEMANTIC_CHUNKING_MIN_SENTENCES,
        )
        assert SEMANTIC_CHUNKING_ENABLED is True
        assert 0.0 < SEMANTIC_CHUNKING_THRESHOLD < 1.0
        assert SEMANTIC_CHUNKING_MAX_TOKENS > 0
        assert SEMANTIC_CHUNKING_MIN_SENTENCES >= 1

    def test_doc_chat_config_higher_than_rag(self):
        from src.engine.inference.config import RAG_TOP_K, RAG_CONTEXT_DOCS, DOC_CHAT_TOP_K, DOC_CHAT_CONTEXT_DOCS
        assert DOC_CHAT_TOP_K > RAG_TOP_K, "Doc chat needs broader retrieval than curated KB RAG"
        assert DOC_CHAT_CONTEXT_DOCS > RAG_CONTEXT_DOCS, "Doc chat needs more context docs than curated KB RAG"


# ---------------------------------------------------------------------------
# Semantic chunking delimiter config
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSemanticChunkingDelimiters:
    def test_chunker_uses_paragraph_delimiters(self, temp_chroma_dir, mock_small_language_model_client):
        """The SemanticChunker should use \\n\\n as primary delimiter."""
        try:
            from chonkie import SemanticChunker
        except ImportError:
            pytest.skip("chonkie not installed")

        from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
        with patch("src.engine.knowledge.semantic_embeddings.httpx.Client") as mock_cls:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 768}]}
            mock_response.raise_for_status = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_cls.return_value = mock_instance

            emb = LlamaServerEmbeddings(base_url="http://localhost:9092")
            proc = _make_processor_with_semantic(temp_chroma_dir, mock_small_language_model_client, semantic_emb=emb)
            assert proc._semantic_chunker is not None
            # Verify delimiter config includes paragraph boundary
            chunker = proc._semantic_chunker
            assert "\n\n" in chunker.delim
