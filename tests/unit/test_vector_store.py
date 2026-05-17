"""
Unit tests for VectorStore.

ChromaDB is a real dependency but runs in-process (no server needed).
SmallLanguageModelClient is mocked so no llama-server calls are made.
"""

import pytest

from src.engine.knowledge.vector_store import Document, VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(chroma_dir: str, mock_client) -> VectorStore:
    store = VectorStore(persist_dir=chroma_dir)
    store.set_client(mock_client)
    return store


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVectorStoreSetup:
    async def test_empty_store_count_is_zero(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        assert await store.count() == 0

    async def test_requires_client_before_add(self, temp_chroma_dir):
        store = VectorStore(persist_dir=temp_chroma_dir)
        with pytest.raises(RuntimeError, match="set_client"):
            await store.add_document(Document(id="x", content="test"))

    async def test_requires_client_before_search(self, temp_chroma_dir):
        store = VectorStore(persist_dir=temp_chroma_dir)
        with pytest.raises(RuntimeError, match="set_client"):
            await store.search("test query")


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexing:
    async def test_add_single_document(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        doc = Document(id="doc-001", content="Hello world", metadata={"source": "test"})
        await store.add_document(doc)
        assert await store.count() == 1

    async def test_add_multiple_documents(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        docs = [
            Document(id=f"doc-{i}", content=f"Content {i}")
            for i in range(5)
        ]
        await store.add_documents(docs)
        assert await store.count() == 5

    async def test_embed_batch_called_once_for_bulk_add(self, temp_chroma_dir, mock_small_language_model_client):
        """Bulk indexing should call embed_batch once, not embed() per document."""
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        docs = [Document(id=f"d{i}", content=f"doc {i}") for i in range(3)]
        await store.add_documents(docs)
        mock_small_language_model_client.embed_batch.assert_called_once()
        mock_small_language_model_client.embed.assert_not_called()

    async def test_document_exists_after_add(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        await store.add_document(Document(id="exists", content="content"))
        assert await store.document_exists("exists") is True

    async def test_document_not_exists_before_add(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        assert await store.document_exists("never-added") is False


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieval:
    async def test_search_empty_store_returns_empty(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        results = await store.search("anything")
        assert results == []

    async def test_search_returns_documents(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        await store.add_documents([
            Document(id="a", content="Alpha document"),
            Document(id="b", content="Beta document"),
        ])
        results = await store.search("query", top_k=2)
        assert len(results) >= 1
        assert all(isinstance(r, Document) for r in results)

    async def test_search_respects_top_k(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        for i in range(10):
            await store.add_document(Document(id=f"d{i}", content=f"Doc number {i}"))
        results = await store.search("doc", top_k=3)
        assert len(results) <= 3

    async def test_search_result_has_score(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        await store.add_document(Document(id="scored", content="test content"))
        results = await store.search("test")
        for doc in results:
            assert doc.score is not None
            assert 0.0 <= doc.score <= 1.0

    async def test_search_result_preserves_metadata(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        await store.add_document(Document(
            id="meta-doc",
            content="with metadata",
            metadata={"title": "My Doc", "category": "test"},
        ))
        results = await store.search("metadata")
        assert len(results) >= 1
        assert results[0].metadata.get("title") == "My Doc"


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClear:
    async def test_clear_removes_all_documents(self, temp_chroma_dir, mock_small_language_model_client):
        store = _make_store(temp_chroma_dir, mock_small_language_model_client)
        await store.add_documents([
            Document(id="x", content="one"),
            Document(id="y", content="two"),
        ])
        assert await store.count() == 2
        await store.clear()
        assert await store.count() == 0
