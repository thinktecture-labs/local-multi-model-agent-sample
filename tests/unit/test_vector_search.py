"""
Unit tests for VectorSearchTool.

Tests the execute() method, top_k clamping, schema generation, and error
handling — using a mocked VectorStore so no embedding server is needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.engine.tools.vector_search import VectorSearchTool
from src.engine.tools.tool_result import ToolResult
from src.engine.knowledge.vector_store import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.search = AsyncMock(return_value=[
        Document(id="doc-1", content="First doc.", metadata={"title": "Doc 1"}, score=0.95),
        Document(id="doc-2", content="Second doc.", metadata={"title": "Doc 2"}, score=0.80),
    ])
    return store


@pytest.fixture
def tool(mock_store):
    return VectorSearchTool(mock_store)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_name(self, tool):
        assert tool.name == "vector_search"

    def test_schema_has_function_type(self, tool):
        schema = tool.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "vector_search"

    def test_schema_requires_query(self, tool):
        schema = tool.get_schema()
        params = schema["function"]["parameters"]
        assert "query" in params["properties"]
        assert "query" in params["required"]

    def test_schema_has_top_k(self, tool):
        schema = tool.get_schema()
        params = schema["function"]["parameters"]
        assert "top_k" in params["properties"]


# ---------------------------------------------------------------------------
# Execute — happy path
# ---------------------------------------------------------------------------

class TestExecute:
    async def test_returns_documents(self, tool, mock_store):
        result = await tool.execute(query="enterprise plan")
        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0].id == "doc-1"

    async def test_passes_query_to_store(self, tool, mock_store):
        await tool.execute(query="test query", top_k=3)
        mock_store.search.assert_called_once_with("test query", top_k=3)

    async def test_default_top_k(self, tool, mock_store):
        await tool.execute(query="test")
        mock_store.search.assert_called_once_with("test", top_k=5)


# ---------------------------------------------------------------------------
# top_k clamping
# ---------------------------------------------------------------------------

class TestTopKClamping:
    async def test_clamp_to_minimum_1(self, tool, mock_store):
        await tool.execute(query="test", top_k=0)
        mock_store.search.assert_called_once_with("test", top_k=1)

    async def test_clamp_negative_to_1(self, tool, mock_store):
        await tool.execute(query="test", top_k=-5)
        mock_store.search.assert_called_once_with("test", top_k=1)

    async def test_clamp_to_maximum(self, tool, mock_store):
        from src.engine.inference.config import VECTOR_SEARCH_MAX_K
        await tool.execute(query="test", top_k=100)
        mock_store.search.assert_called_once_with("test", top_k=VECTOR_SEARCH_MAX_K)

    async def test_valid_top_k_passed_through(self, tool, mock_store):
        await tool.execute(query="test", top_k=7)
        mock_store.search.assert_called_once_with("test", top_k=7)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_store_exception_returns_failure(self, mock_store):
        mock_store.search = AsyncMock(side_effect=RuntimeError("ChromaDB offline"))
        tool = VectorSearchTool(mock_store)
        result = await tool.execute(query="test")
        assert result.success is False
        assert "ChromaDB offline" in result.error

    async def test_error_has_no_data(self, mock_store):
        mock_store.search = AsyncMock(side_effect=ValueError("bad query"))
        tool = VectorSearchTool(mock_store)
        result = await tool.execute(query="test")
        assert result.data is None

    async def test_empty_results_still_successful(self, mock_store):
        mock_store.search = AsyncMock(return_value=[])
        tool = VectorSearchTool(mock_store)
        result = await tool.execute(query="nothing here")
        assert result.success is True
        assert result.data == []


# ---------------------------------------------------------------------------
# Upload merge with relevance threshold
# ---------------------------------------------------------------------------

class TestUploadMergeThreshold:
    """Upload chunks only merge into general RAG when score >= UPLOAD_MERGE_MIN_SCORE."""

    async def test_high_score_uploads_merge(self, mock_store):
        """Upload chunks scoring above threshold should appear in results."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=5)
        upload_store.search = AsyncMock(return_value=[
            Document(id="up-1", content="Relevant upload.", metadata={"title": "upload.txt"}, score=0.90),
        ])
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        result = await tool.execute(query="test")
        assert result.success
        ids = [d.id for d in result.data]
        assert "up-1" in ids, "High-score upload should be merged"

    async def test_low_score_uploads_filtered(self, mock_store):
        """Upload chunks scoring below threshold should NOT appear in results."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=5)
        upload_store.search = AsyncMock(return_value=[
            Document(id="up-1", content="Weak match.", metadata={"title": "agenda.txt"}, score=0.79),
            Document(id="up-2", content="Another weak.", metadata={"title": "agenda.txt"}, score=0.75),
        ])
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        result = await tool.execute(query="test")
        assert result.success
        ids = [d.id for d in result.data]
        assert "up-1" not in ids, "Low-score upload should be filtered out"
        assert "up-2" not in ids

    async def test_mixed_scores_only_high_merge(self, mock_store):
        """Only upload chunks above threshold merge; others are dropped."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=5)
        upload_store.search = AsyncMock(return_value=[
            Document(id="up-good", content="Strong match.", metadata={}, score=0.90),
            Document(id="up-bad", content="Weak match.", metadata={}, score=0.70),
        ])
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        result = await tool.execute(query="test")
        ids = [d.id for d in result.data]
        assert "up-good" in ids
        assert "up-bad" not in ids

    async def test_no_uploads_returns_kb_only(self, mock_store):
        """When upload_store is empty, only KB results returned."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=0)
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        result = await tool.execute(query="test")
        assert result.success
        assert len(result.data) == 2  # just the two mock KB docs
        assert all(d.id.startswith("doc-") for d in result.data)

    async def test_no_upload_store_returns_kb_only(self, mock_store):
        """When upload_store is None, only KB results returned."""
        tool = VectorSearchTool(mock_store, upload_store=None)
        result = await tool.execute(query="test")
        assert result.success
        assert len(result.data) == 2


class TestIncludeAllUploads:
    """Escalation path: include_all_uploads=True bypasses UPLOAD_MERGE_MIN_SCORE."""

    async def test_low_score_uploads_pass_when_flag_set(self, mock_store):
        """Below-threshold upload chunks are included when include_all_uploads=True."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=2)
        upload_store.search = AsyncMock(return_value=[
            Document(id="up-low", content="Below threshold but relevant.",
                     metadata={"title": "uploaded.pdf"}, score=0.78),
            Document(id="up-mid", content="Borderline relevance.",
                     metadata={"title": "uploaded.pdf"}, score=0.72),
        ])
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        result = await tool.execute(query="test", include_all_uploads=True)
        assert result.success
        ids = [d.id for d in result.data]
        assert "up-low" in ids, "Below-threshold upload must merge when flag is set"
        assert "up-mid" in ids, "Even lower-scoring upload must merge when flag is set"

    async def test_default_behaviour_unchanged(self, mock_store):
        """Without the flag, the threshold filter still applies (regression guard)."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=1)
        upload_store.search = AsyncMock(return_value=[
            Document(id="up-low", content="Below threshold.", metadata={}, score=0.78),
        ])
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        # Default call (escalation flag NOT set)
        result = await tool.execute(query="test")
        assert result.success
        ids = [d.id for d in result.data]
        assert "up-low" not in ids, "Below-threshold upload must still be filtered by default"

    async def test_explicit_false_matches_default(self, mock_store):
        """include_all_uploads=False is identical to default behaviour."""
        upload_store = MagicMock()
        upload_store.count = AsyncMock(return_value=1)
        upload_store.search = AsyncMock(return_value=[
            Document(id="up-low", content="Below threshold.", metadata={}, score=0.78),
        ])
        tool = VectorSearchTool(mock_store, upload_store=upload_store)
        result = await tool.execute(query="test", include_all_uploads=False)
        ids = [d.id for d in result.data]
        assert "up-low" not in ids
