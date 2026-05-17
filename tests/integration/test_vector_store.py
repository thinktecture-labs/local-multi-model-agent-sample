"""
Integration tests for VectorStore — real ChromaDB, mocked embeddings.

These tests do NOT require a llama-server or GPU. They exercise the
real ChromaDB query path and verify the distance-to-similarity conversion
produces scores in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.engine.knowledge.vector_store import Document, VectorStore


@pytest.fixture
def mock_client():
    """Embed client that returns deterministic 4-dim unit vectors."""
    client = MagicMock()
    # Each call returns the same placeholder; override per-test where needed.
    client.embed = AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0])
    client.embed_batch = AsyncMock(
        side_effect=lambda texts: [[1.0, 0.0, 0.0, 0.0]] * len(texts)
    )
    return client


@pytest.fixture
async def populated_store(tmp_path, mock_client):
    """A VectorStore with 3 documents already indexed."""
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    vs.set_client(mock_client)
    await vs.add_documents([
        Document(id="doc-1", content="Nextera Enterprise plan pricing", metadata={"title": "Enterprise"}),
        Document(id="doc-2", content="Professional plan features overview", metadata={"title": "Professional"}),
        Document(id="doc-3", content="FAQ about billing and invoices", metadata={"title": "FAQ"}),
    ])
    return vs


# ---------------------------------------------------------------------------
# Basic query correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_returns_results(populated_store):
    """A populated store returns results for any query."""
    results = await populated_store.search("pricing", top_k=3)
    assert len(results) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_top_k_respected(populated_store):
    """top_k caps the number of results returned."""
    results = await populated_store.search("plan", top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_returns_document_fields(populated_store):
    """Each result has id, content, metadata, and score."""
    results = await populated_store.search("Enterprise", top_k=3)
    for doc in results:
        assert doc.id is not None
        assert doc.content
        assert isinstance(doc.metadata, dict)
        assert doc.score is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_empty_store_returns_no_results(tmp_path, mock_client):
    """Querying an empty store returns an empty list, not an error."""
    vs = VectorStore(persist_dir=str(tmp_path / "empty_chroma"))
    vs.set_client(mock_client)
    results = await vs.search("anything", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Score range — verifies the distance→similarity conversion is correct
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
async def test_all_scores_in_unit_interval(populated_store):
    """All returned similarity scores must be in [0, 1].

    ChromaDB cosine distance ∈ [0, 2]; naive `1 - distance` gives [-1, 1].
    The correct formula `1 - distance/2` guarantees [0, 1].
    """
    results = await populated_store.search("plan pricing features", top_k=5)
    for doc in results:
        assert 0.0 <= doc.score <= 1.0, (
            f"Score {doc.score!r} for doc {doc.id!r} is outside [0, 1]"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_identical_query_gives_score_near_one(tmp_path):
    """A query embedding identical to a stored doc should score close to 1."""
    client = MagicMock()
    vec = [0.5, 0.5, 0.5, 0.5]
    client.embed_batch = AsyncMock(return_value=[vec])
    client.embed = AsyncMock(return_value=vec)

    vs = VectorStore(persist_dir=str(tmp_path / "chroma_identical"))
    vs.set_client(client)
    await vs.add_documents([Document(id="exact", content="match", metadata={})])

    results = await vs.search("match", top_k=1)
    assert len(results) == 1
    assert results[0].score >= 0.99, f"Expected near-1 score, got {results[0].score}"


# ---------------------------------------------------------------------------
# Production LogReg model file sanity check
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_production_logreg_model_loadable():
    """Verify the production LogReg model file exists and can be loaded by joblib.

    A missing or corrupt file causes a silent runtime crash on the first
    request — this test surfaces the problem at CI time instead.
    """
    import joblib

    model_path = Path("models/intent-logreg/model.joblib")
    assert model_path.exists(), (
        f"LogReg model not found at {model_path}. "
        "Run: python -m training.train_intent_logreg"
    )
    model = joblib.load(model_path)
    # Verify the loaded object is a sklearn estimator with predict_proba
    assert hasattr(model, "predict_proba"), (
        f"Loaded object {type(model)} is not a classifier with predict_proba"
    )
