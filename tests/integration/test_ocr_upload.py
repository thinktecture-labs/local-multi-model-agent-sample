"""
Integration tests for document upload with OCR.

Tests the /upload-document and /query endpoints with document_id
using FastAPI TestClient with mocked components.
No llama-server instances needed.
"""

import io
import json

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.engine.agent import AgentResponse, Intent, ExecutionStep
from src.engine.inference.client import SmallLanguageModelRole, LLMResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with mocked lifespan including upload_store and ocr_client."""
    import src.server as server_module

    mock_client = MagicMock()
    mock_client.models = {
        SmallLanguageModelRole.INFERENCE: "gemma3:1b-it",
        SmallLanguageModelRole.FUNCTION: "qwen",
        SmallLanguageModelRole.EMBEDDING: "embeddinggemma",
        SmallLanguageModelRole.VISION: "gemma3-4b-vision",
    }
    mock_client.check_health = AsyncMock(return_value={
        "INFERENCE": True, "FUNCTION": True, "EMBEDDING": True, "VISION": True,
    })
    mock_client.embed = AsyncMock(return_value=[0.1] * 768)
    mock_client.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 768] * len(texts))
    mock_client.generate_synthesis = AsyncMock(return_value=LLMResponse(
        content="The document says NRR is 126%.",
        model="gemma3-4b",
        tokens_used=20,
        prompt_tokens=15,
        completion_tokens=5,
    ))
    mock_client.generate_synthesis_stream = MagicMock()

    mock_vs = MagicMock()
    mock_vs.count = AsyncMock(return_value=13)
    mock_vs.set_client = MagicMock()
    mock_vs._collection = MagicMock()
    mock_vs._collection.get = MagicMock(return_value={"ids": [], "metadatas": []})

    # Upload store — separate mock
    mock_upload_store = MagicMock()
    mock_upload_store.count = AsyncMock(return_value=0)
    mock_upload_store.set_client = MagicMock()
    mock_upload_store.add_documents = AsyncMock()
    mock_upload_store.clear = AsyncMock()
    mock_upload_store._collection = MagicMock()
    mock_upload_store._collection.get = MagicMock(return_value={"ids": [], "metadatas": []})
    mock_upload_store._collection.delete = MagicMock()

    from src.engine.knowledge.vector_store import Document
    mock_upload_store.search = AsyncMock(return_value=[
        Document(id="chunk_1", content="NRR is 126%", metadata={"title": "test.pdf", "document_id": "test"}, score=0.92),
    ])

    mock_tools = MagicMock()
    mock_tools.list_tools = MagicMock(return_value=["calculator", "sql_query", "vector_search"])
    mock_tools.get_all_schemas = MagicMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.process = AsyncMock(return_value=AgentResponse(
        query="test", intent=Intent.DIRECT_ANSWER, response="Hello!",
        steps=[ExecutionStep(action="direct_response", model="gemma3:1b-it")],
        execution_time_ms=42.0,
    ))
    mock_agent.interaction_count = 0
    mock_agent.eviction_count = 0

    @asynccontextmanager
    async def mock_lifespan(app):
        server_module._state.client = mock_client
        server_module._state.vector_store = mock_vs
        server_module._state.upload_store = mock_upload_store
        server_module._state.tools = mock_tools
        server_module._state.agent = mock_agent
        server_module._state.ocr_available = False
        server_module._state.ocr_client = None
        server_module._state.cloud_bytes_sent = 0
        server_module._state.network_mode = "online"
        server_module._state.routing_mode = "local-only"
        server_module._state.model_mode = "finetuned"
        server_module._state.qwen_available = False
        server_module._state.qwen_agent = None
        server_module._state.cloud_orchestrator = None
        yield

    original_lifespan = server_module.app.router.lifespan_context
    server_module.app.router.lifespan_context = mock_lifespan

    with TestClient(server_module.app, raise_server_exceptions=False) as tc:
        tc._mock_upload_store = mock_upload_store
        tc._mock_client = mock_client
        yield tc

    server_module.app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# Upload endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUploadEndpoint:

    def test_upload_txt_returns_sse_events(self, client):
        """Upload a .txt file, verify SSE events are returned."""
        content = ("This is test content with enough text. " * 10).encode()
        resp = client.post(
            "/upload-document",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        events = [line for line in resp.text.split("\n") if line.startswith("data:")]
        assert len(events) >= 2  # at least parsing + indexed

    def test_upload_txt_no_ocr_stage(self, client):
        """TXT uploads should never emit ocr_extraction stage."""
        content = ("Text content for testing. " * 20).encode()
        resp = client.post(
            "/upload-document",
            files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
        )
        stages = []
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                stages.append(data.get("stage"))
        assert "ocr_extraction" not in stages

    def test_upload_unsupported_format_returns_400(self, client):
        """Uploading a .docx should return 400."""
        resp = client.post(
            "/upload-document",
            files={"file": ("doc.docx", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_too_large_returns_400(self, client):
        """Files over 10MB should be rejected."""
        big = b"x" * (11 * 1024 * 1024)
        resp = client.post(
            "/upload-document",
            files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Health endpoint OCR status
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHealthOCRStatus:

    def test_health_includes_ocr_field(self, client):
        """Health response should include OCR availability."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "OCR" in data["models"]

    def test_health_ocr_false_when_not_available(self, client):
        """OCR should be false when ocr_client is not set."""
        resp = client.get("/health")
        assert resp.json()["models"]["OCR"] is False


# ---------------------------------------------------------------------------
# Document chat via /query
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDocumentChat:

    def test_query_with_document_id_returns_document_chat(self, client):
        """Query with document_id should return intent=document_chat."""
        resp = client.post("/query", json={
            "query": "What is the NRR?",
            "document_id": "test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "document_chat"

    def test_query_with_document_id_has_document_search_step(self, client):
        """Document chat should include a document_search step."""
        resp = client.post("/query", json={
            "query": "What is the NRR?",
            "document_id": "test",
        })
        data = resp.json()
        actions = [s["action"] for s in data["steps"]]
        assert "document_search" in actions

    def test_query_with_document_id_includes_documents_in_step(self, client):
        """Document search step should include retrieved documents."""
        resp = client.post("/query", json={
            "query": "What is the NRR?",
            "document_id": "test",
        })
        data = resp.json()
        search_step = next(s for s in data["steps"] if s["action"] == "document_search")
        assert "documents" in search_step["details"]

    def test_query_without_document_id_uses_normal_pipeline(self, client):
        """Query without document_id should use normal agent pipeline."""
        resp = client.post("/query", json={"query": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] != "document_chat"


# ---------------------------------------------------------------------------
# Clear uploads
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestClearUploads:

    def test_delete_uploads_returns_status(self, client):
        """DELETE /uploads should return cleared status."""
        resp = client.delete("/uploads")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cleared"


# ---------------------------------------------------------------------------
# Upload status endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUploadStatus:

    def test_status_idle_when_no_upload(self, client):
        """GET /uploads/status returns idle when nothing is uploading."""
        resp = client.get("/uploads/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["filename"] is None

    def test_status_after_upload_completes(self, client):
        """After a successful upload, status should be idle (task completed)."""
        content = ("Test content for status endpoint check. " * 10).encode()
        client.post(
            "/upload-document",
            files={"file": ("status_test.txt", io.BytesIO(content), "text/plain")},
        )
        resp = client.get("/uploads/status")
        data = resp.json()
        # After upload completes synchronously in TestClient, status should be idle or completed
        assert data["status"] in ("idle", "completed")


# ---------------------------------------------------------------------------
# Upload resilience (background task)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUploadResilience:

    def test_upload_produces_indexed_event(self, client):
        """Upload must produce an indexed event as the final SSE event."""
        content = ("Resilience test content with enough text for chunking. " * 10).encode()
        resp = client.post(
            "/upload-document",
            files={"file": ("resilience.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
        stages = [e["stage"] for e in events]
        assert "indexed" in stages, f"Expected indexed event, got stages: {stages}"

    def test_upload_error_event_on_corrupt_pdf(self, client):
        """Uploading a corrupt PDF should produce parsing but still handle gracefully."""
        corrupt_pdf = b"%PDF-1.4 this is not a valid pdf"
        resp = client.post(
            "/upload-document",
            files={"file": ("corrupt.pdf", io.BytesIO(corrupt_pdf), "application/pdf")},
        )
        assert resp.status_code == 200
        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
        stages = [e["stage"] for e in events]
        # Should have at least parsing stage, and either error or indexed
        assert "parsing" in stages


# ---------------------------------------------------------------------------
# Data extraction endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestExtractEndpoint:

    def test_extract_returns_404_for_missing_document(self, client):
        """POST /extract with unknown document_id returns 404."""
        client._mock_upload_store.search = AsyncMock(return_value=[])
        resp = client.post("/extract", json={"document_id": "nonexistent-doc"})
        assert resp.status_code == 404

    def test_extract_returns_extraction_result(self, client):
        """POST /extract with a valid document_id returns extraction result."""
        # Mock upload_store to return docs for this document_id
        from src.engine.knowledge.vector_store import Document
        client._mock_upload_store.search = AsyncMock(return_value=[
            Document(
                id="chunk_1",
                content="Snowflake FY2025: total revenue $3.63 billion, 29% growth YoY. "
                        "Net revenue retention rate 126%. 580 customers with >$1M trailing revenue. "
                        "Product revenue $3.46 billion. Free cash flow $884 million.",
                metadata={"title": "snowflake.pdf", "document_id": "snowflake-fy2025"},
                score=0.92,
            ),
        ])

        # Mock the LLM synthesis to return extraction JSON
        from src.engine.inference.client import LLMResponse
        client._mock_client.generate_synthesis = AsyncMock(return_value=LLMResponse(
            content='{"company": "Snowflake", "fiscal_year": 2025, "revenue": 3630000000, "nrr": 126}',
            model="gemma3-4b",
            tokens_used=30,
            prompt_tokens=20,
            completion_tokens=10,
        ))

        resp = client.post("/extract", json={"document_id": "snowflake-fy2025"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stored"] is True
        assert data["extracted"]["company"] == "Snowflake"
        assert data["extracted"]["nrr"] == 126
        assert data["execution_time_ms"] > 0

    def test_extract_returns_raw_output(self, client):
        """The raw LLM output should be included in the response for debug UI."""
        from src.engine.knowledge.vector_store import Document
        from src.engine.inference.client import LLMResponse
        client._mock_upload_store.search = AsyncMock(return_value=[
            Document(id="c1", content="Revenue data here", metadata={"document_id": "test"}, score=0.9),
        ])
        raw = '{"company": "Test", "fiscal_year": 2025}'
        client._mock_client.generate_synthesis = AsyncMock(return_value=LLMResponse(
            content=raw, model="gemma3-4b", tokens_used=10, prompt_tokens=5, completion_tokens=5,
        ))

        resp = client.post("/extract", json={"document_id": "test"})
        data = resp.json()
        assert data["raw_output"] == raw


# ---------------------------------------------------------------------------
# Competitors endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCompetitorsEndpoint:

    def test_competitors_returns_list(self, client):
        """GET /competitors returns a list (possibly empty)."""
        resp = client.get("/competitors")
        assert resp.status_code == 200
        data = resp.json()
        assert "competitors" in data
        assert "count" in data
        assert isinstance(data["competitors"], list)
