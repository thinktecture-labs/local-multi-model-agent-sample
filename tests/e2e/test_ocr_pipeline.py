"""
End-to-end tests for the GLM-OCR document processing pipeline.

Requires:
  - All core llama-server instances running (ports 9090-9093)
  - GLM-OCR server running on port 9098
  - Demo data seeded (python -m data.loader)

All tests auto-skip when servers are unreachable.

Run:
  bash scripts/start_servers.sh --bg
  pytest tests/e2e/test_ocr_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engine.inference.client import SmallLanguageModelClient
from src.engine.knowledge.vector_store import VectorStore
from src.engine.knowledge.ocr_client import OCRClient
from src.engine.knowledge.document_processor import DocumentProcessor
from src.engine.tools.tool_registry import create_default_registry
from src.engine.agent import SmallLanguageModelAgentOrchestrator
from src.engine.inference.config import SCENARIO_CONFIG


_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
DEMO_DOCS = _DATA_ROOT / "demo-documents"
NEXTERA_PDF = DEMO_DOCS / "nextera_quarterly_report.pdf"
# Use the Nextera quarterly report as the OCR test fixture — exercises the same
# OCR + extraction pipeline as any other PDF, against a known-public synthetic doc.
OCR_TEST_PDF = NEXTERA_PDF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_client(servers_available):
    """SmallLanguageModelClient connected to live llama-servers."""
    if not servers_available:
        pytest.skip("llama-servers not running")
    return SmallLanguageModelClient.create_with_auto_detection()


@pytest.fixture(scope="module")
def live_vector_store(live_client):
    """VectorStore with real embeddinggemma for OCR e2e tests."""
    vs = VectorStore(collection_name="test_ocr_e2e", persist_dir=SCENARIO_CONFIG.chroma_dir)
    vs.set_client(live_client)
    return vs


@pytest.fixture(scope="module")
def ocr_processor(live_vector_store, ocr_server_available):
    """DocumentProcessor with a real OCR client (requires GLM-OCR server)."""
    if not ocr_server_available:
        pytest.skip("GLM-OCR server not running on port 9098")
    ocr_client = OCRClient()
    return DocumentProcessor(live_vector_store, ocr_client=ocr_client)


@pytest.fixture(scope="module")
def pypdf_processor(live_vector_store):
    """DocumentProcessor without OCR — pypdf fallback path."""
    return DocumentProcessor(live_vector_store, ocr_client=None)


@pytest.fixture(scope="module")
def agent(live_client, live_vector_store):
    """Fully-wired agent for querying OCR-uploaded content."""
    tools = create_default_registry(vector_store=live_vector_store)
    return SmallLanguageModelAgentOrchestrator(client=live_client, tools=tools)


# ---------------------------------------------------------------------------
# Tests: pypdf fallback (no OCR server needed — just llama-servers)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestPypdfFallbackPipeline:
    """Upload pipeline without OCR — verifies pypdf path with real embeddings."""

    async def test_upload_nextera_pdf_without_ocr(self, pypdf_processor):
        """Upload Nextera quarterly report via pypdf, verify indexing works."""
        if not NEXTERA_PDF.is_file():
            pytest.skip(f"Nextera PDF not found: {NEXTERA_PDF}")

        events = []
        async for event in pypdf_processor.process_file(NEXTERA_PDF.name, NEXTERA_PDF.read_bytes()):
            events.append(event)

        stages = [e.stage for e in events]
        assert "ocr_extraction" not in stages, "OCR stage should not appear without OCR client"
        assert "indexed" in stages

        indexed = next(e for e in events if e.stage == "indexed")
        assert indexed.detail["total_chunks"] > 0, "Should have indexed at least 1 chunk"

    async def test_uploaded_pdf_is_queryable(self, pypdf_processor, agent):
        """After pypdf upload, agent can answer questions about the PDF content."""
        pdf_path = NEXTERA_PDF
        query = "What was total revenue in Q4 2024?"
        keywords = ["103,200", "103200", "103.2"]

        if not pdf_path.is_file():
            pytest.skip(f"PDF not found: {pdf_path}")

        # Upload first
        async for _ in pypdf_processor.process_file(pdf_path.name, pdf_path.read_bytes()):
            pass

        # Query
        result = await agent.process(query)
        assert result.success
        assert any(kw in result.response for kw in keywords), (
            f"Expected one of {keywords} in response, got: {result.response[:200]}"
        )

    async def test_cross_validation_ocr_vs_sql(self, pypdf_processor, agent):
        """OCR-path and SQL-path should return the same Q3 2024 revenue figure."""
        if not NEXTERA_PDF.is_file():
            pytest.skip(f"Nextera PDF not found: {NEXTERA_PDF}")

        # Upload the PDF
        async for _ in pypdf_processor.process_file(NEXTERA_PDF.name, NEXTERA_PDF.read_bytes()):
            pass

        # Query via RAG (from uploaded PDF)
        rag_result = await agent.process("What was Q3 2024 revenue according to the quarterly report?")

        # Query via SQL (from database)
        sql_result = await agent.process("What was the revenue in Q3 2024?")

        # Both should reference 84,900
        for result, label in [(rag_result, "RAG"), (sql_result, "SQL")]:
            assert result.success, f"{label} query failed"


# ---------------------------------------------------------------------------
# Tests: GLM-OCR (requires OCR server on port 9098)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestOCREndToEnd:
    """Full OCR pipeline tests — require GLM-OCR server on port 9098."""

    async def test_upload_real_pdf_extracts_text(self, ocr_processor):
        """Upload a real PDF and verify OCR extracts text."""
        if not OCR_TEST_PDF.is_file():
            pytest.skip(f"Demo PDF not found: {OCR_TEST_PDF}")

        pdf_bytes = OCR_TEST_PDF.read_bytes()
        events = []
        async for event in ocr_processor.process_file(OCR_TEST_PDF.name, pdf_bytes):
            events.append(event)

        stages = [e.stage for e in events]
        assert "ocr_extraction" in stages, f"Expected ocr_extraction stage, got: {stages}"

        indexed = next(e for e in events if e.stage == "indexed")
        assert indexed.detail["total_chunks"] > 0

    async def test_ocr_extraction_has_timing(self, ocr_processor):
        """Verify OCR extraction event reports method and page info.

        Smart OCR may decide no pages need OCR (pypdf-only) — the event
        still has method, total_pages, and ocr_pages fields.
        """
        if not OCR_TEST_PDF.is_file():
            pytest.skip(f"Demo PDF not found: {OCR_TEST_PDF}")

        pdf_bytes = OCR_TEST_PDF.read_bytes()
        async for event in ocr_processor.process_file(OCR_TEST_PDF.name, pdf_bytes):
            if event.stage == "ocr_extraction":
                assert "method" in event.detail
                assert "total_pages" in event.detail
                assert "ocr_pages" in event.detail
                assert "ocr_ms" in event.detail
                # If OCR ran, it should complete in under 2 min
                if event.detail["ocr_pages"] > 0:
                    assert event.detail["ocr_ms"] < 120_000, f"OCR took too long: {event.detail['ocr_ms']}ms"
                break

    async def test_ocr_indexed_event_has_timing(self, ocr_processor):
        """Final indexed event should contain total_ms and per-stage timing."""
        if not OCR_TEST_PDF.is_file():
            pytest.skip(f"Demo PDF not found: {OCR_TEST_PDF}")

        pdf_bytes = OCR_TEST_PDF.read_bytes()
        events = []
        async for event in ocr_processor.process_file(OCR_TEST_PDF.name, pdf_bytes):
            events.append(event)

        indexed = next(e for e in events if e.stage == "indexed")
        assert "total_ms" in indexed.detail
        assert "parse_ms" in indexed.detail
        assert "embed_ms" in indexed.detail
        # ocr_ms is only present when OCR actually ran (ocr_ms > 0)
        # For PDFs where pypdf is sufficient, it's correctly absent
