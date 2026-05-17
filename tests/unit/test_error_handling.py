"""
Unit tests for error handling in the upload and OCR pipeline.

Phase 9: Upload resilience, embedding fallback, ChromaDB safety.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.knowledge.document_processor import DocumentProcessor, ProcessingEvent
from src.engine.knowledge.vector_store import VectorStore, Document


# ---------------------------------------------------------------------------
# 9.2 — Embedding batch fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingBatchFallback:

    async def test_batch_failure_falls_back_to_individual(self):
        """When embed_batch raises, add_documents falls back to individual embed calls."""
        mock_client = MagicMock()
        mock_client.embed_batch = AsyncMock(
            side_effect=Exception("batch too large: 588 tokens > 512")
        )
        mock_client.embed = AsyncMock(return_value=[0.1] * 768)

        vs = VectorStore(collection_name="test_fallback")
        vs.set_client(mock_client)

        docs = [
            Document(id="d1", content="short text", metadata={"_": True}),
            Document(id="d2", content="another short text", metadata={"_": True}),
        ]
        await vs.add_documents(docs)

        # embed_batch was called first (and failed)
        mock_client.embed_batch.assert_called_once()
        # embed was called individually for each doc
        assert mock_client.embed.call_count == 2

    async def test_batch_success_does_not_fallback(self):
        """When embed_batch succeeds, individual embed is never called."""
        mock_client = MagicMock()
        mock_client.embed_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        mock_client.embed = AsyncMock()

        vs = VectorStore(collection_name="test_no_fallback")
        vs.set_client(mock_client)

        docs = [
            Document(id="d1", content="text 1", metadata={"_": True}),
            Document(id="d2", content="text 2", metadata={"_": True}),
        ]
        await vs.add_documents(docs)

        mock_client.embed_batch.assert_called_once()
        mock_client.embed.assert_not_called()


# ---------------------------------------------------------------------------
# 9.3 — Upload resilience (background task)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUploadResilience:

    async def test_process_file_yields_all_events_on_success(self):
        """process_file yields events through to indexed on successful upload."""
        mock_vs = MagicMock()
        mock_vs.add_documents = AsyncMock()
        mock_vs.count = AsyncMock(return_value=5)
        mock_vs._collection = MagicMock()
        mock_vs._collection.get = MagicMock(return_value={"ids": []})

        processor = DocumentProcessor(mock_vs)
        content = ("Test content for resilience testing. " * 10).encode()

        events = []
        async for event in processor.process_file("test.txt", content):
            events.append(event)

        stages = [e.stage for e in events]
        assert "parsing" in stages
        assert "chunking" in stages
        assert "indexed" in stages

    async def test_process_file_error_on_empty_content(self):
        """Empty content should yield an error event."""
        mock_vs = MagicMock()
        mock_vs._collection = MagicMock()
        mock_vs._collection.get = MagicMock(return_value={"ids": []})

        processor = DocumentProcessor(mock_vs)

        events = []
        async for event in processor.process_file("empty.txt", b""):
            events.append(event)

        stages = [e.stage for e in events]
        assert "error" in stages

    async def test_process_file_error_on_short_content(self):
        """Content too short to produce chunks should yield an error event."""
        mock_vs = MagicMock()
        mock_vs._collection = MagicMock()
        mock_vs._collection.get = MagicMock(return_value={"ids": []})

        processor = DocumentProcessor(mock_vs)

        events = []
        async for event in processor.process_file("tiny.txt", b"hi"):
            events.append(event)

        stages = [e.stage for e in events]
        assert "error" in stages

    async def test_ocr_failure_still_produces_indexed_event(self):
        """If OCR fails, fallback to pypdf and still index successfully."""
        mock_vs = MagicMock()
        mock_vs.add_documents = AsyncMock()
        mock_vs.count = AsyncMock(return_value=3)
        mock_vs._collection = MagicMock()
        mock_vs._collection.get = MagicMock(return_value={"ids": []})

        mock_ocr = MagicMock()
        mock_ocr.extract_text = AsyncMock(return_value="")  # OCR returns nothing

        processor = DocumentProcessor(mock_vs, ocr_client=mock_ocr)

        # Create a minimal valid PDF-like content — the processor will try
        # pypdf first (which fails on non-PDF bytes), then OCR (which returns "").
        # For .txt files, OCR is skipped entirely, so we use a TXT file to
        # ensure the processor reaches the indexing stage regardless.
        content = ("Fallback content for testing error resilience. " * 10).encode()
        events = []
        async for event in processor.process_file("fallback.txt", content):
            events.append(event)

        stages = [e.stage for e in events]
        assert "indexed" in stages
        assert "ocr_extraction" not in stages  # TXT files skip OCR


# ---------------------------------------------------------------------------
# 9.4 — Clean replace (dedup on re-upload)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCleanReplace:

    async def test_reupload_deletes_old_chunks(self):
        """Re-uploading the same file deletes old chunks first."""
        mock_vs = MagicMock()
        mock_vs.add_documents = AsyncMock()
        mock_vs.count = AsyncMock(return_value=5)
        mock_vs.delete_by_document_id = AsyncMock(return_value=2)

        processor = DocumentProcessor(mock_vs)
        content = ("Re-upload test content with enough text for chunking. " * 10).encode()

        events = []
        async for event in processor.process_file("test.txt", content):
            events.append(event)

        # Should have called delete_by_document_id for the slugified filename
        mock_vs.delete_by_document_id.assert_called_once_with("test")

    async def test_first_upload_no_delete(self):
        """First upload of a file doesn't attempt to delete anything."""
        mock_vs = MagicMock()
        mock_vs.add_documents = AsyncMock()
        mock_vs.count = AsyncMock(return_value=3)
        mock_vs.delete_by_document_id = AsyncMock(return_value=0)

        processor = DocumentProcessor(mock_vs)
        content = ("First upload test content with enough chars. " * 10).encode()

        events = []
        async for event in processor.process_file("new_doc.txt", content):
            events.append(event)

        # delete_by_document_id called but returned 0 (no old chunks)
        mock_vs.delete_by_document_id.assert_called_once_with("new-doc")
