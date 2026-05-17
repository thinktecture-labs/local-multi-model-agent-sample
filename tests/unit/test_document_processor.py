"""
Unit tests for DocumentProcessor.

Tests the parse, chunk, and index pipeline without requiring
llama-server or real PDF files. SmallLanguageModelClient is mocked,
ChromaDB runs in-process.
"""

import pytest

from src.engine.knowledge.document_processor import DocumentProcessor, ProcessingEvent, TextChunk
from src.engine.knowledge.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processor(chroma_dir: str, mock_client, ocr_client=None) -> DocumentProcessor:
    store = VectorStore(persist_dir=chroma_dir)
    store.set_client(mock_client)
    return DocumentProcessor(store, ocr_client=ocr_client)


async def _collect_events(processor, filename, content) -> list[ProcessingEvent]:
    """Run process_file and collect all emitted events."""
    events = []
    async for event in processor.process_file(filename, content):
        events.append(event)
    return events


def _make_txt_content(text: str) -> bytes:
    return text.encode("utf-8")


# Minimum viable content (>100 chars to pass chunk filter)
SAMPLE_TEXT = (
    "Acme Analytics is building a next-generation customer data platform "
    "to serve enterprise SaaS customers. The platform integrates with "
    "existing CRM systems and provides real-time analytics."
)


def _make_long_text(chars: int = 2000) -> str:
    """Generate text long enough to produce multiple chunks."""
    sentence = "This is a test sentence with enough words to be meaningful. "
    return (sentence * (chars // len(sentence) + 1))[:chars]


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTextParsing:
    async def test_txt_file_produces_events(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        events = await _collect_events(proc, "test.txt", _make_txt_content(SAMPLE_TEXT))
        stages = [e.stage for e in events]
        assert "parsing" in stages
        assert "chunking" in stages
        assert "indexed" in stages

    async def test_md_file_works(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        events = await _collect_events(proc, "readme.md", _make_txt_content("# Title\n\n" + SAMPLE_TEXT))
        assert events[-1].stage == "indexed"

    async def test_empty_file_yields_error(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        events = await _collect_events(proc, "empty.txt", b"   \n  \n  ")
        stages = [e.stage for e in events]
        assert "error" in stages

    async def test_utf8_decode_errors_handled(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        # Invalid UTF-8 bytes should not crash — uses errors="replace"
        content = SAMPLE_TEXT.encode("utf-8") + b" \xff\xfe additional content after invalid bytes"
        events = await _collect_events(proc, "binary.txt", content)
        assert events[-1].stage == "indexed"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChunking:
    def test_short_text_single_chunk(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        chunks = proc._chunk_text(SAMPLE_TEXT, "test.txt", 1)
        assert len(chunks) == 1
        assert SAMPLE_TEXT in chunks[0].text

    def test_long_text_multiple_chunks(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = _make_long_text(2000)
        chunks = proc._chunk_text(text, "test.txt", 1)
        assert len(chunks) > 1

    def test_chunk_metadata(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        chunks = proc._chunk_text(SAMPLE_TEXT, "doc.pdf", 3)
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].page == 3
        assert chunks[0].chunk_index == 0

    def test_empty_text_no_chunks(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        chunks = proc._chunk_text("", "test.txt", 1)
        assert len(chunks) == 0

    def test_whitespace_only_no_chunks(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        chunks = proc._chunk_text("   \n  \t  ", "test.txt", 1)
        assert len(chunks) == 0

    def test_chunk_size_respected(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = _make_long_text(3000)
        chunks = proc._chunk_text(text, "test.txt", 1)
        # Chunks should be roughly chunk_size (512) with some tolerance for boundary seeking
        for chunk in chunks:
            assert len(chunk.text) <= 900  # 800 + boundary tolerance

    def test_no_infinite_loop_on_edge_cases(self, temp_chroma_dir, mock_small_language_model_client):
        """Ensure chunking terminates even on adversarial input."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        # Single very long word (no break points)
        text = "a" * 2000
        chunks = proc._chunk_text(text, "test.txt", 1)
        assert len(chunks) >= 1  # should not hang


# ---------------------------------------------------------------------------
# Full pipeline (indexing)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPipeline:
    async def test_indexed_event_has_chunk_count(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = _make_long_text(2000)
        events = await _collect_events(proc, "doc.txt", _make_txt_content(text))
        indexed = [e for e in events if e.stage == "indexed"]
        assert len(indexed) == 1
        assert indexed[0].detail["total_chunks"] > 1

    async def test_documents_in_chromadb_after_indexing(self, temp_chroma_dir, mock_small_language_model_client):
        store = VectorStore(persist_dir=temp_chroma_dir)
        store.set_client(mock_small_language_model_client)
        proc = DocumentProcessor(store)
        text = _make_long_text(2000)
        await _collect_events(proc, "doc.txt", _make_txt_content(text))
        count = await store.count()
        assert count > 0

    async def test_embedding_progress_events(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = _make_long_text(6000)  # enough for multiple batches
        events = await _collect_events(proc, "big.txt", _make_txt_content(text))
        embed_events = [e for e in events if e.stage == "embedding"]
        assert len(embed_events) >= 1
        # Last embedding event should have progress close to 1.0
        assert embed_events[-1].detail["progress"] > 0.5

    async def test_event_order_is_correct(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        events = await _collect_events(proc, "doc.txt", _make_txt_content(SAMPLE_TEXT))
        stages = [e.stage for e in events]
        # Must follow: parsing -> chunking -> embedding(s) -> indexed
        assert stages[0] == "parsing"
        assert stages[1] == "chunking"
        assert stages[-1] == "indexed"
        for s in stages[2:-1]:
            assert s == "embedding"

    async def test_indexed_detail_has_timings(self, temp_chroma_dir, mock_small_language_model_client):
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        events = await _collect_events(proc, "doc.txt", _make_txt_content(SAMPLE_TEXT))
        indexed = [e for e in events if e.stage == "indexed"][0]
        assert "parse_ms" in indexed.detail
        assert "chunk_ms" in indexed.detail
        assert "embed_ms" in indexed.detail
        assert "total_ms" in indexed.detail

    async def test_document_ids_are_namespaced(self, temp_chroma_dir, mock_small_language_model_client):
        """Uploaded document IDs should start with 'upload_' to avoid collisions with seeded docs."""
        store = VectorStore(persist_dir=temp_chroma_dir)
        store.set_client(mock_small_language_model_client)
        proc = DocumentProcessor(store)
        await _collect_events(proc, "report.txt", _make_txt_content(SAMPLE_TEXT))
        # Check that document_exists works with the expected ID pattern
        assert await store.document_exists("upload_report_p1_c0")


# ---------------------------------------------------------------------------
# PDF parsing (requires pypdf)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPdfParsing:
    def test_parse_pdf_requires_pypdf(self, temp_chroma_dir, mock_small_language_model_client):
        """If pypdf is installed, _parse_pdf should not raise ImportError."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        # Create a minimal valid PDF in memory
        try:
            import pypdf
        except ImportError:
            pytest.skip("pypdf not installed")

        # Minimal PDF bytes (empty single-page document)
        from pypdf import PdfWriter
        import io
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        pages = proc._parse_pdf(pdf_bytes, "blank.pdf")
        # A blank page has no text, so pages list may be empty
        assert isinstance(pages, list)


# ---------------------------------------------------------------------------
# OCR-enhanced pipeline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCREnhancedPipeline:

    async def test_pdf_with_ocr_emits_ocr_stage(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """PDFs with OCR client should emit 5 stages: parsing, ocr_extraction, chunking, embedding, indexed."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        # Use a minimal valid PDF
        try:
            from pypdf import PdfWriter
            import io
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
        except ImportError:
            pytest.skip("pypdf not installed")

        events = await _collect_events(proc, "report.pdf", pdf_bytes)
        stages = [e.stage for e in events]
        assert "ocr_extraction" in stages
        # Order: parsing before ocr_extraction before chunking
        ocr_idx = stages.index("ocr_extraction")
        assert stages.index("parsing") < ocr_idx
        assert ocr_idx < stages.index("chunking")

    async def test_pdf_without_ocr_skips_stage(
        self, temp_chroma_dir, mock_small_language_model_client,
    ):
        """Without OCR client, pipeline has original 4 stages."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=None)
        content = ("This is a test document with enough text " * 10).encode()
        events = await _collect_events(proc, "doc.txt", content)
        stages = [e.stage for e in events]
        assert "ocr_extraction" not in stages
        assert "parsing" in stages
        assert "chunking" in stages

    async def test_txt_upload_skips_ocr(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """TXT uploads should never trigger OCR even if client is available."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        content = ("Text file content with enough text. " * 10).encode()
        events = await _collect_events(proc, "doc.txt", content)
        stages = [e.stage for e in events]
        assert "ocr_extraction" not in stages
        mock_ocr_client.extract_text.assert_not_called()

    async def test_md_upload_skips_ocr(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """MD uploads should never trigger OCR even if client is available."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        content = ("# Heading\n\nMarkdown content with enough text. " * 10).encode()
        events = await _collect_events(proc, "doc.md", content)
        stages = [e.stage for e in events]
        assert "ocr_extraction" not in stages
        mock_ocr_client.extract_text.assert_not_called()

    async def test_ocr_failure_falls_back_to_pypdf(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """When OCR returns empty for all pages, method reports fallback."""
        mock_ocr_client.extract_text.return_value = ""  # OCR returns nothing
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        try:
            from pypdf import PdfWriter
            import io
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
        except ImportError:
            pytest.skip("pypdf not installed")

        events = await _collect_events(proc, "report.pdf", pdf_bytes)
        stages = [e.stage for e in events]
        assert "ocr_extraction" in stages
        # Get the LAST ocr_extraction event (final summary, not intermediate per-page)
        ocr_events = [e for e in events if e.stage == "ocr_extraction"]
        assert ocr_events[-1].detail["method"] in ("fallback-pypdf", "pypdf-only", "hybrid")

    async def test_good_pypdf_skips_ocr_calls(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """When pypdf extracts good text, OCR is not called for those pages."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        try:
            from pypdf import PdfWriter
            import io
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas as rl_canvas
        except ImportError:
            pytest.skip("pypdf or reportlab not installed")

        # Create a PDF with actual text content (pypdf will extract it fine)
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=letter)
        c.drawString(72, 700, "This is a test page with substantial text content that pypdf can extract perfectly well without any need for OCR processing.")
        c.showPage()
        c.save()
        pdf_bytes = buf.getvalue()

        events = await _collect_events(proc, "report.pdf", pdf_bytes)
        stages = [e.stage for e in events]
        assert "ocr_extraction" in stages
        ocr_event = next(e for e in events if e.stage == "ocr_extraction")
        # Good pypdf text → method should be pypdf-only (no OCR calls)
        assert ocr_event.detail["method"] == "pypdf-only"
        assert ocr_event.detail["ocr_pages"] == 0

    async def test_ocr_detail_has_page_counts(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """OCR event detail should contain page breakdown counts."""
        mock_ocr_client.extract_text.return_value = "OCR extracted text content."
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        try:
            from pypdf import PdfWriter
            import io
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)  # blank → triggers OCR
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
        except ImportError:
            pytest.skip("pypdf not installed")

        events = await _collect_events(proc, "report.pdf", pdf_bytes)
        ocr_event = next(e for e in events if e.stage == "ocr_extraction")
        assert "total_pages" in ocr_event.detail
        assert "pypdf_pages" in ocr_event.detail
        assert "ocr_pages" in ocr_event.detail
        assert "ocr_extracted" in ocr_event.detail
        assert "ocr_ms" in ocr_event.detail
        assert "method" in ocr_event.detail

    async def test_ocr_timing_in_indexed_event(
        self, temp_chroma_dir, mock_small_language_model_client, mock_ocr_client,
    ):
        """Final indexed event should contain ocr_ms when OCR was used."""
        mock_ocr_client.extract_text.return_value = "OCR text with enough content to pass the minimum chunk threshold for indexing purposes and retrieval quality."
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client, ocr_client=mock_ocr_client)
        try:
            from pypdf import PdfWriter
            import io
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)  # blank → triggers OCR
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
        except ImportError:
            pytest.skip("pypdf not installed")

        events = await _collect_events(proc, "report.pdf", pdf_bytes)
        indexed_events = [e for e in events if e.stage == "indexed"]
        if indexed_events:
            assert "ocr_ms" in indexed_events[0].detail
            assert indexed_events[0].detail["ocr_ms"] >= 0


# ---------------------------------------------------------------------------
# Table-aware chunking
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTableAwareChunking:

    def test_markdown_table_stays_atomic(self, temp_chroma_dir, mock_small_language_model_client):
        """A Markdown table should remain in a single chunk."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = (
            "Introduction paragraph with enough text to be meaningful.\n\n"
            "| Name | Price | Category |\n"
            "|------|-------|----------|\n"
            "| Starter | 299 | platform |\n"
            "| Professional | 999 | platform |\n"
            "| Enterprise | 3500 | platform |\n\n"
            "Conclusion paragraph with additional context and details."
        )
        chunks = proc._chunk_with_tables(text, "test.md", 1)
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) == 1
        assert "| Starter |" in table_chunks[0].text
        assert "| Enterprise |" in table_chunks[0].text

    def test_table_chunk_has_type_metadata(self, temp_chroma_dir, mock_small_language_model_client):
        """Chunks containing tables should have chunk_type='table'."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "| 3 | 4 |"
        )
        chunks = proc._chunk_with_tables(text, "test.md", 1)
        for c in chunks:
            if "|" in c.text and "---" in c.text:
                assert c.chunk_type == "table"

    def test_prose_chunks_have_type_text(self, temp_chroma_dir, mock_small_language_model_client):
        """Non-table chunks should have chunk_type='text'."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = "Just regular prose text without any tables. " * 20
        chunks = proc._chunk_with_tables(text, "test.md", 1)
        for c in chunks:
            assert c.chunk_type == "text"

    def test_mixed_content_preserves_order(self, temp_chroma_dir, mock_small_language_model_client):
        """Prose → table → prose should produce chunks in that order."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        # Each prose section must be >= 100 chars to pass the minimum chunk threshold.
        # Build separate strings to avoid * 5 repeating the table.
        prose_a = "First section of prose content with enough text to create a meaningful chunk for retrieval and indexing purposes. " * 3
        table = "| Col1 | Col2 |\n|------|------|\n| val1 | val2 |\n| val3 | val4 |"
        prose_b = "Second section of prose content with enough text to create another meaningful chunk for retrieval and indexing purposes. " * 3
        text = prose_a.strip() + "\n" + table + "\n" + prose_b.strip()
        chunks = proc._chunk_with_tables(text, "test.md", 1)
        types = [c.chunk_type for c in chunks]
        # Should have at least: text, table, text
        assert "table" in types
        table_idx = types.index("table")
        assert any(t == "text" for t in types[:table_idx])  # text before table
        assert any(t == "text" for t in types[table_idx + 1:])  # text after table

    def test_no_table_falls_through_to_standard_chunking(self, temp_chroma_dir, mock_small_language_model_client):
        """Text without tables should use standard chunking."""
        proc = _make_processor(temp_chroma_dir, mock_small_language_model_client)
        text = "Regular text content. " * 50
        chunks_table = proc._chunk_with_tables(text, "test.md", 1)
        chunks_standard = proc._chunk_text(text, "test.md", 1)
        # Both should produce the same result for non-table text
        assert len(chunks_table) == len(chunks_standard)
        for ct, cs in zip(chunks_table, chunks_standard):
            assert ct.text == cs.text


# ---------------------------------------------------------------------------
# Smart OCR page quality detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsPoorPypdf:
    """Tests for DocumentProcessor._is_poor_pypdf() heuristic."""

    def test_empty_text_is_poor(self):
        assert DocumentProcessor._is_poor_pypdf("") is True

    def test_short_text_is_poor(self):
        assert DocumentProcessor._is_poor_pypdf("Page 1") is True

    def test_whitespace_only_is_poor(self):
        assert DocumentProcessor._is_poor_pypdf("   \n\n  ") is True

    def test_good_prose_is_not_poor(self):
        text = "This is a well-extracted paragraph with normal English text. " * 5
        assert DocumentProcessor._is_poor_pypdf(text) is False

    def test_garbled_table_is_poor(self):
        """pypdf garbles tables into runs of pipes, dashes, and spaces."""
        text = "|||---|||---|||---|||" * 20
        assert DocumentProcessor._is_poor_pypdf(text) is True

    def test_barely_above_threshold_is_not_poor(self):
        text = "A" * 101  # just above MIN_PYPDF_CHARS
        assert DocumentProcessor._is_poor_pypdf(text) is False

    def test_barely_below_threshold_is_poor(self):
        text = "A" * 99  # just below MIN_PYPDF_CHARS
        assert DocumentProcessor._is_poor_pypdf(text) is True

    def test_mixed_content_with_enough_text(self):
        text = "Revenue Q4 2024: $103,200. New customers: 11. Churn: 0.7%. " * 3
        assert DocumentProcessor._is_poor_pypdf(text) is False
