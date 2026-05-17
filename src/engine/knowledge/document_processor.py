"""
DocumentProcessor — Parse, chunk, and index uploaded files via embeddinggemma.

Accepts PDF, TXT, and MD files as raw bytes. Yields ProcessingEvent objects
at each stage for real-time progress reporting via SSE.

When GLM-OCR is available (ocr_client provided), PDFs are processed via OCR
for superior table/image/scanned-PDF extraction. Falls back to pypdf when
OCR is not available or fails.

Pipeline stages:
  parsing → [ocr_extraction] → chunking → embedding → indexed

All processing is local. No data leaves the machine.
"""

import asyncio
import io
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from .vector_store import Document, VectorStore

if TYPE_CHECKING:
    from .ocr_client import OCRClient
    from .semantic_embeddings import LlamaServerEmbeddings

logger = logging.getLogger(__name__)

# Markdown table line: starts with |, contains at least one more |
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|", re.MULTILINE)
# Separator line in Markdown tables: |---|---|
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:]+\|", re.MULTILINE)


@dataclass
class ProcessingEvent:
    """A progress event emitted during document processing."""
    stage: str          # "parsing", "ocr_extraction", "chunking", "embedding", "indexed", "error"
    message: str        # Human-readable status
    detail: dict = field(default_factory=dict)


@dataclass
class TextChunk:
    """A chunk of text with source metadata."""
    text: str
    source: str
    page: int
    chunk_index: int
    chunk_type: str = "text"  # "text" or "table"


class DocumentProcessor:
    """
    Parse, chunk, and index uploaded documents.

    Uses the existing VectorStore (embeddinggemma + ChromaDB) for embedding
    and storage. The process_file() async generator yields events at each
    pipeline stage, enabling real-time SSE progress in the Observatory UI.

    When an OCRClient is provided, PDFs are processed via GLM-OCR for
    high-quality text + table extraction. Falls back to pypdf silently.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        ocr_client: "OCRClient | None" = None,
        semantic_embeddings: "LlamaServerEmbeddings | None" = None,
    ):
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_client = ocr_client
        self._semantic_chunker = None

        # Initialize chonkie SemanticChunker if semantic embeddings are provided
        if semantic_embeddings is not None and semantic_embeddings.dimension > 0:
            try:
                from chonkie import SemanticChunker
                from ..inference.config import (
                    SEMANTIC_CHUNKING_MAX_TOKENS,
                    SEMANTIC_CHUNKING_MIN_SENTENCES,
                    SEMANTIC_CHUNKING_THRESHOLD,
                )

                self._semantic_chunker = SemanticChunker(
                    embedding_model=semantic_embeddings,
                    threshold=SEMANTIC_CHUNKING_THRESHOLD,
                    chunk_size=SEMANTIC_CHUNKING_MAX_TOKENS,
                    min_sentences_per_chunk=SEMANTIC_CHUNKING_MIN_SENTENCES,
                    min_characters_per_sentence=12,
                    # Use paragraph boundaries (\n\n) as the primary delimiter
                    # so blank-line-separated blocks (e.g. conference sessions,
                    # structured records) stay atomic. Sentence-end delimiters
                    # split within long prose paragraphs.
                    delim=["\n\n", ". ", "! ", "? "],
                )
                logger.info(
                    "Semantic chunking enabled (threshold=%.2f, max_tokens=%d)",
                    SEMANTIC_CHUNKING_THRESHOLD,
                    SEMANTIC_CHUNKING_MAX_TOKENS,
                )
            except Exception as exc:
                logger.warning("Semantic chunking init failed, using fixed-size: %s", exc)
                self._semantic_chunker = None

    # ------------------------------------------------------------------
    # Main pipeline (async generator for SSE)
    # ------------------------------------------------------------------

    async def process_file(
        self, filename: str, content: bytes,
    ) -> AsyncIterator[ProcessingEvent]:
        """
        Full pipeline: parse → [ocr_extraction] → chunk → embed → index.

        Yields ProcessingEvent at each stage so the frontend can show
        real-time progress via Server-Sent Events.

        OCR extraction is only attempted for PDFs when an OCR client is
        available. For TXT/MD files, OCR is skipped entirely.
        """
        suffix = Path(filename).suffix.lower()
        stem = Path(filename).stem
        is_pdf = suffix == ".pdf"

        # Clean replace: delete any existing chunks from a prior upload of the
        # same file. Uses the slugified document_id to match.
        doc_id = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
        try:
            deleted = await self.vector_store.delete_by_document_id(doc_id)
            if deleted:
                logger.info("Replaced %d existing chunks for document_id=%s", deleted, doc_id)
        except Exception:
            pass  # collection may not support where filter yet, or doc_id doesn't exist

        # Stage 1: Parse (pypdf or plain text — fast baseline)
        t0 = time.perf_counter()
        yield ProcessingEvent(
            stage="parsing",
            message=f"Extracting text from {filename}...",
            detail={"filename": filename, "size_bytes": len(content)},
        )

        if is_pdf:
            pypdf_pages = self._parse_pdf(content, filename)
        else:
            pypdf_pages = self._parse_text(content, filename)

        parse_ms = (time.perf_counter() - t0) * 1000

        # Stage 1b: Smart OCR — only OCR pages where pypdf produces poor output.
        # pypdf handles text-based PDFs well (~1s for 250 pages). OCR is slow
        # (~7s/page on Metal) but essential for scanned/image-based pages and
        # pages with tables that pypdf garbles. The hybrid approach: use pypdf
        # text when it's good (>= MIN_PYPDF_CHARS), OCR only the rest.
        ocr_ms = 0.0
        pages = pypdf_pages  # default: use pypdf result

        if is_pdf and self.ocr_client is not None:
            # _smart_ocr_stream yields per-page progress events, then a final result
            async for item in self._smart_ocr_stream(content, pypdf_pages):
                if isinstance(item, ProcessingEvent):
                    yield item
                else:
                    # Final result tuple: (pages, ocr_ms, ocr_detail)
                    pages, ocr_ms, _ = item

        total_chars = sum(len(text) for _, text in pages)

        # Stage 2: Chunk
        t1 = time.perf_counter()
        yield ProcessingEvent(
            stage="chunking",
            message=f"Chunking {len(pages)} page(s), {total_chars:,} chars...",
            detail={
                "pages": len(pages),
                "total_chars": total_chars,
                "parse_ms": round(parse_ms, 1),
            },
        )

        all_chunks: list[TextChunk] = []
        for page_num, text in pages:
            if text.strip():
                # Prefer semantic chunking; fall back to table-aware fixed-size
                chunks = self._chunk_semantic(text, filename, page_num)
                if chunks is None:
                    chunks = self._chunk_with_tables(text, filename, page_num)
                all_chunks.extend(chunks)

        chunk_ms = (time.perf_counter() - t1) * 1000

        if not all_chunks:
            yield ProcessingEvent(
                stage="error",
                message="No text content found in document.",
                detail={},
            )
            return

        # Stage 3: Embed + Index (in batches for progress reporting)
        batch_size = 10
        total = len(all_chunks)

        # Stable document_id for metadata filtering and per-document deletion
        doc_id = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")

        documents = [
            Document(
                id=f"upload_{stem}_p{c.page}_c{c.chunk_index}",
                content=c.text,
                metadata={
                    "title": filename,
                    "document_id": doc_id,
                    "source": "upload",
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "type": c.chunk_type,
                },
            )
            for c in all_chunks
        ]

        t2 = time.perf_counter()
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start : batch_start + batch_size]
            progress = min(1.0, (batch_start + len(batch)) / len(documents))

            yield ProcessingEvent(
                stage="embedding",
                message=f"Embedding chunks {batch_start + 1}\u2013{batch_start + len(batch)} of {total}...",
                detail={
                    "progress": round(progress, 3),
                    "current": batch_start + len(batch),
                    "total": total,
                },
            )

            await self.vector_store.add_documents(batch)

        embed_ms = (time.perf_counter() - t2) * 1000
        total_count = await self.vector_store.count()
        total_ms = parse_ms + ocr_ms + chunk_ms + embed_ms

        # Stage 4: Done
        detail: dict = {
            "total_chunks": total,
            "total_documents": total_count,
            "parse_ms": round(parse_ms, 1),
            "chunk_ms": round(chunk_ms, 1),
            "embed_ms": round(embed_ms, 1),
            "total_ms": round(total_ms, 1),
        }
        if ocr_ms > 0:
            detail["ocr_ms"] = round(ocr_ms, 1)

        yield ProcessingEvent(
            stage="indexed",
            message=f"{total} chunks indexed in {round(total_ms)}ms",
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_pdf(
        self, content: bytes, filename: str,
    ) -> list[tuple[int, str]]:
        """Extract (page_num, text) pairs from PDF bytes.

        Strips repeated headers/footers that PDF generators add to every
        page — these waste chunk space and confuse retrieval.
        """
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))

        # Detect repeated header/footer by comparing first 3 pages
        raw_pages = []
        for page in reader.pages:
            raw_pages.append(page.extract_text() or "")

        header = self._detect_repeated_header(raw_pages)

        pages: list[tuple[int, str]] = []
        for page_num, text in enumerate(raw_pages, start=1):
            # Strip detected header from each page
            if header and text.startswith(header):
                text = text[len(header):]
            # Also strip "Page N" markers
            text = re.sub(r"^Page\s+\d+\s*\n?", "", text.strip())
            if text.strip():
                pages.append((page_num, text))
        return pages

    @staticmethod
    def _detect_repeated_header(pages: list[str], min_pages: int = 3) -> str:
        """Find common prefix string repeated across pages (header/footer)."""
        if len(pages) < min_pages:
            return ""
        # Compare first few pages line-by-line to find common prefix
        lines_per_page = [p.split("\n") for p in pages[:min(5, len(pages))]]
        common_lines: list[str] = []
        for line_idx in range(min(len(lp) for lp in lines_per_page)):
            candidates = [lp[line_idx] for lp in lines_per_page]
            if len(set(candidates)) == 1 and candidates[0].strip():
                common_lines.append(candidates[0])
            else:
                break
        if common_lines:
            return "\n".join(common_lines) + "\n"
        return ""

    def _parse_text(
        self, content: bytes, filename: str,
    ) -> list[tuple[int, str]]:
        """Parse TXT/MD as a single page."""
        text = content.decode("utf-8", errors="replace")
        return [(1, text)]

    # ------------------------------------------------------------------
    # Smart OCR — hybrid pypdf + GLM-OCR
    # ------------------------------------------------------------------

    # Minimum chars from pypdf for a page to be considered "good enough".
    # Pages below this threshold are sent to OCR.
    _MIN_PYPDF_CHARS = 100

    # Ratio of non-alphanumeric chars that signals garbled pypdf output.
    # Tables extracted by pypdf often produce runs of spaces/pipes/dashes.
    _MAX_GARBLE_RATIO = 0.5

    @staticmethod
    def _is_poor_pypdf(text: str) -> bool:
        """Check if pypdf output is too short or garbled to be useful.

        Returns True if the page should be sent to OCR instead.
        """
        stripped = text.strip()
        if len(stripped) < DocumentProcessor._MIN_PYPDF_CHARS:
            return True
        # Check for garbled output: high ratio of non-alphanumeric chars
        alnum = sum(c.isalnum() or c.isspace() for c in stripped)
        if alnum / len(stripped) < (1 - DocumentProcessor._MAX_GARBLE_RATIO):
            return True
        return False

    async def _smart_ocr_stream(
        self,
        pdf_bytes: bytes,
        pypdf_pages: list[tuple[int, str]],
    ) -> AsyncIterator[ProcessingEvent | tuple]:
        """Streaming version of _smart_ocr that yields per-page progress events.

        Yields:
          - ProcessingEvent for each OCR page ("OCR: page 3/14...")
          - ProcessingEvent with final OCR summary
          - A tuple (pages, ocr_ms, detail) as the final item
        """
        import time as _time

        pypdf_map: dict[int, str] = {pn: text for pn, text in pypdf_pages}
        needs_ocr: list[int] = []
        for pn, text in pypdf_pages:
            if self._is_poor_pypdf(text):
                needs_ocr.append(pn)

        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pdf_pages = len(doc)
            doc.close()
            pypdf_page_nums = set(pypdf_map.keys())
            for pn in range(1, total_pdf_pages + 1):
                if pn not in pypdf_page_nums and pn not in needs_ocr:
                    needs_ocr.append(pn)
        except ImportError:
            total_pdf_pages = len(pypdf_pages)
        except Exception:
            total_pdf_pages = len(pypdf_pages)

        ocr_ms = 0.0
        ocr_pages_extracted = 0
        ocr_failures = 0

        if not needs_ocr:
            # No OCR needed — emit summary and return
            detail = {
                "total_pages": total_pdf_pages,
                "pypdf_pages": total_pdf_pages,
                "ocr_pages": 0, "ocr_extracted": 0, "ocr_failures": 0,
                "ocr_ms": 0.0, "method": "pypdf-only",
            }
            yield ProcessingEvent(
                stage="ocr_extraction",
                message=f"OCR: not needed ({total_pdf_pages} pages OK via pypdf)",
                detail=detail,
            )
            merged = [(pn, text) for pn, text in sorted(pypdf_map.items()) if text.strip()]
            yield (merged, 0.0, detail)
            return

        # Emit initial OCR stage with page count
        yield ProcessingEvent(
            stage="ocr_extraction",
            message=f"OCR: 0/{len(needs_ocr)} pages (scanning...)",
            detail={
                "total_pages": total_pdf_pages,
                "pypdf_pages": total_pdf_pages - len(needs_ocr),
                "ocr_pages": len(needs_ocr),
                "ocr_extracted": 0, "ocr_failures": 0,
                "ocr_ms": 0.0, "method": "hybrid",
                "current_page": 0,
            },
        )

        if self.ocr_client is not None:
            t_ocr = _time.perf_counter()
            try:
                import fitz
                import base64
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                for page_idx, pn in enumerate(sorted(needs_ocr)):
                    if pn - 1 >= len(doc):
                        continue
                    try:
                        page = doc[pn - 1]
                        text = ""
                        for dpi in (150, 100):
                            pix = page.get_pixmap(dpi=dpi)
                            png_data = pix.tobytes("png")
                            b64 = base64.b64encode(png_data).decode()
                            text = await self.ocr_client.extract_text(b64, mode="text")
                            if text.strip():
                                break
                        if text.strip():
                            pypdf_map[pn] = text
                            ocr_pages_extracted += 1
                        else:
                            ocr_failures += 1
                    except Exception as exc:
                        ocr_failures += 1
                        logger.warning("OCR failed for page %d: %s", pn, exc)

                    # Yield per-page progress
                    elapsed = (_time.perf_counter() - t_ocr) * 1000
                    yield ProcessingEvent(
                        stage="ocr_extraction",
                        message=f"OCR: page {page_idx + 1}/{len(needs_ocr)} (page {pn})...",
                        detail={
                            "total_pages": total_pdf_pages,
                            "pypdf_pages": total_pdf_pages - len(needs_ocr),
                            "ocr_pages": len(needs_ocr),
                            "ocr_extracted": ocr_pages_extracted,
                            "ocr_failures": ocr_failures,
                            "ocr_ms": round(elapsed, 1),
                            "method": "hybrid",
                            "current_page": page_idx + 1,
                        },
                    )
                doc.close()
            except ImportError:
                logger.info("pymupdf not installed — skipping OCR")
            except Exception as exc:
                logger.warning("Smart OCR failed: %s", exc)

            ocr_ms = (_time.perf_counter() - t_ocr) * 1000

        # Final summary
        if ocr_pages_extracted > 0:
            method = "hybrid"
        elif self.ocr_client is None:
            method = "pypdf-only"
        else:
            method = "fallback-pypdf"

        detail = {
            "total_pages": total_pdf_pages,
            "pypdf_pages": total_pdf_pages - len(needs_ocr),
            "ocr_pages": len(needs_ocr),
            "ocr_extracted": ocr_pages_extracted,
            "ocr_failures": ocr_failures,
            "ocr_ms": round(ocr_ms, 1),
            "method": method,
        }
        yield ProcessingEvent(
            stage="ocr_extraction",
            message=(
                f"OCR: {ocr_pages_extracted}/{len(needs_ocr)} pages extracted ({method}, {ocr_ms:.0f}ms)"
                if ocr_ms > 0 else f"OCR: {method}"
            ),
            detail=detail,
        )

        merged = [(pn, text) for pn, text in sorted(pypdf_map.items()) if text.strip()]
        yield (merged, ocr_ms, detail)

    # ------------------------------------------------------------------
    # Table-aware chunking
    # ------------------------------------------------------------------

    def _chunk_with_tables(
        self, text: str, source: str, page: int,
    ) -> list[TextChunk]:
        """Chunk text with Markdown table awareness.

        If the text contains Markdown tables (from OCR output), tables are
        kept as atomic chunks (never split across boundaries) with
        chunk_type="table". Prose sections use the standard overlapping
        chunker.

        Falls through to _chunk_text() if no Markdown tables are detected.
        """
        # Fast path: no tables → standard chunking
        if not _TABLE_SEP_RE.search(text):
            return self._chunk_text(text, source, page)

        # Split into alternating prose/table segments
        segments = self._split_tables(text)
        chunks: list[TextChunk] = []
        chunk_num = 0

        for segment_text, is_table in segments:
            segment_text = segment_text.strip()
            if not segment_text:
                continue

            if is_table:
                # Tables are atomic — one chunk, never split
                # Allow up to 2x chunk_size for tables; log a warning if larger
                if len(segment_text) > self.chunk_size * 2:
                    logger.warning(
                        "Large table (%d chars) exceeds 2x chunk_size (%d) — "
                        "keeping atomic but retrieval quality may degrade",
                        len(segment_text), self.chunk_size,
                    )
                if len(segment_text) >= 50:  # skip tiny table fragments
                    chunks.append(TextChunk(
                        text=segment_text,
                        source=source,
                        page=page,
                        chunk_index=chunk_num,
                        chunk_type="table",
                    ))
                    chunk_num += 1
            else:
                # Prose: use standard chunker
                prose_chunks = self._chunk_text(segment_text, source, page)
                for c in prose_chunks:
                    chunks.append(TextChunk(
                        text=c.text,
                        source=source,
                        page=page,
                        chunk_index=chunk_num,
                        chunk_type="text",
                    ))
                    chunk_num += 1

        return chunks

    @staticmethod
    def _split_tables(text: str) -> list[tuple[str, bool]]:
        """Split text into alternating (content, is_table) segments.

        A "table" is a contiguous block of lines where each line starts
        with | and contains at least one more |.
        """
        lines = text.split("\n")
        segments: list[tuple[str, bool]] = []
        current_lines: list[str] = []
        in_table = False

        for line in lines:
            line_is_table = bool(_TABLE_LINE_RE.match(line))

            if line_is_table != in_table:
                # State change — flush current segment
                if current_lines:
                    segments.append(("\n".join(current_lines), in_table))
                    current_lines = []
                in_table = line_is_table

            current_lines.append(line)

        # Flush final segment
        if current_lines:
            segments.append(("\n".join(current_lines), in_table))

        return segments

    # ------------------------------------------------------------------
    # Semantic chunking (chonkie — embedding-based topic boundaries)
    # ------------------------------------------------------------------

    def _chunk_semantic(
        self, text: str, source: str, page: int,
    ) -> list[TextChunk] | None:
        """Split text using chonkie SemanticChunker.

        Returns None if semantic chunking is unavailable or fails,
        signalling the caller to fall back to fixed-size chunking.

        Skips semantic chunking when text contains Markdown tables
        (from OCR output), deferring to _chunk_with_tables() which
        keeps tables as atomic chunks.
        """
        if self._semantic_chunker is None:
            return None

        # Markdown tables need table-aware chunking — skip semantic path
        if _TABLE_SEP_RE.search(text):
            return None

        try:
            chunks = self._semantic_chunker.chunk(text)
        except Exception as exc:
            logger.warning("Semantic chunking failed, falling back to fixed-size: %s", exc)
            return None

        if not chunks:
            return None

        result: list[TextChunk] = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.text.strip()
            if chunk_text and len(chunk_text) >= 50:
                result.append(TextChunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_index=i,
                ))
        return result if result else None

    # ------------------------------------------------------------------
    # Standard chunking (ported from local-doc-assistant, simplified)
    # ------------------------------------------------------------------

    def _chunk_text(
        self, text: str, source: str, page: int,
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks with sentence-boundary awareness.

        Ported from local-and-edge-ai/samples/local-doc-assistant, simplified
        (no boilerplate filtering — that was domain-specific to medical papers).
        """
        text = self._clean_text(text)
        if not text:
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_num = 0
        prev_start = -1

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                end = self._find_break_point(text, start, end)
            else:
                end = len(text)

            chunk_text = text[start:end].strip()
            # Skip tiny fragments (< 100 chars) -- they add noise to retrieval
            if chunk_text and len(chunk_text) >= 100:
                chunks.append(TextChunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_index=chunk_num,
                ))
                chunk_num += 1

            # Move to next chunk with overlap
            next_start = end - self.chunk_overlap
            if next_start <= prev_start:
                next_start = end  # avoid infinite loop
            prev_start = start
            start = next_start

        return chunks

    def _clean_text(self, text: str) -> str:
        """Basic text cleanup: collapse whitespace, remove noise."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.\-_=]{10,}", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """
        Find a good break point near the target end position.

        Tries sentence boundaries first, then paragraph boundaries,
        then word boundaries.
        """
        search_start = max(start, end - 100)
        search_text = text[search_start : end + 50]

        # Try sentence boundary (.!? followed by space)
        sentence_ends = list(re.finditer(r"[.!?]\s+", search_text))
        if sentence_ends:
            for match in reversed(sentence_ends):
                pos = search_start + match.end()
                if pos <= end + 20:
                    return pos

        # Try paragraph boundary
        para_ends = list(re.finditer(r"\n\s*\n", search_text))
        if para_ends:
            for match in reversed(para_ends):
                pos = search_start + match.end()
                if pos <= end + 20:
                    return pos

        # Fall back to word boundary
        if end < len(text):
            space_pos = text.find(" ", end)
            if space_pos != -1 and space_pos < end + 50:
                return space_pos

        return end
