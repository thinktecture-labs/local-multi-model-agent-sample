"""Shared text chunking utilities for seed loaders and DocumentProcessor.

Provides fixed-size chunking with sentence-boundary awareness.
Used by:
  - data/loader.py for seed-time chunking
  - DocumentProcessor as fallback when semantic chunking is unavailable

Semantic chunking (chonkie SemanticChunker) is preferred for uploaded
documents but requires a running embeddinggemma server. Seed loaders
run at startup when the server may still be loading, so they use the
simpler fixed-size approach here.
"""

import re

# Defaults match DocumentProcessor (chunk_size=800, overlap=80)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80
MIN_CHUNK_CHARS = 100  # Skip fragments shorter than this


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split text into overlapping chunks with sentence-boundary awareness.

    Returns a list of chunk strings. If the text is shorter than
    *chunk_size*, returns the original text as a single-element list
    (no splitting needed).

    This is the same algorithm as DocumentProcessor._chunk_text, extracted
    so seed loaders can reuse it without importing the full processor.
    """
    text = _clean_text(text)
    if not text:
        return []

    # Short texts don't need splitting
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    prev_start = -1

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            end = _find_break_point(text, start, end)
        else:
            end = len(text)

        chunk_text_str = text[start:end].strip()
        if chunk_text_str and len(chunk_text_str) >= min_chars:
            chunks.append(chunk_text_str)

        # Move to next chunk with overlap
        next_start = end - chunk_overlap
        if next_start <= prev_start:
            next_start = end  # avoid infinite loop
        prev_start = start
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers (ported from DocumentProcessor)
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Basic text cleanup: collapse whitespace, remove noise."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.\-_=]{10,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_break_point(text: str, start: int, end: int) -> int:
    """Find a good break point near the target end position.

    Tries sentence boundaries first, then paragraph boundaries,
    then word boundaries.
    """
    search_start = max(start, end - 100)
    search_text = text[search_start: end + 50]

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
