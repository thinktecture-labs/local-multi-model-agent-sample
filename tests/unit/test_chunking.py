"""
Unit tests for src/engine/knowledge/chunking.py.

Verifies chunk sizes, overlap, sentence-boundary awareness, and
short-text passthrough — the same algorithm used by both seed loaders
and (via DocumentProcessor) the upload pipeline.
"""

import pytest

from src.engine.knowledge.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_CHARS,
    chunk_text,
)


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestChunkTextBasic:
    def test_empty_string_returns_empty(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n  ") == []

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should NOT be split."""
        short = "This is a short document about Nextera pricing."
        result = chunk_text(short)
        assert len(result) == 1
        assert result[0] == short

    def test_text_exactly_chunk_size_returns_single_chunk(self):
        text = "A" * DEFAULT_CHUNK_SIZE
        result = chunk_text(text)
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self):
        # 3000 chars should produce 4-5 chunks at 800 chars each
        text = "Dies ist ein langer Text. " * 150  # ~3900 chars
        result = chunk_text(text)
        assert len(result) >= 3
        assert len(result) <= 8

    def test_all_chunks_below_max_size(self):
        """No chunk should exceed chunk_size + break tolerance (50 chars)."""
        text = "Dieser Satz ist lang genug. " * 200  # ~5600 chars
        result = chunk_text(text)
        for i, chunk in enumerate(result):
            assert len(chunk) <= DEFAULT_CHUNK_SIZE + 50, (
                f"Chunk {i} is {len(chunk)} chars, exceeds max "
                f"{DEFAULT_CHUNK_SIZE + 50}"
            )

    def test_all_chunks_above_min_size(self):
        """No chunk should be smaller than MIN_CHUNK_CHARS."""
        text = "Dieser Satz ist ein Beispiel. " * 100
        result = chunk_text(text)
        for i, chunk in enumerate(result):
            assert len(chunk) >= MIN_CHUNK_CHARS, (
                f"Chunk {i} is only {len(chunk)} chars, below min {MIN_CHUNK_CHARS}"
            )


# ---------------------------------------------------------------------------
# Content preservation
# ---------------------------------------------------------------------------

class TestContentPreservation:
    def test_no_content_lost(self):
        """All words from the original text should appear in at least one chunk."""
        text = "Alpha Bravo Charlie Delta Echo Foxtrot. " * 50
        result = chunk_text(text)
        combined = " ".join(result)
        for word in ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]:
            assert word in combined, f"Word '{word}' lost during chunking"

    def test_whitespace_normalized(self):
        """Multiple spaces and newlines should be collapsed."""
        text = "Hello    world.\n\n\nThis   is    a   test.  " * 30
        result = chunk_text(text)
        for chunk in result:
            assert "    " not in chunk
            assert "\n\n" not in chunk


# ---------------------------------------------------------------------------
# Sentence boundary awareness
# ---------------------------------------------------------------------------

class TestSentenceBoundaries:
    def test_prefers_sentence_boundaries(self):
        """Chunks should end at sentence boundaries when possible."""
        # Build a text where sentence boundaries fall near chunk_size
        sentences = [f"Sentence number {i} is here." for i in range(50)]
        text = " ".join(sentences)
        result = chunk_text(text)
        # Most chunks should end with a period or start of next sentence
        ends_at_sentence = sum(1 for c in result if c.rstrip().endswith("."))
        assert ends_at_sentence >= len(result) // 2, (
            f"Only {ends_at_sentence}/{len(result)} chunks end at sentence boundaries"
        )


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

class TestCustomParameters:
    def test_smaller_chunk_size(self):
        text = "Word " * 500  # 2500 chars
        result = chunk_text(text, chunk_size=200, chunk_overlap=20, min_chars=20)
        assert len(result) >= 8  # 2500 / (200-20) = ~14 chunks
        for chunk in result:
            assert len(chunk) <= 250  # 200 + tolerance

    def test_larger_chunk_size(self):
        text = "Word " * 500  # 2500 chars
        result = chunk_text(text, chunk_size=2000)
        assert len(result) >= 1
        assert len(result) <= 3

    def test_zero_overlap(self):
        text = "Abcdef. " * 200  # ~1600 chars
        result = chunk_text(text, chunk_size=400, chunk_overlap=0, min_chars=20)
        assert len(result) >= 3


# ---------------------------------------------------------------------------
# German-language text (chunking should handle non-English correctly)
# ---------------------------------------------------------------------------

class TestGermanText:
    GERMAN_SAMPLE = (
        "Die deutsche Sprache verwendet zusammengesetzte Substantive, die "
        "oft ungewoehnlich lang werden koennen. Beispiele sind "
        "Donaudampfschifffahrtsgesellschaftskapitaen oder "
        "Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz. "
        "Solche Wortbildungen koennen das automatische Chunking erschweren, "
        "weil viele Token-basierte Algorithmen mit kurzen Woertern rechnen. "
        "Ein guter Chunker sollte mit beliebigen Sprachen und Wortlaengen "
        "umgehen koennen, ohne dass die Ergebnisse stark variieren. "
        "Die Pruefung anhand realer Texte ist daher entscheidend. "
    )

    def test_german_text_chunks_correctly(self):
        # Repeat to exceed chunk_size
        text = self.GERMAN_SAMPLE * 10  # ~5000 chars
        result = chunk_text(text)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= DEFAULT_CHUNK_SIZE + 50

    def test_short_german_doc_not_split(self):
        """A single short German document should not be split."""
        assert len(self.GERMAN_SAMPLE) < DEFAULT_CHUNK_SIZE
        result = chunk_text(self.GERMAN_SAMPLE)
        assert len(result) == 1
