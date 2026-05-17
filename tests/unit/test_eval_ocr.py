"""
Unit tests for the OCR evaluation test set integrity.

Validates:
  - No duplicate queries
  - All entries have required fields
  - Categories are valid
  - Expected keywords are non-empty
  - Keyword checker works correctly
"""

from pathlib import Path

import pytest

from finetune.eval_ocr import TEST_SET, CATEGORIES, check_keywords


DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "demo-documents"


@pytest.mark.unit
class TestEvalOCRTestSet:

    def test_no_duplicate_queries(self):
        """Every query string is unique within the test set."""
        queries = [item["query"] for item in TEST_SET]
        assert len(queries) == len(set(queries)), (
            f"Duplicate queries found: "
            f"{[q for q in queries if queries.count(q) > 1]}"
        )

    def test_all_entries_have_required_fields(self):
        """Every entry has: document, query, expected_keywords, category."""
        required = {"document", "query", "expected_keywords", "category"}
        for i, item in enumerate(TEST_SET):
            missing = required - set(item.keys())
            assert not missing, f"Entry {i} missing fields: {missing}"

    def test_categories_are_valid(self):
        """All categories are in the allowed set."""
        valid = set(CATEGORIES)
        for i, item in enumerate(TEST_SET):
            assert item["category"] in valid, (
                f"Entry {i} has invalid category: {item['category']}"
            )

    def test_expected_keywords_are_nonempty(self):
        """Every entry has at least 1 expected keyword."""
        for i, item in enumerate(TEST_SET):
            assert len(item["expected_keywords"]) > 0, (
                f"Entry {i} has empty expected_keywords"
            )

    def test_document_names_are_pdfs(self):
        """All document names end with .pdf."""
        for i, item in enumerate(TEST_SET):
            assert item["document"].endswith(".pdf"), (
                f"Entry {i} document is not a PDF: {item['document']}"
            )

@pytest.mark.unit
class TestKeywordChecker:

    def test_keyword_hit(self):
        assert check_keywords("PostgreSQL 16 was chosen", ["PostgreSQL", "16"]) is True

    def test_keyword_miss(self):
        assert check_keywords("The database is MySQL", ["PostgreSQL"]) is False

    def test_case_insensitive(self):
        assert check_keywords("the answer is POSTGRESQL", ["postgresql"]) is True

    def test_partial_match(self):
        """Keywords match as substrings."""
        assert check_keywords("Revenue was $3.63 billion", ["3.63"]) is True

    def test_any_keyword_sufficient(self):
        """Only one keyword needs to match."""
        assert check_keywords("The answer is 580", ["580", "five hundred"]) is True
