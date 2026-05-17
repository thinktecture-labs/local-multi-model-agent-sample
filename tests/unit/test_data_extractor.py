"""
Unit tests for the DataExtractor and parse_extraction_json.
"""

import json
import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.engine.knowledge.data_extractor import (
    DataExtractor,
    ExtractionResult,
    parse_extraction_json,
    store_competitor,
)


# ---------------------------------------------------------------------------
# parse_extraction_json
# ---------------------------------------------------------------------------

class TestParseExtractionJson:

    def test_clean_json(self):
        raw = '{"company": "Snowflake", "fiscal_year": 2025, "revenue": 3500000000}'
        result = parse_extraction_json(raw)
        assert result["company"] == "Snowflake"
        assert result["fiscal_year"] == 2025
        assert result["revenue"] == 3500000000

    def test_json_with_code_fences(self):
        raw = '```json\n{"company": "Acme", "fiscal_year": 2024}\n```'
        result = parse_extraction_json(raw)
        assert result["company"] == "Acme"
        assert result["fiscal_year"] == 2024

    def test_json_with_bare_fences(self):
        raw = '```\n{"company": "Test", "fiscal_year": 2023}\n```'
        result = parse_extraction_json(raw)
        assert result["company"] == "Test"

    def test_json_embedded_in_text(self):
        raw = 'Here is the extracted data:\n{"company": "Embedded", "fiscal_year": 2025}\nEnd of extraction.'
        result = parse_extraction_json(raw)
        assert result["company"] == "Embedded"

    def test_null_fields(self):
        raw = '{"company": "Partial", "fiscal_year": 2025, "revenue": null, "nrr": null}'
        result = parse_extraction_json(raw)
        assert result["company"] == "Partial"
        assert result["revenue"] is None
        assert result["nrr"] is None

    def test_empty_string_returns_none(self):
        assert parse_extraction_json("") is None

    def test_plain_text_returns_none(self):
        assert parse_extraction_json("I could not find any financial data.") is None

    def test_malformed_json_returns_none(self):
        assert parse_extraction_json('{"company": "Broken"') is None

    def test_array_extracts_first_object(self):
        """When model wraps output in an array, extract the first object."""
        result = parse_extraction_json('[{"company": "Array"}]')
        assert result["company"] == "Array"

    def test_whitespace_handling(self):
        raw = '  \n  {"company": "Trimmed", "fiscal_year": 2025}  \n  '
        result = parse_extraction_json(raw)
        assert result["company"] == "Trimmed"

    def test_all_fields_present(self):
        raw = json.dumps({
            "company": "Full",
            "fiscal_year": 2025,
            "revenue": 1000000,
            "revenue_growth_pct": 25.5,
            "nrr": 110,
            "customers_1m_plus": 50,
            "total_customers": 500,
            "product_revenue": 800000,
            "gross_margin_pct": 72.0,
            "free_cash_flow": 200000,
        })
        result = parse_extraction_json(raw)
        assert result["revenue"] == 1000000
        assert result["nrr"] == 110
        assert result["gross_margin_pct"] == 72.0


# ---------------------------------------------------------------------------
# store_competitor
# ---------------------------------------------------------------------------

class TestStoreCompetitor:

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, db_path):
        import aiosqlite

        data = {
            "company": "TestCo",
            "fiscal_year": 2025,
            "revenue": 1000000,
            "revenue_growth_pct": 20.0,
            "nrr": 110,
            "customers_1m_plus": 10,
            "total_customers": 100,
            "product_revenue": 800000,
            "gross_margin_pct": 70.0,
            "free_cash_flow": 50000,
            "source_document": "test.pdf",
            "extracted_at": "2026-03-23T00:00:00Z",
        }
        await store_competitor(db_path, data)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM competitors WHERE company = 'TestCo'") as c:
                row = dict(await c.fetchone())
        assert row["company"] == "TestCo"
        assert row["revenue"] == 1000000
        assert row["nrr"] == 110

    @pytest.mark.asyncio
    async def test_upsert_replaces_on_same_company_year(self, db_path):
        import aiosqlite

        base = {
            "company": "UpsertCo",
            "fiscal_year": 2025,
            "revenue": 100,
            "source_document": "v1.pdf",
            "extracted_at": "2026-03-23T00:00:00Z",
        }
        await store_competitor(db_path, base)

        updated = {**base, "revenue": 200, "source_document": "v2.pdf"}
        await store_competitor(db_path, updated)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT COUNT(*) as n FROM competitors WHERE company = 'UpsertCo'") as c:
                count = (await c.fetchone())["n"]
            async with db.execute("SELECT revenue, source_document FROM competitors WHERE company = 'UpsertCo'") as c:
                row = dict(await c.fetchone())
        assert count == 1
        assert row["revenue"] == 200
        assert row["source_document"] == "v2.pdf"

    @pytest.mark.asyncio
    async def test_store_with_null_fields(self, db_path):
        import aiosqlite

        data = {
            "company": "NullCo",
            "fiscal_year": 2025,
            "source_document": "test.pdf",
            "extracted_at": "2026-03-23T00:00:00Z",
        }
        await store_competitor(db_path, data)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM competitors WHERE company = 'NullCo'") as c:
                row = dict(await c.fetchone())
        assert row["revenue"] is None
        assert row["nrr"] is None


# ---------------------------------------------------------------------------
# DataExtractor (with mocked LLM)
# ---------------------------------------------------------------------------

class TestDataExtractor:

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "extract.db")

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.generate_synthesis = AsyncMock()
        return client

    def _make_llm_response(self, content):
        resp = MagicMock()
        resp.content = content
        return resp

    @pytest.mark.asyncio
    async def test_successful_extraction(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            '{"company": "TestCorp", "fiscal_year": 2025, "revenue": 5000000, "nrr": 115}'
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("Some financial text", "test.pdf")

        assert result.success is True
        assert result.stored is True
        assert result.extracted["company"] == "TestCorp"
        assert result.extracted["revenue"] == 5000000

    @pytest.mark.asyncio
    async def test_missing_company_fails(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            '{"fiscal_year": 2025, "revenue": 100}'
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "test.pdf")

        assert result.success is False
        assert "company" in result.error

    @pytest.mark.asyncio
    async def test_missing_fiscal_year_fails(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            '{"company": "NoCo"}'
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "test.pdf")

        assert result.success is False
        assert "fiscal_year" in result.error

    @pytest.mark.asyncio
    async def test_unparseable_output_fails(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            "I cannot extract any financial data from this text."
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "test.pdf")

        assert result.success is False
        assert "parse" in result.error.lower()

    @pytest.mark.asyncio
    async def test_llm_error_handled(self, mock_client, db_path):
        mock_client.generate_synthesis.side_effect = RuntimeError("model crashed")
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "test.pdf")

        assert result.success is False
        assert "model crashed" in result.error

    @pytest.mark.asyncio
    async def test_text_truncation(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            '{"company": "Trunc", "fiscal_year": 2025}'
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        long_text = "x" * 10000
        await extractor.extract(long_text, "test.pdf")

        # Verify the text was truncated before sending to LLM
        call_args = mock_client.generate_synthesis.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "[...truncated...]" in user_msg
        assert len(user_msg) < 10000

    @pytest.mark.asyncio
    async def test_raw_output_preserved(self, mock_client, db_path):
        raw = '{"company": "Raw", "fiscal_year": 2025}'
        mock_client.generate_synthesis.return_value = self._make_llm_response(raw)
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "test.pdf")

        assert result.raw_output == raw

    @pytest.mark.asyncio
    async def test_code_fenced_output_parsed(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            '```json\n{"company": "Fenced", "fiscal_year": 2025, "revenue": 999}\n```'
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "test.pdf")

        assert result.success is True
        assert result.extracted["company"] == "Fenced"
        assert result.extracted["revenue"] == 999

    @pytest.mark.asyncio
    async def test_source_document_and_timestamp_added(self, mock_client, db_path):
        mock_client.generate_synthesis.return_value = self._make_llm_response(
            '{"company": "Meta", "fiscal_year": 2025}'
        )
        extractor = DataExtractor(client=mock_client, db_path=db_path)
        result = await extractor.extract("text", "my-report.pdf")

        assert result.extracted["source_document"] == "my-report.pdf"
        assert "extracted_at" in result.extracted
