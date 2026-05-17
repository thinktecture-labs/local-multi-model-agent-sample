"""
DataExtractor — Extract structured competitor data from document text.

Part of the document ingestion pipeline (alongside DocumentProcessor and
OCRClient). Not a query-time tool — triggered after upload when the user
requests data extraction.

Uses gemma3-4B to extract financial metrics from unstructured text into
structured fields, then stores in the competitors SQL table.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from ..inference.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE

logger = logging.getLogger(__name__)

# Financial highlights are typically in the first few pages.
_MAX_EXTRACTION_CHARS = 6000


@dataclass
class ExtractionResult:
    """Result of a data extraction attempt."""
    success: bool
    extracted: Optional[dict] = None
    raw_output: Optional[str] = None
    stored: bool = False
    error: Optional[str] = None


class DataExtractor:
    """
    Extract structured financial data from document text and store it.

    Usage:
        extractor = DataExtractor(client=slm_client, db_path="./data/business.db")
        result = await extractor.extract("...document text...", "snowflake-fy2025.pdf")
    """

    def __init__(self, client, db_path: str = "./data/business.db") -> None:
        self._client = client
        self.db_path = db_path

    async def extract(self, text: str, source_document: str) -> ExtractionResult:
        """Extract financial metrics from text, parse JSON, store in DB."""
        if len(text) > _MAX_EXTRACTION_CHARS:
            text = text[:_MAX_EXTRACTION_CHARS] + "\n\n[...truncated...]"

        # Call 4B model
        try:
            response = await self._client.generate_synthesis(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(
                        text=text, source_document=source_document)},
                ],
            )
            raw_output = response.content.strip()
        except Exception as exc:
            logger.error(f"Extraction LLM call failed: {exc}")
            return ExtractionResult(success=False, error=f"LLM extraction failed: {exc}")

        # Parse JSON
        extracted = parse_extraction_json(raw_output)
        if extracted is None:
            return ExtractionResult(
                success=False, raw_output=raw_output,
                error="Failed to parse JSON from model output",
            )

        # Validate required fields
        if not extracted.get("company"):
            return ExtractionResult(
                success=False, extracted=extracted, raw_output=raw_output,
                error="Missing required field: company",
            )
        if not extracted.get("fiscal_year"):
            return ExtractionResult(
                success=False, extracted=extracted, raw_output=raw_output,
                error="Missing required field: fiscal_year",
            )

        # Store
        extracted["source_document"] = source_document
        extracted["extracted_at"] = datetime.now(timezone.utc).isoformat()

        try:
            await store_competitor(self.db_path, extracted)
        except Exception as exc:
            logger.error(f"Failed to store extraction: {exc}")
            return ExtractionResult(
                success=False, extracted=extracted, raw_output=raw_output,
                error=f"Database storage failed: {exc}",
            )

        return ExtractionResult(
            success=True,
            extracted=extracted,
            raw_output=raw_output,
            stored=True,
        )


def parse_extraction_json(raw: str) -> dict | None:
    """Parse JSON from LLM output, handling code fences and stray text."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


async def store_competitor(db_path: str, data: dict) -> None:
    """Insert or replace competitor data in the database."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS competitors (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                company             TEXT    NOT NULL,
                fiscal_year         INTEGER NOT NULL,
                revenue             REAL,
                revenue_growth_pct  REAL,
                nrr                 REAL,
                customers_1m_plus   INTEGER,
                total_customers     INTEGER,
                product_revenue     REAL,
                gross_margin_pct    REAL,
                free_cash_flow      REAL,
                source_document     TEXT    NOT NULL,
                extracted_at        TEXT    NOT NULL,
                UNIQUE(company, fiscal_year)
            )
        """)

        await db.execute("""
            INSERT OR REPLACE INTO competitors
                (company, fiscal_year, revenue, revenue_growth_pct, nrr,
                 customers_1m_plus, total_customers, product_revenue,
                 gross_margin_pct, free_cash_flow, source_document, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["company"],
            data["fiscal_year"],
            data.get("revenue"),
            data.get("revenue_growth_pct"),
            data.get("nrr"),
            data.get("customers_1m_plus"),
            data.get("total_customers"),
            data.get("product_revenue"),
            data.get("gross_margin_pct"),
            data.get("free_cash_flow"),
            data["source_document"],
            data["extracted_at"],
        ))
        await db.commit()
