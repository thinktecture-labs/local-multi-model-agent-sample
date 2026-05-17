# Structured Data Extraction

> **Status**: Implemented (2026-03-23) on `feature/structured-extraction`
> **Scope**: Demo/keynote — financial metrics from earnings reports and annual reports
> **Model**: gemma3-4B (same model used for RAG synthesis and vision)

## What It Does

Extracts structured financial metrics from unstructured PDF documents and stores them in a SQL table. This enables cross-source queries: "How does our revenue growth compare to Snowflake's?" joins Nextera's internal `sales` table with the extracted `competitors` table.

**Pipeline:**
```
Upload PDF → OCR (GLM-OCR) → chunks (ChromaDB) → DataExtractor (4B) → JSON → competitors table
                                                                          ↓
                                            Query: "Compare our growth to Snowflake"
                                                                          ↓
                                            Qwen SQL: SELECT ... FROM sales JOIN competitors
```

## Use Cases

### Primary: Competitive Benchmarking (Keynote Demo)

1. Upload Snowflake's annual report (250 pages, OCR extracts text)
2. Trigger extraction → model returns structured JSON with revenue, NRR, customer counts
3. Data stored in `competitors` table alongside Nextera's `sales` data
4. Agent answers cross-source questions via SQL JOINs

**Demo beat**: *"We uploaded a competitor's annual report. The agent extracted the key metrics into structured data. Now watch — the agent joins our internal sales with the competitor's extracted data to answer: How does our growth compare?"*

### Secondary: Document-to-Database Pipeline

Any financial document (earnings press release, quarterly report, investor presentation) can be processed. The extraction is scoped to SaaS/financial metrics for the demo, but the architecture generalizes to any schema.

### Not In Scope (Current Implementation)

- **Industry benchmark extraction** (e.g. aggregate B2B SaaS benchmark reports) — different schema needed (metric/segment/percentile vs company/year/revenue). Those reports remain in the OCR + document chat path only.
- **Non-financial documents** — medical records, legal contracts, etc. would require different extraction schemas and prompts.
- **Schema auto-detection** — the fields to extract are hardcoded in the prompt. A future version could infer the schema from the document type.

## Architecture

### Components

| Component | File | Role |
|-----------|------|------|
| `DataExtractor` | `src/engine/knowledge/data_extractor.py` | Service class: calls LLM, parses JSON, stores in DB |
| `ExtractionResult` | `src/engine/knowledge/data_extractor.py` | Dataclass with success, extracted, raw_output, stored, error |
| `parse_extraction_json()` | `src/engine/knowledge/data_extractor.py` | Handles code fences, embedded JSON, stray text |
| `store_competitor()` | `src/engine/knowledge/data_extractor.py` | INSERT OR REPLACE into competitors table |
| `EXTRACTION_SYSTEM_PROMPT` | `src/engine/inference/prompts.py` | Field definitions, scaling rules, output format |
| `EXTRACTION_USER_TEMPLATE` | `src/engine/inference/prompts.py` | Template with `{source_document}` and `{text}` |
| `POST /extract` | `src/server/agent_routes.py` | HTTP endpoint: reads uploads, calls extractor, returns result |
| `GET /competitors` | `src/server/agent_routes.py` | HTTP endpoint: lists all extracted competitor data |

### Design Decisions

**DataExtractor is NOT a tool.** It's part of the document ingestion pipeline (alongside `DocumentProcessor` and `OCRClient`), not the query pipeline. Qwen doesn't select it during tool routing — it's triggered explicitly by the user or the upload flow. This is why it lives in `src/engine/knowledge/`, not `src/engine/tools/`.

**Source filename passed as context.** The LLM prompt includes the filename so the model can infer the company name even when the text passage doesn't mention it. Example: "Fiscal Year 2025 Business Highlights" with no company name → model reads `snowflake-fy2025-annual-report.pdf` → extracts `company: "Snowflake"`.

**Revenue scaling instruction.** Small models are bad at unit conversion. The prompt explicitly says: "EUR311,500 means 311500, not 311500000. $3,626,396 thousands means 3626396000." This fixed a 1000x scaling error on the Nextera report.

**Text truncation.** Input text is truncated to 6,000 chars. Financial highlights are typically in the first few pages of earnings reports. The 4B model's context window (8K tokens) can't handle a full 250-page report anyway. The top-10 chunks from vector search provide the densest financial content.

**UNIQUE(company, fiscal_year) with INSERT OR REPLACE.** Re-extracting the same document overwrites the previous extraction. No duplicate rows.

## Database Schema

```sql
CREATE TABLE competitors (
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
);
```

The SQL tool's schema description (`sql_query.py`) includes this table so Qwen can generate queries against it.

## Extraction Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `company` | TEXT | Company name (inferred from filename if needed) | "Snowflake" |
| `fiscal_year` | INTEGER | Fiscal year | 2025 |
| `revenue` | REAL | Total revenue, raw number | 3626396000 |
| `revenue_growth_pct` | REAL | YoY revenue growth % | 30.0 |
| `nrr` | REAL | Net revenue retention % | 126 |
| `customers_1m_plus` | INTEGER | Customers >$1M trailing revenue | 580 |
| `total_customers` | INTEGER | Total customer count | 745 |
| `product_revenue` | REAL | Product-specific revenue | 3462422000 |
| `gross_margin_pct` | REAL | Gross margin % | 66.0 |
| `free_cash_flow` | REAL | Free cash flow (negative = burn) | 884100000 |
| `source_document` | TEXT | PDF filename | "snowflake-fy2025.pdf" |
| `extracted_at` | TEXT | ISO 8601 timestamp | "2026-03-23T10:23:35Z" |

## API

### POST /extract

Extract structured data from an uploaded document.

**Request:**
```json
{"document_id": "snowflake-fy2025-annual-report"}
```

**Response:**
```json
{
  "success": true,
  "extracted": {
    "company": "Snowflake",
    "fiscal_year": 2025,
    "revenue": 3626396000,
    "revenue_growth_pct": 30.0,
    "nrr": 126,
    "customers_1m_plus": 580,
    "total_customers": 745,
    "product_revenue": 3462422000,
    "gross_margin_pct": null,
    "free_cash_flow": 884100000,
    "source_document": "snowflake-fy2025-annual-report",
    "extracted_at": "2026-03-23T10:23:35.419430+00:00"
  },
  "raw_output": "{\"company\": \"Snowflake\", ...}",
  "stored": true,
  "error": null,
  "execution_time_ms": 1105.3
}
```

The `raw_output` field contains the exact LLM output before parsing — displayed in the pipeline trace UI as a JSON debug view.

### GET /competitors

List all extracted competitor data.

**Response:**
```json
{
  "competitors": [
    {"company": "Snowflake", "fiscal_year": 2025, "revenue": 3626396000, ...}
  ],
  "count": 1
}
```

## Cross-Source Queries

After extraction, Qwen can generate SQL that joins Nextera's internal data with extracted competitor data:

```sql
-- "How does our revenue growth compare to Snowflake?"
SELECT 'Nextera' as company, SUM(revenue) as total_revenue, MAX(arr_growth_pct) as growth
FROM sales WHERE year = 2024
UNION ALL
SELECT company, revenue, revenue_growth_pct
FROM competitors WHERE fiscal_year = 2025
```

```sql
-- "Which company has more customers?"
SELECT 'Nextera' as company, COUNT(*) as customers FROM customers
UNION ALL
SELECT company, total_customers FROM competitors
```

## Eval Results

**Extraction accuracy** (5 test cases, 29 fields, M5 Max Metal):

| Metric | Value |
|--------|-------|
| Field accuracy | **100%** (29/29) |
| Case accuracy | **100%** (5/5) |
| Mean latency | 1,016ms |

Test cases cover: Snowflake CEO letter (4 fields), Snowflake financial highlights (8 fields), Nextera quarterly (4 fields), synthetic startup (10 fields, including negative FCF), minimal info (3 fields).

Previous version (before prompt fixes): 93.1% (27/29) — two misses:
1. Company name null when text didn't mention it (fixed by passing filename)
2. EUR311,500 scaled to 311,500,000 (fixed by explicit scaling instruction)

**Eval script:** `python -m finetune.eval_extraction --verbose`

## Tests

| Suite | Tests | Coverage |
|-------|-------|----------|
| `tests/unit/test_data_extractor.py` | 23 | JSON parser (12 cases), store_competitor (3 cases), DataExtractor with mocked LLM (8 cases) |
| `tests/integration/test_ocr_upload.py` | 4 | /extract 404, extraction result, raw_output, /competitors listing |
| **Total** | **27** | |

## Developer Guide

### Running Extraction

```python
from src.engine.inference.client import SmallLanguageModelClient
from src.engine.knowledge.data_extractor import DataExtractor

client = SmallLanguageModelClient.create_with_auto_detection()
extractor = DataExtractor(client=client, db_path="./data/business.db")

result = await extractor.extract(
    text="Snowflake FY2025: revenue $3.63B, NRR 126%, 580 customers >$1M",
    source_document="snowflake-fy2025-annual-report.pdf",
)

if result.success:
    print(result.extracted)  # dict with structured fields
    print(result.raw_output) # raw LLM JSON for debug view
```

### Adding New Extraction Fields

1. Add the field to `EXTRACTION_SYSTEM_PROMPT` in `src/engine/inference/prompts.py`
2. Add the column to the `CREATE TABLE competitors` in both `data/loader.py` and `data_extractor.py`
3. Add the column to the `INSERT OR REPLACE` statement in `store_competitor()`
4. Add the field to the SQL tool description in `sql_query.py`
5. Add test cases to `tests/unit/test_data_extractor.py`
6. Add eval cases to `finetune/eval_extraction.py`

### Adding New Document Types

The current schema is hardcoded for financial/SaaS metrics. To support a different document type (e.g., medical records, legal contracts):

1. Define a new extraction prompt in `prompts.py`
2. Create a new table schema (or extend `competitors` with a `doc_type` discriminator)
3. Create a new `DataExtractor` subclass or add a `schema` parameter
4. Add eval cases for the new document type

This is intentionally not implemented — the demo scope is financial documents only.

## Known Limitations

1. **6,000 char context limit.** The extractor truncates input to 6K chars. For documents where key metrics appear deep in the text (page 50+), the top-10 vector search chunks may not include them. Workaround: search with more targeted queries.

2. **Single-company schema.** The `competitors` table assumes one company per row. Industry benchmark reports with aggregate data across many companies don't fit this schema.

3. **No validation against ground truth.** The extractor trusts the LLM output. If the model hallucinates a number, it gets stored. The eval catches this on known documents, but novel documents may have errors.

4. **Currency ambiguity.** The prompt doesn't handle currency conversion. EUR and USD values are stored as-is in the same column. The cross-source query comparing Nextera (EUR) with Snowflake (USD) is comparing different currencies.
