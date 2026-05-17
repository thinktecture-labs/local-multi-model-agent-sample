"""
Sample data loader — seeds the demo with a realistic knowledge base.

The fictional company "Nextera Platform" sells a SaaS AI infrastructure product.
This domain is intentionally broad enough to exercise all three intent types:

  rag_query     → product features, pricing, integrations, support
  tool_use      → SQL queries, calculations (routed via the tool-calling model)
  direct_answer → greetings, capability questions

Run `python -m data.loader` to populate both the vector store and SQLite DB.
"""

import asyncio
import glob
import os
import sys

import aiosqlite

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.knowledge.chunking import chunk_text
from src.engine.knowledge.vector_store import Document, VectorStore
from src.engine.inference.client import SmallLanguageModelClient


# ---------------------------------------------------------------------------
# Knowledge base documents — loaded from data/business-documents/*.md
# ---------------------------------------------------------------------------

_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "business-documents")


def _parse_doc(filepath: str) -> dict:
    """Parse a markdown doc with YAML-style front matter (title, category)."""
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip().strip('"').strip("'")
                    meta[key.strip()] = val
            text = parts[2].strip()

    doc_id = os.path.splitext(os.path.basename(filepath))[0]
    return {
        "id": doc_id,
        "title": meta.get("title", doc_id),
        "content": text,
        "category": meta.get("category", "general"),
    }


def load_knowledge_base(docs_dir: str = _DOCS_DIR) -> list[dict]:
    """Load all .md files from the docs directory."""
    files = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    return [_parse_doc(f) for f in files]


KNOWLEDGE_BASE: list[dict] = load_knowledge_base()


# ---------------------------------------------------------------------------
# SQLite seed data
# ---------------------------------------------------------------------------

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id              INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    price_monthly   REAL    NOT NULL,
    price_annual    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    tier        TEXT    NOT NULL,
    mrr         REAL    NOT NULL,
    joined_date TEXT    NOT NULL,
    industry    TEXT
);

CREATE TABLE IF NOT EXISTS sales (
    id             INTEGER PRIMARY KEY,
    year           INTEGER NOT NULL,
    quarter        TEXT    NOT NULL,
    revenue        REAL    NOT NULL,
    new_customers  INTEGER NOT NULL,
    churn_rate     REAL    NOT NULL,
    arr_growth_pct REAL    NOT NULL
);

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
    extracted_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(company, fiscal_year)
);
"""

SQL_SEED = """
INSERT OR IGNORE INTO products VALUES
    (1, 'Nextera Starter',       'platform', 299.0,  2990.0),
    (2, 'Nextera Professional',  'platform', 999.0,  9990.0),
    (3, 'Nextera Enterprise',    'platform', 3500.0, 35000.0),
    (4, 'Fine-Tuning Add-on',  'addon',    500.0,  5000.0),
    (5, 'GPU Hours (A100)',    'compute',  4.5,    NULL);

INSERT OR IGNORE INTO customers VALUES
    (1,  'Acme Corp',           'enterprise',    3500,  '2023-01-15', 'Manufacturing'),
    (2,  'BrightHealth GmbH',   'enterprise',    7000,  '2023-03-01', 'Healthcare'),
    (3,  'CodeStack Ltd',       'professional',  999,   '2023-05-20', 'Software'),
    (4,  'DataFlow AG',         'professional',  999,   '2023-07-11', 'Analytics'),
    (5,  'EduTech Berlin',      'starter',       299,   '2023-09-05', 'Education'),
    (6,  'FinVault SA',         'enterprise',    5000,  '2024-01-08', 'Finance'),
    (7,  'GreenOps BV',         'professional',  1499,  '2024-02-14', 'Energy'),
    (8,  'Horizon AI',          'enterprise',    4200,  '2024-03-22', 'Technology'),
    (9,  'InsureBase Inc',      'professional',  999,   '2024-04-01', 'Insurance'),
    (10, 'JetLog Systems',      'starter',       299,   '2024-06-18', 'Logistics');

INSERT OR IGNORE INTO sales VALUES
    (1, 2023, 'Q1', 18500,  3, 2.1,  NULL),
    (2, 2023, 'Q2', 24700,  5, 1.8,  33.5),
    (3, 2023, 'Q3', 31200,  4, 1.5,  26.3),
    (4, 2023, 'Q4', 42800,  6, 1.2,  37.2),
    (5, 2024, 'Q1', 55100,  7, 1.0,  28.7),
    (6, 2024, 'Q2', 68300,  8, 0.9,  23.9),
    (7, 2024, 'Q3', 84900,  9, 0.8,  24.3),
    (8, 2024, 'Q4', 103200, 11, 0.7, 21.6);
"""


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

async def seed_vector_store(
    client: SmallLanguageModelClient,
    vector_store: VectorStore,
    force_reload: bool = False,
) -> int:
    """
    Index the knowledge base documents into ChromaDB.

    Skips already-indexed documents unless force_reload=True.
    Returns the number of newly indexed documents.
    """
    vector_store.set_client(client)

    documents_to_add = []
    for item in KNOWLEDGE_BASE:
        doc_id = item["id"]
        if not force_reload and (
            await vector_store.document_exists(doc_id)
            or await vector_store.document_exists(f"{doc_id}_c0")
        ):
            continue

        # Chunk large documents for better retrieval precision and to
        # keep RAG synthesis context within the model's context window.
        # Small documents (<= chunk_size) pass through as a single chunk.
        chunks = chunk_text(item["content"])
        if len(chunks) <= 1:
            # Single chunk — use doc_id directly (backwards compatible)
            documents_to_add.append(Document(
                id=doc_id,
                content=chunks[0] if chunks else item["content"],
                metadata={"title": item["title"], "category": item["category"]},
            ))
        else:
            for i, chunk in enumerate(chunks):
                documents_to_add.append(Document(
                    id=f"{doc_id}_c{i}",
                    content=chunk,
                    metadata={
                        "title": item["title"],
                        "category": item["category"],
                        "document_id": doc_id,
                        "chunk_index": i,
                    },
                ))

    if documents_to_add:
        await vector_store.add_documents(documents_to_add)

    return len(documents_to_add)


async def seed_sql_database(db_path: str = "./data/business.db") -> None:
    """
    Create and seed the SQLite demo database.

    Safe to call multiple times — drops and recreates all tables to prevent
    duplicate rows in AUTOINCREMENT tables (competitors) that INSERT OR IGNORE
    cannot deduplicate.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        for table in ("competitors", "sales", "customers", "products"):
            await db.execute(f"DROP TABLE IF EXISTS {table}")
        await db.executescript(SQL_SCHEMA)
        await db.executescript(SQL_SEED)
        await db.commit()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    print("Seeding Nextera Platform demo data...")

    from src.engine.inference.config import SCENARIO_CONFIG
    client = SmallLanguageModelClient()
    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)

    print("  → Seeding vector store (this embeds documents with embeddinggemma)...")
    n = await seed_vector_store(client, vector_store)
    total = await vector_store.count()
    print(f"  ✓ Vector store: {n} new docs indexed, {total} total")

    print("  → Seeding SQLite database...")
    await seed_sql_database()
    print("  ✓ SQLite database seeded at ./data/business.db")

    print("\nDemo data ready. Run `python demo.py` to start.")


if __name__ == "__main__":
    asyncio.run(_main())
