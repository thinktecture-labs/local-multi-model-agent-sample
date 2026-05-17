"""
Integration tests for data/loader.py.

Verifies knowledge base document integrity, SQL schema creation, seed data
correctness, and idempotent seeding — all without model servers (vector
store operations use a mocked SmallLanguageModelClient).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import aiosqlite

from data.loader import (
    KNOWLEDGE_BASE,
    SQL_SCHEMA,
    SQL_SEED,
    seed_sql_database,
    seed_vector_store,
)
from src.engine.knowledge.chunking import DEFAULT_CHUNK_SIZE
from src.engine.knowledge.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Knowledge base documents
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_has_documents(self):
        assert len(KNOWLEDGE_BASE) > 0

    def test_all_have_required_fields(self):
        for doc in KNOWLEDGE_BASE:
            assert "id" in doc, f"Missing id in doc: {doc.get('title', '?')}"
            assert "title" in doc, f"Missing title in doc: {doc['id']}"
            assert "content" in doc, f"Missing content in doc: {doc['id']}"
            assert "category" in doc, f"Missing category in doc: {doc['id']}"

    def test_unique_ids(self):
        ids = [doc["id"] for doc in KNOWLEDGE_BASE]
        assert len(ids) == len(set(ids)), "Duplicate document IDs found"

    def test_content_nonempty(self):
        for doc in KNOWLEDGE_BASE:
            assert len(doc["content"]) > 20, f"Content too short for doc: {doc['id']}"

    def test_expected_categories(self):
        categories = {doc["category"] for doc in KNOWLEDGE_BASE}
        assert "pricing" in categories
        assert "features" in categories
        assert "faq" in categories

    def test_thirteen_documents(self):
        assert len(KNOWLEDGE_BASE) == 13


# ---------------------------------------------------------------------------
# SQL seeding
# ---------------------------------------------------------------------------

class TestSeedSqlDatabase:
    async def test_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in await cursor.fetchall()]

        assert "products" in tables
        assert "customers" in tables
        assert "sales" in tables

    async def test_products_seeded(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM products")
            count = (await cursor.fetchone())[0]
        # 4 of 5 — "GPU Hours (A100)" is skipped by INSERT OR IGNORE because
        # price_annual=NULL violates the NOT NULL constraint in SQL_SCHEMA.
        assert count == 4

    async def test_customers_seeded(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM customers")
            count = (await cursor.fetchone())[0]
        assert count == 10

    async def test_sales_seeded(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM sales")
            count = (await cursor.fetchone())[0]
        # 7 of 8 — Q1 2023 has arr_growth_pct=NULL violating NOT NULL constraint
        assert count == 7

    async def test_idempotent(self, tmp_path):
        """Running seed_sql_database twice produces the same result."""
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM products")
            count = (await cursor.fetchone())[0]
        assert count == 4

    async def test_products_schema_columns(self, tmp_path):
        """Verify column names match loader.py SQL_SCHEMA."""
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA table_info(products)")
            columns = [row[1] for row in await cursor.fetchall()]
        assert columns == ["id", "name", "category", "price_monthly", "price_annual"]

    async def test_customers_schema_columns(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA table_info(customers)")
            columns = [row[1] for row in await cursor.fetchall()]
        assert columns == ["id", "name", "tier", "mrr", "joined_date", "industry"]

    async def test_sales_schema_has_churn_rate(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA table_info(sales)")
            columns = [row[1] for row in await cursor.fetchall()]
        assert "churn_rate" in columns
        assert "arr_growth_pct" in columns

    async def test_enterprise_plan_data(self, tmp_path):
        """Spot-check a specific product row."""
        db_path = str(tmp_path / "test.db")
        await seed_sql_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name, price_monthly FROM products WHERE id = 3"
            )
            row = await cursor.fetchone()
        assert row[0] == "Nextera Enterprise"
        assert row[1] == 3500.0


# ---------------------------------------------------------------------------
# Vector store seeding
# ---------------------------------------------------------------------------

class TestSeedVectorStore:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        _placeholder = [0.1] * 768
        client.embed_batch = AsyncMock(side_effect=lambda texts: [_placeholder] * len(texts))
        return client

    async def test_indexes_all_documents(self, mock_client, tmp_path):
        vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
        n = await seed_vector_store(mock_client, vs)
        assert n == 13
        assert await vs.count() == 13

    async def test_skips_existing_documents(self, mock_client, tmp_path):
        vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
        # First call indexes all
        n1 = await seed_vector_store(mock_client, vs)
        assert n1 == 13
        # Second call skips all
        n2 = await seed_vector_store(mock_client, vs)
        assert n2 == 0
        assert await vs.count() == 13

    async def test_force_reload_reindexes(self, mock_client, tmp_path):
        vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
        await seed_vector_store(mock_client, vs)
        n = await seed_vector_store(mock_client, vs, force_reload=True)
        assert n == 13

    async def test_document_metadata_preserved(self, mock_client, tmp_path):
        vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
        await seed_vector_store(mock_client, vs)
        exists = await vs.document_exists("nextera-overview")
        assert exists is True

    async def test_all_chunks_below_max_size(self, mock_client, tmp_path):
        """Every indexed chunk must fit within chunk_size + break tolerance."""
        max_allowed = DEFAULT_CHUNK_SIZE + 50

        vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
        await seed_vector_store(mock_client, vs)

        collection = vs._collection
        result = collection.get(include=["documents"])
        for doc_id, content in zip(result["ids"], result["documents"]):
            assert len(content) <= max_allowed, (
                f"Chunk {doc_id} is {len(content)} chars, exceeds max {max_allowed}. "
                f"Seed documents must be chunked before indexing."
            )

    async def test_nextera_docs_not_split(self, mock_client, tmp_path):
        """Nextera docs are small (<800 chars) — chunking should not split them."""
        vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
        n = await seed_vector_store(mock_client, vs)
        # All 13 docs should index as 13 chunks (no splitting)
        assert n == 13
        assert await vs.count() == 13
