"""
Shared fixtures and configuration for the test suite.

Fixture scopes:
  - mock_small_language_model_client   — async mock of SmallLanguageModelClient (unit tests)
  - temp_db             — fresh SQLite database pre-seeded with demo data
  - temp_chroma_dir     — temp directory for ChromaDB (cleaned up after test)
  - servers_available   — True only when all three llama-server instances are reachable (e2e gating)
  - vision_server_available — True only when the vision llama-server on port 9093 is reachable
  - sample_image_b64    — base64-encoded PNG for vision tests (revenue_chart.png or 1x1 fallback)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from src.engine.inference.client import StreamChunk


def make_stream(content: str, tokens_used: int = 10, prompt_tokens: int = 5, completion_tokens: int = 5):
    """Return a callable that yields content as StreamChunks (for mocking generate_stream)."""
    async def _gen(**kwargs):
        yield StreamChunk(text=content)
        yield StreamChunk(done=True, tokens_used=tokens_used,
                          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return _gen

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

# pytest-asyncio ≥ 0.21 requires explicit mode; pytest.ini sets asyncio_mode=auto.
# This file just ensures the marker is available.


# ---------------------------------------------------------------------------
# Mock SmallLanguageModelClient
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_small_language_model_client():
    """
    A fully-mocked SmallLanguageModelClient that never touches llama-server.

    Defaults:
      generate()     → LLMResponse(content="direct_answer", model="gemma3:1b-it", tokens_used=5)
      call_function() → LLMResponse(content="", function_call=None)
      embed()        → [0.1] * 768   (unit vector placeholder)
      embed_batch()  → [[0.1] * 768] * n
    """
    from src.engine.inference.client import SmallLanguageModelRole, LLMResponse

    client = MagicMock()
    client.models = {
        SmallLanguageModelRole.INFERENCE:  "gemma3:1b-it",
        SmallLanguageModelRole.FUNCTION:   "qwen",
        SmallLanguageModelRole.EMBEDDING:  "embeddinggemma",
        SmallLanguageModelRole.VISION:     "gemma3-4b-vision",
    }

    # Default text generation — returns "direct_answer" (safe fallback intent)
    client.generate = AsyncMock(return_value=LLMResponse(
        content="direct_answer",
        model="gemma3:1b-it",
        tokens_used=5,
    ))

    # Default function calling — no tool selected
    client.call_function = AsyncMock(return_value=LLMResponse(
        content="",
        model="qwen",
        tokens_used=0,
        function_call=None,
    ))

    # Default embeddings — 768-dim placeholder vector
    _placeholder = [0.1] * 768
    client.embed       = AsyncMock(return_value=_placeholder)
    client.embed_batch = AsyncMock(side_effect=lambda texts: [_placeholder] * len(texts))

    # Default vision generation — describes image content
    client.generate_vision = AsyncMock(return_value=LLMResponse(
        content="This image shows a revenue chart with quarterly data.",
        model="gemma3-4b-vision",
        tokens_used=15,
    ))

    return client


# ---------------------------------------------------------------------------
# Temporary SQLite database
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """
    Create a temporary SQLite database with the same schema and seed data
    as the production demo database. Cleaned up automatically after the test.

    Returns the path to the database file.
    """
    import aiosqlite

    db_path = tmp_path / "test_business.db"

    async def _seed(path: str) -> None:
        async with aiosqlite.connect(path) as db:
            # Schema matches data/loader.py SQL_SCHEMA exactly
            await db.executescript("""
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

                INSERT INTO products VALUES
                    (1, 'Nextera Starter',       'platform', 299.0,  2990.0),
                    (2, 'Nextera Professional',  'platform', 999.0,  9990.0),
                    (3, 'Nextera Enterprise',    'platform', 3500.0, 35000.0);

                INSERT INTO customers VALUES
                    (1, 'Acme Corp',       'enterprise',    3500, '2023-01-15', 'Manufacturing'),
                    (2, 'BrightHealth GmbH', 'enterprise',  7000, '2023-03-01', 'Healthcare'),
                    (3, 'CodeStack Ltd',   'professional',  999,  '2023-05-20', 'Software');

                INSERT INTO sales VALUES
                    (1, 2024, 'Q1', 55100,  7,  1.0, 28.7),
                    (2, 2024, 'Q2', 68300,  8,  0.9, 23.9),
                    (3, 2024, 'Q3', 84900,  9,  0.8, 24.3),
                    (4, 2024, 'Q4', 103200, 11, 0.7, 21.6);
            """)
            await db.commit()

    asyncio.run(_seed(str(db_path)))
    return str(db_path)


# ---------------------------------------------------------------------------
# Temporary ChromaDB directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provide a fresh, empty directory for ChromaDB persistence."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    return str(chroma_dir)


# ---------------------------------------------------------------------------
# llama-server availability gate (e2e only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def servers_available():
    """
    Check once per session whether the three core llama-server instances are running.

    Checks the /health endpoint on INFERENCE_PORT, FUNCTION_PORT, EMBEDDING_PORT
    (from .env, defaulting to 9090/9091/9092). E2e tests skip if any is unreachable.

    The vision server (port 9093) is NOT required here — vision tests use the
    separate ``vision_server_available`` fixture and skip independently.

    Start servers with: bash scripts/start_servers.sh --bg [--ft]
    """
    import httpx

    ports = [
        int(os.getenv("INFERENCE_PORT", 9090)),
        int(os.getenv("FUNCTION_PORT",  9091)),
        int(os.getenv("EMBEDDING_PORT", 9092)),
    ]
    try:
        for port in ports:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
            if resp.status_code != 200:
                return False
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def vision_server_available():
    """
    Check once per session whether the vision llama-server (port 9093) is running.

    Vision e2e tests skip when this returns False, but all other e2e tests
    continue to run normally.
    """
    import httpx

    port = int(os.getenv("VISION_PORT", 9093))
    try:
        resp = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GLM-OCR server availability gate (e2e only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ocr_server_available():
    """
    Check once per session whether the GLM-OCR server (port 9098) is running.

    OCR e2e tests skip when this returns False, but all other e2e tests
    continue to run normally.
    """
    import httpx

    port = int(os.getenv("OCR_PORT", 9098))
    try:
        resp = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mock OCR client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ocr_client():
    """A fully-mocked OCRClient that returns canned Markdown text."""
    client = MagicMock()
    client.extract_text = AsyncMock(
        return_value="# Sample Document\n\nExtracted text content."
    )
    client.extract_pages = AsyncMock(return_value=[
        (1, "# Page 1\n\nSome text with a table:\n\n| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n| 3 | 4 |\n\nMore text after the table."),
        (2, "# Page 2\n\nJust prose content here. This is a paragraph with enough text to pass the minimum chunk size threshold for indexing."),
    ])
    client.check_health = AsyncMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Sample image fixture (base64-encoded PNG for vision tests)
# ---------------------------------------------------------------------------

def _make_1x1_red_png() -> bytes:
    """Build a minimal valid 1x1 red PNG entirely in Python (no PIL needed)."""
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return chunk_len + chunk_type + data + crc

    # IHDR: 1x1, 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: filter-byte 0 + RGB(255, 0, 0)
    raw_row = b"\x00\xff\x00\x00"
    idat = _chunk(b"IDAT", zlib.compress(raw_row))

    # IEND
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


@pytest.fixture
def sample_image_b64():
    """
    Load data/demo-images/revenue_chart.png as a base64 string.

    Falls back to a synthetic 1x1 red PNG if the file does not exist, so
    vision unit/integration tests can run without the full data directory.
    """
    image_path = Path(__file__).resolve().parent.parent / "data" / "demo-images" / "revenue_chart.png"
    if image_path.is_file():
        return base64.b64encode(image_path.read_bytes()).decode()
    return base64.b64encode(_make_1x1_red_png()).decode()


# ---------------------------------------------------------------------------
# Interaction log fixture for data_prep tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_interactions():
    """
    A minimal but representative list of interaction log records,
    mirroring exactly what SmallLanguageModelAgentOrchestrator._log_interaction produces.
    """
    return [
        {
            "timestamp": "2025-01-01T12:00:00",
            "query": "What features does the Enterprise plan include?",
            "intent": "rag_query",
            "response": "The Enterprise plan includes unlimited users and 24/7 support.",
            "steps": [
                {
                    "action": "rewrite_query",
                    "model": "gemma3:1b-it",
                    "details": {"original": "Enterprise plan features?", "rewritten": "Enterprise plan features limits"},
                },
                {
                    "action": "vector_search",
                    "model": "embeddinggemma",
                    "details": {
                        "query": "Enterprise plan features limits",
                        "documents": [
                            {"id": "doc-001", "content": "Enterprise plan starts at €3,500/month.", "metadata": {}},
                            {"id": "doc-002", "content": "Enterprise includes unlimited users.", "metadata": {}},
                        ],
                    },
                },
            ],
            "models_used": ["gemma3:1b-it", "embeddinggemma"],
        },
        {
            "timestamp": "2025-01-01T12:05:00",
            "query": "What were total sales in Q3 2024?",
            "intent": "tool_use",
            "response": "Total Q3 2024 sales were €84,900.",
            "steps": [
                {
                    "action": "select_tool",
                    "model": "qwen",
                    "details": {
                        "tool": "sql_query",
                        "arguments": {"query": "SELECT revenue FROM sales WHERE year=2024 AND quarter='Q3'"},
                    },
                },
                {
                    "action": "execute_tool",
                    "model": "local_execution",
                    "details": {"success": True, "error": None},
                },
            ],
            "models_used": ["qwen", "gemma3:1b-it"],
        },
        {
            "timestamp": "2025-01-01T12:10:00",
            "query": "Hello! What can you help me with?",
            "intent": "direct_answer",
            "response": "I can help you with product information and data queries.",
            "steps": [
                {
                    "action": "direct_response",
                    "model": "gemma3:1b-it",
                    "details": {},
                }
            ],
            "models_used": ["gemma3:1b-it"],
        },
    ]
