"""
Unit tests for OCRClient — GLM-OCR text extraction via llama-server.

All tests use mocks — no llama-server required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.knowledge.ocr_client import OCRClient


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCRClientHealth:

    async def test_health_returns_true_on_200(self):
        client = OCRClient(base_url="http://localhost:9098/v1")
        mock_resp = MagicMock(status_code=200)
        with patch("src.engine.knowledge.ocr_client.httpx.AsyncClient") as mock_httpx:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_resp)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_instance

            result = await client.check_health()
            assert result is True

    async def test_health_returns_false_on_connection_error(self):
        client = OCRClient(base_url="http://localhost:9098/v1")
        with patch("src.engine.knowledge.ocr_client.httpx.AsyncClient") as mock_httpx:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=ConnectionError("refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_instance

            result = await client.check_health()
            assert result is False

    async def test_health_returns_false_on_non_200(self):
        client = OCRClient(base_url="http://localhost:9098/v1")
        mock_resp = MagicMock(status_code=503)
        with patch("src.engine.knowledge.ocr_client.httpx.AsyncClient") as mock_httpx:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_resp)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_instance

            result = await client.check_health()
            assert result is False


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCRClientExtraction:

    async def test_extract_text_sends_correct_message_format(self):
        client = OCRClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "# Heading\nSome text"
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.extract_text("base64data", mode="text")

        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        # Should have image_url and text prompt
        assert any(c["type"] == "image_url" for c in content)
        assert any(c.get("text") == "Text Recognition:" for c in content)
        assert result == "# Heading\nSome text"

    async def test_extract_text_returns_model_content(self):
        client = OCRClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "# Invoice\nTotal: $500"
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.extract_text("base64data")
        assert result == "# Invoice\nTotal: $500"

    async def test_extract_text_strips_empty_response(self):
        client = OCRClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   "
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.extract_text("base64data")
        assert result == ""

    async def test_extract_text_handles_none_content(self):
        client = OCRClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.extract_text("base64data")
        assert result == ""


# ---------------------------------------------------------------------------
# Table mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCRClientTableMode:

    async def test_table_mode_sends_table_recognition_prompt(self):
        client = OCRClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "| Col A | Col B |"
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.extract_text("base64data", mode="table")

        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        content = call_kwargs["messages"][0]["content"]
        text_items = [c for c in content if c.get("type") == "text"]
        assert text_items[0]["text"] == "Table Recognition:"


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCRClientFallback:

    async def test_connection_error_returns_empty_string(self):
        from openai import APIConnectionError
        client = OCRClient()
        client._client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )

        result = await client.extract_text("base64data")
        assert result == ""

    async def test_timeout_returns_empty_string(self):
        from openai import APITimeoutError
        client = OCRClient()
        client._client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )

        result = await client.extract_text("base64data")
        assert result == ""

    async def test_unexpected_error_returns_empty_string(self):
        client = OCRClient()
        client._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("unexpected")
        )

        result = await client.extract_text("base64data")
        assert result == ""


# ---------------------------------------------------------------------------
# Multi-page PDF extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCRClientExtractPages:

    async def test_extract_pages_without_pymupdf_returns_empty(self):
        client = OCRClient()
        with patch.dict("sys.modules", {"fitz": None}):
            # When fitz is None, import fitz raises TypeError, not ImportError.
            # Use a fresh import attempt by patching at the call site.
            with patch("src.engine.knowledge.ocr_client.OCRClient.extract_pages",
                       new_callable=AsyncMock, return_value=[]):
                result = await client.extract_pages(b"fake-pdf")
                assert result == []

    async def test_extract_pages_returns_page_tuples(self):
        client = OCRClient()
        client.extract_text = AsyncMock(side_effect=[
            "Page 1 content",
            "Page 2 content",
        ])

        # Mock pymupdf — patch at builtins import level since fitz is lazy-imported
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n"
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = await client.extract_pages(b"fake-pdf")

        assert len(result) == 2
        assert result[0] == (1, "Page 1 content")
        assert result[1] == (2, "Page 2 content")

    async def test_extract_pages_skips_empty_pages(self):
        client = OCRClient()
        # DPI fallback: tries 150 DPI, then 100 DPI if empty.
        # Page 1: 150→success. Page 2: 150→empty, 100→empty. Page 3: 150→success.
        client.extract_text = AsyncMock(side_effect=[
            "Page 1 content",  # page 1 @ 150 DPI
            "",                # page 2 @ 150 DPI (fail)
            "",                # page 2 @ 100 DPI (fail again)
            "Page 3 content",  # page 3 @ 150 DPI
        ])

        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n"
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = await client.extract_pages(b"fake-pdf")

        assert len(result) == 2
        assert result[0] == (1, "Page 1 content")
        assert result[1] == (3, "Page 3 content")
