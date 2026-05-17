"""
OCR Client — Extract text and tables from PDF pages via GLM-OCR.

Standalone async client for the GLM-OCR vision model served by llama-server.
Used only at document upload time (not during queries). Gracefully degrades
to empty results when the OCR server is not running — the caller falls back
to pypdf.

GLM-OCR is a 0.9B encoder-decoder model (#1 on OmniDocBench V1.5) that
extracts text, tables, formulas, and code blocks from document images.
It runs as a standard llama-server on port 9098 with --mmproj.

Prompt modes:
  "Text Recognition:"  — full document text extraction (Markdown output)
  "Table Recognition:" — structured table extraction (Markdown tables)
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

logger = logging.getLogger(__name__)

_PROMPT_TEXT = "Text Recognition:"
_PROMPT_TABLE = "Table Recognition:"


class OCRClient:
    """Async client for GLM-OCR text extraction via llama-server."""

    def __init__(
        self,
        base_url: str = "http://localhost:9098/v1",
        model: str = "glm-ocr",
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = AsyncOpenAI(
            base_url=base_url, api_key="no-key", timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self) -> bool:
        """Probe the OCR server health endpoint."""
        base = self._base_url.replace("/v1", "")
        try:
            async with httpx.AsyncClient(timeout=3.0) as http:
                resp = await http.get(f"{base}/health")
                return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Single-image extraction
    # ------------------------------------------------------------------

    async def extract_text(
        self, image_b64: str, *, mode: str = "text",
    ) -> str:
        """Extract text from a base64-encoded image via GLM-OCR.

        Args:
            image_b64: Base64-encoded PNG/JPEG image data.
            mode: "text" for full text extraction, "table" for table extraction.

        Returns:
            Extracted Markdown text, or "" on failure (never raises).
        """
        prompt = _PROMPT_TABLE if mode == "table" else _PROMPT_TEXT

        # Resize oversized images to prevent CUDA vision pipeline failures.
        # Decodes base64 → resizes → re-encodes only when needed.
        try:
            raw = base64.b64decode(image_b64)
            resized = self._resize_if_needed(raw, max_dim=1792)
            safe_b64 = base64.b64encode(resized).decode()
        except Exception:
            safe_b64 = image_b64  # fallback: use original if decode/resize fails

        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{safe_b64}"},
            },
            {"type": "text", "text": prompt},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except (APIConnectionError, APITimeoutError) as exc:
            logger.warning("OCR extraction failed (%s): %s", type(exc).__name__, exc)
            return ""
        except Exception as exc:
            logger.warning("OCR extraction unexpected error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Multi-page PDF extraction
    # ------------------------------------------------------------------

    async def extract_pages(
        self, pdf_bytes: bytes,
    ) -> list[tuple[int, str]]:
        """Convert PDF pages to images and extract text via OCR.

        Returns a list of (page_number, markdown_text) tuples.
        Pages with no extractable text are skipped.
        Returns [] if pymupdf is not installed or on any error.
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            logger.info(
                "pymupdf not installed — OCR page extraction unavailable. "
                "Install with: pip install pymupdf"
            )
            return []

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.warning("Failed to open PDF for OCR: %s", exc)
            return []

        pages: list[tuple[int, str]] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Try 150 DPI first; fall back to 100 DPI if image is too large
                # for the OCR model's context window (8192 tokens).
                text = ""
                for dpi in (150, 100):
                    pix = page.get_pixmap(dpi=dpi)
                    png_bytes = pix.tobytes("png")
                    b64 = base64.b64encode(png_bytes).decode()
                    # Resize happens inside extract_text() — no need here
                    text = await self.extract_text(b64, mode="text")
                    if text.strip():
                        break

                if text.strip():
                    pages.append((page_num + 1, text))
                else:
                    logger.debug("OCR page %d: no text extracted at any DPI", page_num + 1)
        finally:
            doc.close()

        return pages

    @staticmethod
    def _resize_if_needed(png_bytes: bytes, max_dim: int = 1792) -> bytes:
        """Resize PNG image if either dimension exceeds max_dim.

        CUDA vision pipelines in llama-server fail on images >~1792px.
        Metal handles larger images fine, but capping uniformly ensures
        cross-platform compatibility.

        Returns the original bytes if already within limits.
        """
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes))
            w, h = img.size
            if w <= max_dim and h <= max_dim:
                return png_bytes
            ratio = min(max_dim / w, max_dim / h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            logger.debug("Resized OCR image from %dx%d to %dx%d", w, h, *new_size)
            return buf.getvalue()
        except ImportError:
            return png_bytes  # PIL not available — send as-is
        except Exception as exc:
            logger.warning("Image resize failed: %s — sending original", exc)
            return png_bytes
