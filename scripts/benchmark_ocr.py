#!/usr/bin/env python3
"""
OCR Benchmark — measure per-page OCR speed, upload throughput, and document chat latency.

Runs on the local machine and produces a JSON results file + Rich console report.
Designed for cross-platform comparison (M5 Max, RTX PRO 6000, DGX Spark, MS-S1 MAX).

Usage:
  python scripts/benchmark_ocr.py                              # run all benchmarks
  python scripts/benchmark_ocr.py --save results/ocr_bench_m5max.json
  python scripts/benchmark_ocr.py --pages 5                    # limit OCR pages
  python scripts/benchmark_ocr.py --skip-ocr                   # skip per-page OCR (no server needed)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.inference.config import SCENARIO_CONFIG


DOCS_DIR = Path(SCENARIO_CONFIG.demo_documents_dir)
NEXTERA_PDF = DOCS_DIR / "nextera_quarterly_report.pdf"
SNOWFLAKE_PDF = DOCS_DIR / "snowflake-fy2025-first50.pdf"


def get_platform_info() -> dict:
    """Gather system info for the benchmark report."""
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    # Try to get GPU info
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
            info["backend"] = "CUDA"
    except Exception:
        if platform.system() == "Darwin":
            info["backend"] = "Metal"
        else:
            info["backend"] = "CPU"
    return info


async def benchmark_per_page_ocr(pdf_path: Path, max_pages: int = 10, dpi: int = 150) -> list[dict]:
    """Benchmark OCR extraction speed per page."""
    from src.engine.knowledge.ocr_client import OCRClient

    ocr = OCRClient()
    if not await ocr.check_health():
        print("  GLM-OCR server not available — skipping per-page benchmark")
        return []

    import fitz
    doc = fitz.open(str(pdf_path))
    n_pages = min(max_pages, len(doc))

    results = []
    print(f"  OCR per-page benchmark: {n_pages} pages from {pdf_path.name} @ {dpi} DPI")

    for i in range(n_pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode()
        img_kb = len(png_bytes) / 1024

        t0 = time.perf_counter()
        text = await ocr.extract_text(b64, mode="text")
        ms = (time.perf_counter() - t0) * 1000

        results.append({
            "page": i + 1,
            "dpi": dpi,
            "image_kb": round(img_kb, 1),
            "ocr_ms": round(ms, 1),
            "chars": len(text),
        })
        print(f"    page {i+1}/{n_pages}: {ms:.0f}ms, {len(text)} chars, {img_kb:.0f} KB")

    doc.close()
    return results


async def benchmark_upload(pdf_path: Path, ocr_client=None) -> dict:
    """Benchmark full upload pipeline (parse → OCR → chunk → embed → index)."""
    from src.engine.inference.client import SmallLanguageModelClient
    from src.engine.knowledge.vector_store import VectorStore
    from src.engine.knowledge.document_processor import DocumentProcessor

    client = SmallLanguageModelClient.create_with_auto_detection()
    vs = VectorStore(collection_name="bench_upload", persist_dir=SCENARIO_CONFIG.chroma_dir)
    vs.set_client(client)

    # Clear previous benchmark data
    try:
        await vs.clear()
    except Exception:
        pass

    processor = DocumentProcessor(vs, ocr_client=ocr_client)
    content = pdf_path.read_bytes()

    timings: dict[str, float] = {}
    total_chunks = 0

    t0 = time.perf_counter()
    async for event in processor.process_file(pdf_path.name, content):
        if event.stage == "indexed":
            total_chunks = event.detail.get("total_chunks", 0)
            timings = {
                "parse_ms": event.detail.get("parse_ms", 0),
                "ocr_ms": event.detail.get("ocr_ms", 0),
                "chunk_ms": event.detail.get("chunk_ms", 0),
                "embed_ms": event.detail.get("embed_ms", 0),
                "total_ms": event.detail.get("total_ms", 0),
            }
    wall_ms = (time.perf_counter() - t0) * 1000

    # Cleanup
    try:
        await vs.clear()
    except Exception:
        pass

    return {
        "file": pdf_path.name,
        "size_kb": round(len(content) / 1024, 1),
        "total_chunks": total_chunks,
        "wall_ms": round(wall_ms, 1),
        "ocr_enabled": ocr_client is not None,
        **timings,
    }


async def benchmark_document_chat(n_queries: int = 5) -> list[dict]:
    """Benchmark document chat query latency (upload → query cycle)."""
    from src.engine.inference.client import SmallLanguageModelClient
    from src.engine.knowledge.vector_store import VectorStore
    from src.engine.knowledge.document_processor import DocumentProcessor
    from src.engine.inference.prompts import RAG_SYNTHESIS_SYSTEM_PROMPT, RAG_SYNTHESIS_USER_TEMPLATE

    if not NEXTERA_PDF.is_file():
        print("  Nextera PDF not found — skipping document chat benchmark")
        return []

    client = SmallLanguageModelClient.create_with_auto_detection()
    vs = VectorStore(collection_name="bench_chat", persist_dir=SCENARIO_CONFIG.chroma_dir)
    vs.set_client(client)

    try:
        await vs.clear()
    except Exception:
        pass

    # Upload the Nextera PDF
    processor = DocumentProcessor(vs)
    async for _ in processor.process_file(NEXTERA_PDF.name, NEXTERA_PDF.read_bytes()):
        pass

    queries = [
        "What was total revenue in Q4 2024?",
        "Which customer has the highest MRR?",
        "What is the Enterprise plan monthly price?",
        "How many product tiers does Nextera offer?",
        "What was the churn rate in Q2 2024?",
    ][:n_queries]

    results = []
    for q in queries:
        t0 = time.perf_counter()

        docs = await vs.search(q, top_k=5, where={"document_id": "nextera-quarterly-report"})
        search_ms = (time.perf_counter() - t0) * 1000

        if docs:
            context = "\n\n".join(
                f"[Source: {d.metadata.get('title', d.id)}]\n{d.content}"
                for d in docs[:5]
            )
            t1 = time.perf_counter()
            resp = await client.generate_synthesis(
                messages=[
                    {"role": "system", "content": RAG_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": RAG_SYNTHESIS_USER_TEMPLATE.format(
                        context=context, query=q)},
                ],
            )
            synth_ms = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
        else:
            synth_ms = 0
            total_ms = search_ms

        results.append({
            "query": q,
            "search_ms": round(search_ms, 1),
            "synthesis_ms": round(synth_ms, 1),
            "total_ms": round(total_ms, 1),
        })
        print(f"    {total_ms:6.0f}ms (search: {search_ms:.0f}ms, synth: {synth_ms:.0f}ms) | {q[:60]}")

    # Cleanup
    try:
        await vs.clear()
    except Exception:
        pass

    return results


def print_report(bench: dict) -> None:
    """Print a formatted benchmark report."""
    print("\n" + "=" * 70)
    print(f"  OCR Benchmark — {bench['platform']['hostname']}")
    print(f"  {bench['platform'].get('backend', 'unknown')} | {bench['platform'].get('gpu', bench['platform'].get('processor', ''))}")
    print("=" * 70)

    if bench.get("per_page"):
        pages = bench["per_page"]
        times = [p["ocr_ms"] for p in pages]
        print(f"\n  Per-page OCR ({len(pages)} pages):")
        print(f"    Mean:   {mean(times):.0f}ms")
        print(f"    Median: {median(times):.0f}ms")
        print(f"    Min:    {min(times):.0f}ms")
        print(f"    Max:    {max(times):.0f}ms")
        print(f"    Pages/sec: {1000 / mean(times):.2f}")

    if bench.get("uploads"):
        print(f"\n  Upload pipeline:")
        for u in bench["uploads"]:
            ocr_label = " (OCR)" if u.get("ocr_enabled") else " (pypdf)"
            print(f"    {u['file']:45s} {u['wall_ms']:8.0f}ms  {u['total_chunks']:3d} chunks{ocr_label}")
            if u.get("ocr_ms", 0) > 0:
                print(f"      parse={u.get('parse_ms',0):.0f}ms  ocr={u.get('ocr_ms',0):.0f}ms  chunk={u.get('chunk_ms',0):.0f}ms  embed={u.get('embed_ms',0):.0f}ms")

    if bench.get("document_chat"):
        chats = bench["document_chat"]
        times = [c["total_ms"] for c in chats]
        print(f"\n  Document chat ({len(chats)} queries):")
        print(f"    Mean:   {mean(times):.0f}ms")
        print(f"    Median: {median(times):.0f}ms")
        search_times = [c["search_ms"] for c in chats]
        synth_times = [c["synthesis_ms"] for c in chats]
        print(f"    Search: {mean(search_times):.0f}ms mean")
        print(f"    Synth:  {mean(synth_times):.0f}ms mean")

    print()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="OCR benchmark")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--pages", type=int, default=10, help="Max pages for per-page OCR benchmark")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip per-page OCR (no OCR server needed)")
    args = parser.parse_args()

    bench: dict = {
        "timestamp": datetime.now().isoformat(),
        "platform": get_platform_info(),
    }

    # 1. Per-page OCR speed
    if not args.skip_ocr and SNOWFLAKE_PDF.is_file():
        print("\n1. Per-page OCR benchmark")
        bench["per_page"] = await benchmark_per_page_ocr(SNOWFLAKE_PDF, max_pages=args.pages)
    else:
        print("\n1. Per-page OCR: skipped")
        bench["per_page"] = []

    # 2. Upload pipeline (pypdf-only + OCR)
    print("\n2. Upload pipeline benchmark")
    bench["uploads"] = []

    if NEXTERA_PDF.is_file():
        print(f"  Uploading {NEXTERA_PDF.name} (pypdf-only)...")
        bench["uploads"].append(await benchmark_upload(NEXTERA_PDF, ocr_client=None))

    if SNOWFLAKE_PDF.is_file():
        print(f"  Uploading {SNOWFLAKE_PDF.name} (pypdf-only)...")
        bench["uploads"].append(await benchmark_upload(SNOWFLAKE_PDF, ocr_client=None))

        if not args.skip_ocr:
            from src.engine.knowledge.ocr_client import OCRClient
            ocr = OCRClient()
            if await ocr.check_health():
                print(f"  Uploading {SNOWFLAKE_PDF.name} (with OCR)...")
                bench["uploads"].append(await benchmark_upload(SNOWFLAKE_PDF, ocr_client=ocr))

    # 3. Document chat latency
    print("\n3. Document chat benchmark")
    bench["document_chat"] = await benchmark_document_chat()

    # Report
    print_report(bench)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(bench, f, indent=2)
        print(f"  Results saved to {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
