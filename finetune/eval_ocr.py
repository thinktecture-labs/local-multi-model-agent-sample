"""
OCR Quality Evaluator — measure GLM-OCR extraction + RAG accuracy on uploaded documents.

29 queries across 5 Nextera demo documents (factual, real_world, table_extraction,
benchmark_report).

Evaluates:
  1. OCR extraction quality — does OCR produce correct text from the PDF?
  2. End-to-end RAG quality — after OCR upload, can the agent answer questions correctly?

Usage:
  python -m finetune.eval_ocr                                     # run + print report
  python -m finetune.eval_ocr --save results/ocr_baseline.json    # save raw results
  python -m finetune.eval_ocr --pypdf-only                        # force pypdf (no OCR)
  python -m finetune.eval_ocr --document-chat                     # use document_id (bypass classifier)
  python -m finetune.eval_ocr --full-ocr                          # OCR every page (no smart filtering)
  python -m finetune.eval_ocr --compare results/pypdf.json results/ocr.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.inference.config import SCENARIO_CONFIG
from finetune._scenario import SCENARIO_NAME as _SCENARIO
from finetune.eval_base import (
    compute_latency_stats,
    fmt_latency,
    fmt_pct as _fmt_pct,
    load_eval_jsonl,
    load_results,
    save_results,
)


# ---------------------------------------------------------------------------
# Paths & categories — scenario-aware
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = _PROJECT_ROOT / "data" / "demo-documents"
CATEGORIES = ["factual", "real_world", "table_extraction", "benchmark_report"]
_EVAL_FILE = "eval_ocr.jsonl"


# ---------------------------------------------------------------------------
# Fixed test set — loaded from data/eval-data/<scenario-specific>.jsonl
# ---------------------------------------------------------------------------

TEST_SET: list[dict] = load_eval_jsonl(_EVAL_FILE)


# ---------------------------------------------------------------------------
# Keyword checker
# ---------------------------------------------------------------------------

def check_keywords(response: str, expected: list[str]) -> bool:
    """Check if any expected keyword appears in the response (case-insensitive).

    Keywords prefixed with ``r/`` are treated as regex patterns.
    Example: ``"r/3[.,]?6\\d{2}"`` matches "3,626" and "3.63" and "3626".
    """
    import re as _re
    response_lower = response.lower()
    for kw in expected:
        if kw.startswith("r/"):
            pattern = kw[2:]
            if _re.search(pattern, response, _re.IGNORECASE):
                return True
        else:
            if kw.lower() in response_lower:
                return True
    return False


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_eval(
    agent,
    vector_store,
    ocr_client=None,
    *,
    skip_missing: bool = True,
    document_chat: bool = False,
    full_ocr: bool = False,
    slm_client=None,
) -> dict:
    """
    Run every test query through the full pipeline.

    1. Upload each unique document via DocumentProcessor
    2. Query the agent (or use document_chat mode) for each test case
    3. Score keyword hits

    Modes:
      - default: agent pipeline (classifier → handler)
      - document_chat: bypass classifier, direct vector search by document_id
      - full_ocr: disable smart filtering, OCR every page
    """
    import re
    from src.engine.knowledge.document_processor import DocumentProcessor
    from src.engine.inference.prompts import RAG_SYNTHESIS_SYSTEM_PROMPT, RAG_SYNTHESIS_USER_TEMPLATE

    # Deduplicate documents
    docs_to_upload = sorted(set(item["document"] for item in TEST_SET))

    # Upload phase
    uploaded: set[str] = set()
    upload_timings: dict[str, float] = {}

    for doc_name in docs_to_upload:
        doc_path = DOCS_DIR / doc_name
        if not doc_path.is_file():
            if skip_missing:
                print(f"  Skipping {doc_name} (not found)")
                continue
            else:
                raise FileNotFoundError(f"Document not found: {doc_path}")

        t0 = time.perf_counter()
        effective_ocr = ocr_client
        if full_ocr and ocr_client is not None:
            # For full-OCR mode, set _MIN_PYPDF_CHARS very high so all pages go through OCR
            processor = DocumentProcessor(vector_store, ocr_client=ocr_client)
            processor._MIN_PYPDF_CHARS = 999999  # force all pages through OCR
        else:
            processor = DocumentProcessor(vector_store, ocr_client=effective_ocr)
        async for event in processor.process_file(doc_name, doc_path.read_bytes()):
            if event.stage == "ocr_extraction":
                method = event.detail.get("method", "")
                ocr_pages = event.detail.get("ocr_pages", 0)
                print(f"    OCR: {ocr_pages} pages ({method})")
        upload_ms = (time.perf_counter() - t0) * 1000
        uploaded.add(doc_name)
        upload_timings[doc_name] = upload_ms
        print(f"  Uploaded {doc_name} ({upload_ms:.0f}ms)")

    # Query phase
    predictions: list[dict] = []
    for item in TEST_SET:
        if item["document"] not in uploaded:
            continue

        t0 = time.perf_counter()

        if document_chat and slm_client is not None:
            # Document chat mode: bypass classifier, search uploads directly
            stem = Path(item["document"]).stem
            doc_id = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
            results = await vector_store.search(
                item["query"], top_k=5, where={"document_id": doc_id},
            )
            if results:
                context = "\n\n".join(
                    f"[Source: {d.metadata.get('title', d.id)}]\n{d.content}"
                    for d in results[:5]
                )
                resp = await slm_client.generate_synthesis(
                    messages=[
                        {"role": "system", "content": RAG_SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": RAG_SYNTHESIS_USER_TEMPLATE.format(
                            context=context, query=item["query"])},
                    ],
                )
                response_text = resp.content.strip()
                intent_val = "document_chat"
            else:
                response_text = "No results found."
                intent_val = "document_chat"
        else:
            result = await agent.process(item["query"])
            response_text = result.response
            intent_val = result.intent.value

        latency_ms = (time.perf_counter() - t0) * 1000

        correct = check_keywords(response_text, item["expected_keywords"])
        predictions.append({
            "document": item["document"],
            "query": item["query"],
            "expected_keywords": item["expected_keywords"],
            "category": item["category"],
            "response": response_text,
            "correct": correct,
            "latency_ms": round(latency_ms, 1),
            "intent": intent_val,
        })

    mode = "document_chat" if document_chat else ("full_ocr" if full_ocr else "default")
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "documents_uploaded": list(uploaded),
        "upload_timings_ms": upload_timings,
        "ocr_enabled": ocr_client is not None,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    """Compute accuracy metrics from eval results."""
    preds = results["predictions"]
    if not preds:
        return {"overall": 0.0, "per_category": {}, "n": 0}

    correct = sum(1 for p in preds if p["correct"])
    overall = correct / len(preds)

    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_preds = [p for p in preds if p["category"] == cat]
        if cat_preds:
            cat_correct = sum(1 for p in cat_preds if p["correct"])
            per_cat[cat] = {
                "accuracy": cat_correct / len(cat_preds),
                "correct": cat_correct,
                "total": len(cat_preds),
            }

    latencies = [p["latency_ms"] for p in preds]
    return {
        "overall": overall,
        "correct": correct,
        "total": len(preds),
        "per_category": per_cat,
        "latency": compute_latency_stats(latencies),
        "ocr_enabled": results.get("ocr_enabled", False),
    }


def print_report(scores: dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"  OCR Evaluation — {'with GLM-OCR' if scores['ocr_enabled'] else 'pypdf only'}")
    print("=" * 60)
    print(f"\n  Overall: {_fmt_pct(scores['overall'])} ({scores['correct']}/{scores['total']})")

    if scores.get("per_category"):
        print("\n  Per category:")
        for cat, data in scores["per_category"].items():
            bar = "█" * int(data["accuracy"] * 20) + "░" * (20 - int(data["accuracy"] * 20))
            print(f"    {cat:20s} {bar} {_fmt_pct(data['accuracy'])} ({data['correct']}/{data['total']})")

    if scores.get("latency"):
        print(f"\n  Latency: {fmt_latency(scores['latency'])}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OCR quality evaluation")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--pypdf-only", action="store_true", help="Force pypdf (no OCR)")
    parser.add_argument("--document-chat", action="store_true",
                        help="Use document_id mode (bypass classifier, direct vector search)")
    parser.add_argument("--full-ocr", action="store_true",
                        help="OCR every page (disable smart filtering)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="Compare two result files")
    args = parser.parse_args()

    if args.compare:
        before = load_results(args.compare[0])
        after = load_results(args.compare[1])
        print("\n--- BEFORE ---")
        print_report(score(before))
        print("--- AFTER ---")
        print_report(score(after))
        s_before, s_after = score(before), score(after)
        delta = s_after["overall"] - s_before["overall"]
        print(f"  Delta: {'+' if delta >= 0 else ''}{delta*100:.1f}%")
        return

    from src.engine.inference.client import SmallLanguageModelClient
    from src.engine.knowledge.vector_store import VectorStore
    from src.engine.tools import create_default_registry
    from src.engine.agent import SmallLanguageModelAgentOrchestrator

    print("Setting up agent...")
    client = SmallLanguageModelClient.create_with_auto_detection()
    vector_store = VectorStore(
        collection_name="eval_ocr",
        persist_dir=SCENARIO_CONFIG.chroma_dir,
    )
    vector_store.set_client(client)
    tools = create_default_registry(vector_store=vector_store)
    agent = SmallLanguageModelAgentOrchestrator(client, tools)

    ocr_client = None
    if not args.pypdf_only:
        from src.engine.knowledge.ocr_client import OCRClient
        ocr = OCRClient()
        if await ocr.check_health():
            ocr_client = ocr
            print("  GLM-OCR available — using OCR extraction")
        else:
            print("  GLM-OCR not available — falling back to pypdf")

    mode_label = "document_chat" if args.document_chat else ("full_ocr" if args.full_ocr else "default")
    print(f"\nRunning eval ({len(TEST_SET)} queries, mode={mode_label})...")
    results = await run_eval(
        agent, vector_store, ocr_client=ocr_client,
        document_chat=args.document_chat,
        full_ocr=args.full_ocr,
        slm_client=client,
    )
    scores_dict = score(results)
    print_report(scores_dict)

    if args.save:
        save_results(results, args.save)
        print(f"  Results saved to {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
