"""
Retrieval Quality Evaluator — measure embeddinggemma MRR and Recall before/after fine-tuning.

Uses a fixed set of (query, correct_document) pairs and a small corpus to measure how
well embeddinggemma ranks the correct document for each query.

Metrics:
  MRR@10    — Mean Reciprocal Rank: average of 1/rank for each correct document
  Recall@5  — % of queries where the correct document is in the top-5 results

Usage:
  python -m finetune.eval_embeddinggemma                          # run + print report
  python -m finetune.eval_embeddinggemma --save results/baseline_embeddinggemma.json
  python -m finetune.eval_embeddinggemma --compare before.json after.json

Demo talk workflow:
  1. python -m finetune.eval_embeddinggemma --save results/baseline_embeddinggemma.json
  2. python -m finetune.train_embeddinggemma
  3. bash finetune/convert_embeddinggemma_to_gguf.sh
  4. bash scripts/start_servers.sh --bg --ft && python -m data.loader
  5. python -m finetune.eval_embeddinggemma --save results/finetuned_embeddinggemma.json
  6. python -m finetune.eval_embeddinggemma --compare results/baseline_embeddinggemma.json results/finetuned_embeddinggemma.json
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from finetune.eval_base import (
    compute_latency_stats,
    fmt_latency,
    fmt_pct as _fmt_pct,
    load_eval_json,
    load_eval_jsonl,
    load_results,
    save_results,
)


# ---------------------------------------------------------------------------
# Fixed test corpus and query-document pairs
# Loaded from data/eval-data/ (not inline).
# ---------------------------------------------------------------------------

CORPUS: list[str] = load_eval_json("eval_embeddinggemma_corpus.json")

TEST_PAIRS: list[dict] = load_eval_jsonl("eval_embeddinggemma_pairs.jsonl")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_eval(client, test_pairs: list[dict] | None = None) -> dict:
    """
    Embed all corpus passages and queries, then measure retrieval quality.

    For each query, ranks all corpus passages by cosine similarity and records
    the rank of the correct passage.
    """
    if test_pairs is None:
        test_pairs = TEST_PAIRS

    try:
        from src.engine.inference.client import SmallLanguageModelRole
        model_name = client.models.get(SmallLanguageModelRole.EMBEDDING, "unknown")
    except Exception:
        model_name = "unknown"

    # Embed corpus (batch for efficiency)
    print("  Embedding corpus…")
    corpus_embeddings = await client.embed_batch(CORPUS)

    # Evaluate each query
    predictions = []
    for item in test_pairs:
        t0 = time.perf_counter()
        q_emb = await client.embed(item["query"])
        # Cosine similarity (embeddings are already normalized by embeddinggemma)
        sims = [_cosine(q_emb, c_emb) for c_emb in corpus_embeddings]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        rank = ranked.index(item["correct_idx"]) + 1  # 1-indexed
        latency_ms = (time.perf_counter() - t0) * 1000
        predictions.append({
            "query":       item["query"],
            "correct_idx": item["correct_idx"],
            "rank":        rank,
            "in_top_5":    rank <= 5,
            "in_top_10":   rank <= 10,
            "mrr":         1.0 / rank if rank <= 10 else 0.0,
            "latency_ms":  round(latency_ms, 1),
        })

    return {
        "timestamp":   datetime.now().isoformat(),
        "model":       model_name,
        "n":           len(predictions),
        "predictions": predictions,
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two pre-normalized vectors = cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(results: dict) -> dict:
    preds = results["predictions"]
    n = len(preds)
    if n == 0:
        return {"mrr_at_10": 0.0, "recall_at_5": 0.0, "n": 0}

    mrr_at_10  = sum(p["mrr"]      for p in preds) / n
    recall_at_5 = sum(1 for p in preds if p["in_top_5"]) / n

    return {
        "mrr_at_10":  mrr_at_10,
        "recall_at_5": recall_at_5,
        "n":          n,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(before: dict, after: dict) -> dict:
    s_before = score(before)
    s_after  = score(after)
    return {
        "mrr_delta":    s_after["mrr_at_10"]   - s_before["mrr_at_10"],
        "recall_delta": s_after["recall_at_5"] - s_before["recall_at_5"],
        "before":       s_before,
        "after":        s_after,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "embeddinggemma Evaluation") -> None:
    s = score(results)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Model : {results.get('model', '?')}")
    print(f"  Run   : {results.get('timestamp', '?')[:19]}")
    print(f"{'='*60}")
    print(f"\n  MRR@10:    {s['mrr_at_10']:.4f}  (higher = correct doc ranked first)")
    print(f"  Recall@5:  {_fmt_pct(s['recall_at_5'])}  ({s['n']} queries)")

    latencies = [p["latency_ms"] for p in results["predictions"] if "latency_ms" in p]
    if latencies:
        print(f"  Latency:   {fmt_latency(compute_latency_stats(latencies))}")
    print()

    hard = [p for p in results["predictions"] if p["rank"] > 5]
    if hard:
        print(f"  Not in top-5 ({len(hard)}):")
        for p in hard:
            print(f"    rank={p['rank']:2d}  \"{p['query'][:70]}\"")
    print()


def print_comparison(comparison: dict, labels: tuple[str, str] = ("Before", "After")) -> None:
    before_label, after_label = labels
    c = comparison
    print(f"\n{'='*60}")
    print(f"  embeddinggemma:  {before_label}  →  {after_label}")
    print(f"{'='*60}")
    sign_mrr    = "+" if c["mrr_delta"]    >= 0 else ""
    sign_recall = "+" if c["recall_delta"] >= 0 else ""
    arrow_mrr    = "▲" if c["mrr_delta"]    > 0.001 else ("▼" if c["mrr_delta"]    < -0.001 else "═")
    arrow_recall = "▲" if c["recall_delta"] > 0.001 else ("▼" if c["recall_delta"] < -0.001 else "═")
    print(f"\n  MRR@10:   {c['before']['mrr_at_10']:.4f}  →  {c['after']['mrr_at_10']:.4f}"
          f"   {arrow_mrr} {sign_mrr}{c['mrr_delta']:+.4f}")
    print(f"  Recall@5: {_fmt_pct(c['before']['recall_at_5'])}  →  {_fmt_pct(c['after']['recall_at_5'])}"
          f"   {arrow_recall} {sign_recall}{_fmt_pct(c['recall_delta'])}")
    print()




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate embeddinggemma retrieval quality (MRR@10, Recall@5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--save",    metavar="PATH", help="Save raw results to JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved result files — no servers required")
    args = parser.parse_args()

    if args.compare:
        before_data = load_results(args.compare[0])
        after_data  = load_results(args.compare[1])
        print_report(before_data, title=f"Before  ({Path(args.compare[0]).name})")
        print_report(after_data,  title=f"After   ({Path(args.compare[1]).name})")
        print_comparison(
            compare(before_data, after_data),
            labels=(Path(args.compare[0]).stem, Path(args.compare[1]).stem),
        )
    else:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.engine.inference.client import SmallLanguageModelClient

        async def _main() -> None:
            client = SmallLanguageModelClient.create_with_auto_detection()
            print(f"\nRunning embeddinggemma retrieval eval "
                  f"({len(TEST_PAIRS)} queries, {len(CORPUS)} corpus passages)…")
            results = await run_eval(client)
            print_report(results)
            if args.save:
                save_results(results, args.save)

        asyncio.run(_main())
