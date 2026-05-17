"""
Fine-Tune embeddinggemma — semantic retrieval quality.

Unlike the other two models (which use LoRA for instruction following),
embeddinggemma is fine-tuned as a bi-encoder using contrastive learning.

Supported training data formats:

  1. Pair format  (query, positive):
       {"query": "...", "positive": "..."}
     Loss: MultipleNegativesRankingLoss
           — treats all other positives in the batch as hard negatives
           — no explicit negative mining required

  2. Triplet format  (anchor, positive, negative):
       {"anchor": "...", "positive": "...", "negative": "..."}
     Loss: TripletLoss
           — uses explicit hard negatives for stronger training signal
           — requires negative mining or manual curation

Both formats can be mixed in the same JSONL file — the loader detects
which loss to apply per record based on key presence.

Run:
  python -m finetune.train_embeddinggemma
  python -m finetune.train_embeddinggemma --epochs 5 --batch-size 32

Output: ./models/embeddinggemma-300m-ft-merged/
        (fine-tuned model ready for: bash finetune/convert_embeddinggemma_to_gguf.sh)

Prerequisites:
  pip install -r requirements-finetune.txt

─────────────────────────────────────────────────────────────────────────
Why fine-tuning may show little improvement — and how to fix it
─────────────────────────────────────────────────────────────────────────

google/embeddinggemma-300m is a purpose-built embedding model that already
achieves ~95% MRR on typical business-domain retrieval tasks out of the box.
With a small synthetic dataset (< 50 pairs) and no hard negatives, training
usually converges at initialisation — the model is already near-optimal for
the given examples.  The internal eval MRR stays flat across all epochs and
save_best_model saves the original weights.

To produce measurable improvement, address each of the following in order:

1.  ADD HARD NEGATIVES (highest impact)
    Switch from pair to triplet format in data/training-data/embeddinggemma_retrieval.jsonl.
    A hard negative is a passage that looks superficially relevant but is wrong:

      # Pair (weak signal):
      {"query": "enterprise plan pricing", "positive": "Enterprise: €3,500/month..."}

      # Triplet (strong signal):
      {"anchor": "enterprise plan pricing",
       "positive": "Enterprise: €3,500/month, unlimited users, 24/7 SLA...",
       "negative": "Starter plan: €299/month, up to 5 users, community support..."}

    Both formats can coexist in the same file — the loader auto-detects per record.
    A good negative is one that shares keywords with the query but answers a
    different question (e.g. pricing for the wrong tier, compliance for wrong plan).

2.  MORE TRAINING PAIRS (200+ recommended, 500+ for clear gains)
    37 pairs produce ≈ 16 in-batch negatives per step at batch_size=16.
    MultipleNegativesRankingLoss needs many in-batch negatives to create a
    strong learning signal.  Target: 200–500 pairs minimum.

    Best source: real user interactions after running the demo.
      python -m data.loader        # seed demo data
      python demo.py --interactive  # collect real queries (logged automatically)
      python -m finetune.data_prep  # extract query/document pairs from logs
    After 100+ RAG interactions, data_prep produces realistic query→document
    pairs that reflect actual user language, not synthetic keyword phrases.

3.  LARGER BATCH SIZE (32+)
    In-batch negatives scale with batch_size.  batch_size=32 gives ~31 negatives
    per positive vs. ~15 at batch_size=16.  On a GPU with 8+ GB VRAM, use:
      python -m finetune.train_embeddinggemma --batch-size 32

4.  TUNE LEARNING RATE AND EPOCHS
    embeddinggemma-300m has already learned strong general representations.
    A high learning rate destroys this quickly.  Recommended starting point:
      learning_rate = 5e-6   (half of the default 2e-5)
      num_epochs    = 3–5    (not 10 — early stopping via save_best_model catches the peak)
    If the internal MRR eval is still flat after epoch 2, the data is the
    bottleneck, not the hyperparameters.

5.  GGUF CONVERSION NOTE
    The sentence-transformers trainer adds Dense projection layers (768→3072→768)
    that are NOT included in the GGUF — llama-server performs mean pooling directly
    on the backbone output.  These projection layers do carry some domain adaptation.
    To preserve them, consider using the model's .encode() method directly (bypassing
    llama-server) and keeping the full sentence-transformers model for embedding at
    inference time (requires a custom SmallLanguageModelClient.embed() implementation).
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from finetune._scenario import TRAINING_DIR as _TRAINING_DIR, SUFFIX as _SUFFIX

import numpy as np
import torch

try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingGemmaTrainConfig:
    # Base model — the exact production model: google/embeddinggemma-300m
    # This is Google's dedicated 308M embedding model with bidirectional attention,
    # the same base as the GGUF served on port 9092. Fine-tuning it on domain
    # query-document pairs improves retrieval accuracy for Nextera Platform queries.
    base_model:       str   = "google/embeddinggemma-300m"

    num_epochs:       int   = 10
    batch_size:       int   = 16      # larger batches = more in-batch negatives = stronger signal
    learning_rate:    float = 5e-6
    warmup_steps:     int   = 100

    max_seq_length:   int   = 512     # embeddinggemma supports up to 2K tokens
    show_progress:    bool  = True

    data_path:        str   = field(default_factory=lambda: f"./{_TRAINING_DIR}/embeddinggemma_retrieval{_SUFFIX}.jsonl")
    output_dir:       str   = "./models/embeddinggemma-300m-ft-merged"
    eval_during_training: bool = True


# ---------------------------------------------------------------------------
# Dataset loader — supports both pair and triplet formats
# ---------------------------------------------------------------------------

def _load_examples(filepath: str):
    """
    Load training examples from JSONL, detecting format per record.

    Returns (pair_examples, triplet_examples) as separate lists:
      - pair_examples:    list of InputExample with texts=[query, positive]
      - triplet_examples: list of InputExample with texts=[anchor, positive, negative]
    """
    if not _DEPS_AVAILABLE:
        return [], []

    pair_examples     = []
    triplet_examples  = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if "anchor" in item and "positive" in item and "negative" in item:
                # Triplet format — explicit hard negative
                triplet_examples.append(InputExample(
                    texts=[item["anchor"], item["positive"], item["negative"]]
                ))
            elif "query" in item and "positive" in item:
                # Pair format — in-batch negatives via MNRL
                pair_examples.append(InputExample(
                    texts=[item["query"], item["positive"]]
                ))

    return pair_examples, triplet_examples


# ---------------------------------------------------------------------------
# Evaluation — manual MRR@10 on held-out pairs
# ---------------------------------------------------------------------------

def _evaluate_retrieval(model, eval_pairs: list, top_k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank (MRR@top_k) on a small held-out set.

    For each query, the model encodes all passages and ranks them by
    cosine similarity. MRR measures how high the correct passage ranks.
    """
    if not eval_pairs:
        return 0.0

    try:
        import numpy as np
    except ImportError:
        return 0.0

    queries   = [ex.texts[0] for ex in eval_pairs]
    positives = [ex.texts[1] for ex in eval_pairs]

    q_embs = model.encode(queries,   batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    p_embs = model.encode(positives, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

    # Cosine similarity matrix: (n_queries, n_passages)
    sim_matrix = q_embs @ p_embs.T

    mrr_scores = []
    for i in range(len(queries)):
        scores = sim_matrix[i]
        ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        rank   = ranked.index(i) + 1  # 1-indexed rank of the true positive
        mrr_scores.append(1.0 / rank if rank <= top_k else 0.0)

    return float(sum(mrr_scores) / len(mrr_scores))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(config: Optional[EmbeddingGemmaTrainConfig] = None) -> None:
    if not _DEPS_AVAILABLE:
        print(
            "ERROR: Fine-tuning dependencies not installed.\n"
            "Run: pip install -r requirements-finetune.txt"
        )
        return

    if config is None:
        config = EmbeddingGemmaTrainConfig()

    if not Path(config.data_path).exists():
        print(f"  Dataset not found: {config.data_path}")
        print("  Run: python -m finetune.data_prep first")
        return

    print(f"\n{'='*60}")
    print("Fine-tuning embeddinggemma — semantic retrieval")
    print(f"{'='*60}")
    print(f"  Batch size: {config.batch_size}  (larger = stronger in-batch negatives)")
    print(f"  Epochs:     {config.num_epochs}")

    # Load base model as SentenceTransformer
    print(f"\n  Loading {config.base_model}…")
    model = SentenceTransformer(config.base_model)
    model.max_seq_length = config.max_seq_length

    # Load dataset — detect pair vs. triplet format
    print(f"  Loading examples: {config.data_path}")
    pair_examples, triplet_examples = _load_examples(config.data_path)
    total = len(pair_examples) + len(triplet_examples)

    if total == 0:
        print("  No examples found — aborting.")
        return

    print(f"  Pair examples:    {len(pair_examples)}")
    print(f"  Triplet examples: {len(triplet_examples)}")

    # Build train objectives — one per format present in the data
    train_objectives = []

    if pair_examples:
        # Hold out 10% of pairs for MRR evaluation (shuffle for random split)
        shuffled_pairs = list(pair_examples)
        random.Random(42).shuffle(shuffled_pairs)
        split_idx  = max(1, int(len(shuffled_pairs) * 0.9))
        train_pairs = shuffled_pairs[:split_idx]
        eval_pairs  = shuffled_pairs[split_idx:]
        pair_loader = DataLoader(train_pairs, shuffle=True, batch_size=config.batch_size)
        mnrl_loss   = losses.MultipleNegativesRankingLoss(model=model)
        train_objectives.append((pair_loader, mnrl_loss))
        print(f"\n  Loss (pairs):    MultipleNegativesRankingLoss  [{len(train_pairs)} train / {len(eval_pairs)} eval]")
    else:
        eval_pairs = []

    if triplet_examples:
        triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=config.batch_size)
        triplet_loss   = losses.TripletLoss(model=model)
        train_objectives.append((triplet_loader, triplet_loss))
        print(f"  Loss (triplets): TripletLoss  [{len(triplet_examples)} examples]")

    # Baseline MRR before training
    baseline_mrr = 0.0
    if eval_pairs:
        baseline_mrr = _evaluate_retrieval(model, eval_pairs)
        print(f"\n  Baseline MRR@10: {baseline_mrr:.4f}")

    # Optional InformationRetrievalEvaluator for tracking during training
    evaluator = None
    if eval_pairs and config.eval_during_training:
        try:
            from sentence_transformers.evaluation import InformationRetrievalEvaluator

            queries  = {str(i): ex.texts[0] for i, ex in enumerate(eval_pairs)}
            corpus   = {str(i): ex.texts[1] for i, ex in enumerate(eval_pairs)}
            relevant = {str(i): {str(i)} for i in range(len(eval_pairs))}

            evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant,
                name="retrieval-eval",
                show_progress_bar=False,
            )
        except Exception:
            pass  # evaluator is optional

    # Wrap evaluator to track scores for convergence monitoring
    eval_history: list[dict] = []
    if evaluator is not None:
        _base_evaluator = evaluator

        class _TrackingEvaluator:
            """Wraps InformationRetrievalEvaluator to capture scores.

            Implements __iter__ for sentence-transformers >=3.x which
            may iterate over the evaluator (expecting a list-like).
            """
            def __init__(self, base):
                self._base = base
            def __call__(self, model, output_path="", epoch=-1, steps=-1, **kwargs):
                result = self._base(model, output_path, epoch, steps, **kwargs)
                # sentence-transformers >=3.x returns a dict of metrics;
                # extract the primary score for convergence tracking.
                if isinstance(result, dict):
                    score = result.get("eval_sequential_score",
                            result.get("eval_retrieval-eval_cosine_mrr@10", 0.0))
                    score = float(score) if score is not None else 0.0
                else:
                    score = float(result)
                eval_history.append({"epoch": epoch, "steps": steps, "score": score})
                return result
            def __iter__(self):
                return iter([self._base])

        evaluator = _TrackingEvaluator(_base_evaluator)

    # Determine evaluation_steps from the largest loader
    all_loaders = [obj[0] for obj in train_objectives]
    max_loader_len = max(len(ldr) for ldr in all_loaders) if all_loaders else 1
    eval_steps = max(1, max_loader_len // 2) if evaluator else 0

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Backup existing model before overwriting ---
    from finetune._scenario import EMBEDDING_GGUF_FT
    gguf_path = EMBEDDING_GGUF_FT
    if os.path.isfile(gguf_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{gguf_path}.bak.{ts}"
        shutil.copy2(gguf_path, backup)
        print(f"\n  Backed up existing GGUF → {backup}")

    # Train
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"\n  Training for {config.num_epochs} epochs…")

    model.fit(
        train_objectives=train_objectives,
        epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=config.output_dir,
        evaluator=evaluator,
        evaluation_steps=eval_steps,
        show_progress_bar=config.show_progress,
        save_best_model=True,
    )

    print(f"\n  Model saved → {config.output_dir}")

    # --- Convergence summary ---
    if eval_history:
        from finetune.training_utils import save_training_curves
        # Convert eval_history to the standard curves format
        curves = [{"step": e["steps"], "epoch": e["epoch"], "eval_score": e["score"]} for e in eval_history]
        save_training_curves(curves, config.output_dir, "embeddinggemma")

        best_entry = max(eval_history, key=lambda e: e["score"])
        print(f"\n  {'='*50}")
        print(f"  Convergence Summary — embeddinggemma")
        print(f"  {'='*50}")
        print(f"  Eval checkpoints: {len(eval_history)}")
        if len(eval_history) >= 2:
            print(f"  First eval score: {eval_history[0]['score']:.4f}")
            print(f"  Final eval score: {eval_history[-1]['score']:.4f}")
        print(f"  Best eval score:  {best_entry['score']:.4f} (epoch {best_entry['epoch']})")
        print(f"  {'='*50}")

    # Final MRR
    if eval_pairs:
        final_model = SentenceTransformer(config.output_dir)
        final_mrr   = _evaluate_retrieval(final_model, eval_pairs)
        improvement = (final_mrr - baseline_mrr) / baseline_mrr * 100 if baseline_mrr > 0 else 0
        print(f"\n  Final MRR@10:    {final_mrr:.4f}  (N={len(eval_pairs)} pairs, in-distribution slice)")
        print(f"  Improvement:     +{improvement:.1f}%")
        print(f"\n  ⚠️  This MRR is a training-loop diagnostic — a held-out 10% slice")
        print(f"     of the SAME shuffled training pairs, with the eval corpus being")
        print(f"     just those held-out positives. It is NOT a production-grade number.")
        print(f"     For the headline benchmark, run:")
        print(f"       python -m finetune.eval_embeddinggemma")
        print(f"     against the official 25-query / 26-passage eval set.")

    _print_next_steps(config)


def _print_next_steps(config: EmbeddingGemmaTrainConfig) -> None:
    print(f"""
  ─────────────────────────────────────────────────────
  Next: convert to GGUF and serve with llama-server

    bash finetune/convert_embeddinggemma_to_gguf.sh

  Restart all servers with fine-tuned models:

    bash scripts/start_servers.sh --bg --ft

  Re-index vector store with fine-tuned embeddings:

    python -m data.loader

  Evaluate retrieval improvement:

    python -m finetune.eval_embeddinggemma --save results/finetuned_embeddinggemma.json
    python -m finetune.eval_embeddinggemma --compare \\
        results/baseline_embeddinggemma.json results/finetuned_embeddinggemma.json
  ─────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# Quick test — encode a few queries without full training
# ---------------------------------------------------------------------------

def test_embeddings(model_path: str = "./models/embeddinggemma-300m-ft-merged") -> None:
    """Sanity-check the fine-tuned model by ranking a few queries."""
    if not _DEPS_AVAILABLE:
        print("sentence-transformers not installed")
        return

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    model = SentenceTransformer(model_path)

    queries = [
        "enterprise plan pricing",
        "LoRA fine-tuning support",
        "Kubernetes deployment",
    ]
    passages = [
        "Enterprise plan starts at €3,500/month with unlimited users and 24/7 support.",
        "Professional and Enterprise plans include LoRA and QLoRA fine-tuning.",
        "Nextera supports Kubernetes via Helm chart with auto-scaling.",
        "Starter plan costs €299/month for up to 5 users.",
    ]

    q_embs = model.encode(queries,   normalize_embeddings=True)
    p_embs = model.encode(passages,  normalize_embeddings=True)
    sim    = q_embs @ p_embs.T

    print("\nSimilarity matrix (queries × passages):")
    print(f"{'':30s}", end="")
    for p in passages:
        print(f"  {p[:25]:25s}", end="")
    print()
    for i, q in enumerate(queries):
        print(f"{q:30s}", end="")
        for j in range(len(passages)):
            print(f"  {sim[i][j]:8.4f}             ", end="")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune embeddinggemma with contrastive learning")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--test",       action="store_true", help="Test the fine-tuned model")
    args = parser.parse_args()

    if args.test:
        test_embeddings()
    else:
        cfg = EmbeddingGemmaTrainConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        train(config=cfg)
