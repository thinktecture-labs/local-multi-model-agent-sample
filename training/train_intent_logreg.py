"""Train a LogisticRegression intent classifier on embeddinggemma embeddings.

Replaces the generative gemma3-1B classification step with a fully
deterministic, <25ms classifier: embed query → LogReg predict.

Usage:
    python -m training.train_intent_logreg                  # default settings
    python -m training.train_intent_logreg --max-iter 2000  # tune regularization

Produces:
    models/intent-logreg/model.joblib   — trained sklearn classifier
    models/intent-logreg/meta.json      — class labels, embedding dim, training stats

Requires:
    - embeddinggemma server running on port 9092 (or EMBEDDING_PORT env var)
    - scikit-learn, joblib (pip install scikit-learn)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.inference.client import SmallLanguageModelClient
from src.engine.inference.config import SCENARIO_CONFIG


# ---------------------------------------------------------------------------
# Paths (scenario-aware)
# ---------------------------------------------------------------------------

from finetune._scenario import TRAINING_DIR as _TRAINING_DIR, SUFFIX as _SUFFIX, LOGREG_MODEL_DIR as _LOGREG_DIR

TRAINING_DATA = Path(f"{_TRAINING_DIR}/gemma3_intent{_SUFFIX}.jsonl")
HARD_NEGATIVES = Path(f"{_TRAINING_DIR}/intent_hard_negatives{_SUFFIX}.jsonl")
HOLDOUT_DATA = Path(f"{_TRAINING_DIR}/intent_eval_holdout{_SUFFIX}.jsonl")
OUTPUT_DIR = Path(_LOGREG_DIR)

MODEL_PATH = OUTPUT_DIR / "model.joblib"
META_PATH = OUTPUT_DIR / "meta.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(path: Path) -> tuple[list[str], list[str]]:
    """Load intent training data from JSONL. Returns (texts, labels)."""
    valid_labels = {"rag_query", "tool_use", "direct_answer"}
    texts: list[str] = []
    labels: list[str] = []
    skipped = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            label = entry["output"]
            if label not in valid_labels:
                skipped += 1
                continue
            texts.append(entry["input"])
            labels.append(label)
    if skipped:
        print(f"  Skipped {skipped} entries with invalid labels")
    return texts, labels


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def encode_texts(
    client: SmallLanguageModelClient,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts via embeddinggemma API in batches."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = await client.embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    labels: list[str],
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train LogisticRegression and report cross-validation accuracy."""
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=75)

    # 5-fold cross-validation
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean() * 100:.1f}% ± {scores.std() * 100:.1f}%")
    print(f"  Per-fold: {', '.join(f'{s*100:.1f}%' for s in scores)}")

    # Train final model on all data
    clf.fit(X, labels)
    print(f"  Classes: {list(clf.classes_)}")
    print(f"  Training accuracy: {clf.score(X, labels) * 100:.1f}%")

    return clf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(max_iter: int = 1000) -> None:
    print("=" * 60)
    print("  Intent LogReg Classifier Training")
    print("=" * 60)

    # 1. Load training data + hard negatives
    print(f"\n[1/4] Loading training data...")
    texts, labels = load_training_data(TRAINING_DATA)
    print(f"  Base: {len(texts)} from {TRAINING_DATA.name}")
    if HARD_NEGATIVES.exists():
        hn_texts, hn_labels = load_training_data(HARD_NEGATIVES)
        texts.extend(hn_texts)
        labels.extend(hn_labels)
        print(f"  Hard negatives: +{len(hn_texts)} from {HARD_NEGATIVES.name}")
    label_counts = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"  Total: {len(texts)} examples: {label_counts}")

    holdout_texts, holdout_labels = [], []
    if HOLDOUT_DATA.exists():
        holdout_texts, holdout_labels = load_training_data(HOLDOUT_DATA)
        print(f"  Holdout (blind eval): {len(holdout_texts)} from {HOLDOUT_DATA.name}")

    # 2. Encode via embeddinggemma
    print("\n[2/4] Encoding utterances via embeddinggemma...")
    client = SmallLanguageModelClient.create_with_auto_detection()
    t0 = time.perf_counter()
    all_texts = texts + holdout_texts
    X_all = await encode_texts(client, all_texts)
    encode_ms = (time.perf_counter() - t0) * 1000
    X = X_all[: len(texts)]
    X_holdout = X_all[len(texts) :]
    print(f"  Encoded {len(all_texts)} texts in {encode_ms:.0f}ms")
    print(f"  Embedding dimension: {X.shape[1]}")

    # 3. Train classifier
    print("\n[3/4] Training LogisticRegression...")
    t0 = time.perf_counter()
    clf = train(X, labels, max_iter=max_iter)
    train_ms = (time.perf_counter() - t0) * 1000
    print(f"  Trained in {train_ms:.0f}ms")

    # Holdout evaluation (blind — these examples never seen during training)
    if len(holdout_texts) > 0:
        holdout_acc = clf.score(X_holdout, holdout_labels)
        preds = clf.predict(X_holdout)
        wrong = [(holdout_texts[i], holdout_labels[i], preds[i])
                 for i in range(len(holdout_texts)) if preds[i] != holdout_labels[i]]
        print(f"\n  Holdout accuracy (blind): {holdout_acc * 100:.1f}%  ({len(holdout_texts) - len(wrong)}/{len(holdout_texts)})")
        if wrong:
            print(f"  Misclassified ({len(wrong)}):")
            for text, true, pred in wrong:
                print(f"    ✗ got={pred:15s} expected={true:15s} | {text[:60]}")

    # 4. Save
    print("\n[4/4] Saving model...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"  Model → {MODEL_PATH}")

    meta = {
        "classes": list(clf.classes_),
        "embedding_dim": int(X.shape[1]),
        "n_training_examples": len(texts),
        "label_distribution": label_counts,
        "cv_accuracy_mean": float(cross_val_score(clf, X, labels, cv=5).mean()),
        "holdout_accuracy": float(clf.score(X_holdout, holdout_labels)) if len(holdout_texts) > 0 else None,
        "n_holdout_examples": len(holdout_texts),
        "max_iter": max_iter,
        # Coupling guard: the LogReg weights are only valid against the exact
        # embedder used to produce these training vectors. The load-time check
        # in intent_classifier_logreg.py compares this against the running
        # scenario's embedding_gguf_ft and refuses to load on mismatch.
        "embedding_model": SCENARIO_CONFIG.embedding_gguf_ft,
        "scenario": SCENARIO_CONFIG.name,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"  Meta  → {META_PATH}")
    print("\n  Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LogReg intent classifier")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max solver iterations")
    args = parser.parse_args()
    asyncio.run(main(max_iter=args.max_iter))
