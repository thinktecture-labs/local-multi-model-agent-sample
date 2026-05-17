"""
LogReg intent classifier — fully deterministic, <25ms.

Uses a pre-trained scikit-learn LogisticRegression model on embeddinggemma
embeddings. Same input → same output, every time. No temperature, no
sampling, no prompt sensitivity.

Falls back to the generative IntentClassifier if the model file is missing
or the embedding server is unreachable.

Train first:  python -m training.train_intent_logreg

COUPLING: This classifier depends on the exact embedding model (embeddinggemma-ft).
Changing the embedding model (swap, re-fine-tune, quantization change) will produce
different embedding vectors, making the trained LogReg weights invalid. Any embedding
model change requires retraining the classifier: python -m training.train_intent_logreg.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .types import Intent

if TYPE_CHECKING:
    from ..inference.client import SmallLanguageModelClient, LLMResponse

from ..inference.config import SCENARIO_CONFIG

logger = logging.getLogger(__name__)

MODEL_PATH = Path(SCENARIO_CONFIG.logreg_model_dir) / "model.joblib"
META_PATH = Path(SCENARIO_CONFIG.logreg_model_dir) / "meta.json"

# Low-confidence predictions are rerouted to DIRECT_ANSWER (safe fallback).
# Adversarial/OOD inputs tend to produce diffuse probability distributions
# (max ~0.33–0.45). Genuine boundary cases (e.g. "price difference between
# plans") can land at 0.63–0.64 — still clearly a single winner. Threshold
# lowered from 0.65 → 0.60 to pass these without admitting true OOD inputs.
CONFIDENCE_THRESHOLD = 0.60

# Label string → Intent enum mapping
_LABEL_TO_INTENT = {
    "rag_query": Intent.RAG_QUERY,
    "tool_use": Intent.TOOL_USE,
    "direct_answer": Intent.DIRECT_ANSWER,
}


class LogRegIntentClassifier:
    """Deterministic intent classifier using LogReg on embedding vectors."""

    def __init__(self, client: SmallLanguageModelClient) -> None:
        self._client = client
        self._clf = None
        self._available = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained classifier from disk.

        Refuses to load when meta.json records a different `embedding_model`
        than the running scenario's `embedding_gguf_ft`: LogReg weights are
        only valid against the exact embedder used at training time. A mismatch
        means silently degraded accuracy; falling back to the generative
        classifier is safer than serving wrong intents.

        Trust model: `joblib.load` uses Python pickle under the hood and will
        execute arbitrary code embedded in a malicious `model.joblib`. In this
        repo the file is produced locally by `training.train_intent_logreg`
        and committed alongside its `meta.json` — treat it like source code,
        not an opaque artifact. **Never** load a `model.joblib` from an
        untrusted source (PR, fork, download). See SECURITY.md.
        """
        if not MODEL_PATH.exists():
            logger.info(
                "LogReg intent model not found at %s — will use generative fallback",
                MODEL_PATH,
            )
            return
        if not self._check_embedding_match():
            return
        try:
            import joblib
            self._clf = joblib.load(MODEL_PATH)
            self._available = True
            logger.info(
                "LogReg intent classifier loaded: classes=%s",
                list(self._clf.classes_),
            )
        except Exception:
            logger.exception("Failed to load LogReg intent model")

    def _check_embedding_match(self) -> bool:
        """Verify meta.json's embedding_model matches the running scenario.

        Returns True if the check passes or is inapplicable (legacy meta
        without the field — logs a warning). Returns False on mismatch,
        forcing the generative fallback.
        """
        if not META_PATH.exists():
            logger.warning(
                "LogReg meta.json not found at %s — cannot verify embedding "
                "coupling; loading anyway (consider retraining)",
                META_PATH,
            )
            return True
        try:
            meta = json.loads(META_PATH.read_text())
        except Exception:
            logger.exception("Failed to read LogReg meta.json")
            return True
        recorded = meta.get("embedding_model")
        if recorded is None:
            logger.warning(
                "LogReg meta.json is missing `embedding_model` field — "
                "cannot verify embedding coupling; loading anyway "
                "(retrain to populate: python -m training.train_intent_logreg)"
            )
            return True
        if recorded != SCENARIO_CONFIG.embedding_gguf_ft:
            logger.error(
                "LogReg embedding mismatch: model was trained on %r but the "
                "running scenario uses %r. Refusing to load — predictions "
                "would be silently wrong. Retrain: "
                "python -m training.train_intent_logreg",
                recorded, SCENARIO_CONFIG.embedding_gguf_ft,
            )
            return False
        return True

    @property
    def available(self) -> bool:
        """Whether the LogReg model is loaded and ready."""
        return self._available

    async def classify(self, query: str) -> tuple[Intent, LLMResponse | None]:
        """
        Classify intent using LogReg on the embedding vector.

        Returns (intent, None) — the LLMResponse is None because no
        generative model was used. Token tracking returns zeros.
        """
        if not self._available:
            raise RuntimeError("LogReg model not loaded")

        # Embed the query via embeddinggemma
        embedding = await self._client.embed(query)
        X = np.array([embedding])

        # Predict
        label = self._clf.predict(X)[0]
        probas = self._clf.predict_proba(X)[0]
        confidence = float(max(probas))

        intent = _LABEL_TO_INTENT.get(label, Intent.DIRECT_ANSWER)

        # Low confidence → safe fallback to DIRECT_ANSWER.
        # OOD/adversarial inputs produce diffuse probability distributions
        # that rarely exceed the threshold for rag_query or tool_use.
        if confidence < CONFIDENCE_THRESHOLD and intent != Intent.DIRECT_ANSWER:
            logger.info(
                "LogReg low confidence (%.3f < %.2f) for %s — "
                "overriding to DIRECT_ANSWER",
                confidence, CONFIDENCE_THRESHOLD, intent.value,
            )
            intent = Intent.DIRECT_ANSWER

        logger.debug(
            "LogReg intent: %s (confidence=%.3f, probas=%s)",
            intent.value,
            confidence,
            {cls: f"{p:.3f}" for cls, p in zip(self._clf.classes_, probas)},
        )

        return intent, None
