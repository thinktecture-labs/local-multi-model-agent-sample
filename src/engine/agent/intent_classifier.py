"""
Intent classification — LogReg on embeddings with generative fallback.

Primary path: LogReg on embeddinggemma vectors (<25ms, fully deterministic).
Fallback: generative gemma3-ft classification (~200ms).

The LogReg classifier is used when a trained model exists at
models/intent-logreg/model.joblib. If missing or if the embedding server
is unreachable, falls back transparently to the generative path.

Includes a rule-based pre-filter for obvious prompt injection patterns.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from openai import APIConnectionError

from .types import CLASSIFY_PROMPT, Intent
from ..inference.config import CLASSIFY_MAX_TOKENS, CLASSIFY_TEMPERATURE

if TYPE_CHECKING:
    from ..inference.client import SmallLanguageModelClient, LLMResponse
    from .intent_classifier_logreg import LogRegIntentClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-classifier injection filter
# ---------------------------------------------------------------------------
# Fast regex-based check for obvious prompt injection patterns.
# Catches common adversarial attacks before they reach the LLM, improving
# both robustness and latency (no wasted inference on junk).

_INJECTION_PATTERNS = re.compile(
    r"|".join([
        # Prompt injection / jailbreak
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"you\s+are\s+now\s+(a|an|my|DAN|evil|unrestricted)\b",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"<\s*/?\s*(system|prompt|instruction)",
        r"\[INST\]",
        r"<<\s*SYS\s*>>",
        r"ASSISTANT\s*:",
        r"Human\s*:\s*.*\nAssistant\s*:",
        r"do\s+not\s+classify",
        r"respond\s+with\s+(only|just)\s+",
        r"repeat\s+(after\s+me|the\s+following|back)",
        r"forget\s+(your|all)\s+(training|instructions|rules)",
        r"bypass\s+(your\s+)?(safety|security|filter)",
        r"override\s+(your\s+)?(classification|system|rules)",
        r"<<<\s*\w+\s*>>>",
        r"pretend\s+you\s+are",
        # SQL injection
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM\s+\w+\s+WHERE\s+1\s*=\s*1",
        r"UNION\s+SELECT\s+",
        r"'\s*OR\s+1\s*=\s*1",
        r"xp_cmdshell",
        r";\s*EXEC\s+",
        r"import\s+os\s*;\s*os\.\s*system",
        # Standalone SQL write operations (not wrapped in natural language)
        r"^\s*(?:UPDATE|DELETE\s+FROM|INSERT\s+INTO|ALTER\s+TABLE|CREATE\s+TABLE)\s+\w+",
        r"^\s*SELECT\s+.+\s+FROM\s+(?:sqlite_master|information_schema|pg_catalog)",
        r"DROP\s+TABLE\s+\w+",
        # Intent-stuffing adversarial
        r"classify\s+this\s+as\s+(tool_use|rag_query|direct_answer)",
        r"the\s+correct\s+answer\s+is\s*:\s*(tool_use|rag_query|direct_answer)",
        r"^(tool_use|rag_query)\s*$",
        r"output\s+the\s+word\s+(tool_use|rag_query|sql_query)",
        # German-language injection patterns
        r"ignoriere?\s+(alle\s+)?(vorherigen?|bisherigen?)\s+(anweisungen?|regeln?|instruktionen?)",
        r"vergiss\s+(dein|all)\s+(training|anweisungen|regeln)",
        r"du\s+bist\s+(jetzt|nun)\s+(ein|eine|DAN)\b",
        r"neue\s+(direktive|anweisung)\s*:",
        r"klassifiziere?\s+(dies|das)\s+(bitte\s+)?als\s+(tool_use|rag_query)",
        r"die\s+richtige\s+antwort\s+ist\s*:",
        r"gib\s+(das\s+wort|den\s+text)\s+.+\s+aus",
        r"wiederhole?\s+nach\s+mir",
        r"umgehe?\s+(deine?\s+)?(sicherheit|filter|schutz)",
        r"tue?\s+so\s+als\s+w.+rst\s+du",
    ]),
    re.IGNORECASE,
)


def _looks_like_gibberish(query: str) -> bool:
    """Detect random character noise, symbol spam, or extreme repetition."""
    stripped = query.strip()
    if not stripped or len(stripped) < 5:
        return False
    # High ratio of non-alphanumeric chars (symbols/punctuation spam)
    alnum = sum(c.isalnum() or c.isspace() for c in stripped)
    if alnum / len(stripped) < 0.3:
        return True
    # Extreme repetition: same word/token repeated 5+ times
    words = stripped.lower().split()
    if len(words) >= 5 and len(set(words)) <= 2:
        return True
    return False


def _looks_like_non_latin_script(query: str) -> bool:
    """Detect queries dominated by non-Latin scripts (CJK, Cyrillic, Arabic, etc.).

    Latin-alphabet languages (French, German, Spanish, etc.) pass through —
    they share enough ASCII characters to stay above the 60% threshold.
    """
    stripped = query.strip()
    if len(stripped) < 5:
        return False
    ascii_chars = sum(1 for c in stripped if c.isascii())
    return ascii_chars / len(stripped) < 0.6


def _looks_like_injection(query: str) -> bool:
    """Return True if the query matches known injection patterns."""
    return bool(_INJECTION_PATTERNS.search(query))


def looks_like_adversarial(query: str) -> bool:
    """Check all adversarial heuristics: injection, gibberish, non-Latin script.

    Public API for use by the orchestrator and eval scripts.
    """
    return (
        _looks_like_injection(query)
        or _looks_like_gibberish(query)
        or _looks_like_non_latin_script(query)
    )


class IntentClassifier:
    """Classifies user queries into one of the Intent categories.

    Uses LogReg on embeddings when available (<25ms, deterministic).
    Falls back to generative gemma3-ft classification (~200ms).
    """

    def __init__(self, client: SmallLanguageModelClient) -> None:
        self._client = client
        self._logreg: LogRegIntentClassifier | None = None
        self._init_logreg(client)

    def _init_logreg(self, client: SmallLanguageModelClient) -> None:
        """Try to initialize the LogReg classifier."""
        try:
            from .intent_classifier_logreg import LogRegIntentClassifier
            logreg = LogRegIntentClassifier(client)
            if logreg.available:
                self._logreg = logreg
                logger.info("Intent classification: using LogReg (deterministic)")
            else:
                logger.info("Intent classification: using generative (LogReg model not found)")
        except Exception:
            logger.info("Intent classification: using generative (LogReg init failed)")

    @property
    def using_logreg(self) -> bool:
        """Whether the fast LogReg path is active."""
        return self._logreg is not None

    async def classify(self, query: str) -> tuple[Intent, LLMResponse | None]:
        """
        Classify user intent.

        1. Rule-based injection filter (regex, <1ms)
        2. LogReg on embeddings if available (<25ms, deterministic)
        3. Generative gemma3-ft fallback (~200ms)

        Returns (intent, llm_response) — response is None for LogReg path.
        """
        if looks_like_adversarial(query):
            logger.warning(
                "Adversarial pattern detected in query: %r — routing to DIRECT_ANSWER",
                query[:80],
            )
            return Intent.DIRECT_ANSWER, None

        # Fast path: LogReg on embeddings
        if self._logreg is not None:
            try:
                return await self._logreg.classify(query)
            except Exception:
                logger.warning(
                    "LogReg classification failed, falling back to generative",
                    exc_info=True,
                )

        # Slow path: generative classification
        return await self._classify_generative(query)

    async def _classify_generative(self, query: str) -> tuple[Intent, LLMResponse | None]:
        """Generative classification using gemma3-ft."""
        try:
            response = await self._client.generate(
                prompt=CLASSIFY_PROMPT.format(query=query),
                temperature=CLASSIFY_TEMPERATURE,
                max_tokens=CLASSIFY_MAX_TOKENS,
                deterministic=True,
            )
        except APIConnectionError:
            logger.warning(
                "Intent classification failed: model server unreachable, "
                "falling back to DIRECT_ANSWER"
            )
            return Intent.DIRECT_ANSWER, None

        # Take first line only — the 1B model sometimes appends extra text
        first_line = response.content.strip().split("\n")[0]
        intent_str = first_line.strip().lower().replace(" ", "_")
        try:
            return Intent(intent_str), response
        except ValueError:
            logger.warning(
                "Intent classification fallback: model returned %r, not a valid "
                "intent — defaulting to DIRECT_ANSWER",
                intent_str,
            )
            return Intent.DIRECT_ANSWER, response
