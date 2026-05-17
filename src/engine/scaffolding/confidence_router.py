"""
ConfidenceRouter — Score response confidence and decide on cloud escalation.

Uses text-based heuristics (no logprobs needed) to estimate how confident
the local model is in its response. When confidence falls below threshold
in hybrid routing mode, the query is escalated to a cloud LLM.

Ported from local-and-edge-ai/samples/hybrid-ai-inference, simplified for
the Observatory agent pipeline.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ConfidenceResult:
    """Result of confidence assessment on a local response."""
    score: float                               # 0.0–1.0
    factors: dict[str, float] = field(default_factory=dict)
    should_escalate: bool = False

    @property
    def score_pct(self) -> int:
        return round(self.score * 100)


# ---------------------------------------------------------------------------
# Scoring heuristics (8 factors)
# ---------------------------------------------------------------------------

_STRONG_UNCERTAINTY = [
    "i don't know", "i do not know",
    "i'm not sure", "i am not sure",
    "i cannot answer", "i can't answer",
    "i don't have information", "i do not have information",
    "beyond my knowledge", "outside my knowledge",
    "i'm unable to", "i am unable to",
    "i cannot provide", "i can't provide",
    "no way of knowing",
    "cannot predict", "can't predict",
    "no definitive answer",
    "subjective", "depends on", "varies",
    # Negation patterns — model says "the document doesn't contain X"
    "does not provide", "does not contain",
    "does not mention", "does not include",
    "does not specify", "does not state",
    # Plural / present-tense variants — "the sources do not provide ..."
    "do not provide", "do not contain",
    "do not mention", "do not include",
    "do not specify", "do not state",
    # Contractions
    "doesn't provide", "doesn't contain",
    "doesn't mention", "doesn't include",
    "doesn't specify", "doesn't state",
    "don't provide", "don't contain",
    "don't mention", "don't include",
    "don't specify", "don't state",
    "no information about", "no information on", "no information regarding",
    "no details about", "no details on",
    "not mentioned", "not provided",
    "not specified", "not stated",
    "there is no", "there are no",
]

_HEDGING = [
    r"\bmight\b", r"\bcould be\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\bmaybe\b", r"\bprobably\b", r"\bi think\b", r"\bi believe\b",
    r"\bit seems\b", r"\bit appears\b", r"\bgenerally\b", r"\btypically\b",
    r"\busually\b", r"\boften\b",
]

_APOLOGIES = [
    "i apologize", "i'm sorry", "i am sorry",
    "please note", "disclaimer", "important to note",
    "keep in mind", "be aware",
]

_CLARIFICATION = [
    r"could you clarify", r"can you clarify",
    r"what do you mean", r"could you provide more",
    r"do you mean", r"are you asking",
]

_CONFIDENT = [
    r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b",
    r"\bthe answer is\b", r"\byes,\b", r"\bno,\b",
    r"\bcorrect\b", r"\bexactly\b",
]


def score_confidence(
    response: str,
    query: str,
    threshold: float = 0.6,
    context_docs: list[str] | None = None,
) -> ConfidenceResult:
    """
    Score the confidence of a local model response using 8-factor heuristics.

    Returns a ConfidenceResult with the score, factor breakdown, and
    whether the response should be escalated to a cloud model.
    """
    factors: dict[str, float] = {}
    text = response.lower()

    # Base confidence — empirical baseline: well-formed local-model responses typically
    # score 0.70–0.95. Cloud escalation triggers below 0.60 (configurable via POST /routing-mode).
    base = 0.85
    factors["base"] = base

    # Factor 1: Strong uncertainty phrases (-0.35)
    factors["strong_uncertainty"] = 0.0
    for phrase in _STRONG_UNCERTAINTY:
        if phrase in text:
            factors["strong_uncertainty"] = -0.35
            break

    # Factor 2: Hedging language (-0.03 each, max -0.15)
    hedge_count = sum(1 for p in _HEDGING if re.search(p, text))
    factors["hedging"] = -min(0.15, hedge_count * 0.03)

    # Factor 3: Apology/disclaimer (-0.10)
    factors["apology"] = 0.0
    for phrase in _APOLOGIES:
        if phrase in text:
            factors["apology"] = -0.10
            break

    # Factor 4: Asks for clarification (-0.20)
    factors["clarification"] = 0.0
    for pattern in _CLARIFICATION:
        if re.search(pattern, text):
            factors["clarification"] = -0.20
            break

    # Factor 5: Response length vs query complexity
    words = len(response.split())
    query_words = len(query.split())
    if query_words > 10 and words < 10:
        factors["response_length"] = -0.15
    elif words < 5:
        factors["response_length"] = -0.10
    else:
        factors["response_length"] = 0.0

    # Factor 6: Confident language (bonus, +0.05 each, max +0.10)
    confident_count = sum(1 for p in _CONFIDENT if re.search(p, text))
    factors["confident_language"] = min(0.10, confident_count * 0.05) if confident_count else 0.0

    # Factor 7: Structured response (bonus, +0.05)
    has_structure = bool(
        re.search(r"^\s*[-*]\s", response, re.MULTILINE)
        or re.search(r"^\s*\d+[.)]\s", response, re.MULTILINE)
        or "```" in response
    )
    factors["structured"] = 0.05 if has_structure else 0.0

    # Factor 8: Entity grounding — do key query entities appear in retrieved context?
    # Catches hallucination when the model invents answers about entities not in the KB.
    factors["entity_grounding"] = 0.0
    if context_docs is not None:
        context_blob = " ".join(context_docs).lower()
        # Extract proper nouns and named entities from the query.
        # Strategy: find individual capitalized words, then also all-caps acronyms.
        _SKIP = {"what", "how", "why", "when", "where", "which", "who",
                 "compare", "show", "list", "the", "and", "for", "with",
                 "between", "about", "does", "from", "into", "that",
                 "this", "their", "there", "these", "those", "have",
                 "not", "can", "are", "was", "were",
                 "calculate", "find", "get", "tell", "explain",
                 "describe", "give", "help", "should", "would", "could"}
        # Capitalized words (proper nouns)
        caps = re.findall(r"\b[A-Z][a-z]{2,}\b", query)
        # All-caps acronyms (AWS, HIPAA, etc.)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", query)
        entities = [w for w in caps + acronyms if w.lower() not in _SKIP]
        if entities:
            found = sum(1 for e in entities if e.lower() in context_blob)
            ratio = found / len(entities)
            if ratio < 0.34:
                factors["entity_grounding"] = -0.40  # most entities missing
            elif ratio < 0.67:
                factors["entity_grounding"] = -0.25  # significant entities missing

    # Final score
    adjustment = sum(v for k, v in factors.items() if k != "base")
    score = max(0.1, min(1.0, base + adjustment))

    return ConfidenceResult(
        score=round(score, 3),
        factors=factors,
        should_escalate=score < threshold,
    )
