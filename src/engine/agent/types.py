"""
Shared data types and constants for the agent pipeline.

This module is the single source of truth for:
  - Intent taxonomy (Intent enum)
  - Execution trace types (ExecutionStep, AgentResponse)
  - Classification prompt template (CLASSIFY_PROMPT)

All agent sub-modules and eval scripts import from here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class Intent(Enum):
    """
    Categories of user intent — 4-way classification.

    Image routing is deterministic (presence of images → IMAGE_QUERY).
    For text-only queries, gemma3 handles the routing decision:
      RAG_QUERY     → embeddinggemma retrieves docs, gemma3 synthesizes
      TOOL_USE      → the tool-calling model picks calculator or sql_query
      DIRECT_ANSWER → gemma3 responds directly

    This architecture gives each model a clear, achievable task:
    gemma3 separates "knowledge" from "data" from "chitchat",
    the tool-calling model separates "arithmetic" from "SQL",
    gemma3-4B handles all visual understanding.
    """
    RAG_QUERY     = "rag_query"      # Needs knowledge-base search
    TOOL_USE      = "tool_use"       # Needs calculator or SQL query
    DIRECT_ANSWER = "direct_answer"  # Answerable without tools
    IMAGE_QUERY   = "image_query"    # Has image(s) → vision model


# ---------------------------------------------------------------------------
# Classification prompt — shared with eval scripts
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = (
    "Classify the user intent into: rag_query, tool_use, "
    "direct_answer\n\n{query}"
)

INTENT_LABELS = ["rag_query", "tool_use", "direct_answer"]


# ---------------------------------------------------------------------------
# Execution trace types
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStep:
    """One step in the agent's execution trace."""
    action: str
    model: str
    details: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class AgentResponse:
    """Complete, structured response from the agent pipeline."""
    query:            str
    intent:           Intent
    response:         str
    steps:            list[ExecutionStep] = field(default_factory=list)
    success:          bool  = True
    execution_time_ms: float = 0.0
    total_tokens:     int   = 0
    prompt_tokens:    int   = 0
    completion_tokens: int  = 0
    request_id:       str   = field(default_factory=_new_request_id)

    def __post_init__(self):
        self.total_tokens = sum(s.tokens_used for s in self.steps)
        self.prompt_tokens = sum(s.prompt_tokens for s in self.steps)
        self.completion_tokens = sum(s.completion_tokens for s in self.steps)
