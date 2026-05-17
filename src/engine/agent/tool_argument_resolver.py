"""
Helpers for normalising tool arguments before/after the function model.

Earlier iterations also defined ExpressionResolver / SQLResolver protocols
and Null implementations used to gate deterministic pre-routers. Those have
been removed — Qwen3.5-4B FT handles tool selection and argument generation
natively. See src/engine/scaffolding/README.md for the history.
"""

from __future__ import annotations

import re


def normalize_tool_query(query: str) -> str:
    """Normalize a query before passing to the function model.

    Strip currency symbols, replace Unicode operators, remove
    digit-grouping commas, and expand contractions.
    """
    normalized = query
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = normalized.replace("$", "").replace("€", "").replace("£", "")
    normalized = re.sub(r"(\d),(\d{3})", r"\1\2", normalized)
    normalized = re.sub(r"(?i)\bwhat'?s\b", "What is", normalized)
    return normalized


def rephrase_for_sql(step: str) -> str:
    """Rephrase a decomposed step description for SQL routing.

    The decomposition prompt produces imperative phrases like
    "Find the best product...", "Count total customers", "Look up revenue...".
    This normalises them to interrogative form ("How many...", "Show...").
    """
    s = step.strip().rstrip(".")

    # "Count X" → "How many X are there?"
    m = re.match(r"(?i)^count\s+(.+)", s)
    if m:
        return f"How many {m.group(1)} are there?"

    # "Find/Look up/Fetch/Retrieve/Get X" → "Show X."
    m = re.match(r"(?i)^(?:find|look\s*up|fetch|retrieve|get)\s+(.+)", s)
    if m:
        return f"Show {m.group(1)}."

    return step


def patch_calculator_expression(tool_args: dict, normalized: str) -> dict:
    """Fix incomplete calculator expressions from the function model.

    If the model returned a bare percentage like "15%" but the query
    says "15% of 45000", reconstruct the full expression.
    """
    if "expression" not in tool_args:
        return tool_args
    expr = tool_args["expression"].strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?%", expr):
        m = re.search(
            r"(\d[\d,.]*)\s*%\s*(?:of|×|x|\*)\s*\$?€?£?(\d[\d,.]*)",
            normalized, re.IGNORECASE,
        )
        if m:
            tool_args["expression"] = f"{m.group(1)}% of {m.group(2)}"
    return tool_args
