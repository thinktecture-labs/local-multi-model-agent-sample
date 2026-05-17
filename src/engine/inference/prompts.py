"""
All LLM prompt strings in one place.

Import from here — never inline prompts in handler code.
"""

from .config import SCENARIO_CONFIG

# ─── Direct answer ────────────────────────────────────────────────────────────
DIRECT_ANSWER_SYSTEM_PROMPT = SCENARIO_CONFIG.direct_answer_system_prompt

# ─── RAG ──────────────────────────────────────────────────────────────────────
RAG_REWRITE_PROMPT_TEMPLATE = SCENARIO_CONFIG.rag_rewrite_prompt

RAG_SYNTHESIS_SYSTEM_PROMPT = SCENARIO_CONFIG.rag_synthesis_system_prompt

if SCENARIO_CONFIG.language == "de":
    RAG_SYNTHESIS_USER_TEMPLATE = (
        "QUELLEN:\n{context}\n\n"
        "FRAGE: {query}\n\n"
        "Antworte auf Deutsch. Nenne alle relevanten Details aus den Quellen. "
        "Zitiere konkrete Zahlen, Werte und Fakten exakt wie sie in den Quellen stehen. "
        "Verwende keine Informationen aus quellfremden Abschnitten."
    )
else:
    RAG_SYNTHESIS_USER_TEMPLATE = (
        "SOURCES:\n{context}\n\n"
        "QUESTION: {query}\n\n"
        "Include all items from the most relevant source. "
        "Quote specific numbers and values exactly as they appear in the sources. "
        "Do not pull in details from unrelated sources."
    )


def build_rag_messages(docs: list, query: str) -> tuple[list[dict], str]:
    """Build the messages list and context string for RAG synthesis.

    Returns (messages, context) so callers can log context_docs count.
    Shared by RAGHandler, _query_uploaded_document, and _stream_document_chat.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('title', doc.id)}]\n{doc.content}"
        for doc in docs
    )
    messages = [
        {"role": "system", "content": RAG_SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_SYNTHESIS_USER_TEMPLATE.format(context=context, query=query)},
    ]
    return messages, context

# ─── Tool use ─────────────────────────────────────────────────────────────────
TOOL_FORMAT_PROMPT_TEMPLATE = SCENARIO_CONFIG.tool_format_prompt

MULTI_STEP_SYNTHESIS_PROMPT_TEMPLATE = SCENARIO_CONFIG.multi_step_synthesis_prompt

# ─── Data extraction ─────────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = SCENARIO_CONFIG.extraction_system_prompt

EXTRACTION_USER_TEMPLATE = (
    "Source filename: {source_document}\n\n"
    "Extract financial metrics from this document:\n\n"
    "{text}\n\n"
    "Output a single JSON object with the fields: company, fiscal_year, revenue, "
    "revenue_growth_pct, nrr, customers_1m_plus, total_customers, product_revenue, "
    "gross_margin_pct, free_cash_flow."
)
