"""
Data preparation for embeddinggemma — contrastive retrieval pairs.
"""

from __future__ import annotations

import os
import random

from finetune.data_prep_shared import load_interactions, save_jsonl


# ---------------------------------------------------------------------------
# Contrastive retrieval dataset (embeddinggemma)
# ---------------------------------------------------------------------------

# Hard-coded positive pairs (query, relevant passage) for retrieval training.
# The contrastive loss treats other passages in the batch as negatives.
_RETRIEVAL_PAIRS: list[dict[str, str]] = [
    {
        "query": "enterprise plan features unlimited storage",
        "positive": (
            "The Nextera Enterprise plan starts at €3,500 per month. "
            "It includes: unlimited concurrent users, unlimited vector storage, "
            "support for all model sizes including 405B+ parameter models, "
            "24/7 phone and Slack support with 1-hour SLA."
        ),
    },
    {
        "query": "LoRA fine-tuning local models",
        "positive": (
            "Nextera Professional and Enterprise plans include on-device fine-tuning. "
            "Supported methods: LoRA, QLoRA (4-bit quantized LoRA), and full "
            "fine-tuning for models under 7B. Typical fine-tuning time: 30–60 minutes "
            "for 500 examples on a single A100."
        ),
    },
    {
        "query": "docker kubernetes deployment requirements",
        "positive": (
            "Nextera can be deployed as a single-node Docker container on any Linux, "
            "macOS, or Windows machine with 16 GB+ RAM. "
            "Kubernetes deployment uses a Helm chart with auto-scaling. "
            "Minimum 8 GB RAM for small models; 32 GB + 16 GB VRAM GPU for production."
        ),
    },
    {
        "query": "SOC2 HIPAA compliance security",
        "positive": (
            "Nextera is designed for strict data security. "
            "SOC 2 Type II compliance (Enterprise), GDPR-ready architecture, "
            "HIPAA BAA available on Enterprise. "
            "Security: AES-256 at rest, TLS 1.3 in transit, RBAC, audit logging."
        ),
    },
    {
        "query": "starter plan pricing monthly cost",
        "positive": (
            "The Nextera Starter plan costs €299 per month (or €2,990 per year). "
            "Includes: up to 5 concurrent users, 10 GB vector storage, "
            "support for models up to 7B parameters, community Discord support."
        ),
    },
    {
        "query": "OpenAI compatible API SDK integration",
        "positive": (
            "Nextera exposes a fully OpenAI-compatible REST API. Any existing "
            "application built for OpenAI works with a single base_url change. "
            "Official SDKs: Python, TypeScript/Node, Go, and .NET C#. "
            "Native integrations: LangChain, LlamaIndex, Semantic Kernel, Haystack."
        ),
    },
    {
        "query": "why use local AI instead of cloud GPT privacy cost",
        "positive": (
            "Privacy: Prompts and documents never leave your network — critical for "
            "healthcare, legal, finance. "
            "Cost: After hardware, inference is free. A team of 50 doing 100 queries/day "
            "pays ~€5,000/month on cloud APIs but €0 locally. "
            "Latency: Local GPU achieves <200ms first token vs 500–2000ms cloud."
        ),
    },
    {
        "query": "RAG pipeline document chunking hybrid search",
        "positive": (
            "Nextera RAG pipeline features: automatic document chunking (configurable overlap), "
            "hybrid search (dense + sparse BM25), re-ranking with cross-encoders, "
            "query rewriting, multi-document synthesis, citation tracking, streaming responses. "
            "Supported formats: PDF, DOCX, TXT, Markdown, HTML, CSV."
        ),
    },
    {
        "query": "professional plan email support SLA",
        "positive": (
            "Professional plan: email support with 8-hour response SLA (business hours), "
            "access to private support portal, monthly office hours, "
            "fine-tuning support (LoRA), SSO via SAML 2.0, audit logs."
        ),
    },
    {
        "query": "function calling agent tools multi-step reasoning",
        "positive": (
            "Nextera agent framework: intent classification, tool registry, "
            "function calling (OpenAI-compatible), ReAct-style reasoning loops, "
            "multi-agent orchestration, execution tracing. "
            "Built-in tools: SQL query, vector search, web search, code execution (sandboxed), HTTP API calls."
        ),
    },
    # Additional pairs for stronger domain signal
    {
        "query": "plan comparison professional vs enterprise",
        "positive": (
            "Professional plan: €1,499/month, up to 50 users, 100 GB storage, LoRA fine-tuning, "
            "SSO via SAML 2.0, 8-hour email SLA. "
            "Enterprise plan: €3,500/month, unlimited users, unlimited storage, all fine-tuning methods, "
            "dedicated support engineer, 1-hour SLA, custom contracts."
        ),
    },
    {
        "query": "air-gapped offline deployment",
        "positive": (
            "Nextera Enterprise supports fully air-gapped deployments. "
            "All model weights, the API server, and the vector database run on-premises with no internet access. "
            "Updates are delivered as signed offline packages via USB or private registry."
        ),
    },
    {
        "query": "GDPR data residency on-premises",
        "positive": (
            "Nextera is GDPR-ready: all data remains on the customer's infrastructure. "
            "No telemetry, no prompt logging to external servers. "
            "Data residency guarantees are built into the architecture — "
            "nothing leaves your network unless you choose to send it."
        ),
    },
    {
        "query": "streaming responses latency first token",
        "positive": (
            "Nextera supports streaming responses via server-sent events (SSE). "
            "On a single NVIDIA GPU, first-token latency is 80–200ms. "
            "Typical query response time end-to-end: 1–3 seconds including retrieval and synthesis."
        ),
    },
    {
        "query": "multi-tenant access control RBAC",
        "positive": (
            "Nextera Enterprise supports role-based access control (RBAC). "
            "Teams can be isolated with separate knowledge bases, model configurations, and API keys. "
            "Audit logs capture all queries and responses for compliance review."
        ),
    },
    {
        "query": "model size supported parameters 7B 70B",
        "positive": (
            "Starter: up to 7B parameter models. "
            "Professional: up to 70B parameter models. "
            "Enterprise: all sizes including 405B+ parameter models. "
            "Quantized (4-bit) models run in 40–50% less VRAM — e.g., 70B runs in ~40 GB with Q4."
        ),
    },
    {
        "query": "pricing annual discount subscription",
        "positive": (
            "All plans offer annual billing at a 20% discount. "
            "Starter: €299/month (€2,990/year). "
            "Professional: €1,499/month (€14,990/year). "
            "Enterprise: €3,500/month (€35,000/year, or custom pricing for large deployments)."
        ),
    },
    {
        "query": "vector database ChromaDB embedding search",
        "positive": (
            "Nextera uses an embedded vector database for semantic search. "
            "Documents are chunked, embedded via the local embedding model, "
            "and stored with cosine similarity indexing. "
            "Supports metadata filtering, hybrid BM25+dense search, and re-ranking."
        ),
    },
    {
        "query": "SQL database query structured data",
        "positive": (
            "Nextera includes a read-only SQL tool for querying structured business data. "
            "The agent generates SELECT statements from natural language, "
            "executes them against SQLite (or PostgreSQL for Enterprise), "
            "and formats the result as a human-readable answer."
        ),
    },
    {
        "query": "calculator arithmetic math expression",
        "positive": (
            "Nextera includes a sandboxed calculator tool for precise arithmetic. "
            "The agent extracts the mathematical expression from the query, "
            "evaluates it in an isolated Python environment, "
            "and returns the result — no hallucination possible on pure arithmetic."
        ),
    },
    {
        "query": "intent classification routing agent pipeline",
        "positive": (
            "Nextera routes every query through an intent classifier before routing to tools. "
            "Intents: rag_query (knowledge base search), tool_use (calculator/SQL), "
            "direct_answer (no tools needed). "
            "The fine-tuned 1B classifier achieves 75%+ accuracy on domain queries."
        ),
    },
    {
        "query": "query rewriting semantic search improvement",
        "positive": (
            "Before vector search, Nextera rewrites the user query into a dense keyword phrase. "
            "Example: 'What is the best plan for a growing startup?' becomes "
            "'Professional Enterprise plan comparison scalability pricing'. "
            "Query rewriting improves retrieval accuracy by 20–30% on complex questions."
        ),
    },
    {
        "query": "customer onboarding professional services",
        "positive": (
            "Enterprise plan includes a dedicated customer success engineer, "
            "onboarding workshop (2 days), custom fine-tuning of the intent classifier "
            "on your domain data, and quarterly business reviews. "
            "Professional plan includes self-service onboarding and monthly office hours."
        ),
    },
]


class EmbeddingGemmaDataPreparer:
    """
    Builds a contrastive retrieval dataset for embeddinggemma.

    Format: {query, positive} pairs. The sentence-transformers library
    (MultipleNegativesRankingLoss) treats all other positives in the batch
    as hard negatives — no explicit negative mining required.

    Augments hand-crafted pairs with real query/document pairs from logs.
    """

    def __init__(
        self,
        interactions_path: str = "./data/interactions.json",
        output_dir:        str = "./data/training-data",
    ) -> None:
        self.interactions = load_interactions(interactions_path)
        self.output_dir   = output_dir

    def prepare(self) -> int:
        pairs: list[dict] = list(_RETRIEVAL_PAIRS)

        # Extract real query→document pairs from RAG logs
        # agent.py stores retrieved docs under details["documents"]
        for interaction in self.interactions:
            if interaction.get("intent") != "rag_query":
                continue
            query = interaction.get("query", "")
            for step in interaction.get("steps", []):
                if step.get("action") == "vector_search":
                    documents = step.get("details", {}).get("documents", [])
                    for doc in documents[:2]:  # top-2 = most relevant positives
                        if isinstance(doc, dict) and doc.get("content"):
                            pairs.append({
                                "query":    query,
                                "positive": doc["content"][:500],
                            })

        random.shuffle(pairs)
        os.makedirs(self.output_dir, exist_ok=True)
        out = os.path.join(self.output_dir, "embeddinggemma_retrieval.jsonl")
        count = save_jsonl(pairs, out)
        print(f"  [embeddinggemma] Retrieval pairs: {count} examples → {out}")
        return count
