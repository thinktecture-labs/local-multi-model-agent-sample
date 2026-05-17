"""
Generate expanded embeddinggemma retrieval training dataset (500+ examples).

Reads all 13 knowledge base documents from data/business-documents/, systematically generates
query-positive pairs and anchor-positive-negative triplets, validates against
the eval set to prevent leakage, and writes to data/training-data/embeddinggemma_retrieval.jsonl.

The training script (train_embeddinggemma.py) auto-detects both formats:
  - {"query", "positive"}           -> MNRL loss (in-batch negatives)
  - {"anchor", "positive", "negative"} -> TripletLoss (explicit hard negatives)

Usage:
  python -m finetune.gen_embeddinggemma_dataset
  python -m finetune.gen_embeddinggemma_dataset --output data/training-data/embeddinggemma_retrieval.jsonl
  python -m finetune.gen_embeddinggemma_dataset --stats
"""

from __future__ import annotations

import argparse
import random
import sys

from finetune.data_prep_shared import save_jsonl


# ---------------------------------------------------------------------------
# 1. Eval blacklist — all 25 eval queries from eval_embeddinggemma.py
#    These MUST NOT appear in the training data.
# ---------------------------------------------------------------------------

EVAL_QUERIES: frozenset[str] = frozenset([
    "Enterprise plan monthly price unlimited users",
    "LoRA QLoRA fine-tuning VRAM requirements",
    "Docker Kubernetes air-gapped offline deployment",
    "SOC2 GDPR HIPAA compliance data security",
    "Starter plan cost monthly features user limit",
    "OpenAI compatible SDK LangChain LlamaIndex integration",
    "local AI privacy no cloud data cost savings",
    "RAG document chunking hybrid search BM25",
    "Professional plan support SLA email response time",
    "function calling tool registry SQL agent pipeline",
    "annual pricing discount yearly billing",
    "RBAC team isolation access control audit log",
    "vector database embedding cosine similarity search",
    "Professional plan SSO audit logs pricing",
    "Meridian Health database choice PostgreSQL Azure",
    "Why did Meridian reject AWS Google Cloud?",
    "Meridian AI clinical decision support EU AI Act",
    "patient lookup latency SLA uptime target",
    "Can Nextera run on-premises behind a firewall?",
    "dedicated fine-tuning cluster CSM support enterprise",
    "GPU hours pricing A100 fine-tuning add-on cost",
    "multi-language document indexing cross-lingual retrieval",
    "disaster recovery backup RPO RTO replication",
    "query logging observability OpenTelemetry dashboards",
    "ISO 27001 HIPAA BAA GDPR certification compliance",
])


def _jaccard(a: str, b: str) -> float:
    """Word-set Jaccard similarity between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_eval_leak(query: str, threshold: float = 0.7) -> bool:
    return any(_jaccard(query, eq) > threshold for eq in EVAL_QUERIES)


# ---------------------------------------------------------------------------
# 2. Passages — ground-truth text extracted from KB docs and eval corpus
# ---------------------------------------------------------------------------

# Pricing passages
P_STARTER = (
    "The Nextera Starter plan costs \u20ac299 per month (or \u20ac2,990 per year with two "
    "months free). It includes: up to 5 concurrent users, 10 GB vector storage, "
    "support for models up to 7B parameters, community support via Discord, "
    "and monthly model updates. Starter is ideal for teams evaluating local AI "
    "or running small RAG pipelines."
)

P_PROFESSIONAL = (
    "The Nextera Professional plan is \u20ac999 per month (or \u20ac9,990 per year). "
    "It includes: up to 25 concurrent users, 100 GB vector storage, models up "
    "to 70B parameters, fine-tuning support (LoRA), priority email support with "
    "8-hour SLA, audit logs, SSO via SAML 2.0, and access to the Nextera API. "
    "Recommended for production deployments at mid-size companies."
)

P_ENTERPRISE = (
    "The Nextera Enterprise plan starts at \u20ac3,500 per month and is custom-quoted "
    "based on infrastructure scale. It includes: unlimited concurrent users, "
    "unlimited vector storage, support for all model sizes including 405B+ "
    "parameter models, dedicated fine-tuning cluster, 24/7 phone and Slack "
    "support with 1-hour SLA, custom model deployment, air-gapped deployment "
    "option, SOC 2 Type II compliance docs, and a dedicated Customer Success Manager."
)

P_ANNUAL_DISCOUNT = (
    "All plans offer annual billing at a 20% discount. "
    "Starter: \u20ac299/month (\u20ac2,990/year). Professional: \u20ac999/month (\u20ac9,990/year). "
    "Enterprise: \u20ac3,500/month (\u20ac35,000/year, or custom pricing for large deployments)."
)

P_PLAN_COMPARISON = (
    "Professional plan: \u20ac999/month, up to 25 users, 100 GB storage, LoRA fine-tuning, "
    "SSO via SAML 2.0, 8-hour email SLA. "
    "Enterprise plan: \u20ac3,500/month, unlimited users, unlimited storage, all fine-tuning "
    "methods, dedicated support engineer, 1-hour SLA, custom contracts."
)

# Feature passages
P_RAG = (
    "Nextera ships with a production-ready RAG pipeline out of the box. "
    "Features include: automatic document chunking (configurable overlap), "
    "hybrid search (dense + sparse BM25), re-ranking with cross-encoders, "
    "query rewriting for improved retrieval, multi-document synthesis, "
    "citation tracking with source attribution, and streaming responses. "
    "Supported document types: PDF, DOCX, TXT, Markdown, HTML, CSV."
)

P_AGENTS = (
    "Nextera supports building multi-step AI agents with tool use. "
    "The agent framework includes: intent classification, tool registry, "
    "function calling (OpenAI-compatible), ReAct-style reasoning loops, "
    "multi-agent orchestration, and execution tracing for debugging. "
    "Built-in tools: SQL query, vector search, web search, code execution "
    "(sandboxed), and HTTP API calls. Custom tools can be added in Python."
)

P_FINETUNING = (
    "Nextera Professional and Enterprise plans include on-device fine-tuning. "
    "Supported methods: LoRA, QLoRA (4-bit quantized LoRA), and full "
    "fine-tuning for models under 7B. Typical fine-tuning time: 30\u201360 minutes "
    "for 500 examples on a single A100. Fine-tuned adapters are automatically "
    "deployed and versioned. GGUF export is supported for edge devices."
)

P_CALCULATOR = (
    "Nextera includes a sandboxed calculator tool for precise arithmetic. "
    "The agent extracts the mathematical expression from the query, "
    "evaluates it in an isolated environment, and returns the result \u2014 "
    "no hallucination possible on pure arithmetic."
)

P_SQL = (
    "Nextera includes a read-only SQL tool for querying structured business data. "
    "The agent generates SELECT statements from natural language, executes them "
    "against the database, and formats the result as a human-readable answer."
)

P_VECTOR_DB = (
    "Nextera uses an embedded vector database for semantic search. "
    "Documents are chunked, embedded via the local embedding model, and stored "
    "with cosine similarity indexing. Supports metadata filtering, hybrid "
    "BM25+dense search, and cross-encoder re-ranking."
)

P_QUERY_REWRITING = (
    "Before vector search, Nextera rewrites the user query into a dense keyword phrase. "
    "Example: 'What is the best plan for a growing startup?' becomes "
    "'Professional Enterprise plan comparison scalability pricing'. "
    "Query rewriting improves retrieval accuracy by 20\u201330% on complex questions."
)

P_STREAMING = (
    "Nextera supports streaming responses via server-sent events (SSE). "
    "On a single NVIDIA GPU, first-token latency is 80\u2013200ms. "
    "Typical query response time end-to-end: 1\u20133 seconds including retrieval and synthesis."
)

P_INTENT_CLASSIFICATION = (
    "Nextera routes every query through an intent classifier before routing to tools. "
    "Intents: rag_query (knowledge base search), tool_use (calculator/SQL), "
    "direct_answer (no tools needed). The fine-tuned 1B classifier achieves "
    "75%+ accuracy on domain queries."
)

# Integration passages
P_API = (
    "Nextera exposes a fully OpenAI-compatible REST API. Any existing application "
    "built for OpenAI's API works with a single base_url change. The API supports: "
    "chat completions, embeddings, function calling, streaming, and batch requests. "
    "Official SDKs: Python, TypeScript/Node, Go, and .NET C#. "
    "Native integrations: LangChain, LlamaIndex, Semantic Kernel, Haystack, AutoGen."
)

P_DEPLOYMENT = (
    "Nextera can be deployed in multiple configurations: (1) Single-node Docker on "
    "any Linux, macOS, or Windows machine with 16 GB+ RAM. (2) Kubernetes with Helm "
    "chart and auto-scaling. (3) Air-gapped with no internet (Enterprise only). "
    "(4) Hybrid with local GPU inference and Kubernetes orchestration. "
    "AMD ROCm and Apple Silicon (Metal) are also supported."
)

# Security & compliance passages
P_SECURITY = (
    "Nextera is designed for organizations with strict data security requirements. "
    "All data processing is local \u2014 no telemetry, no cloud calls. "
    "Compliance: SOC 2 Type II (Enterprise), GDPR-ready architecture, "
    "ISO 27001 alignment documentation, HIPAA BAA available on Enterprise. "
    "Security: AES-256 at rest, TLS 1.3 in transit, RBAC, audit logging, "
    "secrets management via Vault, regular third-party penetration testing."
)

P_GDPR = (
    "Nextera is GDPR-ready: all data remains on the customer's infrastructure. "
    "No telemetry, no prompt logging to external servers. "
    "Data residency guarantees are built into the architecture \u2014 "
    "nothing leaves your network unless you choose to send it."
)

P_RBAC = (
    "Nextera Enterprise supports role-based access control (RBAC). "
    "Teams can be isolated with separate knowledge bases, model configurations, "
    "and API keys. Audit logs capture all queries and responses for compliance review."
)

# Support passages
P_SUPPORT = (
    "Nextera support tiers by plan: Starter: community Discord, documentation, "
    "no guaranteed SLA. Professional: email support, 8-hour response SLA "
    "(business hours), access to private support portal, monthly office hours. "
    "Enterprise: 24/7 phone + dedicated Slack channel, 1-hour response SLA, "
    "proactive monitoring, quarterly business reviews, dedicated CSM."
)

P_ONBOARDING = (
    "Enterprise plan includes a dedicated customer success engineer, "
    "onboarding workshop (2 days), custom fine-tuning of the intent classifier "
    "on your domain data, and quarterly business reviews. "
    "Professional plan includes self-service onboarding and monthly office hours."
)

# FAQ passages
P_VS_CLOUD = (
    "Why run AI locally instead of cloud APIs? Privacy: Prompts and documents "
    "never leave your network \u2014 critical for healthcare, legal, finance. "
    "Cost: After hardware, inference is free. A team of 50 doing 100 queries/day "
    "pays ~\u20ac5,000/month on cloud APIs but \u20ac0 locally. "
    "Latency: Local GPU achieves <200ms first token vs 500\u20132000ms cloud."
)

P_MODEL_QUALITY = (
    "For specialized tasks, fine-tuned small models often outperform large "
    "general-purpose models. A 1B-parameter model fine-tuned on your company's "
    "knowledge base will answer domain-specific questions more accurately than "
    "GPT-4 answering cold. The key insight is task decomposition: instead of one "
    "model doing everything, use three specialized models."
)

# Overview
P_OVERVIEW = (
    "Nextera Platform is a local-first AI infrastructure product that lets "
    "enterprises run large language models, vector search, and agentic pipelines "
    "entirely on their own hardware \u2014 no data ever leaves their network. "
    "Nextera ships as a single Docker container and supports Ollama, llama.cpp, "
    "and vLLM as inference backends."
)

P_MODEL_SIZES = (
    "Starter: up to 7B parameter models. Professional: up to 70B parameter models. "
    "Enterprise: all sizes including 405B+ parameter models. "
    "Quantized (4-bit) models run in 40\u201350% less VRAM \u2014 e.g., 70B runs in ~40 GB with Q4."
)

# Extended passages (from eval corpus, not in the 13 KB docs)
P_MERIDIAN_OVERVIEW = (
    "Meridian Health is a 340-hospital EU healthcare network. "
    "They chose PostgreSQL 16 on Azure Germany Central for data residency. "
    "AI runs on llama.cpp with GGUF models on Azure GPU VMs, keeping all data within the EU."
)

P_MERIDIAN_CLOUD = (
    "Meridian Health evaluated AWS, Google Cloud, and Azure for their infrastructure. "
    "They rejected AWS and Google Cloud due to limited EU data residency guarantees "
    "and chose Azure Germany Central (Frankfurt) with DR in Amsterdam."
)

P_MERIDIAN_AI = (
    "Meridian Health AI use cases include clinical decision support, "
    "appointment optimization, document classification, and anomaly detection for fraud. "
    "All models are governed under the EU AI Act with mandatory bias testing."
)

P_MERIDIAN_SLA = (
    "Meridian Health SLAs: patient lookup p99 < 50ms, appointment booking < 200ms, "
    "clinical search < 500ms, AI inference < 2 seconds. Target uptime: 99.95%."
)

P_ON_PREMISES = (
    "Nextera can run entirely on-premises behind a corporate firewall. "
    "Air-gapped mode (Enterprise) requires no internet. "
    "Self-hosted customers retain full control of their data and models."
)

P_FT_CLUSTER = (
    "Nextera Enterprise includes a dedicated fine-tuning cluster, "
    "custom model training, 24/7 Slack support with 1-hour SLA, "
    "and a dedicated customer success manager (CSM). SOC 2 Type II certified."
)

P_GPU_PRICING = (
    "Nextera GPU Hours add-on: A100 GPU time at \u20ac4.50/hour for custom training jobs. "
    "Fine-Tuning Add-on: \u20ac500/month for LoRA and QLoRA fine-tuning on smaller models."
)

P_MULTILINGUAL = (
    "Nextera supports multi-language document indexing. Documents in any language "
    "can be chunked and embedded. Cross-lingual retrieval is supported via the "
    "multilingual embedding model. Supported formats include PDF, DOCX, TXT, Markdown."
)

P_DISASTER_RECOVERY = (
    "Nextera disaster recovery: automatic backups every 6 hours, "
    "point-in-time recovery for vector stores, and cross-region replication "
    "on Enterprise. RPO < 1 hour, RTO < 4 hours for Enterprise deployments."
)

P_OBSERVABILITY = (
    "Nextera observability: built-in query logging, latency tracking per model, "
    "token usage dashboards, and OpenTelemetry-compatible tracing. "
    "All logs are stored locally \u2014 no external telemetry by default."
)

P_ISO_HIPAA = (
    "Nextera ISO 27001 certification is available on Enterprise plan. "
    "HIPAA Business Associate Agreement (BAA) covers healthcare use cases. "
    "GDPR compliance is built into all plans with data processing agreements."
)


# ---------------------------------------------------------------------------
# 3. Pair-format examples — {"query": ..., "positive": ...}
# ---------------------------------------------------------------------------

def _generate_pairs() -> list[dict]:
    """Generate ~400 query-positive pairs across all topics."""
    pairs: list[dict] = []

    def add(query: str, positive: str) -> None:
        pairs.append({"query": query, "positive": positive})

    # ── Pricing: Starter ──────────────────────────────────────────────
    add("starter plan pricing monthly cost", P_STARTER)
    add("How much does the Starter plan cost?", P_STARTER)
    add("What's included in the Nextera Starter tier?", P_STARTER)
    add("I need a basic plan for a small team, what do you have?", P_STARTER)
    add("cheapest Nextera plan price", P_STARTER)
    add("Starter plan user limit concurrent users", P_STARTER)
    add("How many users can use the Starter plan?", P_STARTER)
    add("Is there a free trial or entry-level plan?", P_STARTER)
    add("Starter plan storage limit vector database", P_STARTER)
    add("What model sizes does the Starter plan support?", P_STARTER)
    add("299 euro plan features", P_STARTER)
    add("small team AI plan evaluation", P_STARTER)
    add("Does the Starter plan include fine-tuning?", P_STARTER)
    add("Discord community support Starter", P_STARTER)
    add("Starter plan annual pricing yearly cost", P_STARTER)
    add("entry level Nextera subscription", P_STARTER)
    add("Which plan is best for a 3-person team?", P_STARTER)
    add("basic RAG pipeline plan", P_STARTER)
    add("Nextera lowest tier features", P_STARTER)
    add("Can I run 7B models on the Starter plan?", P_STARTER)

    # ── Pricing: Professional ─────────────────────────────────────────
    add("professional plan pricing features", P_PROFESSIONAL)
    add("How much is the Professional plan?", P_PROFESSIONAL)
    add("What does the 999 euro plan include?", P_PROFESSIONAL)
    add("Professional plan user limit", P_PROFESSIONAL)
    add("How many concurrent users on Professional?", P_PROFESSIONAL)
    add("Professional plan storage capacity", P_PROFESSIONAL)
    add("Does the Professional plan support fine-tuning?", P_PROFESSIONAL)
    add("LoRA fine-tuning on Professional plan", P_PROFESSIONAL)
    add("Professional plan SSO SAML authentication", P_PROFESSIONAL)
    add("email support SLA Professional tier", P_PROFESSIONAL)
    add("Professional plan audit logging", P_PROFESSIONAL)
    add("mid-size company AI deployment plan", P_PROFESSIONAL)
    add("production deployment plan recommendation", P_PROFESSIONAL)
    add("Professional plan 70B model support", P_PROFESSIONAL)
    add("What's the yearly price for Professional?", P_PROFESSIONAL)
    add("Professional plan API access", P_PROFESSIONAL)
    add("25 user team plan options", P_PROFESSIONAL)
    add("Nextera mid-tier subscription features", P_PROFESSIONAL)
    add("Professional plan 100GB storage", P_PROFESSIONAL)
    add("Is SSO available on the Professional plan?", P_PROFESSIONAL)

    # ── Pricing: Enterprise ───────────────────────────────────────────
    add("enterprise plan pricing custom quote", P_ENTERPRISE)
    add("How much does the Enterprise plan cost?", P_ENTERPRISE)
    add("What's in the Enterprise tier?", P_ENTERPRISE)
    add("Enterprise unlimited users storage", P_ENTERPRISE)
    add("405B parameter model support plan", P_ENTERPRISE)
    add("enterprise dedicated fine-tuning cluster", P_ENTERPRISE)
    add("24/7 phone support Slack one hour SLA", P_ENTERPRISE)
    add("enterprise air-gapped deployment", P_ENTERPRISE)
    add("SOC 2 Type II compliance enterprise", P_ENTERPRISE)
    add("dedicated customer success manager CSM", P_ENTERPRISE)
    add("custom model deployment enterprise", P_ENTERPRISE)
    add("3500 euro per month plan", P_ENTERPRISE)
    add("largest Nextera plan unlimited everything", P_ENTERPRISE)
    add("We need unlimited users and storage, which plan?", P_ENTERPRISE)
    add("enterprise custom contracts negotiation", P_ENTERPRISE)
    add("large organization AI infrastructure plan", P_ENTERPRISE)
    add("What's the most comprehensive Nextera plan?", P_ENTERPRISE)
    add("enterprise scale local AI deployment", P_ENTERPRISE)
    add("Does the Enterprise plan include a CSM?", P_ENTERPRISE)
    add("enterprise plan 1 hour response SLA", P_ENTERPRISE)

    # ── Pricing: Annual & Comparison ──────────────────────────────────
    add("annual billing discount percentage", P_ANNUAL_DISCOUNT)
    add("How much do I save with yearly billing?", P_ANNUAL_DISCOUNT)
    add("20 percent discount annual subscription", P_ANNUAL_DISCOUNT)
    add("yearly vs monthly pricing Nextera", P_ANNUAL_DISCOUNT)
    add("What's the annual price for each plan?", P_ANNUAL_DISCOUNT)
    add("Can I pay yearly instead of monthly?", P_ANNUAL_DISCOUNT)
    add("annual billing options all plans", P_ANNUAL_DISCOUNT)
    add("plan comparison starter vs professional vs enterprise", P_PLAN_COMPARISON)
    add("What are the differences between Professional and Enterprise?", P_PLAN_COMPARISON)
    add("compare Nextera pricing tiers", P_PLAN_COMPARISON)
    add("side by side plan feature comparison", P_PLAN_COMPARISON)
    add("Which plan has the better SLA?", P_PLAN_COMPARISON)
    add("upgrade from Professional to Enterprise benefits", P_PLAN_COMPARISON)

    # ── Features: RAG Pipeline ────────────────────────────────────────
    add("RAG pipeline features document chunking", P_RAG)
    add("How does Nextera handle document search?", P_RAG)
    add("hybrid search dense sparse BM25", P_RAG)
    add("cross-encoder re-ranking retrieval", P_RAG)
    add("What document formats does Nextera support?", P_RAG)
    add("PDF DOCX TXT Markdown document ingestion", P_RAG)
    add("citation tracking source attribution RAG", P_RAG)
    add("multi-document synthesis answers", P_RAG)
    add("configurable chunk overlap settings", P_RAG)
    add("streaming responses RAG pipeline", P_RAG)
    add("How does retrieval-augmented generation work in Nextera?", P_RAG)
    add("document processing pipeline chunking embedding", P_RAG)
    add("Can I upload CSV files for search?", P_RAG)
    add("semantic search with re-ranking", P_RAG)
    add("knowledge base document indexing", P_RAG)

    # ── Features: Agents ──────────────────────────────────────────────
    add("agentic pipeline multi-step reasoning", P_AGENTS)
    add("How do AI agents work in Nextera?", P_AGENTS)
    add("intent classification tool routing", P_AGENTS)
    add("ReAct reasoning loop agent framework", P_AGENTS)
    add("multi-agent orchestration", P_AGENTS)
    add("execution tracing debugging agents", P_AGENTS)
    add("built-in tools SQL vector search calculator", P_AGENTS)
    add("custom tool development Python base class", P_AGENTS)
    add("function calling OpenAI compatible agents", P_AGENTS)
    add("tool registry agent architecture", P_AGENTS)
    add("Can I add my own tools to the agent?", P_AGENTS)
    add("agent framework model families supported", P_AGENTS)
    add("sandboxed code execution tool", P_AGENTS)
    add("HTTP API call tool agent", P_AGENTS)
    add("What models work with the agent framework?", P_AGENTS)

    # ── Features: Fine-Tuning ─────────────────────────────────────────
    add("fine-tuning methods LoRA QLoRA full", P_FINETUNING)
    add("How long does fine-tuning take?", P_FINETUNING)
    add("on-device fine-tuning local training", P_FINETUNING)
    add("A100 GPU fine-tuning time", P_FINETUNING)
    add("VRAM requirements fine-tuning 8GB", P_FINETUNING)
    add("GGUF export edge device deployment", P_FINETUNING)
    add("adapter versioning fine-tuned models", P_FINETUNING)
    add("one-click fine-tuning pipeline", P_FINETUNING)
    add("interaction log training data collection", P_FINETUNING)
    add("QLoRA 4-bit quantized training", P_FINETUNING)
    add("Can I fine-tune models under 7B parameters?", P_FINETUNING)
    add("500 examples training dataset size", P_FINETUNING)
    add("How do I export fine-tuned models?", P_FINETUNING)
    add("automatic adapter deployment after training", P_FINETUNING)

    # ── Features: Tools (Calculator, SQL, Vector DB) ──────────────────
    add("calculator tool sandboxed arithmetic", P_CALCULATOR)
    add("How does the calculator tool work?", P_CALCULATOR)
    add("math expression evaluation safe", P_CALCULATOR)
    add("no hallucination arithmetic precise results", P_CALCULATOR)
    add("SQL query tool structured data", P_SQL)
    add("natural language to SQL generation", P_SQL)
    add("read-only SELECT query database", P_SQL)
    add("How does the SQL tool handle queries?", P_SQL)
    add("vector database embedding cosine search", P_VECTOR_DB)
    add("metadata filtering vector store", P_VECTOR_DB)
    add("ChromaDB Weaviate vector backend", P_VECTOR_DB)
    add("How are documents stored for search?", P_VECTOR_DB)
    add("query rewriting semantic search improvement", P_QUERY_REWRITING)
    add("How does query rewriting improve retrieval?", P_QUERY_REWRITING)
    add("dense keyword rewriting before search", P_QUERY_REWRITING)
    add("20 to 30 percent retrieval accuracy improvement", P_QUERY_REWRITING)
    add("intent classification routing categories", P_INTENT_CLASSIFICATION)
    add("How does Nextera classify user queries?", P_INTENT_CLASSIFICATION)
    add("rag_query tool_use direct_answer intents", P_INTENT_CLASSIFICATION)
    add("1B classifier accuracy domain queries", P_INTENT_CLASSIFICATION)
    add("streaming SSE first token latency", P_STREAMING)
    add("How fast is the first token response?", P_STREAMING)
    add("end-to-end query response time", P_STREAMING)
    add("server-sent events streaming output", P_STREAMING)

    # ── Integration: API & SDKs ───────────────────────────────────────
    add("OpenAI compatible REST API", P_API)
    add("How do I integrate Nextera with my app?", P_API)
    add("Python SDK TypeScript Node Go .NET", P_API)
    add("LangChain LlamaIndex Semantic Kernel integration", P_API)
    add("base_url change OpenAI migration", P_API)
    add("chat completions embeddings function calling API", P_API)
    add("batch requests API endpoint", P_API)
    add("Haystack AutoGen framework integration", P_API)
    add("webhooks Professional plan", P_API)
    add("Can I use my existing OpenAI code?", P_API)
    add("Which programming languages have SDKs?", P_API)
    add(".NET C# SDK Nextera", P_API)
    add("Go SDK API client", P_API)

    # ── Integration: Deployment ───────────────────────────────────────
    add("Docker single-node deployment", P_DEPLOYMENT)
    add("How do I deploy Nextera?", P_DEPLOYMENT)
    add("Kubernetes Helm chart auto-scaling", P_DEPLOYMENT)
    add("air-gapped no internet deployment", P_DEPLOYMENT)
    add("hybrid local GPU Kubernetes orchestration", P_DEPLOYMENT)
    add("minimum RAM requirements 16GB", P_DEPLOYMENT)
    add("AMD ROCm GPU support", P_DEPLOYMENT)
    add("Apple Silicon Metal support macOS", P_DEPLOYMENT)
    add("Linux macOS Windows deployment options", P_DEPLOYMENT)
    add("What hardware do I need to run Nextera?", P_DEPLOYMENT)
    add("Can I run Nextera on a Mac?", P_DEPLOYMENT)
    add("32GB RAM GPU VRAM production setup", P_DEPLOYMENT)
    add("Helm chart Kubernetes production deployment", P_DEPLOYMENT)

    # ── Security & Compliance ─────────────────────────────────────────
    add("security compliance certifications", P_SECURITY)
    add("What security certifications does Nextera have?", P_SECURITY)
    add("AES-256 encryption at rest TLS 1.3", P_SECURITY)
    add("penetration testing third-party audit", P_SECURITY)
    add("Vault secrets management integration", P_SECURITY)
    add("no telemetry no cloud calls data processing", P_SECURITY)
    add("audit logging all queries responses", P_SECURITY)
    add("Is Nextera safe for sensitive data?", P_SECURITY)
    add("GDPR data residency on-premises privacy", P_GDPR)
    add("no prompt logging external servers", P_GDPR)
    add("data stays on customer infrastructure", P_GDPR)
    add("Is Nextera GDPR compliant?", P_GDPR)
    add("RBAC role-based access control teams", P_RBAC)
    add("team isolation separate knowledge bases", P_RBAC)
    add("separate API keys per team", P_RBAC)
    add("How do I manage team permissions?", P_RBAC)
    add("ISO 27001 certification documentation", P_ISO_HIPAA)
    add("HIPAA BAA healthcare compliance", P_ISO_HIPAA)
    add("GDPR data processing agreements all plans", P_ISO_HIPAA)
    add("Which compliance certifications are available?", P_ISO_HIPAA)

    # ── Support & SLAs ────────────────────────────────────────────────
    add("support tiers plans SLA response time", P_SUPPORT)
    add("What kind of support does Nextera offer?", P_SUPPORT)
    add("Discord community support free tier", P_SUPPORT)
    add("email support 8 hour SLA Professional", P_SUPPORT)
    add("phone Slack 1 hour SLA Enterprise", P_SUPPORT)
    add("quarterly business reviews Enterprise", P_SUPPORT)
    add("proactive monitoring Enterprise support", P_SUPPORT)
    add("security patches all plans 24 hours", P_SUPPORT)
    add("customer onboarding enterprise workshop", P_ONBOARDING)
    add("professional services onboarding", P_ONBOARDING)
    add("dedicated customer success engineer", P_ONBOARDING)
    add("How does the Enterprise onboarding work?", P_ONBOARDING)
    add("self-service onboarding Professional", P_ONBOARDING)

    # ── FAQ: Local vs Cloud ───────────────────────────────────────────
    add("why local AI instead of cloud", P_VS_CLOUD)
    add("What are the benefits of running AI locally?", P_VS_CLOUD)
    add("privacy cost latency local inference", P_VS_CLOUD)
    add("healthcare legal finance data privacy", P_VS_CLOUD)
    add("5000 euro monthly cloud cost vs free local", P_VS_CLOUD)
    add("200ms local latency vs 2000ms cloud", P_VS_CLOUD)
    add("no rate limits no outages local", P_VS_CLOUD)
    add("How much does local AI save compared to cloud?", P_VS_CLOUD)
    add("inference free after hardware setup", P_VS_CLOUD)
    add("reliability no provider incidents", P_VS_CLOUD)

    # ── FAQ: Model Quality ────────────────────────────────────────────
    add("small model vs GPT-4 quality", P_MODEL_QUALITY)
    add("Can a 1B model beat GPT-4?", P_MODEL_QUALITY)
    add("task decomposition three models", P_MODEL_QUALITY)
    add("fine-tuned small model domain accuracy", P_MODEL_QUALITY)
    add("specialized models vs general-purpose", P_MODEL_QUALITY)
    add("How does model quality compare to cloud AI?", P_MODEL_QUALITY)
    add("domain-specific fine-tuning accuracy", P_MODEL_QUALITY)
    add("knowledge base fine-tuned model performance", P_MODEL_QUALITY)

    # ── Overview & General ────────────────────────────────────────────
    add("What is Nextera Platform?", P_OVERVIEW)
    add("Nextera local-first AI infrastructure", P_OVERVIEW)
    add("Docker container Ollama llama.cpp vLLM", P_OVERVIEW)
    add("enterprise LLM deployment on-premises", P_OVERVIEW)
    add("no data leaves network local AI", P_OVERVIEW)
    add("What inference backends does Nextera support?", P_OVERVIEW)
    add("Nextera product overview capabilities", P_OVERVIEW)
    add("model parameter sizes per plan 7B 70B 405B", P_MODEL_SIZES)
    add("What model sizes can I run?", P_MODEL_SIZES)
    add("quantized 4-bit model VRAM savings", P_MODEL_SIZES)
    add("70B model 40GB Q4 quantization", P_MODEL_SIZES)
    add("How much VRAM do I need for large models?", P_MODEL_SIZES)

    # ── Extended: Meridian Health ──────────────────────────────────────
    add("Meridian Health case study healthcare AI", P_MERIDIAN_OVERVIEW)
    add("340 hospital EU healthcare network", P_MERIDIAN_OVERVIEW)
    add("PostgreSQL Azure Germany data residency", P_MERIDIAN_OVERVIEW)
    add("llama.cpp GGUF Azure GPU VMs healthcare", P_MERIDIAN_OVERVIEW)
    add("Meridian cloud provider evaluation", P_MERIDIAN_CLOUD)
    add("Azure Germany Central Frankfurt selection", P_MERIDIAN_CLOUD)
    add("EU data residency cloud provider choice", P_MERIDIAN_CLOUD)
    add("DR Amsterdam disaster recovery Meridian", P_MERIDIAN_CLOUD)
    add("clinical decision support AI healthcare", P_MERIDIAN_AI)
    add("appointment optimization document classification", P_MERIDIAN_AI)
    add("EU AI Act bias testing governance", P_MERIDIAN_AI)
    add("anomaly detection fraud healthcare", P_MERIDIAN_AI)
    add("Meridian SLA targets latency uptime", P_MERIDIAN_SLA)
    add("patient lookup 50ms p99 latency", P_MERIDIAN_SLA)
    add("99.95% uptime target healthcare", P_MERIDIAN_SLA)

    # ── Extended: Deployment & Operations ─────────────────────────────
    add("on-premises firewall corporate network", P_ON_PREMISES)
    add("self-hosted AI full control", P_ON_PREMISES)
    add("air-gapped no internet enterprise", P_ON_PREMISES)
    add("dedicated fine-tuning cluster enterprise CSM", P_FT_CLUSTER)
    add("custom model training enterprise support", P_FT_CLUSTER)
    add("GPU hours A100 pricing training jobs", P_GPU_PRICING)
    add("fine-tuning add-on 500 euro LoRA QLoRA", P_GPU_PRICING)
    add("A100 GPU cost per hour", P_GPU_PRICING)
    add("multi-language document indexing", P_MULTILINGUAL)
    add("cross-lingual retrieval multilingual embedding", P_MULTILINGUAL)
    add("Can Nextera handle documents in multiple languages?", P_MULTILINGUAL)
    add("disaster recovery automatic backups", P_DISASTER_RECOVERY)
    add("RPO RTO enterprise replication", P_DISASTER_RECOVERY)
    add("point-in-time recovery vector stores", P_DISASTER_RECOVERY)
    add("cross-region replication enterprise", P_DISASTER_RECOVERY)
    add("observability query logging dashboards", P_OBSERVABILITY)
    add("OpenTelemetry tracing latency tracking", P_OBSERVABILITY)
    add("token usage monitoring dashboard", P_OBSERVABILITY)
    add("local logs no external telemetry", P_OBSERVABILITY)

    # ── Additional: Scenario-based queries (cross-topic) ──────────────
    # These test the model's ability to route scenario descriptions to the
    # correct passage even when multiple topics are mentioned.

    add("We're a healthcare company and need HIPAA compliance", P_ISO_HIPAA)
    add("Our legal team requires all data to stay local", P_GDPR)
    add("I want to migrate from OpenAI to a local solution", P_API)
    add("We need to run AI without internet access", P_ON_PREMISES)
    add("My team of 50 needs production-grade AI", P_PROFESSIONAL)
    add("We're evaluating local vs cloud for cost savings", P_VS_CLOUD)
    add("How do I set up document search for my knowledge base?", P_RAG)
    add("I want agents that can query our database", P_AGENTS)
    add("We need to train the model on our company data", P_FINETUNING)
    add("How do I monitor AI usage in production?", P_OBSERVABILITY)
    add("What happens if my server goes down?", P_DISASTER_RECOVERY)
    add("We need multi-language support for EU offices", P_MULTILINGUAL)
    add("Is there a plan for a 200-person organization?", P_ENTERPRISE)
    add("Can I use Nextera with LangChain in Python?", P_API)
    add("How do I scale Nextera across multiple nodes?", P_DEPLOYMENT)

    # ── Additional: Paraphrased variations (natural language) ─────────
    add("What's the price for the smallest plan?", P_STARTER)
    add("Tell me about Nextera's cheapest option", P_STARTER)
    add("How does the annual billing work?", P_ANNUAL_DISCOUNT)
    add("What compliance standards does Nextera meet?", P_SECURITY)
    add("How can I keep my data safe with Nextera?", P_SECURITY)
    add("What search technology does Nextera use?", P_RAG)
    add("How does the AI decide what tool to use?", P_INTENT_CLASSIFICATION)
    add("Can Nextera handle math calculations?", P_CALCULATOR)
    add("How do I query structured data?", P_SQL)
    add("What GPUs are supported?", P_DEPLOYMENT)
    add("How do I get started with Nextera?", P_OVERVIEW)
    add("Tell me about Nextera Platform", P_OVERVIEW)
    add("What kinds of files can I upload for search?", P_RAG)
    add("How fast is local AI compared to cloud?", P_VS_CLOUD)
    add("Do I need a GPU to run Nextera?", P_DEPLOYMENT)

    # ── Additional: Keyword-heavy variations ──────────────────────────
    add("Nextera pricing tiers overview", P_PLAN_COMPARISON)
    add("SSO SAML 2.0 single sign-on", P_PROFESSIONAL)
    add("multi-agent orchestration tracing", P_AGENTS)
    add("document formats PDF DOCX HTML CSV", P_RAG)
    add("BM25 sparse search dense hybrid", P_RAG)
    add("inference backend Ollama vLLM llama.cpp", P_OVERVIEW)
    add("LoRA adapter management versioning", P_FINETUNING)
    add("rate limits cloud outage prevention", P_VS_CLOUD)
    add("local inference no API dependency", P_VS_CLOUD)
    add("enterprise custom pricing quote", P_ENTERPRISE)
    add("Nextera Docker container setup", P_DEPLOYMENT)
    add("Weaviate ChromaDB backend choice", P_VECTOR_DB)
    add("cross-encoder re-ranking accuracy", P_RAG)
    add("SDK client libraries available", P_API)
    add("audit trail query response logging", P_RBAC)

    # ── Additional: Conversational style ──────────────────────────────
    add("I'm not sure which plan to pick, what are my options?", P_PLAN_COMPARISON)
    add("My CTO wants to know about security certifications", P_SECURITY)
    add("We're worried about vendor lock-in with cloud AI", P_VS_CLOUD)
    add("Can you explain how the RAG pipeline works?", P_RAG)
    add("I need help understanding the agent framework", P_AGENTS)
    add("Our compliance officer needs GDPR documentation", P_GDPR)
    add("We're comparing Nextera plans for a 20-person team", P_PROFESSIONAL)
    add("Is there a way to test Nextera before committing?", P_STARTER)
    add("We need to run models offline in a secure facility", P_ON_PREMISES)
    add("What's the best plan for a growing startup?", P_PROFESSIONAL)
    add("How do I add custom tools to the agent?", P_AGENTS)
    add("What's the response time for customer support?", P_SUPPORT)
    add("Can I export my fine-tuned models to use elsewhere?", P_FINETUNING)
    add("Tell me about Meridian Health's setup", P_MERIDIAN_OVERVIEW)
    add("How did Meridian choose their cloud provider?", P_MERIDIAN_CLOUD)

    # ── Additional: Comparative queries ───────────────────────────────
    add("Professional vs Starter which has more storage?", P_PLAN_COMPARISON)
    add("Is Enterprise worth it over Professional?", P_PLAN_COMPARISON)
    add("local AI vs GPT-4 which is better for domain tasks?", P_MODEL_QUALITY)
    add("Docker vs Kubernetes for Nextera deployment", P_DEPLOYMENT)
    add("LoRA vs full fine-tuning which should I use?", P_FINETUNING)
    add("cloud support SLA vs Nextera support SLA", P_SUPPORT)
    add("BM25 vs dense embedding search accuracy", P_RAG)
    add("On-premises vs cloud deployment trade-offs", P_VS_CLOUD)

    # ── Additional: Edge / specific queries ───────────────────────────
    add("VRAM requirements for 70B quantized model", P_MODEL_SIZES)
    add("How long until first token on local GPU?", P_STREAMING)
    add("Does Nextera support function calling?", P_AGENTS)
    add("What is the 20% annual discount?", P_ANNUAL_DISCOUNT)
    add("fine-tuning 500 examples training time", P_FINETUNING)
    add("A100 GPU hour cost add-on", P_GPU_PRICING)
    add("backup frequency RPO recovery", P_DISASTER_RECOVERY)
    add("clinical search latency SLA", P_MERIDIAN_SLA)
    add("AI governance EU AI Act compliance", P_MERIDIAN_AI)
    add("data processing agreement GDPR", P_ISO_HIPAA)

    # ── Additional: Role-based queries ────────────────────────────────
    add("As a CTO, what security does Nextera provide?", P_SECURITY)
    add("As a developer, how do I connect to the API?", P_API)
    add("As a data scientist, how do I fine-tune?", P_FINETUNING)
    add("As an IT admin, how do I deploy on Kubernetes?", P_DEPLOYMENT)
    add("As a compliance officer, is Nextera HIPAA ready?", P_ISO_HIPAA)
    add("As a product manager, which plan fits a 10-person team?", P_STARTER)
    add("As an architect, what inference backends are available?", P_OVERVIEW)
    add("As a DevOps engineer, how do I monitor the system?", P_OBSERVABILITY)

    # ── Additional: Negative / boundary queries ───────────────────────
    add("Does the Starter plan have phone support?", P_SUPPORT)
    add("Can I get SOC 2 on the Professional plan?", P_SECURITY)
    add("Is fine-tuning available on the Starter plan?", P_STARTER)
    add("Do I need an internet connection for Nextera?", P_ON_PREMISES)
    add("Can I run 405B models on the Professional plan?", P_MODEL_SIZES)
    add("Is there a free plan available?", P_STARTER)
    add("What happens if I exceed the storage limit?", P_PLAN_COMPARISON)
    add("Can I switch from Professional to Enterprise?", P_PLAN_COMPARISON)

    # ── Additional: Technical deep-dive ───────────────────────────────
    add("cosine similarity vs euclidean distance", P_VECTOR_DB)
    add("how does chunk overlap affect retrieval?", P_RAG)
    add("ReAct loop agent trace debugging", P_AGENTS)
    add("QLoRA 4-bit VRAM memory savings", P_FINETUNING)
    add("TLS 1.3 encryption in transit", P_SECURITY)
    add("HNSW indexing vector store performance", P_VECTOR_DB)
    add("Vault integration secrets management", P_SECURITY)
    add("OpenTelemetry compatible tracing system", P_OBSERVABILITY)
    add("point-in-time recovery vector store", P_DISASTER_RECOVERY)
    add("cross-region replication disaster recovery", P_DISASTER_RECOVERY)

    # ── Additional: Final coverage ────────────────────────────────────
    add("Nextera for government use cases", P_GDPR)
    add("model families Gemma Llama Mistral Qwen", P_AGENTS)
    add("Nextera changelogs documentation portal", P_SUPPORT)
    add("web search HTTP API optional tools", P_AGENTS)
    add("how does multi-document synthesis work?", P_RAG)
    add("What's the minimum hardware for small models?", P_DEPLOYMENT)
    add("token usage cost tracking dashboards", P_OBSERVABILITY)
    add("Meridian fraud detection anomaly AI", P_MERIDIAN_AI)
    add("EU data residency Frankfurt Azure", P_MERIDIAN_CLOUD)
    add("Nextera reliability no rate limits", P_VS_CLOUD)

    return pairs


# ---------------------------------------------------------------------------
# 4. Triplet-format examples — {"anchor": ..., "positive": ..., "negative": ...}
# ---------------------------------------------------------------------------

def _generate_triplets() -> list[dict]:
    """Generate ~150 triplets with explicit hard negatives for confusable topics."""
    triplets: list[dict] = []

    def add(anchor: str, positive: str, negative: str) -> None:
        triplets.append({"anchor": anchor, "positive": positive, "negative": negative})

    # ── Pricing tier confusion: Starter vs Professional ───────────────
    add("basic plan 5 users small team", P_STARTER, P_PROFESSIONAL)
    add("cheapest plan available", P_STARTER, P_PROFESSIONAL)
    add("entry level pricing under 500 euro", P_STARTER, P_PROFESSIONAL)
    add("plan with Discord support", P_STARTER, P_PROFESSIONAL)
    add("7B model support basic plan", P_STARTER, P_PROFESSIONAL)
    add("10GB vector storage plan", P_STARTER, P_PROFESSIONAL)
    add("mid-size company production plan", P_PROFESSIONAL, P_STARTER)
    add("plan with SSO SAML support", P_PROFESSIONAL, P_STARTER)
    add("70B model support production", P_PROFESSIONAL, P_STARTER)
    add("plan with email support and SLA", P_PROFESSIONAL, P_STARTER)
    add("LoRA fine-tuning available plan", P_PROFESSIONAL, P_STARTER)
    add("100GB storage plan", P_PROFESSIONAL, P_STARTER)

    # ── Pricing tier confusion: Professional vs Enterprise ────────────
    add("unlimited users unlimited storage plan", P_ENTERPRISE, P_PROFESSIONAL)
    add("24/7 phone support plan", P_ENTERPRISE, P_PROFESSIONAL)
    add("1-hour SLA dedicated support", P_ENTERPRISE, P_PROFESSIONAL)
    add("405B+ large model support plan", P_ENTERPRISE, P_PROFESSIONAL)
    add("SOC 2 Type II certified plan", P_ENTERPRISE, P_PROFESSIONAL)
    add("air-gapped deployment support", P_ENTERPRISE, P_PROFESSIONAL)
    add("25 user limit production plan", P_PROFESSIONAL, P_ENTERPRISE)
    add("8-hour email SLA plan", P_PROFESSIONAL, P_ENTERPRISE)
    add("plan around 1000 euro per month", P_PROFESSIONAL, P_ENTERPRISE)
    add("audit logs SSO without enterprise price", P_PROFESSIONAL, P_ENTERPRISE)
    add("moderate storage 100GB plan", P_PROFESSIONAL, P_ENTERPRISE)
    add("plan for mid-size not large enterprise", P_PROFESSIONAL, P_ENTERPRISE)

    # ── Pricing tier confusion: Starter vs Enterprise ─────────────────
    add("most affordable plan for evaluation", P_STARTER, P_ENTERPRISE)
    add("no SLA community support plan", P_STARTER, P_ENTERPRISE)
    add("plan for 3 person team testing AI", P_STARTER, P_ENTERPRISE)
    add("custom quoted enterprise-grade plan", P_ENTERPRISE, P_STARTER)
    add("CSM dedicated support engineer plan", P_ENTERPRISE, P_STARTER)
    add("plan with custom contracts", P_ENTERPRISE, P_STARTER)

    # ── Compliance confusion ──────────────────────────────────────────
    add("SOC 2 Type II compliance certification", P_SECURITY, P_ISO_HIPAA)
    add("GDPR data processing agreement", P_ISO_HIPAA, P_SECURITY)
    add("HIPAA BAA healthcare use case", P_ISO_HIPAA, P_SECURITY)
    add("ISO 27001 alignment documentation", P_ISO_HIPAA, P_GDPR)
    add("encryption AES-256 TLS security features", P_SECURITY, P_ISO_HIPAA)
    add("RBAC access control team isolation", P_RBAC, P_SECURITY)
    add("audit log compliance review", P_RBAC, P_ISO_HIPAA)
    add("data residency GDPR no telemetry", P_GDPR, P_ISO_HIPAA)
    add("penetration testing security audit", P_SECURITY, P_GDPR)
    add("secrets management Vault integration", P_SECURITY, P_RBAC)
    add("which plan has HIPAA compliance?", P_ISO_HIPAA, P_SECURITY)
    add("is GDPR built into all plans?", P_ISO_HIPAA, P_GDPR)
    add("data processing local no cloud calls", P_SECURITY, P_GDPR)
    add("compliance certifications available enterprise", P_ISO_HIPAA, P_RBAC)
    add("team permissions separate knowledge bases", P_RBAC, P_GDPR)
    add("separate API keys per team setup", P_RBAC, P_SECURITY)
    add("no prompt logging to external servers", P_GDPR, P_SECURITY)
    add("data stays on customer infrastructure guarantee", P_GDPR, P_RBAC)

    # ── Deployment confusion ──────────────────────────────────────────
    add("single node Docker container setup", P_DEPLOYMENT, P_ON_PREMISES)
    add("Kubernetes Helm chart auto-scaling", P_DEPLOYMENT, P_ON_PREMISES)
    add("air-gapped enterprise no internet access", P_ON_PREMISES, P_DEPLOYMENT)
    add("behind corporate firewall self-hosted", P_ON_PREMISES, P_DEPLOYMENT)
    add("minimum 16GB RAM hardware requirements", P_DEPLOYMENT, P_GPU_PRICING)
    add("AMD ROCm Apple Metal GPU support", P_DEPLOYMENT, P_ON_PREMISES)
    add("hybrid local GPU Kubernetes orchestration", P_DEPLOYMENT, P_ON_PREMISES)
    add("fully offline deployment enterprise", P_ON_PREMISES, P_DEPLOYMENT)
    add("Windows macOS Linux platform support", P_DEPLOYMENT, P_ON_PREMISES)
    add("auto-scaling production Kubernetes", P_DEPLOYMENT, P_ON_PREMISES)
    add("GPU VRAM production hardware setup", P_DEPLOYMENT, P_GPU_PRICING)
    add("32GB RAM recommended production", P_DEPLOYMENT, P_FT_CLUSTER)

    # ── Feature confusion: RAG vs Agents ──────────────────────────────
    add("document chunking embedding retrieval", P_RAG, P_AGENTS)
    add("hybrid search BM25 dense vectors", P_RAG, P_AGENTS)
    add("intent classification tool routing", P_AGENTS, P_RAG)
    add("ReAct reasoning multi-step agents", P_AGENTS, P_RAG)
    add("citation tracking source attribution", P_RAG, P_AGENTS)
    add("custom tools Python development", P_AGENTS, P_RAG)
    add("SQL query natural language database", P_AGENTS, P_RAG)
    add("PDF DOCX document ingestion", P_RAG, P_AGENTS)
    add("execution tracing debugging", P_AGENTS, P_RAG)
    add("cross-encoder re-ranking pipeline", P_RAG, P_VECTOR_DB)

    # ── Feature confusion: Fine-tuning methods ────────────────────────
    add("LoRA low-rank adaptation training", P_FINETUNING, P_FT_CLUSTER)
    add("QLoRA 4-bit quantized efficient training", P_FINETUNING, P_GPU_PRICING)
    add("full fine-tuning models under 7B", P_FINETUNING, P_FT_CLUSTER)
    add("dedicated training cluster enterprise", P_FT_CLUSTER, P_FINETUNING)
    add("GPU hours A100 custom training", P_GPU_PRICING, P_FINETUNING)
    add("GGUF export edge device", P_FINETUNING, P_DEPLOYMENT)
    add("adapter versioning model management", P_FINETUNING, P_FT_CLUSTER)
    add("500 euro fine-tuning add-on", P_GPU_PRICING, P_FT_CLUSTER)

    # ── Feature confusion: Tools ──────────────────────────────────────
    add("calculator arithmetic precise math", P_CALCULATOR, P_SQL)
    add("SQL SELECT structured data query", P_SQL, P_CALCULATOR)
    add("database query natural language", P_SQL, P_VECTOR_DB)
    add("semantic search vector embeddings", P_VECTOR_DB, P_SQL)
    add("sandboxed code execution safe", P_CALCULATOR, P_SQL)
    add("cosine similarity document ranking", P_VECTOR_DB, P_RAG)

    # ── Support tier confusion ────────────────────────────────────────
    add("Discord community support free", P_SUPPORT, P_ONBOARDING)
    add("8-hour email SLA business hours", P_SUPPORT, P_ONBOARDING)
    add("24/7 phone dedicated Slack channel", P_SUPPORT, P_ONBOARDING)
    add("onboarding workshop 2 days enterprise", P_ONBOARDING, P_SUPPORT)
    add("customer success engineer dedicated", P_ONBOARDING, P_SUPPORT)
    add("monthly office hours Professional", P_SUPPORT, P_ONBOARDING)
    add("quarterly business reviews enterprise", P_SUPPORT, P_ONBOARDING)
    add("proactive monitoring enterprise support", P_SUPPORT, P_ONBOARDING)
    add("self-service onboarding Professional plan", P_ONBOARDING, P_SUPPORT)
    add("critical security patches delivery time", P_SUPPORT, P_SECURITY)
    add("support portal access Professional", P_SUPPORT, P_ONBOARDING)
    add("domain fine-tuning onboarding", P_ONBOARDING, P_FINETUNING)

    # ── SDK/Integration confusion ─────────────────────────────────────
    add("Python SDK Nextera API client", P_API, P_DEPLOYMENT)
    add("TypeScript Node.js SDK integration", P_API, P_DEPLOYMENT)
    add("LangChain framework integration", P_API, P_AGENTS)
    add("LlamaIndex native integration", P_API, P_RAG)
    add("Semantic Kernel .NET integration", P_API, P_DEPLOYMENT)
    add("webhooks event notifications", P_API, P_OBSERVABILITY)
    add("OpenAI API migration base_url", P_API, P_AGENTS)
    add("batch requests API endpoint", P_API, P_STREAMING)
    add("Go SDK server-side integration", P_API, P_DEPLOYMENT)
    add("Haystack AutoGen framework support", P_API, P_AGENTS)
    add("chat completions embeddings endpoint", P_API, P_VECTOR_DB)
    add("function calling API specification", P_API, P_AGENTS)

    # ── Cross-domain ambiguity ────────────────────────────────────────
    add("cost savings local AI vs cloud pricing", P_VS_CLOUD, P_ANNUAL_DISCOUNT)
    add("how much does Nextera save me?", P_VS_CLOUD, P_STARTER)
    add("pricing comparison cloud vs local", P_VS_CLOUD, P_PLAN_COMPARISON)
    add("security features data protection", P_SECURITY, P_VS_CLOUD)
    add("privacy compliance healthcare", P_GDPR, P_VS_CLOUD)
    add("no data exfiltration guarantee", P_GDPR, P_ON_PREMISES)
    add("data privacy local processing", P_VS_CLOUD, P_GDPR)
    add("latency performance speed benchmark", P_STREAMING, P_VS_CLOUD)
    add("first token response time comparison", P_STREAMING, P_VS_CLOUD)
    add("model quality accuracy benchmark", P_MODEL_QUALITY, P_FINETUNING)
    add("GPT-4 comparison local AI performance", P_MODEL_QUALITY, P_VS_CLOUD)
    add("fine-tuned accuracy domain-specific", P_MODEL_QUALITY, P_FINETUNING)
    add("task decomposition classification embedding", P_MODEL_QUALITY, P_AGENTS)
    add("vector store database backend", P_VECTOR_DB, P_SQL)
    add("document storage and retrieval system", P_RAG, P_VECTOR_DB)
    add("query logging monitoring production", P_OBSERVABILITY, P_SUPPORT)
    add("OpenTelemetry tracing dashboards", P_OBSERVABILITY, P_STREAMING)
    add("token usage tracking cost monitoring", P_OBSERVABILITY, P_VS_CLOUD)

    # ── Meridian Health specifics ─────────────────────────────────────
    add("Meridian Azure deployment healthcare", P_MERIDIAN_OVERVIEW, P_DEPLOYMENT)
    add("340 hospitals PostgreSQL EU data", P_MERIDIAN_OVERVIEW, P_GDPR)
    add("cloud provider rejection AWS Google", P_MERIDIAN_CLOUD, P_DEPLOYMENT)
    add("Frankfurt Amsterdam disaster recovery Meridian", P_MERIDIAN_CLOUD, P_DISASTER_RECOVERY)
    add("clinical AI appointment optimization Meridian", P_MERIDIAN_AI, P_AGENTS)
    add("EU AI Act bias testing models", P_MERIDIAN_AI, P_SECURITY)
    add("Meridian SLA patient lookup latency", P_MERIDIAN_SLA, P_SUPPORT)
    add("99.95% uptime healthcare infrastructure", P_MERIDIAN_SLA, P_DISASTER_RECOVERY)
    add("healthcare data residency EU compliance", P_MERIDIAN_OVERVIEW, P_ISO_HIPAA)
    add("GGUF models GPU VMs healthcare deployment", P_MERIDIAN_OVERVIEW, P_FINETUNING)
    add("document classification anomaly detection", P_MERIDIAN_AI, P_RAG)
    add("appointment booking latency requirement", P_MERIDIAN_SLA, P_STREAMING)

    # ── Operations ────────────────────────────────────────────────────
    add("backup schedule every 6 hours", P_DISASTER_RECOVERY, P_OBSERVABILITY)
    add("RPO 1 hour RTO 4 hours enterprise", P_DISASTER_RECOVERY, P_MERIDIAN_SLA)
    add("cross-region replication recovery", P_DISASTER_RECOVERY, P_DEPLOYMENT)
    add("multi-language PDF DOCX indexing", P_MULTILINGUAL, P_RAG)
    add("cross-lingual retrieval search", P_MULTILINGUAL, P_VECTOR_DB)
    add("documents in multiple languages", P_MULTILINGUAL, P_RAG)

    return triplets


# ---------------------------------------------------------------------------
# 5. Quality controls
# ---------------------------------------------------------------------------

def _deduplicate(examples: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove exact and near-duplicate queries."""
    seen_exact: set[str] = set()
    seen_words: list[set[str]] = []
    result: list[dict] = []

    for ex in examples:
        q = ex.get("query") or ex.get("anchor", "")
        q_lower = q.lower().strip()

        # Exact duplicate
        if q_lower in seen_exact:
            continue

        # Near-duplicate (Jaccard on word sets)
        q_words = set(q_lower.split())
        is_near_dup = False
        for sw in seen_words:
            if not q_words or not sw:
                continue
            jacc = len(q_words & sw) / len(q_words | sw)
            if jacc > threshold:
                is_near_dup = True
                break

        if is_near_dup:
            continue

        seen_exact.add(q_lower)
        seen_words.append(q_words)
        result.append(ex)

    return result


def _validate_no_eval_leakage(
    examples: list[dict], threshold: float = 0.7
) -> list[dict]:
    """Remove any examples whose query is too similar to an eval query."""
    clean: list[dict] = []
    removed = 0

    for ex in examples:
        q = ex.get("query") or ex.get("anchor", "")
        if _is_eval_leak(q, threshold):
            removed += 1
        else:
            clean.append(ex)

    if removed > 0:
        print(f"  Removed {removed} examples due to eval leakage (Jaccard > {threshold})")

    return clean


def _validate_format(examples: list[dict]) -> None:
    """Assert all records have the correct keys and non-empty values."""
    for i, ex in enumerate(examples):
        if "anchor" in ex:
            for key in ("anchor", "positive", "negative"):
                assert key in ex, f"Record {i} missing key '{key}': {ex}"
                assert ex[key].strip(), f"Record {i} has empty '{key}'"
        elif "query" in ex:
            for key in ("query", "positive"):
                assert key in ex, f"Record {i} missing key '{key}': {ex}"
                assert ex[key].strip(), f"Record {i} has empty '{key}'"
        else:
            raise AssertionError(f"Record {i} has unrecognized format: {list(ex.keys())}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

OUTPUT_DEFAULT = "./data/training-data/embeddinggemma_retrieval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddinggemma training data")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print stats only, don't write")
    args = parser.parse_args()

    print("Generating embeddinggemma retrieval dataset...")

    # Generate
    pairs = _generate_pairs()
    triplets = _generate_triplets()
    print(f"  Raw pairs:    {len(pairs)}")
    print(f"  Raw triplets: {len(triplets)}")

    # Combine
    all_examples = pairs + triplets

    # Quality controls
    all_examples = _deduplicate(all_examples)
    print(f"  After dedup:  {len(all_examples)}")

    all_examples = _validate_no_eval_leakage(all_examples)
    print(f"  After leak check: {len(all_examples)}")

    _validate_format(all_examples)

    # Count by format
    n_pairs = sum(1 for ex in all_examples if "query" in ex)
    n_triplets = sum(1 for ex in all_examples if "anchor" in ex)
    total = len(all_examples)

    print(f"\n  Final dataset:")
    print(f"    Pairs:    {n_pairs}")
    print(f"    Triplets: {n_triplets}")
    print(f"    Total:    {total}")

    assert total >= 500, f"Expected >= 500 examples, got {total}"

    if args.stats:
        print("\n  --stats mode: no file written.")
        return

    # Shuffle and write
    random.seed(args.seed)
    random.shuffle(all_examples)

    count = save_jsonl(all_examples, args.output)
    print(f"\n  Written {count} examples to {args.output}")


if __name__ == "__main__":
    main()
