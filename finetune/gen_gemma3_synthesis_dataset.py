"""
Generate synthesis training data for gemma3 — response generation from context.

Produces 200+ examples across 5 synthesis categories, using the EXACT prompt
templates from agent.py.  All outputs are hand-crafted and grounded in the
13 KB documents and the SQL seed data from data/loader.py.

Run:
    python -m finetune.gen_gemma3_synthesis_dataset
    python -m finetune.gen_gemma3_synthesis_dataset --stats
    python -m finetune.gen_gemma3_synthesis_dataset --output path/to/out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random

# ---------------------------------------------------------------------------
# Exact prompt templates from src/agent.py
# ---------------------------------------------------------------------------

# RAG synthesis — agent.py L368-372
INSTR_RAG = (
    "You are a factual assistant for Nextera platform questions. "
    "Answer precisely from the provided sources. Use bullet points for lists. "
    "Never invent information not in the sources."
)

# Tool result formatting — agent.py L956
INSTR_TOOL = "Turn this tool result into a clear, helpful answer for the user."

# Multi-step synthesis — agent.py L846
INSTR_MULTI = "Combine these tool results into a clear, helpful answer."

# Direct answer — agent.py L1036
INSTR_DIRECT = "You are a helpful, concise AI assistant."

# ---------------------------------------------------------------------------
# KB passage constants (from data/business-documents/*.md, stripped of YAML front matter)
# Truncated to ~300 chars to match agent.py L353 behavior.
# ---------------------------------------------------------------------------

DOC_STARTER = (
    "The Nextera Starter plan costs €299 per month (or €2,990 per year with "
    "two months free). It includes: up to 5 concurrent users, 10 GB vector "
    "storage, support for models up to 7B parameters, community support via "
    "Discord, and monthly model updates. Starter is ideal for teams evaluating "
    "local AI or running small RAG pipelines."
)

DOC_PROFESSIONAL = (
    "The Nextera Professional plan is €999 per month (or €9,990 per year). "
    "It includes: up to 25 concurrent users, 100 GB vector storage, models "
    "up to 70B parameters, fine-tuning support (LoRA), priority email support "
    "with 8-hour SLA, audit logs, SSO via SAML 2.0, and access to the Nextera "
    "API. Recommended for production deployments at mid-size companies."
)

DOC_ENTERPRISE = (
    "The Nextera Enterprise plan starts at €3,500 per month and is "
    "custom-quoted based on infrastructure scale. It includes: unlimited "
    "concurrent users, unlimited vector storage, support for all model sizes "
    "including 405B+ parameter models, dedicated fine-tuning cluster, 24/7 "
    "phone and Slack support with 1-hour SLA, custom model deployment, "
    "air-gapped deployment option, SOC 2 Type II compliance docs, and a "
    "dedicated Customer Success Manager."
)

DOC_RAG = (
    "Nextera ships with a production-ready RAG pipeline out of the box. "
    "Features include: automatic document chunking (configurable overlap), "
    "hybrid search (dense + sparse BM25), re-ranking with cross-encoders, "
    "query rewriting for improved retrieval, multi-document synthesis, "
    "citation tracking with source attribution, and streaming responses. "
    "Supported document types: PDF, DOCX, TXT, Markdown, HTML, CSV."
)

DOC_AGENTS = (
    "Nextera supports building multi-step AI agents with tool use. The agent "
    "framework includes: intent classification, tool registry, function calling "
    "(OpenAI-compatible), ReAct-style reasoning loops, multi-agent orchestration, "
    "and execution tracing for debugging. Built-in tools: SQL query, vector "
    "search, web search (optional), code execution (sandboxed), and HTTP API calls."
)

DOC_FINETUNING = (
    "Nextera Professional and Enterprise plans include on-device fine-tuning. "
    "Supported methods: LoRA, QLoRA (4-bit quantized LoRA), and full fine-tuning "
    "for models under 7B. The platform automatically collects interaction logs "
    "that can be used as training data, with a one-click fine-tuning pipeline. "
    "Typical fine-tuning time: 30-60 minutes for 500 examples on a single A100."
)

DOC_API = (
    "Nextera exposes a fully OpenAI-compatible REST API. Any existing application "
    "built for OpenAI's API works with Nextera with a single base_url change — "
    "no code modifications needed. The API supports: chat completions, embeddings, "
    "function calling, streaming, and batch requests. Official SDKs: Python, "
    "TypeScript/Node, Go, and .NET C#. Native integrations: LangChain, LlamaIndex, "
    "Semantic Kernel, Haystack, and AutoGen."
)

DOC_DEPLOYMENT = (
    "Nextera can be deployed in multiple configurations: (1) Single-node Docker — "
    "runs on any Linux, macOS, or Windows machine with 16 GB+ RAM. "
    "(2) Kubernetes — Helm chart provided, supports auto-scaling. "
    "(3) Air-gapped — no internet connection required (Enterprise only). "
    "(4) Hybrid — inference on local GPU, orchestration on Kubernetes. "
    "Hardware requirements: minimum 8 GB RAM for small models (<7B), "
    "recommended 32 GB + NVIDIA GPU with 16 GB VRAM for production."
)

DOC_SECURITY = (
    "Nextera is designed for organizations with strict data security requirements. "
    "All data processing is local — no telemetry, no cloud calls, no model API usage. "
    "Compliance: SOC 2 Type II (Enterprise), GDPR-ready architecture, ISO 27001 "
    "alignment documentation, and HIPAA BAA available on Enterprise. Security "
    "features: end-to-end encryption at rest (AES-256), TLS 1.3 in transit, "
    "role-based access control (RBAC), audit logging."
)

DOC_SUPPORT = (
    "Nextera support tiers by plan: Starter: community Discord, documentation, "
    "no guaranteed SLA. Professional: email support, 8-hour response SLA "
    "(business hours), access to private support portal, monthly office hours. "
    "Enterprise: 24/7 phone + dedicated Slack channel, 1-hour response SLA, "
    "proactive monitoring, quarterly business reviews, dedicated CSM."
)

DOC_OVERVIEW = (
    "Nextera Platform is a local-first AI infrastructure product that lets "
    "enterprises run large language models, vector search, and agentic pipelines "
    "entirely on their own hardware — no data ever leaves their network. Nextera "
    "ships as a single Docker container and supports Ollama, llama.cpp, and vLLM "
    "as inference backends."
)

DOC_FAQ_QUALITY = (
    "For specialized tasks, fine-tuned small models often outperform large "
    "general-purpose models. A 1B-parameter model fine-tuned on your company's "
    "knowledge base will answer domain-specific questions more accurately than "
    "GPT-4 answering cold. The key insight is task decomposition: instead of one "
    "model doing everything, use three specialized models."
)

DOC_FAQ_CLOUD = (
    "Why run AI locally instead of using cloud APIs? Privacy: Your prompts, "
    "documents, and outputs never leave your network. Critical for healthcare, "
    "legal, finance, and government use cases. Cost: After initial hardware, "
    "inference is free. A team of 50 running 100 queries/day would spend "
    "~€5,000/month on cloud APIs but €0 on local inference after setup. "
    "Latency: Local inference on a modern GPU achieves <200ms first token."
)


# ---------------------------------------------------------------------------
# Helper: build RAG context block (matches agent.py L353-356)
# ---------------------------------------------------------------------------

def _rag_context(*docs: tuple[str, str]) -> str:
    """Build a SOURCES block from (title, content) pairs."""
    parts = []
    for title, content in docs:
        parts.append(f"[Source: {title}]\n{content[:300]}")
    return "\n\n---\n\n".join(parts)


def _rag_input(context: str, question: str) -> str:
    """Build the full RAG user prompt matching agent.py L359-367."""
    return (
        "Answer the question using ONLY the sources below.\n"
        "Rules: use exact facts from the sources. If the question asks about "
        "multiple items, list each one. Cite [Source: title] for key facts.\n\n"
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def _tool_input(question: str, tool_name: str, raw_result: str) -> str:
    """Build the tool formatting user prompt matching agent.py L956-963."""
    return (
        f"User's question: {question}\n"
        f"Tool used: {tool_name}\n"
        f"Raw result:\n{raw_result}\n\n"
        "Write a concise, human-readable answer:"
    )


def _multi_input(question: str, results_str: str) -> str:
    """Build the multi-step synthesis prompt matching agent.py L846-849."""
    return (
        f"User's question: {question}\n\n"
        f"Results:\n{results_str}\n\n"
        "Write a concise answer that integrates all results:"
    )


# ---------------------------------------------------------------------------
# 1. RAG Synthesis examples (~80)
# ---------------------------------------------------------------------------

def _generate_rag_examples() -> list[dict]:
    examples = []

    def add(question: str, sources: list[tuple[str, str]], answer: str) -> None:
        ctx = _rag_context(*sources)
        examples.append({
            "instruction": INSTR_RAG,
            "input": _rag_input(ctx, question),
            "output": answer,
        })

    # --- Pricing: Starter ---
    add(
        "How much does the Starter plan cost?",
        [("Starter Plan Pricing", DOC_STARTER), ("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "The Nextera Starter plan costs €299 per month, or €2,990 per year (two months free) [Source: Starter Plan Pricing].",
    )
    add(
        "What's included in the Starter plan?",
        [("Starter Plan Pricing", DOC_STARTER), ("Support Tiers and SLAs", DOC_SUPPORT)],
        "The Starter plan includes:\n- Up to 5 concurrent users\n- 10 GB vector storage\n- Models up to 7B parameters\n- Community support via Discord\n- Monthly model updates\n[Source: Starter Plan Pricing]",
    )
    add(
        "Is the Starter plan right for a small team evaluating AI?",
        [("Starter Plan Pricing", DOC_STARTER), ("Nextera Platform Overview", DOC_OVERVIEW)],
        "Yes, the Starter plan is ideal for teams evaluating local AI or running small RAG pipelines [Source: Starter Plan Pricing]. It provides core capabilities at €299/month with up to 5 concurrent users.",
    )
    add(
        "What is the annual price for the Starter plan?",
        [("Starter Plan Pricing", DOC_STARTER)],
        "The Starter plan costs €2,990 per year, which includes two months free compared to monthly billing (€299/month) [Source: Starter Plan Pricing].",
    )

    # --- Pricing: Professional ---
    add(
        "What does the Professional plan include?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL), ("Support Tiers and SLAs", DOC_SUPPORT)],
        "The Professional plan (€999/month) includes:\n- Up to 25 concurrent users\n- 100 GB vector storage\n- Models up to 70B parameters\n- Fine-tuning support (LoRA)\n- Priority email support with 8-hour SLA\n- Audit logs and SSO via SAML 2.0\n- Access to the Nextera API\n[Source: Professional Plan Pricing]",
    )
    add(
        "Does the Professional plan support fine-tuning?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL), ("Fine-Tuning Capabilities", DOC_FINETUNING)],
        "Yes, the Professional plan includes fine-tuning support via LoRA [Source: Professional Plan Pricing]. Supported methods include LoRA, QLoRA, and full fine-tuning for models under 7B [Source: Fine-Tuning Capabilities].",
    )
    add(
        "How much storage comes with the Professional plan?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "The Professional plan includes 100 GB of vector storage [Source: Professional Plan Pricing].",
    )
    add(
        "What is the annual cost of the Professional plan?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "The Professional plan costs €9,990 per year [Source: Professional Plan Pricing].",
    )

    # --- Pricing: Enterprise ---
    add(
        "What are the features of the Enterprise plan?",
        [("Enterprise Plan Pricing", DOC_ENTERPRISE), ("Support Tiers and SLAs", DOC_SUPPORT)],
        "The Enterprise plan (starting at €3,500/month) includes:\n- Unlimited concurrent users and vector storage\n- Support for all model sizes including 405B+ parameters\n- Dedicated fine-tuning cluster\n- 24/7 phone and Slack support with 1-hour SLA\n- Custom model deployment and air-gapped deployment\n- SOC 2 Type II compliance docs\n- Dedicated Customer Success Manager\n[Source: Enterprise Plan Pricing]",
    )
    add(
        "Can the Enterprise plan be deployed without internet?",
        [("Enterprise Plan Pricing", DOC_ENTERPRISE), ("Deployment Options", DOC_DEPLOYMENT)],
        "Yes, the Enterprise plan supports air-gapped deployment — no internet connection required [Source: Enterprise Plan Pricing]. Nextera can be deployed fully offline with all models running on-premises [Source: Deployment Options].",
    )
    add(
        "Does Enterprise include a dedicated support person?",
        [("Enterprise Plan Pricing", DOC_ENTERPRISE), ("Support Tiers and SLAs", DOC_SUPPORT)],
        "Yes, the Enterprise plan includes a dedicated Customer Success Manager [Source: Enterprise Plan Pricing] and a dedicated Slack channel with 1-hour response SLA, plus quarterly business reviews [Source: Support Tiers and SLAs].",
    )

    # --- Pricing: Comparisons ---
    add(
        "What is the difference between the Starter and Professional plans?",
        [("Starter Plan Pricing", DOC_STARTER), ("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "Key differences:\n- **Price**: Starter €299/month vs Professional €999/month\n- **Users**: 5 vs 25 concurrent users\n- **Storage**: 10 GB vs 100 GB vector storage\n- **Models**: Up to 7B vs up to 70B parameters\n- **Fine-tuning**: Not included vs LoRA support\n- **Support**: Discord community vs email with 8-hour SLA\n- **Features**: Professional adds SSO, audit logs, and API access\n[Source: Starter Plan Pricing] [Source: Professional Plan Pricing]",
    )
    add(
        "Compare Professional and Enterprise plans.",
        [("Professional Plan Pricing", DOC_PROFESSIONAL), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "Key differences:\n- **Price**: Professional €999/month vs Enterprise €3,500+/month\n- **Users**: 25 vs unlimited\n- **Storage**: 100 GB vs unlimited\n- **Models**: Up to 70B vs all sizes (405B+)\n- **Support**: 8-hour email SLA vs 1-hour phone/Slack SLA\n- **Enterprise adds**: Dedicated fine-tuning cluster, air-gapped deployment, SOC 2 Type II, dedicated CSM\n[Source: Professional Plan Pricing] [Source: Enterprise Plan Pricing]",
    )
    add(
        "Which plan is best for a 50-person company?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "For a 50-person company, the Enterprise plan would be needed since the Professional plan supports only up to 25 concurrent users [Source: Professional Plan Pricing]. The Enterprise plan offers unlimited users and includes dedicated support [Source: Enterprise Plan Pricing].",
    )
    add(
        "What are all the pricing tiers?",
        [("Starter Plan Pricing", DOC_STARTER), ("Professional Plan Pricing", DOC_PROFESSIONAL), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "Nextera offers three pricing tiers:\n- **Starter**: €299/month (€2,990/year) — up to 5 users, 7B models [Source: Starter Plan Pricing]\n- **Professional**: €999/month (€9,990/year) — up to 25 users, 70B models, LoRA fine-tuning [Source: Professional Plan Pricing]\n- **Enterprise**: €3,500+/month — unlimited users, all model sizes, air-gapped deployment [Source: Enterprise Plan Pricing]",
    )

    # --- Features: RAG ---
    add(
        "What RAG features does Nextera include?",
        [("RAG Pipeline Features", DOC_RAG), ("Nextera Platform Overview", DOC_OVERVIEW)],
        "Nextera includes a production-ready RAG pipeline with:\n- Automatic document chunking with configurable overlap\n- Hybrid search (dense + sparse BM25)\n- Re-ranking with cross-encoders\n- Query rewriting for improved retrieval\n- Multi-document synthesis\n- Citation tracking with source attribution\n- Streaming responses\n[Source: RAG Pipeline Features]",
    )
    add(
        "What document types can Nextera process?",
        [("RAG Pipeline Features", DOC_RAG)],
        "Nextera supports the following document types: PDF, DOCX, TXT, Markdown, HTML, and CSV [Source: RAG Pipeline Features].",
    )
    add(
        "Does Nextera support hybrid search?",
        [("RAG Pipeline Features", DOC_RAG)],
        "Yes, Nextera supports hybrid search combining dense vector search with sparse BM25 for improved retrieval accuracy [Source: RAG Pipeline Features].",
    )

    # --- Features: Agents ---
    add(
        "What agent capabilities does Nextera offer?",
        [("Agentic Pipeline Features", DOC_AGENTS), ("Nextera Platform Overview", DOC_OVERVIEW)],
        "Nextera's agent framework includes:\n- Intent classification\n- Tool registry and function calling (OpenAI-compatible)\n- ReAct-style reasoning loops\n- Multi-agent orchestration\n- Execution tracing for debugging\n- Built-in tools: SQL query, vector search, web search, code execution (sandboxed), HTTP API calls\n[Source: Agentic Pipeline Features]",
    )
    add(
        "Can I add custom tools to the agent framework?",
        [("Agentic Pipeline Features", DOC_AGENTS)],
        "Yes, custom tools can be added in Python with a simple base class [Source: Agentic Pipeline Features].",
    )
    add(
        "What model families does the agent framework support?",
        [("Agentic Pipeline Features", DOC_AGENTS)],
        "The agent framework supports Gemma, Llama, Mistral, and Qwen model families [Source: Agentic Pipeline Features].",
    )

    # --- Features: Fine-tuning ---
    add(
        "What fine-tuning methods are supported?",
        [("Fine-Tuning Capabilities", DOC_FINETUNING), ("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "Nextera supports three fine-tuning methods:\n- LoRA (Low-Rank Adaptation)\n- QLoRA (4-bit quantized LoRA)\n- Full fine-tuning for models under 7B\nFine-tuning is available on Professional and Enterprise plans [Source: Fine-Tuning Capabilities].",
    )
    add(
        "How long does fine-tuning take?",
        [("Fine-Tuning Capabilities", DOC_FINETUNING)],
        "Typical fine-tuning time is 30-60 minutes for 500 examples on a single A100 GPU [Source: Fine-Tuning Capabilities].",
    )
    add(
        "Can I export fine-tuned models to other formats?",
        [("Fine-Tuning Capabilities", DOC_FINETUNING)],
        "Yes, GGUF export is supported for deployment to edge devices. Fine-tuned adapters are also automatically deployed to Ollama and versioned [Source: Fine-Tuning Capabilities].",
    )

    # --- Integration: API ---
    add(
        "Is the Nextera API compatible with OpenAI?",
        [("Nextera API & Integrations", DOC_API), ("Nextera Platform Overview", DOC_OVERVIEW)],
        "Yes, Nextera exposes a fully OpenAI-compatible REST API. Any existing application built for OpenAI's API works with a single base_url change — no code modifications needed [Source: Nextera API & Integrations].",
    )
    add(
        "What SDKs does Nextera provide?",
        [("Nextera API & Integrations", DOC_API)],
        "Nextera provides official SDKs for:\n- Python\n- TypeScript/Node\n- Go\n- .NET C#\n[Source: Nextera API & Integrations]",
    )
    add(
        "Does Nextera integrate with LangChain?",
        [("Nextera API & Integrations", DOC_API)],
        "Yes, Nextera has native integrations with LangChain, LlamaIndex, Semantic Kernel, Haystack, and AutoGen [Source: Nextera API & Integrations].",
    )
    add(
        "What API features does Nextera support?",
        [("Nextera API & Integrations", DOC_API)],
        "The Nextera API supports chat completions, embeddings, function calling, streaming, and batch requests. Webhooks are available on Professional and higher plans [Source: Nextera API & Integrations].",
    )

    # --- Deployment ---
    add(
        "How do I deploy Nextera?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "Nextera supports four deployment modes:\n1. **Single-node Docker** — any Linux, macOS, or Windows machine with 16 GB+ RAM\n2. **Kubernetes** — Helm chart with auto-scaling\n3. **Air-gapped** — no internet required (Enterprise only)\n4. **Hybrid** — local GPU inference with Kubernetes orchestration\n[Source: Deployment Options]",
    )
    add(
        "What are the hardware requirements for Nextera?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "Minimum: 8 GB RAM for small models (<7B). Recommended for production: 32 GB RAM + NVIDIA GPU with 16 GB VRAM. AMD ROCm and Apple Silicon (Metal) are also supported [Source: Deployment Options].",
    )
    add(
        "Does Nextera support Kubernetes?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "Yes, Nextera provides a Helm chart for Kubernetes deployment with auto-scaling support [Source: Deployment Options].",
    )
    add(
        "Can Nextera run on Apple Silicon?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "Yes, Apple Silicon (Metal) is supported alongside NVIDIA CUDA and AMD ROCm [Source: Deployment Options].",
    )

    # --- Security ---
    add(
        "What compliance certifications does Nextera have?",
        [("Security and Compliance", DOC_SECURITY), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "Nextera's compliance certifications include:\n- SOC 2 Type II (Enterprise plan)\n- GDPR-ready architecture\n- ISO 27001 alignment documentation\n- HIPAA BAA available on Enterprise\n[Source: Security and Compliance]",
    )
    add(
        "How does Nextera handle data security?",
        [("Security and Compliance", DOC_SECURITY)],
        "All data processing is local — no telemetry, no cloud calls, no model API usage. Security features include:\n- AES-256 encryption at rest\n- TLS 1.3 in transit\n- Role-based access control (RBAC)\n- Audit logging\n- Secrets management via Vault integration\n- Regular third-party penetration testing\n[Source: Security and Compliance]",
    )
    add(
        "Is Nextera HIPAA compliant?",
        [("Security and Compliance", DOC_SECURITY), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "HIPAA BAA is available on the Enterprise plan [Source: Security and Compliance]. Combined with the local-only data processing architecture, Nextera can meet HIPAA requirements for healthcare organizations.",
    )
    add(
        "Does Nextera send any data to external servers?",
        [("Security and Compliance", DOC_SECURITY), ("FAQ: Local AI vs Cloud AI", DOC_FAQ_CLOUD)],
        "No. All data processing is local — no telemetry, no cloud calls, no model API usage [Source: Security and Compliance]. Your prompts, documents, and outputs never leave your network [Source: FAQ: Local AI vs Cloud AI].",
    )

    # --- Support ---
    add(
        "What support options are available?",
        [("Support Tiers and SLAs", DOC_SUPPORT)],
        "Support tiers by plan:\n- **Starter**: Community Discord, documentation, no guaranteed SLA\n- **Professional**: Email support with 8-hour SLA, private support portal, monthly office hours\n- **Enterprise**: 24/7 phone + Slack, 1-hour SLA, proactive monitoring, quarterly reviews, dedicated CSM\nAll plans include access to the documentation portal. Critical security patches are delivered within 24 hours [Source: Support Tiers and SLAs].",
    )
    add(
        "What is the Enterprise SLA response time?",
        [("Support Tiers and SLAs", DOC_SUPPORT), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "Enterprise support has a 1-hour response SLA with 24/7 availability via phone and a dedicated Slack channel [Source: Support Tiers and SLAs].",
    )
    add(
        "Does the Starter plan include any support?",
        [("Support Tiers and SLAs", DOC_SUPPORT), ("Starter Plan Pricing", DOC_STARTER)],
        "The Starter plan includes community Discord support and access to the public documentation portal, but has no guaranteed SLA [Source: Support Tiers and SLAs].",
    )

    # --- FAQ ---
    add(
        "Are local models as good as GPT-4?",
        [("FAQ: Are Local Models as Good as GPT-4?", DOC_FAQ_QUALITY)],
        "For specialized tasks, fine-tuned small models often outperform large general-purpose models. A 1B-parameter model fine-tuned on your knowledge base answers domain-specific questions more accurately than GPT-4 answering cold. The key is task decomposition: three specialized models together produce GPT-4-level results on domain tasks at a fraction of the cost [Source: FAQ: Are Local Models as Good as GPT-4?].",
    )
    add(
        "Why should I use local AI instead of cloud APIs?",
        [("FAQ: Local AI vs Cloud AI", DOC_FAQ_CLOUD)],
        "Benefits of local AI:\n- **Privacy**: Prompts and documents never leave your network\n- **Cost**: €0 inference after hardware setup (vs ~€5,000/month on cloud APIs for a 50-person team)\n- **Latency**: <200ms first token on GPU (vs 500-2000ms cloud)\n- **Reliability**: No rate limits or provider outages\n- **Control**: Fine-tune on proprietary data without sharing it\n[Source: FAQ: Local AI vs Cloud AI]",
    )
    add(
        "How much money can I save by running AI locally?",
        [("FAQ: Local AI vs Cloud AI", DOC_FAQ_CLOUD)],
        "A team of 50 running 100 queries/day would spend approximately €5,000/month on cloud APIs, but €0 on local inference after the initial hardware setup [Source: FAQ: Local AI vs Cloud AI].",
    )
    add(
        "What is Nextera Platform?",
        [("Nextera Platform Overview", DOC_OVERVIEW), ("FAQ: Local AI vs Cloud AI", DOC_FAQ_CLOUD)],
        "Nextera Platform is a local-first AI infrastructure product that lets enterprises run large language models, vector search, and agentic pipelines entirely on their own hardware. No data ever leaves your network. It ships as a single Docker container and supports Ollama, llama.cpp, and vLLM as inference backends [Source: Nextera Platform Overview].",
    )

    # --- Cross-doc ---
    add(
        "Which plan supports HIPAA compliance and unlimited users?",
        [("Enterprise Plan Pricing", DOC_ENTERPRISE), ("Security and Compliance", DOC_SECURITY)],
        "The Enterprise plan — it includes HIPAA BAA, unlimited concurrent users, and unlimited vector storage [Source: Enterprise Plan Pricing]. SOC 2 Type II compliance documentation is also included [Source: Security and Compliance].",
    )
    add(
        "Can I use Nextera with my existing OpenAI-based application and deploy on Kubernetes?",
        [("Nextera API & Integrations", DOC_API), ("Deployment Options", DOC_DEPLOYMENT)],
        "Yes. Nextera's API is fully OpenAI-compatible — change only the base_url [Source: Nextera API & Integrations]. For deployment, a Helm chart with auto-scaling is provided for Kubernetes [Source: Deployment Options].",
    )
    add(
        "What fine-tuning features are available on the Professional plan?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL), ("Fine-Tuning Capabilities", DOC_FINETUNING)],
        "The Professional plan includes LoRA fine-tuning [Source: Professional Plan Pricing]. The platform supports LoRA, QLoRA, and full fine-tuning for models under 7B, with automatic interaction log collection and a one-click pipeline [Source: Fine-Tuning Capabilities].",
    )
    add(
        "How does Nextera ensure data privacy for a healthcare company?",
        [("Security and Compliance", DOC_SECURITY), ("FAQ: Local AI vs Cloud AI", DOC_FAQ_CLOUD), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "Nextera ensures healthcare data privacy through:\n- All processing is local — no telemetry, no cloud calls [Source: Security and Compliance]\n- HIPAA BAA available on Enterprise [Source: Security and Compliance]\n- Prompts and documents never leave your network [Source: FAQ: Local AI vs Cloud AI]\n- Air-gapped deployment option for maximum isolation [Source: Enterprise Plan Pricing]",
    )
    add(
        "What is task decomposition and how does it work with Nextera?",
        [("FAQ: Are Local Models as Good as GPT-4?", DOC_FAQ_QUALITY), ("Agentic Pipeline Features", DOC_AGENTS)],
        "Task decomposition means using multiple specialized models instead of one large model. Nextera uses three specialized models — a classification model, a function-calling model, and an embedding model — that together produce GPT-4-level results on domain tasks [Source: FAQ: Are Local Models as Good as GPT-4?]. The agent framework orchestrates them with intent classification and tool routing [Source: Agentic Pipeline Features].",
    )
    add(
        "Does Nextera support both Docker and Kubernetes deployments?",
        [("Deployment Options", DOC_DEPLOYMENT), ("Nextera Platform Overview", DOC_OVERVIEW)],
        "Yes. Nextera can be deployed as a single-node Docker container or via a Kubernetes Helm chart with auto-scaling. A hybrid option is also available with local GPU inference and Kubernetes orchestration [Source: Deployment Options].",
    )
    add(
        "What vector store does Nextera use for RAG?",
        [("RAG Pipeline Features", DOC_RAG), ("Nextera Platform Overview", DOC_OVERVIEW)],
        "The RAG pipeline's vector store is backed by ChromaDB or Weaviate (configurable) [Source: RAG Pipeline Features].",
    )
    add(
        "How does re-ranking work in the RAG pipeline?",
        [("RAG Pipeline Features", DOC_RAG)],
        "The RAG pipeline uses cross-encoder re-ranking to improve result quality. After initial hybrid retrieval (dense + sparse BM25), a cross-encoder re-ranks the results for better precision [Source: RAG Pipeline Features].",
    )
    add(
        "What is the cheapest plan for a 3-person team that needs RAG?",
        [("Starter Plan Pricing", DOC_STARTER), ("RAG Pipeline Features", DOC_RAG)],
        "The Starter plan at €299/month would work — it supports up to 5 concurrent users and includes the full RAG pipeline [Source: Starter Plan Pricing]. All RAG features (chunking, hybrid search, re-ranking, query rewriting) are available on every plan [Source: RAG Pipeline Features].",
    )
    add(
        "What encryption does Nextera use?",
        [("Security and Compliance", DOC_SECURITY)],
        "Nextera uses AES-256 encryption at rest and TLS 1.3 for data in transit [Source: Security and Compliance].",
    )
    add(
        "Does Nextera support streaming responses?",
        [("RAG Pipeline Features", DOC_RAG), ("Nextera API & Integrations", DOC_API)],
        "Yes, the RAG pipeline supports streaming responses [Source: RAG Pipeline Features], and the API also supports streaming for chat completions [Source: Nextera API & Integrations].",
    )
    add(
        "What inference backends does Nextera support?",
        [("Nextera Platform Overview", DOC_OVERVIEW)],
        "Nextera supports Ollama, llama.cpp, and vLLM as inference backends [Source: Nextera Platform Overview].",
    )
    add(
        "Can I use Nextera with Semantic Kernel?",
        [("Nextera API & Integrations", DOC_API)],
        "Yes, Nextera has a native integration with Semantic Kernel, along with LangChain, LlamaIndex, Haystack, and AutoGen [Source: Nextera API & Integrations].",
    )
    add(
        "What GPU hardware is recommended for production?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "For production use, a machine with 32 GB RAM and an NVIDIA GPU with at least 16 GB VRAM is recommended. AMD ROCm and Apple Silicon (Metal) are also supported [Source: Deployment Options].",
    )

    # --- Additional RAG examples for diversity ---
    add(
        "What model sizes are supported on each plan?",
        [("Starter Plan Pricing", DOC_STARTER), ("Professional Plan Pricing", DOC_PROFESSIONAL), ("Enterprise Plan Pricing", DOC_ENTERPRISE)],
        "Model size limits by plan:\n- **Starter**: Up to 7B parameters [Source: Starter Plan Pricing]\n- **Professional**: Up to 70B parameters [Source: Professional Plan Pricing]\n- **Enterprise**: All sizes including 405B+ parameters [Source: Enterprise Plan Pricing]",
    )
    add(
        "Does Nextera support audit logging?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL), ("Security and Compliance", DOC_SECURITY)],
        "Yes, audit logging is available on the Professional plan and above [Source: Professional Plan Pricing]. The Security and Compliance features include audit logging as part of the security framework [Source: Security and Compliance].",
    )
    add(
        "How does query rewriting improve retrieval?",
        [("RAG Pipeline Features", DOC_RAG)],
        "The RAG pipeline includes automatic query rewriting to improve retrieval accuracy. User queries are reformulated into more effective search phrases before being used for document retrieval [Source: RAG Pipeline Features].",
    )
    add(
        "What is the difference between LoRA and QLoRA?",
        [("Fine-Tuning Capabilities", DOC_FINETUNING)],
        "LoRA (Low-Rank Adaptation) trains adapter matrices at full precision. QLoRA is a 4-bit quantized version that uses less memory while maintaining similar quality. Both are supported on Professional and Enterprise plans [Source: Fine-Tuning Capabilities].",
    )
    add(
        "Does Nextera support SSO?",
        [("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "Yes, SSO via SAML 2.0 is included in the Professional plan and above [Source: Professional Plan Pricing].",
    )
    add(
        "Can I deploy Nextera on Windows?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "Yes, the single-node Docker deployment option runs on Linux, macOS, or Windows with 16 GB+ RAM [Source: Deployment Options].",
    )
    add(
        "What is the minimum RAM for running Nextera?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "The minimum is 8 GB RAM for small models under 7B parameters. For production use, 32 GB RAM is recommended [Source: Deployment Options].",
    )
    add(
        "Does Nextera support citation tracking?",
        [("RAG Pipeline Features", DOC_RAG)],
        "Yes, the RAG pipeline includes citation tracking with source attribution, so you can verify where answers come from [Source: RAG Pipeline Features].",
    )
    add(
        "What sandboxed tools does the agent framework include?",
        [("Agentic Pipeline Features", DOC_AGENTS)],
        "The agent framework includes several built-in tools: SQL query, vector search, web search (optional), code execution (sandboxed), and HTTP API calls. Custom tools can also be added [Source: Agentic Pipeline Features].",
    )
    add(
        "Is there a free trial of Nextera?",
        [("Starter Plan Pricing", DOC_STARTER), ("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "The sources do not mention a free trial. The most affordable option is the Starter plan at €299/month (or €2,990/year with two months free) [Source: Starter Plan Pricing].",
    )
    add(
        "How does Nextera handle document chunking?",
        [("RAG Pipeline Features", DOC_RAG)],
        "Nextera performs automatic document chunking with configurable overlap as part of the RAG pipeline. This splits large documents into smaller pieces optimized for retrieval [Source: RAG Pipeline Features].",
    )
    add(
        "What orchestration options does Nextera support for agents?",
        [("Agentic Pipeline Features", DOC_AGENTS)],
        "The agent framework supports multi-agent orchestration with ReAct-style reasoning loops, intent classification, and execution tracing for debugging [Source: Agentic Pipeline Features].",
    )
    add(
        "Can I use Nextera with Haystack?",
        [("Nextera API & Integrations", DOC_API)],
        "Yes, Nextera has a native integration with Haystack, along with LangChain, LlamaIndex, Semantic Kernel, and AutoGen [Source: Nextera API & Integrations].",
    )
    add(
        "Does the Professional plan include quarterly business reviews?",
        [("Support Tiers and SLAs", DOC_SUPPORT), ("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "No, quarterly business reviews are included only with the Enterprise plan. The Professional plan includes monthly office hours and email support with an 8-hour SLA [Source: Support Tiers and SLAs].",
    )
    add(
        "How quickly are critical security patches delivered?",
        [("Support Tiers and SLAs", DOC_SUPPORT)],
        "Critical security patches are delivered to all plans within 24 hours [Source: Support Tiers and SLAs].",
    )
    add(
        "What is Nextera's approach to data privacy?",
        [("Security and Compliance", DOC_SECURITY), ("FAQ: Local AI vs Cloud AI", DOC_FAQ_CLOUD)],
        "Nextera's approach is local-first: all data processing happens on your hardware with no telemetry, no cloud calls, and no model API usage [Source: Security and Compliance]. Your prompts, documents, and outputs never leave your network — critical for healthcare, legal, finance, and government use cases [Source: FAQ: Local AI vs Cloud AI].",
    )
    add(
        "What Vault integration does Nextera support?",
        [("Security and Compliance", DOC_SECURITY)],
        "Nextera supports secrets management via HashiCorp Vault integration as part of its security features [Source: Security and Compliance].",
    )
    add(
        "Does Nextera offer penetration testing?",
        [("Security and Compliance", DOC_SECURITY)],
        "Nextera undergoes regular third-party penetration testing as part of its security program [Source: Security and Compliance].",
    )
    add(
        "What is the hybrid deployment option?",
        [("Deployment Options", DOC_DEPLOYMENT)],
        "The hybrid deployment option runs inference on a local GPU while using Kubernetes for orchestration, combining local GPU performance with Kubernetes scalability [Source: Deployment Options].",
    )
    add(
        "Can Nextera run models larger than 70B parameters?",
        [("Enterprise Plan Pricing", DOC_ENTERPRISE), ("Professional Plan Pricing", DOC_PROFESSIONAL)],
        "Models larger than 70B parameters are supported only on the Enterprise plan, which handles all model sizes including 405B+ parameters [Source: Enterprise Plan Pricing]. The Professional plan supports up to 70B [Source: Professional Plan Pricing].",
    )
    add(
        "What batch request support does the API have?",
        [("Nextera API & Integrations", DOC_API)],
        "The Nextera API supports batch requests alongside chat completions, embeddings, function calling, and streaming [Source: Nextera API & Integrations].",
    )
    add(
        "How does interaction log collection work for fine-tuning?",
        [("Fine-Tuning Capabilities", DOC_FINETUNING)],
        "The platform automatically collects interaction logs that can be used as training data. A one-click fine-tuning pipeline then uses these logs to improve model accuracy on your domain [Source: Fine-Tuning Capabilities].",
    )
    add(
        "Does Nextera support webhooks?",
        [("Nextera API & Integrations", DOC_API)],
        "Yes, webhooks are available on the Professional plan and above [Source: Nextera API & Integrations].",
    )

    return examples


# ---------------------------------------------------------------------------
# 2. SQL Tool Formatting examples (~50)
# ---------------------------------------------------------------------------

def _generate_sql_examples() -> list[dict]:
    examples = []

    def add(question: str, result: str, answer: str) -> None:
        examples.append({
            "instruction": INSTR_TOOL,
            "input": _tool_input(question, "sql_query", result),
            "output": answer,
        })

    # --- Revenue queries ---
    add("What was the total revenue in Q4 2024?",
        '[[103200]]',
        "The total revenue in Q4 2024 was €103,200.")
    add("What was total revenue in 2024?",
        '[[311500.0]]',
        "The total revenue for 2024 was €311,500.")
    add("What was total revenue in 2023?",
        '[[117200.0]]',
        "The total revenue for 2023 was €117,200.")
    add("Show me quarterly revenue for 2024.",
        '[["Q1", 55100], ["Q2", 68300], ["Q3", 84900], ["Q4", 103200]]',
        "Quarterly revenue for 2024:\n- Q1: €55,100\n- Q2: €68,300\n- Q3: €84,900\n- Q4: €103,200")
    add("Which quarter had the highest revenue in 2024?",
        '[["Q4", 103200]]',
        "Q4 2024 had the highest revenue at €103,200.")
    add("What was the revenue in Q1 2024?",
        '[[55100]]',
        "Revenue in Q1 2024 was €55,100.")
    add("What was the revenue in Q3 2024?",
        '[[84900]]',
        "Revenue in Q3 2024 was €84,900.")
    add("Show me the revenue trend for 2023.",
        '[["Q1", 18500], ["Q2", 24700], ["Q3", 31200], ["Q4", 42800]]',
        "Revenue trend for 2023:\n- Q1: €18,500\n- Q2: €24,700\n- Q3: €31,200\n- Q4: €42,800\nRevenue grew consistently throughout the year.")

    # --- Customer queries ---
    add("How many customers do we have?",
        '[[10]]',
        "You currently have 10 customers in total.")
    add("List all Enterprise customers.",
        '[["Acme Corp", 3500], ["BrightHealth GmbH", 7000], ["FinVault SA", 5000], ["Horizon AI", 4200]]',
        "There are 4 Enterprise customers:\n- Acme Corp (€3,500 MRR)\n- BrightHealth GmbH (€7,000 MRR)\n- FinVault SA (€5,000 MRR)\n- Horizon AI (€4,200 MRR)")
    add("Which customer has the highest MRR?",
        '[["BrightHealth GmbH", 7000.0]]',
        "BrightHealth GmbH has the highest monthly recurring revenue at €7,000.")
    add("List all Professional tier customers.",
        '[["CodeStack Ltd", 999], ["DataFlow AG", 999], ["GreenOps BV", 1499], ["InsureBase Inc", 999]]',
        "There are 4 Professional tier customers:\n- CodeStack Ltd (€999 MRR)\n- DataFlow AG (€999 MRR)\n- GreenOps BV (€1,499 MRR)\n- InsureBase Inc (€999 MRR)")
    add("How many customers are on the Starter plan?",
        '[[2]]',
        "There are 2 customers on the Starter plan.")
    add("Who was the most recent customer to join?",
        '[["JetLog Systems", "2024-06-18"]]',
        "The most recent customer is JetLog Systems, who joined on June 18, 2024.")
    add("List customers who joined in 2024.",
        '[["FinVault SA", "2024-01-08"], ["GreenOps BV", "2024-02-14"], ["Horizon AI", "2024-03-22"], ["InsureBase Inc", "2024-04-01"], ["JetLog Systems", "2024-06-18"]]',
        "Five customers joined in 2024:\n- FinVault SA (January 8)\n- GreenOps BV (February 14)\n- Horizon AI (March 22)\n- InsureBase Inc (April 1)\n- JetLog Systems (June 18)")
    add("What industries are our customers in?",
        '[["Manufacturing", 1], ["Healthcare", 1], ["Software", 1], ["Analytics", 1], ["Education", 1], ["Finance", 1], ["Energy", 1], ["Technology", 1], ["Insurance", 1], ["Logistics", 1]]',
        "Your customers span 10 industries: Manufacturing, Healthcare, Software, Analytics, Education, Finance, Energy, Technology, Insurance, and Logistics — one customer in each.")

    # --- Aggregation queries ---
    add("What is the average MRR per customer?",
        '[[2649.5]]',
        "The average monthly recurring revenue per customer is €2,649.50.")
    add("What is our total MRR?",
        '[[26495.0]]',
        "The total monthly recurring revenue across all customers is €26,495.")
    add("How many new customers joined in Q4 2024?",
        '[[11]]',
        "11 new customers were acquired in Q4 2024.")
    add("What is the total number of new customers in 2024?",
        '[[35]]',
        "A total of 35 new customers were acquired throughout 2024.")
    add("What was the churn rate in Q4 2024?",
        '[[0.7]]',
        "The churn rate in Q4 2024 was 0.7%.")
    add("What is the average churn rate for 2024?",
        '[[0.85]]',
        "The average churn rate across all quarters of 2024 was 0.85%.")
    add("What was the ARR growth percentage in Q3 2024?",
        '[[24.3]]',
        "The annual recurring revenue (ARR) growth rate in Q3 2024 was 24.3%.")

    # --- Product queries ---
    add("What products does Nextera offer?",
        '[["Nextera Starter", "platform", 299.0], ["Nextera Professional", "platform", 999.0], ["Nextera Enterprise", "platform", 3500.0], ["Fine-Tuning Add-on", "addon", 500.0], ["GPU Hours (A100)", "compute", 4.5]]',
        "Nextera offers 5 products:\n- Nextera Starter: €299/month (platform)\n- Nextera Professional: €999/month (platform)\n- Nextera Enterprise: €3,500/month (platform)\n- Fine-Tuning Add-on: €500/month\n- GPU Hours (A100): €4.50/hour")
    add("What is the price of the Fine-Tuning Add-on?",
        '[[500.0, 5000.0]]',
        "The Fine-Tuning Add-on costs €500 per month or €5,000 per year.")
    add("How much do GPU Hours cost?",
        '[[4.5]]',
        "GPU Hours (A100) cost €4.50 per hour.")

    # --- Tier distributions ---
    add("Show me the customer breakdown by tier.",
        '[["enterprise", 4], ["professional", 4], ["starter", 2]]',
        "Customer distribution by tier:\n- Enterprise: 4 customers\n- Professional: 4 customers\n- Starter: 2 customers")
    add("What is the MRR breakdown by tier?",
        '[["enterprise", 19700.0], ["professional", 4496.0], ["starter", 598.0]]',
        "MRR by tier:\n- Enterprise: €19,700 (largest share)\n- Professional: €4,496\n- Starter: €598")

    # --- Year-over-year ---
    add("Compare Q1 revenue between 2023 and 2024.",
        '[[2023, 18500], [2024, 55100]]',
        "Q1 revenue grew significantly year-over-year: €18,500 in 2023 vs €55,100 in 2024 — a 197.8% increase.")
    add("How many total new customers did we get in 2023?",
        '[[18]]',
        "A total of 18 new customers were acquired in 2023.")
    add("What was the churn rate trend in 2024?",
        '[["Q1", 1.0], ["Q2", 0.9], ["Q3", 0.8], ["Q4", 0.7]]',
        "Churn rate improved consistently throughout 2024: 1.0% → 0.9% → 0.8% → 0.7%, showing strong customer retention improvement.")

    # --- Specific customer lookups ---
    add("Tell me about BrightHealth GmbH.",
        '[["BrightHealth GmbH", "enterprise", 7000.0, "2023-03-01", "Healthcare"]]',
        "BrightHealth GmbH is an Enterprise customer in the Healthcare industry. They joined on March 1, 2023, with a monthly recurring revenue of €7,000.")
    add("When did Acme Corp join?",
        '[["Acme Corp", "2023-01-15"]]',
        "Acme Corp joined on January 15, 2023.")
    add("What tier is FinVault SA on?",
        '[["enterprise"]]',
        "FinVault SA is on the Enterprise tier.")

    # --- Empty/edge results ---
    add("Are there any customers in the automotive industry?",
        '[]',
        "No customers were found in the automotive industry.")
    add("Show me revenue for Q1 2025.",
        '[]',
        "No revenue data is available for Q1 2025 — the data only covers through Q4 2024.")

    # --- Revenue ratios ---
    add("What percentage of 2024 revenue came from Q4?",
        '[[33.14]]',
        "Q4 2024 accounted for 33.1% of total 2024 revenue.")
    add("Show the year-over-year revenue growth rate.",
        '[[165.7]]',
        "Revenue grew 165.7% year-over-year from 2023 to 2024.")
    add("What was the average quarterly revenue in 2024?",
        '[[77875.0]]',
        "The average quarterly revenue in 2024 was €77,875.")
    add("What was the revenue in Q2 2024?",
        '[[68300]]',
        "Revenue in Q2 2024 was €68,300.")
    add("What was Q2 2023 revenue?",
        '[[24700]]',
        "Revenue in Q2 2023 was €24,700.")

    # --- Customer details ---
    add("What tier is JetLog Systems on?",
        '[["starter"]]',
        "JetLog Systems is on the Starter tier.")
    add("Tell me about CodeStack Ltd.",
        '[["CodeStack Ltd", "professional", 999.0, "2023-06-01", "Software"]]',
        "CodeStack Ltd is a Professional customer in the Software industry. They joined on June 1, 2023, with an MRR of €999.")
    add("When did DataFlow AG become a customer?",
        '[["DataFlow AG", "2023-09-15"]]',
        "DataFlow AG became a customer on September 15, 2023.")
    add("Which customer generates the least revenue?",
        '[["JetLog Systems", 299.0]]',
        "JetLog Systems generates the least revenue at €299/month — they are on the Starter plan.")
    add("Show me all Starter tier customers and their MRR.",
        '[["EduSpark OÜ", 299], ["JetLog Systems", 299]]',
        "There are 2 Starter tier customers:\n- EduSpark OÜ (€299 MRR)\n- JetLog Systems (€299 MRR)")

    # --- Complex aggregations ---
    add("What is the revenue share by tier?",
        '[["enterprise", 74.3], ["professional", 17.0], ["starter", 2.3]]',
        "Revenue share by tier:\n- Enterprise: 74.3%\n- Professional: 17.0%\n- Starter: 2.3%\nEnterprise customers dominate revenue despite being only 4 of 10 customers.")
    add("How many customers joined each quarter of 2024?",
        '[["Q1", 3], ["Q2", 4], ["Q3", 3], ["Q4", 3]]',
        "New customer acquisitions by quarter in 2024:\n- Q1: 3\n- Q2: 4\n- Q3: 3\n- Q4: 3\nTotal: 13 new customers in 2024.")

    return examples


# ---------------------------------------------------------------------------
# 3. Calculator Tool Formatting examples (~30)
# ---------------------------------------------------------------------------

def _generate_calculator_examples() -> list[dict]:
    examples = []

    def add(question: str, expression: str, result, answer: str) -> None:
        raw = json.dumps({"expression": expression, "result": result})
        examples.append({
            "instruction": INSTR_TOOL,
            "input": _tool_input(question, "calculator", raw),
            "output": answer,
        })

    # --- Percentages ---
    add("What is 15% of 84900?",
        "0.15 * 84900", 12735.0,
        "15% of 84,900 is 12,735.")
    add("What is 23% of 100000?",
        "0.23 * 100000", 23000.0,
        "23% of 100,000 is 23,000.")
    add("What is 10% of 55100?",
        "0.10 * 55100", 5510.0,
        "10% of 55,100 is 5,510.")
    add("What is 20% of 311500?",
        "0.20 * 311500", 62300.0,
        "20% of 311,500 is 62,300.")

    # --- Annual costs ---
    add("If I have 30 users on the Professional plan, what is my annual spend?",
        "999 * 12", 11988,
        "With 30 users on the Professional plan, the annual cost is €11,988 (€999/month × 12 months). Note: the Professional plan supports up to 25 concurrent users.")
    add("What is the annual cost of the Enterprise plan?",
        "3500 * 12", 42000,
        "The annual cost of the Enterprise plan is €42,000 (€3,500/month × 12 months).")
    add("How much would 50 users on the Starter plan cost per year?",
        "299 * 12", 3588,
        "The Starter plan costs €3,588 per year (€299/month × 12). Note: the Starter plan supports up to 5 concurrent users, so 50 users would need the Enterprise plan.")
    add("What is the monthly cost of 10 A100 GPU hours per day?",
        "4.5 * 10 * 30", 1350.0,
        "At 10 GPU hours per day for 30 days, the monthly cost would be €1,350.")

    # --- Growth projections ---
    add("If Q4 revenue was 103200, what would 15% growth look like?",
        "103200 * 1.15", 118680.0,
        "With 15% growth, Q4 revenue of €103,200 would increase to €118,680.")
    add("What would revenue be if it doubled from Q1 2024?",
        "55100 * 2", 110200,
        "If Q1 2024 revenue doubled, it would be €110,200.")
    add("What is 25% growth on 84900?",
        "84900 * 1.25", 106125.0,
        "With 25% growth, €84,900 would become €106,125.")

    # --- Unit economics ---
    add("What is 50 * 999 * 12?",
        "50 * 999 * 12", 599400,
        "The result of 50 × 999 × 12 is 599,400. This represents the annual revenue from 50 Professional plan customers.")
    add("What is 4 * 3500 * 12?",
        "4 * 3500 * 12", 168000,
        "The result is 168,000. This represents the annual revenue from 4 Enterprise customers (€3,500/month each).")
    add("Calculate the total ARR from 10 customers at 2649.50 MRR each.",
        "10 * 2649.50 * 12", 317940.0,
        "The total annual recurring revenue from 10 customers at €2,649.50 average MRR is €317,940.")

    # --- Comparisons ---
    add("What is the difference between Enterprise and Starter monthly cost?",
        "3500 - 299", 3201,
        "The monthly cost difference between Enterprise (€3,500) and Starter (€299) is €3,201.")
    add("How much more expensive is Professional than Starter per year?",
        "9990 - 2990", 7000,
        "The Professional plan costs €7,000 more per year than the Starter plan (€9,990 vs €2,990).")
    add("What is the total MRR of all Enterprise customers?",
        "3500 + 7000 + 5000 + 4200", 19700,
        "The total MRR of all Enterprise customers is €19,700.")

    # --- Savings calculation ---
    add("How much would a team of 50 save per month using local AI vs cloud?",
        "5000 - 0", 5000,
        "A team of 50 would save approximately €5,000 per month by running AI locally instead of using cloud APIs.")
    add("What is the annual savings from local AI for a 50-person team?",
        "5000 * 12", 60000,
        "The annual savings from local AI vs cloud APIs for a 50-person team would be approximately €60,000.")

    # --- Misc arithmetic ---
    add("What is the square root of 144?",
        "sqrt(144)", 12.0,
        "The square root of 144 is 12.")
    add("What is 2 to the power of 10?",
        "2 ** 10", 1024,
        "2 to the power of 10 is 1,024.")

    return examples


# ---------------------------------------------------------------------------
# 4. Multi-Step Synthesis examples (~30)
# ---------------------------------------------------------------------------

def _generate_multi_step_examples() -> list[dict]:
    examples = []

    def add(question: str, step1_tool: str, step1_result: str,
            step2_tool: str, step2_result: str, answer: str) -> None:
        results_str = (
            f"Step 1 ({step1_tool}): {step1_result}\n\n"
            f"Step 2 ({step2_tool}): {step2_result}"
        )
        examples.append({
            "instruction": INSTR_MULTI,
            "input": _multi_input(question, results_str),
            "output": answer,
        })

    # --- Lookup + growth ---
    add(
        "What was Q4 2024 revenue, and what would 15% growth look like?",
        "sql_query", json.dumps({"result": [[103200]]}),
        "calculator", json.dumps({"expression": "103200 * 1.15", "result": 118680.0}),
        "Q4 2024 revenue was €103,200. With 15% growth, it would increase to €118,680.",
    )
    add(
        "What was Q3 2024 revenue, and what would a 20% increase be?",
        "sql_query", json.dumps({"result": [[84900]]}),
        "calculator", json.dumps({"expression": "84900 * 1.20", "result": 101880.0}),
        "Q3 2024 revenue was €84,900. A 20% increase would bring it to €101,880.",
    )
    add(
        "What was Q1 2024 revenue, and what would it be if it doubled?",
        "sql_query", json.dumps({"result": [[55100]]}),
        "calculator", json.dumps({"expression": "55100 * 2", "result": 110200}),
        "Q1 2024 revenue was €55,100. If doubled, it would be €110,200.",
    )

    # --- Count + multiply ---
    add(
        "How many Enterprise customers do we have, and what's their combined annual revenue?",
        "sql_query", json.dumps({"result": [[4]]}),
        "calculator", json.dumps({"expression": "4 * 3500 * 12", "result": 168000}),
        "You have 4 Enterprise customers. At €3,500/month each, their combined annual revenue is €168,000.",
    )
    add(
        "How many customers do we have, and what's the total ARR at average MRR?",
        "sql_query", json.dumps({"result": [[10, 2649.5]]}),
        "calculator", json.dumps({"expression": "10 * 2649.5 * 12", "result": 317940.0}),
        "You have 10 customers with an average MRR of €2,649.50. The total annual recurring revenue is €317,940.",
    )
    add(
        "How many Professional customers do we have, and what's their total annual spend?",
        "sql_query", json.dumps({"result": [[4]]}),
        "calculator", json.dumps({"expression": "4 * 999 * 12", "result": 47952}),
        "There are 4 Professional customers. Their combined annual spend is €47,952 (€999/month × 4 × 12).",
    )

    # --- Revenue + percentage ---
    add(
        "What was total 2024 revenue, and what would a 10% discount be?",
        "sql_query", json.dumps({"result": [[311500.0]]}),
        "calculator", json.dumps({"expression": "311500 * 0.10", "result": 31150.0}),
        "Total 2024 revenue was €311,500. A 10% discount would amount to €31,150.",
    )
    add(
        "What was Q4 2024 revenue, and what percentage of total 2024 revenue is that?",
        "sql_query", json.dumps({"result": [[103200, 311500.0]]}),
        "calculator", json.dumps({"expression": "(103200 / 311500) * 100", "result": 33.14}),
        "Q4 2024 revenue was €103,200, which represents 33.1% of the total 2024 revenue of €311,500.",
    )

    # --- Customer + cost ---
    add(
        "What is BrightHealth's MRR, and what would their annual cost be?",
        "sql_query", json.dumps({"result": [["BrightHealth GmbH", 7000.0]]}),
        "calculator", json.dumps({"expression": "7000 * 12", "result": 84000}),
        "BrightHealth GmbH has an MRR of €7,000. Their annual cost would be €84,000.",
    )
    add(
        "What is the highest MRR customer, and what's their annual value?",
        "sql_query", json.dumps({"result": [["BrightHealth GmbH", 7000.0]]}),
        "calculator", json.dumps({"expression": "7000 * 12", "result": 84000}),
        "BrightHealth GmbH has the highest MRR at €7,000/month. Their annual value is €84,000.",
    )

    # --- Comparison + calculation ---
    add(
        "What's the revenue difference between Q4 and Q1 2024, and what growth rate is that?",
        "sql_query", json.dumps({"result": [[103200, 55100]]}),
        "calculator", json.dumps({"expression": "((103200 - 55100) / 55100) * 100", "result": 87.3}),
        "Q4 2024 revenue was €103,200 and Q1 was €55,100 — a difference of €48,100. That's an 87.3% growth rate from Q1 to Q4.",
    )
    add(
        "What were the highest and lowest quarterly revenues in 2024, and what's the gap?",
        "sql_query", json.dumps({"result": [["Q4", 103200], ["Q1", 55100]]}),
        "calculator", json.dumps({"expression": "103200 - 55100", "result": 48100}),
        "The highest quarter was Q4 at €103,200 and the lowest was Q1 at €55,100. The gap between them is €48,100.",
    )

    # --- Churn + impact ---
    add(
        "What is the current churn rate, and how many customers might we lose from 10 at that rate?",
        "sql_query", json.dumps({"result": [[0.7]]}),
        "calculator", json.dumps({"expression": "10 * 0.007", "result": 0.07}),
        "The current churn rate is 0.7%. With 10 customers, that means statistically 0.07 customers per quarter — essentially negligible monthly churn.",
    )

    # --- New customer + revenue ---
    add(
        "How many new customers joined in Q4 2024, and what revenue did they contribute?",
        "sql_query", json.dumps({"result": [[11, 103200]]}),
        "calculator", json.dumps({"expression": "103200 / 11", "result": 9381.82}),
        "11 new customers joined in Q4 2024, with total revenue of €103,200. That's approximately €9,382 per new customer.",
    )

    # --- Plan comparison + savings ---
    add(
        "What's the monthly price difference between Enterprise and Professional, and what's the annual gap?",
        "sql_query", json.dumps({"result": [[3500.0, 999.0]]}),
        "calculator", json.dumps({"expression": "(3500 - 999) * 12", "result": 30012}),
        "The Enterprise plan costs €3,500/month and Professional costs €999/month — a €2,501 monthly difference. Annually, that's a €30,012 gap.",
    )

    # --- Trend + projection ---
    add(
        "What was the ARR growth rate in Q4 2024, and what would next quarter's revenue be at that rate?",
        "sql_query", json.dumps({"result": [[21.6, 103200]]}),
        "calculator", json.dumps({"expression": "103200 * 1.216", "result": 125491.2}),
        "The ARR growth rate in Q4 2024 was 21.6%. If that rate continues, next quarter's revenue would be approximately €125,491.",
    )

    # --- Total + per-unit ---
    add(
        "What is total MRR and the average per customer?",
        "sql_query", json.dumps({"result": [[26495.0, 10]]}),
        "calculator", json.dumps({"expression": "26495.0 / 10", "result": 2649.5}),
        "Total MRR is €26,495 across 10 customers. The average MRR per customer is €2,649.50.",
    )

    add(
        "What's the total 2023 revenue, and how much less is it than 2024?",
        "sql_query", json.dumps({"result": [[117200.0, 311500.0]]}),
        "calculator", json.dumps({"expression": "311500 - 117200", "result": 194300.0}),
        "Total 2023 revenue was €117,200 and 2024 was €311,500. Revenue increased by €194,300 year-over-year — a significant growth trajectory.",
    )

    # --- Starter revenue projection ---
    add(
        "How many Starter customers do we have, and what's their combined annual revenue?",
        "sql_query", json.dumps({"result": [[2]]}),
        "calculator", json.dumps({"expression": "2 * 299 * 12", "result": 7176}),
        "There are 2 Starter customers. Their combined annual revenue is €7,176 (€299/month × 2 × 12).",
    )

    # --- Upgrade opportunity ---
    add(
        "What is the average MRR per Enterprise customer, and how does it compare to Professional?",
        "sql_query", json.dumps({"result": [[4925.0, 1124.0]]}),
        "calculator", json.dumps({"expression": "4925.0 / 1124.0", "result": 4.38}),
        "Enterprise customers average €4,925 MRR while Professional customers average €1,124 MRR. Enterprise customers generate 4.4x more revenue per customer.",
    )

    # --- Revenue per customer trend ---
    add(
        "What was the average revenue per new customer in Q1 2024 vs Q4 2024?",
        "sql_query", json.dumps({"result": [[55100, 8, 103200, 11]]}),
        "calculator", json.dumps({"expression": "(103200/11) - (55100/8)", "result": 2494.32}),
        "Average revenue per new customer: Q1 2024 was €6,888 (€55,100 / 8 customers) and Q4 2024 was €9,382 (€103,200 / 11 customers). That's a €2,494 increase per new customer.",
    )

    # --- Churn cost ---
    add(
        "What is the current churn rate and what revenue would we lose if we churned one Enterprise customer?",
        "sql_query", json.dumps({"result": [[0.7, 4925.0]]}),
        "calculator", json.dumps({"expression": "4925.0 * 12", "result": 59100.0}),
        "The churn rate is 0.7%. Losing one average Enterprise customer (€4,925/month MRR) would cost €59,100 in annual revenue.",
    )

    # --- Growth needed ---
    add(
        "What is our total ARR and how much more do we need to reach €500K ARR?",
        "sql_query", json.dumps({"result": [[26495.0]]}),
        "calculator", json.dumps({"expression": "500000 - (26495 * 12)", "result": 182060.0}),
        "Current total MRR is €26,495 which equals €317,940 ARR. You need €182,060 more to reach €500K ARR.",
    )

    return examples


# ---------------------------------------------------------------------------
# 5. Direct Answer examples (~30)
# ---------------------------------------------------------------------------

def _generate_direct_examples() -> list[dict]:
    examples = []

    def add(query: str, answer: str) -> None:
        examples.append({
            "instruction": INSTR_DIRECT,
            "input": query,
            "output": answer,
        })

    # --- Greetings ---
    add("Hello!", "Hello! I'm your Nextera AI assistant. I can help with product questions, search your knowledge base, run SQL queries, and perform calculations. How can I help?")
    add("Hi there!", "Hi! How can I assist you today?")
    add("Good morning!", "Good morning! What can I help you with?")
    add("Good afternoon!", "Good afternoon! Ready to help with your questions.")
    add("Hey", "Hey! What would you like to know?")

    # --- Farewells ---
    add("Thanks for the help!", "You're welcome! Let me know if you need anything else.")
    add("Goodbye!", "Goodbye! Have a great day.")
    add("Thank you!", "Happy to help! Feel free to ask anytime.")
    add("That's all I need.", "Great! Don't hesitate to come back if you have more questions.")

    # --- Capability questions ---
    add("What can you do?", "I can help you with:\n- Searching the Nextera knowledge base for product and feature information\n- Running SQL queries against your business database\n- Performing calculations\n- Answering general questions about Nextera Platform")
    add("What tools do you have?", "I have access to three tools:\n1. **Vector search** — semantic search across the Nextera knowledge base\n2. **SQL query** — read-only queries against the business database\n3. **Calculator** — precise arithmetic calculations")
    add("How fast are you?", "On a GPU, each model call takes 10-200ms. A full query including classification, search, and synthesis completes in under 1 second.")
    add("Are you an AI?", "Yes, I'm an AI assistant powered by fine-tuned Gemma models running locally on your hardware. All processing happens on-device — no data is sent to external servers.")
    add("How do you work?", "I use a pipeline of specialized models: one for understanding your intent, one for selecting tools, one for document retrieval, and one for generating responses. Each model is optimized for its specific task.")

    # --- Domain knowledge ---
    add("What is RAG?", "RAG stands for Retrieval-Augmented Generation. It combines a knowledge base search with language model generation — the model retrieves relevant documents first, then uses them to generate an accurate, grounded answer.")
    add("What is LoRA?", "LoRA (Low-Rank Adaptation) is a fine-tuning technique that trains only small adapter matrices instead of all model weights. This makes fine-tuning faster and requires less memory while preserving the model's original capabilities.")
    add("What is a vector database?", "A vector database stores document embeddings — numerical representations of text meaning. When you search, your query is also converted to a vector, and the database finds the most semantically similar documents using cosine similarity.")
    add("What is function calling?", "Function calling is a capability where the AI model selects and invokes specific tools (like SQL queries or calculators) based on the user's request, then formats the result into a natural language answer.")

    # --- Edge cases ---
    add("Tell me a joke.", "I'm better at answering questions about Nextera Platform! Try asking about pricing, features, or your business data.")
    add("What's the weather?", "I don't have access to weather information. I can help with Nextera product questions, database queries, and calculations.")
    add("I don't understand.", "No problem! Feel free to rephrase your question. I can help with:\n- Product and feature questions\n- Business data queries\n- Calculations and comparisons")
    add("Can you help me with something else?", "Of course! I'm here to help. You can ask about Nextera features, pricing, run database queries, or perform calculations. What do you need?")
    add("Never mind.", "No problem! Let me know if you need anything later.")
    add("", "It looks like you sent an empty message. How can I help you?")

    # --- Meta questions ---
    add("Who built you?", "I was built using fine-tuned Google Gemma models as part of the Nextera Platform local AI stack. I run entirely on local hardware with no cloud dependencies.")
    add("What model are you?", "I'm powered by a fine-tuned Gemma 3 1B model for intent classification, a Gemma 3 4B model for response synthesis, a Qwen 3.5 4B model for tool routing, and an EmbeddingGemma 300M model for document retrieval. All models run locally.")
    add("Do you send my data anywhere?", "No. All processing happens locally on your hardware. Your queries, documents, and responses never leave your network.")
    add("Can you learn from our conversations?", "Yes, your interactions are logged locally and can be used to fine-tune the models for improved accuracy on your specific domain. The data never leaves your infrastructure.")

    # --- Clarifying questions ---
    add("What database do you query?", "I query a local SQLite database that contains your business data — products, customers, and sales records. All queries are read-only.")
    add("How accurate are your answers?", "My accuracy depends on the data available. For knowledge base questions, I cite sources directly. For SQL queries, results come straight from the database. I'll let you know when information isn't available.")

    return examples


# ---------------------------------------------------------------------------
# Quality controls
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dedup_key(ex: dict) -> str:
    """Extract the discriminating text for dedup comparison.

    For RAG examples the full input is mostly shared template + source passages,
    so comparing full input causes false-positive dedup.  We compare the output
    (the model's answer) which is the truly distinctive part.  Two examples that
    produce near-identical outputs are genuinely redundant for training.
    """
    return ex.get("output", "")


def _deduplicate(examples: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate examples based on output Jaccard similarity."""
    kept: list[dict] = []
    for ex in examples:
        text = _dedup_key(ex)
        if any(_jaccard(text, _dedup_key(k)) >= threshold for k in kept):
            continue
        kept.append(ex)
    return kept


def _validate_format(examples: list[dict]) -> None:
    """Assert all examples have the required Alpaca format fields."""
    valid_instructions = {INSTR_RAG, INSTR_TOOL, INSTR_MULTI, INSTR_DIRECT}
    for i, ex in enumerate(examples):
        assert "instruction" in ex, f"Example {i} missing 'instruction'"
        assert "input" in ex, f"Example {i} missing 'input'"
        assert "output" in ex, f"Example {i} missing 'output'"
        assert ex["instruction"] in valid_instructions, (
            f"Example {i} has unknown instruction: {ex['instruction'][:50]}..."
        )
        assert ex["output"].strip(), f"Example {i} has empty output"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthesis training data for gemma3"
    )
    parser.add_argument(
        "--output", default="./data/training-data/gemma3_synthesis.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate all categories
    rag = _generate_rag_examples()
    sql = _generate_sql_examples()
    calc = _generate_calculator_examples()
    multi = _generate_multi_step_examples()
    direct = _generate_direct_examples()

    all_examples = rag + sql + calc + multi + direct
    print(f"Raw examples: {len(all_examples)} "
          f"(rag={len(rag)}, sql={len(sql)}, calc={len(calc)}, "
          f"multi={len(multi)}, direct={len(direct)})")

    # Deduplicate
    deduped = _deduplicate(all_examples)
    removed = len(all_examples) - len(deduped)
    if removed:
        print(f"Removed {removed} near-duplicates (Jaccard >= 0.85)")

    # Validate format
    _validate_format(deduped)

    # Assert minimum count
    assert len(deduped) >= 200, (
        f"Only {len(deduped)} examples after dedup — need >= 200"
    )

    # Shuffle
    random.shuffle(deduped)

    # Write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(deduped)} examples → {args.output}")

    if args.stats:
        from collections import Counter
        instr_counts = Counter()
        for ex in deduped:
            instr = ex["instruction"]
            if instr == INSTR_RAG:
                instr_counts["RAG synthesis"] += 1
            elif instr == INSTR_TOOL:
                instr_counts["Tool formatting"] += 1
            elif instr == INSTR_MULTI:
                instr_counts["Multi-step synthesis"] += 1
            elif instr == INSTR_DIRECT:
                instr_counts["Direct answer"] += 1
        print("\nBy category:")
        for cat, count in instr_counts.most_common():
            print(f"  {cat}: {count}")

        # Output length stats
        lengths = [len(ex["output"]) for ex in deduped]
        print(f"\nOutput length: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.0f} chars")


if __name__ == "__main__":
    main()
