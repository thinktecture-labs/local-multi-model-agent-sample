# Solution Overview — Multi-Model Local AI Agent

> ⚠️ **Conference talk demo — not production code.** Accompanies a conference
> keynote on local on-device AI. Reference for architectural patterns, not a
> production-ready system. See [`SECURITY.md`](SECURITY.md) for the threat model.

---

> A privacy-preserving agentic AI system built on a stack of specialized small language models. All inference runs locally — zero data leaves the machine unless the user explicitly approves cloud escalation.

## The Thesis

Task decomposition beats monolithic models on domain tasks. Instead of one large model doing everything, five small models (1B–4B parameters) each handle what they're best at. The result: faster, cheaper, more accurate, and fully private.

## Use Cases

### 1. Knowledge Base Q&A

Users ask questions about the company's products, pricing, integrations, compliance, and support. The agent retrieves relevant documents from a curated knowledge base via semantic search and synthesizes a natural-language answer.

*"What features are included in the Enterprise plan?"* → vector search finds the pricing doc → 4B model synthesizes a complete answer with bullet points.

### 2. Structured Data Queries

Users ask questions about sales figures, customer lists, revenue trends, and other tabular data. The agent generates SQL queries, executes them against a local database, and formats the results.

*"What were the total sales in Q3 2024?"* → Qwen generates `SELECT revenue FROM sales WHERE quarter='Q3' AND year=2024` → executes → formats "EUR 84,900".

### 3. Calculations

Users ask math questions — percentages, discounts, compound growth. The agent evaluates expressions directly without hallucination risk.

*"What's 15% of $45,000?"* → Qwen selects calculator tool → safe eval → "$6,750".

### 4. Image Understanding

Users attach images (screenshots, diagrams, charts) and ask questions about them. The vision model analyzes the image and responds.

*"Explain what this system diagram shows"* → 4B vision model describes the architecture.

### 5. Document Upload and Chat

Users upload PDF, TXT, or MD files. The system extracts text (with OCR for scanned pages), chunks it, embeds it, and indexes it. The user can then chat with the uploaded document — all queries are scoped to that document's content.

*Upload the first 50 pages of Snowflake's FY2025 annual report → "What is Snowflake's net revenue retention rate?"* → answer (126%) extracted from the uploaded document.

### 6. Structured Data Extraction

After uploading a financial document (earnings report, annual report), the user can extract structured metrics — revenue, growth rate, NRR, customer counts — into a SQL table. This enables cross-source queries that join internal data with extracted competitor data.

*Upload Snowflake annual report → click "Extract structured data" → 9 fields stored → "How does our revenue growth compare to Snowflake?"* → SQL JOIN across internal sales and extracted competitor data.

### 7. Voice Interaction

Users speak to the agent. Speech is transcribed locally (whisper), processed through the agent pipeline, and the response is spoken back (piper TTS). Full voice-to-voice loop, entirely local.

### 8. Competitive Benchmarking

Combines document upload, OCR, structured extraction, and cross-source queries into a complete workflow: upload a competitor's report, extract their metrics, compare against internal data. All without any data leaving the machine.

### 9. Hybrid Cloud Escalation

When the local agent's confidence is low (entity not in knowledge base, ambiguous query), the system can escalate to a cloud model — but only with explicit user approval. A confidence score and "Escalate?" button appear. The user decides whether data leaves the machine.

### 10. Model Comparison

Side-by-side comparison of different backends on the same query: multi-model pipeline vs single Qwen model vs cloud (GPT-5.4). Shows latency, token count, cost, and response quality differences in real time.

### 11. Fine-Tuning Flywheel

Every interaction is logged. Logged interactions can be exported as training data, used to fine-tune the models, and deployed — creating a continuous improvement loop: Use → Log → Train → Deploy.

---

## Architecture

### Five Specialized Models + a deterministic Router

| Role | Model | Size | Responsibility |
|------|-------|------|----------------|
| **Router** | LogReg on FT-embeddinggemma vectors | sklearn .joblib (~150 KB) | **Primary** intent classifier (~10 ms, deterministic, ~93% of traffic) |
| **Thinker** | gemma3-ft | 1B | Direct answers, query decomposition, tool-result synthesis, intent-classification fallback when LogReg model is absent |
| **Doer** | Qwen3.5-4B-toolcalling-ft (v8) | 4B | Tool selection, argument extraction, SQL generation (99.4% routing) |
| **Librarian** | embeddinggemma-ft | 308M | Semantic search, document retrieval |
| **Eye** | gemma3-4B-ft | 4B | RAG synthesis from retrieved context; image understanding via mmproj (base channel) |
| **Reader** | GLM-OCR | 0.9B | PDF text + table extraction (upload-time only) |

Supporting models (not part of the agent pipeline):
- **whisper.cpp** — speech-to-text transcription
- **piper TTS** — text-to-speech synthesis

### Query Pipeline

```
User Query
    │
    ├── Image attached? ──→ Vision model ──→ Response
    │
    ├── Voice input? ──→ Whisper STT ──→ Text query
    │
    ▼
Intent Classification (LogReg on embeddings, <5ms)
    │
    ├── rag_query ──→ Query rewrite ──→ Vector search ──→ 4B Synthesis ──→ Response
    │
    ├── tool_use ──→ Qwen tool selection ──→ Execute (SQL/Calculator) ──→ Format ──→ Response
    │
    └── direct_answer ──→ 1B generates response ──→ Response
```

### Document Pipeline

```
PDF Upload
    │
    ├── Text extraction (pypdf)
    │
    ├── Smart OCR (only pages where pypdf fails)
    │       └── GLM-OCR per page (150 DPI, fallback 100 DPI)
    │
    ├── Chunking (semantic via chonkie when embeddinggemma available, fixed-size 800 chars fallback)
    │
    ├── Embedding (embeddinggemma, batches of 10)
    │
    ├── Indexing (ChromaDB uploads collection)
    │
    └── [Optional] Structured extraction (4B model → JSON → competitors table)
```

### Multi-Scenario Architecture

The system supports multiple domain scenarios via JSON-driven configuration. No code changes — switching scenario changes the data, prompts, SQL schema, and fine-tuned GGUFs.

```bash
bash scripts/start_servers.sh --scenario nextera    # English SaaS analytics (shipped)
```

Each scenario defines: RAG documents, SQL tables, system prompts (including the decomposer fewshot and concretize examples in the active language), tool descriptions, adversarial refusal messages, and per-model GGUF paths. Configuration lives in `scenarios/<name>.json`. The four LoRA-fine-tuned models (Gemma3-1B, Gemma3-4B, Qwen3.5-4B-toolcalling, EmbeddingGemma) plus the LogReg intent classifier are all trained independently per scenario.

| Scenario | Language | Domain | RAG Docs | SQL Tables | Models |
| --- | --- | --- | --- | --- | --- |
| **nextera** | English | SaaS platform analytics | 13 | 4 (products, customers, sales, competitors) | 4 FT GGUFs + LogReg .joblib |

The shipped Nextera scenario is the reference implementation. For adding new
scenarios (legal, healthcare, manufacturing, …), see `docs/guides/SCENARIO_PLAYBOOK.md`.

### Prompt Architecture

Prompts are organized **per task**, not per model. Each pipeline step has one prompt template, shared across all models that execute it. The same `tool_format_prompt` is used whether gemma3-1B or Qwen3.5-35B formats the response.

| Prompt | Used by | Purpose |
|--------|---------|---------|
| `direct_answer_system` | gemma3-1B | System prompt for direct answers |
| `rag_synthesis_system` | gemma3-4B | System prompt for RAG synthesis from retrieved context |
| `rag_rewrite_prompt` | gemma3-1B | Rewrite user query into dense keyword phrase for retrieval |
| `tool_format_prompt` | gemma3-1B | Format raw tool results into human-readable answers |
| `multi_step_synthesis_prompt` | Qwen3.5-4B FT v9 (FUNCTION role) | Combine multiple tool results into a single answer — routed off gemma3-1B in commit `118b6a1` after gemma3-1B was found to plagiarise fewshot example numbers |
| `extraction_system` | gemma3-4B | Extract structured data from uploaded documents |
| `adversarial_refusal` | (none — canned) | Static refusal message for adversarial inputs |

All prompts live in `scenarios/<name>.json` under the `prompts` key. Switching scenario switches all prompts — including the operating language. No model-specific or mode-specific (base vs fine-tuned) prompt variants exist; fine-tuning compensates for model capability differences.

The **cloud orchestrator** uses the scenario's `rag_synthesis_system` prompt and `sql_tool_description` to build its system prompt, keeping it domain-aware without hardcoded strings.

Tool schemas (function-calling descriptions sent to Qwen) are also scenario-driven: `sql.tool_description` and `sql.parameter_description` describe the domain-specific tables and columns.

### Storage

| Store | Technology | Content |
|-------|-----------|---------|
| Knowledge base | ChromaDB (cosine) | Curated domain documents (scenario-specific) |
| Uploads | ChromaDB (cosine, separate collection) | User-uploaded document chunks |
| Business data | SQLite | Scenario-specific tables (scenario-specific path) |
| Interactions | JSON file | Logged query/response pairs for fine-tuning |

### Inference

All models run via **llama.cpp** (llama-server) with the OpenAI-compatible API. Each model gets its own server process on a dedicated port. Models are served as GGUF files (quantized for efficiency).

### Platforms

| Machine | GPU | Memory | Agent p50 | OCR/page | Doc-chat p50 |
|---------|-----|--------|-----------|----------|--------------|
| MacBook Pro M5 Max | 40-core Metal | 128 GB unified | 1121ms | 2715ms | 378ms |
| NVIDIA RTX PRO 6000 | Blackwell CUDA | 96 GB VRAM | 465ms | 1008ms | 507ms |
| NVIDIA DGX Spark | GB10 CUDA | 128 GB unified | 2315ms | 4274ms | 2115ms |
| Minisforum MS-S1 MAX | Strix Halo RDNA 3.5 | 128 GB unified | ~2400ms | 8498ms | 1822ms |

---

## Features

### Agent Capabilities

- **Three-way intent classification** — LogReg on embeddings (<5ms, 99.4% accuracy post-2026-05-15 retrain) with generative gemma3-ft fallback at 96.7%
- **RAG with dual-query search** — searches with both original and rewritten query, deduplicates, keeps best results
- **Native function calling** — Qwen3.5-4B fine-tuned on 1,372 domain examples (99.4% tool routing)
- **Multi-step tool chains** — complex queries decomposed into sequential tool calls; **97.5% chain accuracy with 100% SQL execution validity** (v9 post-2026-05-15 retrain — matches v8's chain-shape number AND fixes the v8-era `FROM revenue` training-data bug). Deterministic — verified byte-identical across 3 back-to-back runs.
- **Adversarial robustness** — 5-layer pre-classifier defense stack: 30 regex injection patterns + gibberish detector + non-ASCII filter + LogReg confidence threshold (0.60) + canned refusal. Matched queries route directly to `direct_answer` without inference. Pipeline robustness 93.3% on the 60-query adversarial set (vs 43.3% generative-only baseline).

### Document Processing

- **Smart hybrid OCR** — pypdf first (fast), GLM-OCR only for pages where pypdf fails
- **Separate upload collection** — uploaded documents don't pollute the curated knowledge base
- **Document chat mode** — queries scoped to a specific uploaded document via document_id
- **Re-upload deduplication** — uploading the same file replaces previous chunks
- **Structured extraction** — 4B model extracts financial metrics into SQL table (100% field accuracy on eval)

### Observatory UI

- **Real-time pipeline trace** — every step visible: intent, tool selection, SQL query, retrieved documents, synthesis
- **Upload progress widget** — per-stage progress with live timer, per-page OCR status
- **Extract Data button** — one-click structured extraction with inline JSON result display
- **Collapsible model status bar** — 10 model pills (LogReg router, gemma3-1B, Qwen3.5-4B FT, embeddinggemma, gemma3-4B vision, GLM-OCR, whisper, piper, Qwen-35B comparison, cloud) with health dots and role tooltips
- **Suggestion chips** — collapsible sample queries, loaded from the active scenario's JSON
- **Scenario-driven branding** — brand label, logo, favicon, title, and suggestion chips all driven by `scenarios/<name>.json`
- **GPU/energy panel** — VRAM usage, utilization, temperature, power draw, CO2 estimates
- **Cost comparison** — local ($0.00) vs estimated cloud cost per query
- **Privacy badge** — tracks bytes sent externally (0 in local-only mode)
- **Backend selector** — switch between Multi-Models, Qwen (single), Cloud, or All (side-by-side)
- **Mode toggles** — airplane mode (offline), local-only/hybrid routing, base/fine-tuned model swap
- **Voice input** — push-to-talk microphone with streaming STT → agent → TTS response
- **Wake word detection** — OpenWakeWord browser-side keyword spotting ("Hey Jarvis") via in-browser ONNX inference, auto-starts recording on detection. MIT licensed, no API key required.
- **Show Mode** — cinematic full-screen keynote demo with animated orb, 5-model activity strip, smart cards (KPI, bar chart, ranked bars, table), dark/light theme toggle
- **Dark/light theme**

### Fine-Tuning & Evaluation

- **LoRA/QLoRA training** — gemma3 (intent), Qwen3.5 (tool calling), embeddinggemma (retrieval)
- **Zero-downtime model swap** — dual-port architecture, toggle base ↔ fine-tuned instantly
- **7 eval suites** — intent (180q), tool routing (160q), retrieval MRR (25q), multi-step (80q), adversarial (60q), OCR (29q), extraction (29 fields)
- **Eval-train leakage checks** — automated verification that eval and training sets don't overlap
- **A/B comparison dashboard** — before/after eval results side-by-side

### Privacy & Sovereignty

- **100% local by default** — no API keys needed, no cloud calls, no telemetry
- **Explicit cloud escalation** — user must approve before any data leaves the machine
- **Byte-level tracking** — UI shows exactly how many bytes were sent externally
- **Offline mode** — airplane mode kills all external connectivity, agent keeps working
- **On-device training** — fine-tuning data never leaves the machine

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Process a user query (with optional document_id for document chat) |
| `POST` | `/query/stream` | Streaming query via SSE |
| `POST` | `/query/compare-all` | Three-path comparison (multi-models, qwen, cloud) |
| `POST` | `/compare` | Two-path comparison (local vs cloud) |
| `POST` | `/escalate` | Cloud escalation with confidence check |
| `POST` | `/escalate/stream` | Streaming cloud escalation via SSE |
| `GET` | `/health` | Model availability, document count |
| `GET` | `/tools` | List registered tools with schemas |
| `GET` | `/scenario` | Active scenario metadata, suggestions, logo, branding |
| `POST` | `/upload-document` | Upload & index a file with SSE progress |
| `POST` | `/extract` | Extract structured data from uploaded document |
| `GET` | `/competitors` | List extracted competitor data |
| `DELETE` | `/uploads` | Clear uploaded document chunks |
| `GET` | `/uploads/status` | Check upload processing status |
| `POST` | `/documents` | Add a document to the knowledge base |
| `POST` | `/models/swap` | Zero-downtime base ↔ fine-tuned swap |
| `GET` | `/models/mode` | Current model mode (base/finetuned) |
| `POST` | `/network-mode` | Toggle online/offline |
| `POST` | `/routing-mode` | Toggle local-only/hybrid |
| `POST` | `/voice/chat` | Voice-to-voice interaction |
| `POST` | `/voice/synthesize` | Text-to-speech synthesis |
| `GET` | `/voice/audio/{id}` | Retrieve cached TTS audio |
| `POST` | `/export-training-data` | Export interaction logs for fine-tuning |
| `POST` | `/train` | Trigger fine-tuning (SSE progress) |
| `GET` | `/train/status` | Training progress |
| `POST` | `/eval` | Run evaluation suite |
| `GET` | `/eval/results` | Stored eval results (before/after) |
| `POST` | `/eval/reset` | Clear eval results |
| `GET` | `/gpu` | GPU stats (VRAM, utilization, temperature, power) |
| `GET` | `/energy` | Energy consumption and CO2 estimates |
| `GET` | `/privacy` | Privacy stats (queries, tokens, external bytes) |
| `WS` | `/ws/stats` | WebSocket for live GPU/energy stats |

---

## Eval Results Summary

| Eval | Queries | Score | Baseline |
|------|---------|----------|----------|
| Intent classification (LogReg primary path) | 180 | **99.4%** (was 97.2% pre-2026-05-15 retrain) | n/a (no FT — sklearn fit on FT-embeddinggemma vectors) |
| Intent classification (gemma3-1B FT fallback path) | 180 | **96.7%** (was 93.3% pre-2026-05-15 retrain) | ~0% (base gemma3-1B-it) |
| Tool routing — Qwen3.5-4B FT v9 | 160 | **99.4%** | ~60% (zero-shot Qwen base) |
| Multi-step tool chains — Qwen3.5-4B FT v9 | 80 | **97.5% chain / 98.8% decompose / 100% SQL exec** (deterministic) | ~55% (base) — matches v8's chain-shape number AND adds 100% SQL valid execution. See [FT_INSIGHTS §3b](docs/benchmarks/FINE_TUNING_INSIGHTS.md#3b-qwen35-4b-ft-v8--current-production-2026-03-19). |
| Retrieval MRR@10 — EmbeddingGemma FT | 25 q / 26-passage corpus | **98.0%** | 95.3% (base). **Corpus caveat:** small eval set; production KB ~120 chunks |
| Adversarial robustness — pipeline | 60 | **93.3%** | 43.3% (generative-only baseline) |
| Vision (image keyword match) | 10 | **100%** [95% CI: 69.2-100%] | n/a |
| OCR document chat | 29 | 100% | n/a |
| Structured extraction | 29 fields | 100% | n/a |

---

## Keynote Demo Flow

1. **Standard queries** — RAG, SQL, calculator, direct answer. Show the pipeline trace.
2. **Vision** — attach an image, ask about it.
3. **Voice** — speak to the agent, hear the response.
4. **Document upload + OCR** — drop a PDF, watch OCR process, chat with it.
5. **Structured extraction** — extract competitor metrics, cross-source SQL query.
6. **Kill the WiFi** — airplane mode, agent keeps working.
7. **Hybrid escalation** — complex query, low confidence, user approves cloud escalation.
8. **Three-path comparison** — "All" mode: same query across Multi-Models, MoE, Cloud side-by-side. Show latency, accuracy, and cost differences.
9. **Model swap** — toggle base ↔ fine-tuned, show accuracy difference.
10. **Cost comparison** — local $0.00 vs cloud $0.0041 per query.
