# GLM-OCR Integration — Research & Proposal

> Date: 2026-03-16
> **Status: IMPLEMENTED (2026-03-20)**
> See `src/engine/knowledge/ocr_client.py`, `scripts/setup_ocr.sh`, and `finetune/eval_ocr.py`.

---

## What is GLM-OCR?

- **Size:** 0.9B parameters (lightweight, edge-deployable)
- **Architecture:** GLM-V encoder-decoder — CogViT visual encoder + GLM-0.5B language decoder
- **Benchmark:** #1 on OmniDocBench V1.5 (94.62 score)
- **Output:** Raw text, Markdown, or structured JSON (schema-driven)
- **Speed:** 1.86 pages/sec official (PDF), 0.67 images/sec. Our measured speeds via llama-server GGUF: RTX PRO 6000 0.99 pages/sec, M5 Max Metal 0.37 pages/sec, DGX Spark 0.23 pages/sec, MS-S1 MAX Vulkan 0.12 pages/sec
- **llama.cpp/GGUF supported** — merged in [llama.cpp PR #19677](https://github.com/ggml-org/llama.cpp/pull/19677) (Feb 2026). Also runs via Ollama, vLLM, mlx-vlm (Apple Silicon), or HuggingFace Transformers

Capabilities beyond basic OCR:
- Tables, formulas, code blocks
- Scanned/image-based PDFs
- Multi-column layouts, seals, complex figures
- Structured data extraction via JSON schema

---

## Where It Fits in Our Architecture

**One specific place: the document upload pipeline — as a preprocessor before embedding.**

Current flow:
```
PDF uploaded → pypdf (text extraction) → chunking → embeddinggemma → ChromaDB
```

With GLM-OCR:
```
PDF uploaded → GLM-OCR (structured extraction) → clean text + tables as JSON
             → chunking → embeddinggemma → ChromaDB
```

Nothing in the query-time architecture changes. RAG quality improves on complex documents.

**Why pypdf falls short:**
- Misses tables entirely (extracts garbled text or nothing)
- Fails on scanned/image-based PDFs
- Loses multi-column structure
- Drops embedded charts and figures

**Why it does NOT belong in the vision query path:**
gemma3-4B already handles real-time image queries adequately. Adding a GLM-OCR hop to every image query adds latency with marginal benefit for conversational use.

---

## Integration Architecture

GLM-OCR runs via llama-server (same as all other models) — OpenAI-compatible `/v1/chat/completions` with `image_url`. No new runtime dependency.

```bash
llama-server -hf ggml-org/GLM-OCR-GGUF --port 9098
```

GGUF files (from `ggml-org/GLM-OCR-GGUF`):

- `GLM-OCR-Q8_0.gguf` — 950 MB (text model)
- `mmproj-GLM-OCR-Q8_0.gguf` — 484 MB (vision projector)

Prompt modes (not a general-purpose vision model):

- `"Text Recognition:"` — full document text extraction
- `"Formula Recognition:"` — LaTeX formula extraction
- `"Table Recognition:"` — structured table extraction

Integration points:

- New port in `.env` (e.g. `OCR_PORT=9098`) — note: 9097 is already used by whisper
- Reuse existing `SmallLanguageModelClient.generate_vision()` — same API shape
- Called only during `/upload-document` — not at query time
- `DocumentProcessor` gets an optional OCR pre-processing stage

This adds a 5th specialized model to the stack, each doing one job — directly reinforces the "usecase-optimized" architecture thesis.

---

## Demo Narrative

> *"Our agent already answers questions about Nextera's internal data. Now watch what happens when we upload a real analyst report — the agent can immediately answer questions about how Nextera compares to industry benchmarks."*

This is a materially stronger demo than a static knowledge base. It shows:
1. Live document ingestion with complex structure
2. Immediate queryability after upload
3. A real-world business workflow (competitive benchmarking, financial analysis)

---

## Concrete Demo Documents

### 1. Snowflake FY2025 Annual Report ← best fit overall

**URL:** https://s26.q4cdn.com/463892824/files/doc_financials/2025/ar/Snowflake-2025-Annual-Report-and-Proxy-Web-Version.pdf

Snowflake *is* Nextera — cloud analytics platform, B2B SaaS, tiered customers. Contains exactly the metrics the agent already understands: MRR/ARR, customer tiers, NRR, RPO.

Key data inside:
- Total revenue: $3.63B FY2025, +29% YoY
- Customer tiers: 11,159 total; 580 with >$1M trailing revenue; 745 Forbes Global 2000
- Net Revenue Retention: **126%**
- Remaining Performance Obligations: $6.9B, +33% YoY

Example demo queries:
- *"How many Snowflake customers spend more than $1M ARR?"*
- *"What is Snowflake's NRR and how does it compare to Nextera's?"*
- *"What percentage of revenue comes from large enterprises?"*

---

### 2. HubSpot Q4 & FY2024 Earnings Press Release ← best for table-heavy demo

**URL:** https://ir.hubspot.com/static-files/69db7b03-79a2-4a05-88ff-3c085107a3be

Short, dense, six verified financial tables — exactly the structure pypdf garbles and GLM-OCR handles cleanly.

Key data inside:
- Total revenue: $703M Q4 / $2.63B FY2024
- 247,939 customers (+21% YoY), ARPU $11,312
- GAAP operating loss ($67.6M) vs. non-GAAP operating income $460.2M (17.5% margin)
- FY2025 guidance: $2.985B–$2.995B revenue

Example demo queries:
- *"What was HubSpot's subscription revenue in Q4 2024?"*
- *"How many customers does HubSpot have and what's the ARPU?"*
- *"What is the GAAP to non-GAAP operating income reconciliation?"*

---

### Bonus: Datadog FY2024 10-K (SEC EDGAR)

**URL:** https://www.sec.gov/Archives/edgar/data/1561550/000156155025000110/dd_annualxreportx2024.pdf

Pure-play B2B SaaS/cloud monitoring — close analogue to Nextera. Public by law (SEC filing).

Key data: Total revenue $2.68B (+26% YoY), ARR cohort tiers (462 customers >$1M ARR), non-GAAP operating margin 25%, free cash flow $775M.

---

## Summary Table

| Document | Organization | Key Complex Elements | Best Demo Query |
|----------|-------------|----------------------|-----------------|
| FY2025 Annual Report | Snowflake | Customer tier tables, NRR 126%, RPO $6.9B | "How many customers >$1M ARR?" |
| Q4 & FY2024 Press Release | HubSpot | 6 financial tables, GAAP/non-GAAP, guidance | "What was subscription revenue in Q4?" |
| FY2024 10-K | Datadog | ARR cohort tiers, non-GAAP reconciliation | "How many customers have ARR over $1M?" |

---

## Open Questions Before Building

1. **Intentional scope?** Is this a demo feature or a production feature? Scoping matters for how much effort to put into error handling and fallback behavior.
2. **JSON schema design:** GLM-OCR can output structured JSON. Defining schemas per document type (earnings report vs. analyst report vs. annual report) would improve chunk quality significantly.
3. **Chunking strategy for tables:** Tables extracted as JSON need different chunking than prose. A table row is not a natural chunk boundary. This needs a dedicated chunking path in `DocumentProcessor`.
