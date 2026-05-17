# Nextera Platform — Business Scenario & Demo Guide

This document defines the complete business scenario for the Multi-Model
Local AI Agent demo. It serves as the single reference for what the demo showcases,
which data is used, and how to run each scenario.

## Company: Nextera Platform

**Nextera Platform** is a fictional local-first AI infrastructure SaaS product that lets
enterprises run large language models, vector search, and agentic pipelines entirely
on their own hardware — no data ever leaves their network.

- **Tiers**: Starter (€299/mo), Professional (€999/mo), Enterprise (€3,500/mo)
- **Add-ons**: Fine-Tuning Add-on (€500/mo), GPU Hours A100 (€4.50/hr)
- **Customers**: 10 companies across Manufacturing, Healthcare, Software, Analytics,
  Education, Finance, Energy, Technology, Insurance, Logistics
- **Sales history**: Q1 2023 – Q4 2024 (8 quarters, €18.5k → €103.2k quarterly revenue)

## Agent Architecture

The agent uses four specialized Google Gemma models, each optimized for a specific task:

| Role | Model | Params | Task |
|------|-------|--------|------|
| **Thinker** | gemma3-1B-it (fine-tuned) | 1B | Intent classification, query rewriting, tool-result formatting |
| **Doer** | Qwen3.5-4B (fine-tuned v8, LoRA) | 4B | Tool selection, parameter extraction (99.4% accuracy) |
| **Librarian** | embeddinggemma-308M (fine-tuned) | 308M | Semantic embeddings for document retrieval |
| **Vision** | gemma3-4B-it (vision) | 4B | Multimodal image understanding, RAG synthesis |

Combined: ~9.3B parameters, ~6-8 GB VRAM (Q4/Q8 quantized).

## Intent Types & Use Cases

### 1. RAG Query (rag_query) — Knowledge Base Search

The agent searches the Nextera product documentation via semantic vector search
(embeddinggemma) and synthesizes answers (gemma3-4B).

**Data source**: `data/business-documents/*.md` (13 documents) → ChromaDB vector store

**Sample scenarios**:
- "What features are included in the Enterprise plan?"
- "What integrations does the platform support?"
- "What support SLAs are available?"
- "Which plan should a 15-person startup choose?"

### 2. Tool Use — SQL (tool_use → sql_query)

The agent routes to qwen3.5-4b for tool selection, executes SQL queries against
the business database, and formats results with gemma3-1B.

**Data source**: `data/business.db` (SQLite) — tables: products, customers, sales

**Sample scenarios**:
- "What were the total sales revenue figures for 2024?"
- "How many new customers joined in Q3 and Q4 of 2024?"
- "Show top 3 customers by revenue"
- "Which quarter in 2024 had the biggest revenue jump?"

### 3. Tool Use — Calculator (tool_use → calculator)

Same routing via qwen3.5-4b, but the calculator tool handles arithmetic.

**Sample scenarios**:
- "If I have 50 customers paying €999/month, what is my ARR?"
- "What is 23% of 84900?"
- "What's 15% of $45,000?"

### 4. Direct Answer (direct_answer) — Conversational

Gemma3-1B responds directly without any tool or retrieval.

**Sample scenarios**:
- "Hello! What can you help me with?"
- "What is the difference between LoRA and full fine-tuning?"
- "Thanks, that was really helpful!"

### 5. Image Query (image_query) — Visual Understanding

When images are present, the agent bypasses text classification and routes directly
to gemma3-4B (vision model) for multimodal analysis.

**Data source**: `data/demo-images/` (3 sample images generated from actual demo data)

**Sample scenarios**:

| Scenario | Image | Query | Expected insight |
|----------|-------|-------|------------------|
| **Dashboard analysis** | `revenue_chart.png` | "What trends do you see in this revenue chart?" | Consistent growth from €18.5k to €103.2k, accelerating in 2024 |
| **Document reading** | `pricing_table.png` | "Summarize the pricing tiers shown in this table" | Three tiers (Starter/Professional/Enterprise) with features comparison |
| **Architecture review** | `architecture_diagram.png` | "Explain what this system diagram shows" | Four-model agent: Thinker, Doer, Librarian, Vision with routing logic |

**API note**: Images are sent as base64-encoded strings in the JSON request body
(`POST /query { query: "...", images: ["base64..."] }`), not as multipart form uploads.
This keeps the API simple and avoids a separate upload endpoint — practical for demo-sized
images (50-100 KB). For production use with large images, consider a multipart endpoint
or pre-upload to a storage service.

## Sample Data Inventory

| File | Type | Content | Used by |
|------|------|---------|---------|
| `data/business-documents/*.md` | Markdown (13 files) | Nextera product docs (overview, pricing, features, FAQ, security, support) | RAG queries via ChromaDB |
| `data/business.db` | SQLite | products (5), customers (10), sales (8 quarters) | SQL tool queries |
| `data/training-data-data/*.jsonl` | JSONL (5 files) | Fine-tuning data for gemma3, qwen3.5-4b, embeddinggemma | LoRA training |
| `data/demo-images/revenue_chart.png` | PNG | Quarterly revenue bar chart (Q1 2023 – Q4 2024) | Image query demo |
| `data/demo-images/pricing_table.png` | PNG | Nextera tier comparison table | Image query demo |
| `data/demo-images/architecture_diagram.png` | PNG | Four-model agent architecture | Image query demo |

## Demo Walkthrough

### Quick showcase (all intent types)

```bash
# Start all model servers (cheat mode: base + fine-tuned for instant swap)
bash scripts/start_servers.sh --all --bg

# Run the full demo (7 text queries + 3 image queries)
python demo.py
```

### Single image query

```bash
python demo.py -q "What trends do you see?" --image data/demo-images/revenue_chart.png
```

### Interactive mode

```bash
python demo.py --interactive
# Type queries, or 'showcase' for preset queries
```

### Benchmarking

```bash
python scripts/benchmark.py --runs 3 --json results/bench.json
```

### Evaluation

```bash
# Intent classification accuracy (text queries — 60 test queries)
python -m finetune.eval_gemma3

# Vision accuracy (image queries — 10 test queries)
python -m finetune.eval_vision
```
