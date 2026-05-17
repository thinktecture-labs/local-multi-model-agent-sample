# Recommendations for Domain Independence

> How to make the multi-model agent solution reusable across business domains — extending the current Nextera domain and implementing entirely new ones.

**Status: IMPLEMENTED.** All coupling points identified below have been resolved via JSON-driven scenario configuration (`scenarios/*.json`). See `docs/guides/SCENARIO_PLAYBOOK.md` for the operational guide and lessons learned.

## Current State (as of initial analysis)

The engine infrastructure (orchestrator, handlers, client, tools, streaming, eval framework) is already domain-agnostic. Domain coupling lives almost entirely in **data, prompts, and tool descriptions** — all of which can be externalized into configuration without touching the engine code.

### Coupling Heatmap

| Component | Coupling Level | Fix Effort | Phase |
|-----------|---------------|------------|-------|
| Knowledge base docs | **Low** (file-based) | Trivial | 1 |
| SQL seed data | **Medium** (in loader.py) | Low | 1 |
| Prompt templates | **High** (hardcoded strings) | Low | 1 |
| Adversarial refusal message | **High** (hardcoded) | Trivial | 1 |
| SQL tool description + schema | **High** (hardcoded) | Medium | 2 |
| ALLOWED_TABLES | **High** (hardcoded set) | Low | 2 |
| Extraction schema | **High** (hardcoded) | Medium | 3 |
| Training data | **Complete** (hand-crafted) | High | 4 |
| Eval datasets | **Complete** (hand-crafted) | High | 4 |
| Scaffolding builders | **Complete** (regex per domain) | N/A | 5 (optional) |
| Intent taxonomy | **None** (generic) | None | — |
| Handler architecture | **None** (generic) | None | — |
| Inference client | **None** (generic) | None | — |
| Tool registry | **None** (generic) | None | — |
| Confidence router | **None** (generic) | None | — |
| Streaming pipeline | **None** (generic) | None | — |
| UI (Observatory) | **None** (generic) | None | — |

---

## Phase 1: Configuration-Driven Domain (1–2 days)

**Goal:** A new domain can be deployed by editing config files and data — no Python changes.

### 1.1 Create a `domain/` directory structure

```
domain/
  nextera/                   # current domain
    config.yaml              # domain metadata, entity name, description
    prompts.yaml             # all prompt templates
    schema.sql               # CREATE TABLE + seed INSERT statements
    documents/               # *.md knowledge base articles
    training/                # *.jsonl fine-tuning data
    evals/                   # *.jsonl evaluation datasets
  healthcare/                # example second domain
    config.yaml
    prompts.yaml
    schema.sql
    documents/
    training/
    evals/
```

Move the current Nextera content into `domain/nextera/`:
- `data/business-documents/*.md` → `domain/nextera/documents/`
- SQL schema + seed from `data/loader.py` → `domain/nextera/schema.sql`
- `data/training-data/*.jsonl` → `domain/nextera/training/`
- Eval test sets from `finetune/eval_*.py` → `domain/nextera/evals/`

### 1.2 Extract a `DomainConfig` dataclass

```python
@dataclass
class DomainConfig:
    name: str                          # "Nextera Analytics"
    description: str                   # one-line domain summary for prompt injection
    allowed_tables: set[str]           # {"products", "customers", "sales", "competitors"}
    tool_descriptions: dict[str, str]  # per-tool description overrides
    sql_schema_description: str        # full schema string for function model _get_parameters()
    extraction_fields: list[dict]      # field definitions for structured extraction
    rag_system_prompt: str             # RAG synthesis system prompt
    adversarial_refusal: str           # canned refusal message
    direct_answer_system_prompt: str   # direct answer system prompt
    documents_dir: str                 # path to knowledge base markdown files
    schema_sql_path: str               # path to schema.sql
    training_dir: str                  # path to training JSONL files
    evals_dir: str                     # path to eval JSONL files
```

### 1.3 Create `prompts.yaml` for the Nextera domain

Extract every hardcoded prompt from `src/engine/inference/prompts.py` and `src/engine/agent/orchestrator.py` into a YAML file:

```yaml
# domain/nextera/prompts.yaml

rag_synthesis_system: |
  You are a concise, factual assistant for Nextera platform questions.
  Answer using ONLY the provided sources.
  Include ALL relevant details from the sources — do not omit items.
  Use natural prose with bullet points where lists are appropriate.
  Only use sources whose title is relevant to the question.
  Never invent information not in the sources.

rag_synthesis_user: |
  SOURCES:
  {context}

  QUESTION: {query}

  Include all items from the most relevant source.
  Do not pull in details from unrelated sources.

direct_answer_system: "You are a helpful, concise AI assistant."

adversarial_refusal: |
  I can only help with questions about Nextera Analytics —
  products, pricing, customers, and sales data.
  Please rephrase your question or ask something else.

rag_rewrite: |
  Rewrite this query as a short, dense keyword phrase for
  semantic document search. Output ONLY the rewritten phrase.
  No explanation, no quotes, no punctuation at the end.

  Original: {query}
  Rewritten:

tool_format: |
  Turn this tool result into a clear, helpful answer for the user.

  User's question: {query}
  Tool used: {tool_name}
  Raw result:
  {result_str}

  Write a concise, human-readable answer:

multi_step_synthesis: |
  Combine these tool results into a clear, helpful answer.

  User's question: {query}

  Results:
  {results_str}

  Write a concise answer that integrates all results:
```

### 1.4 Load domain at startup via env var

```bash
DOMAIN=nextera  # or DOMAIN=healthcare
```

The server reads `domain/{DOMAIN}/config.yaml` and injects it into the client, tools, handlers, and prompts. A `load_domain(name: str) -> DomainConfig` function reads the YAML files and returns a populated `DomainConfig`.

### 1.5 Make `loader.py` domain-aware

Replace the hardcoded `SQL_SCHEMA`, `SQL_SEED`, and `_DOCS_DIR` with:

```python
def load_domain_data(domain: DomainConfig):
    # Read schema + seed from domain.schema_sql_path
    # Read documents from domain.documents_dir
    # Seed vector store and SQLite from these sources
```

### 1.6 Wire `DomainConfig` into existing code

Pass `DomainConfig` into the components that currently hardcode domain strings:

| File | What to change |
|------|---------------|
| `src/engine/inference/prompts.py` | Read templates from `domain_config` instead of constants |
| `src/engine/agent/orchestrator.py` | Read `_ADVERSARIAL_REFUSAL` from `domain_config.adversarial_refusal` |
| `src/engine/tools/sql_query.py` | Read `ALLOWED_TABLES` from `domain_config.allowed_tables` |
| `src/engine/tools/sql_query.py` | Read schema description from `domain_config.sql_schema_description` |
| `src/engine/tools/calculator.py` | Read tool description from `domain_config.tool_descriptions["calculator"]` |
| `src/engine/tools/vector_search.py` | Read tool description from `domain_config.tool_descriptions["vector_search"]` |
| `src/engine/knowledge/data_extractor.py` | Read extraction fields from `domain_config.extraction_fields` |
| `data/loader.py` | Read docs dir and schema from `domain_config` |

---

## Phase 2: Dynamic Schema Introspection (1 day)

**Goal:** SQL tool descriptions and safety rules derive from the actual database.

### 2.1 Auto-generate `ALLOWED_TABLES`

Introspect the SQLite database at startup:

```python
async def discover_tables(db_path: str) -> set[str]:
    async with aiosqlite.connect(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return {r[0] for r in rows}
```

Use `DomainConfig.allowed_tables` as an override/allowlist. If not specified, all discovered tables are allowed.

### 2.2 Auto-generate schema description for the function model

Read `PRAGMA table_info(...)` for each allowed table and build the schema string dynamically:

```python
async def describe_schema(db_path: str, tables: set[str]) -> str:
    lines = []
    async with aiosqlite.connect(db_path) as db:
        for table in sorted(tables):
            rows = await db.execute_fetchall(f"PRAGMA table_info({table})")
            cols = ", ".join(r[1] for r in rows)
            lines.append(f"{table}({cols})")
    return "; ".join(lines)
```

This eliminates the hardcoded schema string in `SQLQueryTool._get_parameters()`. The tool description and schema description update automatically when the database changes.

### 2.3 Store overrides in DomainConfig

Some tables should remain hidden even if they exist in SQLite (e.g., internal metadata tables). `DomainConfig.allowed_tables` serves as the allowlist; `discover_tables()` is used only as a fallback when no allowlist is configured.

---

## Phase 3: Pluggable Extraction Schemas (1 day)

**Goal:** Structured extraction works for any domain, not just SaaS financials.

### 3.1 Move extraction field definitions into config

```yaml
# domain/nextera/extraction.yaml

table: competitors
unique_key: [company, fiscal_year]
fields:
  - name: company
    type: string
    description: "Company name. If not in the text, infer from the filename."
    required: true
  - name: fiscal_year
    type: integer
    description: "Fiscal year (e.g. 2025)"
    required: true
  - name: revenue
    type: number
    description: "Total revenue as a number. Preserve exact value from text."
  - name: revenue_growth_pct
    type: number
    description: "Year-over-year revenue growth percentage (e.g. 29.0)"
  - name: nrr
    type: number
    description: "Net revenue retention rate percentage (e.g. 126)"
  - name: customers_1m_plus
    type: integer
    description: "Customers with >$1M trailing 12-month revenue"
  - name: total_customers
    type: integer
    description: "Total number of customers"
  - name: product_revenue
    type: number
    description: "Product-specific revenue"
  - name: gross_margin_pct
    type: number
    description: "Gross margin percentage (e.g. 66.0)"
  - name: free_cash_flow
    type: number
    description: "Free cash flow (negative if cash burn)"
```

A healthcare domain might define:

```yaml
# domain/healthcare/extraction.yaml

table: clinical_trials
unique_key: [trial_id, phase]
fields:
  - name: trial_id
    type: string
    description: "Clinical trial identifier (e.g. NCT12345678)"
    required: true
  - name: phase
    type: string
    description: "Trial phase (Phase I, II, III, IV)"
    required: true
  - name: enrollment
    type: integer
    description: "Number of enrolled participants"
  - name: primary_endpoint_met
    type: boolean
    description: "Whether the primary endpoint was met"
  - name: p_value
    type: number
    description: "Primary endpoint p-value"
```

### 3.2 Generate the extraction prompt dynamically

Replace the hardcoded `EXTRACTION_SYSTEM_PROMPT` with a generator:

```python
def build_extraction_prompt(fields: list[dict]) -> str:
    field_lines = []
    for f in fields:
        line = f"- {f['name']}: {f['description']}"
        if f.get("type"):
            line += f" ({f['type']})"
        field_lines.append(line)

    return (
        "You are a data extraction model. Given document text and a source "
        "filename, extract the following fields into a JSON object. Use null for any "
        "field not found in the text. Do not guess — only extract values explicitly stated.\n\n"
        "Fields to extract:\n"
        + "\n".join(field_lines)
        + "\n\nOutput ONLY a single JSON object. No markdown, no explanation, no code fences."
    )
```

### 3.3 Generate CREATE TABLE and INSERT dynamically

Replace `store_competitor()` with a generic `store_extraction()`:

```python
async def store_extraction(db_path: str, table: str, unique_key: list[str],
                           fields: list[dict], data: dict) -> None:
    col_defs = []
    for f in fields:
        sql_type = {"string": "TEXT", "integer": "INTEGER", "number": "REAL",
                    "boolean": "INTEGER"}.get(f["type"], "TEXT")
        nullable = "" if f.get("required") else ""
        col_defs.append(f"{f['name']} {sql_type}{nullable}")

    create_sql = f"CREATE TABLE IF NOT EXISTS {table} (...)"
    # ... build INSERT OR REPLACE from field names and data dict
```

---

## Phase 4: Training Data Generation Framework (2–3 days)

**Goal:** Fine-tuning data can be generated for any domain, not just hand-crafted for Nextera.

This is the hardest phase. Currently, training data is hand-crafted:
- `gemma3_intent.jsonl` — 500+ intent classification examples
- `qwen35_toolcalling.jsonl` — 1,372 tool-calling examples
- `embeddinggemma_retrieval.jsonl` — retrieval pairs

### 4.1 Intent classification data generator

Given a `DomainConfig` with example queries per intent, use a large model to generate synthetic training examples:

```
Domain: {domain.description}
Tables: {domain.table_descriptions}
Documents: {domain.document_titles}

Generate 50 diverse queries that should be classified as {intent}.
Vary phrasing, complexity, and specificity.
Do not repeat patterns.
```

Implement as `finetune/gen_intent_dataset.py` that reads the domain config, calls a large model (Claude API or local large model), and outputs JSONL.

### 4.2 Tool-calling data generator

Given the SQL schema and domain description, generate (query, tool_call) pairs:

```
Database schema:
{auto-generated schema description}

Generate 50 diverse natural-language queries that require SQL to answer.
For each query, provide the correct SQL SELECT statement.
Vary: aggregations, filters, joins, ordering, limits.
```

The current `data_prep_qwen35_toolcalling.py` already has a template-based approach — make it parameterized on domain config instead of hardcoded Nextera queries.

### 4.3 Eval data generator

Same approach as training data, but:
- Generated in a separate pass to avoid train/eval overlap
- Requires manual review before use (automated generation can introduce subtle errors)
- Should cover edge cases specific to the domain

### 4.4 Preserve the eval infrastructure

The eval framework is already domain-agnostic:
- Wilson confidence intervals, McNemar tests, latency stats
- Leakage checks (eval vs. training overlap detection)
- Before/after comparison with `--save` and `--compare`

Only the test data itself needs to change per domain. Refactor eval scripts to load test sets from `domain/{DOMAIN}/evals/` instead of hardcoded Python lists.

---

## Phase 5: Scaffolding Builders — RESOLVED (modules removed)

Earlier iterations of the agent shipped two deterministic pattern-matching
pre-routers — `expression_builder.py` and `sql_builder.py` — that were
inherently domain-specific (hand-written regex pattern sets for Nextera
queries). They were retired in commit `68d52a5` after Qwen3.5-4B FT v8
took over native tool-argument generation. Production now uses
`NullExpressionResolver` / `NullSQLResolver` from
`src/engine/agent/tool_argument_resolver.py`.
- New domains simply don't have scaffolding builders — Qwen FT handles tool routing at 99.4% without them

### Option B: Auto-generated scaffolding (high effort, marginal benefit)

Build a "scaffolding builder generator" that reads the SQL schema and generates common query patterns automatically. This would cover:
- Revenue/metric queries for any numeric column
- Count queries for any table
- Filter queries for any categorical column
- Rank/sort queries for any orderable column

This is a significant effort for marginal benefit given that the fine-tuned function model already achieves 99.4% routing accuracy.

---

## Implementation Checklist

### Phase 1 — Configuration-Driven Domain
- [ ] Create `domain/nextera/` directory structure
- [ ] Move business documents to `domain/nextera/documents/`
- [ ] Extract SQL schema + seed to `domain/nextera/schema.sql`
- [ ] Create `domain/nextera/prompts.yaml` with all prompt templates
- [ ] Create `domain/nextera/config.yaml` with domain metadata
- [ ] Implement `DomainConfig` dataclass and `load_domain()` loader
- [ ] Wire `DomainConfig` into `prompts.py` (replace hardcoded strings)
- [ ] Wire `DomainConfig` into `orchestrator.py` (adversarial refusal)
- [ ] Wire `DomainConfig` into `sql_query.py` (allowed tables, schema description, tool description)
- [ ] Wire `DomainConfig` into `calculator.py` and `vector_search.py` (tool descriptions)
- [ ] Wire `DomainConfig` into `data/loader.py` (document and schema paths)
- [ ] Add `DOMAIN` env var to `.env.example`
- [ ] Verify all existing tests pass with `DOMAIN=nextera`

### Phase 2 — Dynamic Schema Introspection
- [ ] Implement `discover_tables()` SQLite introspection
- [ ] Implement `describe_schema()` for auto-generated schema strings
- [ ] Replace hardcoded `ALLOWED_TABLES` with introspection + config override
- [ ] Replace hardcoded schema string in `SQLQueryTool._get_parameters()`
- [ ] Add integration test: schema description matches actual DB

### Phase 3 — Pluggable Extraction
- [ ] Create `domain/nextera/extraction.yaml`
- [ ] Implement `build_extraction_prompt()` from field definitions
- [ ] Implement generic `store_extraction()` replacing `store_competitor()`
- [ ] Generate `CREATE TABLE` dynamically from extraction schema
- [ ] Verify extraction eval still passes (100% field accuracy)

### Phase 4 — Training Data Generation
- [ ] Move training JSONL files to `domain/nextera/training/`
- [ ] Move eval test sets to `domain/nextera/evals/`
- [ ] Refactor eval scripts to load from `domain/{DOMAIN}/evals/`
- [ ] Implement `finetune/gen_intent_dataset.py` (synthetic intent data)
- [ ] Implement `finetune/gen_toolcalling_dataset.py` (synthetic tool-calling data)
- [ ] Implement `finetune/gen_retrieval_dataset.py` (synthetic retrieval pairs)
- [ ] Add eval/training leakage check for generated datasets

### Phase 5 — Scaffolding Builders (resolved)

- [x] Modules retired in commit `68d52a5`; production uses `NullExpressionResolver` / `NullSQLResolver` from `src/engine/agent/tool_argument_resolver.py` (Qwen3.5-4B FT v8 generates tool arguments natively).

---

## Example: Adding a Healthcare Domain

To validate the architecture, here is what creating a second domain would look like after Phases 1–3:

```
domain/healthcare/
  config.yaml                # name: "MedAssist", allowed_tables: {patients, diagnoses, ...}
  prompts.yaml               # "You are a medical information assistant..."
  schema.sql                 # patients, diagnoses, procedures, medications tables
  extraction.yaml            # clinical trial fields
  documents/
    hipaa-compliance.md
    treatment-protocols.md
    drug-interactions.md
    insurance-coverage.md
    ...
```

```bash
# Deploy the healthcare domain
DOMAIN=healthcare bash scripts/start_app.sh
```

No Python code changes required. The engine reads the healthcare config, loads the healthcare documents, seeds the healthcare database, and serves the healthcare agent — with the same multi-model pipeline, streaming UI, eval framework, and fine-tuning flywheel.
