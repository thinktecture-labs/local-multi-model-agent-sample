# Scenario Playbook — Adding a New Domain to the Multi-Model Agent

This guide captures everything learned from building scenarios alongside the original Nextera scenario. It documents not just the steps, but the mistakes, the debugging, and the hard-won insights that cost hours to discover.

**Audience**: Developers or AI agents tasked with adding a third (or nth) scenario.
**Time estimate**: 2-3 days for a competent developer with domain knowledge.

---

## Prerequisites

Before starting, you need:

- Domain documents (PDFs, Markdown) — minimum 7, ideally 15+
- Domain expertise to write SQL schemas, training data, and eval queries
- Access to the RTX PRO 6000 (or equivalent GPU with 24+ GB VRAM) for fine-tuning
- Understanding of the 5-model architecture (see SOLUTION_OVERVIEW.md)

---

## Phase 1: Scenario Configuration (30 minutes)

Create `scenarios/<name>.json`. Copy from `scenarios/nextera.json` and adapt every field.

```json
{
  "name": "<name>",
  "label": "<Display Name>",
  "brand": "<Brand>",
  "language": "en|de|...",
  "paths": { ... },
  "models": { ... },
  "prompts": { ... },
  "sql": { ... }
}
```

### Key fields

| Field | Example | Notes |
| --- | --- | --- |
| `paths.db` | `./data/<name>.db` | SQLite database path |
| `paths.docs_dir_name` | `<name>-documents` | Markdown docs for RAG |
| `paths.training_data_dir` | `data/training-data-<suffix>` | All JSONL training files |
| `paths.training_data_suffix` | `_<suffix>` | Appended to JSONL filenames |
| `paths.data_loader_module` | `data.loader_<name>` | Python module that seeds the DB |
| `paths.chroma_dir` | `./chroma_db_<name>` | ChromaDB vector store path |
| `models.*` | `models/<model>-ft-<name>.gguf` | Per-scenario GGUF paths |
| `prompts.*` | Domain-specific prompts in target language | See lesson below |
| `sql.allowed_tables` | `["table1", "table2"]` | Whitelist for SQL injection prevention |
| `sql.parameter_description` | Full schema documentation | **Critical** — see lesson below |

### Lesson: The SQL parameter_description is the most important field

The Qwen tool-calling model generates SQL based almost entirely on `sql.parameter_description`. If your schema docs are wrong, the model will generate SQL with wrong column names, wrong JOIN paths, and wrong status enum values. We spent days debugging SQL column errors that turned out to be documentation mismatches.

**Include**: Every table, every column with type and example values, all status enums verbatim, JOIN hints with FK relationships, and example queries.

**Do not**: Abbreviate, omit columns, or describe the schema from memory. Copy-paste from the actual `CREATE TABLE` statements.

---

## Phase 2: Domain Data (2-4 hours)

### 2a. Documents for RAG

Place Markdown files in `data/<name>-documents/`. Each file needs YAML front matter:

```yaml
---
title: "Document Title"
category: category_name
---
```

**Minimum**: 7 documents. **Recommended**: 13+. Each document should be 1-3 pages of dense, factual content.

**Lesson: Dense tables chunk poorly.** If your scenario has wide multi-column tables (compliance matrices, fuel consumption charts, spec sheets, anything where one row carries the whole answer), a single row can get split across chunks. The retrieval finds the right document but the answer is in a different chunk than the question context. Short, focused documents perform measurably better on RAG ground-truth.

**Recommendation**: Keep documents focused. One topic per document. If you have large tables, consider splitting them into separate documents by category.

### 2b. SQL Database

Create `data/loader_<name>.py` that seeds a SQLite database. The loader must:
- Create all tables
- Insert seed data (enough for realistic demo queries)
- Be idempotent (drop + recreate on each run)

**Lesson: Status enum values must match exactly.** The SQL parameter_description says `status = 'einsatzbereit'` but the loader inserts `status = 'Einsatzbereit'` — SQLite is case-sensitive by default. We had 3 SQL failures from this mismatch alone.

### 2c. Demo images

Place in `data/demo-images-<suffix>/`. At least 3-5 images for vision queries (charts, forms, equipment photos).

---

## Phase 3: Training Data (1-2 days — the hardest part)

You need 12 JSONL files per scenario. Here's what each one does, how many examples it needs, and what goes wrong:

### 3a. Intent Classification

**File**: `gemma3_intent_<suffix>.jsonl` (~2000 examples)

Format:
```json
{"instruction": "Classify intent as: rag_query, tool_use, direct_answer", "input": "user query", "output": "rag_query"}
```

Distribution target: ~40% rag_query, ~40% tool_use, ~20% direct_answer.

**CRITICAL LESSON: Vocabulary distribution bias.**

This was our most expensive mistake. The training data generator created calculator-style tool_use examples using domain vocabulary (e.g. "Fuel consumption: 5 vehicles over 50 km at 24 L/100km?") but zero rag_query examples with the same words. Result: any query mentioning the domain numbers ("fuel", "budget", "consumption", "litres") was classified as tool_use — even when the user was asking about a document.

**The fix**: Ensure domain vocabulary appears in BOTH rag_query AND tool_use examples. The structural differentiator should be:

- **rag_query**: References documents/regulations ("according to the docs", "per the SOP", "as stated in the manual")
- **tool_use**: Contains numbers to calculate with, or asks for database operations ("Show all", "How many X are Y")

Check starter patterns before training:
```python
for pattern in ["How much", "How many", "Show all", "List all"]:
    rag_count = sum(1 for q in rag if q.startswith(pattern))
    tool_count = sum(1 for q in tool if q.startswith(pattern))
    print(f'"{pattern}": rag={rag_count} tool={tool_count}')
```

If any pattern is >80% one class, you have a bias problem. Target 30-50% rag for ambiguous starters.

**CRITICAL LESSON: Corrupted labels.**

Nextera's training data had 181 entries where the output label was a literal model answer ("The Enterprise plan starts at EUR 3,500 per month...") instead of `rag_query`/`tool_use`/`direct_answer`. These were synthesis training examples that leaked into the intent file. Always validate:

```python
valid = {'rag_query', 'tool_use', 'direct_answer'}
for d in data:
    assert d['output'] in valid, f"Corrupted label: {d['output'][:50]}"
```

### 3b. Hard Negatives

**File**: `intent_hard_negatives_<suffix>.jsonl` (~200 examples)

Same format as intent. Focus on boundary cases where the classification is genuinely ambiguous. Include examples from all three classes.

### 3c. Intent Eval Holdout

**File**: `intent_eval_holdout_<suffix>.jsonl` (~40 examples)

Blind evaluation set. Never used in training. Must have zero overlap with training data.

### 3d. Tool Routing (Qwen)

**File**: `qwen35_toolcalling_<suffix>.jsonl` (~200 examples)

Teaches Qwen which tool to call (sql_query vs calculator) and what arguments to pass.

### 3e. Synthesis (Gemma3 1B)

**File**: `gemma3_synthesis_<suffix>.jsonl` (~200 examples)

Teaches the 1B model domain-specific response style. Alpaca format with instruction/input/output.

### 3f. 4B Synthesis (Extractive QA)

**File**: `gemma3_4b_synthesis_<suffix>.jsonl` (~200+ examples)

Teaches the 4B RAG synthesis model to cite exact numbers from context instead of hallucinating.

**CRITICAL LESSON: Gemma3-4B is multimodal.**

The 4B model (`google/gemma-3-4b-it`) is `Gemma3ForConditionalGeneration` (vision + text). We wasted hours because:

1. `Gemma3ForCausalLM` (text-only) loads from the same checkpoint but **produces garbage after LoRA merge** due to weight mapping issues.
2. `Gemma3ForConditionalGeneration` requires `token_type_ids` during training. For text-only inputs, set to all zeros via a custom data collator.
3. The vision encoder (SigLIP) must be excluded from LoRA: `exclude_modules=["vision_tower", "multi_modal_projector"]`.
4. EVA initialization fails on the multimodal model (hooks can't parse vision layers). Use standard rsLoRA instead.
5. GGUF conversion needs `tokenizer.model` (SPM file) from HuggingFace — the BPE path fails with an unrecognized pre-tokenizer hash.

See `finetune/train_gemma3_4b.py` for the working implementation.

### 3g. Embedding Training

**File**: `embeddinggemma_retrieval_<suffix>.jsonl`

Positive pairs for embedding model fine-tuning. Query + relevant document chunk.

### 3h. Tool Routing Variants

**Files**: `tool_routing_2tool_<suffix>.jsonl`, `tool_routing_multi_turn_<suffix>.jsonl`

Two-tool scenarios and multi-turn conversations for tool routing.

### 3i. Multi-Step Eval

**File**: `eval_multi_step_<suffix>.jsonl`

Queries that require decomposition into multiple sub-queries. Include `multi_step: true/false` field.

**Lesson: German compound words break regex patterns.** The QueryDecomposer uses regex to detect multi-step queries. German words like "Durchschnittskosten" don't have word boundaries after the stem — trailing `\b` fails. Use open-ended patterns without trailing `\b` for German.

---

## Phase 4: Eval Data (2-4 hours)

You need 8-9 eval files per scenario. **These must have ZERO overlap with training data.**

### Minimum query counts for statistical validity

| Eval | Minimum | Recommended | Why |
| --- | --- | --- | --- |
| Intent (gemma3) | 60 | 180 (60 per class) | 60 per class gives CI ±12pp |
| Tool routing | 80 | 160 | Need both sql_query and calculator coverage |
| RAG ground-truth | **80** | 100+ | 20 queries gave CI ±20pp — **statistically meaningless**. 80 gives ±10pp. |
| Response quality | 30 | 50+ | Mix of all 3 intent types |
| Adversarial | 40 | 60 | Diverse attack types |

**CRITICAL LESSON: 20 eval queries is not enough.**

We initially had 20 RAG ground-truth queries per scenario. Results fluctuated wildly (60% to 90%) depending on which queries were included. With 80 queries, the 95% CI narrowed from ±20pp to ±10pp — still wide but actionable. The 20-query numbers were inflated because the small set was biased toward easier queries.

### Contamination checking

Every eval query must be checked against ALL training data at Jaccard similarity threshold 0.6:

```python
def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0
```

Run `pytest tests/unit/test_eval_overlap.py` — this checks all eval files against all training files. **Do not skip this.** We found contamination in every batch of generated eval queries — 10 violations in BW, 21 in Nextera. The fix is always to rephrase the eval query, never to remove the training example.

**Lesson: Agent-generated eval queries are heavily contaminated.** When using an AI agent to generate eval queries, expect 10-25% contamination rate against training data. The agent naturally produces similar phrasings. Always run the Jaccard check and rephrase violations.

### RAG ground-truth format

```json
{"query": "Specific factual question", "expected_keywords": ["exact_number", "exact_name"], "source_doc": "filename.md"}
```

Every `expected_keyword` must appear verbatim in the source document. Use specific tokens (numbers, codes, proper nouns), not common words.

---

## Phase 5: Fine-Tuning (2-3 hours compute)

### Training order

1. **Embedding model** — needed for intent LogReg and RAG retrieval
2. **Intent LogReg** — fast (<2 min), needs embedding model
3. **Gemma3 1B** (intent + synthesis) — LoRA, ~5 min on RTX PRO 6000
4. **Qwen 3.5B** (tool routing) — LoRA, ~5 min
5. **Gemma3 4B** (RAG synthesis) — LoRA, ~1 min (small dataset)

### GGUF conversion

Each model needs conversion after training:
```bash
PIP=.venv/bin/pip PYTHON=.venv/bin/python3 SCENARIO=<name> bash finetune/convert_<model>_to_gguf.sh
```

**Lesson: Always use `.venv/bin/pip` and `.venv/bin/python3`.** System Python on RTX is externally-managed (Debian) and refuses pip install.

**Lesson: tokenizer.model must be downloaded from HuggingFace.** The SPM tokenizer file is not saved by `tokenizer.save_pretrained()`. The conversion scripts handle this, but if you write a new one, remember to download it.

### Hyperparameters that work

| Model | LoRA r | Alpha | LR | Epochs | Batch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma3 1B | 8 | 16 | 5e-5 | 5 | 4 | EVA init works for 1B |
| Qwen 3.5B | 8 | 16 | 5e-5 | 5 | 2 | |
| Gemma3 4B | 16 | 32 | 5e-5 | 3 | 2 | No EVA (multimodal), all linear layers |
| EmbeddingGemma | 8 | 16 | 5e-5 | 5 | 4 | |

### Quality gates after training

| Metric | Minimum | Action if below |
| --- | --- | --- |
| Training loss convergence | >50% reduction | More data or check data quality |
| Intent LogReg CV | >93% | Check vocabulary bias (Section 3a) |
| Tool routing accuracy | >95% | Check SQL parameter_description |
| RAG ground-truth | >65% | Check routing first, then retrieval, then synthesis |

---

## Phase 6: Debugging RAG Failures

When RAG ground-truth accuracy is low, the failures come from exactly three places. Diagnose in this order:

### Step 1: Is it routing?

Hit the `/query` API endpoint and check the `intent` field in the response. If the query gets routed to `tool_use` or `direct_answer` instead of `rag_query`, it's a routing problem.

Use `scripts/analyze_intent_classifier.py` to understand why:
- Check per-class probabilities and decision margins
- Find nearest training neighbors by embedding similarity
- Analyze vocabulary distribution bias

### Step 2: Is it retrieval?

Check the `vector_search` step in the execution trace. Are the right documents in the top-5? Is the expected keyword in the retrieved chunks?

Common retrieval failures:
- Answer is in a table row that got split across chunks
- Query uses different vocabulary than the document (semantic gap)
- Embedding model wasn't fine-tuned on domain data

### Step 3: Is it synthesis?

The right chunks were retrieved but the model invents wrong numbers. This is the hardest to fix. Options:
- Fine-tune 4B on extractive QA (what we did — modest improvement)
- Use a larger model (7B, 12B) for synthesis
- Add confidence-threshold fallback (try RAG first for borderline routing)

**Lesson: Most RAG failures are routing, not synthesis.** In our experience, 7 of 8 initial failures were routing errors. Fix routing first — it's cheaper and more impactful than model improvements.

---

## Phase 7: Running the Full Eval Suite

```bash
# Start servers
bash scripts/start_servers.sh --scenario <name> --ft

# Start app server
SCENARIO=<name> .venv/bin/python3 -m uvicorn src.server:app --host 127.0.0.1 --port 8000

# Run all evals
SCENARIO=<name> .venv/bin/python3 -m finetune.eval_gemma3
SCENARIO=<name> .venv/bin/python3 -m finetune.eval_tool_routing
SCENARIO=<name> .venv/bin/python3 -m finetune.eval_rag_groundtruth
SCENARIO=<name> .venv/bin/python3 -m finetune.eval_response_quality
SCENARIO=<name> .venv/bin/python3 -m finetune.eval_adversarial
SCENARIO=<name> .venv/bin/python3 scripts/benchmark.py --runs 1

# Tests
SCENARIO=<name> .venv/bin/python3 -m pytest tests/ -q
```

**Lesson: ALWAYS start ALL servers before ANY eval/test.** The eval scripts hit the HTTP API. Missing servers = silent failures or hangs. Start all 5 llama-servers + uvicorn.

---

## Checklist: New Scenario Readiness

- [ ] `scenarios/<name>.json` — all fields populated
- [ ] `data/<name>-documents/` — 7+ Markdown files with front matter
- [ ] `data/loader_<name>.py` — seeds SQLite with correct schema
- [ ] Training data — 12 JSONL files, vocabulary bias checked
- [ ] Eval data — 8-9 files, 80+ RAG ground-truth queries, 0 contamination
- [ ] All 5 models fine-tuned and converted to GGUF
- [ ] `pytest tests/unit/test_eval_overlap.py` — 0 contamination violations
- [ ] Intent LogReg CV >93%
- [ ] Tool routing >95%
- [ ] RAG ground-truth >65% on 80+ queries
- [ ] Response quality >90%
- [ ] `--scenario <name>` flag works in start_servers.sh and start_app.sh
- [ ] Frontend shows correct branding via `/scenario` endpoint
- [ ] Demo script runs end-to-end

---

## Common Mistakes (and how long they took to debug)

| Mistake | Symptom | Debug time | Fix |
| --- | --- | --- | --- |
| SQL column names wrong in parameter_description | SQL errors "no such column" | 2 days | Copy-paste from CREATE TABLE |
| Status enum case mismatch (loader vs docs) | SQL returns 0 rows | 4 hours | Standardize to lowercase |
| Vocabulary bias in intent training | Document queries routed to SQL | 6 hours | Analyze with analyze_intent_classifier.py, rebalance |
| Corrupted labels in intent data | Classification accuracy drops | 3 hours | Validate all labels are in {rag_query, tool_use, direct_answer} |
| 20 eval queries | "90% accuracy!" (actually 68-99% CI) | 4 hours | Minimum 80 queries per eval |
| Gemma3ForCausalLM for 4B | Model outputs garbage after merge | 5 hours | Use Gemma3ForConditionalGeneration + token_type_ids collator |
| EVA init on multimodal model | IndexError in hook registration | 2 hours | Skip EVA, use standard rsLoRA |
| BPE tokenizer path for GGUF | "BPE pre-tokenizer not recognized" | 1 hour | Download tokenizer.model (SPM) from HuggingFace |
| Agent-generated eval queries | 10-25% contamination rate | 2 hours per batch | Always run Jaccard check, rephrase violations |
| German regex patterns with \b | Multi-step detection fails | 3 hours | Drop trailing \b for compound words |
| Training/eval on same queries | Inflated accuracy | Caught by CI | Run test_eval_overlap.py before every commit |
