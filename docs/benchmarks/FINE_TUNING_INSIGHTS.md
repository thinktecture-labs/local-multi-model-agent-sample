# Fine-Tuning Insights — Multi-Model Local AI Agent

> Hard-won lessons from fine-tuning three specialized Gemma models for a local agentic RAG system.
> Every insight was discovered empirically across 20+ training runs, 5 architecture pivots, and 3 complete dataset rewrites.
>
> **Updated 2026-05-15:** Section 3 (FunctionGemma 270M) is now a historical
> reference. Qwen3.5-4B FT **v9** is the production tool caller (retrained on
> corrected training data; replaced v8 as of the 2026-05-15 retrain). Section
> 2 (gemma3-ft) likewise carries a historical framing — intent is served by
> LogReg over FT-EmbeddingGemma vectors today, with gemma3-ft as the fallback.
> See [EVAL_RESULTS_2026-04-05.md](EVAL_RESULTS_2026-04-05.md) and [Section 3b](#3b-qwen35-4b-ft-v8--current-production-2026-03-19) for the Qwen training details.

---

## 1. Architecture: Why Three Small Models Beat One Big One

### The hypothesis

A single 7B or 70B model can do intent classification, tool calling, document retrieval, AND response synthesis — but it does all of them mediocrely. What if we decompose the problem into three tasks and assign each to the smallest model that can solve it?

### The four-model stack (three fine-tuned + one base vision)

| Model | Base | Params | Task | Why this model |
|-------|------|--------|------|---------------|
| **gemma3-ft** | `google/gemma-3-1b-it` | 1B | Direct answers, query decomposition, tool-result synthesis, intent-classification fallback (primary intent is now LogReg) | Instruct-tuned for generation; LoRA preserves classification while 4B handles RAG synthesis |
| **qwen3.5-4b-toolcalling-ft** | `Qwen/Qwen3.5-4B` | 4B | 2-tool routing (calculator / sql_query) with native function calling | Native OpenAI-compatible tool-call format; 99.4% routing accuracy in production. See [Section 3b](#3b-qwen35-4b-ft-v8--current-production-2026-03-19). |
| **embeddinggemma-ft** | `google/embeddinggemma-300m` | 308M | Semantic document retrieval | Bidirectional attention (encoder-style), 768-dim vectors, pre-trained for embedding |

A fourth model (`google/gemma-3-4b-it`, 4B params) handles vision/multimodal queries and RAG synthesis at inference time without fine-tuning. The 4B model's superior multi-document comprehension prevents fact cross-contamination that the 1B model exhibited when synthesizing across multiple source documents.

**Combined parameter count: ~9.3B** across the four FT GGUFs (gemma3-1B + gemma3-4B + Qwen3.5-4B + EmbeddingGemma 308M), plus a ~150KB LogReg classifier on top of the FT EmbeddingGemma vectors. The "small specialists" framing still holds — the 1B and the 308M embedder cover ~93% of traffic via the LogReg primary path; the two 4B models only spin up for tool calling and RAG/vision synthesis.

### The key architectural insight: task boundary design

The hardest problem wasn't training — it was deciding **where to draw the lines between models**.

Our first design had the legacy FunctionGemma 270M tool caller routing between three tools: calculator, sql_query, and vector_search. The 270M model couldn't distinguish sql_query from vector_search — they overlap linguistically ("show me revenue" vs "tell me about pricing"). Every training run produced 0% accuracy on vector_search.

**The fix**: move vector_search to the intent level. gemma3 (1B) handles it as `rag_query` — a fundamentally different decision than tool routing. This reduced the tool caller to a 2-way decision (calculator vs sql_query), which the 270M model solved reliably. (The current production tool caller is Qwen3.5-4B FT v8 — see [§3b](#3b-qwen35-4b-ft-v8--current-production-2026-03-19) — which retains the 2-way schema while running natively at 4B.)

**Lesson**: when a small model fails at N-way classification, the answer isn't more data — it's reducing N by redesigning the task boundary.

### Performance: why small specialists are fast

End-to-end latency for a full agent query (classify + route + execute + synthesize) across **four** hardware configurations, measured 2026-05-17 with `scripts/benchmark.py` (48 queries / machine, FT stack, byte-deterministic, **Q4_K_M synthesis production + llama.cpp b9196**):

| Path | RTX PRO 6000 (CUDA Blackwell) | MBP M5 Max (Metal) | DGX Spark (GB10) | Strix Halo (Vulkan/RDNA 3.5) |
| ---- | ----------------------------- | ------------------- | ---------------- | ---------------------------- |
| **Direct** | **92–123 ms** (p50 123) | 180–246 ms (p50 244) | 450–624 ms (p50 620) | 361–495 ms (p50 494) |
| **Tool use (calc)** | **194–196 ms** (p50 195) | 432–437 ms (p50 434) | 867–879 ms (p50 871) | 828–838 ms (p50 834) |
| **Tool use (SQL)** | **199–348 ms** (p50 310) | 434–745 ms (p50 675) | 881–1658 ms (p50 1463) | 842–1461 ms (p50 1323) |
| **RAG (4B synthesis)** | **393–515 ms** (p50 456) | 688–979 ms (p50 817) | 1660–6787 ms (p50 1997) | 1478–1893 ms (p50 1640) |
| **Image (4B vision)** | **650–1515 ms** (p50 922) | 991–2748 ms (p50 1648) | 2341–6960 ms (p50 4141) | 2184–5498 ms (p50 3628) |
| **bench.py overall** | **median 329 ms / mean 437 ms** | median 688 / mean 791 | median 1657 / mean 2018 | median 1455 / mean 1630 |

> **Note:** All numbers measured 2026-05-17 with the Q4_K_M synthesis production stack on `vendor/llama.cpp` pinned at **b9196** (commit `bef86db`), Qwen3.5-4B FT v8 on port 9091, NullResolvers active. The b9196 pin bump (from March-era `cf21cdf36`) cut bench median 14–53% on top of the Q4_K_M gains, with **identical-or-better RAG quality** across the fleet (rag_groundtruth strict: RTX +3 / DGX +1 / Strix +1 / MBP 0). Strix Halo gained the most (-53% median) because the newer Vulkan kernels improve Qwen tool-routing throughput on RDNA 3.5. Visualisation: [`benchmark_visualization.html`](benchmark_visualization.html). Raw matrix: [`benchmark_matrix.json`](benchmark_matrix.json).

**Per-step breakdown (p50, latest run per machine):**

| Step | RTX PRO 6000 | MBP M5 Max | DGX Spark | Strix Halo |
| ---- | ------------ | ---------- | --------- | ---------- |
| classify_intent (LogReg) | 5 ms | 6 ms | 7 ms | 7 ms |
| rewrite_query (1B) | 64 ms | 130 ms | 315 ms | 250 ms |
| vector_search (embed) | 11 ms | 14 ms | 26 ms | 18 ms |
| synthesize_response (4B Q4_K_M) | 380 ms | 691 ms | 1681 ms | 1491 ms |
| select_tool (Qwen FT) | 189 ms | 430 ms | 866 ms | 829 ms |
| execute_tool | 0 ms | 0 ms | 1 ms | 1 ms |
| format_response (1B) | 26 ms | 50 ms | 128 ms | 102 ms |
| direct_response (1B) | 119 ms | 241 ms | 614 ms | 488 ms |
| analyse_image (4B vision Q4_K_M) | 922 ms | 1647 ms | 4141 ms | 3628 ms |
| **bench.py overall (median)** | **329 ms** | **688 ms** | **1657 ms** | **1455 ms** |

→ **Ranking (lower = better):** RTX < MBP < Strix < DGX. The b9196 bump re-ordered the bottom of the leaderboard — Strix Halo's Vulkan/RDNA-3.5 path now beats DGX Spark's GB10 CUDA on the overall median. **RTX ~2× faster than MBP**, ~5× faster than DGX, ~4.4× faster than Strix.

For comparison, a single 7B model doing everything in one forward pass: 1-3 seconds per query. Our four-model pipeline is **3-10x faster on GPU** because each model is small enough for near-instant inference, and the pipeline is sequential (no parallel GPU contention).

**CPU-only mode**: the models are small enough (270M-4B, Q8_0/Q4) that CPU inference is viable. Calculator calls stay under 700 ms; RAG queries are ~4x slower than Metal (median 3.8s vs 942ms) because 4B synthesis dominates. SQL and direct queries are ~2.5x slower. Start servers with `bash scripts/start_servers.sh --cpu` to force CPU-only (sets `--n-gpu-layers 0`). Benchmark: `python scripts/benchmark.py --json results/bench-cpu.json`.

**VRAM budget**: all four models combined use ~5 GB — leaving the rest of GPU memory available for batching or other workloads. A 7B model at fp16 needs ~14 GB; a 70B needs ~140 GB. In CPU-only mode, VRAM is not used at all — the models run entirely in system RAM (~3 GB).

### Query rewriting: diminishing returns and the dual-query fix

The RAG handler rewrites user queries via gemma3-1B before vector search: "What compliance certifications does Nextera have?" → "Nextera compliance certifications SOC2 HIPAA GDPR". The original claim was "20-30% retrieval improvement" — but that was measured against the base embeddinggemma model (MRR 0.95) before fine-tuning. With the fine-tuned embeddings at MRR 0.98, the rewrite's marginal value is much smaller.

**For uploaded documents, the rewrite is actively harmful.** The 1B model was fine-tuned on Nextera vocabulary and hallucinates keywords for unfamiliar domains:
- "How many Snowflake customers spend more than $1M ARR?" → "arr-growth shelf-ranking limits pricing total-bill total-revenue" (hallucinated)
- "What is Snowflake's net revenue retention rate?" → "snowflake net retention rate" (dropped "revenue")

These bad rewrites produce worse embeddings than the original query, causing the correct chunks to rank outside the synthesis window.

**Fix (2026-03-21):** Dual-query search — the RAG handler now searches with both the rewritten query AND the original query, deduplicates by document ID, and keeps the top results by score. The rewrite can only help (when it adds useful keywords), never hurt (the original query always participates). Cost: one extra vector search (~5ms) per RAG query.

Additionally, `RAG_TOP_K` was increased from 5 to 7 (later to 15) and `RAG_CONTEXT_DOCS` from 4 to 5 (later to 10) to accommodate the merged result set and support aggregation queries over scattered facts (e.g. "how many sessions does speaker X have?" across a conference agenda). The `document_chat` route was also updated to use these config values instead of hardcoded top-5.

**Open question:** Whether to remove the rewrite step entirely. With dual-query search it's safe but costs ~22ms (CUDA) / ~46ms (Metal) per RAG query for an LLM call that produces marginal-to-harmful rewrites on non-Nextera content. The latency savings would be noticeable on stage. A simpler alternative: skip the rewrite when the original query already contains specific technical terms (detected by a simple heuristic).

---

## 2. gemma3-ft — Intent Classification + Response Synthesis

> **Historical framing.** This section documents the **v5 training arc** when gemma3-ft was
> the primary intent classifier and reached **95.0%** on the 180-query eval. In production
> today (post-2026-05-15 retrain on corrected training data), intent is served by a
> **LogReg head over the FT EmbeddingGemma vectors** at **99.4%** (~5ms, deterministic —
> the primary path, ~93% of traffic), and gemma3-ft is the **fallback path at 96.7%**
> for queries the LogReg head abstains on. The headline numbers below (95%, 95.0%) describe
> the v5 model when it was the sole classifier; the current production numbers live in
> [§10 Quick Reference](#10-quick-reference). The lessons themselves (catastrophic
> forgetting, multi-task ratios, EVA + rsLoRA) are unchanged.

### The dual-role challenge

gemma3 serves two roles in the pipeline:
1. **Classifier** — output one of three labels (`rag_query`, `tool_use`, `direct_answer`) — now the fallback path; primary intent is the LogReg head
2. **Synthesizer** — generate natural-language responses from tool results and direct answers (RAG synthesis now uses gemma3-4B instead)

These are fundamentally different tasks: classification wants short, constrained output; synthesis wants fluent, creative text. Training for one destroys the other unless handled carefully.

### The winning recipe

```bash
python -m finetune.train_gemma3 --task intent --epochs 7 --lr 5e-5
```

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | `google/gemma-3-1b-it` | **Instruct model, not base** — retains generation ability after LoRA |
| LoRA | r=8, alpha=16, all 4 attention projections | Modifies attention without destroying pretrained weights |
| **rsLoRA** | `use_rslora=True` | Uses `alpha/sqrt(r)` scaling instead of `alpha/r` — stabilizes gradient norms across training ([Kalajdzievski, 2023](https://arxiv.org/abs/2312.03732)) |
| **EVA initialization** | `init_lora_weights="eva"` | Data-driven SVD on activation vectors — replaces random Kaiming init with directions capturing the most variance ([Paischer et al., 2024](https://arxiv.org/abs/2410.07170)) |
| **Cosine LR decay** | `lr_scheduler_type="cosine"` | ~5% accuracy improvement over linear decay; smooth convergence avoids late-training oscillations |
| **Deterministic training** | Full CUDA determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `cudnn.deterministic` + manual seeds |
| Learning rate | 5e-5 | Higher than typical LoRA (1e-4 was too aggressive for 1B-it) |
| Epochs | 7 | Sweet spot: enough for classification signal, short enough to preserve synthesis |
| QLoRA | Disabled | bitsandbytes quantized weights can't be converted to GGUF by llama.cpp |
| Effective batch size | 16 (4 x 4 grad accum) | Stability threshold per Databricks research |

### Results

**95% accuracy on 180 unseen queries** *(v5 era — see historical framing at the top of this section; current fallback-path number is 93.3%)* (60 per class, zero overlap with training data verified by Jaccard similarity check):

| Intent | Accuracy | Notes |
|--------|----------|-------|
| `rag_query` | **95%** (19/20) | One boundary query misrouted to tool_use |
| `tool_use` | **100%** (20/20) | Perfectly stable across all training runs |
| `direct_answer` | **90%** (18/20) | +10% over v4 — model learned better direct/tool boundary |

**Run-to-run variance: 3.3%** (91.7-95.0% across seeds and hardware) — down from 8.3% before EVA/rsLoRA. The remaining variance comes from EVA initialization (SVD on activations varies across hardware/library versions).

### Why EVA + rsLoRA matter for small models

Standard LoRA initializes `lora_A` with random Kaiming uniform values. On a 1B model with only 1.5M trainable parameters (0.15% of total), this random starting point has outsized impact — the optimization landscape is small and every initial direction matters.

**EVA** (Explained Variance Adaptation) replaces random init with an SVD of the model's own activation patterns on your training data. It finds the directions where the model's representations vary most, then initializes LoRA matrices along those directions. This means training starts from a "warm" position that already captures domain-relevant patterns.

**rsLoRA** fixes a scaling bug in standard LoRA: the default `alpha/r` scaling factor causes gradient collapse at higher ranks. The `alpha/sqrt(r)` fix ensures gradients maintain consistent magnitude regardless of rank, which matters when r=8 is already on the edge for a 1B model.

The combination produces measurably more stable training: tool_use is **perfectly stable at 100%** across all seeds. rag_query and direct_answer vary by ±5% between runs (95-100% and 80-90% respectively), with failures on genuinely ambiguous queries like "What happens if one of the models crashes?" (could be rag_query about system architecture or direct_answer about runtime behavior).

### What we tried and why it failed

| Attempt | Result | Root cause |
|---------|--------|------------|
| **Full fine-tuning** (all weights) | 95% classification, **empty synthesis** | Catastrophic forgetting — classification signal overwrites generation ability |
| **LoRA on base model** (`-pt`) | 90% classification, **gibberish synthesis** | Base model has no instruction-following ability to preserve |
| **QLoRA** (4-bit) | Training works, **GGUF conversion fails** | `convert_hf_to_gguf.py` raises `NotImplementedError: bitsandbytes` |
| **30 training examples** | **0% accuracy** | Insufficient signal to shift a 1B model's behavior |
| **Classification-only training** | High accuracy, **empty responses** | Model learned "output one word then stop" — killed generation |
| **Random LoRA init** (default Kaiming) | 86-95% accuracy, **3-5% variance** | Random starting point leads to different convergence basins per run |

### The multi-task training breakthrough

The critical discovery: if you train a model for classification only, it forgets how to generate text. Even with LoRA (which preserves most weights), the adapter learns to suppress generation.

**Fix**: include generation examples in the training data alongside classification examples.

**Intent dataset** (classification + generation reminders):

```
1878 total examples:
  1697 classification  →  "Classify: rag_query / tool_use / direct_answer"
   166 generation      →  "Hello!" → "Hi there! How can I assist you?"
    15 tool formatting →  "Turn this result into a sentence" → "The total revenue was EUR 311,500."
```

The 15 tool-result formatting examples were the most impactful per-example — they match the exact prompt pattern from `_handle_tool_use()` step 3, which was the specific failure point. Without them, the model returned empty strings when asked to format calculator/SQL results into human-readable responses.

**Ratio insight**: 90:9:1 (classify:generate:format) works because the model's base instruct ability handles generation; it just needs periodic reminders not to forget. Too many generation examples (>20%) dilutes the classification signal.

### Dedicated synthesis dataset

The intent dataset's 181 generation examples (166 + 15) are reminders to prevent catastrophic forgetting — they aren't enough to teach the model HOW to synthesize well. A dedicated synthesis dataset trains the model on all 4 production prompt patterns from `agent.py`:

**Synthesis dataset** (generated by `finetune/gen_gemma3_synthesis_dataset.py`):

```
201 total examples:
   79 RAG synthesis        →  "Answer from SOURCES..." → grounded answer with [Source: ...] citations
   69 tool formatting      →  "Turn this tool result..." → natural language from raw JSON (SQL + calculator)
   23 multi-step synthesis →  "Combine these tool results..." → integrated answer from 2 tool steps
   30 direct answer        →  "You are a helpful, concise AI assistant." → greetings, capability, domain knowledge
```

**Critical fix**: the original 7 synthesis examples used a mismatched instruction prompt (`"You are a helpful AI assistant for Nextera Platform..."`) that didn't match what `agent.py` actually sends at inference time (`"You are a factual assistant for Nextera platform questions..."`). This is the same category of bug that caused 0% accuracy on the legacy FunctionGemma tool caller (section 3). The new 201 examples use the **exact** prompt templates extracted from agent.py.

**What FT teaches vs what RAG provides**: fine-tuning teaches the model _how_ to respond (format, tone, citation style, tool output formatting). RAG provides _what_ to respond with (current facts at query time). They're complementary — RAG without FT gives correct facts poorly formatted; FT without RAG gives good formatting with hallucinated facts.

### Synthesis training experiments and production recommendation

Three training configurations were tested on the RTX PRO 6000:

| Configuration | Command | Intent Accuracy | Notes |
| --- | --- | --- | --- |
| **Intent-only (v5)** | `--task intent` | **95.0%** (95/100/90) | Production model — best classification |
| Intent-only (v4) | `--task intent` | 93.3% (100/100/80) | Previous best — same config, different EVA init |
| Combined (both) | `--task both` | 90.0% (100/95/75) | -5.0% vs v5 — synthesis dilutes classification signal |
| Synthesis-only | `--task synthesis` | 15-17% (0/40-45/5) | Catastrophic forgetting — destroys classification |

**Production recommendation: use `--task both`** (2,252 examples) — *updated 2026-05-15*. This recommendation flipped from `--task intent` because the architectural role of gemma3-ft changed: it is no longer the primary intent classifier. **LogReg over FT-EmbeddingGemma vectors handles ~93% of intent traffic at 99.4%**; gemma3-ft now serves the intent **fallback** plus query rewriting, decomposition, and direct-answer generation. Synthesis quality matters more than peak classification accuracy in that role. The 2026-05-15 retrain on `--task both` measured **96.7%** intent on the fallback path (180-query eval) — higher than the v5 `--task intent` 95.0% — because the corrected training data (commit `62addc1`) outweighed the synthesis dilution. The v5 era table above (`--task intent` 95.0% vs `--task both` 90.0%) reflects the pre-data-correction world.

The `--task both` mode (2,079 concatenated examples) exists for cases where synthesis quality is measurably poor. The `--task synthesis` mode should never be used for production — it exists only for isolated synthesis experiments.

**Open question**: a synthesis quality eval (measuring response format, citation accuracy, tool output formatting) would determine whether `--task both` provides measurable synthesis improvement worth the 5.0% classification cost. Until then, intent-only is the safer choice.

### Output parsing for 1B models

The 1B model sometimes outputs multiple lines when asked for a single label. We take only the first line:

```python
first_line = response.content.strip().split("\n")[0]
intent_str = first_line.strip().lower().replace(" ", "_")
```

Without this, multi-line output like `"rag_query\nDirect_answer\ntool_use"` fails to match any intent. This is a 1B-specific issue — the legacy 270M tool caller was more constrained.

---

## 3. FunctionGemma 270M — Legacy 2-Tool Routing (superseded by Qwen3.5-4B FT v8, see [§3b](#3b-qwen35-4b-ft-v8--current-production-2026-03-19))

> **Historical context.** This section documents the **FunctionGemma 270M** training arc that
> preceded the current production tool caller. It is preserved because the lessons
> (format-mismatch disaster, hard negatives, single-turn architecture, capability ceiling)
> motivated the move to Qwen3.5-4B FT v8. The legacy GGUF and training script are no longer
> shipped; `finetune/data_prep_qwen35_toolcalling.py` carries forward the curated dataset.

### Why this model existed

FunctionGemma 270M was Google's purpose-built function-calling model. Unlike general LLMs that learn tool calling as a side task, FunctionGemma had native control tokens:

```
<start_function_declaration> / <end_function_declaration>  →  tool definitions
<start_function_call> / <end_function_call>                →  model output (tool invocation)
```

These tokens were baked into the tokenizer and the model's pretraining. Fine-tuning on our specific tools (calculator, sql_query) taught the model WHICH tool to pick and WHAT arguments to extract — without needing to learn the function-calling format itself.

### The winning recipe (legacy)

```bash
# Historical command — the legacy module was removed; do not run.
python -m finetune.train_functiongemma --epochs 3 --batch-size 4
# Full FT, lr=1e-5, 3 epochs, batch=4, grad_accum=4, completion_only_loss=True
```

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | `google/functiongemma-270m-it` | Purpose-built for function calling |
| Method | **Full fine-tuning** (not LoRA) | Structured output format needs all weights to shift |
| Learning rate | 1e-5 | Google's official Colab notebook recommendation |
| Batch size | 4 | Google uses 4; matches their Colab notebook |
| Grad accumulation | 4 | Effective batch 16; Google uses 8 (eff 32) on 9.6K examples |
| Epochs | 3 | Sweet spot: 2-3 epochs. 6 epochs causes catastrophic overfitting (72.5%) |
| max_seq_length | 1024 | Google uses 997; actual data is 157-218 tokens |
| completion_only_loss | `True` | Train only on the function call output, not the prompt tokens |
| Dataset | 1,266 balanced (609 calc + 642 sql + 15 hard negatives) | Near-equal representation + targeted hard negatives |
| bf16 | `True` on CUDA/MPS | fp16=True crashes on bfloat16-native model |
| Deterministic CUDA | `CUBLAS_WORKSPACE_CONFIG=:4096:8` + seeds | Reduces run-to-run variance |

### Results

**~91% tool selection accuracy** on 160 unseen queries (80 per tool, zero overlap verified).
Reproducible range: **90.6-91.2%** across consecutive training runs (seed=42, CUBLAS deterministic):

| Tool       | Accuracy              | Notes                                                 |
|------------|-----------------------|-------------------------------------------------------|
| sql_query  | **83-85%** (67-68/80) | Business-vocabulary queries misrouted to calculator    |
| calculator | **97.5%** (78/80)     | Rock-solid; entity-mentioning queries mostly resolved  |

Training variance note: ~1-2% run-to-run despite deterministic seeds. Early stopping
settings are critical — patience=3, threshold=0.01 (patience=5 causes 10%+ accuracy drop).

### The format mismatch disaster (biggest bug we found)

Our biggest breakthrough was discovering the training format was wrong:

```
OUR FORMAT:    call:tool{key:<escape>val<escape>}<end_of_turn>
CORRECT:       call:tool{                    {"key": "val"}}<start_function_response>
```

Three differences: argument structure, escape handling, end token. The model was learning a different output format during training than what llama-server expected at inference. Even at 85% training accuracy, inference produced 0% correct function calls.

**The fix**: use `tokenizer.apply_chat_template(messages, tools=schemas)` for BOTH prompt AND completion during training — exactly as Google's official Colab notebook does. This produces the native function-calling format with proper control tokens.

**Lesson**: for models with specialized control tokens, NEVER manually construct the training format. Always use the model's own tokenizer to generate it.

### Why LoRA didn't work for FunctionGemma 270M

We tried LoRA (r=16, targeting attention + MLP layers) and it capped at ~75% accuracy vs 95% with full FT. The reason: structured output format (JSON arguments inside function-call tokens) required shifting weight distributions throughout the entire network. LoRA's low-rank updates couldn't capture the full distribution shift needed for precise structured output at this parameter count. (At 4B — see §3b — LoRA r=16 is sufficient.)

`completion_only_loss=True` is critical: without it, the loss is dominated by predicting the long tool-declaration prompt (~80% of tokens), diluting the gradient signal for the actual tool selection and argument extraction (~20% of tokens).

### Training data quality beats quantity

| Dataset | Examples | Accuracy | Lesson |
|---------|----------|----------|--------|
| Template-expanded | 3,481 | 43% | Syntactic variations (same SQL with different phrasings) teach surface patterns, not routing |
| Curated v1 | 567 | 77% | Hand-crafted examples with controlled diversity beat automated expansion |
| **Balanced 2-tool + hard negatives** | **1,266** | **95%** | Equal calc/sql + boundary disambiguation examples |

**567 curated examples beat 3,481 template-expanded ones.** The expanded dataset's syntactic variations taught the model to pattern-match on surface features ("total" = calculator) instead of understanding the semantic routing decision ("does the user provide numbers or ask to look them up?").

### Hard negatives: the boundary disambiguation technique

The 3 remaining failures were all sql_query queries misrouted to calculator. The pattern: queries about "total revenue", "average MRR", and "quarterly figures" — words that appear in both calculator and SQL contexts.

**Fix**: add 10 hard-negative calculator examples that use business vocabulary but provide explicit numbers:
- "If total revenue was EUR 311,500 and costs were EUR 200,000, what is the profit?" → calculator
- "What is the average of 55100, 68300, 84900, and 103200?" → calculator

These teach the decision boundary: **numbers provided by user = calculator; numbers to be looked up from database = sql_query.**

### Tool description alignment

Tool descriptions in training data must **exactly match** inference descriptions from `src/engine/tools/*.py`. Any mismatch causes the model to learn a different decision boundary than what it sees at inference time.

We discovered that inline examples in the calculator description (`'100 * 0.15'`, `'sqrt(144)'`) acted as a "magnet" — when the model was uncertain, it copied these as default arguments instead of extracting arguments from the user query.

### Multi-step architecture: single-turn calls + expression builder

A critical architectural finding: the 270M FunctionGemma model **could not handle multi-turn conversation history**. Google's documentation confirmed it was not trained on multi-turn interactions. Passing prior tool results as conversation context caused the model to generate garbled or repetitive output.

**The fix (historical, since retired)**: each step got a fresh single-turn call to FunctionGemma. Prior results were injected via a deterministic **expression builder** module (a pattern matcher that replaced LLM-generated expressions). For multi-step queries it extracted variables from SQL results (e.g., `revenue = 84900`) and built math expressions from 12 domain-specific patterns; for single-step calculator queries 16 additional patterns overrode FunctionGemma's expressions (which were only 38.9% correct). An attempt to use gemma3-ft for expression generation scored just 10% — fine-tuning for intent classification destroyed general math capability. The expression-builder module was retired in commit `68d52a5` after Qwen3.5-4B FT v8 took over native expression generation.

Additionally, step 1 of multi-step queries restricted the FunctionGemma schema to `sql_query` only (since step 1 is structurally always a data lookup), and `rephrase_for_sql()` (`src/engine/agent/tool_argument_resolver.py`) converted decomposed step descriptions ("Find X") into FunctionGemma training vocabulary ("Show X").

These three changes — single-turn calls, expression builder, schema restriction — eliminated ~800ms of LLM latency per multi-step query and improved multi-step tool chain accuracy from ~55% to ~97.5%.

---

### 3b. Qwen3.5-4B FT v8 — Current Production (2026-03-19)

FunctionGemma 270M (§3 above) was superseded by Qwen3.5-4B FT v8. The fundamental issue was a **capability ceiling** at 270M parameters: the model could select tools (88.8-92.5% accuracy) but generated wrong expressions 60% of the time — requiring 1,615 lines of deterministic scaffolding to compensate.

**Training approach**: QLoRA r=16 (Unsloth), unlike full fine-tuning used for FunctionGemma. The larger base model (4B vs 270M) can learn argument generation reliably via LoRA.

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | `Qwen/Qwen3.5-4B` (4B params) | Native function calling support, strong code/reasoning |
| Method | **LoRA r=16** (Unsloth) | 4B model retains reasoning; LoRA sufficient (unlike 270M) |
| Epochs | 2 | Script default; a 3-epoch experiment on 2026-05-15 regressed tool routing by 1.3pp and multi-step by 1.2pp despite lower train_loss. Eval_loss bottoms around epoch 2 (0.0038 → 0.0033 then upticks at 0.0033). |
| Examples | 1,372 (1,331 single + 41 multi-turn) | Extended from FunctionGemma's 1,266 |
| Quantization | Q4_K_M GGUF | Standard production quantization. The shipped file is `qwen3.5-4b-toolcalling-ft-<scenario>-q4_k_m.gguf` (~2.5 GB). Prior releases used `-f16.gguf` as the filename for the same Q4_K_M bytes — a legacy naming artefact that was corrected in commit `577cb2b`. |
| Inference flags | `--jinja --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0` | Disable thinking mode (2x speedup), native Qwen3.5 chat template |

**Results (v9, post-2026-05-15 retrain on corrected training data):**

| Metric | FunctionGemma 270M FT (legacy) | Qwen3.5-4B FT v8 (pre-data-fix) | Qwen3.5-4B FT v9 (current) |
|--------|----------------------|------------------|------------------|
| Tool routing accuracy | 88.8-92.5% | **99.4%** | **99.4%** ✓ |
| Multi-step tool chain | 70.0% | **97.5%** (78/80) | **97.5%** (78/80) ✓ |
| SQL execution validity | n/a | 100% (caveat: training data had `FROM revenue` bug) | **100%** (79/79) ✓ |
| Expression correctness | 39.5% (model alone) | ~95% | **95.0%** (76/80) ✓ |
| Latency (CUDA p50) | ~100ms | ~381ms | ~381ms |
| Latency (Metal p50) | ~100ms | ~1115ms | ~1115ms |
| Latency (DGX Spark p50) | ~100ms | ~2410ms | ~2410ms |

**v9 multi-step matches v8's 97.5% AND adds 100% SQL execution validity — a Pareto improvement.** Critical caveat about how this was achieved: the v9 retrain on corrected training data, in isolation, initially measured ~90% (5 failures) on the first eval run, because the corrected training data shifted Qwen's decomposition style enough to break some chain shapes. The 97.5% recovery required four prompt-engineering commits to land alongside the retrain:

| Commit | Change |
|---|---|
| `3db64e4` | Route `concretize_step` through Qwen FT (FUNCTION role) — gemma3-1B hallucinated growth assumptions and value-substitution; Qwen reads SQL-result structured context faithfully |
| `e6a3276` | Diversify `decomposer_fewshot` (the original 2 examples bled "best product last quarter" and "10 customers × 999/month" onto unrelated queries) |
| `c8d4eb5` + `0c644fe` | Drop anchor examples from `multi_step_synthesis_prompt` (the `[10 customers, 999, 119880]` example sentence was copied verbatim into unrelated answers); add explicit unit-preservation rules |
| `118b6a1` | Route multi-step synthesis through Qwen FT — gemma3-1B's prose-layer plagiarism of fewshot numbers was a capability ceiling no prompt could close |
| `f69fd41` | Decomposer: enforce exactly-2 steps + explicit "use the user's EXACT numbers, do NOT substitute from examples" rule |

The system is **fully deterministic** (`deterministic=True` plumbed end-to-end via `client.generate(temperature=0, seed=42, top_k=1, top_p=1)` and `client.call_function(deterministic=True)`); three back-to-back eval runs produce byte-identical predictions (SHA256 `a4bfb61c972b0949` × 3). So 97.5% is the stable measurement, not a sample.

The 2 remaining chain-shape failures (out of 80):
1. **"Show total 2024 revenue and the mean quarterly revenue"** → `[sql_query]` only — the model correctly answers both parts via a single `SUM/AVG` aggregation; the eval gold chain (`[sql_query, calculator]`) is overly prescriptive. Eval false-positive.
2. **"How much more does the Enterprise plan cost compared to Starter?"** → `[calculator]` instead of `[sql_query]` — the model uses hardcoded prices (3500-299) from training memory rather than querying the DB. Numerically correct, semantically wrong. Same failure appears in the single-step tool-routing eval (double-counted).

**Architectural changes that landed alongside the v9 retrain (2026-05-15):**

| Change | Commit | Rationale |
|---|---|---|
| Route `concretize_step` through Qwen FT (FUNCTION role) | `3db64e4` | gemma3-1B hallucinated growth assumptions (e.g. "15% growth per year") and confused multi-row SQL results. Qwen reads structured context faithfully. |
| Route multi-step synthesis through Qwen FT (FUNCTION role) | `118b6a1` | gemma3-1B copied few-shot example numbers verbatim into synthesized answers regardless of prompt rules. Qwen at ~95% digit fidelity in a 40-query audit. |
| Diversify `decomposer_fewshot`, drop synthesis examples | `e6a3276`, `c8d4eb5` | Few-shot leakage from anchor examples ("best product last quarter", "10 customers × 999 = 119,880") onto unrelated queries. |
| Enforce `steps[:2]` in decomposer + "use exact user numbers" rule | `f69fd41` | Decomposer sometimes duplicated step 2 or substituted time periods from fewshot. |

**Known limitations (v9):**

- **Decomposer time-period substitution (partial):** despite the explicit "use the user's exact numbers" rule, the decomposer occasionally substitutes from the fewshot anchor (e.g. user asks "2 years", decomposer emits "3 years" in the sub-task). Synthesizer's final prose preserves the user's wording, but the chain math may use the substituted value.
- **Concretize value hallucination:** when SQL didn't fetch a value referenced in a sub-task (e.g. user asks "average MRR per customer" but the SQL only `COUNT(*)`s), Qwen-routed concretize fills in from training memory (`3500` from the Enterprise plan price). Recommendation: future iteration should detect missing variables and re-route to fetch them.
- **Multi-step synthesizer digit fidelity:** ~95% on a 40-query audit (1/40 genuine hallucination, 2/40 chain-corruption symptoms). Production-acceptable for curated keynote-demo queries; worth flagging for unconstrained user input.
- **Multi-step is keynote-demo-quality, not production-grade** for unconstrained input. Single-step tool calling (99.4%) and intent classification (99.4% LogReg / 96.7% gemma3 fallback) are production-grade.

#### Architectural-justification audit (2026-05-15)

The two Qwen-routing commits above (`3db64e4` concretize, `118b6a1` multi-step synthesis) were initially justified by observations that I later realised had been framed loosely — "different runs produce different outputs" elided the fact that the runs had different prompts. Before committing the architecture changes long-term, an A/B/C/D test was run on the same servers to falsify whether each commit was actually doing real work:

| Config | What's reverted | Multi-step chain (eval) | Q1 €252,000 | Q2′ €59,940 | Q3′ €359,640 | Spot-check pass |
|---|---|---|---|---|---|---|
| **T0** current | nothing | **78/80 (97.5%)** | ✅ | ❌ `3,496,500` (concretize hallucinated `3500`) | ✅ | **2/3** |
| **T1** | revert `118b6a1` only | 78/80 (97.5%) | ❌ `84,000` (1 yr not 3) | ❌ `349,650` (invented) | ✅ | **1/3** |
| **T2** | + revert `3db64e4` | 77/80 (96.2%) | ❌ "what would grow to in 3 yrs?" | ❌ `119,880` (canonical fewshot leak BACK) | ❌ `2,997/month annually` | **0/3** |
| **T3** | + revert `0c644fe` | 77/80 (96.2%) | ❌ hallucinated customer "HealthTech Solutions" | ❌ `119,880` (canonical leak) | ❌ `29,970 annually` (wrong) | **0/3** |

**The chain-shape eval (78/80 → 77/80) barely registers the difference — it scores tool names, not answer correctness.** The spot-check is what matters: T0 → T3 progressively degrades from 2/3 correct → 0/3 correct. Notably, **the `119,880` canonical fewshot leak comes back the moment synthesis runs on gemma3-1B (T2, T3)** — proving `c8d4eb5` (drop synth examples) needed `118b6a1` (route to Qwen) to actually take effect. Q1 ("3 years of spend") is only correct when synthesis is on Qwen — gemma3-1B forgets the "× 3" multiplier despite the SQL value being right.

The Q2′ failure in T0 (`3,496,500`) is the documented "concretize value hallucination" — Qwen routing improves the failure modes but doesn't fully eliminate them when the SQL didn't fetch the needed value. The chain-shape eval is therefore a poor proxy for real correctness; the architectural changes' impact is in *what numbers the user sees*, not in chain shape.

**Scaffolding status**: `NullExpressionResolver` and `NullSQLResolver` are active in production — Qwen3.5-4B FT handles all tool selection and argument generation natively. The earlier deterministic pre-routers (`expression_builder.py`, `sql_builder.py`) were retired in commit `68d52a5`; their narrative role is preserved here as the "before" half of the FT story.

#### Latency audit (2026-05-15)

After the T0-T3 quality audit above, the same A/B/C/D matrix was re-run to measure latency cost of the two Qwen-routing commits (RTX PRO 6000, CUDA, 80-query multi-step eval each):

| Config | What's reverted | Multi-step p50 latency |
|---|---|---|
| **T0** current production | nothing | 668 ms |
| **T1** revert `118b6a1` (synth back to gemma3-1B) | synth | 665 ms (−3 ms) |
| **T2** + revert `3db64e4` (concretize back to gemma3-1B) | + concretize | **718 ms (+50 ms)** |
| **T3** + revert `0c644fe` (unit-preservation rules) | + rules | 717 ms (+49 ms) |

**Surprising result:** the Qwen-routings are essentially free, possibly slightly faster on p50. Hypothesis: gemma3-1B's malformed concretize outputs (e.g. multi-sentence prose instead of `Calculate X * Y * Z`) trigger additional downstream tool-routing cycles in `select_tool`, so total work goes up despite individual gemma3 calls being cheaper than Qwen ones. The Qwen-routed concretize emits clean expressions that the next stage parses without retry.

**Cross-machine latency comparison — NOT AVAILABLE.** An earlier draft of this section compared today's numbers against `results/bench-2026-03-08-cuda.json`, but on inspection that bench file was generated with the **Bundeswehr scenario** (German military queries, different KB, different SQL schema, different image content) — the queries in the bench script were swapped to Nextera English in the public-release cleanup. So that comparison was apples-to-oranges and the inferred "+74-106% latency regression" was an artefact of the workload change, not the architecture. Retracted. The current bench JSON is at [`bench_2026-05-15_post-architecture_rtx.json`](bench_2026-05-15_post-architecture_rtx.json); to do a real same-scenario comparison we'd need to re-run a Bundeswehr-language bench against today's code (unlikely to be useful given that scenario has been removed) or compare against another Nextera-scenario bench run when one is available.

What we **can** say from the T0-T3 same-day data above: routing concretize + synth through Qwen FT is essentially free on p50 multi-step latency, possibly slightly faster than the gemma3-1B alternative. Per-step latencies in [§10 Quick Reference](#10-quick-reference) are the current measurements.

---

## 4. embeddinggemma-ft — Semantic Retrieval

### Already excellent out of the box

| Metric | Base Score |
|--------|-------|
| MRR@10 | 0.9533 |
| Recall@5 | 100% |

`google/embeddinggemma-308m` is a purpose-built embedding model with bidirectional attention (encoder-style, like BERT) and 768-dimensional output vectors. Unlike generative models repurposed for embeddings, this model was pretrained specifically for semantic similarity.

### Fine-tuning now produces measurable improvement

The original 37 synthetic (query, passage) pairs with `MultipleNegativesRankingLoss` converged at initialization — too few in-batch negatives to shift a model already at 95%+ MRR.

**The fix**: `finetune/gen_embeddinggemma_dataset.py` systematically generates **507 training examples** from the 13 knowledge base documents:
- **373 query-positive pairs** across pricing, features, integrations, security, support, FAQ, Meridian Health, and operations topics — using keyword, natural language, conversational, comparative, and scenario query styles
- **134 hard-negative triplets** (`{"anchor", "positive", "negative"}`) targeting confusable topic pairs: pricing tier confusion, compliance certs, deployment methods, RAG vs agents, tool disambiguation, support tiers, SDKs, and cross-domain ambiguity

Quality controls: Jaccard deduplication (0.85 threshold), eval leakage prevention (0.7 threshold against all 25 eval queries), format validation. Both formats are auto-detected by `train_embeddinggemma.py` — pairs use MNRL loss, triplets use TripletLoss.

**Results** (RTX PRO 6000 Blackwell, 5 epochs, 14.29s training):

| Metric | Baseline | Fine-Tuned | Change |
|--------|----------|------------|--------|
| MRR@10 | 0.9533 | **0.9800** | **+2.67%** |
| Recall@5 | 100% | 100% | — |

The GGUF conversion pipeline preserves the improvement — fine-tuned weights survive Dense projection stripping.

The eval suite covers 25 corpus passages and 25 query-document pairs (expanded from 13/13 to include Meridian Health ADR, compliance, disaster recovery, and observability topics). The additional test pairs serve as a more thorough benchmark; the actual ChromaDB knowledge base still contains the original 13 seeded documents from `data/business-documents/`.

> **Eval-corpus caveat.** The eval indexes 26 passages. A domain-tuned encoder on a 26-passage corpus has very little room to be wrong — the headline MRR@10=0.98 is therefore best read as a sanity check that the FT did not regress retrieval, not as a production benchmark. The live ChromaDB indexes ~120 chunks across 13 KB documents at runtime; how much of the +2.67pp uplift survives at that corpus size has not been measured separately. The `train_embeddinggemma.py` script also prints a "Final MRR" at end-of-training that is computed against an even smaller, in-distribution held-out slice of the training pairs — that figure is a training-loop diagnostic, not the published benchmark.

**For the demo narrative**: this is a two-part story — (1) the base model already scores 95%+ MRR out of the box on the small eval set, proving Google's purpose-built embedding model works for business-domain retrieval, and (2) targeted fine-tuning with hard negatives pushes MRR to 98% on that same set, demonstrating that domain-specific contrastive signal can shift even a strong baseline. The story is honest at the eval-corpus scale; it is not a generalisation to arbitrary production corpora.

### What made the difference

| Change | Impact | Status |
|--------|--------|--------|
| **Hard negatives** (triplet format) | Highest — forces the model to distinguish similar-but-wrong passages | **Done** — 134 triplets covering 9 confusion categories |
| **500+ training examples** | High — more in-batch negatives per step | **Done** — 507 examples (was 37) |
| **Diverse query styles** | High — keyword, natural, conversational, comparative, scenario | **Done** — 4-6 styles per fact slot |
| **Fewer epochs (5)** | Medium — `save_best_model` catches the peak | **Done** — 5 epochs, loss 3.152 |
| **Real user queries** from demo logs | Medium — after 100+ RAG interactions, `data_prep` extracts real phrasing | Available via `data_prep_embeddinggemma.py` |
| **Batch size 32+** | Medium — doubles in-batch negatives | Worth trying on GPU with 8+ GB VRAM |

### GGUF conversion caveat

The sentence-transformers trainer adds Dense projection layers (768 to 3072 to 768) that are **not included** in the GGUF — llama-server performs mean pooling directly on the backbone output. These projection layers carry some domain adaptation. To preserve them, use the model's `.encode()` method directly instead of llama-server for embedding.

`--pooling mean` flag is NOT supported in our llama.cpp version — omit it; pooling is auto-detected from the model architecture. Also download `tokenizer.model` from Google — sentence-transformers saves only `tokenizer.json`, not the SPM file needed for GGUF conversion.

---

## 5. Training Reproducibility — Eliminating Run-to-Run Variance

### The problem

With default LoRA settings, identical training runs on the same data produced accuracy ranging from 86.7% to 95% — a **8.3 percentage point spread**. This made it impossible to tell whether a change improved the model or just got lucky.

### Sources of non-determinism

| Source | Impact | Fix |
|--------|--------|-----|
| LoRA matrix initialization | **High** — random Kaiming init lands in different convergence basins | EVA initialization (data-driven SVD) |
| cuBLAS workspace algorithms | **Medium** — CUDA matrix multiplication uses non-deterministic algorithms | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| cuDNN algorithm selection | **Medium** — `benchmark=True` picks fastest algorithm per shape, varies between runs | `cudnn.deterministic=True`, `benchmark=False` |
| Gradient accumulation order | **Low** — floating-point non-associativity | `torch.use_deterministic_algorithms(True, warn_only=True)` |
| Data shuffling | **Low** — already controlled by `seed=42` | Fixed seed in `train_test_split` and `SFTConfig` |

### The fix: 4 techniques combined

```python
# 1. Deterministic CUDA
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2. EVA initialization (data-driven SVD on activations)
lora_cfg = LoraConfig(
    init_lora_weights="eva",
    eva_config=EvaConfig(rho=2.0),
    use_rslora=True,           # 3. rsLoRA: alpha/sqrt(r) scaling
    ...
)

# 4. Cosine LR scheduler
sft_config = SFTConfig(lr_scheduler_type="cosine", ...)
```

### Result

| Metric | Before (random init) | After (EVA + rsLoRA + cosine + deterministic) |
|--------|---------------------|-----------------------------------------------|
| Accuracy range | 86.7% - 95.0% | 91.7% - 95.0% |
| **Variance** | **8.3 pp** | **3.3 pp** |
| Token accuracy | 86.5% | 91.9% |
| Eval loss | 0.62 | 0.39 |
| rag_query stability | 95-100% | **100% (stable)** |
| tool_use stability | 95-100% | **100% (stable)** |

**Best-of-3 selection**: train with 3 different seeds, pick the best GGUF. LoRA adapters are ~4MB so storing 3 candidates costs nothing.

---

## 6. Training Data Design — Hard Lessons

### Train/eval contamination (the most dangerous bug)

We discovered that **18 out of 20** eval queries were present in the training data. This produced a perfect-looking 100% accuracy that was pure memorization.

**Detection**: programmatic overlap check comparing eval queries against all training inputs (case-insensitive, stripped).

**Fix**: 180 completely new eval queries (60 per class), zero overlap verified via Jaccard word-set similarity (threshold 0.7). Also verified that e2e test queries and demo showcase queries don't appear in training data.

**Best practice**: automated overlap check is now built into the eval pipeline (`check_eval_training_overlap()` in `eval_base.py`). Unit tests verify zero overlap on every run.

### Multi-task training prevents catastrophic forgetting

When training a model for classification AND synthesis:

| Training data | Classification | Synthesis | Why |
|---------------|---------------|-----------|-----|
| Classification only | High | **Broken** (empty) | Model learns "output one word then stop" |
| Synthesis only | **Broken** (17%) | Good | Model learns "generate text" — forgets short-label classification |
| + generation examples | Good | Works | Model sees both short and long outputs |
| + formatting examples | **Best** | **Best** | Exact prompt patterns from production code |

**Catastrophic forgetting works in both directions.** Training synthesis-only (`--task synthesis`, 201 examples) destroyed intent classification:

| Intent | Before (intent-only FT, v5) | After (synthesis-only FT) | Delta |
| ------------- | ----------------------- | ------------------------- | ---------- |
| Overall | **95.0%** | 16.7% | **-78.3%** |
| rag_query | 95% | 0% | -95% |
| tool_use | 100% | 45% | -55% |
| direct_answer | 90% | 5% | -85% |

The root cause was a bug in `train_gemma3.py`: `--task both` trained intent and synthesis **sequentially** (two separate LoRA adapters from the same base model), so the second adapter overwrote the first when merged. **Fix**: `--task both` now concatenates both datasets into a single training set, producing one unified LoRA adapter that covers all prompt patterns.

**Combined training (`--task both`) partially recovers** — 90.0% accuracy (100/95/75), a 5.0% drop from the intent-only 95.0% (v5). The 201 synthesis examples (10% of the 2,079 total) diluted the classification signal. See section 2 "Synthesis training experiments" for the full comparison and production recommendation.

**The key insight**: training examples must cover **all the prompt patterns** the model will see in production. If the model never sees a "Turn this tool result into a clear answer" prompt during training, it won't know how to respond to it at inference time. Conversely, if it never sees classification prompts, it forgets how to classify.

### Data balance

- **qwen3.5-4b**: 609 calculator + 642 sql_query = balanced, with hard negatives for boundary disambiguation
- **gemma3 intent**: 1697 classify + 166 generate + 15 format = 90:9:1 ratio works because the model's base instruct ability handles generation; it just needs periodic reminders not to forget
- **gemma3 synthesis**: 79 RAG + 69 tool formatting + 23 multi-step + 30 direct = 201 examples covering all 4 agent.py prompt patterns

### The "quality > quantity" principle

| Evidence | Lesson |
|----------|--------|
| 567 curated examples beat 3,481 template-expanded ones (77% vs 43%) | Diverse, well-chosen examples teach semantic patterns; noisy expansion teaches surface patterns |
| Adding 60 direct_answer examples **decreased** gemma3 accuracy by 3.3% | Flooding an underrepresented class can shift the boundary and destabilize other classes |
| 40 targeted sql_query + hard-negative examples improved qwen3.5-4b by 5% | Surgical additions at the decision boundary are more effective than broad expansion |

---

## 7. GGUF Conversion — Gotchas That Cost Hours

### Use the SPM path, not BPE

```bash
# CORRECT — SPM path produces tokenizer.ggml.model='llama' with correct token IDs
python convert_hf_to_gguf.py MODEL_DIR --outfile output.gguf --outtype f16

# WRONG — BPE path fails for Gemma3 (missing hash in llama.cpp, see issue #19152)
```

### tokenizer_config.json must come from Google

`GemmaTokenizer.save_pretrained()` omits `add_bos_token` and `added_tokens_decoder`, which llama.cpp needs for correct tokenization. Always download the original:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("google/gemma-3-1b-it", "tokenizer_config.json")
```

### chat_template must be injected

Not saved during LoRA training. Without it, `/v1/chat/completions` returns errors:

```python
tc["chat_template"] = "{{ bos_token }}{% for message in messages %}..."
```

For qwen3.5-4b: the native template (`<start_function_declaration>` format) is stored inside the tokenizer OBJECT, not the JSON file. Load via `AutoTokenizer.from_pretrained()` and inject `tok.chat_template` into `tokenizer_config.json`.

### QLoRA weights don't convert

```
NotImplementedError: Quant method is not yet supported: 'bitsandbytes'
```

**Fix**: omit the `--qlora` flag (QLoRA is off by default) to use full-precision LoRA, or merge adapter into base model and save in float16 before converting.

### Don't use `ollama create` from safetensors

Ollama's built-in GGUF converter for Gemma3 silently produces incorrect token vocabulary ordering, which breaks fine-tuned models. Always use llama.cpp's `convert_hf_to_gguf.py` directly.

---

## 8. llama-server Configuration

### Temperature management

```bash
# WRONG — locks temperature at server level, overrides all API requests
launch "inference" "$PORT" "$GGUF" --temp 0.0

# CORRECT — let the API control temperature per-request
launch "inference" "$PORT" "$GGUF"
```

Classification needs temp=0.0 (deterministic), but synthesis needs temp=0.7 (creative). Server-level `--temp 0.0` prevents the API from overriding.

### Achieving true determinism (greedy decoding)

`temperature=0.0` alone does NOT guarantee deterministic output in llama.cpp. You need all four parameters:

```python
# In the OpenAI SDK call:
temperature=0.0,    # disable sampling
seed=42,            # fix RNG
top_p=1.0,          # disable nucleus sampling
extra_body={"top_k": 1}  # force greedy (single highest-probability token)
```

This is implemented as `deterministic=True` in `SmallLanguageModelClient.generate()` and `call_function()`. Applied to intent classification, query rewriting, RAG synthesis, and tool routing. Additionally, all llama-server instances run with `--parallel 1` (mandatory — the default is now auto/multi-slot) to prevent batch-size-dependent floating-point variation across requests. E2e determinism tests hard-assert intent classification and tool selection across 5 runs per query (65 total inferences); response text determinism is reported but not asserted (synthesis wording may vary without affecting routing correctness).

See: [Achieving Deterministic Inference with Local LLMs](content/deterministic-local-inference.md) for the full technical deep-dive, or the original [LinkedIn article](https://www.linkedin.com/pulse/achieving-determinism-local-slm-llm-deployments-using-christian-weyer-quoxe/).

### qwen3.5-4b requires specific flags

```bash
launch "qwen3.5-4b" "$PORT" "$GGUF" --temp 1.0 --top-k 64 --top-p 0.95 --jinja
```

The `--jinja` flag enables the native Qwen tool caller chat template with function declaration tokens. Without it, the model receives a generic Gemma template and can't produce function calls.

### Build with static linking

```bash
cmake -DBUILD_SHARED_LIBS=OFF ...
```

Dynamic linking produces `libmtmd.so.0 not found` at runtime. Static builds are self-contained and portable across machines (important for MBP demo deployment).

### Cross-platform compatibility

The demo stack runs on both Linux (CUDA) and macOS (Metal). Key compatibility points:
- `build_llama.sh` auto-detects CUDA vs Metal vs CPU
- `--n-gpu-layers 999` works with both CUDA and Metal backends
- Port collision detection uses `lsof` (available on both Linux and macOS)
- All Python dependencies are platform-agnostic (no CUDA-specific packages in `requirements.txt`)

---

## 9. What We'd Do Differently

1. **Start with proper train/eval split from day one** — never train on eval data, even accidentally. The most dangerous metric is a perfect one.
2. **Include generation examples from the start** — don't discover catastrophic forgetting after 10 training runs.
3. **Use the instruct model from the beginning** — base models can classify but can't synthesize. LoRA on `-it` preserves both abilities.
4. **Test e2e quality alongside classification accuracy** — 100% classification means nothing if synthesis returns empty strings.
5. **Verify training data format matches inference format** — the biggest qwen3.5-4b bug was invisible in training metrics but caused 0% accuracy at inference.
6. **Use EVA + rsLoRA from the start** — eliminates the "lucky seed" problem that makes results unreproducible.
7. **Design task boundaries before choosing models** — reducing a 3-way decision to 2-way was worth more than any amount of training data.
8. **Invest in hard negatives, not more examples** — 40 targeted boundary examples improved accuracy more than 2,000 generic ones.

---

## 10. Quick Reference

### Reproduction commands

```bash
# Train gemma3-ft (intent fallback path — 93.3% in current production; v5 reached 95.0% when
# gemma3-ft was the sole classifier, before the LogReg primary head landed)
python -m finetune.train_gemma3 --task intent --epochs 7 --lr 5e-5
# Train both (concatenated intent + synthesis — 90.0% accuracy, -5.0% vs intent-only)
python -m finetune.train_gemma3 --task both --epochs 7 --lr 5e-5
# Train synthesis only (WARNING: destroys intent classification — experimental only)
python -m finetune.train_gemma3 --task synthesis --epochs 7 --lr 5e-5
# Train Qwen3.5-4B FT v8 (QLoRA r=16, 2-tool routing + multi-turn, 3 epochs)
python -m finetune.train_qwen35_toolcalling

# Train embeddinggemma-ft (contrastive — optional, already at ceiling)
python -m finetune.train_embeddinggemma

# Convert all to GGUF
bash finetune/convert_gemma3_to_gguf.sh
bash finetune/convert_qwen35_to_gguf.sh
bash finetune/convert_embeddinggemma_to_gguf.sh

# Serve with fine-tuned models
bash scripts/start_servers.sh --bg --ft

# Evaluate (60 intent + 40 tool + 25 retrieval queries, zero overlap)
python -m finetune.eval_gemma3 --save results/finetuned_gemma3.json
python -m finetune.eval_tool_routing --save results/finetuned_tool_routing.json
python -m finetune.eval_embeddinggemma --save results/finetuned_embeddinggemma.json
```

### Key numbers

| Metric | Value |
|--------|-------|
| gemma3 intent accuracy _(generative fallback path)_ | **96.7%** (180-query eval, post-2026-05-15 retrain; v5 era was 93.3% pre-data-correction) |
| LogReg intent accuracy _(primary path)_ | **99.4%** (180-query eval, ~5ms, deterministic; v5 era was 97.2% on the pre-data-correction embedding space) |
| Qwen3.5-4B FT v9 tool accuracy _(current production)_ | **99.4%** routing / **100%** SQL valid execution / **95.0%** calculator exact (160-query single-step). Multi-step: **97.5%** (78/80) chain shape — deterministic (verified byte-identical across 3 back-to-back runs). See §3b. |
| embeddinggemma MRR@10 | **98%** (base 95.3% → fine-tuned 98% with 507 examples + hard negatives — 25 queries) |
| Training variance (gemma3) | **3.3%** across seeds + hardware (was 8.3%) |
| Total training time (all 3) | ~6 min on NVIDIA RTX PRO 6000 |
| Total inference VRAM | ~3 GB for all 3 models |
| Combined parameter count | ~9.3B (gemma3-1B + gemma3-4B + Qwen3.5-4B + EmbeddingGemma 308M) + ~150KB LogReg head |

### Hardware

All training and inference runs on:
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q (~96 GB VRAM)
- **CUDA**: 13.1
- **Training time**: gemma3-ft ~5 min, qwen3.5-4b-ft ~1 min, embeddinggemma ~10 min
- **Inference verified on** (all optimized builds, llama.cpp b8384, isolated benchmarks):
  - NVIDIA RTX PRO 6000 (CUDA 13.1, `GGML_CUDA_F16`, `--no-mmap --flash-attn on`) — **465 ms** overall p50
  - MacBook Pro M5 Max 128GB (Metal, `GGML_METAL_EMBED_LIBRARY`) — **1121 ms** overall p50
  - NVIDIA DGX Spark GB10 (CUDA 13.0, ARM SVE2 flags, `--no-mmap --flash-attn on`) — **2315 ms** overall p50
  - MacBook Pro M3 Max 128GB (Metal, llama.cpp b8117) — median ~650 ms RAG, ~170 ms calculator
  - MacBook Pro M3 Max 128GB (CPU-only, `--cpu` flag) — median ~2 s RAG, ~320 ms calculator
- **Build optimizations**: auto-detected per platform in `build_llama.sh` and `start_servers.sh`. See [EVAL_RESULTS_2026-04-05.md](EVAL_RESULTS_2026-04-05.md) for full details.

---

## 7. Eval/Training Data Decontamination (2026-04-04)

All eval test sets were audited for overlap with their corresponding training JSONL files using Jaccard word-set similarity (`|A∩B| / |A∪B|` on lowercased word sets). Any eval query appearing verbatim or near-verbatim in training data inflates eval metrics — the model is being tested on what it memorized, not what it learned.

### Methodology

- **Tool**: `check_eval_training_overlap()` from `finetune/eval_base.py`
- **Threshold**: Jaccard >= 0.7 (0.6 produces false positives from German sentence structures like "Was ist der Unterschied zwischen X und Y?")
- **Scope**: All 12 eval scripts × all relevant training JSONL files, across every shipped scenario

### Results

| Eval Script | Queries | Contaminated (before) | After cleanup |
|-------------|---------|----------------------|---------------|
| eval_tool_routing | 160 | 57+ queries | 0 |
| eval_response_quality | 30 | 6 queries | 0 |
| eval_multi_step | 80 | 20 queries | 0 |
| eval_adversarial | 60 | 0 (already clean) | 0 |
| eval_vision | 10 | 0 | 0 |
| eval_ocr | 22 | 0 | 0 |
| eval_embeddinggemma | 25 | 0 | 0 |

**Total: 80+ contaminated queries replaced** with rephrased variants that test the same capability without appearing in training data.

### CI Guard

`tests/unit/test_eval_overlap.py` covers all eval/training pairs. Any future training data addition that overlaps with an eval query will fail CI.

### Eval Data Externalization (2026-04-05)

All eval test sets with >20 entries were moved from inline Python literals to external files in `data/eval-data/`. This separates eval logic from eval data, making test sets easier to edit, diff, and lint independently.

Eval files are named with a per-scenario suffix (e.g. `eval_gemma3_<scenario>.jsonl`); the Nextera reference set is shipped in this repo.

| File pattern | Entries | Format |
|------|---------|--------|
| `eval_gemma3_<scenario>.jsonl` | 180 | JSONL |
| `eval_tool_routing_<scenario>.jsonl` | 160 | JSONL |
| `eval_calculator_<scenario>.json` | 80 | JSON (query→answer dict) |
| `eval_adversarial_<scenario>.jsonl` | 60 | JSONL |
| `eval_response_quality_<scenario>.jsonl` | 30 | JSONL |
| `eval_embeddinggemma_corpus.json` / `_pairs.jsonl` | 25 / 25 | JSON + JSONL |
| `eval_ocr.jsonl` | 29 | JSONL |

Loaders: `load_eval_jsonl()` and `load_eval_json()` in `finetune/eval_base.py`.

Small sets kept inline: `eval_extraction` (5 entries).

### Eval Coverage Pattern

For each new scenario added under `scenarios/<name>.json`, 11 eval scripts cover:
intent classification, tool routing, multi-step decomposition, calculator,
SQL execution, RAG retrieval, RAG ground-truth, response quality, adversarial
robustness, vision, and OCR — each loading per-scenario JSONL files from
`data/eval-data/`.

`finetune/eval_rag_groundtruth.py` is the per-document accuracy breakdown:
each query targets a specific source document with expected keywords drawn
from the indexed content. Decontamination is checked by `tests/unit/test_eval_overlap.py`
with Jaccard >= 0.7 — any future training data addition that overlaps with an
eval query will fail CI.

### Full Numbers Run (2026-04-05, RTX PRO 6000)

> **Updated 2026-05-15 retrain summary.** After commit `62addc1` corrected the training data
> (`FROM revenue` → `FROM sales`, `€1,499` → `€999`, real 2024 quarterly numbers), all four FT
> models were retrained. Headline deltas vs the 2026-04-05 baseline below:
>
> | Metric | 2026-04-05 (v8 era) | 2026-05-15 (v9 era) | Δ |
> |---|---|---|---|
> | LogReg intent (primary) | 97.2% | **99.4%** | +2.2pp |
> | gemma3-ft intent (fallback) | 93.3% | **96.7%** | +3.4pp |
> | Qwen tool routing | 99.4% | **99.4%** | ≈0 |
> | Qwen SQL execution validity | 100% (caveat: training had `FROM revenue` bug) | **100%** (79/79, valid SQL throughout) | clean |
> | Qwen multi-step tool chain | 97.5% (78/80) | **97.5%** (78/80) | match v8 chain-shape number AND gain 100% SQL valid execution (v8 era had `FROM revenue` crashes the eval didn't catch). Deterministic — verified byte-identical across 3 back-to-back runs. See §3b for the architectural commits that closed the gap. (Earlier draft of this row said v8 was 96.2% (78/80); 78/80 = 97.5% exactly — internal arithmetic typo, corrected.) |
> | Embedding retrieval MRR | 0.98 | **0.98** | flat (FT was skipped per RETRAIN_NOTES) |
> | Vision (n=10) | 100% | **100%** | flat (no retrain) |
>
> Architectural changes alongside the retrain: `concretize_step` and multi-step `synthesize_response`
> both routed through Qwen FT (FUNCTION role) instead of gemma3-1B. See [§3b](#3b-qwen35-4b-ft-v8--current-production-2026-03-19)
> "Architectural changes that landed alongside the v9 retrain" for the full commit list and the
> known-limitations callout.

First complete evaluation after Phase 11–13 (decontamination, coverage gaps, RAG deepening). The numbers below are the Nextera reference scenario.

#### Test Suite — zero failures

| Suite | Result |
|-------|---------|
| Unit tests | 1254 passed |
| Integration tests | 180 passed |
| E2E tests | 66 passed |
| **Total** | **1500** |

#### Eval results

| Eval | Result | Assessment |
|------|---------|------------|
| **Intent (gemma3-ft 1B, fallback path)** | 93.3% | direct_answer over-routes to rag_query — the dominant remaining error |
| **Intent (LogReg, primary path)** | 97.2% | Deterministic, ~10 ms, production path |
| **Tool routing (Qwen3.5-4B FT v8)** | 99.4% | 2 sql_query → calculator misroutes on ambiguous queries |
| **Calculator expressions** | 92.5% | Most failures are percentage-vs-decimal format mismatches (e.g. "0.275" → "27.5") |
| **Adversarial (generative-only baseline)** | 70.0% | Generative model alone leaks ~30% of adversarial queries to tool_use |
| **Adversarial (full pipeline)** | 93.3% | LogReg + 30-regex injection pre-filter + gibberish/non-ASCII filters + canned refusal |
| **Vision** | 100% | All 10 keyword-checked image queries |
| **Embedding retrieval** | MRR 0.98, R@5 100% (on 26-passage eval set) | Scenario-independent FT EmbeddingGemma. See §4 eval-corpus caveat — production KB is ~120 chunks and the uplift at that scale has not been measured. |
| **Response quality** | 95.7% (46 q) | 2 tool_use grounding failures |
| **Multi-step decomposition** | 97.5% | 11 regex patterns + LLM-decomposition fallback |
| **Multi-step tool chain** | 96.2% (Qwen3.5-4B FT v8) | 3 chain failures on ambiguous decomposition |
| **Expression pipeline (E2E)** | 92.5% | ~5% are decimal/percentage format mismatches |
| **RAG ground-truth (80 q)** | **78.8% (63 / 80)** | Expanded from 20 → 80 queries for statistical validity. 95% CI: 68–87%. Intent rebalancing + 4B FT + label cleanup. |

#### Known Issues — Status After Fixes (2026-04-05)

| Issue | Before | After | Fix Applied |
| ----- | ------ | ----- | ----------- |
| **1. RAG Ground-Truth routing** | 65.0% (13/20) | **90.0% (18/20)** | Two-part fix: (a) 4B synthesis FT with `Gemma3ForConditionalGeneration` + extractive QA training data; (b) intent training data rebalancing — added 50 rag_query examples + removed 47 corrupted labels + relabeled 134. See deep analysis below. |
| **2. Generative intent path imbalance** | 45% rag_query | 45% rag_query | **Accepted** — training data imbalance on the generative path. LogReg (97.2%) handles production traffic; the 1B generative classifier is a load-time fallback only. Re-tuning would require rebalancing `gemma3_intent.jsonl` and retraining the 1B FT. Low priority. |

**Root cause analysis (RAG ground-truth fix):** The intent training data had a vocabulary distribution bias — domain-fact questions like "What is the price of X?" were classified as `direct_answer` because the early training data treated product-knowledge questions as conversational. Combined with corrupted labels (literal answer strings instead of valid intent enums) and a `"How much"` distribution of only 7% rag vs 93% tool, the model systematically routed pricing/feature questions away from RAG. The rebalancing pass added rag_query examples with document-reference signals ("according to the pricing page", "as documented") to teach the classifier that the distinguishing signal is **document-lookup intent**, not the surface form of the question.

#### Deep Analysis: RAG Ground-Truth 65% → 90%

**Problem decomposition.** All 7 original failures were traced through the full pipeline using `scripts/analyze_intent_classifier.py`. Every failure was **routing → direct_answer** (never tool_use):

| Query | Intent | Should be |
| --- | --- | --- |
| What is the annual price for the Starter plan? | direct_answer | rag_query |
| What is the starting monthly price for the Enterprise plan? | direct_answer | rag_query |
| How long does fine-tuning typically take for 500 examples? | direct_answer | rag_query |
| Which file types can be ingested by the RAG pipeline? | direct_answer | rag_query |
| What model families does the agentic pipeline support? | direct_answer | rag_query |
| What is the minimum RAM requirement for small models? | direct_answer | rag_query |
| How quickly are critical security patches delivered? | direct_answer | rag_query |

**No contamination**: 0 violations at Jaccard 0.6 against eval queries. All 23 CI overlap tests pass. The new examples are general patterns, not paraphrases of eval queries.

**Result: 65% → 90% (18/20).** Remaining 2 failures sit at the rag/tool boundary — text-classification alone cannot decide between "look this up in a document" and "compute this from the database" when the phrasing is genuinely ambiguous.

#### Alternatives for Further Improvement

Three options exist if 90% is insufficient:

1. **Accept current results** — the remaining failures are genuinely ambiguous queries where text-based classification cannot determine the correct data source. This is the current choice for the reference scenario.

2. **Confidence-threshold fallback** — when LogReg confidence is borderline (e.g., rag probability > 0.3 but another class wins), try RAG first. If RAG returns relevant chunks with high similarity, use them; otherwise fall through to the predicted class. This would catch borderline cases without retraining. **Trade-off**: adds ~200 ms latency for borderline queries due to speculative RAG search.

3. **Add more rag_query training examples** — continue rebalancing the vocabulary toward rag_query. **Trade-off**: risks making rag_query too dominant, causing real tool_use queries ("How many customers joined in Q3?") to be misclassified as rag_query. Every shift toward rag_query for ambiguous patterns pulls genuinely numeric DB queries along with it.

**Recommendation**: Option 2 (confidence-threshold fallback) is the most principled approach. It resolves ambiguity at runtime rather than at training time, preserving the clean tool_use / direct_answer boundary for unambiguous queries.

#### Latency (M5 Max p50)

| Intent Path | Median | Bottleneck |
|-------------|---------|------------|
| rag_query | 464 ms | `synthesize_response` 422 ms |
| tool_sql | 626 ms | `select_tool` 396 ms |
| tool_calc | 382 ms | balanced |
| direct | 98 ms | `direct_response` 94 ms |
| image_query | 1,692 ms | `analyse_image` (mmproj on 4B) |
| **Overall** | **559 ms** | sub-second median |

Scenario-independent steps: intent classification (4 ms), vector search (10 ms), tool execution (<1 ms). Synthesis is the dominant latency driver.

---

## Faithfulness eval methodology change (2026-05-16)

The `eval_rag_groundtruth` test was the weakest link in the project's eval rigor. It is now fixed; the headline number dropped 11 pp once the scoring became honest, but the sub-metric story is more interesting than the headline.

### The change

**Before (`check_keywords`):**

```python
def check_keywords(response, expected):
    return any(kw.lower() in response.lower() for kw in expected)
```

ANY expected keyword → pass. A response saying *"Enterprise costs €3,500/month and includes the new Quantum Module"* (faithful price + fabricated feature) graded **identically** to *"Enterprise costs €3,500/month."* (faithful only). Coverage and fabrication were both invisible.

**After (`_coverage` + `_value_grounded` in [`eval_rag_groundtruth.py`](../../finetune/eval_rag_groundtruth.py)):**

A query is correct only when:

1. **Coverage = 100%** — every expected keyword from the ground-truth set appears in the response (substring, case-insensitive).
2. **value_grounded** — every numeric value in the response is sourced from either the expected-keyword set OR the retrieved `vector_search` context. No fabricated numbers.

Sub-metrics (`full_coverage`, `any_coverage`, `value_grounded`) are surfaced separately so a regression points at the failing axis.

The old "any keyword" scoring is preserved as `--legacy` for back-compat plots.

### Headline result (2026-05-16, scenario `nextera`, 80 queries)

| Machine | Strict (new headline) | Legacy "any keyword" | Δ |
|---|---|---|---|
| RTX/CUDA | **75.0%** (60/80) | 86.2% (69/80) | −11.2 pp |
| MBP/Metal | **67.5%** (54/80) | 80.0% (64/80) | −12.5 pp |

Sources: `results/rtx_rag_gt_strict_20260516T101130.json`, `results/mbp_rag_gt_strict_20260516T091218.json`.

### The decomposition that matters

| Failure axis | RTX | MBP |
|---|---|---|
| `coverage < 100%` (model omits expected facts) | 25.0% | 32.5% |
| `value_grounded == False` (model invents numbers) | **7.5%** | **7.5%** |

Inverted: **value_grounded = 92.5% on both machines.** The model rarely fabricates specific values — it more often *omits* them.

For a privacy-pitched local-RAG agent, omission is the dramatically better failure mode:

- Omission is detectable from the response itself — the missing fact is visible by inspection against the source document.
- Fabrication is only catchable against ground truth (which production users don't have).

The 7.5% fabrication rate is the real number that needs work. The 25–32.5% omission gap is fixable with better synthesis prompting or a larger synthesis model, and is the natural target for the next round of "data lever" work.

### Why the headline drop is good news, not bad

The previous 81–86% number was telling a story that wasn't true. A response that included *one* expected keyword and invented two facts of its own was graded the same as a faithful summary. The strict scoring is the number we can defend in public — anyone running `python -m finetune.eval_rag_groundtruth` against this repo gets the honest baseline. The `--legacy` flag is one keystroke away for back-compat plots.

### What this means for the "+27 pp data lever" slide claim

The Three-Levers slide cites "65% → 92% RAG ground-truth (PROJECTED, with one more round of label cleanup)". Against the honest baseline, the claim is now **75% → 92% on RTX** or **67.5% → 92% on MBP** — the data lever is **8–17 pp**, not the original **+27 pp**. The 92% projection is the same target; what changed is the honest starting point. The presenter note in `slides/slides_v1.3.md:1727` reflects this.

---

### References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [rsLoRA: A Rank Stabilization Scaling Factor](https://arxiv.org/abs/2312.03732) — Kalajdzievski, 2023
- [EVA: Explained Variance Adaptation](https://arxiv.org/abs/2410.07170) — Paischer et al., 2024
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka
- [Efficient Fine-Tuning with LoRA: Optimal Parameter Selection](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) — Databricks
- [HuggingFace PEFT — LoRA Developer Guide](https://huggingface.co/docs/peft/developer_guides/lora)
