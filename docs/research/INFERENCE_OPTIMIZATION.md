# Inference Optimization — Techniques, Patterns, and Opportunities

> Research date: 2026-04-12. Focus: local SLM/LLM inference on consumer/workstation GPUs (Metal, CUDA).

---

## What We Already Have

| Technique | Status | Implementation |
|-----------|--------|---------------|
| Flash Attention | **Active** | `--flash-attn on` on CUDA |
| GPU offloading | **Active** | `--n-gpu-layers 999` (all layers on GPU) |
| Prompt caching | **Active** | `--swa-full` + `--cache-reuse 256` + `n_keep` per stage |
| Per-stage token limits | **Active** | Classify=20, rewrite=40, RAG=600, tool=800, etc. |
| Parallel slots | **Active** | `--parallel` with `-kvu` on CUDA |
| Model quantization | **Active** | Q8_0 for small models, Q4_K_M for vision |
| Unified KV cache | **Active** | `-kvu` when parallel > 1 on CUDA |
| Batch size tuning | **Active** | `--ubatch-size 2048` for embedding model |

---

## Optimization Techniques Catalog

### 1. Speculative Decoding

**What:** A small "draft" model proposes N tokens ahead; the target model verifies them in one forward pass. Since LLM inference is memory-bandwidth bound, verifying N tokens costs roughly the same as generating 1.

**Impact:** 1.5-3x tokens/sec improvement. Best when draft model is from the same family and much smaller.

**Our opportunity:** 
- Use Gemma 4 E2B (2.3B) as draft model for Gemma 4 E4B (4.5B) tool-calling/synthesis
- Or use qwen3.5-4b as draft for gemma-3-1b classification
- llama.cpp supports this via `--model-draft` flag

**Effort:** Low — just add a draft model to start_servers.sh. No code changes.

**Risk:** Draft model must share tokenizer with target. Same-family requirement limits options. Acceptance rate varies by task.

**Verdict: HIGH PRIORITY** — easy win for the synthesis step (our slowest per-token stage).

---

### 2. KV Cache Quantization

**What:** Store KV cache entries in lower precision (Q8_0, Q4_0, or even 3.5-bit TurboQuant) instead of FP16. Reduces memory per token, allowing longer contexts or more parallel slots.

**Impact:** 
- Q8_0: ~50% memory reduction, minimal quality loss
- Q4_0: ~75% memory reduction, noticeable on long context
- TurboQuant 3.5-bit: 4.9x compression vs FP16, new in llama.cpp (ICLR 2026)

**Our opportunity:**
- Already in TODO: `--cache-type-k q8_0 --cache-type-v q8_0`
- Would free VRAM for running more models simultaneously on RTX (96GB)
- Enables running 26B A4B MoE alongside all other models

**Effort:** Trivial — two flags in start_servers.sh. Test quality on eval suite.

**Risk:** Slight quality degradation on very long contexts. Unlikely to matter at our context lengths (~2-4K effective).

**Verdict: HIGH PRIORITY** — already planned, zero code changes, measurable VRAM savings.

---

### 3. APEX Quantization (MoE-Specific)

**What:** Assigns different precision per tensor type and layer position in MoE models. Routed experts get aggressive compression (Q3-Q6_K), shared experts stay Q8_0, attention layers at Q6_K. Edge layers (first/last 5) get higher precision than middle layers.

**Impact:** On Qwen3.5-35B-A3B: 21.3GB APEX Quality **beats** 45.3GB Unsloth UD-Q8_K_XL on HellaSwag (83.5% vs 82.5%) while being 2x smaller. APEX Mini at 12.2GB reaches 74.4 tok/s.

**Our opportunity:**
- Directly applicable to Qwen3.5-35B (MoE comparison column) and Gemma 4 26B A4B
- Could shrink MoE from ~20GB to ~12-16GB with equal or better quality
- Faster inference due to better cache utilization at smaller size

**Effort:** Medium — need to generate APEX-quantized GGUFs. Uses stock llama.cpp (no code changes needed), just per-layer tensor type assignments.

**Risk:** Need to run evals on APEX quants. Research is new (April 2026).

**Verdict: MEDIUM PRIORITY** — significant for MoE column, but only affects the comparison path, not core pipeline.

---

### 4. Brevity Fine-Tuning (OogaBoogaLM approach)

**What:** Fine-tune the model to produce shorter, more direct responses *in the weights* rather than via system prompt instructions. QLoRA on ~500 examples, 10 minutes of training.

**Impact:** 5.7x token reduction (82.8 vs 473.7 tokens mean output) with preserved quality. Every saved output token = saved inference time.

**Our opportunity:**
- **Directly applicable to synthesis step.** Our format_response model (Gemma/Qwen) often generates verbose output. Training conciseness into weights means:
  - Faster responses (fewer tokens to generate)
  - Less reliance on "be concise" system prompt instructions
  - More consistent output length
- **Cloud orchestrator problem:** GPT-5.4 dumps verbose per-record details (18s). Can't fine-tune GPT, but for local models this approach would help.
- Could also apply to RAG synthesis — train model to summarize rather than regurgitate context.

**Effort:** Low — QLoRA fine-tune with existing infrastructure (Unsloth). Need ~500 curated concise examples per scenario.

**Risk:** Too aggressive compression loses important details. Need careful example curation.

**Verdict: HIGH PRIORITY** — reduces the dominant cost (output token generation) at the source.

---

### 5. Structured Output / Constrained Decoding (GBNF)

**What:** Force model output to conform to a grammar (JSON schema, enum values, etc.) at the token level. Prevents invalid output, reduces retries.

**Impact:** Near-zero overhead with modern engines (XGrammar, llguidance). Eliminates JSON parse failures. Can speed up generation by constraining the vocabulary at each step.

**Our opportunity:**
- **Tool calling step** — force valid JSON for SQL queries and calculator arguments
- **Intent classification** — constrain to valid intent enum values
- **Already partially done** — Qwen FT is trained on structured output. But GBNF grammar would guarantee validity without relying on fine-tuning alone.

**Effort:** Low-medium — define GBNF grammars for each step, pass via llama-server API.

**Risk:** Overly tight grammars can reduce quality (model fights the constraint). Leading whitespace issue: 5-10% quality drop if not handled.

**Verdict: MEDIUM PRIORITY** — useful for reliability, moderate speed benefit.

---

### 6. Continuous Batching

**What:** Insert new requests as soon as any slot completes, instead of waiting for all slots to finish. Keeps GPU at ~100% utilization.

**Impact:** 3-10x throughput improvement under concurrent load.

**Our opportunity:**
- Currently using `--parallel` with slot-based scheduling
- Only matters under concurrent users (keynote demo is single-user)
- Would matter for production deployment

**Effort:** Low — llama.cpp supports it natively.

**Risk:** None.

**Verdict: LOW PRIORITY** — no benefit for single-user demo. Important for production.

---

### 7. Host-Memory KV Cache Offloading

**What:** Spill KV cache from GPU VRAM to system RAM. Enables longer contexts and more parallel slots without running out of VRAM.

**Impact:** Extends effective context window significantly. Slower than pure GPU cache but much cheaper.

**Our opportunity:**
- RTX PRO 6000 has 96GB VRAM (not a constraint today)
- M5 Max has 128GB unified (VRAM and RAM are the same)
- Useful if we add 26B A4B MoE alongside existing 5 models

**Effort:** Low — llama-server flag.

**Risk:** Slight latency increase for cache misses.

**Verdict: LOW PRIORITY** — we're not VRAM-constrained yet.

---

### 8. MoE Expert Offloading

**What:** Keep active experts on GPU, offload inactive experts to CPU/RAM. Since MoE only activates 4-8 of 128 experts per token, most weights can live in RAM.

**Impact:** HOBBIT system shows 9.93x speedup over naive offloading. MoEpic saves ~50% GPU cost with 37% lower latency.

**Our opportunity:**
- Enables running Gemma 4 26B A4B on smaller GPUs (even 16GB consumer cards)
- RTX PRO 6000 doesn't need this (26B fits in 96GB easily)
- But M5 Max running multiple models simultaneously might benefit

**Effort:** Medium — llama.cpp `--n-cpu-moe` flag exists. Needs tuning.

**Risk:** CPU-bound token generation for cache-miss experts.

**Verdict: LOW PRIORITY** — we have enough VRAM. Relevant if targeting consumer hardware.

---

### 9. Token Budget Management

**What:** Enforce a global token budget across the entire pipeline (classify + route + execute + synthesize). If early steps are cheap, later steps get more budget. If context is long, auto-reduce synthesis length.

**Impact:** Prevents runaway generation. More predictable latency. Better UX.

**Our opportunity:**
- Currently per-stage max_tokens (20/40/600/800) are static
- A dynamic budget could: reduce synthesis tokens when SQL returns many rows, increase when answer is complex
- Could also cap total pipeline tokens (e.g., 1200 total) and allocate dynamically

**Effort:** Medium — needs pipeline-level token tracking and dynamic limit adjustment.

**Risk:** Complexity. Hard to tune right.

**Verdict: MEDIUM PRIORITY** — would improve latency consistency but complex to implement well.

---

### 10. Prompt Compression / Context Distillation

**What:** Compress long system prompts or RAG context into fewer tokens while preserving semantic content. Approaches: LLMLingua, prompt distillation, or training a compressor model.

**Impact:** 2-5x reduction in prompt tokens → proportional reduction in prefill time and KV cache usage.

**Our opportunity:**
- BW scenario prompts are 20-30% longer than Nextera (German compound words)
- RAG context can be large when multiple documents match
- Compressing the RAG context before synthesis would speed up the slowest step

**Effort:** High — needs a compression model or careful prompt engineering.

**Risk:** Lossy compression may drop important context.

**Verdict: LOW PRIORITY** — our prompts are already reasonably sized. Diminishing returns.

---

## Priority Matrix

| Technique | Impact | Effort | Priority | Notes |
|-----------|--------|--------|----------|-------|
| **Speculative decoding** | High (1.5-3x tok/s) | Low | **P1** | Same-family draft model |
| **KV cache quantization** | Medium (50% VRAM) | Trivial | **P1** | Already in TODO |
| **Brevity fine-tuning** | High (2-5x fewer tokens) | Low | **P1** | OogaBoogaLM approach |
| **APEX MoE quantization** | High (2x smaller, equal quality) | Medium | **P2** | For MoE comparison column |
| **Constrained decoding (GBNF)** | Medium (reliability + speed) | Low-Med | **P2** | For tool calling / classification |
| **Token budget management** | Medium (consistency) | Medium | **P2** | Dynamic per-pipeline allocation |
| **Continuous batching** | High (3-10x throughput) | Low | **P3** | Only for production, not demo |
| **Host-memory KV offload** | Low (more headroom) | Low | **P3** | Not VRAM-constrained yet |
| **MoE expert offloading** | Low (smaller GPU support) | Medium | **P3** | Not needed on RTX 96GB |
| **Prompt compression** | Low-Med (faster prefill) | High | **P3** | Prompts already reasonable |

---

## Recommended Next Steps

### Quick wins (can do this week)
1. **KV cache quantization** — Add `--cache-type-k q8_0 --cache-type-v q8_0` to all generation servers in start_servers.sh. Run eval suite to verify no quality regression.
2. **Speculative decoding** — Test gemma-3-1b as draft for Qwen3.5-4B tool calling. Measure tok/s improvement on RTX.

### Medium-term (next sprint)
3. **Brevity fine-tuning** — Create 500 concise synthesis examples per scenario. QLoRA fine-tune. Measure token reduction and quality.
4. **APEX quants** — Generate APEX-quantized GGUFs for Qwen3.5-35B / Gemma 4 26B A4B. Benchmark against current Q4_K_M.

### Future
5. **Constrained decoding** — Define GBNF grammars for tool call JSON schema and intent classification.
6. **Dynamic token budgets** — Instrument pipeline with token tracking, implement adaptive limits.

---

## Sources

- [Inference Engine Deep Dive](https://femiadeniran.com/blog/inference-engine-deep-dive-blog.html) — How inference works at the matrix multiplication level
- [APEX Quantization](https://github.com/mudler/apex-quant) — MoE-specific per-layer precision assignment
- [OogaBoogaLM](https://github.com/Mintzs/oogaboogalm) — Brevity baked into weights via QLoRA
- [TurboQuant (ICLR 2026)](https://github.com/ggml-org/llama.cpp/discussions/20969) — Extreme KV cache quantization (3.5-bit)
- [llama.cpp Speculative Decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [Host-Memory Prompt Caching Tutorial](https://github.com/ggml-org/llama.cpp/discussions/20574)
- [HOBBIT: MoE Expert Offloading](https://arxiv.org/html/2411.01433v2)
- [NVIDIA NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [Speculative Decoding: 2-3x Faster (2026)](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/)
- [JSONSchemaBench: Constrained Decoding Benchmark](https://arxiv.org/html/2501.10868v1)
