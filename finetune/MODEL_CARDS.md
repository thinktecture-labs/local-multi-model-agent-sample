# Model Cards — Fine-Tuned Demo Stack

**⚠️ Conference talk demo — not production weights.** These models were trained
for a conference keynote on local on-device AI. They are published as a
reference for the fine-tuning patterns shown on stage — domain specialisation
of small models, deterministic + LLM hybrid routing, contrastive retrieval —
**not** for production deployment. See the project's
[`SECURITY.md`](../SECURITY.md) for the threat model and what's out of scope.

One section per model in the demo stack. Copy the relevant section into the
HuggingFace repo's README for each model (HF renders the repo README as the
model card automatically).

> Numbers reported here are reproducible from this repo via the canonical
> sequence in [`README.md`](README.md). Last updated alongside the
> Nextera reference scenario.

---

## Gemma3-1B FT (f16) — Direct Answer + Tool-Result Synthesis (intent fallback)

| | |
|---|---|
| **Base model** | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) (1.0B params) |
| **License** | Gemma Terms of Use — see [`MODEL_LICENSES.md`](MODEL_LICENSES.md) |
| **Training script** | [`finetune/train_gemma3.py`](train_gemma3.py) |
| **Method** | LoRA r=8, α=16, 7 epochs, lr=1e-4 |
| **Training data** | `data/training-data/gemma3_intent_{scenario}.jsonl` + `gemma3_synthesis_{scenario}.jsonl` |
| **Hardware tested** | RTX PRO 6000 (CUDA), Apple M-series (Metal/MPS fallback) |
| **Intended use** | (1) Direct-answer responses (no-retrieval-needed questions, chitchat); (2) tool-result synthesis (turning JSON tool outputs into natural-language answers); (3) generative intent-classification **fallback** when the LogReg classifier isn't loaded. Primary intent path is the [LogReg classifier](https://huggingface.co/thinktecture/intent-logreg-nextera) — it handles ~93% of traffic deterministically in <25ms. |
| **Out of scope** | RAG synthesis from multi-document context (cross-contaminates facts — use Gemma3-4B FT for that). General-purpose chat. Multilingual beyond the scenario's training language. |
| **Reference eval (Nextera)** | Intent (fallback path): 96.7% / 174 of 180 (post-2026-05-15 retrain on corrected training data; pre-retrain was 93.3%). Direct-answer: 95%+. See `docs/benchmarks/FINE_TUNING_INSIGHTS.md` §10. |
| **Known failure modes** | On the fallback intent path: confuses tool_use vs rag_query when the question contains a number AND a topical noun (e.g. "What's the cost of GDPR compliance?"). LogReg primary handles these cleanly. |

---

## Gemma3-4B FT (f16) — RAG Synthesis (+ Vision)

| | |
|---|---|
| **Base model** | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) (4.3B params, multimodal: text + vision via mmproj) |
| **License** | Gemma Terms of Use |
| **Training script** | [`finetune/train_gemma3_4b.py`](train_gemma3_4b.py) |
| **Method** | LoRA r=16, α=32, 3 epochs, lr=5e-5 |
| **Training data** | `data/training-data/gemma3_4b_synthesis_{scenario}.jsonl` (RAG passages + grounded answers) |
| **Hardware tested** | RTX PRO 6000 (CUDA). MPS works but slow; QLoRA via `--qlora` for ≤24GB VRAM |
| **Intended use** | RAG response synthesis — given retrieved passages and a user question, produce a grounded, source-faithful answer. The vision channel (mmproj) remains base-only. **Status: kept for A/B comparison; the Q4_K_M variant below is the production default since 2026-05-17.** |
| **Out of scope** | Tool calling (delegated to Qwen3.5-4B FT). Free-form chat without retrieved context. |
| **Reference eval (Nextera)** | RAG keyword grounding: 96% on 25-query holdout. See `docs/benchmarks/EVAL_RESULTS_*.md`. |
| **Known failure modes** | Will occasionally synthesise across documents that share lexical overlap but different domains — mitigated by the rewrite-query step that pre-filters retrieval. |

---

## Gemma3-4B FT (Q4_K_M) — RAG Synthesis (+ Vision) — **production**

| | |
|---|---|
| **Base model** | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) (4.3B params, multimodal: text + vision via mmproj) |
| **License** | Gemma Terms of Use |
| **Provenance** | `llama-quantize` from the F16 sibling GGUF (no separate training run — quantization only). See [`finetune/convert_gemma3_4b_to_gguf.sh`](convert_gemma3_4b_to_gguf.sh). |
| **File size** | 2.49 GB (vs 7.77 GB for F16) — ~3× memory-bandwidth headroom on decode |
| **Hardware tested** | RTX PRO 6000 (Blackwell sm_120), MBP M5 Max (Metal), DGX Spark (GB10 sm_121), Strix Halo (Vulkan/RDNA 3.5) — byte-deterministic across all four |
| **Intended use** | Production RAG response synthesis. Points-of-use: `scenarios/<scenario>.json:synthesis_4b_gguf_ft`. The vision channel uses the same GGUF (multimodal via the same mmproj as the F16 variant). |
| **Out of scope** | Same as F16 sibling — tool calling, free-form chat without retrieved context. |
| **Reference eval (Nextera, 2026-05-17)** | **Identical-quality to F16** on the 80-query RAG groundtruth set (MBP same-machine F16-vs-Q4_K_M A/B: 55/80 vs 54/80 = 1-query phrasing noise, zero semantic regression). Cumulative realized perf gains (F16 → Q4_K_M → b9196 pin): bench median dropped 14–53% across the four-backend fleet; RAG p50 dropped 30–60%; image-query p50 dropped 35–59%. RAG strict quality flat-or-better on every machine (+3 on RTX, +1 on DGX, +1 on Strix, flat on MBP). |
| **Known failure modes** | Same as F16 sibling. Q4_K_M-specific quantization artifacts not observed in our evals; would expect them most on rare-token tail behavior. |

---

## Qwen3.5-4B FT (Q4_K_M) — Tool Calling

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (4.0B params) |
| **License** | Tongyi Qianwen License — see [`MODEL_LICENSES.md`](MODEL_LICENSES.md) |
| **Training script** | [`finetune/train_qwen35_toolcalling.py`](train_qwen35_toolcalling.py) |
| **Method** | QLoRA r=16, α=16, 2 epochs, lr=2e-4 (via Unsloth — CUDA only) |
| **Training data** | `data/training-data/qwen35_toolcalling_{scenario}.jsonl` (~1,300 hand-curated tool-call examples) |
| **Hardware** | **CUDA required** (Unsloth dependency). Tested on RTX PRO 6000. |
| **Intended use** | Tool selection (sql_query / calculator) + argument generation. Native OpenAI tool-calling format. `enable_thinking=False` to keep output clean for llama.cpp's autoparser. |
| **Out of scope** | Free-form chat, RAG synthesis, intent classification. The model is trained only on tool-call outputs. |
| **Reference eval (Nextera, v9 post-2026-05-15 retrain)** | Tool routing: 99.4%. Multi-step decomposition (gemma3-ft): 98.8%. Multi-step chain shape: 97.5% (78/80, deterministic — verified byte-identical across 3 runs). SQL exec validity: 100% (79/79). Calculator expression correctness: 95.0%. |
| **Known failure modes** | Occasionally generates `<think>` blocks despite `enable_thinking=false` — the `_strip_thinking` filter in `src/engine/inference/client.py` handles this at parse time. Will refuse to answer if the query is clearly outside both tools (correct behaviour, but eval treats as "wrong tool"). |

---

## EmbeddingGemma 300M FT (q8_0) — RAG Retrieval

| | |
|---|---|
| **Base model** | [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) (308M params) |
| **License** | Gemma Terms of Use |
| **Training script** | [`finetune/train_embeddinggemma.py`](train_embeddinggemma.py) |
| **Method** | Contrastive (sentence-transformers MultipleNegativesRankingLoss), 10 epochs max with save_best, lr=5e-6 |
| **Training data** | `data/training-data/embeddinggemma_retrieval_{scenario}.jsonl` (query↔passage triplets with hard negatives) |
| **Hardware tested** | Works on CPU (slow), MPS (medium), CUDA (fast). 308M params is small enough that hardware rarely matters. |
| **Intended use** | Encoding documents and queries for semantic retrieval in ChromaDB. Output: 768-dim L2-normalised vectors. |
| **Out of scope** | Text generation (it's an encoder-only model). Cross-domain retrieval — the FT specialises it for the scenario's domain. |
| **Reference eval (Nextera)** | MRR@10: 0.9533 → 0.9800 (base → FT). Recall@5: ~98%. **Eval-corpus caveat:** measured on the held-out 25-query / 26-passage eval set — small by design, for fast iteration. Production retrieval against the live KB (~120 indexed chunks across 13 documents) was not measured separately; the MRR uplift against a real-size corpus may differ. |
| **Known failure modes** | The FT narrows the model's domain — out-of-domain queries (e.g. medical questions on the Nextera-FT model) retrieve nonsense with high confidence. Use the base model or a different scenario's FT for cross-domain queries. |

---

## LogReg Intent Classifier

| | |
|---|---|
| **Base** | scikit-learn `LogisticRegression`, multinomial, L2 penalty |
| **License** | Apache-2.0 (this repo) — but inputs are EmbeddingGemma vectors so the [Gemma Terms](MODEL_LICENSES.md) cover the embedding step |
| **Training script** | [`training/train_intent_logreg.py`](../training/train_intent_logreg.py) |
| **Method** | LogReg on FT-EmbeddingGemma's 768-dim output vectors. Held-out 90/10 split. ~2 minutes on CPU. |
| **Training data** | Same as Gemma3-1B intent: `data/training-data/gemma3_intent_{scenario}.jsonl` (re-embedded with the FT EmbeddingGemma) |
| **Hardware** | CPU is sufficient. Requires the FT EmbeddingGemma llama-server running on port 9092/9096 to embed training examples. |
| **Intended use** | Replaces the 1B generative classifier as the primary intent router. ~10ms per query (vs ~200ms for the 1B). Same accuracy on the standard eval set. |
| **Out of scope** | Anything that requires generation (it's a 3-way classifier). Low-confidence predictions (< 0.60 threshold, configurable in `intent_classifier_logreg.py`) are overridden to `direct_answer` as a safe fallback intent. The 1B generative classifier is only used as a load-time fallback when the LogReg model file is absent, not as a per-query confidence fallback. |
| **Reference eval (Nextera)** | 99.4% on 180-query eval set (post-2026-05-15 retrain on corrected training data; pre-retrain was 97.2%). ~5ms per classification (CUDA inference of embedding + single-threaded CPU LogReg fit). |
| **Known failure modes** | When the EmbeddingGemma FT changes, the LogReg weights become invalid — `intent_classifier_logreg.py:13-15` warns about this coupling. Re-train both together. |

---

## Filling these in for a new scenario

When you add a new scenario:

1. Update the **Training data** row in each card with your scenario-specific filename(s)
2. Re-run the canonical eval sequence after training and replace the **Reference eval** row with your numbers
3. If your domain has different known failure modes, document them in **Known failure modes** — that's where real ML engineers look first

The structure is intentionally minimal. A good model card answers four questions:
*what is it, what data trained it, what is it for, when does it break.*
