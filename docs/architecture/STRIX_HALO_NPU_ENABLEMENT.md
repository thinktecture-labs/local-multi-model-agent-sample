# Enabling the Solution on Strix Halo NPU — Deep Research

> Date: 2026-03-25
> Hardware: Minisforum MS-S1 MAX — AMD Ryzen AI Max+ 395 ("Strix Halo")
> Current state: All 5 models running on RDNA 3.5 iGPU via Vulkan. XDNA2 NPU (50 TOPS) idle.

---

## 1. Current State: Vulkan iGPU Only

The solution currently runs on Strix Halo using **Vulkan on the RDNA 3.5 iGPU** (40 CUs). The `start_servers.sh` already detects Vulkan:

```bash
elif [[ "$(uname -s)" == "Linux" ]] && vulkaninfo --summary 2>/dev/null | grep -q "INTEGRATED_GPU"; then
    EXTRA_LLAMA_ARGS="--no-mmap"
    info "Vulkan integrated GPU detected — using --no-mmap (UMA)"
fi
```

The XDNA2 NPU (50 TOPS) sits completely idle. This is the single biggest untapped performance opportunity.

---

## 2. The NPU Landscape (as of March 2026)

There are **three viable paths** to NPU inference on Strix Halo:

### Path A: FastFlowLM via Lemonade Server (Most Promising)

**What it is:** AMD's NPU-native LLM runtime, available since October 2025 (Windows) and March 11, 2026 (Linux). Ships as a 17 MB runtime with OpenAI-compatible API.

**Why it matters for this project:**
- **Supports all five model roles.** FastFlowLM officially supports: Gemma 3 (vision), Qwen 3.5, EmbeddingGemma (RAG embeddings), and Whisper (audio). GLM-OCR would need testing but vision models are supported.
- **OpenAI-compatible API.** `flm serve` exposes endpoints at `POST /api/v1/chat/completions`, `POST /api/v1/embeddings`, and `POST /api/v1/audio/transcriptions` — which is exactly what `SmallLanguageModelClient` already speaks via the OpenAI Python SDK.
- **Concurrent model slots.** FLM supports 1 LLM + 1 Audio + 1 Embedding model simultaneously on the NPU.
- **GGUF to FLM conversion.** The `FLM_Q4NX_Converter` tool converts GGUF Q4_0/Q4_1 models to the FLM Q4NX format for NPU execution.
- **Context length up to 256K tokens** (model-dependent).
- **Tool calling supported** in chat completions.
- **Power efficiency:** ~20-80 tok/s at <2W (CPU+NPU), vs the Vulkan iGPU which draws significantly more.

**Critical limitations:**
- NPU model slots are **mutually exclusive by type**: loading a new LLM evicts the current one. You cannot run Gemma3 1B (inference) + Qwen3.5 4B (function) + Gemma3 4B (vision) simultaneously on the NPU.
- Context length capped at **2K-3K tokens** for hybrid NPU mode (as of current releases).
- Licensing: free for companies under $10M annual revenue; commercial license needed above that.

### Path B: ONNX Runtime with Ryzen AI Software 1.7

**What it is:** AMD's Vitis AI Execution Provider for ONNX Runtime, running INT4-quantized models on the NPU.

**Why it's less suitable:**
- Requires converting all models to ONNX INT4 format (significant effort, potential accuracy loss)
- Context length capped at 2K-3K tokens
- No direct OpenAI-compatible API — would need a custom serving layer
- Better suited for single-model deployment (MoE models like GPT-OSS-20B)

### Path C: Hybrid NPU+iGPU Orchestration (Best Architecture)

**What it is:** Using the NPU for specific roles and the iGPU for others, exploiting the fact that they don't compete for compute resources.

**Why this is the right answer for this project:**

The multi-model architecture is a *perfect* fit for hybrid NPU+iGPU because different models have different latency profiles:

| Model | Current (Vulkan iGPU) | NPU Candidate? | Rationale |
|-------|----------------------|----------------|-----------|
| **embeddinggemma** (308M) | Fast | **Yes — ideal** | FLM has a dedicated NPU embedding slot. Frees iGPU bandwidth. |
| **gemma3-ft** (1B inference) | Fast | Maybe | Small model, but needs fast response. NPU LLM slot could work. |
| **Qwen3.5-4B** (function) | ~200ms Vulkan | **No** | Needs tool calling with complex JSON. Keep on iGPU where llama.cpp is proven. |
| **gemma3-4B** (vision) | Medium | **No** | Multimodal. Keep on iGPU — llama.cpp's mmproj is battle-tested. |
| **GLM-OCR** (0.9B) | Upload-time only | Maybe | Not latency-critical. Could offload to NPU to free iGPU during uploads. |
| **Whisper** | Optional | **Yes — ideal** | FLM has dedicated NPU audio slot. Zero iGPU impact. |

---

## 3. Recommended Implementation Plan

### Phase 1: EmbeddingGemma on NPU (Highest Impact, Lowest Risk)

**Why first:** EmbeddingGemma is called on every query (intent classification via LogReg, plus RAG search). Moving it to the NPU frees iGPU bandwidth for the heavier models and reduces power draw. FLM has a dedicated NPU embedding slot that doesn't conflict with LLM or audio.

**Steps:**
1. Install Lemonade Server + FastFlowLM on MS-S1 MAX (Linux)
2. Convert `embeddinggemma-ft.gguf` to FLM Q4NX format using `FLM_Q4NX_Converter`
3. Start FLM embedding server: `flm serve` with the converted model
4. Point `EMBEDDING_PORT` to the FLM server port (OpenAI-compatible, should work with the existing `AsyncOpenAI` client)
5. Run the embeddinggemma eval suite to verify MRR@10 holds
6. Benchmark: measure embedding latency and power draw vs Vulkan baseline

**Risk:** Low. Embedding is a simple `POST /v1/embeddings` call. The OpenAI API compatibility should mean zero code changes.

### Phase 2: Whisper on NPU (Free Performance)

**Why:** FLM has a dedicated NPU audio slot (concurrent with LLM + embedding). This is free — no conflicts.

**Steps:**
1. Convert whisper model to FLM format
2. Update `WHISPER_URL` to point to FLM audio endpoint
3. Verify voice pipeline works end-to-end

**Risk:** Very low. Whisper is optional and audio transcription is a standard API.

### Phase 3: Gemma3 1B (Intent + Synthesis) on NPU — Experimental

**Why:** The 1B inference model handles classification (~5ms on LogReg, but generative fallback is ~200ms) and tool-result formatting. On the NPU, it could run at <2W while freeing the iGPU entirely for Qwen and vision.

**Steps:**
1. Convert `gemma3-ft.gguf` to FLM format
2. Test intent classification accuracy (must match 96.7% on the gemma3-ft fallback path; LogReg primary on the same eval is 99.4%)
3. Test tool-result synthesis quality
4. Measure TTFT and tok/s on NPU vs Vulkan

**Risk:** Medium. The NPU LLM slot is shared — if Gemma3 1B is loaded, Qwen3.5 4B can't also be on the NPU. But since we're keeping Qwen on iGPU anyway, this works.

**Blocker:** Context length. If FLM caps at 2-3K tokens for the NPU LLM slot, RAG synthesis (which needs context from retrieved docs) might be too constrained. The current `ctx-size` is 4096. Need to verify FLM's actual context limits for Gemma3 1B.

### Phase 4: Architecture Change — NPU-Aware Model Routing

If Phases 1-3 succeed, add NPU awareness to the infrastructure:

**SmallLanguageModelClient model placement:**
```
INFERENCE  = gemma3         → NPU LLM slot (Phase 3)
FUNCTION   = qwen3.5-ft    → iGPU (Vulkan/ROCm) — tool calling needs proven backend
EMBEDDING  = embeddinggemma → NPU embedding slot (Phase 1)
VISION     = gemma3-4b      → iGPU (Vulkan/ROCm) — multimodal needs mmproj
```

**start_servers.sh addition:**
```bash
# NPU detection (Strix Halo XDNA2)
if command -v flm &>/dev/null && flm check 2>/dev/null; then
    info "XDNA2 NPU detected — starting embedding + whisper on NPU"
    flm serve embeddinggemma --port $EMBEDDING_PORT &
    # ... Whisper on NPU audio slot
fi
```

**config.py addition:**
```python
NPU_ENABLED = os.getenv("NPU_ENABLED", "false").lower() == "true"
NPU_EMBEDDING_PORT = int(os.getenv("NPU_EMBEDDING_PORT", "52625"))
```

---

## 4. Expected Impact

| Metric | Current (Vulkan-only) | With NPU Hybrid | Change |
|--------|-----------------------|-----------------|--------|
| **Embedding latency** | ~15ms (Vulkan) | ~5-10ms (NPU) | -30-50% |
| **iGPU contention** | 5 models sharing 40 CUs | 3 models on iGPU + 2 on NPU | ~40% less contention |
| **Power draw** | ~30-50W (all on iGPU) | ~20-35W (NPU at <2W for embedding+whisper) | -30% |
| **Whisper STT** | iGPU (competes with LLMs) | NPU (dedicated audio slot) | Zero impact on LLM latency |
| **Concurrent query capacity** | Limited by iGPU VRAM/bandwidth | Better — embedding offloaded | +30-50% |

---

## 5. What Won't Work (and Why)

1. **Running all 5 models on NPU.** FLM's NPU has only 3 slots (1 LLM + 1 Embedding + 1 Audio). You can't run Gemma3 1B + Qwen3.5 4B + Gemma3 4B vision simultaneously on NPU.

2. **llama.cpp native NPU backend.** Doesn't exist. [GitHub issue #14377](https://github.com/ggml-org/llama.cpp/issues/14377) was filed June 2025 requesting this, but no implementation has landed. The GGML project has an OpenVINO backend for Intel NPUs but nothing for AMD XDNA.

3. **Qwen3.5-4B tool calling on NPU.** While FLM supports tool calling, the 4B model with complex JSON function calling schemas is best kept on the iGPU where llama.cpp's autoparser is battle-tested. The NPU runtime is newer and less proven for structured output.

4. **Vision/multimodal on NPU.** FLM supports Gemma3 vision on NPU, but the `mmproj` (multimodal projector) integration is more mature in llama.cpp. Risk of regression.

---

## 6. Actionable Next Steps

**Immediate (this week):**
1. Install Lemonade Server + FastFlowLM on MS-S1 MAX
2. Convert embeddinggemma-ft to FLM Q4NX format
3. Test embedding API compatibility (should be zero code changes)
4. Run embeddinggemma eval (MRR@10 must hold at ≥98%)

**Short-term (next 2 weeks):**
5. Convert Whisper to FLM and test voice pipeline on NPU
6. Benchmark: embedding latency, power draw, concurrent query throughput
7. Add NPU detection to `start_servers.sh`

**Medium-term:**
8. Experiment with Gemma3 1B inference on NPU LLM slot
9. Add `NPU_ENABLED` config flag and NPU port configuration
10. Run full benchmark matrix with NPU hybrid configuration

---

## 7. Key Resources

- [Lemonade by AMD: A Unified API for Local AI Developers](https://www.amd.com/en/developer/resources/technical-articles/2026/lemonade-for-local-ai.html)
- [FastFlowLM GitHub](https://github.com/FastFlowLM/FastFlowLM) — NPU-native runtime
- [FastFlowLM Supported Models](https://fastflowlm.com/models/) — Gemma3, Qwen3.5, EmbeddingGemma, Whisper
- [Lemonade Server API Spec](https://lemonade-server.ai/docs/server/server_spec/) — OpenAI-compatible endpoints
- [AMD Ryzen AI NPU on Linux via Lemonade 10.0](https://agent-wars.com/news/2026-03-14-amd-ryzen-ai-npu-linux-lemonade-10-fastflowlm) — Linux support since March 11, 2026
- [Strix Halo LLM Benchmark Results (Level1Techs)](https://forum.level1techs.com/t/strix-halo-ryzen-ai-max-395-llm-benchmark-results/233796)
- [Strix Halo GPU Performance Tests (Framework)](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)
- [Strix Halo LLM Optimization Guide](https://www.hardware-corner.net/strix-halo-llm-optimization/)
- [llama.cpp NPU Support Request (Issue #14377)](https://github.com/ggml-org/llama.cpp/issues/14377)
- [AMD GPT-OSS-20B on NPU Technical Article](https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-gpt-oss-20b-on-amd-ryzen-ai-npus.html)
- [Lemonade SDK on PyPI](https://pypi.org/project/lemonade-sdk/)
- [Strix Halo Mini PCs for Local LLM Inference (Starry Hope)](https://www.starryhope.com/minipcs/strix-halo-local-llm-inference-2026/)
- [AMD Strix Halo Setup Guide (Medium)](https://medium.com/@cdbb_writes/from-bare-metal-to-ai-powerhouse-setting-up-amds-strix-halo-a58a1f3bc675)
- [Strix Halo Unleashed: Real LLM Workflows (Medium)](https://medium.com/@orami98/strix-halo-unleashed-real-llm-workflows-on-128gb-ryzen-ai-max-395-mini-pcs-and-laptops-5dabdd3fcae3)

---

## 8. Why This Project Is Uniquely Suited

The multi-model architecture is a *perfect* natural fit for Strix Halo's heterogeneous compute:

```
NPU (XDNA2, 50 TOPS, <2W):
  ├── Embedding slot  → embeddinggemma (308M) — every query
  ├── Audio slot      → Whisper STT — voice pipeline
  └── LLM slot        → gemma3-ft (1B) — intent + synthesis (experimental)

iGPU (RDNA 3.5, 40 CUs, ~30W):
  ├── Qwen3.5-4B FT   — tool calling (needs proven backend)
  └── gemma3-4B        — vision + RAG synthesis (needs mmproj)
```

No other local AI project is positioned to exploit this heterogeneous architecture as naturally as one that already separates models by role.
