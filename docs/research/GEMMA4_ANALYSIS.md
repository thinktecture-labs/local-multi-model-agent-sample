# Gemma 4 — Analysis for Local Multi-Model Agent

> Research date: 2026-04-12. Based on the April 2, 2026 release.

---

## Model Lineup

| Variant | Params (effective) | Total | Type | Context | Audio | Image/Video | License |
|---------|-------------------|-------|------|---------|-------|-------------|---------|
| **E2B** | 2.3B | 5.1B | Dense | 128K | Yes | Yes | Apache 2.0 |
| **E4B** | 4.5B | 8B | Dense | 128K | Yes | Yes | Apache 2.0 |
| **26B A4B** | 4B active | 26B | MoE | 256K | No | Yes | Apache 2.0 |
| **31B** | 31B | 31B | Dense | 256K | No | Yes | Apache 2.0 |

All ship as base + instruction-tuned (IT). 140+ languages. Native function calling, structured JSON, extended thinking.

## Architecture Highlights

- **Hybrid attention**: Alternating local sliding-window (512/1024 tokens) + global full-context layers
- **Per-Layer Embeddings (PLE)**: Second embedding table feeding residual signals into every decoder layer
- **Shared KV Cache**: Last N layers reuse K/V tensors from earlier layers — significant memory savings
- **Proportional RoPE**: Enables long context on global layers
- **Vision encoder**: Learned 2D positions + multidimensional RoPE, native aspect ratios, configurable image token budgets (70/140/280/560/1120)
- **Audio encoder**: USM-style conformer (E2B/E4B only), max 30s

## Benchmarks vs Gemma 3

| Benchmark | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 E4B | Gemma 4 E2B | Gemma 3 27B |
|-----------|-------------|-----------------|-------------|-------------|-------------|
| MMLU Pro | 85.2% | 82.6% | 69.4% | 60.0% | 67.6% |
| AIME 2026 | 89.2% | 88.3% | 42.5% | 37.5% | 20.8% |
| GPQA Diamond | 84.3% | 82.3% | 58.6% | 43.4% | 42.4% |
| MMMU Pro (vision) | 76.9% | 73.8% | 52.6% | 44.2% | 49.7% |
| MATH-Vision | 85.6% | 82.4% | 59.5% | 52.4% | 46.0% |
| LMArena Elo | ~1452 | ~1441 | — | — | ~1300 |

Key: 26B A4B matches 31B quality with only 4B active parameters. E4B already beats Gemma 3 27B on vision tasks.

## Framework Support (Day 1)

- **llama.cpp** — GGUF quantized (Q4_K_M, Q8_0, etc.)
- **MLX** — Apple Silicon with TurboQuant KV cache
- **Transformers.js** — Browser/WebGPU via ONNX
- **vLLM** — Production serving
- **ONNX** — Edge devices
- **mistral.rs** — Rust-native, full multimodal

---

## Mapping to Our System

### Current Architecture (5 models, 3 families)

| Model | Role | Family | Size |
|-------|------|--------|------|
| gemma-3-1b-it | Intent classification + response synthesis | Gemma 3 | 1B |
| qwen3.5-4b | Tool selection | Gemma | 270M |
| Qwen3.5-4B FT | SQL generation + tool calling | Qwen | 4B |
| embeddinggemma | Vector embeddings | Gemma | 300M |
| gemma-3-4b-it | Vision / OCR | Gemma 3 | 4B |
| *(comparison)* Qwen3.5-35B | MoE column in "All" mode | Qwen | 35B |
| *(STT)* whisper.cpp | Speech-to-text | OpenAI | ~1.5B |

### Potential Gemma 4 Architecture (2-3 models, 1 family)

| Gemma 4 variant | Replaces | Why |
|-----------------|----------|-----|
| **E2B** (2.3B) | gemma-3-1b + qwen3.5-4b | Better reasoning + native function calling. One model does classify + route + synthesize |
| **E4B** (4.5B) | Qwen3.5-4B FT + gemma-3-4b vision | Native function calling + SQL + vision + audio. May need less fine-tuning for tool routing |
| **26B A4B** | Qwen3.5-35B (MoE comparison) | Same family, 4B active params, 26B quality. Keeps "All" mode comparison within Gemma family |
| **E4B audio** | whisper.cpp | Native audio input (30s max). STT + intent in one pass |
| *Keep as-is* | embeddinggemma | Gemma 4 is generative, not an embedding model |

### What This Enables

1. **Single model family story**: "Gemma 4 at every scale" instead of "Gemma + Qwen + Qwen tool caller + Whisper"
2. **Fewer models to manage**: 5 llama-server instances → potentially 2-3
3. **Native function calling**: Less fine-tuning needed for tool routing (Gemma 4 does it out of the box)
4. **Audio consolidation**: E4B handles STT natively — skip whisper-server entirely
5. **Vision upgrade**: E4B vision beats Gemma 3 27B on MMMU Pro and MATH-Vision

### What It Does NOT Help With

- **Embeddings** — Still need a dedicated embedding model (embeddinggemma or similar)
- **TTS** — No speech output. Piper/Chatterbox still needed
- **Fine-tuned SQL patterns** — Domain-specific SQL (BW schema, Nextera schema) still benefits from fine-tuning. Native function calling helps routing, not domain SQL generation
- **Speed of tiny models** — qwen3.5-4b at 270M is extremely fast (~5ms). E2B at 2.3B will be slower for tool selection, even if more capable

---

## Impact on Keynote Story

### Current narrative
"5 specialized small models with task decomposition beat one big cloud model"

### Upgraded narrative
"One open model family — Gemma 4 — handles text, vision, audio, and tools. From 2B in your browser tab, to 4B on your GPU, to 26B MoE activating only 4B parameters. Same architecture at every scale. No cloud."

### Specific keynote beats

1. **Model consolidation**: "We went from 5 models across 3 families to 3 models from one family. Simpler. Same quality."

2. **Native function calling**: "No fine-tuning needed for tool routing. Gemma 4 understands function schemas out of the box. We only fine-tune for domain-specific SQL."

3. **Audio without Whisper**: "The same 4B model that writes SQL also transcribes your voice. One model, two modalities."

4. **MoE efficiency**: "26 billion parameters, but only 4 billion activate per token. It runs at the speed of a 4B model with the quality of a 26B model."

5. **Browser demo**: "Gemma 4 E2B in the browser via WebGPU. Vision-capable. It reads a document from your camera and detects PII — in a browser tab."

6. **Three-path comparison upgraded**: Multi-Models (Gemma 4 E2B+E4B) vs MoE (Gemma 4 26B A4B) vs Cloud (GPT-5.4). All local paths are now the same family.

### Architecture slide addition

```
┌─────────────────────────────────────────────────────┐
│                    Gemma 4 Family                     │
│                                                       │
│  Browser    ┌─────┐  On-Device  ┌─────┐  ┌─────┐    │
│  (WebGPU)   │ E2B │  (GPU)      │ E4B │  │26B  │    │
│             │ 2.3B│             │ 4.5B│  │A4B  │    │
│             └──┬──┘             └──┬──┘  └──┬──┘    │
│                │                   │        │        │
│  Text+Audio    │  Text+Vision     │  Text+  │        │
│  +Vision       │  +Audio+Tools    │  Vision │        │
│                │  +SQL+STT        │  (MoE)  │        │
└─────────────────────────────────────────────────────┘
         │                │               │
    PII Scanner      Agent Pipeline   Quality Check
    (Act 5b)         (Acts 1-4)       ("All" mode)
```

---

## Migration Path

### Phase 1: Drop-in upgrades (low risk)
- Replace gemma-3-1b-it → Gemma 4 E2B for intent classification + synthesis
- Replace gemma-3-4b-it → Gemma 4 E4B for vision/OCR
- Replace Qwen3.5-35B → Gemma 4 26B A4B for MoE comparison column

### Phase 2: Consolidation (medium risk, needs testing)
- Test E4B native function calling against Qwen3.5-4B FT on tool routing eval
- If comparable: replace Qwen3.5-4B FT → Gemma 4 E4B (with domain fine-tuning for SQL)
- Merge qwen3.5-4b tool selection into E2B intent classification step

### Phase 3: Audio consolidation (needs eval)
- Test E4B audio transcription quality vs whisper.cpp on English + German
- If comparable: route audio directly to E4B for STT + intent in one pass
- Keep whisper as fallback for >30s audio

### Phase 4: WebGPU upgrade
- Replace LFM 2.5 1.2B → Gemma 4 E2B in browser demo
- Gains: vision capability for PII redaction demo, audio input, better quality
- Same Transformers.js pipeline, just swap the model

---

## Open Questions

1. **E4B function calling quality vs fine-tuned Qwen3.5-4B** — Need to run the existing tool routing eval suite. If Gemma 4 matches without fine-tuning, that's a major simplification.
2. **E4B audio quality vs whisper.cpp** — 30s limit is fine for queries but test accuracy on German speech.
3. **Fine-tuning E4B for domain SQL** — Does TRL/Unsloth support Gemma 4 FT already? Can we use existing JSONL training data?
4. **GGUF availability** — GGUF quants exist via ggml-org and unsloth. Verify multimodal (vision+audio) works in llama-server, not just llama-cli.
5. **Memory budget** — Running E2B + E4B + 26B A4B simultaneously on RTX PRO 6000 (96GB). Estimate: E2B Q8 (~5GB) + E4B Q8 (~8GB) + 26B A4B Q4 (~15GB) = ~28GB. Comfortable fit.
6. **WebGPU E2B vision** — Does the ONNX Q4 export include the vision encoder? Check onnx-community/gemma-4-E2B-it-ONNX.
