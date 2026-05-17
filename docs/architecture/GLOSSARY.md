# Glossary

Technical terms and concepts used across this repository. Organized by category with references to where each term appears in the codebase.

---

## AI / ML Fundamentals

**Attention mechanism** -- The core operation in transformer architectures where each token attends to all others. LoRA targets the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

**Backpropagation** -- The algorithm for computing gradients of the loss function with respect to model weights. Drives all fine-tuning in this project.

**bfloat16 (bf16)** -- A 16-bit floating point format used for training. Qwen3.5-4B FT v8 requires bf16 -- using fp16 causes crashes on Apple MPS for bfloat16-native models. See `finetune/train_qwen35_toolcalling.py`.

**Catastrophic forgetting** -- When fine-tuning on a new task causes a model to lose previously learned capabilities. Training gemma3 for classification only caused it to forget text generation. Fixed by mixing generation examples into the classification dataset at a 90:9:1 ratio. See `docs/benchmarks/FINE_TUNING_INSIGHTS.md`.

**Chat template** -- A model-specific format that structures system/user/assistant turns into tokens. For Gemma: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...`. Must be re-injected into `tokenizer_config.json` before GGUF conversion.

**Completion-only loss** -- Training only on the model's response tokens, ignoring loss on the instruction/prompt tokens. Used via `DataCollatorForCompletionOnlyLM` in all training scripts.

**Context window** -- The maximum number of tokens a model can process in a single forward pass. gemma3-1B: 32K tokens; gemma3-4B: 128K tokens.

**Cosine similarity** -- The similarity metric used for embedding vector comparison. The embedding model produces 768-dimensional vectors; retrieval ranks documents by cosine similarity to the query vector.

**Embeddings** -- Dense vector representations of text. embeddinggemma produces 768-dimensional vectors used for semantic search in the RAG pipeline. Stored in ChromaDB.

**Gradient accumulation** -- Simulates larger batch sizes by accumulating gradients over multiple forward passes before updating weights. Used in gemma3 training (`gradient_accumulation_steps=4`) to achieve effective batch size of 16 with per-device batch of 4.

**Greedy decoding** -- A generation strategy that always picks the highest-probability next token. Used for classification tasks (deterministic output). Contrast with beam search and sampling.

**Inference** -- Running a trained model to produce predictions. In this project, all inference runs locally via llama-server with no cloud API calls.

**Softmax** -- The function that converts raw model outputs (logits) into probability distributions. Used in the final layer of classification and generation.

**Temperature** -- A parameter controlling randomness in model output. Lower = more deterministic, higher = more creative. Set to 0.0 for classification, 0.7 for synthesis. See `src/engine/inference/config.py`.

**Tokenization** -- Converting text into integer token IDs that the model processes. Each model has its own tokenizer with a different vocabulary.

**Top-p (nucleus) sampling** -- A decoding strategy that samples from the smallest set of tokens whose cumulative probability exceeds p. Used for synthesis generation.

**Transformer** -- The neural network architecture underlying all four models. Based on self-attention layers that process sequences in parallel.

---

## Models and Architectures

**gemma3-1B (gemma3:1b-it)** -- Google's 1-billion parameter instruction-tuned language model. Serves as the planner (intent classification), query decomposer, and tool-use synthesizer. Fine-tuned for 3-class intent classification. RAG synthesis is handled by the 4B model instead. Ports: 9090 (base), 9094 (FT).

**gemma3-4B (gemma3:4b-it)** -- Google's 4-billion parameter instruction-tuned model with vision capabilities. Handles image understanding and RAG synthesis (superior multi-document comprehension vs 1B). Port: 9093.

**Qwen3.5-4B FT v8 (qwen3.5-4b-toolcalling-ft)** -- The production tool caller (`Qwen/Qwen3.5-4B` base, 4B parameters, Alibaba/Qwen team). Routes queries to `sql_query` or `calculator` via native OpenAI-compatible function calling, generating tool arguments end-to-end (no scaffolding pre-routers). Trained via QLoRA r=16 in Unsloth on ~1,372 examples. Ports: 9091 (base), 9095 (FT). The legacy FunctionGemma 270M tool caller it replaced is documented in [`FINE_TUNING_INSIGHTS.md` §3](../benchmarks/FINE_TUNING_INSIGHTS.md#3-functiongemma-270m--legacy-2-tool-routing-superseded-by-qwen35-4b-ft-v8-see-3b).

**embeddinggemma (embeddinggemma:308m)** -- Google's 308M-parameter bidirectional encoder for semantic retrieval. Produces 768-dimensional embeddings for RAG. Fine-tuned with contrastive learning. Ports: 9092 (base), 9096 (FT).

**Task decomposition** -- The core architectural principle: four specialist models each handle a narrow task rather than one monolithic model handling everything. Intent classification -> tool routing -> execution -> synthesis.

**Single-turn inference** -- *Historical.* Each call to the legacy FunctionGemma 270M tool caller got a fresh conversation with no prior history, because the 270M model was not trained for multi-turn conversations. Qwen3.5-4B FT v8 handles multi-turn natively (~41 multi-turn examples in its training set), so the production pipeline now passes prior tool results through normal conversation context.

**Control tokens** -- *Historical.* Specialized tokens in the legacy FunctionGemma vocabulary: `<start_function_declaration>`, `<end_function_declaration>`, `<start_function_call>`, `<end_function_call>`, `<start_function_response>`. Format had to be reproduced exactly via `tokenizer.apply_chat_template()`. Qwen3.5-4B uses its own native Jinja chat template with `--jinja --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0` at the llama-server flags.

---

## Fine-Tuning

**LoRA (Low-Rank Adaptation)** -- A parameter-efficient fine-tuning method that trains small rank-decomposition matrices instead of all model weights. Used for gemma3 (rank 16, alpha 32, targeting attention layers). See `finetune/train_gemma3.py`.

**rsLoRA (Rank-Stabilized LoRA)** -- A LoRA variant that adjusts the scaling factor to `alpha / sqrt(rank)` instead of `alpha / rank`, improving training stability. Enabled via `use_rslora=True`.

**EVA (Explained Variance Adaptation)** -- An initialization strategy for LoRA that uses SVD on a calibration set to find the principal directions of the weight matrices. Achieves 95% accuracy at rank 16 where standard LoRA achieves 93.3%. See `docs/benchmarks/FINE_TUNING_INSIGHTS.md`.

**QLoRA (Quantized LoRA)** -- Combines 4-bit quantization with LoRA for lower VRAM usage. Available but disabled by default in this project because bitsandbytes weights cannot be converted to GGUF.

**Full fine-tuning** -- Training all model weights (no adapter). Used for embeddinggemma (308M is small enough), and was used historically for the FunctionGemma 270M tool caller. Qwen3.5-4B FT v8 instead uses QLoRA r=16 — at 4B parameters LoRA captures the structured-output shift that full FT was needed for at 270M. Contrast with LoRA which only trains adapter matrices.

**SFTTrainer** -- HuggingFace's Supervised Fine-Tuning Trainer class. Used for gemma3 training. Handles chat template formatting, completion-only loss, gradient accumulation. (Qwen3.5-4B FT v8 uses Unsloth's `FastLanguageModel` + SFTTrainer wrapper for QLoRA.)

**Contrastive learning** -- A training approach that teaches embeddings by pulling similar items together and pushing dissimilar items apart. Used for embeddinggemma via MNRL (Multiple Negatives Ranking Loss) and TripletLoss.

**Hard negatives** -- Training examples that are semantically similar but should NOT match. For embeddinggemma: 134 triplets like "Nextera security" matched against the security passage (positive) and the pricing passage (hard negative). Critical for disambiguation.

**MNRL (Multiple Negatives Ranking Loss)** -- A contrastive loss function for embedding models. Each query-positive pair uses all other batch positives as in-batch negatives. Primary loss for embeddinggemma.

**TripletLoss** -- A contrastive loss requiring (anchor, positive, negative) triplets. Used alongside MNRL for embeddinggemma's hard negative examples.

**Alpaca format** -- A JSONL training data format with `instruction`, `input`, `output` fields. Used for gemma3 training data.

**Merge and unload** -- The PEFT method (`merge_and_unload()`) that bakes LoRA adapter weights into the base model, producing a standard HuggingFace model ready for GGUF conversion.

**Epoch** -- One complete pass through the training data. gemma3: 7 epochs (intent), 3 epochs (synthesis); Qwen3.5-4B FT v8: 2-3 epochs; embeddinggemma: 5 epochs.

**Learning rate** -- The step size for weight updates during training. gemma3: 5e-5 (LoRA); Qwen3.5-4B FT v8: 2e-4 (QLoRA); embeddinggemma: 2e-5.

**Early stopping** -- A training callback (`EarlyStoppingCallback`) that halts training when eval loss stops improving for N consecutive evaluations (patience=3, threshold=0.01). Prevents overfitting by stopping before the model memorizes training data. Used in gemma3 and Qwen3.5-4B FT training scripts.

**LossHistoryCallback** -- A custom `TrainerCallback` that captures train/eval loss at each logging step, saves JSON training curves after training, and prints a convergence summary with overfitting detection. Shared across all training scripts via `finetune/training_utils.py`.

**Training curves** -- JSON files recording `{step, train_loss, eval_loss, epoch}` at each logging step. Saved automatically to the model output directory (e.g. `gemma3_training_curves.json`). Used to diagnose convergence, detect overfitting, and compare training runs.

---

## Quantization and Model Formats

**GGUF (GPT-Generated Unified Format)** -- The binary model format used by llama.cpp for CPU/GPU inference. All four models are converted from HuggingFace format to GGUF before deployment. Conversion via `llama-quantize` or `convert_hf_to_gguf.py`.

**Quantization** -- Reducing model precision from float32/float16 to lower bit widths (Q4_K_M, Q8_0) to decrease memory usage and increase inference speed. Trade-off: smaller models may lose accuracy.

**Q4_K_M** -- A 4-bit quantization scheme with medium K-quant optimization. Used for gemma3-1B (~700MB on disk). Good balance of size vs. quality.

**Q8_0** -- An 8-bit quantization scheme. Used for embeddinggemma where precision matters more than size. Qwen3.5-4B FT v8 ships as f16 → Q4_K_M GGUF (precision sufficient at 4B for tool-call structure).

**F16 (float16)** -- Half-precision floating point. Used as an intermediate format during GGUF conversion. Also the default for gemma3-4B vision model.

**llama.cpp** -- The C++ inference engine that serves all four models. Provides OpenAI-compatible API endpoints. Compiled with Metal (macOS), CUDA (NVIDIA), or Vulkan (AMD) acceleration.

**llama-server** -- The HTTP server component of llama.cpp. Each model runs as a separate llama-server process on its own port (9090-9096).

**Vulkan** -- A cross-platform GPU compute API. Used as the inference backend on AMD Strix Halo (MS-S1 MAX) via the Mesa RADV driver. More stable than ROCm/HIP for short-context workloads; requires `--no-mmap` on UMA systems. See `start_servers.sh` auto-detection.

**Strix Halo (MS-S1 MAX)** -- AMD Ryzen AI Max+ 395 APU with 40 RDNA 3.5 CUs, 128 GB LPDDR5x-8000 unified memory (~212 GB/s bandwidth). The 4th benchmark platform. Runs all models via Vulkan/RADV; latency comparable to DGX Spark. Key advantage: 128 GB unified memory for large models without discrete GPU.

---

## Evaluation and Statistics

**Wilson score CI** -- A confidence interval for binomial proportions that is more accurate than the normal approximation for small n or extreme p. Used for all accuracy metrics. Implementation in `finetune/eval_base.py`.

**Bootstrap CI** -- A non-parametric confidence interval computed by resampling with replacement. Uses 10,000 resamples with seed=42 for reproducibility. Implementation in `finetune/eval_base.py`.

**McNemar's test** -- A paired significance test for comparing two models on the same queries. Uses chi-squared with continuity correction. P-value via `math.erfc(sqrt(chi2/2))` for df=1 (no scipy required). Answers: "Is the accuracy difference real or just noise?"

**Jaccard similarity** -- Word-set similarity metric: `|A intersection B| / |A union B|`. Used to detect train/eval data leakage. Threshold: 0.7 (queries above this are considered overlapping).

**Confidence interval (CI)** -- A range expressing the uncertainty in a measured value. In this project, 95% Wilson CIs bracket all accuracy scores. Example: `95.0% [90.8%, 97.3%]` means the true accuracy is likely between 90.8% and 97.3%.

**p-value** -- The probability of observing the measured difference (or more extreme) under the null hypothesis of no real change. If p < 0.05, the difference is "statistically significant." Shown in `compare()` reports.

**MRR@10 (Mean Reciprocal Rank at 10)** -- The primary retrieval quality metric for embeddinggemma. The reciprocal of the rank of the first relevant document in the top-10 results, averaged across queries. FT score: 0.9800 (HuggingFace), 0.9533 (GGUF via llama-server — Dense projection layers lost in conversion). **Eval-corpus caveat:** scores are measured on the held-out 25-query / 26-passage eval set (intentionally small for fast iteration). The live ChromaDB indexes ~120 chunks across 13 KB documents — production-corpus MRR was not measured separately and the uplift may differ at that scale.

**Recall@K** -- The fraction of queries where the relevant document appears in the top-K results. Recall@5 = 100% means every query's answer is in the top 5 documents.

**Accuracy** -- The fraction of queries where the model's prediction matches the expected label. Overall accuracy is the primary metric for intent classification and tool routing.

**Per-class accuracy** -- Accuracy broken down by intent class or tool type. Reveals if a model is strong on one class but weak on another.

**F1 score / Harmonic mean** -- Qwen3.5-4B FT's `overall` score is the harmonic mean of tool accuracy and argument key accuracy. If either is zero, the harmonic mean is zero.

**Confusion matrix** -- A matrix showing predicted vs. expected labels. Generated by `eval_multi_step.py` for the full pipeline.

**Eval set** -- The fixed labeled test queries used to measure model accuracy. gemma3 intent: 180 queries (60/class); Qwen3.5-4B FT tool routing: 240 queries spanning single-step + multi-step SQL + calculator. Stored as JSONL in `data/eval-data/` and loaded via `finetune/eval_base.py`.

**Train/eval overlap** -- Data leakage where eval queries appear in (or are too similar to) training data, producing inflated accuracy scores. Checked via Jaccard similarity with threshold 0.7.

---

## Project Architecture

**ChromaDB** -- The vector database storing document embeddings for RAG retrieval. In-process (no separate server). Uses embeddinggemma vectors for semantic search.

**Circuit breaker** -- A resilience pattern that tracks consecutive failures per model and temporarily stops sending requests after a threshold. States: closed (normal) → open (failing, reject requests) → half_open (test recovery). Implemented per-model in `src/engine/inference/client.py`.

**Confidence router** -- An 8-factor heuristic scoring system (`src/engine/scaffolding/confidence_router.py`) that decides between `rag_only`, `tool_only`, and `combined` query handling strategies based on classification confidence, context quality, and tool coverage.

**Deterministic pre-router** -- Historical (keynote demo mode only). Implemented in `src/engine/agent/handlers/tool_use.py` (added 2026-03-12, superseded 2026-03-17). When `USE_SCAFFOLDING_BUILDERS=true`, tries: (1) expr-builder (28 regex patterns, 0ms) for calculator queries, (2) sql-builder (25 regex patterns, 0ms) for SQL queries. **In production `USE_SCAFFOLDING_BUILDERS` defaults to `false` — NullResolvers are active and Qwen3.5-4B FT v8 handles 100% of tool routing end-to-end.** The builders are retained in the codebase for keynote comparison demos only and are never reached on the production code path.

**Dual-port model swap** -- Zero-downtime architecture where base models (ports 9090-9093) and fine-tuned models (ports 9094-9096) run simultaneously. `POST /models/swap` toggles between them instantly.

**Error boundary** -- A React component that catches JavaScript errors in its child component tree and displays a fallback UI instead of crashing. `src/clients/observatory-react/src/components/ErrorBoundary.tsx` wraps the entire Observatory app. React 19 still requires class components for error boundaries.

**ExpressionResolver / SQLResolver** -- Protocol interfaces (`src/engine/agent/tool_argument_resolver.py`) that decouple tool-argument generation from the orchestrator via dependency injection. Production uses the `NullExpressionResolver` / `NullSQLResolver` no-op implementations — Qwen3.5-4B FT v8 generates expressions and SQL natively, end-to-end. The earlier deterministic pattern-matching pre-routers (`expression_builder.py`, `sql_builder.py`) were retired in commit `68d52a5` once the FT model superseded them.

**SmallLanguageModelAgentOrchestrator** -- Thin router class (`src/engine/agent/orchestrator.py`, ~218 lines) that validates input, classifies intent, dispatches to the appropriate handler, and logs interactions. Business logic lives in four handler classes under `src/engine/agent/handlers/`: `DirectAnswerHandler`, `RAGHandler` (rewrite → vector search → 4B synthesis), `ToolUseHandler` (single-step + multi-step chained execution), and `VisionHandler` (image analysis). Also delegates to `IntentClassifier`, `QueryDecomposer`, `InteractionLogger`, and protocol-based `ExpressionResolver`/`SQLResolver`. Shared types and prompts live in `src/engine/agent/types.py`.

**SmallLanguageModelClient** -- The HTTP client class (`src/engine/inference/client.py`) that communicates with all four llama-server instances. Provides `classify()`, `select_tool()`, `embed()`, `generate()`, and `describe_image()` methods. The `create_with_auto_detection()` classmethod probes FT ports (9094-9096) via `/health` and falls back to base ports (9090-9092) — used by all evals, E2E tests, benchmark, demo.py, and `src/server/` startup. Per-model `asyncio.Semaphore` concurrency limits (default 4, configurable via `MODEL_CONCURRENCY_LIMIT`) prevent request queuing overload. Circuit breaker tracks consecutive failures per model and calls `record_success()` on the first successful stream chunk (not before consumption) to avoid false positives.

**Adversarial defense (pre-classifier)** -- A 5-layer defense stack in `src/engine/agent/intent_classifier.py`: (1) 30 regex injection patterns, (2) gibberish detector, (3) non-ASCII filter, (4) LogReg confidence threshold (0.60), (5) canned refusal. Matched queries are routed directly to `direct_answer` without inference, saving latency and preventing misrouting. Pipeline robustness: 93.3% (up from 43.3% generative-only baseline).

**Intent classification** -- The first step in the agent pipeline. Primary path: LogReg classifier on embeddinggemma embeddings (deterministic, <5ms, 99.4% accuracy post-2026-05-15 retrain). Fallback: gemma3-ft generative classification (96.7% post-retrain). Classifies into `rag_query`, `tool_use`, or `direct_answer`.

**IntentClassifier** -- Extracted module (`src/engine/agent/intent_classifier.py`) handling 3-way intent classification with LogReg primary path and generative fallback. Includes 5-layer adversarial defense (30 regex + gibberish + non-ASCII + LogReg confidence threshold + canned refusal). 93.3% adversarial robustness.

**IntentClassifierLogReg** -- LogReg-based intent classifier (`src/engine/agent/intent_classifier_logreg.py`) trained on embeddinggemma embeddings. Pure math -- no temperature, no sampling, same input = same output. Model stored in `models/intent-logreg/model.joblib`. 99.4% accuracy post-2026-05-15 retrain, <5ms inference. **Coupling:** tightly coupled to the embedding model — changing embeddinggemma (swap, re-fine-tune, or quantization change) invalidates the trained weights and requires retraining via `python -m training.train_intent_logreg`.

**InteractionLogger** -- Extracted module (`src/engine/agent/interaction_logger.py`) providing thread-safe interaction logging with `log()`, `export()`, token accumulation, and JSON file export.

**Multi-step decomposition** -- Breaking complex queries into sequential tool calls. Example: "Compare Q1 and Q2 revenue" -> step 1: SQL for Q1, step 2: SQL for Q2, step 3: calculator for difference. Detection and decomposition in `src/engine/agent/query_decomposer.py`, execution orchestrated by `src/engine/agent/orchestrator.py`.

**Observatory** -- The browser-based UI for interacting with the agent system. A React 19 + TypeScript SPA (`src/clients/observatory-react/`). Shows agent reasoning steps, model traces, three-path comparison (multi-models / Qwen / cloud), energy tracking, and response synthesis.

**Per-model semaphore** -- An `asyncio.Semaphore` per model role (default: 4 concurrent requests) that prevents unbounded request queuing when a llama-server instance is saturated. Configured via `MODEL_CONCURRENCY_LIMIT` env var in `src/engine/inference/config.py`. Wraps all API methods in `src/engine/inference/client.py`.

**QueryDecomposer** -- Extracted module (`src/engine/agent/query_decomposer.py`) handling multi-step detection (11 regex patterns), query decomposition via LLM with mechanical fallback, and step concretization.

**RAG (Retrieval-Augmented Generation)** -- A technique where the model retrieves relevant documents before generating a response. For `rag_query` intents: embeddinggemma finds relevant passages in ChromaDB, then gemma3-4B (vision model) synthesizes an answer citing those passages.

**Request ID** -- A 12-character hex UUID (`uuid.uuid4().hex[:12]`) generated per agent request and included in `AgentResponse`. Threaded through the API response (`QueryResponse.request_id`) for end-to-end observability. Defined in `src/engine/agent/types.py`.

**SQL parameterization** -- Defense-in-depth pattern used historically by the now-retired `sql_builder.py`: returned `(sql_template, params_tuple)` with `?` placeholders instead of f-string interpolated values, preventing SQL injection even if user input reached the SQL builder. Production no longer routes through that builder — Qwen3.5-4B FT v8 generates SQL natively, and the `sql_query` tool enforces a read-only SELECT whitelist at the executor (`src/engine/tools/sql_query.py`).

**SSE (Server-Sent Events)** -- The streaming protocol used by `POST /chat` to deliver real-time agent reasoning steps to the Observatory UI. Each step (classification, tool call, synthesis) is a separate SSE event.

**Synthesis** -- The final step where the agent combines tool results, retrieved documents, or general knowledge into a natural-language response. RAG synthesis uses gemma3-4B (superior multi-doc comprehension); **multi-step** tool-result synthesis is routed through Qwen3.5-4B FT v9 (FUNCTION role) since commit `118b6a1` — gemma3-1B was unreliable on the structured-context-into-prose task. **Single-step** tool-result formatting still uses gemma3-1B (simpler input, no chained reasoning).

**Tool routing** -- The second step for `tool_use` queries. Qwen3.5-4B FT v8 handles all tool selection and argument generation natively via OpenAI-compatible function calling. The earlier deterministic pre-routers were retired once the FT model superseded them; `NullExpressionResolver` and `NullSQLResolver` are active in production. select_tool p50: 381ms (RTX), 1115ms (MBP Metal), 2410ms (DGX Spark).

**ToolRegistry / create_default_registry()** -- The tool registration system (`src/engine/tools/tool_registry.py`). `create_default_registry()` is the single factory that enforces consistent tool order (vector_search → sql_query → calculator) to prevent Qwen3.5-4B FT positional bias. Used by `src/server/`, demo.py, evals, and benchmark.

---

## Voice Pipeline

**STT (Speech-to-Text)** -- Converting spoken audio to text via whisper-server (whisper.cpp). Port 9097. Auto-detects English and German.

**TTS (Text-to-Speech)** -- Converting text to spoken audio via Piper TTS (CPU, ONNX runtime). English voice: `en_US-lessac-medium`; German voice: `de_DE-thorsten-high`.

**whisper.cpp** -- A C++ implementation of OpenAI's Whisper speech recognition model. Runs locally for on-device STT.

**Piper TTS** -- A fast, local text-to-speech engine using ONNX models. Produces WAV audio files that auto-expire after 120 seconds.

**Voice-to-voice** -- The full round-trip: audio input -> STT transcription -> agent pipeline -> text response -> TTS audio output. Exposed via `POST /voice/chat` SSE endpoint.

---

## Business Domain (Nextera)

**Nextera Platform** -- The fictional SaaS AI platform used as the demo scenario. Three pricing tiers: Starter (EUR 299/mo), Professional (EUR 999/mo), Enterprise (EUR 3,500/mo). All RAG documents, SQL data, and eval queries reference this scenario.

**ARR (Annual Recurring Revenue)** -- Annual subscription revenue. Frequently computed in demo queries: `MRR x 12`.

**MRR (Monthly Recurring Revenue)** -- Monthly subscription revenue per customer. Key metric in the `customers` SQL table.

**Churn rate** -- The percentage of customers who cancel per quarter. Tracked in the `sales` SQL table. Featured in many eval and demo queries.

**Meridian Health** -- A fictional 340-hospital EU healthcare network used as a case study in the RAG corpus. Demonstrates data residency requirements and HIPAA compliance.

---

## Testing

**Unit tests** -- Tests requiring no external services. ~1001 tests in `tests/unit/`. Cover scoring math, tool contracts, SQL injection fuzzing, data prep, vector search, eval statistics, adversarial/OOD eval, concurrency stress tests, and extracted module tests (intent classifier, interaction logger, query decomposer, tool argument resolver).

**Integration tests** -- Tests requiring local services only (SQLite, server endpoints with mocks). ~108 tests in `tests/integration/`.

**E2E (end-to-end) tests** -- Tests requiring llama-server instances with seeded data. ~43 tests in `tests/e2e/`. Auto-detect FT servers (9094-9096) when available, falling back to base ports (9090-9092). Auto-skip when no servers are reachable.

**FT regression tests** -- `tests/e2e/test_ft_regression.py`: 18 test functions exercising 351+ golden queries against live fine-tuned servers. Auto-detects FT ports. Accuracy thresholds catch regressions before deployment.

**Golden queries** -- A fixed set of labeled queries with known-correct answers used to detect model regressions. Imported directly from eval scripts (`TEST_SET`).

**Markers** -- pytest decorators (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`) for selective test execution.

**Fixtures** -- pytest shared setup objects in `tests/conftest.py`: mock SmallLanguageModelClient, temporary SQLite database, fake VectorStore.

---

## Document Upload & OCR

**GLM-OCR** -- A 0.9B vision model (zai-org/GLM-OCR) for extracting text and tables from PDF page images. Runs as a llama-server on port 9098. The 5th specialized model in the stack, used at upload time only. See `src/engine/knowledge/ocr_client.py`.

**Smart hybrid OCR** -- The document processor first tries pypdf (fast, <1s for 250 pages). Only pages where pypdf produces poor output (<100 chars or garbled) are sent to GLM-OCR (~7s/page on Metal). Reduces a 50-page upload from 4.4min (full OCR) to ~9s (hybrid). See `DocumentProcessor._smart_ocr_stream()`.

**Uploads collection** -- A separate ChromaDB collection (`uploads`) for user-uploaded documents. Isolated from the curated 13-doc knowledge base (`knowledge_base`). Chunks include `document_id` metadata for per-document filtering.

**Document chat** -- A query mode where `document_id` is passed to `/query`, bypassing intent classification entirely. Queries go directly to vector search (top 15 candidates, top 10 to synthesis) against the specific uploaded document's chunks, then 4B synthesis. The Observatory UI auto-scopes queries after upload.

**Semantic chunking** -- When embeddinggemma is available at upload time, documents are chunked using chonkie's SemanticChunker instead of fixed-size splitting. The chunker uses paragraph boundaries (`\n\n`) as primary delimiters and embedding similarity (Savitzky-Golay smoothed) to detect topic boundaries. This keeps related content (e.g. a conference session's title + speaker + description) in a single chunk. Falls back to fixed-size chunking (800 chars, 80 overlap) if the embedding server is unreachable. Config: `SEMANTIC_CHUNKING_ENABLED`, `SEMANTIC_CHUNKING_THRESHOLD` (0.7), `SEMANTIC_CHUNKING_MAX_TOKENS` (256). See `LlamaServerEmbeddings` in `src/engine/knowledge/semantic_embeddings.py`.

**Paste-to-upload** -- The Observatory UI accepts Cmd-V paste of text content (e.g. copied from a webpage). The pasted text is wrapped as a `.txt` file with a timestamp filename and uploaded through the standard document pipeline. Minimum 50 characters to avoid accidental pastes.

**document_id** -- A slugified identifier derived from the uploaded filename (e.g., `snowflake-fy2025-first50`). Used for metadata filtering in ChromaDB and clean-replace on re-upload.

**OCRClient** -- Standalone async client for GLM-OCR (`src/engine/knowledge/ocr_client.py`). Uses AsyncOpenAI pointed at `http://localhost:9098/v1`. Supports text and table extraction modes via prompt prefix (`"Text Recognition:"` / `"Table Recognition:"`). DPI fallback: tries 150 DPI, falls back to 100 DPI for large pages.

---

## Hardware

**RTX PRO 6000 Blackwell** -- NVIDIA workstation GPU (96GB VRAM). Primary training and inference hardware. Training: gemma3 ~5 min, Qwen3.5-4B FT v8 ~5-15 min (QLoRA on 4B), embeddinggemma ~15s.

**Apple M5 Max** -- Apple Silicon chip (128GB unified memory, Metal4 with Neural Accelerators). Primary laptop inference hardware. Optimized build p50: 1121ms overall, 1115ms select_tool, 922ms synthesis. OCR: 2715ms/page (0.37 pages/sec), doc-chat p50=378ms (fastest of all machines). See [EVAL_RESULTS_2026-04-05.md](../benchmarks/EVAL_RESULTS_2026-04-05.md).

**DGX Spark** -- NVIDIA's compact Grace Blackwell server (128 GB unified memory, GB10 GPU, ARM64 Grace CPU). Deployed and benchmarked: full pipeline runs at p50 2315ms overall (5x slower than RTX due to LPDDR5X bandwidth). Requires optimized build flags and `--no-mmap` at runtime. See [EVAL_RESULTS_2026-04-05.md](../benchmarks/EVAL_RESULTS_2026-04-05.md).

**Metal** -- Apple's GPU compute framework. Used by llama.cpp for inference acceleration on macOS.

**CUDA** -- NVIDIA's GPU compute platform. Used by llama.cpp and PyTorch for training and inference on NVIDIA GPUs.

## UI Features

**Show Mode** -- Cinematic full-screen keynote demo interface (Cmd+Shift+P). Animated canvas orb (audio-reactive waveform during recording, spinning during processing), 5-model activity strip, response panel with smart cards (KPI, bar chart, ranked bars, table) auto-detected from SQL result shape, dark/light theme toggle. See `src/clients/observatory-react/src/components/ShowMode/`.

**OpenWakeWord** -- Wake word / keyword spotting library running entirely in-browser via ONNX Runtime WASM. `openwakeword-wasm-browser` provides AudioWorklet-based detection pipeline (melspectrogram → embedding → VAD → keyword scoring). "Hey Jarvis" keyword used for voice activation. MIT licensed, no API key required, no vendor lock-in. See `src/clients/observatory-react/src/hooks/useWakeWord.ts`.

**Smart Cards** -- Auto-generated visualizations in Show Mode response panel. Detects card type from SQL result shape: single value → KPI (animated counter), time series → bar chart (CSS animated bars), ranked list → horizontal bars, multi-row → styled table. Pure CSS animations, no charting library.
