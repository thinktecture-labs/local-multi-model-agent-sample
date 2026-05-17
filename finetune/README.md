# Fine-Tuning Pipeline

End-to-end fine-tuning of the production model stack for a domain scenario.

> **Read first**: [`docs/benchmarks/FINE_TUNING_INSIGHTS.md`](../docs/benchmarks/FINE_TUNING_INSIGHTS.md)
> — the technical narrative behind the choices in this directory (EVA + rsLoRA,
> format-mismatch traps, train/eval overlap detection, run-to-run variance fixes).
> If you read one file before training, read that one.

---

## Model → script map

The production stack has four trained models per scenario plus one LogReg classifier.
Pick the script that matches the model you want to fine-tune.

| Model | Role | Train | Convert to GGUF | Eval |
|------|------|-------|------------------|------|
| **gemma3-1B** | Direct answer + tool-result synthesis (intent fallback) | `train_gemma3.py` | `convert_gemma3_to_gguf.sh` | `eval_gemma3.py` |
| **gemma3-4B** | RAG synthesis (+ vision) | `train_gemma3_4b.py` | `convert_gemma3_4b_to_gguf.sh` | `eval_rag_groundtruth.py` |
| **Qwen3.5-4B** | Tool calling (sql_query / calculator) | `train_qwen35_toolcalling.py` | `convert_qwen35_to_gguf.sh` | `eval_tool_routing.py` |
| **EmbeddingGemma 300M** | RAG retrieval | `train_embeddinggemma.py` | `convert_embeddinggemma_to_gguf.sh` | `eval_embeddinggemma.py` |
| **LogReg classifier** | Fast intent (replaces 1B generative) | `training/train_intent_logreg.py` | (joblib, not GGUF) | `eval_gemma3.py --logreg` |

Plus an orchestrator: [`pipeline.py`](pipeline.py) runs the full sequence (`--train-all`).

---

## Skip the training: download published FTs

If you're just trying the demo, you don't need to retrain anything. The
fine-tuned GGUFs for the reference Nextera scenario are published in the
[Thinktecture AG HuggingFace org](https://huggingface.co/thinktecture), grouped
in a single collection for easy discovery.

```bash
# Download (defaults to HF_NAMESPACE=thinktecture, SCENARIO=nextera)
bash scripts/download_ft_models.sh

# Start FT servers + run the demo
bash scripts/start_servers.sh --bg --ft
python demo.py
```

The download pulls five HF repos (one per FT model) from the
`thinktecture/*-ft-nextera` collection. Each repo carries its own model card
adapted from [`MODEL_CARDS.md`](MODEL_CARDS.md) and ships under the relevant
base-model license — see [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

> **Re-publishing FT models yourself** (e.g. for a new scenario): the
> [`upload_ft_to_hf.sh`](upload_ft_to_hf.sh) script handles the push side,
> including auto-creating an HF Collection that groups the five models. You'll
> need to be a member of `HF_NAMESPACE` and accept the Gemma Terms once on
> your HF account before uploading any Gemma derivative.
>
> Example for the Thinktecture AG flow:
> ```bash
> # Dry-run first (prints what would be uploaded, no push)
> DRY_RUN=1 bash finetune/upload_ft_to_hf.sh
> # Real upload — defaults to HF_NAMESPACE=thinktecture, creates a collection
> bash finetune/upload_ft_to_hf.sh
> ```

---

## Canonical end-to-end sequence (train from scratch)

```bash
source .venv/bin/activate
pip install -r ../requirements-finetune.txt

# 1. Prepare data (gemma3 + embedding from interaction logs; qwen35 uses curated set)
python -m finetune.data_prep

# 2. Baseline eval — run BEFORE training so you have a comparison point
bash scripts/start_servers.sh --bg              # base models, ports 9090–9093
bash scripts/run_all_evals.sh --label baseline

# 3. Train the full stack (RTX/CUDA recommended; runtime ~6h on RTX PRO 6000)
python -m finetune.pipeline --train-all

# 4. Train the LogReg intent classifier (uses the FT EmbeddingGemma)
python -m training.train_intent_logreg

# 5. Convert each FT model to GGUF
bash finetune/convert_gemma3_to_gguf.sh
bash finetune/convert_gemma3_4b_to_gguf.sh
bash finetune/convert_qwen35_to_gguf.sh           # produced in-band by Unsloth
bash finetune/convert_embeddinggemma_to_gguf.sh

# 6. Restart with FT servers, re-run evals
bash scripts/start_servers.sh --bg --ft
bash scripts/run_all_evals.sh --label finetuned

# 7. Compare
python -m finetune.collect_results --matrix results/matrix.json
```

After step 7 you have:
- `results/baseline_*.json` and `results/finetuned_*.json` — per-eval JSONs
- `results/matrix.json` — aggregated comparison
- Headline numbers in `docs/benchmarks/FINE_TUNING_INSIGHTS.md` are reproducible from this set

---

## Hardware matrix

| Step | RTX PRO 6000 (96 GB) | Apple M-series (Metal) | Notes |
|------|---------------------|------------------------|-------|
| **EmbeddingGemma** | ~5 min | ~30 min | Sentence-Transformers; CPU/Metal works |
| **Gemma3-1B** | ~25 min (LoRA, EVA init) | ~3 h (QLoRA fallback) | r=16 default |
| **Gemma3-4B** | ~45 min (LoRA) | ~5 h | r=16, target_modules=q/v |
| **Qwen3.5-4B** | ~35 min (Unsloth QLoRA) | not recommended | Unsloth requires CUDA |
| **LogReg** | ~2 min (on either) | ~2 min | scikit-learn on EmbeddingGemma vectors |
| **Eval (full matrix)** | ~25 min | ~70 min | All 12 eval scripts |

> Unsloth is **required** for the Qwen3.5 training path. It only supports CUDA.
> Add `unsloth>=2024.x` to your environment manually — it's intentionally not
> in [`requirements-finetune.txt`](../requirements-finetune.txt) so non-CUDA
> users don't get a confusing install failure on import.

---

## Determinism

All four training scripts call a shared determinism helper:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for cuBLAS determinism
```

Same seed + same input data → bit-identical FT outputs on the same GPU.
See `FINE_TUNING_INSIGHTS.md` §5 for the rationale and the failure modes
(EVA random init, dataloader shuffle without seed, cuDNN nondeterminism).

---

## What's tracked vs. generated

The whole [`models/`](../models/) directory is gitignored. Both the base GGUFs
(downloaded by `setup.sh`) and the fine-tuned outputs (produced by training)
stay local. Everything in `models/` is reproducible from the scripts in
[`scripts/`](../scripts/) and this directory:

| Path | Produced by |
|---|---|
| `models/gemma3/`, `models/embeddinggemma/`, `models/gemma3-4b/` | `bash setup.sh` (HF download) |
| `models/qwen3.5-4b/`, `models/qwen3.5-35b-a3b/` | `bash scripts/download_qwen_models.sh` |
| `models/whisper/`, `models/piper/` | `bash scripts/setup_voice.sh` |
| `models/glm-ocr/` | `bash scripts/setup_ocr.sh` |
| `models/gemma3-1b-ft-merged/`, `models/gemma3-4b-ft-merged/`, `models/qwen3.5-4b-toolcalling-ft-merged/`, `models/embeddinggemma-300m-ft-merged/` | `python -m finetune.pipeline --train-all` (slow) OR `bash scripts/download_ft_models.sh` (fast) |
| `models/intent-logreg/model.joblib` + `meta.json` | `python -m training.train_intent_logreg` (requires FT EmbeddingGemma running) |

`meta.json` holds the LogReg classifier's training statistics (class labels,
embedding dim, dataset size, CV accuracy). It's not a secret, but it's
regenerated every training run, so we keep it gitignored.

---

## Add a new scenario

Each scenario is a single JSON file under [`scenarios/`](../scenarios/). To add one:

1. Copy `scenarios/nextera.json` → `scenarios/<your-name>.json`. Edit
   `brand`, `language`, `paths.docs_dir_name`, `paths.training_data_dir`,
   `paths.training_data_suffix` (e.g. `_yourname`), prompts, and SQL schema.
2. Create `data/<paths.docs_dir_name>/` and drop your knowledge base markdown there.
3. Create `data/<paths.training_data_dir>/` with parallel `_<suffix>.jsonl` files
   for each training set. Eval scripts auto-discover via `f"eval_X_{SCENARIO}.jsonl"`.
4. Implement your scenario's SQL schema in a Python loader (see `data/loader.py`).
5. Run the canonical sequence above with `SCENARIO=<your-name>` in the environment
   or `--scenario <your-name>` on each shell script.

No engine code changes are required for a new scenario. The
[`SCENARIO_PLAYBOOK`](../docs/guides/SCENARIO_PLAYBOOK.md) has the long-form
walkthrough with the gotchas.

---

## Licensing

Fine-tuned outputs are **derivatives** of their base models. The repo's MIT/Apache
license covers your training code, but the merged GGUFs inherit the base model's terms.
See [`MODEL_LICENSES.md`](MODEL_LICENSES.md) for the per-model details before
publishing or redistributing trained GGUFs.

---

## Where to look when things break

| Symptom | First place to look |
|---------|--------------------|
| FT model returns gibberish at inference | [`FINE_TUNING_INSIGHTS.md`](../docs/benchmarks/FINE_TUNING_INSIGHTS.md) §2 (format mismatch) |
| Eval scores improbably high after training | Eval/training data overlap — run `tests/unit/test_eval_overlap.py` |
| Run-to-run variance > 1pp | `FINE_TUNING_INSIGHTS.md` §5 (determinism) — likely missing CUBLAS_WORKSPACE_CONFIG |
| Tool-call format unrecognised by llama-server | Check that GGUF was converted with the same `chat_template_jinja` |
| `train_qwen35_toolcalling` import fails | Install Unsloth: `pip install unsloth` (CUDA-only) |
| Out-of-memory during gemma3-4B training | Drop to QLoRA: pass `--qlora` to `pipeline.py` |
