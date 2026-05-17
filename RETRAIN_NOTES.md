# Retrain Notes — Training-Data Corrections (post-2026-05-15 architecture review)

> **Status:** the published HuggingFace FT models predate the training-data
> fixes landed in commit [`62addc1`](https://github.com/thinktecture-labs/local-multi-model-agent-slm/commit/62addc1).
> Until a fresh retrain ships, the HF weights and the JSONLs in
> `data/training-data/` are out of sync.

## Why a retrain is needed

The pre-publication architecture review uncovered two semantic bugs in the
training data that have been baked into the currently-published FT models:

1. **€1,499 baked in as the Professional plan price.** Real list price is
   €999 per `data/loader.py` and `scenarios/nextera.json`. The model has
   been trained to answer pricing questions with the wrong number.
   *(Caveat: customer GreenOps BV genuinely pays €1,499 MRR on a custom
   contract — `data/loader.py:135`. That single seed row is correct; the
   other 1,499 mentions in training data were treating it as the plan
   price, which is the bug.)*

2. **`FROM revenue` references a non-existent table.** Real schema is
   `sales(year, quarter, revenue, new_customers, churn_rate,
   arr_growth_pct)`. The Qwen FT model has been trained on multi-turn
   examples that emit SQL the production database cannot execute.
   Additionally those examples used fabricated 2024 quarterly values
   (42 800 / 55 300 / 38 600 / 61 400) that don't match either the real
   2024 seed (55 100 / 68 300 / 84 900 / 103 200) or the 2023 seed.

A third schema bug was caught in the same sweep: `monthly_price` column
doesn't exist (real: `price_monthly`); product example fixtures used
short names like `'Enterprise'` instead of the seeded `'Nextera %'` form.

All three were fixed at source in commit `62addc1`:

- `finetune/data_prep_qwen35_toolcalling.py` (22× `1499→999`, 20×
  `monthly_price→price_monthly`, 4× `FROM revenue→FROM sales`, etc.)
- `data/training-data/qwen35_toolcalling.jsonl` (regenerated from the
  fixed `data_prep`)
- `data/training-data/gemma3_intent.jsonl` (4× `1499→999`)
- `data/training-data/gemma3_synthesis.jsonl` (2× `1499→999`, plus the
  legitimate GreenOps `1499` MRR preserved)
- `data/training-data/tool_routing_2tool.jsonl` (8× `1499→999`)

Pull the latest before retraining:

```bash
git pull origin chore/public-release-cleanup
```

## Which models need retraining

| Model | Training data file | Changed? | Retrain |
|---|---|---|---|
| **Qwen3.5-4B FT v8** (tool calling) | `data/training-data/qwen35_toolcalling.jsonl` | ✅ Yes — SQL schema + numbers + pricing | **Required** |
| **Gemma3-1B FT** (intent + synthesis) | `data/training-data/gemma3_intent.jsonl` + `gemma3_synthesis.jsonl` | ✅ Yes — 6 pricing rows | **Required** |
| EmbeddingGemma 300M FT (retrieval) | `data/training-data/embeddinggemma_retrieval.jsonl` | ❌ No change | Skip |
| Gemma3-4B FT (RAG synthesis + vision) | `data/training-data/gemma3_4b_synthesis.jsonl` | ❌ No change | Skip |
| LogReg intent classifier | depends on FT EmbeddingGemma + `gemma3_intent.jsonl` | EmbeddingGemma unchanged; intent data did change | **Required** (re-runs in seconds — see below) |

## Where to run

The canonical training host is the **RTX PRO 6000** (96 GB VRAM):

- SSH alias: `rtx-pro-6000`
- Repo path: `~/sources/local-multi-model-agent-slm`
- Activate venv via direct path (per
  [`feedback_rtx_venv.md`](../.claude/projects/-Users-christianweyer-sources-local-multi-model-agent-slm/memory/feedback_rtx_venv.md)
  — `source .venv/bin/activate` doesn't work over SSH; use
  `.venv/bin/python3` directly)

The M5 Max also works (Apple Silicon, Metal/MPS) but training is slower
on the 4B Qwen model.

## Retrain procedure

```bash
# ── 1. Sync repo on RTX ─────────────────────────────────────────────────
ssh rtx-pro-6000 'cd ~/sources/local-multi-model-agent-slm && git pull'

# ── 2. Train Qwen3.5-4B FT v8 (QLoRA) ────────────────────────────────────
# ~5–15 min on RTX, ~30–45 min on M5 Max
ssh rtx-pro-6000 '\
  cd ~/sources/local-multi-model-agent-slm && \
  .venv/bin/python3 -m finetune.train_qwen35_toolcalling'

# ── 3. Train Gemma3-1B FT (intent + synthesis) ──────────────────────────
# ~3–5 min on RTX, ~10–15 min on M5 Max
ssh rtx-pro-6000 '\
  cd ~/sources/local-multi-model-agent-slm && \
  .venv/bin/python3 -m finetune.train_gemma3 --task both'

# ── 4. Convert Gemma3 to GGUF (Qwen GGUF is emitted in-band by Unsloth in step 2) ─
# `convert_qwen35_to_gguf.sh` is a debug-only path now (real f16, ~8 GB) and is
# NOT part of the production conversion; production Qwen GGUF is the Q4_K_M
# emitted directly by step 2 above (Unsloth in-band).
ssh rtx-pro-6000 '\
  cd ~/sources/local-multi-model-agent-slm && \
  PIP=.venv/bin/pip PYTHON=.venv/bin/python3 bash finetune/convert_gemma3_to_gguf.sh'
# NOTE 1 (gemma3 tokenizer.model): if the convert aborts with "tokenizer.model not found",
# fetch the SPM file from Google before re-running:
#   ssh rtx-pro-6000 'cd ~/sources/local-multi-model-agent-slm && \
#     .venv/bin/python3 -c "from huggingface_hub import hf_hub_download; import shutil; \
#     p = hf_hub_download(\"google/gemma-3-1b-it\", \"tokenizer.model\"); \
#     shutil.copy(p, \"models/gemma3-1b-ft-merged/tokenizer.model\")"'
# NOTE 2 (Qwen GGUF subdir): Unsloth's `save_pretrained_gguf` writes the file into a
# `<name>_gguf/` subdirectory rather than directly at `<name>.gguf` (the path the
# scenario config expects). After step 2 completes, fix this up:
#   ssh rtx-pro-6000 'cd ~/sources/local-multi-model-agent-slm && \
#     mv models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-nextera-q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf \
#        models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-nextera-q4_k_m.gguf && \
#     rm -rf models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-nextera-q4_k_m_gguf \
#            models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-nextera-q4_k_m'

# ── 5. Retrain the LogReg intent classifier ─────────────────────────────
# (the intent training data shifted; the embedding model didn't)
# Requires the FT EmbeddingGemma server running on port 9092.
# ~2 min total — re-embeds the training set + fits LR.
# IMPORTANT precondition: verify the FT EmbeddingGemma GGUF exists at the
# path declared by `scenarios/<scenario>.json:embedding_gguf_ft` BEFORE
# starting servers. If it's missing (or under an older naming convention
# like `embeddinggemma-ft-merged/` instead of `embeddinggemma-300m-ft-merged/`),
# `start_servers.sh --ft` will silently fall back to the BASE embedder and the
# LogReg refit will be trained against the wrong vector space. Quick check:
#   ssh rtx-pro-6000 'cd ~/sources/local-multi-model-agent-slm && \
#     ls -la $(jq -r ".models.embedding_gguf_ft" scenarios/nextera.json) 2>&1'
ssh rtx-pro-6000 '\
  cd ~/sources/local-multi-model-agent-slm && \
  .venv/bin/python3 -m training.train_intent_logreg'

# ── 6. Re-evaluate on RTX (catches any regression vs documented numbers) ─
# IMPORTANT: do NOT pipe `start_servers.sh --bg` through `grep` etc. — the
# backgrounded llama-server children can die when the pipe closes. Let it run
# straight and check the readiness output:
ssh rtx-pro-6000 '\
  cd ~/sources/local-multi-model-agent-slm && \
  bash scripts/start_servers.sh --bg --ft && \
  bash scripts/run_all_evals.sh'
```

## After retrain — re-publish + reconcile docs

```bash
# ── 7. SCP the regenerated GGUFs to your laptop (or upload from RTX) ─────
# From the laptop:
scp rtx-pro-6000:~/sources/local-multi-model-agent-slm/models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-nextera-q4_k_m.gguf \
    models/qwen3.5-4b-toolcalling-ft-merged/
scp rtx-pro-6000:~/sources/local-multi-model-agent-slm/models/gemma3-1b-ft-merged/gemma3-1b-ft-nextera-f16.gguf \
    models/gemma3-1b-ft-merged/
scp -r rtx-pro-6000:~/sources/local-multi-model-agent-slm/models/intent-logreg \
    models/

# ── 8. Re-upload to Hugging Face ─────────────────────────────────────────
# Uses HF_NAMESPACE=thinktecture, SCENARIO=nextera by default.
bash finetune/upload_ft_to_hf.sh
```

### Reconciling documented eval numbers

After running `scripts/run_all_evals.sh` in step 6, compare the new numbers
against the documented headline metrics. If anything shifts by more than
~1 percentage point of run-to-run noise, update:

- **`finetune/MODEL_CARDS.md`** — per-model "Reference eval (Nextera)" rows
- **`docs/benchmarks/FINE_TUNING_INSIGHTS.md`** — Quick Reference table
  + Section 7 "Full Numbers Run" table
- **`docs/benchmarks/EVAL_RESULTS_2026-04-05.md`** — date-stamped snapshot.
  If the new run replaces the snapshot, rename to
  `EVAL_RESULTS_YYYY-MM-DD.md` and adjust the inbound links from
  `MODEL_CARDS.md` and `FINE_TUNING_INSIGHTS.md`.

### Measured behaviour after the 2026-05-15 retrain

- **Qwen v9 SQL execution: 100% (79/79)** — valid SQL on every chain. No more
  `FROM revenue` crashes.
- **Pricing-question answers** from Gemma3-1B FT now consistently say
  "€999/month" for Professional, not "€1,499/month".
- **LogReg intent accuracy improved to 99.4%** (was 97.2% pre-retrain) — the
  corrected training data + the FT EmbeddingGemma vector space the LR was
  fitted on combined for a small but real gain.
- **Gemma3-ft intent (fallback path) improved to 96.7%** (was 93.3% pre-retrain)
  — the `--task both` retrain on corrected data outperformed the v5
  `--task intent` 95.0%.
- **Multi-step chain accuracy: 97.5% (78/80)** — matches the documented v8 chain-shape
  number AND adds 100% SQL execution validity. v8 era's 96.2% was inflated by
  training on broken SQL (`FROM revenue`) the eval didn't catch; v9 produces
  valid SQL throughout. Recovery to 97.5% required four prompt-engineering
  commits alongside the retrain (3db64e4 concretize→Qwen, e6a3276+c8d4eb5+0c644fe
  fewshot fixes, 118b6a1 synthesis→Qwen, f69fd41 decomposer rules). System is
  fully deterministic — `client.call_function(deterministic=True)` and
  `client.generate(deterministic=True)` plumbed end-to-end; verified
  byte-identical across 3 back-to-back runs. The 2 remaining failures are
  documented edge cases — see [docs/benchmarks/FINE_TUNING_INSIGHTS.md §3b](docs/benchmarks/FINE_TUNING_INSIGHTS.md#3b-qwen35-4b-ft-v8--current-production-2026-03-19).

### Lessons learned (2026-05-15 retrain — items to fold into a future cleanup pass)

These came up during the retrain itself and are tracked for follow-up:

- **Qwen GGUF naming was misleading** — the shipped file was named `…-f16.gguf`
  but the bits were Q4_K_M. Fixed in commit `577cb2b`; the file is now
  `…-q4_k_m.gguf`. Existing HF repos still under the old `-f16` name need
  re-upload via `finetune/upload_ft_to_hf.sh` (which has been updated).
- **Multi-step concretize + synthesis routed through Qwen FT** (commits `3db64e4`,
  `118b6a1`) — gemma3-1B was unreliable at substituting SQL-result values into
  the right arithmetic shape and at quoting calculator results verbatim. Qwen
  reads structured context faithfully. This is an architectural change worth
  knowing about when reading older traces.
- **Decomposer + synthesis fewshot leakage** (commits `e6a3276`, `c8d4eb5`,
  `0c644fe`, `f69fd41`) — anchor examples in `scenarios/<name>.json` prompts
  bled onto unrelated queries. Diversified or removed where appropriate.
- **Convert script lessons:**
  - `convert_gemma3_to_gguf.sh` needs `tokenizer.model` (SPM) which
    `tokenizer.save_pretrained()` doesn't write. See NOTE 1 under step 4 for
    the one-liner that fetches it from HF.
  - On RTX SSH, both convert scripts must be invoked with
    `PIP=.venv/bin/pip PYTHON=.venv/bin/python3` (PEP 668 blocks pip's default
    system-python target).
  - `convert_qwen35_to_gguf.sh` is **debug-only** now (real f16 ~8 GB at a
    `-debug-f16.gguf` path). Production uses Unsloth's in-band Q4_K_M from
    `train_qwen35_toolcalling.py` — see commit `577cb2b` for the rationale.

## Why not retrain everything

EmbeddingGemma's training data (`embeddinggemma_retrieval.jsonl`) and
Gemma3-4B's training data (`gemma3_4b_synthesis.jsonl`) were unaffected by
the data-fix commit — no `1499`, no `FROM revenue`, no schema drift. The
LogReg classifier depends on EmbeddingGemma's vector space; since the
embedding model is unchanged, an LR refit on the corrected intent JSONL
is sufficient (no full EmbeddingGemma retraining needed).
