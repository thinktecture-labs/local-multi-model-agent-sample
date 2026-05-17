#!/usr/bin/env bash
# ============================================================
# download_ft_models.sh — Pull fine-tuned GGUFs from HuggingFace.
#
# Companion to finetune/upload_ft_to_hf.sh — runs the reverse direction
# so consumers of the repo can `bash setup.sh && bash scripts/download_ft_models.sh`
# instead of spending 6 hours fine-tuning to see the demo at full quality.
#
# Optional env:
#   HF_NAMESPACE    HuggingFace org/user hosting the FT models (default: thinktecture)
#   SCENARIO        Scenario to download (default: nextera)
#   HF_TOKEN        Required only if the FT repos are gated/private
#
# Prereqs:
#   1. Run `bash setup.sh` first — base models + venv + huggingface_hub.
#   2. Accept Gemma Terms at https://huggingface.co/google/gemma-3-1b-it
#      (and the 4b + embeddinggemma pages). The FT repos may also require
#      a click-through depending on how the namespace owner published them.
#
# What this pulls:
#   - gemma3-1b-ft-<scenario>-f16              → models/gemma3-1b-ft-merged/
#   - gemma3-4b-ft-<scenario>-q4_k_m           → models/gemma3-4b-ft-merged/
#   - qwen3.5-4b-toolcalling-ft-<scenario>-q4_k_m → models/qwen3.5-4b-toolcalling-ft-merged/
#   - embeddinggemma-300m-ft-<scenario>-q8_0   → models/embeddinggemma-300m-ft-merged/
#   - intent-logreg-<scenario>                 → models/intent-logreg/
#
# The 4B synthesis model is the Q4_K_M production artifact (~2.5 GB) — validated
# identical-quality to F16 with ~3x bandwidth headroom. The F16 variant remains
# published at gemma3-4b-ft-<scenario>-f16 on HF for A/B comparison; pull it
# manually with: HF_NAMESPACE=... GGUF_VARIANT=f16 …
#
# Each repo holds the GGUF plus the model card. Run `bash scripts/start_servers.sh
# --bg --ft` after the download to wire the FT servers up.
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

HF_NAMESPACE="${HF_NAMESPACE:-thinktecture}"
SCENARIO="${SCENARIO:-nextera}"

# ─── HF cache: fall back to repo-local .hf-cache if global is unwritable ────
# Observed in the wild: $HOME/.cache/huggingface/hub gets created by an earlier
# `sudo` invocation and ends up root-owned. Subsequent unprivileged hf_hub_download
# calls then emit "[Errno 13] Permission denied" and the xet sidecar fails with
# 416 errors. Default to a repo-local cache that's always writable by the current
# user; users who want the global cache can still set HF_HOME explicitly.
if [ -z "${HF_HOME:-}" ]; then
    GLOBAL_HF_HUB="${HOME}/.cache/huggingface/hub"
    if [ -d "$GLOBAL_HF_HUB" ] && [ ! -w "$GLOBAL_HF_HUB" ]; then
        export HF_HOME="$(pwd)/.hf-cache"
        mkdir -p "$HF_HOME"
        warn "Global HF cache ($GLOBAL_HF_HUB) is not writable — falling back to HF_HOME=$HF_HOME"
    fi
fi

# ─── Resolve Python with huggingface_hub ────────────────────────────────────
if [ -f .venv/bin/python ] && .venv/bin/python -c "import huggingface_hub" 2>/dev/null; then
    PYTHON=".venv/bin/python"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    PYTHON="python3"
else
    error "huggingface_hub not installed — run: .venv/bin/pip install huggingface_hub"
fi

# ─── Per-model download table ──────────────────────────────────────────────
# Format: "<hf-repo-suffix>|<local-target-dir>|<expected-filename-prefix>"
DOWNLOADS=(
    "gemma3-1b-ft-${SCENARIO}-f16|models/gemma3-1b-ft-merged|gemma3-1b-ft-${SCENARIO}-f16"
    "gemma3-4b-ft-${SCENARIO}-q4_k_m|models/gemma3-4b-ft-merged|gemma3-4b-ft-${SCENARIO}-q4_k_m"
    "qwen3.5-4b-toolcalling-ft-${SCENARIO}-q4_k_m|models/qwen3.5-4b-toolcalling-ft-merged|qwen3.5-4b-toolcalling-ft-${SCENARIO}-q4_k_m"
    "embeddinggemma-300m-ft-${SCENARIO}-q8_0|models/embeddinggemma-300m-ft-merged|embeddinggemma-300m-ft-${SCENARIO}-q8_0"
    "intent-logreg-${SCENARIO}|models/intent-logreg|"
)

echo ""
info "Downloading fine-tuned models from HF namespace: ${HF_NAMESPACE}"
echo "  Scenario: ${SCENARIO}"
echo ""

for entry in "${DOWNLOADS[@]}"; do
    IFS='|' read -r hf_suffix target_dir _prefix <<< "$entry"
    repo_id="${HF_NAMESPACE}/${hf_suffix}"

    mkdir -p "$target_dir"
    echo -e "${BOLD}→${RESET} ${repo_id}"
    echo "  → $target_dir"

    "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${repo_id}',
    local_dir='${target_dir}',
    local_dir_use_symlinks=False,
)
" || warn "Repo not found or access denied: ${repo_id}"

    echo ""
done

info "Done. Next:"
echo "  bash scripts/start_servers.sh --bg --ft   # start FT servers"
echo "  python demo.py                            # smoke test"
