#!/usr/bin/env bash
# ============================================================
# download_qwen_models.sh — Download Qwen 3.5 GGUFs for testing
#
# Downloads from unsloth HuggingFace repos:
#   1. Qwen3.5-2B (Q8_0, 2.01 GB) — Thinker (replaces gemma3-1B)
#   2. Qwen3.5-4B (Q4_K_M, 2.74 GB) — Doer (tool calling)
#
# Embedding model (embeddinggemma) is kept as-is.
#
# Usage:
#   bash scripts/download_qwen_models.sh
#
# After download:
#   cp .env.qwen .env.local
#   bash scripts/start_servers.sh
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

# Find Python with huggingface_hub
if [ -f .venv/bin/python ] && .venv/bin/python -c "import huggingface_hub" 2>/dev/null; then
    PYTHON=".venv/bin/python"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    PYTHON="python3"
else
    error "No Python with huggingface_hub found. Run: .venv/bin/pip install huggingface_hub"
fi

download_gguf() {
    local repo="$1" filename="$2" dest_dir="$3"
    local dest_path="$dest_dir/$filename"

    if [ -f "$dest_path" ]; then
        info "Already exists: $dest_path ($(du -h "$dest_path" | cut -f1))"
        return
    fi

    mkdir -p "$dest_dir"
    echo -e "  Downloading ${BOLD}$filename${RESET} from $repo ..."

    $PYTHON -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$repo', '$filename', local_dir='$dest_dir')
"
    if [ -f "$dest_path" ]; then
        info "Downloaded: $dest_path ($(du -h "$dest_path" | cut -f1))"
    else
        error "Download failed: $dest_path"
    fi
}

echo -e "\n${BOLD}Downloading Qwen 3.5 GGUFs for testing${RESET}\n"

# ─── 1. Qwen3.5-2B (Q8_0) — Thinker ────────────────────────────────────────
# Replaces gemma3-1B for intent classification, query rewriting, formatting
# 2B is the closest Qwen 3.5 size to gemma3-1B (no 1.5B exists)
echo -e "${BOLD}1. Qwen3.5-2B (Thinker — replaces gemma3-1B)${RESET}"
download_gguf \
    "unsloth/Qwen3.5-2B-GGUF" \
    "Qwen3.5-2B-Q8_0.gguf" \
    "models/qwen3.5-2b"

# ─── 2. Qwen3.5-4B (Q4_K_M) — Doer + Vision ────────────────────────────────
# Qwen 3.5-4B for tool calling — uses native OpenAI function calling.
# Also replaces gemma3-4B for RAG synthesis and vision
echo -e "\n${BOLD}2. Qwen3.5-4B (Doer — tool calling)${RESET}"
download_gguf \
    "unsloth/Qwen3.5-4B-GGUF" \
    "Qwen3.5-4B-Q4_K_M.gguf" \
    "models/qwen3.5-4b"

# ─── Summary ────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}VRAM estimate:${RESET}"
echo "  Qwen3.5-2B Q8_0:   ~2.0 GB"
echo "  Qwen3.5-4B Q4_K_M: ~2.7 GB (shared for function + vision)"
echo "  embeddinggemma:     ~0.3 GB (unchanged)"
echo "  Total:              ~5.0 GB (vs ~5.6 GB Gemma)"
echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo "  1. Activate Qwen config:  cp .env.qwen .env.local"
echo "  2. Start servers:         bash scripts/start_servers.sh"
echo "  3. Run evals:"
echo "       python -m finetune.eval_gemma3          # intent classification"
echo "       python -m finetune.eval_tool_routing     # tool routing"
echo "       python -m finetune.eval_vision          # vision (if mmproj available)"
echo ""
echo "  To revert to Gemma:       rm .env.local && bash scripts/start_servers.sh"
