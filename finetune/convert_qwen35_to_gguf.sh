#!/usr/bin/env bash
# ============================================================
# convert_qwen35_to_gguf.sh — Debug/alternate Qwen3.5-4B FT conversion.
#
# *** NOT THE PRODUCTION PIPELINE ***
#
# Production Qwen GGUF is **Q4_K_M (~2.5 GB)** produced by Unsloth's in-band
# export inside `finetune/train_qwen35_toolcalling.py`. That artifact is what
# `scenarios/<name>.json:function_gguf_ft` points at and what
# `start_servers.sh --ft` loads.
#
# This script produces a **real f16 GGUF (~8 GB)** at a debug-suffixed path
# (`*-debug-f16.gguf`) — useful for inspecting weights at higher precision or
# for downstream manual quantization to Q4_K_M / Q5_K_M / Q8_0 via
# `llama-quantize`. It is NOT picked up by the scenario config by default;
# to serve the f16, point FUNCTION_GGUF env var at the f16 file explicitly.
#
# Usage (debug only):
#   bash finetune/convert_qwen35_to_gguf.sh
#
# To run with the venv's Python (e.g. on rtx-pro-6000 over SSH where the
# venv activate doesn't work cleanly):
#   PIP=.venv/bin/pip PYTHON=.venv/bin/python3 bash finetune/convert_qwen35_to_gguf.sh
# ============================================================
set -euo pipefail

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

SCENARIO="${SCENARIO:-nextera}"
MODEL_DIR="models/qwen3.5-4b-toolcalling-ft-merged"
OUTPUT_GGUF="$MODEL_DIR/qwen3.5-4b-toolcalling-ft-${SCENARIO}-debug-f16.gguf"
CONVERT_SCRIPT="vendor/llama.cpp/convert_hf_to_gguf.py"

# ─── Checks ───────────────────────────────────────────────────────────────────
# Unsloth shards as model.safetensors-000XX-of-000XX
ls "$MODEL_DIR"/*.safetensors 2>/dev/null | head -1 | grep -q . || \
    error "No safetensors found in $MODEL_DIR — run python -m finetune.train_qwen35_toolcalling first"

[ -f "$CONVERT_SCRIPT" ] || \
    error "convert_hf_to_gguf.py not found — run: git submodule update --init vendor/llama.cpp"

# ─── Backup ───────────────────────────────────────────────────────────────────
if [ -f "$OUTPUT_GGUF" ]; then
    BACKUP="$OUTPUT_GGUF.bak.$(date +%Y%m%d_%H%M%S)"
    info "Backing up existing GGUF → $BACKUP"
    cp "$OUTPUT_GGUF" "$BACKUP"
fi

# ─── Dependencies ─────────────────────────────────────────────────────────────
.venv/bin/pip install -q vendor/llama.cpp/gguf-py

# ─── Verify chat template ─────────────────────────────────────────────────────
.venv/bin/python - <<'EOF'
import json
from pathlib import Path

cfg_path = Path("models/qwen3.5-4b-toolcalling-ft-merged/tokenizer_config.json")
if not cfg_path.exists():
    print("  tokenizer_config.json not found — skipping template check")
    exit(0)

with open(cfg_path) as f:
    tc = json.load(f)

template = tc.get("chat_template", "")
if "| items" in template and "is mapping" not in template:
    print("  WARNING: Chat template may have the Qwen3.5 '| items' bug.")
    print("  Use --chat-template-file with a fixed template at runtime.")
else:
    print("  Chat template looks clean ✓")
EOF

# ─── Convert ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Converting $MODEL_DIR → $OUTPUT_GGUF…${RESET}"

.venv/bin/python "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_GGUF" \
    --outtype f16

info "GGUF ready: $OUTPUT_GGUF ($(du -h "$OUTPUT_GGUF" | cut -f1))"

# ─── Next steps ───────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo "  1. Start the fine-tuned server (RTX):"
echo "       FUNCTION_PORT=9091 FUNCTION_MODEL=qwen35-toolcalling-ft \\"
echo "       vendor/llama.cpp/build/bin/llama-server \\"
echo "         --model $OUTPUT_GGUF --port 9091 --host 127.0.0.1 \\"
echo "         --parallel 1 --n-gpu-layers 999 --jinja \\"
echo "         --chat-template-kwargs '{\"enable_thinking\":false}' \\"
echo "         --reasoning-budget 0 --top-k 20 --temp 1.0 --top-p 0.95 \\"
echo "         --log-disable &"
echo ""
echo "  2. Evaluate:"
echo "       python -m finetune.eval_tool_routing --save results/qwen35_finetuned.json"
echo ""
echo "  3. Compare against baseline:"
echo "       python -m finetune.eval_tool_routing --compare \\"
echo "           results/qwen35_baseline.json results/qwen35_finetuned.json"
