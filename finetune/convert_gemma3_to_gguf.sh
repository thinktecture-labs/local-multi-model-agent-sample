#!/usr/bin/env bash
# ============================================================
# convert_gemma3_to_gguf.sh — Convert the merged fine-tuned model to GGUF
#
# Run this after `python -m finetune.train_gemma3 --task intent` to produce
# models/gemma3-1b-ft-merged/gemma3-1b-ft-<scenario>-f16.gguf
# which is served by llama-server on port 9090 (--ft flag).
#
# Uses the SPM (SentencePiece) path of the llama.cpp converter —
# the same approach ggml-org uses for official Gemma3 GGUFs.
# This produces tokenizer.ggml.model='llama' with correct token IDs.
#
# Usage:
#   source .venv/bin/activate
#   bash finetune/convert_gemma3_to_gguf.sh
# ============================================================
set -euo pipefail

BOLD="\033[1m"; GREEN="\033[32m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

SCENARIO="${SCENARIO:-nextera}"
MODEL_DIR="models/gemma3-1b-ft-merged"
OUTPUT_GGUF="$MODEL_DIR/gemma3-1b-ft-${SCENARIO}-f16.gguf"
CONVERT_SCRIPT="vendor/llama.cpp/convert_hf_to_gguf.py"

[ -f "$MODEL_DIR/model.safetensors" ] || error "model.safetensors not found — run python -m finetune.train_gemma3 --task intent first"
[ -f "$MODEL_DIR/tokenizer.model"   ] || error "tokenizer.model not found — training should have saved it via tokenizer.save_pretrained()"
[ -f "$CONVERT_SCRIPT" ] || error "convert_hf_to_gguf.py not found — run: git submodule update --init vendor/llama.cpp"

# ─── Backup existing GGUF ────────────────────────────────────────────────────
if [ -f "$OUTPUT_GGUF" ]; then
    BACKUP="$OUTPUT_GGUF.bak.$(date +%Y%m%d_%H%M%S)"
    info "Backing up existing GGUF → $BACKUP"
    cp "$OUTPUT_GGUF" "$BACKUP"
fi

# ─── Dependencies ─────────────────────────────────────────────────────────────

${PIP:-pip} install -q vendor/llama.cpp/gguf-py sentencepiece

# ─── tokenizer_config.json ────────────────────────────────────────────────────
# GemmaTokenizer.save_pretrained() omits add_bos_token and added_tokens_decoder
# which llama.cpp needs for correct BOS handling. We fetch the complete Google
# version and inject our chat_template so /v1/chat/completions works correctly.

${PYTHON:-python} - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, json

p = hf_hub_download("google/gemma-3-1b-it", "tokenizer_config.json")
shutil.copy(p, "models/gemma3-1b-ft-merged/tokenizer_config.json")

cfg_path = "models/gemma3-1b-ft-merged/tokenizer_config.json"
with open(cfg_path) as f:
    tc = json.load(f)
tc["chat_template"] = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)
with open(cfg_path, "w") as f:
    json.dump(tc, f, indent=2)
    f.write("\n")
print("  tokenizer_config.json + chat_template ✓")
EOF

# ─── Convert ──────────────────────────────────────────────────────────────────

${PYTHON:-python} "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_GGUF" \
    --outtype f16

info "GGUF ready: $OUTPUT_GGUF ($(du -h "$OUTPUT_GGUF" | cut -f1))"
echo "  Restart the inference server: bash scripts/start_servers.sh --bg --ft"
