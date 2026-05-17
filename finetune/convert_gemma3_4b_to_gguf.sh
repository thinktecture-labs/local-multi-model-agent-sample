#!/usr/bin/env bash
# ============================================================
# convert_gemma3_4b_to_gguf.sh — Convert merged fine-tuned 4B model to GGUF
#
# Run after `python -m finetune.train_gemma3_4b` to produce both:
#   - models/gemma3-4b-ft-merged/gemma3-4b-ft-<scenario>-f16.gguf      (intermediate)
#   - models/gemma3-4b-ft-merged/gemma3-4b-ft-<scenario>-q4_k_m.gguf   (production synthesis artifact)
#
# The Q4_K_M variant is what scenarios/<scenario>.json points at by default —
# validated identical-quality to F16 on the 80-query RAG groundtruth set with
# ~3x memory-bandwidth headroom. F16 is kept on disk as a rollback target.
#
# Usage:
#   SCENARIO=nextera bash finetune/convert_gemma3_4b_to_gguf.sh
# ============================================================
set -euo pipefail

BOLD="\033[1m"; GREEN="\033[32m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

SCENARIO="${SCENARIO:-nextera}"
MODEL_DIR="models/gemma3-4b-ft-merged"
OUTPUT_GGUF="$MODEL_DIR/gemma3-4b-ft-${SCENARIO}-f16.gguf"
OUTPUT_Q4="$MODEL_DIR/gemma3-4b-ft-${SCENARIO}-q4_k_m.gguf"
CONVERT_SCRIPT="vendor/llama.cpp/convert_hf_to_gguf.py"
QUANTIZE_BIN="vendor/llama.cpp/build/bin/llama-quantize"

[ -f "$MODEL_DIR/model.safetensors" ] || error "model.safetensors not found — run python -m finetune.train_gemma3_4b first"
[ -f "$CONVERT_SCRIPT" ] || error "convert_hf_to_gguf.py not found — run: git submodule update --init vendor/llama.cpp"

# ─── Backup existing GGUF ────────────────────────────────────────────────────
if [ -f "$OUTPUT_GGUF" ]; then
    BACKUP="$OUTPUT_GGUF.bak.$(date +%Y%m%d_%H%M%S)"
    info "Backing up existing GGUF → $BACKUP"
    cp "$OUTPUT_GGUF" "$BACKUP"
fi

# ─── Dependencies ─────────────────────────────────────────────────────────────
${PIP:-pip} install -q vendor/llama.cpp/gguf-py sentencepiece

# ─── Tokenizer files ─────────────────────────────────────────────────────────
# Download tokenizer.model (SPM), tokenizer_config.json from HuggingFace.
# The SPM path is required — BPE path fails (missing pre-tokenizer hash in
# llama.cpp, see FINE_TUNING_INSIGHTS.md). Also inject chat template.
${PYTHON:-python} - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, json

out_dir = "models/gemma3-4b-ft-merged"

# Download both tokenizer files from the 4B model
for fname in ("tokenizer_config.json", "tokenizer.model"):
    p = hf_hub_download("google/gemma-3-4b-it", fname)
    shutil.copy(p, f"{out_dir}/{fname}")
    print(f"  {fname} ✓")

cfg_path = f"{out_dir}/tokenizer_config.json"
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
print("  chat_template injected ✓")
EOF

# ─── Convert ──────────────────────────────────────────────────────────────────
${PYTHON:-python} "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_GGUF" \
    --outtype f16

info "F16 GGUF ready: $OUTPUT_GGUF ($(du -h "$OUTPUT_GGUF" | cut -f1))"

# ─── Quantize to Q4_K_M (production artifact) ────────────────────────────────
# scenarios/<scenario>.json points at the Q4_K_M variant by default. F16 stays
# on disk for A/B rollback. If llama-quantize isn't built yet, the user gets
# a clear pointer instead of a cryptic "file not found" later.
if [ ! -x "$QUANTIZE_BIN" ]; then
    error "llama-quantize not built — run: bash scripts/build_llama.sh"
fi

info "Quantizing F16 → Q4_K_M..."
"$QUANTIZE_BIN" "$OUTPUT_GGUF" "$OUTPUT_Q4" q4_k_m

info "Q4_K_M GGUF ready: $OUTPUT_Q4 ($(du -h "$OUTPUT_Q4" | cut -f1))"
echo "  Restart servers: bash scripts/start_servers.sh --scenario $SCENARIO"
