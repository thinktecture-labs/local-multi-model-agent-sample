#!/usr/bin/env bash
# ============================================================
# convert_embeddinggemma_to_gguf.sh — Convert the fine-tuned
# embeddinggemma model to GGUF for llama-server.
#
# Run this after `python -m finetune.train_embeddinggemma` to produce
# models/embeddinggemma-300m-ft-merged/embeddinggemma-300m-ft-<scenario>-q8_0.gguf
# which is served by llama-server on port 9092 with --embeddings (--ft flag).
#
# embeddinggemma-300m uses bidirectional attention (encoder-style).
# The GGUF uses mean pooling to produce the final embedding vector.
# After swapping, rebuild the ChromaDB index:
#   python -m data.loader
#
# Usage:
#   source .venv/bin/activate
#   bash finetune/convert_embeddinggemma_to_gguf.sh
# ============================================================
set -euo pipefail

BOLD="\033[1m"; GREEN="\033[32m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

SCENARIO="${SCENARIO:-nextera}"
MODEL_DIR="models/embeddinggemma-300m-ft-merged"
OUTPUT_GGUF="$MODEL_DIR/embeddinggemma-300m-ft-${SCENARIO}-q8_0.gguf"
CONVERT_SCRIPT="vendor/llama.cpp/convert_hf_to_gguf.py"

[ -d "$MODEL_DIR" ] || error "Fine-tuned model not found: $MODEL_DIR — run python -m finetune.train_embeddinggemma first"
[ -f "$CONVERT_SCRIPT" ] || error "convert_hf_to_gguf.py not found — run: git submodule update --init vendor/llama.cpp"

# ─── Backup existing GGUF ────────────────────────────────────────────────────
if [ -f "$OUTPUT_GGUF" ]; then
    BACKUP="$OUTPUT_GGUF.bak.$(date +%Y%m%d_%H%M%S)"
    info "Backing up existing GGUF → $BACKUP"
    cp "$OUTPUT_GGUF" "$BACKUP"
fi

# ─── Dependencies ─────────────────────────────────────────────────────────────

pip install -q vendor/llama.cpp/gguf-py sentencepiece

# ─── Tokenizer files ──────────────────────────────────────────────────────────
# sentence-transformers saves tokenizer.json (BPE) but NOT tokenizer.model (SPM).
# Download tokenizer.model from Google to trigger the SPM path in the converter.
# Also download tokenizer_config.json for correct add_bos_token / added_tokens_decoder.

python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil

for filename in ("tokenizer.model", "tokenizer_config.json"):
    p = hf_hub_download("google/embeddinggemma-300m", filename)
    shutil.copy(p, f"models/embeddinggemma-300m-ft-merged/{filename}")
    print(f"  {filename} ✓")
EOF

# ─── Convert ──────────────────────────────────────────────────────────────────
# Pooling type is auto-detected from the model architecture (mean pooling).
# --outtype q8_0: 8-bit quantization matches the original base model GGUF.
# The sentence-transformers Dense projection layers (768→3072→768 linear) are
# not included in the GGUF — llama-server performs mean pooling on the backbone.

python "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_GGUF" \
    --outtype q8_0

info "GGUF ready: $OUTPUT_GGUF ($(du -h "$OUTPUT_GGUF" | cut -f1))"
echo ""
echo "  Next steps:"
echo "  1. Restart servers with fine-tuned models:"
echo "       bash scripts/start_servers.sh --bg --ft"
echo ""
echo "  2. Rebuild vector index with fine-tuned embeddings:"
echo "       python -m data.loader"
echo ""
echo "  3. Evaluate retrieval improvement:"
echo "       python -m finetune.eval_embeddinggemma --save results/finetuned_embeddinggemma.json"
