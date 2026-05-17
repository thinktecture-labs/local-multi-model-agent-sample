#!/usr/bin/env bash
# Download Qwen 3.5 35B-A3B GGUF (Q4_K_M) for three-path comparison demo.
# ~22 GB download. Requires: hf CLI (https://hf.co/cli) or huggingface-cli (pip install huggingface-hub)
set -euo pipefail

MODEL_DIR="models/qwen3.5-35b-a3b"
REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
FILE="Qwen3.5-35B-A3B-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$FILE" ]; then
    echo "Already downloaded: $MODEL_DIR/$FILE"
    exit 0
fi

# Find a HuggingFace download command: prefer hf CLI, fall back to huggingface-cli
HF_CMD=""
for cmd in hf "$HOME/.local/bin/hf"; do
    if command -v "$cmd" &>/dev/null 2>&1 || [ -x "$cmd" ]; then
        HF_CMD="$cmd"
        break
    fi
done

if [ -n "$HF_CMD" ]; then
    echo "Downloading $REPO / $FILE (~22 GB) via $HF_CMD..."
    "$HF_CMD" download "$REPO" "$FILE" --local-dir "$MODEL_DIR"
elif command -v huggingface-cli &>/dev/null; then
    echo "Downloading $REPO / $FILE (~22 GB) via huggingface-cli..."
    huggingface-cli download "$REPO" "$FILE" --local-dir "$MODEL_DIR"
else
    echo "Error: No HuggingFace CLI found."
    echo "Install one of:"
    echo "  curl -LsSf https://hf.co/cli/install.sh | bash    # hf CLI (recommended)"
    echo "  pip install huggingface-hub                         # huggingface-cli"
    exit 1
fi

echo "Done: $MODEL_DIR/$FILE"
