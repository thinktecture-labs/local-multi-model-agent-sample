#!/usr/bin/env bash
# ============================================================
# setup.sh — Bootstrap the local multi-model AI agent demo
#
# What this does:
#   1. Build llama-server from the vendored llama.cpp submodule
#      (auto-detects CUDA / Metal / CPU)
#   2. Download all required GGUFs from HuggingFace
#      (gemma3 inference, embeddinggemma, gemma3-4b vision + mmproj)
#   3. Create a Python virtualenv and install dependencies
#   4. Build the Observatory React UI (requires Node.js)
#   5. Seed the demo database and vector store
#   6. (Optional) Build whisper.cpp + download Piper TTS voices
#   7. (Optional) Download Qwen 3.5 for three-path comparison (~22 GB)
#
# Run: bash setup.sh                    # core setup
#      bash setup.sh --include-voice    # also build whisper.cpp + Piper TTS
#      bash setup.sh --include-qwen     # also download Qwen 3.5 (~22 GB)
#      bash setup.sh --include-ocr      # also download GLM-OCR (~1.4 GB)
#      bash setup.sh --all              # everything
# ============================================================
set -euo pipefail

# ─── Parse flags ──────────────────────────────────────────────────────────────
INCLUDE_VOICE=false
INCLUDE_QWEN=false
INCLUDE_OCR=false
for arg in "$@"; do
    case "$arg" in
        --include-voice) INCLUDE_VOICE=true ;;
        --include-qwen)  INCLUDE_QWEN=true ;;
        --include-ocr)   INCLUDE_OCR=true ;;
        --all)           INCLUDE_VOICE=true; INCLUDE_QWEN=true; INCLUDE_OCR=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

info()    { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error()   { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }
heading() { echo -e "\n${BOLD}$*${RESET}"; }

# ─── 0. Environment config ───────────────────────────────────────────────────

if [ ! -f .env ]; then
    cp .env.example .env
    info "Created .env from .env.example (edit to customise)"
else
    info ".env already exists"
fi

# ─── 1. Build llama-server ────────────────────────────────────────────────────

heading "Step 1 — llama-server"

LLAMA_BIN="vendor/llama.cpp/build/bin/llama-server"

if [ -f "$LLAMA_BIN" ]; then
    info "llama-server already built: $($LLAMA_BIN --version 2>&1 | grep version | head -1)"
else
    echo "  Building llama-server from vendor/llama.cpp…"
    bash scripts/build_llama.sh
fi

# ─── 2. Download GGUF models ──────────────────────────────────────────────────

heading "Step 2 — Downloading GGUF models"
echo "  Models are downloaded from HuggingFace (ggml-org official builds)."
echo ""

# Python is needed for hf_hub_download — check early
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done
[ -z "$PYTHON_CMD" ] && error "Python 3.10+ required. Install from https://python.org"

# Bootstrap minimal venv for downloads
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
# Use $VENV_DIR/bin/python -m pip instead of bare `pip`: covers both
# (a) Ubuntu/Debian/Fedora PEP 668, where bare `pip` falls through to a
# system-managed Python and is blocked; and (b) uv-created venvs, which
# don't ship a `pip` binary by default (only python + `pip` as a module).
"$VENV_DIR/bin/python" -m ensurepip --upgrade --default-pip >/dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --quiet huggingface_hub

download_gguf() {
    local repo="$1" filename="$2" dest_dir="$3"
    local dest="$dest_dir/$filename"
    if [ -f "$dest" ]; then
        info "Already downloaded: $filename"
    else
        echo -e "  ${BOLD}Downloading $filename from ${repo}...${RESET}"
        "$VENV_DIR/bin/python" -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('$dest_dir', exist_ok=True)
hf_hub_download('$repo', '$filename', local_dir='$dest_dir')
"
        info "Downloaded: $dest"
    fi
}

# Fine-tuned intent classifier (produced by finetune/train_gemma3.py + convert)
if [ -f "models/gemma3-1b-ft-merged/gemma3-ft.gguf" ]; then
    info "Already present: gemma3-ft.gguf (fine-tuned)"
else
    warn "models/gemma3-1b-ft-merged/gemma3-ft.gguf not found."
    warn "Run the fine-tuning pipeline first:"
    warn "  pip install -r requirements-finetune.txt"
    warn "  python -m finetune.train_gemma3 --task intent"
    warn "  bash finetune/convert_gemma3_to_gguf.sh"
fi

download_gguf "ggml-org/gemma-3-1b-it-GGUF" \
              "gemma-3-1b-it-Q8_0.gguf" \
              "models/gemma3"

download_gguf "ggml-org/embeddinggemma-300M-GGUF" \
              "embeddinggemma-300M-Q8_0.gguf" \
              "models/embeddinggemma"

# Vision model (gemma3-4B + multimodal projector for image understanding)
download_gguf "ggml-org/gemma-3-4b-it-GGUF" \
              "gemma-3-4b-it-Q4_K_M.gguf" \
              "models/gemma3-4b"

download_gguf "ggml-org/gemma-3-4b-it-GGUF" \
              "mmproj-model-f16.gguf" \
              "models/gemma3-4b"

# ─── 3. Python environment ────────────────────────────────────────────────────

heading "Step 3 — Python environment"

info "Python: $PYTHON_CMD ($("$PYTHON_CMD" --version))"
info "Virtual environment: $VENV_DIR (already active)"

echo "  Installing Python dependencies..."
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet -r requirements.txt
info "Dependencies installed"

# ─── 4. Build Observatory React UI ───────────────────────────────────────────

heading "Step 4 — Observatory UI (React)"

REACT_DIR="src/clients/observatory-react"

# Always ensure npm deps + wake word ONNX models are present
_setup_observatory_deps() {
    echo "  Installing npm dependencies..."
    (cd "$REACT_DIR" && npm install --silent 2>&1 | tail -1)
    echo "  Copying OpenWakeWord ONNX models to public/..."
    mkdir -p "$REACT_DIR/public/openwakeword/models"
    cp "$REACT_DIR/node_modules/openwakeword-wasm-browser/models/melspectrogram.onnx" \
       "$REACT_DIR/node_modules/openwakeword-wasm-browser/models/embedding_model.onnx" \
       "$REACT_DIR/node_modules/openwakeword-wasm-browser/models/silero_vad.onnx" \
       "$REACT_DIR/node_modules/openwakeword-wasm-browser/models/hey_jarvis_v0.1.onnx" \
       "$REACT_DIR/public/openwakeword/models/"
    info "Wake word models ready (hey_jarvis)"
}

if [ -f "$REACT_DIR/dist/index.html" ]; then
    info "Observatory UI already built"
    # Still ensure deps + wake word models are present
    # Check for Node.js
    NODE_CMD=""
    for cmd in node nodejs; do
        if command -v "$cmd" &>/dev/null; then
            node_version=$("$cmd" --version 2>/dev/null | sed 's/v//')
            node_major=$(echo "$node_version" | cut -d. -f1)
            if [ "$node_major" -ge 22 ] 2>/dev/null; then
                NODE_CMD="$cmd"
                break
            fi
        fi
    done
    if [ -n "$NODE_CMD" ] && [ ! -f "$REACT_DIR/public/openwakeword/models/hey_jarvis_v0.1.onnx" ]; then
        _setup_observatory_deps
    fi
else
    # Check for Node.js
    NODE_CMD=""
    for cmd in node nodejs; do
        if command -v "$cmd" &>/dev/null; then
            node_version=$("$cmd" --version 2>/dev/null | sed 's/v//')
            node_major=$(echo "$node_version" | cut -d. -f1)
            if [ "$node_major" -ge 22 ] 2>/dev/null; then
                NODE_CMD="$cmd"
                break
            fi
        fi
    done

    # Try nvm if no system node
    if [ -z "$NODE_CMD" ]; then
        NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
        if [ -s "$NVM_DIR/nvm.sh" ]; then
            source "$NVM_DIR/nvm.sh"
            if command -v node &>/dev/null; then
                NODE_CMD="node"
                info "Using Node.js from nvm: $(node --version)"
            fi
        fi
    fi

    if [ -z "$NODE_CMD" ]; then
        warn "Node.js 18+ not found — skipping Observatory UI build"
        warn "Install Node.js (https://nodejs.org) and run:"
        warn "  cd $REACT_DIR && npm install && npm run build"
    else
        info "Node.js: $($NODE_CMD --version)"
        _setup_observatory_deps
        echo "  Building Observatory UI..."
        (cd "$REACT_DIR" && npm run build 2>&1 | tail -3)
        info "Observatory UI built: $REACT_DIR/dist/"
    fi
fi

# ─── 5. Seed demo data ────────────────────────────────────────────────────────

heading "Step 5 — Demo data"

# Start servers for seeding. Use --ft if fine-tuned GGUFs exist so that
# ChromaDB is indexed with the same embedding model used at query time.
FT_FLAG="--base"
if [ -f "${EMBEDDING_GGUF_FT:-models/embeddinggemma-300m-ft-merged/embeddinggemma-300m-ft-nextera-q8_0.gguf}" ]; then
    FT_FLAG="--ft"
    echo "  Starting llama-server instances (fine-tuned) for seeding..."
else
    echo "  Starting llama-server instances (base) for seeding..."
fi
bash scripts/start_servers.sh --bg $FT_FLAG

echo "  Waiting for llama-server instances to become healthy..."
_PORTS=(9090 9091 9092 9093)
_DEADLINE=$(( $(date +%s) + 60 ))
for _PORT in "${_PORTS[@]}"; do
    while ! curl -sf "http://localhost:${_PORT}/health" >/dev/null 2>&1; do
        if [ "$(date +%s)" -ge "$_DEADLINE" ]; then
            warn "Server on port ${_PORT} did not become healthy within 60s"
            break
        fi
        sleep 1
    done
    info "Port ${_PORT} healthy"
done
unset _PORTS _DEADLINE _PORT

mkdir -p data

echo "  Seeding SQLite database…"
"$VENV_DIR/bin/python" -c "
import asyncio, importlib
from src.engine.inference.config import SCENARIO_CONFIG
loader = importlib.import_module(SCENARIO_CONFIG.data_loader_module)
asyncio.run(loader.seed_sql_database())
print(f'  SQLite database ready: {SCENARIO_CONFIG.db_path}')
"
info "SQLite database seeded"

echo "  Indexing knowledge base into ChromaDB (uses embeddinggemma on :9092)…"
echo "  (This embeds each document — takes ~30 seconds)"
"$VENV_DIR/bin/python" -c "
import asyncio, importlib
from src.engine.inference.config import SCENARIO_CONFIG
from src.engine.inference.client import SmallLanguageModelClient
from src.engine.knowledge.vector_store import VectorStore
loader = importlib.import_module(SCENARIO_CONFIG.data_loader_module)

async def run():
    client = SmallLanguageModelClient()
    vs = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    n = await loader.seed_vector_store(client, vs)
    total = await vs.count()
    print(f'  Indexed {n} new documents ({total} total)')

asyncio.run(run())
"
info "Vector store seeded"

# ─── 6. Voice setup (optional) ─────────────────────────────────────────────────

if [ "$INCLUDE_VOICE" = true ]; then
    heading "Step 6 — Voice (whisper.cpp + Piper TTS)"
    bash scripts/setup_voice.sh
else
    info "Skipping voice setup (pass --include-voice to enable)"
fi

# ─── 7. Qwen comparison model (optional) ─────────────────────────────────────

if [ "$INCLUDE_QWEN" = true ]; then
    heading "Step 7 — Qwen 3.5 comparison model (~22 GB)"
    bash scripts/download_qwen35b.sh
else
    info "Skipping Qwen download (pass --include-qwen to enable)"
fi

# ─── 8. GLM-OCR (optional) ───────────────────────────────────────────────────

if [ "$INCLUDE_OCR" = true ]; then
    heading "Step 8 — GLM-OCR document extraction (~1.4 GB)"
    bash scripts/setup_ocr.sh
else
    info "Skipping OCR setup (pass --include-ocr to enable)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

heading "Setup complete!"
echo ""
echo "  Start everything (servers + API + Observatory UI):"
echo "    bash scripts/start_app.sh              # fine-tuned models (default)"
echo "    bash scripts/start_app.sh --base       # base models (pre fine-tuning)"
echo ""
echo "  Then open:"
echo "    Observatory UI:  http://localhost:8000/app"
echo "    API docs:        http://localhost:8000/docs"
echo ""
echo "  Or start servers only:"
echo "    bash scripts/start_servers.sh --bg     # background"
echo "    bash scripts/start_servers.sh --bg --qwen  # include Qwen comparison"
echo ""
echo "  CLI demo:"
echo "    source .venv/bin/activate"
echo "    python demo.py                          # showcase mode"
echo "    python demo.py --interactive            # REPL for live demos"
echo ""
