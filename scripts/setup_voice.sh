#!/usr/bin/env bash
# ============================================================
# setup_voice.sh — Build whisper.cpp and download voice models
#
# This script:
#   1. Clones & builds whisper.cpp (Metal on macOS, CUDA on Linux)
#   2. Downloads the whisper medium model (1.5 GB, good EN+DE)
#   3. Downloads Piper TTS voices (EN + DE)
#
# Prerequisites:
#   - cmake, git, ffmpeg
#   - pip install piper-tts (for Python API)
#
# Usage:
#   bash scripts/setup_voice.sh
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

# ─── Activate Python env ────────────────────────────────────────────────────

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    info "Activated .venv"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    info "Using virtualenv: $VIRTUAL_ENV"
else
    warn "No .venv found — pip install will use system Python"
fi

# ─── 1. Build whisper.cpp ────────────────────────────────────────────────────

WHISPER_DIR="vendor/whisper.cpp"
WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-server"

if [ -f "$WHISPER_BIN" ]; then
    info "whisper-server already built: $WHISPER_BIN"
else
    echo -e "\n${BOLD}Building whisper.cpp…${RESET}\n"

    if [ ! -d "$WHISPER_DIR" ]; then
        git clone https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
    else
        info "whisper.cpp source already present"
    fi

    cd "$WHISPER_DIR"

    # Detect platform and configure GPU acceleration
    CMAKE_ARGS=""
    if [[ "$(uname)" == "Darwin" ]]; then
        CMAKE_ARGS="-DGGML_METAL=ON"
        info "macOS detected — building with Metal acceleration"
    elif command -v nvidia-smi &>/dev/null; then
        CMAKE_ARGS="-DGGML_CUDA=ON"
        info "NVIDIA GPU detected — building with CUDA acceleration"
    else
        info "No GPU detected — building CPU-only"
    fi

    cmake -B build $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

    cd "$REPO_ROOT"

    if [ -f "$WHISPER_BIN" ]; then
        info "whisper-server built successfully: $WHISPER_BIN"
    else
        # Some whisper.cpp versions name it differently
        ALT_BIN="$WHISPER_DIR/build/bin/server"
        if [ -f "$ALT_BIN" ]; then
            info "whisper server built as: $ALT_BIN (update WHISPER_SERVER in .env)"
        else
            error "Build failed — whisper-server binary not found"
        fi
    fi
fi

# ─── 2. Download whisper model ───────────────────────────────────────────────

WHISPER_MODEL_DIR="models/whisper"
WHISPER_MODEL="$WHISPER_MODEL_DIR/ggml-medium.bin"

mkdir -p "$WHISPER_MODEL_DIR"

if [ -f "$WHISPER_MODEL" ]; then
    info "Whisper model already downloaded: $WHISPER_MODEL"
else
    echo -e "\n${BOLD}Downloading whisper medium model (~1.5 GB)…${RESET}\n"
    curl -L --progress-bar \
        -o "$WHISPER_MODEL" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
    info "Whisper model downloaded: $WHISPER_MODEL"
fi

# ─── 3. Download Piper TTS voices ───────────────────────────────────────────

PIPER_DIR="models/piper"
mkdir -p "$PIPER_DIR"

download_piper_voice() {
    local name="$1"       # e.g. en_US-lessac-medium
    local quality="$2"    # e.g. medium
    local lang="$3"       # e.g. en_US
    local voice="$4"      # e.g. lessac
    local onnx_file="$PIPER_DIR/${name}.onnx"
    local json_file="$PIPER_DIR/${name}.onnx.json"

    if [ -f "$onnx_file" ] && [ -s "$onnx_file" ] && [ -f "$json_file" ]; then
        local size=$(wc -c < "$onnx_file" | tr -d ' ')
        if [ "$size" -gt 1000 ]; then
            info "Piper voice already downloaded: $name ($size bytes)"
            return
        fi
    fi

    echo -e "  Downloading Piper voice: ${BOLD}$name${RESET} ($lang, $quality quality)…"
    # HF path: en/en_US/lessac/medium/en_US-lessac-medium.onnx
    local base_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    local lang2="${lang:0:2}"
    local url_path="${lang2}/${lang}/${voice}/${quality}"

    curl -L --progress-bar -o "$onnx_file" "${base_url}/${url_path}/${name}.onnx"
    curl -sL -o "$json_file" "${base_url}/${url_path}/${name}.onnx.json"

    if [ -f "$onnx_file" ]; then
        local size=$(wc -c < "$onnx_file" | tr -d ' ')
        if [ "$size" -gt 1000000 ]; then
            info "Downloaded: $name ($(( size / 1048576 )) MB)"
        else
            warn "Download may have failed for $name — file only $size bytes"
            rm -f "$onnx_file" "$json_file"
        fi
    else
        warn "Download failed for $name"
    fi
}

echo -e "\n${BOLD}Downloading Piper TTS voices…${RESET}\n"
download_piper_voice "en_US-lessac-medium" "medium" "en_US" "lessac"
download_piper_voice "de_DE-thorsten-high" "high" "de_DE" "thorsten"

# ─── 4. Summary ──────────────────────────────────────────────────────────────

# ─── 4. Install piper-tts Python package ─────────────────────────────────────

if python3 -c "import piper" 2>/dev/null; then
    info "piper-tts already installed"
else
    echo -e "\n${BOLD}Installing piper-tts…${RESET}"
    pip install piper-tts 2>&1 | tail -3
    if python3 -c "import piper" 2>/dev/null; then
        info "piper-tts installed"
    else
        warn "piper-tts installation may have failed — run: pip install piper-tts"
    fi
fi

# ─── 5. Summary ──────────────────────────────────────────────────────────────

echo -e "\n${BOLD}Voice setup complete!${RESET}\n"
echo "  Whisper server: $WHISPER_BIN"
echo "  Whisper model:  $WHISPER_MODEL"
echo "  Piper voices:   $PIPER_DIR/"
echo ""
echo "  To start all servers (including whisper):"
echo "    bash scripts/start_servers.sh"
echo ""
echo "  System dependency needed:"
echo "    ffmpeg  — brew install ffmpeg (macOS) / apt install ffmpeg (Linux)"
echo ""
