#!/usr/bin/env bash
# ============================================================
# start_servers.sh — Start all four llama-server instances
#
# Ports (default, see .env):
#   9090  inference      (gemma3 base / gemma3-ft after fine-tuning)
#   9091  function       (Qwen 3.5-4B base / fine-tuned tool caller)
#   9092  embeddinggemma (base / fine-tuned after training)
#   9093  vision         (gemma3-4B + mmproj for multimodal)
#   9097  whisper        (STT — optional, skipped if not installed)
#
# Run `bash setup.sh` to download the required base models.
# Whisper is optional — run `bash scripts/setup_voice.sh` to install.
# Configuration is read from .env (override with .env.local).
#
# Usage:
#   bash scripts/start_servers.sh          # foreground (Ctrl-C stops all)
#   bash scripts/start_servers.sh --bg     # background (PIDs → .server-pids)
#   bash scripts/start_servers.sh --ft     # use fine-tuned GGUFs (all 3 models)
#                                          # gracefully falls back to base for
#                                          # models not yet fine-tuned
#   bash scripts/start_servers.sh --cpu    # force CPU-only (no Metal/CUDA)
#   bash scripts/start_servers.sh --all    # CHEAT MODE: base + FT on separate ports
#   bash scripts/start_servers.sh --ft-extra # FT servers only (9094-9096)
#   bash scripts/start_servers.sh --qwen               # also start Qwen 3.5 on port 9100
#   bash scripts/start_servers.sh --scenario foo       # use scenarios/foo.json
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

# Portable "find PIDs listening on a port" — lsof (macOS/some Linux) or ss+fuser (Fedora/minimal Linux)
pids_on_port() {
    local port="$1"
    if command -v lsof &>/dev/null; then
        lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true
    elif command -v ss &>/dev/null; then
        ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true
    fi
}

# ─── Load config ──────────────────────────────────────────────────────────────

[ -f .env ]       && source .env
[ -f .env.local ] && source .env.local  # local overrides take precedence

# ─── Parse --scenario flag early (before scenario JSON load) ────────────────
_args=("$@")
for ((i=0; i<${#_args[@]}; i++)); do
    if [[ "${_args[$i]}" == "--scenario" ]]; then
        export SCENARIO="${_args[$((i+1))]}"
    fi
done

# ─── Load scenario-specific FT GGUF paths from JSON ─────────────────────────
SCENARIO="${SCENARIO:-nextera}"
SCENARIO_JSON="scenarios/${SCENARIO}.json"
if [ -f "$SCENARIO_JSON" ]; then
    # Extract FT GGUF paths (jq or python fallback)
    if command -v jq &>/dev/null; then
        INFERENCE_GGUF_FT="$(jq -r '.models.inference_gguf_ft' "$SCENARIO_JSON")"
        FUNCTION_GGUF_FT="$(jq -r '.models.function_gguf_ft' "$SCENARIO_JSON")"
        EMBEDDING_GGUF_FT="$(jq -r '.models.embedding_gguf_ft' "$SCENARIO_JSON")"
        SYNTHESIS_4B_GGUF_FT="$(jq -r '.models.synthesis_4b_gguf_ft // empty' "$SCENARIO_JSON")"
    else
        INFERENCE_GGUF_FT="$(python3 -c "import json; d=json.load(open('$SCENARIO_JSON')); print(d['models']['inference_gguf_ft'])")"
        FUNCTION_GGUF_FT="$(python3 -c "import json; d=json.load(open('$SCENARIO_JSON')); print(d['models']['function_gguf_ft'])")"
        EMBEDDING_GGUF_FT="$(python3 -c "import json; d=json.load(open('$SCENARIO_JSON')); print(d['models']['embedding_gguf_ft'])")"
        SYNTHESIS_4B_GGUF_FT="$(python3 -c "import json; d=json.load(open('$SCENARIO_JSON')); print(d['models'].get('synthesis_4b_gguf_ft',''))")"
    fi
    info "Scenario: ${SCENARIO} — FT GGUFs from ${SCENARIO_JSON}"
else
    error "Scenario config not found: ${SCENARIO_JSON}"
fi

# ─── Parse flags ──────────────────────────────────────────────────────────────

BG=false
USE_FT=true
CPU_ONLY=false
START_ALL=false
FT_EXTRA=false
START_QWEN=false
for arg in "$@"; do
    case "$arg" in
        --bg)       BG=true ;;
        --ft)       USE_FT=true ;;
        --base)     USE_FT=false ;;
        --cpu)      CPU_ONLY=true ;;
        --all)      START_ALL=true; START_QWEN=true; USE_FT=false ;;  # --all: base on 9090-9093, FT on 9094-9096, Qwen on 9100 (USE_FT=false → primary ports use base GGUFs for A/B demo)
        --ft-extra) FT_EXTRA=true ;;
        --qwen)     START_QWEN=true ;;
    esac
done

GPU_LAYERS=999
if $CPU_ONLY; then
    GPU_LAYERS=0
    warn "CPU-only mode: --n-gpu-layers 0 (Metal/CUDA disabled)"
fi

# Platform-specific llama-server flags:
# - CUDA (all): --flash-attn on for 5-20% throughput gain on token generation
# - CUDA + aarch64 (DGX Spark): --no-mmap (mmap 4x slower on unified memory),
#   GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 for full 120 GB pool access
# - CUDA + x86_64 (RTX Blackwell): --no-mmap (mmap assertion bug, #18090)
# - Vulkan (AMD Strix Halo): --no-mmap (UMA system, mmap adds overhead)
# - Metal (Apple Silicon): keep mmap enabled (works well with unified memory)
EXTRA_LLAMA_ARGS=""
if command -v nvidia-smi &>/dev/null; then
    EXTRA_LLAMA_ARGS="--no-mmap --flash-attn on"
    if [[ "$(uname -m)" == "aarch64" ]]; then
        export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
        info "aarch64 + CUDA detected — using --no-mmap, --flash-attn on, unified memory"
    else
        info "x86_64 + CUDA detected — using --no-mmap, --flash-attn on"
    fi
elif [[ "$(uname -s)" == "Linux" ]] && vulkaninfo --summary 2>/dev/null | grep -q "INTEGRATED_GPU"; then
    EXTRA_LLAMA_ARGS="--no-mmap"
    info "Vulkan integrated GPU detected — using --no-mmap (UMA)"
fi

# Select GGUFs: --ft uses fine-tuned versions where available
# Inference fine-tuned is REQUIRED (error if missing); function + embedding
# fall back gracefully to base if their fine-tuned GGUFs aren't ready yet.
if $USE_FT; then
    ACTIVE_INFERENCE_GGUF="${INFERENCE_GGUF_FT}"
    INFERENCE_LABEL="gemma3-ft (fine-tuned)"

    if [ -f "${FUNCTION_GGUF_FT:-}" ]; then
        ACTIVE_FUNCTION_GGUF="${FUNCTION_GGUF_FT}"
        FUNCTION_LABEL="qwen-toolcalling-ft (fine-tuned)"
    else
        ACTIVE_FUNCTION_GGUF="${FUNCTION_GGUF}"
        FUNCTION_LABEL="qwen-toolcalling (base — run finetune/convert_qwen35_to_gguf.sh to upgrade)"
    fi

    if [ -f "${EMBEDDING_GGUF_FT:-}" ]; then
        ACTIVE_EMBEDDING_GGUF="${EMBEDDING_GGUF_FT}"
        EMBEDDING_LABEL="embeddinggemma-ft (fine-tuned)"
    else
        ACTIVE_EMBEDDING_GGUF="${EMBEDDING_GGUF}"
        EMBEDDING_LABEL="embeddinggemma (base — run finetune/convert_embeddinggemma_to_gguf.sh to upgrade)"
    fi
else
    ACTIVE_INFERENCE_GGUF="${INFERENCE_GGUF}"
    ACTIVE_FUNCTION_GGUF="${FUNCTION_GGUF}"
    ACTIVE_EMBEDDING_GGUF="${EMBEDDING_GGUF}"
    INFERENCE_LABEL="gemma3 (base)"
    FUNCTION_LABEL="qwen-toolcalling (base)"
    EMBEDDING_LABEL="embeddinggemma (base)"
fi

# Vision/synthesis model — use FT variant if available, else base
if $USE_FT && [ -n "${SYNTHESIS_4B_GGUF_FT:-}" ] && [ -f "${SYNTHESIS_4B_GGUF_FT:-}" ]; then
    ACTIVE_VISION_GGUF="${SYNTHESIS_4B_GGUF_FT}"
    VISION_LABEL="gemma3-4b-ft (fine-tuned synthesis + vision)"
else
    ACTIVE_VISION_GGUF="${VISION_GGUF:-}"
    VISION_LABEL="gemma3-4b (base)"
fi
VISION_MMPROJ_PATH="${VISION_MMPROJ:-}"

# ─── FT-extra mode: only start FT servers on secondary ports ──────────────────
if $FT_EXTRA; then
    [ -f "$LLAMA_SERVER" ] || error "llama-server not found: $LLAMA_SERVER — run: bash setup.sh"

    # Resolve FT GGUFs (fall back to base if not available)
    FT_INF_GGUF="${INFERENCE_GGUF_FT}"
    [ -f "$FT_INF_GGUF" ] || error "Missing FT inference GGUF: $FT_INF_GGUF — run training first"

    FT_FUNC_GGUF="${FUNCTION_GGUF_FT:-}"
    if [ ! -f "${FT_FUNC_GGUF:-}" ]; then
        FT_FUNC_GGUF="${FUNCTION_GGUF}"
        warn "FT tool-calling model not found, using base"
    fi

    FT_EMB_GGUF="${EMBEDDING_GGUF_FT:-}"
    if [ ! -f "${FT_EMB_GGUF:-}" ]; then
        FT_EMB_GGUF="${EMBEDDING_GGUF}"
        warn "FT embeddinggemma not found, using base"
    fi

    FT_PORTS=("$INFERENCE_PORT_FT" "$FUNCTION_PORT_FT" "$EMBEDDING_PORT_FT")

    # Stop any existing FT servers
    if [ -f .server-pids-ft ]; then
        while IFS= read -r pid; do
            kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
        done < .server-pids-ft
        rm -f .server-pids-ft
        info "Stopped previous FT servers"
    fi
    for port in "${FT_PORTS[@]}"; do
        pids=$(pids_on_port "$port")
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            warn "Killed existing process on port $port"
        fi
    done
    sleep 1

    echo -e "\n${BOLD}Starting FT servers on secondary ports (parallel=$PARALLEL_SLOTS)…${RESET}\n"

    launch() {
        local label="$1" port="$2" gguf="$3" server="${LLAMA_SERVER_OVERRIDE:-$LLAMA_SERVER}"; shift 3
        echo -e "  ${BOLD}$label${RESET}  →  http://localhost:$port"
        # shellcheck disable=SC2086
        "$server" \
            --model    "$gguf" \
            --port     "$port" \
            --host     127.0.0.1 \
            --ctx-size 8192 \
            --parallel "$PARALLEL_SLOTS" \
            --n-gpu-layers "$GPU_LAYERS" \
            --log-disable \
            $KVU_FLAG \
            $EXTRA_LLAMA_ARGS \
            "$@" &
        echo $! >> .server-pids-ft
        unset LLAMA_SERVER_OVERRIDE
    }

    launch "inference-ft"     "$INFERENCE_PORT_FT"  "$FT_INF_GGUF"
    # Tool-calling server: Qwen 3.5 with standard OpenAI tool-calling format.
    FT_FUNC_ARGS=(--temp 1.0 --top-p 0.95 --jinja --top-k 20 \
                  --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0)
    launch "qwen-toolcalling-ft" "$FUNCTION_PORT_FT" "$FT_FUNC_GGUF" \
        "${FT_FUNC_ARGS[@]}"
    launch "embeddinggemma-ft" "$EMBEDDING_PORT_FT" "$FT_EMB_GGUF" \
        --embeddings --ubatch-size 2048

    echo ""
    info "FT servers launching. Waiting for ready…"
    for port in "${FT_PORTS[@]}"; do
        ready=false
        for i in $(seq 1 5); do
            if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
                ready=true; break
            fi
            sleep 1
        done
        if $ready; then info "Port $port — healthy"
        else warn "Port $port — not ready after 5s (still loading?)"; fi
    done

    echo ""
    if $BG; then
        info "FT servers running in background. PIDs saved to .server-pids-ft"
    else
        info "Press Ctrl-C to stop FT servers."
        wait
    fi
    exit 0
fi

# ─── Checks ───────────────────────────────────────────────────────────────────

[ -f "$LLAMA_SERVER" ]           || error "llama-server not found: $LLAMA_SERVER — run: bash setup.sh"
[ -f "$ACTIVE_INFERENCE_GGUF" ]  || error "Missing inference GGUF: $ACTIVE_INFERENCE_GGUF"
[ -f "$ACTIVE_FUNCTION_GGUF" ]   || error "Missing: $ACTIVE_FUNCTION_GGUF — run: bash setup.sh"
[ -f "$ACTIVE_EMBEDDING_GGUF" ]  || error "Missing: $ACTIVE_EMBEDDING_GGUF — run: bash setup.sh"
[ -f "$ACTIVE_VISION_GGUF" ]     || error "Missing vision GGUF: $ACTIVE_VISION_GGUF — run: bash setup.sh"
[ -f "$VISION_MMPROJ_PATH" ]     || error "Missing vision mmproj: $VISION_MMPROJ_PATH — run: bash setup.sh"

# ─── Stop any existing servers ────────────────────────────────────────────────

ALL_PORTS=("$INFERENCE_PORT" "$FUNCTION_PORT" "$EMBEDDING_PORT" "$VISION_PORT")
OPTIONAL_PORTS=()   # optional servers (Whisper, OCR) — longer wait, non-blocking
WHISPER_PRT="${WHISPER_PORT:-9097}"

# Kill servers from previous PID files
for pidfile in .server-pids .server-pids-ft; do
    if [ -f "$pidfile" ]; then
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$pidfile"
        rm -f "$pidfile"
        info "Stopped previous servers ($pidfile)"
    fi
done

# Also kill anything still listening on our ports (covers orphaned processes)
CLEANUP_PORTS=("${ALL_PORTS[@]}" "$WHISPER_PRT")
if $START_ALL; then
    CLEANUP_PORTS+=("${INFERENCE_PORT_FT:-9094}" "${FUNCTION_PORT_FT:-9095}" "${EMBEDDING_PORT_FT:-9096}")
fi
for port in "${CLEANUP_PORTS[@]}"; do
    pids=$(pids_on_port "$port")
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        warn "Killed existing process on port $port"
    fi
done

# Brief pause to let ports release
sleep 1

# ─── Launch ───────────────────────────────────────────────────────────────────
# Note: chat template is embedded in the GGUF — no need to override it.

PARALLEL_SLOTS="${LLAMA_PARALLEL:-1}"

# Unified KV cache (-kvu): when --parallel >1 on CUDA, per-slot KV cache allocation
# causes CUDA Graph warmup instability (90% latency regression on RTX PRO 6000).
# -kvu keeps n_stream=1 (single shared cache), stabilising graphs and eliminating
# the regression entirely. On Metal this is unnecessary (0% regression without -kvu).
# See docs/QWEN35_EVAL_RESULTS.md → "--parallel Slot Testing".
KVU_FLAG=""
if [ "$PARALLEL_SLOTS" -gt 1 ] && command -v nvidia-smi &>/dev/null; then
    KVU_FLAG="-kvu"
    info "parallel=$PARALLEL_SLOTS on CUDA — adding -kvu (unified KV cache)"
fi

echo -e "\n${BOLD}Starting llama-server instances (parallel=$PARALLEL_SLOTS)…${RESET}\n"
[ -f .server-pids ] && rm .server-pids

launch() {
    local label="$1" port="$2" gguf="$3" server="${LLAMA_SERVER_OVERRIDE:-$LLAMA_SERVER}"; shift 3
    echo -e "  ${BOLD}$label${RESET}  →  http://localhost:$port"
    # shellcheck disable=SC2086
    "$server" \
        --model    "$gguf" \
        --port     "$port" \
        --host     127.0.0.1 \
        --ctx-size 8192 \
        --parallel "$PARALLEL_SLOTS" \
        --n-gpu-layers "$GPU_LAYERS" \
        --log-disable \
        $KVU_FLAG \
        $EXTRA_LLAMA_ARGS \
        "$@" &
    echo $! >> .server-pids
    unset LLAMA_SERVER_OVERRIDE
}

launch "inference     ($INFERENCE_LABEL)"  "$INFERENCE_PORT"  "$ACTIVE_INFERENCE_GGUF" \
    --swa-full --cache-reuse 256
sleep 2  # stagger launches to avoid Metal/CUDA memory contention

FUNC_EXTRA_ARGS=(--temp 1.0 --top-p 0.95 --jinja --top-k 20 \
                 --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0)
launch "tool-calling ($FUNCTION_LABEL)"   "$FUNCTION_PORT"   "$ACTIVE_FUNCTION_GGUF" \
    "${FUNC_EXTRA_ARGS[@]}" --cache-reuse 256
sleep 2  # stagger

launch "embeddinggemma ($EMBEDDING_LABEL)" "$EMBEDDING_PORT"  "$ACTIVE_EMBEDDING_GGUF" \
    --embeddings --ubatch-size 2048
sleep 2  # stagger

launch "vision       ($VISION_LABEL)"   "$VISION_PORT"   "$ACTIVE_VISION_GGUF" \
    --mmproj "$VISION_MMPROJ_PATH" --swa-full --cache-reuse 256
sleep 2  # stagger

# ─── Whisper STT (optional — skipped if binary or model not found) ──────────
WHISPER_BIN="${WHISPER_SERVER:-}"
WHISPER_MDL="${WHISPER_MODEL:-}"
if [ -f "${WHISPER_BIN:-}" ] && [ -f "${WHISPER_MDL:-}" ]; then
    echo -e "  ${BOLD}whisper (STT)${RESET}  →  http://localhost:$WHISPER_PRT"
    # whisper.cpp shared libs (libwhisper.so, libggml.so) are not installed system-wide;
    # set LD_LIBRARY_PATH so the dynamic linker can find them at runtime.
    WHISPER_LIB_DIR="$(dirname "$WHISPER_BIN")/.."
    export LD_LIBRARY_PATH="$WHISPER_LIB_DIR/src:$WHISPER_LIB_DIR/ggml/src:$WHISPER_LIB_DIR/ggml/src/ggml-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$WHISPER_BIN" \
        --model "$WHISPER_MDL" \
        --port "$WHISPER_PRT" \
        --host 127.0.0.1 &
    echo $! >> .server-pids
    OPTIONAL_PORTS+=("$WHISPER_PRT")
else
    info "Whisper STT skipped (binary or model not found — run: bash scripts/setup_voice.sh)"
fi

# ─── GLM-OCR (optional — skipped if model not found) ────────────────────────
OCR_GGUF_FILE="${OCR_GGUF:-}"
OCR_MMPROJ_FILE="${OCR_MMPROJ:-}"
OCR_PRT="${OCR_PORT:-9098}"
if [ -f "$LLAMA_SERVER" ] && [ -f "${OCR_GGUF_FILE:-}" ] && [ -f "${OCR_MMPROJ_FILE:-}" ]; then
    echo -e "  ${BOLD}glm-ocr (OCR)${RESET}  →  http://localhost:$OCR_PRT"
    "$LLAMA_SERVER" \
        --model "$OCR_GGUF_FILE" \
        --mmproj "$OCR_MMPROJ_FILE" \
        --port "$OCR_PRT" \
        --host 127.0.0.1 \
        --ctx-size 8192 \
        --n-gpu-layers "$GPU_LAYERS" \
        --log-disable \
        $EXTRA_LLAMA_ARGS &
    echo $! >> .server-pids
    OPTIONAL_PORTS+=("$OCR_PRT")
else
    info "GLM-OCR skipped (model not found — run: bash scripts/setup_ocr.sh)"
fi

# ─── --all: also start FT servers on secondary ports ─────────────────────────
if $START_ALL; then
    FT_INF_GGUF="${INFERENCE_GGUF_FT}"
    [ -f "$FT_INF_GGUF" ] || error "Missing FT inference GGUF: $FT_INF_GGUF — run training first"

    FT_FUNC_GGUF="${FUNCTION_GGUF_FT:-}"
    [ -f "${FT_FUNC_GGUF:-}" ] || FT_FUNC_GGUF="${FUNCTION_GGUF}"

    FT_EMB_GGUF="${EMBEDDING_GGUF_FT:-}"
    [ -f "${FT_EMB_GGUF:-}" ] || FT_EMB_GGUF="${EMBEDDING_GGUF}"

    echo -e "\n  ${BOLD}── FT servers (cheat mode) ──${RESET}"

    FT_PORTS_ARR=("$INFERENCE_PORT_FT" "$FUNCTION_PORT_FT" "$EMBEDDING_PORT_FT")
    for port in "${FT_PORTS_ARR[@]}"; do
        pids=$(pids_on_port "$port")
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            warn "Killed existing process on port $port"
        fi
    done

    launch "inference-ft"      "$INFERENCE_PORT_FT"  "$FT_INF_GGUF"
    sleep 2  # stagger
    ALL_FT_FUNC_ARGS=(--temp 1.0 --top-p 0.95 --jinja --top-k 20 \
                     --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0)
    launch "qwen-toolcalling-ft" "$FUNCTION_PORT_FT" "$FT_FUNC_GGUF" \
        "${ALL_FT_FUNC_ARGS[@]}"
    sleep 2  # stagger
    launch "embeddinggemma-ft" "$EMBEDDING_PORT_FT"  "$FT_EMB_GGUF" \
        --embeddings --ubatch-size 2048
    sleep 2  # stagger

    ALL_PORTS+=("${FT_PORTS_ARR[@]}")
fi

# ─── --qwen: start Qwen 3.5 on dedicated comparison port ─────────────────────
if $START_QWEN; then
    QWEN_PORT="${QWEN_PORT:-9100}"
    QWEN_GGUF="${QWEN_GGUF:-}"

    # Source .env.qwen-compare if it exists
    [ -f "$REPO_ROOT/.env.qwen-compare" ] && source "$REPO_ROOT/.env.qwen-compare"

    if [ -z "$QWEN_GGUF" ] || [ ! -f "$QWEN_GGUF" ]; then
        warn "Qwen GGUF not found: ${QWEN_GGUF:-<not set>} — skipping (set QWEN_GGUF or create .env.qwen-compare)"
    else
        # Kill any existing process on the Qwen port (portable: lsof or ss+fuser)
        pids=$(pids_on_port "$QWEN_PORT")
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            warn "Killed existing process on port $QWEN_PORT"
        fi

        echo -e "\n  ${BOLD}── Qwen comparison ──${RESET}"
        launch "qwen3.5 (comparison)" "$QWEN_PORT" "$QWEN_GGUF" \
            --temp 1.0 --top-p 0.95 --jinja \
            --top-k 20 \
            --chat-template-kwargs '{"enable_thinking":false}' \
            --reasoning-budget 0

        ALL_PORTS+=("$QWEN_PORT")
    fi
fi

echo ""
info "Servers launching. Waiting for ready…"

# Wait for required servers (inference, function, embedding, vision)
all_ok=true
for port in "${ALL_PORTS[@]}"; do
    ready=false
    # Models need 5-15s each; with staggered launches + --all, total can exceed 60s
    max_wait=40
    for i in $(seq 1 "$max_wait"); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 1
    done
    if $ready; then
        info "Port $port — healthy (${i}s)"
    else
        warn "Port $port — not ready after ${max_wait}s (still loading model?)"
        all_ok=false
    fi
done

# Optional servers (Whisper, OCR) — single quick probe, never block startup
for port in "${OPTIONAL_PORTS[@]:-}"; do
    [ -z "$port" ] && continue
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
        info "Port $port — healthy"
    else
        info "Port $port — loading (optional, will appear in /health once ready)"
    fi
done

echo ""
if $BG; then
    info "Servers running in background. PIDs saved to .server-pids"
    echo "  Stop all:  kill \$(cat .server-pids) && rm .server-pids"
else
    info "Press Ctrl-C to stop all servers."
    wait
fi
