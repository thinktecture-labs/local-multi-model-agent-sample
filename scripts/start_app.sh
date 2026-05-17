#!/usr/bin/env bash
# ============================================================
# start_app.sh — Start llama-server instances + FastAPI web app
#
# One command to go from zero to a running agent with Observatory UI.
# Starts the three llama-server instances, waits for health, seeds
# data if needed, then launches the FastAPI server with uvicorn.
#
# Usage:
#   bash scripts/start_app.sh                  # base models
#   bash scripts/start_app.sh --ft             # fine-tuned models
#   bash scripts/start_app.sh --scenario foo   # use scenarios/foo.json
#   bash scripts/start_app.sh --port 3000      # custom API port
#
# Stop:
#   Ctrl-C (stops uvicorn + all llama-server instances)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

# ─── Parse flags ─────────────────────────────────────────────────────────────

SERVER_FLAGS=("--bg")
ALL_ARGS=("$@")  # save before the while-shift loop consumes $@
API_PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ft)        SERVER_FLAGS+=("--ft"); shift ;;
        --base)      SERVER_FLAGS+=("--base"); shift ;;
        --all)       SERVER_FLAGS+=("--all"); shift ;;
        --qwen)      SERVER_FLAGS+=("--qwen"); shift ;;
        --cpu)       SERVER_FLAGS+=("--cpu"); shift ;;
        --scenario)  export SCENARIO="${2:?--scenario requires a name}"; SERVER_FLAGS+=("--scenario" "$SCENARIO"); shift 2 ;;
        --port)      API_PORT="${2:-8000}"; shift 2 ;;
        [0-9]*)      API_PORT="$1"; shift ;;
        *)           shift ;;
    esac
done

# ─── Cleanup on exit ────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo -e "${BOLD}Shutting down…${RESET}"
    if [ -f .server-pids ]; then
        kill $(cat .server-pids) 2>/dev/null || true
        rm -f .server-pids
        info "llama-server instances stopped"
    fi
}
trap cleanup EXIT INT TERM

# ─── Activate Python env ────────────────────────────────────────────────────

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    info "Activated .venv"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    info "Using virtualenv: $VIRTUAL_ENV"
else
    warn "No .venv found. Run: bash setup.sh"
fi

# ─── Start llama-server instances ────────────────────────────────────────────

echo -e "\n${BOLD}${CYAN}═══ Starting llama-server instances ═══${RESET}\n"
bash scripts/start_servers.sh "${SERVER_FLAGS[@]}"

# ─── Wait for optional/slow servers (Whisper, OCR, Qwen) ───────────────────
# start_servers.sh --bg returns once required servers (9090-9093) are healthy,
# but optional servers may still be loading. Wait for them here so uvicorn's
# startup probes see everything ready.

[ -f .env ]       && source .env
[ -f .env.local ] && source .env.local

OPTIONAL_WAIT_PORTS=()
WHISPER_BIN="${WHISPER_SERVER:-}"
WHISPER_MDL="${WHISPER_MODEL:-}"
[ -f "${WHISPER_BIN:-}" ] && [ -f "${WHISPER_MDL:-}" ] && OPTIONAL_WAIT_PORTS+=("${WHISPER_PORT:-9097}")

OCR_GGUF_FILE="${OCR_GGUF:-}"
OCR_MMPROJ_FILE="${OCR_MMPROJ:-}"
[ -f "${OCR_GGUF_FILE:-}" ] && [ -f "${OCR_MMPROJ_FILE:-}" ] && OPTIONAL_WAIT_PORTS+=("${OCR_PORT:-9098}")

for arg in "${ALL_ARGS[@]}"; do
    [[ "$arg" == "--all" || "$arg" == "--qwen" ]] && OPTIONAL_WAIT_PORTS+=("${QWEN_PORT:-9100}")
done

if [ ${#OPTIONAL_WAIT_PORTS[@]} -gt 0 ]; then
    echo ""
    info "Waiting for optional servers to be ready…"
    for port in "${OPTIONAL_WAIT_PORTS[@]}"; do
        ready=false
        printf "  Port %s: " "$port"
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
                ready=true; break
            fi
            printf "."
            sleep 1
        done
        if $ready; then
            echo ""
            info "Port $port — healthy (${i}s)"
        else
            echo ""
            warn "Port $port — not ready after 60s (continuing anyway)"
        fi
    done
fi

# ─── Seed data if needed ────────────────────────────────────────────────────

if [ ! -f data/business.db ]; then
    echo ""
    info "Seeding demo data…"
    .venv/bin/python3 -m data.loader
fi

# ─── Reclaim API port if a stale uvicorn still holds it ─────────────────────
# Prior runs can leak uvicorn (SSH disconnect, kill -9, etc.) — the cleanup trap
# only handles llama-servers via .server-pids. Without this, the next start_app.sh
# fails with "Address already in use".
#
# uvicorn --reload spawns child processes via Python's multiprocessing.spawn /
# `--multiprocessing-fork` mechanism; the parent's argv contains "uvicorn"
# but the children's argv only contains "multiprocessing.spawn" or
# "multiprocessing-fork". A crashed reload cycle can leave a child holding
# :8000 without the parent — happened on DGX after a force-killed eval chain
# and stranded the port (2026-05-17). Match all four uvicorn-fingerprint
# patterns so the reclaim covers the orphan-child case too.

if command -v lsof >/dev/null 2>&1; then
    # -sTCP:LISTEN filters to processes LISTENING on the port (uvicorn is one).
    # Without this, lsof also returns ngrok's outbound ESTABLISHED connection
    # to 127.0.0.1:8000, causing a false positive on the safety check below.
    # `|| true` keeps `set -e + pipefail` from killing the script when lsof
    # finds nothing (exit 1) -- empty result is the common, healthy case.
    stale_pid=$(lsof -ti tcp:"$API_PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)
    if [ -n "$stale_pid" ]; then
        stale_cmd=$(ps -p "$stale_pid" -o command= 2>/dev/null || echo "")
        if echo "$stale_cmd" | grep -qE "uvicorn|src\.server|multiprocessing\.spawn|--multiprocessing-fork"; then
            warn "Reclaiming port $API_PORT from stale uvicorn/multiprocessing orphan (pid=$stale_pid)"
            kill "$stale_pid" 2>/dev/null || true
            sleep 0.5
            kill -9 "$stale_pid" 2>/dev/null || true
        else
            error "Port $API_PORT is in use by a non-uvicorn listener: $stale_cmd"
        fi
    fi
fi

# ─── Start FastAPI server ────────────────────────────────────────────────────

echo -e "\n${BOLD}${CYAN}═══ Starting API server ═══${RESET}\n"
info "API docs:      http://localhost:${API_PORT}/docs"
info "Observatory:   http://localhost:${API_PORT}/app"
echo ""

.venv/bin/python3 -m uvicorn src.server:app --host 0.0.0.0 --port "$API_PORT" --reload --reload-dir src --reload-dir data --reload-dir scenarios
