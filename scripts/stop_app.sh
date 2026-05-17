#!/usr/bin/env bash
# scripts/stop_app.sh — robust stop of the full local stack.
#
# Kills llama-server + whisper-server + uvicorn for this project, via three
# overlapping mechanisms so a stale PID file or a leaked binary can't leave
# orphans behind:
#
#   1. .server-pids / .server-pids-ft files (matches start_servers.sh's tracking)
#   2. pkill -f against the vendored binary paths (catches orphans whose PIDs
#      drifted from the tracking files)
#   3. fuser/lsof on the well-known ports (catches non-vendored listeners
#      bound to our ports — e.g. another uvicorn flavour)
#
# Usage:
#   bash scripts/stop_app.sh

set -uo pipefail   # NB: no `-e` — pkill / kill returning 1 (no match) is expected.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RESET="\033[0m"
info() { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn() { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }

PORTS=(9090 9091 9092 9093 9094 9095 9096 9097 9098 9100 8000)
BINS=(
  "vendor/llama.cpp/build/bin/llama-server"
  "vendor/whisper.cpp/build/bin/whisper-server"
  "uvicorn src.server"
)

# 1) PID files
for pidfile in .server-pids .server-pids-ft; do
    if [ -f "$pidfile" ]; then
        pids=$(cat "$pidfile" 2>/dev/null | tr '\n' ' ')
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null || true
            info "Stopped via $pidfile (${pids// /, })"
        fi
        rm -f "$pidfile"
    fi
done

# 2) pkill by binary
for pat in "${BINS[@]}"; do
    if pkill -f "$pat" 2>/dev/null; then
        info "Killed by name: $pat"
    fi
done

# Brief pause so killed processes release ports before the port sweep
sleep 1

# 3) Anything still listening on our ports
for port in "${PORTS[@]}"; do
    pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
        warn "Killed lingering listener on :$port (pid=$pids)"
    fi
done

# Final verify
echo
remaining=$(lsof -tiTCP -sTCP:LISTEN 2>/dev/null | xargs -I{} ps -p {} -o pid=,command= 2>/dev/null | grep -E "llama-server|whisper-server|uvicorn.*src.server" || true)
if [ -n "$remaining" ]; then
    warn "Still alive:"
    echo "$remaining"
    exit 1
fi
info "Stack stopped cleanly."
