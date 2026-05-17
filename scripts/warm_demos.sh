#!/usr/bin/env bash
# ============================================================
# warm_demos.sh — Pre-warm caches by firing canonical demo queries
#
# After start_app.sh boots, the first query through each layer pays
# a cold-cache tax (~150-700 ms per model). Stage demos hit the warm
# path. This script fires every canonical query the speaker will run
# on stage to prime KV caches and JIT'd code paths.
#
# Run after start_app.sh finishes booting. Eats ~15-20 s upfront,
# claws back ~500-700 ms per stage demo.
#
# Coverage (v1.2):
#   D0  — cold-open voice queries (Q1 + Q2)
#   D1  — Forces teaser (Show Mode chip)
#   D2  — Agent walkthrough layers L1, L2, L4, L5
#   D3  — Lobotomy FT trio (B1.2, B1.3 — B1.1 reuses D2 L2)
#   D4  — Hybrid escalation (B2.1 — B1 reuses D2 L5)
#   D8  — Final Jarvis Q1 reuses D3 B1.2
#
# NOT warmed (intentional — see end of file for why):
#   D2 L6 vision · D5 Browser · D6 iPhone · D7 your-data · D4 cloud · D8 Q2
#
# Usage:
#   bash scripts/warm_demos.sh                          # default localhost:8000
#   bash scripts/warm_demos.sh http://localhost:8000    # explicit URL
#   bash scripts/warm_demos.sh https://...ngrok.io      # via ngrok (RTX)
# ============================================================
set -euo pipefail

API_URL="${1:-http://localhost:8000}"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; DIM="\033[2m"; RESET="\033[0m"

# Probe the API is alive
if ! curl -sf "$API_URL/health" >/dev/null 2>&1; then
    echo -e "${BOLD}${RED}✗${RESET} API not reachable at $API_URL"
    echo "  Start the server first: bash scripts/start_app.sh --all"
    exit 1
fi

echo -e "\n${BOLD}${CYAN}═══ Warming demo caches via $API_URL ═══${RESET}\n"

warm_one() {
    local label="$1"
    local query="$2"
    local payload
    payload=$(printf '{"query":%s}' "$(printf '%s' "$query" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')")

    printf "  %-9s  %-58s  " "$label" "${query:0:55}$([ ${#query} -gt 55 ] && echo "...")"
    local t0 t1 elapsed_ms
    t0=$(python3 -c 'import time; print(time.perf_counter())')
    if curl -sf -X POST "$API_URL/query" -H "Content-Type: application/json" -d "$payload" >/dev/null 2>&1; then
        t1=$(python3 -c 'import time; print(time.perf_counter())')
        elapsed_ms=$(python3 -c "print(round(($t1 - $t0) * 1000))")
        printf "${GREEN}✓${RESET}  ${DIM}%s ms${RESET}\n" "$elapsed_ms"
    else
        printf "${RED}✗${RESET}  (request failed — model may not be ready)\n"
    fi
}

# Demo 0 — cold-open voice (two single-fact queries — fast TTS, no list scramble)
warm_one "D0 Q1"     "Who's our top customer?"
warm_one "D0 Q2"     "What was our revenue in 2023?"

# Demo 1 — Forces teaser ($84,900 single-row Q3 path)
warm_one "D1"        "What was total revenue in Q3 2024?"

# Demo 2 — Agent layer-by-layer
warm_one "D2 L1"     "What can you help me with?"
warm_one "D2 L2"     "How many customers do we have?"
warm_one "D2 L4"     "How many customers do we have, and what would total ARR be if each paid 999 per month?"
warm_one "D2 L5"     "What's the pricing for the Enterprise plan?"

# Demo 3 — Lobotomy FT trio (B1.1 reuses D2 L2; B1.2 also serves D8 Q1)
warm_one "D3 B1.2"   "What was Q3 2024 revenue?"
warm_one "D3 B1.3"   "Calculate 23 deals × \$52,400"

# Demo 4 — Hybrid Escalation (B1 reuses D2 L5; B2.1 hits the confidence-router path)
warm_one "D4 B2.1"   "What is Nextera's uptime SLA on the Enterprise plan, and how does it compare to OpenAI's API availability guarantees?"

echo ""
echo -e "${BOLD}${GREEN}✓${RESET} Caches warm. Stage demos will run fast for the next ~30 minutes."
echo -e "${DIM}  Re-run this script if there's been a long idle gap before going on stage.${RESET}"

# ============================================================
# Why we DON'T warm certain demos via this script:
#
#   D2 L6 (vision)    — needs a PDF dropped via UI (e.g. Snowflake FY2025);
#                       do it pre-stage interactively if you want OCR pre-warmed.
#   D5 (Browser)      — separate WebGPU runtime in Chrome; not on this API.
#                       Pre-cache the model by clicking "Download & Load Model"
#                       once before stage (cold ~30-60s, warm <2s).
#   D6 (iPhone)       — different app, different stack. Pre-flight on device.
#   D7 (your-data)    — needs a plain-text file dropped via UI. Drop it
#                       interactively pre-stage if you want it cached;
#                       otherwise the live ~2.2s indexing IS part of the demo.
#   D4 cloud column   — would burn OpenAI credits. Manual pre-warm: fire one
#                       query in Three-Way Split mode pre-stage, discard.
#   D8 Q2 (canned)    — server-side intercept (regex match in voice_routes.py),
#                       skips the agent entirely. Nothing to warm.
# ============================================================
