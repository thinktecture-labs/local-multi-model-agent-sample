#!/usr/bin/env bash
# ============================================================
# demo_ocr.sh — OCR feature demo (health check, upload, query)
#
# Demonstrates GLM-OCR document extraction:
#   1. Health check (server + OCR availability)
#   2. Upload Nextera quarterly report PDF via /upload-document
#   3. Upload Snowflake annual report (if available)
#   4. Query agent about OCR-extracted content
#   5. Cross-validate OCR vs SQL answers
#
# Prerequisites:
#   - Server running: bash scripts/start_app.sh
#   - OCR model installed: bash scripts/setup_ocr.sh
#
# Usage:
#   bash scripts/demo_ocr.sh                    # default: localhost:8000
#   bash scripts/demo_ocr.sh http://myhost:3000  # custom server URL
# ============================================================
set -euo pipefail

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

SERVER_URL="${1:-http://localhost:8000}"

# ─── Parse --scenario flag ──────────────────────────────────────────────────
_args=("$@")
for ((i=0; i<${#_args[@]}; i++)); do
    if [[ "${_args[$i]}" == "--scenario" ]]; then
        export SCENARIO="${_args[$((i+1))]}"
    fi
done
SCENARIO="${SCENARIO:-nextera}"
# Demo paths + queries — currently only the Nextera scenario ships in this repo.
# Add a new branch here per-scenario if you create more `scenarios/<name>.json`.
DEMO_DOCS_DIR="data/demo-documents"
PRIMARY_PDF="nextera_quarterly_report.pdf"
SECONDARY_PDF="snowflake-fy2025-annual-report.pdf"
QUERIES=(
    "What was total revenue in Q4 2024?"
    "Which customer has the highest MRR?"
    "How many Snowflake customers spend more than 1M ARR?"
    "What is Snowflake's net revenue retention rate?"
)

echo -e "\n${BOLD}${CYAN}═══ GLM-OCR Demo (${SCENARIO}) ═══${RESET}\n"

# ─── 1. Health check ────────────────────────────────────────────────────────

echo -e "${BOLD}1. Health check${RESET}"
HEALTH=$(curl -sf "$SERVER_URL/health" 2>/dev/null) || error "Server not reachable at $SERVER_URL"
OCR_OK=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'].get('OCR', False))")
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
echo "  Status: $STATUS"
echo "  OCR: $OCR_OK"
[ "$OCR_OK" = "True" ] && info "GLM-OCR available" || warn "GLM-OCR not running — will use pypdf fallback"
echo ""

# ─── 2. Upload primary demo document ────────────────────────────────────────

PRIMARY_PATH="$DEMO_DOCS_DIR/$PRIMARY_PDF"
if [ -f "$PRIMARY_PATH" ]; then
    echo -e "${BOLD}2. Uploading ${PRIMARY_PDF}${RESET}"
    curl -sf -N -X POST "$SERVER_URL/upload-document" \
        -F "file=@$PRIMARY_PATH" 2>/dev/null | while IFS= read -r line; do
        if [[ "$line" == data:* ]]; then
            STAGE=$(echo "${line#data: }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stage',''))" 2>/dev/null)
            MSG=$(echo "${line#data: }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',''))" 2>/dev/null)
            echo "  [$STAGE] $MSG"
        fi
    done
    echo ""
else
    warn "${PRIMARY_PDF} not found in ${DEMO_DOCS_DIR}"
fi

# ─── 3. Upload secondary demo document ──────────────────────────────────────

SECONDARY_PATH="$DEMO_DOCS_DIR/$SECONDARY_PDF"
if [ -f "$SECONDARY_PATH" ]; then
    echo -e "${BOLD}3. Uploading ${SECONDARY_PDF}${RESET}"
    curl -sf -N -X POST "$SERVER_URL/upload-document" \
        -F "file=@$SECONDARY_PATH" 2>/dev/null | while IFS= read -r line; do
        if [[ "$line" == data:* ]]; then
            STAGE=$(echo "${line#data: }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stage',''))" 2>/dev/null)
            MSG=$(echo "${line#data: }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',''))" 2>/dev/null)
            echo "  [$STAGE] $MSG"
        fi
    done
    echo ""
else
    warn "${SECONDARY_PDF} not found in ${DEMO_DOCS_DIR}"
fi

# ─── 4. Query OCR-extracted content ─────────────────────────────────────────

echo -e "${BOLD}4. Querying OCR-extracted content${RESET}"
for QUERY in "${QUERIES[@]}"; do
    echo ""
    echo -e "  ${CYAN}Q:${RESET} $QUERY"
    RESP=$(curl -sf -X POST "$SERVER_URL/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$QUERY\"}" 2>/dev/null) || { warn "Query failed"; continue; }
    ANSWER=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])" 2>/dev/null)
    TIMING=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['execution_time_ms'])" 2>/dev/null)
    echo -e "  ${GREEN}A:${RESET} $ANSWER"
    echo -e "  ${YELLOW}T:${RESET} ${TIMING}ms"
done

# ─── 5. Cross-validation ────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}5. Cross-validation: OCR answer vs SQL answer${RESET}"
XVAL_OCR_Q="What was the revenue in Q3 2024?"
XVAL_SQL_Q="What were the total sales in Q3 2024?"
echo "  (Both should return the same Q3 2024 revenue figure)"
echo ""
echo -e "  ${CYAN}OCR path:${RESET} '${XVAL_OCR_Q}' (from uploaded PDF)"
RESP1=$(curl -sf -X POST "$SERVER_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$XVAL_OCR_Q\"}" 2>/dev/null) || true
echo -e "  ${GREEN}→${RESET} $(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])" 2>/dev/null)"
echo ""
echo -e "  ${CYAN}SQL path:${RESET} '${XVAL_SQL_Q}' (from database)"
RESP2=$(curl -sf -X POST "$SERVER_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$XVAL_SQL_Q\"}" 2>/dev/null) || true
echo -e "  ${GREEN}→${RESET} $(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])" 2>/dev/null)"

echo ""
info "Demo complete!"
