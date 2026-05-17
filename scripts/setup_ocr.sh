#!/usr/bin/env bash
# ============================================================
# setup_ocr.sh — Download GLM-OCR model + dependencies
#
# This script:
#   1. Downloads GLM-OCR GGUF files from HuggingFace (~1.4 GB total)
#   2. Installs pymupdf for PDF-to-image conversion
#   3. Downloads the Snowflake FY2025 Annual Report for demo use
#
# Prerequisites:
#   - Python venv with pip (created by setup.sh)
#   - Internet access for HuggingFace and Snowflake CDN
#
# Usage:
#   bash scripts/setup_ocr.sh
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

echo -e "\n${BOLD}GLM-OCR Setup${RESET}\n"

# ─── 1. Download GLM-OCR GGUFs ──────────────────────────────────────────────

OCR_DIR="models/glm-ocr"
OCR_GGUF="$OCR_DIR/GLM-OCR-Q8_0.gguf"
OCR_MMPROJ="$OCR_DIR/mmproj-GLM-OCR-Q8_0.gguf"

mkdir -p "$OCR_DIR"

if [ -f "$OCR_GGUF" ] && [ -f "$OCR_MMPROJ" ]; then
    info "GLM-OCR models already downloaded"
else
    echo "Downloading GLM-OCR GGUFs from ggml-org/GLM-OCR-GGUF..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('$OCR_DIR', exist_ok=True)
hf_hub_download('ggml-org/GLM-OCR-GGUF', 'GLM-OCR-Q8_0.gguf', local_dir='$OCR_DIR')
print('  ✓ GLM-OCR-Q8_0.gguf (~950 MB)')
hf_hub_download('ggml-org/GLM-OCR-GGUF', 'mmproj-GLM-OCR-Q8_0.gguf', local_dir='$OCR_DIR')
print('  ✓ mmproj-GLM-OCR-Q8_0.gguf (~484 MB)')
"
    info "GLM-OCR models downloaded to $OCR_DIR/"
fi

# ─── 2. Install pymupdf (PDF-to-image conversion) ───────────────────────────

if python3 -c "import fitz" 2>/dev/null; then
    info "pymupdf already installed"
else
    echo "Installing pymupdf..."
    pip install pymupdf>=1.24.0
    info "pymupdf installed"
fi

# ─── 3. Download Snowflake FY2025 Annual Report (demo document) ─────────────

SNOWFLAKE_PDF="data/demo-documents/snowflake-fy2025-annual-report.pdf"
if [ -f "$SNOWFLAKE_PDF" ]; then
    info "Snowflake report already downloaded"
else
    echo "Downloading Snowflake FY2025 Annual Report..."
    curl -L -o "$SNOWFLAKE_PDF" \
        "https://s26.q4cdn.com/463892824/files/doc_financials/2025/ar/Snowflake-2025-Annual-Report-and-Proxy-Web-Version.pdf"
    info "Snowflake report → $SNOWFLAKE_PDF"
fi

# ─── Summary ────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}Setup complete!${RESET}"
echo ""
echo "  Model files:"
echo "    $OCR_GGUF"
echo "    $OCR_MMPROJ"
echo ""
echo "  Demo document:"
echo "    $SNOWFLAKE_PDF"
echo ""
echo "  Next steps:"
echo "    bash scripts/start_servers.sh --bg   # OCR server auto-starts if model present"
echo "    bash scripts/demo_ocr.sh             # run OCR demo"
