#!/usr/bin/env bash
# ============================================================
# demo_voice.sh — CLI demo for voice-to-voice pipeline
#
# Tests the voice endpoints against a running server.
# Requires: ffmpeg, a running server (python server.py),
#           and whisper-server + Piper TTS installed.
#
# Usage:
#   bash scripts/demo_voice.sh                    # quick health check + TTS test
#   bash scripts/demo_voice.sh --full             # full round-trip with sample audio
#   bash scripts/demo_voice.sh --record           # record from mic + full round-trip
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; }
step()  { echo -e "\n${BOLD}${CYAN}───${RESET} $* ${CYAN}───${RESET}"; }

# ─── Activate Python env ────────────────────────────────────────────────────

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    info "Activated .venv"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    info "Using virtualenv: $VIRTUAL_ENV"
else
    warn "No .venv found — using system Python"
fi

SERVER_URL="${SERVER_URL:-http://localhost:8000}"
FULL=false
RECORD=false

for arg in "$@"; do
    case "$arg" in
        --full)   FULL=true ;;
        --record) RECORD=true; FULL=true ;;
    esac
done

# ─── 1. Health Check ────────────────────────────────────────────────────────

step "Health Check"

HEALTH=$(curl -sf "$SERVER_URL/health" 2>/dev/null || echo "FAIL")
if [ "$HEALTH" = "FAIL" ]; then
    error "Server not reachable at $SERVER_URL"
    echo "  Start the server:  python server.py"
    exit 1
fi
info "Server healthy"

# Check whisper
WHISPER_OK=$(echo "$HEALTH" | python3 -c "import sys, json; h=json.load(sys.stdin); print(h.get('models',{}).get('WHISPER', False))" 2>/dev/null || echo "False")
if [ "$WHISPER_OK" = "True" ]; then
    info "Whisper STT: available"
else
    warn "Whisper STT: not available (voice chat won't work, TTS-only mode OK)"
fi

# ─── 2. TTS Test — /voice/synthesize ────────────────────────────────────────

step "TTS Test (English)"

TTS_EN=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST "$SERVER_URL/voice/synthesize?text=Hello%20world.%20This%20is%20a%20test%20of%20the%20local%20text%20to%20speech%20system.&language=en" 2>/dev/null || echo "000")

if [ "$TTS_EN" = "200" ]; then
    info "English TTS: working (HTTP 200)"
    # Save and play if possible
    curl -sf -X POST "$SERVER_URL/voice/synthesize?text=Hello%20world.%20This%20is%20a%20test.&language=en" \
        -o /tmp/demo_voice_en.wav 2>/dev/null
    if [ -f /tmp/demo_voice_en.wav ]; then
        SIZE=$(wc -c < /tmp/demo_voice_en.wav | tr -d ' ')
        info "  Saved to /tmp/demo_voice_en.wav ($SIZE bytes)"
        if command -v afplay &>/dev/null; then
            echo -e "  ${BOLD}Playing English TTS...${RESET}"
            afplay /tmp/demo_voice_en.wav 2>/dev/null || true
        elif command -v aplay &>/dev/null; then
            echo -e "  ${BOLD}Playing English TTS...${RESET}"
            aplay /tmp/demo_voice_en.wav 2>/dev/null || true
        fi
    fi
elif [ "$TTS_EN" = "503" ]; then
    warn "English TTS: Piper not available (503) — run: bash scripts/setup_voice.sh"
else
    error "English TTS: unexpected response ($TTS_EN)"
fi

step "TTS Test (German)"

TTS_DE=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST "$SERVER_URL/voice/synthesize?text=Hallo%20Welt.%20Dies%20ist%20ein%20Test%20des%20lokalen%20Sprachsystems.&language=de" 2>/dev/null || echo "000")

if [ "$TTS_DE" = "200" ]; then
    info "German TTS: working (HTTP 200)"
    curl -sf -X POST "$SERVER_URL/voice/synthesize?text=Hallo%20Welt.%20Dies%20ist%20ein%20Test.&language=de" \
        -o /tmp/demo_voice_de.wav 2>/dev/null
    if [ -f /tmp/demo_voice_de.wav ]; then
        SIZE=$(wc -c < /tmp/demo_voice_de.wav | tr -d ' ')
        info "  Saved to /tmp/demo_voice_de.wav ($SIZE bytes)"
        if command -v afplay &>/dev/null; then
            echo -e "  ${BOLD}Playing German TTS...${RESET}"
            afplay /tmp/demo_voice_de.wav 2>/dev/null || true
        elif command -v aplay &>/dev/null; then
            echo -e "  ${BOLD}Playing German TTS...${RESET}"
            aplay /tmp/demo_voice_de.wav 2>/dev/null || true
        fi
    fi
elif [ "$TTS_DE" = "503" ]; then
    warn "German TTS: Piper not available (503)"
else
    error "German TTS: unexpected response ($TTS_DE)"
fi

# ─── 3. Full Voice Round-Trip ───────────────────────────────────────────────

if ! $FULL; then
    echo ""
    info "Quick test done. Run with --full for full voice round-trip."
    exit 0
fi

if [ "$WHISPER_OK" != "True" ]; then
    error "Whisper not available — cannot do full voice round-trip"
    echo "  Run: bash scripts/setup_voice.sh"
    exit 1
fi

# Generate or record test audio
AUDIO_FILE="/tmp/demo_voice_input.wav"

if $RECORD; then
    step "Recording from Microphone"
    echo -e "  ${BOLD}Press Enter to start recording, Ctrl-C to stop...${RESET}"
    read -r
    if command -v ffmpeg &>/dev/null; then
        echo -e "  ${BOLD}Recording... (Ctrl-C to stop)${RESET}"
        # macOS default audio input
        if [[ "$(uname)" == "Darwin" ]]; then
            ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -t 10 "$AUDIO_FILE" -y 2>/dev/null || true
        else
            ffmpeg -f pulse -i default -ar 16000 -ac 1 -t 10 "$AUDIO_FILE" -y 2>/dev/null || true
        fi
        info "Recorded to $AUDIO_FILE"
    else
        error "ffmpeg not found — needed for recording"
        exit 1
    fi
else
    step "Generating Test Audio (TTS → STT round-trip)"
    # Use our own TTS to generate test audio, then feed it back through STT
    curl -sf -X POST "$SERVER_URL/voice/synthesize?text=What%20were%20our%20Q3%20results%20this%20year&language=en" \
        -o "$AUDIO_FILE" 2>/dev/null
    if [ ! -f "$AUDIO_FILE" ] || [ ! -s "$AUDIO_FILE" ]; then
        error "Could not generate test audio via TTS"
        exit 1
    fi
    SIZE=$(wc -c < "$AUDIO_FILE" | tr -d ' ')
    info "Generated test audio: $AUDIO_FILE ($SIZE bytes)"
fi

step "Voice Round-Trip (POST /voice/chat)"

echo -e "  ${BOLD}Sending audio to /voice/chat...${RESET}"
VOICE_RESP=$(curl -sf -X POST "$SERVER_URL/voice/chat" \
    -F "file=@${AUDIO_FILE};type=audio/wav" 2>/dev/null || echo "FAIL")

if [ "$VOICE_RESP" = "FAIL" ]; then
    error "Voice chat request failed"
    exit 1
fi

# Parse SSE events
echo ""
echo "$VOICE_RESP" | while IFS= read -r line; do
    if [[ "$line" == event:* ]]; then
        EVENT="${line#event: }"
        echo -e "  ${BOLD}${CYAN}[$EVENT]${RESET}"
    elif [[ "$line" == data:* ]]; then
        DATA="${line#data: }"
        echo "$DATA" | python3 -m json.tool 2>/dev/null | sed 's/^/    /'
    fi
done

# Extract and play audio response
AUDIO_URL=$(echo "$VOICE_RESP" | grep '"url":' | head -1 | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: '):
        d = json.loads(line[6:])
        if 'url' in d and d['url']:
            print(d['url'])
            break
" 2>/dev/null || echo "")

if [ -n "$AUDIO_URL" ]; then
    step "Playing Agent Response Audio"
    curl -sf "$SERVER_URL$AUDIO_URL" -o /tmp/demo_voice_response.wav 2>/dev/null
    if [ -f /tmp/demo_voice_response.wav ]; then
        SIZE=$(wc -c < /tmp/demo_voice_response.wav | tr -d ' ')
        info "Response audio: /tmp/demo_voice_response.wav ($SIZE bytes)"
        if command -v afplay &>/dev/null; then
            afplay /tmp/demo_voice_response.wav 2>/dev/null || true
        elif command -v aplay &>/dev/null; then
            aplay /tmp/demo_voice_response.wav 2>/dev/null || true
        fi
    fi
fi

echo ""
info "Voice round-trip demo complete!"
