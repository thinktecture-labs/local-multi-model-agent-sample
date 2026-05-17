#!/usr/bin/env bash
# =============================================================================
# run_all_evals.sh — Run the full eval matrix on the current machine.
#
# Probes which servers are running and runs every applicable eval against
# every available model variant.  Results are saved as JSON files in results/
# with machine + model metadata embedded for later aggregation.
#
# Usage:
#   bash scripts/run_all_evals.sh                          # auto-detect all
#   bash scripts/run_all_evals.sh --include-moe            # also probe port 9100
#   bash scripts/run_all_evals.sh --label "gpu-host-1"     # override machine label
#   bash scripts/run_all_evals.sh --skip-slow              # skip vision + adversarial
#
# After running on all machines, aggregate with:
#   python -m finetune.collect_results
# =============================================================================
set -euo pipefail

INCLUDE_MOE=false
SKIP_SLOW=false
MACHINE_LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-moe)   INCLUDE_MOE=true; shift ;;
        --skip-slow)     SKIP_SLOW=true; shift ;;
        --label)         MACHINE_LABEL="$2"; shift 2 ;;
        --label=*)       MACHINE_LABEL="${1#*=}"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ─── Machine label ────────────────────────────────────────────────────────────
if [ -z "$MACHINE_LABEL" ]; then
    MACHINE_LABEL="$(hostname -s)"
fi

# ─── GPU detection ────────────────────────────────────────────────────────────
GPU_LABEL="unknown"
if command -v nvidia-smi &>/dev/null; then
    GPU_LABEL="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '-' || echo 'nvidia')"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    GPU_LABEL="$(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model:' | head -1 | sed 's/.*Chipset Model: //' | tr ' ' '-' || echo 'metal')"
elif command -v vulkaninfo &>/dev/null; then
    GPU_LABEL="$(vulkaninfo --summary 2>/dev/null | grep 'deviceName' | head -1 | sed 's/.*= //' | tr ' ' '-' || echo 'vulkan')"
fi

PLATFORM="$(uname -s)"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# ─── Python interpreter — prefer local venv if present ───────────────────────
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python"
fi

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RESET="\033[0m"
info()    { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
heading() { echo -e "\n${BOLD}${CYAN}▶ $*${RESET}"; }
skip()    { echo -e "  ${YELLOW}—${RESET} $*"; }

echo -e "\n${BOLD}Eval matrix — ${MACHINE_LABEL} / ${GPU_LABEL} / ${PLATFORM}${RESET}"
echo "  Timestamp: $TIMESTAMP"
echo "  Results → $RESULTS_DIR/"

# ─── Port probe helper ────────────────────────────────────────────────────────
port_healthy() {
    curl -sf "http://localhost:$1/health" >/dev/null 2>&1
}

# ─── Metadata injection helper ───────────────────────────────────────────────
# Adds _meta block to a saved JSON file so collect_results.py can aggregate.
inject_meta() {
    local file="$1" eval_type="$2" model_tag="$3" port="$4"
    "$PYTHON" -c "
import json, sys
with open('$file') as f: d = json.load(f)
d['_meta'] = {
    'machine':    '$MACHINE_LABEL',
    'gpu':        '$GPU_LABEL',
    'platform':   '$PLATFORM',
    'eval_type':  '$eval_type',
    'model_tag':  '$model_tag',
    'port':       $port,
    'run_ts':     '$TIMESTAMP',
}
with open('$file', 'w') as f: json.dump(d, f, indent=2)
"
}

# ─── Run one tool-selection eval ─────────────────────────────────────────────
run_tool_sel() {
    local port="$1" model_tag="$2"
    local out="${RESULTS_DIR}/${MACHINE_LABEL}_${model_tag}_toolsel_${TIMESTAMP}.json"
    echo "  Running tool-selection eval → $out"
    "$PYTHON" -m finetune.eval_tool_routing \
        --function-port "$port" \
        --function-model "$model_tag" \
        --save "$out" && inject_meta "$out" "tool_selection" "$model_tag" "$port"
}

# ─── Run one multi-step eval ──────────────────────────────────────────────────
run_multi_step() {
    local port="$1" model_tag="$2"
    local out="${RESULTS_DIR}/${MACHINE_LABEL}_${model_tag}_multistep_${TIMESTAMP}.json"
    echo "  Running multi-step eval → $out"
    "$PYTHON" -m finetune.eval_multi_step \
        --function-port "$port" \
        --function-model "$model_tag" \
        --save "$out" && inject_meta "$out" "multi_step" "$model_tag" "$port"
}

# =============================================================================
# 1. Intent classification — gemma3 fallback path
# (Primary intent path is LogReg, evaluated in Section 2.)
# =============================================================================
heading "Intent classification (gemma3 fallback)"

if port_healthy 9094; then
    info "gemma3-ft detected (port 9094)"
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_gemma3-ft_intent_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_gemma3 --save "$OUT" \
        && inject_meta "$OUT" "intent" "gemma3-ft" 9094
elif port_healthy 9090; then
    warn "gemma3-ft not running — using base (port 9090)"
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_gemma3-base_intent_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_gemma3 --save "$OUT" \
        && inject_meta "$OUT" "intent" "gemma3-base" 9090
else
    skip "No inference server on 9090/9094 — skipping intent eval"
fi

# =============================================================================
# 2. LogReg intent classifier
# =============================================================================
heading "LogReg intent classifier"

if port_healthy 9092 || port_healthy 9096; then
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_logreg_intent_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_intent_logreg --save "$OUT" \
        && inject_meta "$OUT" "logreg_intent" "logreg-v1" 0
else
    skip "No embedding server on 9092/9096 — skipping LogReg eval"
fi

# =============================================================================
# 3. Embedding model
# =============================================================================
heading "Embedding model"

if port_healthy 9096; then
    info "embeddinggemma-ft detected (port 9096)"
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_embeddinggemma-ft_embed_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_embeddinggemma --save "$OUT" \
        && inject_meta "$OUT" "embedding" "embeddinggemma-ft" 9096
elif port_healthy 9092; then
    warn "embeddinggemma-ft not running — using base (port 9092)"
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_embeddinggemma-base_embed_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_embeddinggemma --save "$OUT" \
        && inject_meta "$OUT" "embedding" "embeddinggemma-base" 9092
else
    skip "No embedding server — skipping embedding eval"
fi

# =============================================================================
# 4. Tool selection — all available function models
# =============================================================================
heading "Tool selection"

if port_healthy 9095; then
    info "Qwen3.5-4B FT detected (port 9095)"
    run_tool_sel 9095 "qwen3.5-4b-ft-v8"
elif port_healthy 9091; then
    info "tool-calling model detected (port 9091)"
    run_tool_sel 9091 "qwen-toolcalling-base"
else
    skip "No function server on 9091/9095 — skipping tool-selection eval"
fi

if port_healthy 9091 && port_healthy 9095; then
    info "Both base and FT function servers running — also benchmarking base for comparison"
    run_tool_sel 9091 "qwen-toolcalling-base"
fi

if [ "$INCLUDE_MOE" = true ]; then
    if port_healthy 9100; then
        info "Qwen3.5-35B-A3B MoE detected (port 9100)"
        run_tool_sel 9100 "qwen3.5-35b-a3b-moe"
    else
        warn "MoE server not running on port 9100 — start with: bash scripts/start_servers.sh --qwen --bg"
    fi
fi

# =============================================================================
# 5. Multi-step — same variants as tool selection
# =============================================================================
heading "Multi-step reasoning"

if port_healthy 9095; then
    run_multi_step 9095 "qwen3.5-4b-ft-v8"
elif port_healthy 9091; then
    run_multi_step 9091 "qwen-toolcalling-base"
fi

if port_healthy 9091 && port_healthy 9095; then
    run_multi_step 9091 "qwen-toolcalling-base"
fi

if [ "$INCLUDE_MOE" = true ] && port_healthy 9100; then
    run_multi_step 9100 "qwen3.5-35b-a3b-moe"
fi

# =============================================================================
# 6. Adversarial robustness (uses inference + embedding server)
# =============================================================================
heading "Adversarial robustness"

if [ "$SKIP_SLOW" = true ]; then
    skip "Skipped (--skip-slow)"
elif port_healthy 9090 || port_healthy 9094; then
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_adversarial_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_adversarial --save "$OUT" \
        && inject_meta "$OUT" "adversarial" "full-stack" 0
else
    skip "No inference server — skipping adversarial eval"
fi

# =============================================================================
# 7. Vision (optional — only if vision server running)
# =============================================================================
heading "Vision"

if [ "$SKIP_SLOW" = true ]; then
    skip "Skipped (--skip-slow)"
elif port_healthy 9093; then
    OUT="${RESULTS_DIR}/${MACHINE_LABEL}_vision_${TIMESTAMP}.json"
    "$PYTHON" -m finetune.eval_vision --save "$OUT" \
        && inject_meta "$OUT" "vision" "gemma3-4b-vision" 9093
else
    skip "No vision server on port 9093 — skipping vision eval"
fi

# =============================================================================
# Done — list saved results
# =============================================================================
echo -e "\n${BOLD}Results written:${RESET}"
ls -1 "${RESULTS_DIR}/"*"${TIMESTAMP}"*.json 2>/dev/null | sed 's/^/  /'

echo -e "\nAggregate all machines with:"
echo -e "  ${BOLD}python -m finetune.collect_results${RESET}"
echo -e "  ${BOLD}python -m finetune.collect_results --matrix results/benchmark_matrix.json${RESET}"
