#!/bin/bash
# show_stack.sh — show the 5 production models running on this MBP.
# Stage-demo output: PID · PROCESS · PORT · MODEL · PARAMS · QUANT · CTX · ROLE · RAM.
#
# PARAMS and QUANT are read live from each .gguf file's metadata
# (general.size_label and general.file_type) via the gguf Python package
# in .venv. Falls back to hardcoded values if the read fails.
#
# ROLE is hardcoded — there's no metadata field for "what role this model
# plays in the agent" — that's a narrative choice.
#
# Compatible with macOS default Bash 3.x (no associative arrays).
#
# Usage:  ./scripts/show_stack.sh
# Demo:   between Seven Capabilities reveal and L1 in DEMO #2.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_PY="$REPO_ROOT/vendor/llama.cpp/gguf-py"
VENV_PY="$REPO_ROOT/.venv/bin/python"

# --- Hardcoded fallbacks (used if Python metadata read fails) ---
fallback() {
    case "$1:$2" in
        9094:role)   echo "router (FT)"        ;;
        9094:params) echo "1 B"                ;;
        9094:quant)  echo "F16"                ;;
        9095:role)   echo "tool calling (FT)"  ;;
        9095:params) echo "4 B"                ;;
        9095:quant)  echo "F16"                ;;
        9096:role)   echo "embeddings (FT)"    ;;
        9096:params) echo "308 M"              ;;
        9096:quant)  echo "Q8_0"               ;;
        9093:role)   echo "synthesis + vision" ;;
        9093:params) echo "4 B"                ;;
        9093:quant)  echo "Q4_K_M"             ;;
        9098:role)   echo "OCR"                ;;
        9098:params) echo "0.9 B"              ;;
        9098:quant)  echo "Q8_0"               ;;
    esac
}

# --- Step 1: gather (port, pid, rss, model_path, ctx) for the 5 production processes ---
PROD_PORTS="9094 9095 9096 9093 9098"
PS_DATA=$(ps -eo pid,rss,command -ww 2>/dev/null | grep llama-server | grep -v grep)

# Print one TSV line per running production port: port \t pid \t rss \t path \t ctx
running_data=""
for port in $PROD_PORTS; do
    line=$(echo "$PS_DATA" | grep -- "--port $port " | head -1)
    [ -z "$line" ] && continue
    pid=$(echo "$line" | awk '{print $1}')
    rss=$(echo "$line" | awk '{print $2}')
    path=$(echo "$line" | grep -oE -- '--model [^ ]+' | head -1 | awk '{print $2}')
    ctx_raw=$(echo "$line" | grep -oE -- '--ctx-size [0-9]+' | awk '{print $2}')
    running_data+="$port	$pid	$rss	$path	$ctx_raw"$'\n'
done

# --- Step 2: one Python call → CSV "port,size_label,quant" lines ---
metadata_csv=""
META_HELPER="$REPO_ROOT/scripts/_gguf_meta.py"
if [ -n "$running_data" ] && [ -x "$VENV_PY" ] && [ -f "$META_HELPER" ]; then
    args=()
    while IFS=$'\t' read -r port pid rss path ctx_raw; do
        [ -n "$port" ] && [ -n "$path" ] && args+=("$port=$path")
    done <<< "$running_data"

    if [ ${#args[@]} -gt 0 ]; then
        metadata_csv=$(PYTHONPATH="$GGUF_PY" "$VENV_PY" "$META_HELPER" "${args[@]}" 2>/dev/null)
    fi
fi

# Helper: lookup port_csv field — returns size_label/quant/trained_ctx from metadata_csv
lookup_meta() {
    # $1 = port, $2 = field index (2 = size_label, 3 = quant, 4 = trained_ctx)
    echo "$metadata_csv" | awk -F, -v port="$1" -v idx="$2" '$1==port {print $idx; exit}'
}

# Format trained-context int to human: 8192 → "8K", 131072 → "128K", 262144 → "256K"
format_ctx_human() {
    local n="$1"
    [ -z "$n" ] || [ "$n" = "?" ] && { echo "?"; return; }
    if [ "$n" -ge 1024 ]; then
        echo "$((n / 1024))K"
    else
        echo "$n"
    fi
}

# --- Step 3: print the table ---
echo ""
printf "  %-7s %-13s %-6s %-34s %-7s %-7s %-5s %-5s %-22s %8s\n" \
    "PID" "PROCESS" "PORT" "MODEL" "PARAMS" "QUANT" "CTX" "MAX" "ROLE" "RAM"
printf "  %-7s %-13s %-6s %-34s %-7s %-7s %-5s %-5s %-22s %8s\n" \
    "-------" "-------------" "------" "----------------------------------" \
    "-------" "-------" "-----" "-----" "----------------------" "--------"

total_mb=0
total_params_b=0
count=0

while IFS=$'\t' read -r port pid rss path ctx_raw; do
    [ -z "$port" ] && continue

    process="llama-server"
    model=$(basename "$path" | sed 's/\.gguf$//')
    ram_mb=$((rss / 1024))

    # CTX: 8192 → "8K"
    if [ -n "$ctx_raw" ] && [ "$ctx_raw" -ge 1024 ]; then
        ctx="$((ctx_raw / 1024))K"
    else
        ctx="${ctx_raw:--}"
    fi

    # PARAMS + QUANT + MAX (trained ctx): dynamic from GGUF metadata, fallback to hardcoded
    params=$(lookup_meta "$port" 2)
    quant=$(lookup_meta "$port" 3)
    trained_ctx_raw=$(lookup_meta "$port" 4)
    case "$params" in ""|"?") params=$(fallback "$port" params) ;; esac
    case "$quant"  in ""|"?") quant=$(fallback "$port" quant)   ;; esac
    max_ctx=$(format_ctx_human "$trained_ctx_raw")

    role=$(fallback "$port" role)

    printf "  %-7s %-13s %-6s %-34s %-7s %-7s %-5s %-5s %-22s %5s MB\n" \
        "$pid" "$process" "$port" "$model" "$params" "$quant" "$ctx" "$max_ctx" "$role" "$ram_mb"

    total_mb=$((total_mb + ram_mb))

    # Sum params (parse "1000M", "4.2B", "303M", "891M", "1 B", "0.9 B" etc.)
    p_clean=$(echo "$params" | tr -d ' ')
    bil=0
    if [[ "$p_clean" =~ ^([0-9]+\.?[0-9]*)B$ ]]; then
        bil="${BASH_REMATCH[1]}"
    elif [[ "$p_clean" =~ ^([0-9]+\.?[0-9]*)M$ ]]; then
        bil=$(awk "BEGIN{printf \"%.3f\", ${BASH_REMATCH[1]}/1000}")
    fi
    total_params_b=$(awk "BEGIN{printf \"%.3f\", $total_params_b + $bil}")
    count=$((count + 1))
done <<< "$running_data"

echo ""
if [ "$count" -eq 0 ]; then
    echo "  ⚠  No llama-server processes found on production ports (9093-9098)."
    echo "     Run ./scripts/start_app.sh first."
else
    total_b_display=$(awk "BEGIN{printf \"%.1f\", $total_params_b}")
    echo "  $count models · ~${total_b_display} B parameters · ~$((total_mb / 1024)) GB RAM"
fi
echo ""
