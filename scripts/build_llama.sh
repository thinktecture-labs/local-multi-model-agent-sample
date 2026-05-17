#!/usr/bin/env bash
# ============================================================
# build_llama.sh — Build llama-server from the vendored submodule
#
# Auto-detects the best available backend:
#   CUDA  (NVIDIA GPU — nvcc present)
#   Metal (Apple Silicon — macOS + arm64)
#   CPU   (fallback)
#
# Output: vendor/llama.cpp/build/bin/llama-server
#
# Usage:
#   bash scripts/build_llama.sh
# ============================================================
set -euo pipefail

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RESET="\033[0m"
info() { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn() { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }

SUBMODULE_DIR="vendor/llama.cpp"
BUILD_DIR="$SUBMODULE_DIR/build"
BINARY="$BUILD_DIR/bin/llama-server"
QUANTIZE_BIN="$BUILD_DIR/bin/llama-quantize"

# ─── Ensure nvcc visibility ──────────────────────────────────────────────────
# Non-interactive SSH sessions on Linux often have a stripped PATH that doesn't
# include /usr/local/cuda/bin. Without nvcc on PATH, the `command -v nvcc`
# detection below falls through to the CPU branch even on CUDA-equipped boxes.
# Prepend the standard CUDA bin dir if it exists.
if [ -x /usr/local/cuda/bin/nvcc ] && ! command -v nvcc >/dev/null 2>&1; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# ─── Submodule ────────────────────────────────────────────────────────────────

if [ ! -f "$SUBMODULE_DIR/CMakeLists.txt" ]; then
    echo "Initialising llama.cpp submodule…"
    git submodule update --init --depth=1 vendor/llama.cpp
fi

# ─── Detect backend ───────────────────────────────────────────────────────────
# Use a bash array for CMAKE_ARGS instead of a space-separated string. Earlier
# the aarch64+CUDA branch produced `-DCMAKE_C_FLAGS='-O3 -march=… -mtune=native'`
# in a plain string; when bash word-split that variable into argv, the literal
# single quotes survived into cmake and it rejected `-mtune=native'` as an
# unknown argument. An array preserves each `-DCMAKE_*_FLAGS=…` as a single
# argv element with the embedded spaces intact.

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON)
BACKEND="CPU"

if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    info "NVIDIA CUDA detected ($CUDA_VER) — building with CUDA GPU support"
    CMAKE_ARGS+=(-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA_ALL_QUANTS=ON)
    BACKEND="CUDA"

    if [[ "$(uname -m)" == "aarch64" ]]; then
        # DGX Spark / Grace Blackwell (aarch64 + CUDA, sm_121):
        # - ARM-side: cmake's ARM feature detection fails without explicit -march
        #   flags. The Grace CPU supports ARMv9 with SVE2, i8mm, bf16, and dotprod
        #   — pass them explicitly so ggml-cpu picks up the optimized kernels
        #   instead of falling back to generic NEON.
        # - GPU-side: GGML_CUDA_F16 enables FP16 dequant + matmul paths (default
        #   falls back to FP32 ~2x slower for our F16 4B models). Explicit
        #   121a-real targets GB10 (sm_121) — drops portability bloat and
        #   generates GB10-tuned PTX. Use 121a-real, NOT 120a-real (which broke
        #   after llama.cpp PR #17906; see issue #18398). Forum-vetted on
        #   forums.developer.nvidia.com #363164 and Arm Learning Paths
        #   llama.cpp-on-GB10 guide.
        info "ARM64 + CUDA detected (DGX Spark / Grace) — adding SVE2+i8mm+bf16 + CUDA_F16 + sm_121"
        ARM_FLAGS="-O3 -march=armv9-a+sve2+bf16+i8mm+dotprod -mtune=native"
        CMAKE_ARGS+=(
            -DGGML_CUDA_F16=ON
            -DCMAKE_CUDA_ARCHITECTURES=121a-real
            -DCMAKE_C_FLAGS="$ARM_FLAGS"
            -DCMAKE_CXX_FLAGS="$ARM_FLAGS"
        )
    else
        # RTX PRO 6000 / desktop Blackwell (x86_64 + CUDA, sm_120): use FP16 in
        # CUDA dequant kernels (strong FP16 tensor core throughput on Blackwell).
        # Explicit 120a-real avoids MXFP4 "mma with block scale" compilation failure
        # that occurs with bare sm_120 (github.com/ggml-org/llama.cpp/issues/19662).
        info "x86_64 + CUDA detected (RTX Blackwell) — adding CUDA_F16 + explicit arch"
        CMAKE_ARGS+=(-DGGML_CUDA_F16=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real)
    fi
elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    # Apple Silicon (Metal): embed compiled Metal shaders into the binary to
    # eliminate first-run shader compilation latency (several seconds otherwise).
    info "Apple Silicon detected — building with Metal GPU support + embedded shaders"
    CMAKE_ARGS+=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON)
    BACKEND="Metal"
elif pkg-config --exists vulkan 2>/dev/null && command -v glslc &>/dev/null; then
    # Strix Halo / AMD RDNA 3.5 (Linux, no CUDA, Vulkan SDK present): use Vulkan
    # via Mesa/RADV. The host build outperforms the kyuz0 ROCm toolboxes on our
    # Gemma3-4B and Qwen3.5-4B Q4_K_M models (validated 2026-05-17). Vulkan
    # requires the SDK (libvulkan-dev / vulkan-headers) plus glslc from
    # shaderc-tools for shader compilation. -DGGML_NATIVE=ON picks up the host
    # x86_64 feature flags (AVX-512 etc.) for the CPU-side ggml-cpu fallback.
    info "Vulkan SDK detected (no CUDA) — building with Vulkan GPU support"
    CMAKE_ARGS+=(-DGGML_VULKAN=ON -DGGML_NATIVE=ON)
    BACKEND="Vulkan"
else
    warn "No GPU backend detected — building CPU-only llama-server"
    warn "For NVIDIA: install CUDA toolkit. For Apple Silicon: use macOS arm64."
    warn "For AMD/Intel iGPU: install vulkan-headers + shaderc-tools."
fi

info "Backend: $BACKEND"

# ─── Build ────────────────────────────────────────────────────────────────────
# Earlier versions of this script piped `cmake --build` directly through
# `tail -5`. On long compiles (DGX Spark aarch64+CUDA, Strix Halo Vulkan), the
# tail process closes its read end once it has 5 lines, and the next cmake
# stdout write triggers SIGPIPE — which on non-interactive SSH sessions kills
# the entire build before linking. We now write the full build log to a file
# and tail it afterwards; the failure mode becomes "non-zero exit code from
# cmake --build", not a silent SIGPIPE truncation.

BUILD_LOG="$(mktemp -t llama-build.XXXXXX.log)"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo "Configuring cmake…"
if ! cmake -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" "$SUBMODULE_DIR" >"$BUILD_LOG" 2>&1; then
    echo
    echo "--- cmake configure failed; tail of $BUILD_LOG ---"
    tail -40 "$BUILD_LOG"
    exit 1
fi
grep -E "^-- (ggml|CUDA|Metal|Build)" "$BUILD_LOG" | head -10 || true

echo "Compiling llama-server (using $JOBS cores; log: $BUILD_LOG)…"
if ! cmake --build "$BUILD_DIR" --target llama-server -j"$JOBS" >>"$BUILD_LOG" 2>&1; then
    echo
    echo "--- llama-server compile failed; errors + log tail ---"
    grep -E "error:|FAILED|fatal" "$BUILD_LOG" | head -20
    echo "--- (full log tail) ---"
    tail -30 "$BUILD_LOG"
    echo
    echo "Build failed — full log preserved at: $BUILD_LOG"
    exit 1
fi
tail -5 "$BUILD_LOG"

# llama-quantize is needed by finetune/convert_gemma3_4b_to_gguf.sh to produce
# the production Q4_K_M synthesis artifact alongside the intermediate F16.
echo "Compiling llama-quantize (using $JOBS cores)…"
if ! cmake --build "$BUILD_DIR" --target llama-quantize -j"$JOBS" >>"$BUILD_LOG" 2>&1; then
    echo
    echo "--- llama-quantize compile failed ---"
    grep -E "error:|FAILED|fatal" "$BUILD_LOG" | tail -20
    tail -30 "$BUILD_LOG"
    exit 1
fi
tail -5 "$BUILD_LOG"

info "Built: $BINARY ($BACKEND)"
info "Built: $QUANTIZE_BIN"
"$BINARY" --version
