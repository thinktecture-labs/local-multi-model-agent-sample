#!/usr/bin/env bash
# ============================================================
# upload_ft_to_hf.sh — Publish the fine-tuned GGUFs to HuggingFace.
#
# What this uploads (per scenario):
#   - models/gemma3-1b-ft-merged/gemma3-1b-ft-<scenario>-f16.gguf
#   - models/gemma3-4b-ft-merged/gemma3-4b-ft-<scenario>-f16.gguf     (legacy variant, kept for comparison)
#   - models/gemma3-4b-ft-merged/gemma3-4b-ft-<scenario>-q4_k_m.gguf  (production synthesis artifact)
#   - models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-<scenario>-q4_k_m.gguf
#   - models/embeddinggemma-300m-ft-merged/embeddinggemma-300m-ft-<scenario>-q8_0.gguf
#   - models/intent-logreg/{model.joblib,meta.json}
#
# Each goes to a separate HF repo named:
#   ${HF_NAMESPACE}/<model-base>-ft-<scenario>
#
# Optional env:
#   HF_NAMESPACE      HuggingFace org/user (default: thinktecture)
#   SCENARIO          Scenario name (default: nextera)
#   CREATE_COLLECTION If "1", also create a HF Collection grouping the
#                     uploaded models (recommended for a clean public surface)
#   COLLECTION_TITLE  Collection title (default: derived from scenario)
#   DRY_RUN           If "1", print what would be uploaded but don't push
#
# Required: HF_TOKEN via `hf auth login` OR HF_TOKEN env var
#           (the account must be a member of HF_NAMESPACE)
#
# Prereqs (one-time, on the account you upload from):
#   1. Accept Gemma Terms at https://huggingface.co/google/gemma-3-1b-it
#      (and the 4b + embeddinggemma pages) — this is per-account, can't be
#      automated.
#   2. `pip install -U huggingface_hub` (>=0.32 ships the `hf` CLI)
#   3. `hf auth login` (or export HF_TOKEN=hf_...)
#
# Each uploaded model card MUST include:
#   - Base model attribution + link
#   - License (Gemma Terms / Tongyi Qianwen — see ../MODEL_LICENSES.md)
#   - Intended use (see ../MODEL_CARDS.md for templates)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${BOLD}${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}⚠${RESET}  $*"; }
error() { echo -e "${BOLD}${RED}✗${RESET} $*"; exit 1; }

# ─── Config ──────────────────────────────────────────────────────────────────
HF_NAMESPACE="${HF_NAMESPACE:-thinktecture}"
SCENARIO="${SCENARIO:-nextera}"
DRY_RUN="${DRY_RUN:-0}"
CREATE_COLLECTION="${CREATE_COLLECTION:-1}"
COLLECTION_TITLE="${COLLECTION_TITLE:-Local Multi-Model Agent — ${SCENARIO} fine-tunes}"

# ─── Resolve Python + hf CLI ────────────────────────────────────────────────
# Prefer the venv's hf binary; fall back to system `hf`. The hf CLI ships with
# huggingface_hub >= 0.32 and replaces the deprecated huggingface-cli.
if [ -f .venv/bin/python ] && .venv/bin/python -c "import huggingface_hub" 2>/dev/null; then
    PYTHON=".venv/bin/python"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    PYTHON="python3"
else
    error "huggingface_hub not installed — run: .venv/bin/pip install -U huggingface_hub"
fi
HF_BIN="$(dirname "$PYTHON")/hf"
if [ ! -x "$HF_BIN" ]; then
    if command -v hf >/dev/null 2>&1; then
        HF_BIN="$(command -v hf)"
    else
        error "hf CLI not found — upgrade with: $PYTHON -m pip install -U huggingface_hub"
    fi
fi
info "Using $PYTHON + $HF_BIN"

# ─── Per-model upload table ────────────────────────────────────────────────
# Format: "<local-path> <hf-repo-suffix> <model-card-anchor>"
UPLOADS=(
    "models/gemma3-1b-ft-merged/gemma3-1b-ft-${SCENARIO}-f16.gguf|gemma3-1b-ft-${SCENARIO}-f16|Gemma3-1B-FT"
    "models/gemma3-4b-ft-merged/gemma3-4b-ft-${SCENARIO}-f16.gguf|gemma3-4b-ft-${SCENARIO}-f16|Gemma3-4B-FT"
    "models/gemma3-4b-ft-merged/gemma3-4b-ft-${SCENARIO}-q4_k_m.gguf|gemma3-4b-ft-${SCENARIO}-q4_k_m|Gemma3-4B-FT"
    "models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-${SCENARIO}-q4_k_m.gguf|qwen3.5-4b-toolcalling-ft-${SCENARIO}-q4_k_m|Qwen3.5-4B-FT"
    "models/embeddinggemma-300m-ft-merged/embeddinggemma-300m-ft-${SCENARIO}-q8_0.gguf|embeddinggemma-300m-ft-${SCENARIO}-q8_0|EmbeddingGemma-FT"
    "models/intent-logreg|intent-logreg-${SCENARIO}|LogReg"
)

# ─── Upload loop ────────────────────────────────────────────────────────────
echo ""
info "Uploading fine-tuned models to HF namespace: ${HF_NAMESPACE}"
echo "  Scenario:  ${SCENARIO}"
echo "  Dry run:   ${DRY_RUN}"
echo ""

for entry in "${UPLOADS[@]}"; do
    IFS='|' read -r local_path hf_suffix card_anchor <<< "$entry"
    repo_id="${HF_NAMESPACE}/${hf_suffix}"

    if [ ! -e "$local_path" ]; then
        warn "Skip — not found locally: $local_path"
        continue
    fi

    echo -e "${BOLD}→${RESET} $local_path"
    echo "  → ${repo_id}"
    echo "  → see finetune/MODEL_CARDS.md#${card_anchor} for the card to attach"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN — skipping actual upload]"
        echo ""
        continue
    fi

    # `hf upload <repo> <local> <path-in-repo>` — creates the repo if missing
    # (private by default; flip to public on the HF web UI once the model
    # card is in place).
    # For files: path_in_repo is the basename so the file lands at repo root.
    # For folders: `.` uploads the folder's contents to the repo root.
    if [ -d "$local_path" ]; then
        PATH_IN_REPO="."
    else
        PATH_IN_REPO="$(basename "$local_path")"
    fi
    "$HF_BIN" upload \
        --repo-type model \
        --commit-message "Upload ${SCENARIO} fine-tune" \
        --private \
        "$repo_id" \
        "$local_path" \
        "$PATH_IN_REPO"

    echo ""
done

# ─── Collection (optional) ──────────────────────────────────────────────────
if [ "$CREATE_COLLECTION" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    echo ""
    info "Creating HF Collection: ${COLLECTION_TITLE}"

    REPO_LIST=$(printf '"%s/%s",\n' \
        "$HF_NAMESPACE" "gemma3-1b-ft-${SCENARIO}-f16" \
        "$HF_NAMESPACE" "gemma3-4b-ft-${SCENARIO}-f16" \
        "$HF_NAMESPACE" "gemma3-4b-ft-${SCENARIO}-q4_k_m" \
        "$HF_NAMESPACE" "qwen3.5-4b-toolcalling-ft-${SCENARIO}-q4_k_m" \
        "$HF_NAMESPACE" "embeddinggemma-300m-ft-${SCENARIO}-q8_0" \
        "$HF_NAMESPACE" "intent-logreg-${SCENARIO}")

    "$PYTHON" - <<PYEOF
from huggingface_hub import create_collection, add_collection_item

repos = [
    "${HF_NAMESPACE}/gemma3-1b-ft-${SCENARIO}-f16",
    "${HF_NAMESPACE}/gemma3-4b-ft-${SCENARIO}-f16",
    "${HF_NAMESPACE}/gemma3-4b-ft-${SCENARIO}-q4_k_m",
    "${HF_NAMESPACE}/qwen3.5-4b-toolcalling-ft-${SCENARIO}-q4_k_m",
    "${HF_NAMESPACE}/embeddinggemma-300m-ft-${SCENARIO}-q8_0",
    "${HF_NAMESPACE}/intent-logreg-${SCENARIO}",
]

description = (
    "Fine-tuned model stack for the ${SCENARIO} reference scenario of "
    "thinktecture-labs/local-multi-model-agent-slm."
)  # HF caps collection descriptions at 150 chars.

try:
    collection = create_collection(
        title="${COLLECTION_TITLE}",
        namespace="${HF_NAMESPACE}",
        description=description,
        private=True,  # flip to public on the HF web UI when ready
        exists_ok=True,
    )
    print(f"  Collection slug: {collection.slug}")
    print(f"  URL: https://huggingface.co/collections/{collection.slug}")
    for repo in repos:
        try:
            add_collection_item(
                collection_slug=collection.slug,
                item_id=repo,
                item_type="model",
                exists_ok=True,
            )
            print(f"  + added: {repo}")
        except Exception as exc:
            print(f"  ! could not add {repo}: {exc}")
except Exception as exc:
    print(f"Collection creation failed: {exc}")
    print("(You can create one manually on the HF web UI later.)")
PYEOF
fi

# ─── Next steps ──────────────────────────────────────────────────────────────
echo ""
info "Upload complete. Next:"
echo "  1. On https://huggingface.co/${HF_NAMESPACE}/, flip each repo to public"
echo "     (start private to make sure you've attached the right model card)"
echo "  2. For each repo, paste the matching section from finetune/MODEL_CARDS.md"
echo "     into the repo's README.md (HF auto-renders it as the model card)"
echo "  3. Attach the appropriate LICENSE (Gemma Terms / Qwen License) per"
echo "     finetune/MODEL_LICENSES.md"
echo "  4. If CREATE_COLLECTION=1 was set: flip the collection to public too"
echo "  5. Test the download flow:"
echo "       HF_NAMESPACE=${HF_NAMESPACE} bash scripts/download_ft_models.sh"
