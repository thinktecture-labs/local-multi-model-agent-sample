#!/usr/bin/env bash
# scripts/publish-public.sh — sync the public mirror from private main.
#
# Produces a SINGLE orphan commit on the public repo and force-pushes it.
# This script is for the maintainer of the private source-of-truth repo
# (thinktecture-labs/local-multi-model-agent-slm). If you cloned from the
# public mirror, this script does nothing useful for you.
#
# Usage:
#   bash scripts/publish-public.sh                # interactive (prompts before push)
#   bash scripts/publish-public.sh --yes          # skip confirmation
#   bash scripts/publish-public.sh --dry-run      # build + commit locally, skip push

set -euo pipefail

PUBLIC_REMOTE="https://github.com/thinktecture-labs/local-multi-model-agent-sample.git"
PUBLIC_REPO="thinktecture-labs/local-multi-model-agent-sample"
SOURCE_BRANCH="main"
SUBMODULE_PATH="vendor/llama.cpp"
SUBMODULE_URL="https://github.com/ggerganov/llama.cpp"
TMP_DIR="/tmp/local-multi-model-agent-sample"

DRY_RUN=false
ASSUME_YES=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=true ;;
    --yes|-y)   ASSUME_YES=true ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *)          echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
info() { printf '%s==>%s %s\n' "$BOLD$GREEN" "$RESET" "$*"; }
warn() { printf '%s!!!%s %s\n' "$BOLD$YELLOW" "$RESET" "$*"; }
err()  { printf '%sxxx%s %s\n' "$BOLD$RED" "$RESET" "$*" >&2; exit 1; }

# ─── pre-flight ─────────────────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || err "Not in a git repo."
cd "$REPO_ROOT"

CURR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CURR_BRANCH" == "$SOURCE_BRANCH" ]] || \
  err "Must be on '$SOURCE_BRANCH' (currently '$CURR_BRANCH'). Run: git checkout $SOURCE_BRANCH"

git diff-index --quiet HEAD -- || err "Working tree has uncommitted tracked changes. Commit or stash first."

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git ls-remote origin "refs/heads/$SOURCE_BRANCH" 2>/dev/null | awk '{print $1}')"
if [[ -n "$REMOTE_HEAD" && "$LOCAL_HEAD" != "$REMOTE_HEAD" ]]; then
  err "Local '$SOURCE_BRANCH' ($LOCAL_HEAD) out of sync with origin ($REMOTE_HEAD). git pull first."
fi

command -v gh >/dev/null || err "gh CLI not installed."
gh auth status >/dev/null 2>&1 || err "gh CLI not authenticated. Run: gh auth login"

# Auto-discover the vendor pin from main's tree — keeps script in lockstep with
# whatever commit the private repo is currently pinning.
SUBMODULE_PIN="$(git ls-tree "$SOURCE_BRANCH" -- "$SUBMODULE_PATH" | awk '{print $3}')"
[[ -n "$SUBMODULE_PIN" ]] || err "Could not read $SUBMODULE_PATH pin from $SOURCE_BRANCH tree."

info "Source HEAD:                   $LOCAL_HEAD"
info "$SUBMODULE_PATH pinned at:    $SUBMODULE_PIN"
info "Target public repo:            $PUBLIC_REPO"

# ─── build clean tree ───────────────────────────────────────────────────────
info "Preparing build dir: $TMP_DIR"
rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

info "Exporting $SOURCE_BRANCH tree via git archive | tar..."
git archive --format=tar "$SOURCE_BRANCH" | tar -x -C "$TMP_DIR"

cd "$TMP_DIR"

# Exclusions from the public mirror:
# - slides/            slidev source (PDF in presentations/ is the public artefact)
# - docs/operations/   per-host upgrade + tuning runbooks (internal to the
#                      maintainer's machines: ASUS Ascent GX10, Minisforum
#                      MS-S1 MAX). Fork users on different hardware would
#                      mis-apply them.
info "Applying exclusions: slides/, docs/operations/"
rm -rf slides docs/operations

# tar materialises the submodule gitlink as an empty directory; submodule add
# refuses to create vendor/llama.cpp if anything is already there.
rm -rf "$SUBMODULE_PATH"

# ─── init + vendor submodule ────────────────────────────────────────────────
info "Initializing orphan public-mirror repo..."
git init -b "$SOURCE_BRANCH" -q

# The .gitmodules from the archive is correct content-wise, but we re-create
# it via submodule add so the gitlink + .gitmodules pair are written together.
rm -f .gitmodules

info "Adding $SUBMODULE_PATH submodule (clones llama.cpp; large download)..."
git submodule add -q "$SUBMODULE_URL" "$SUBMODULE_PATH"

info "Pinning $SUBMODULE_PATH to $SUBMODULE_PIN"
git -C "$SUBMODULE_PATH" checkout -q "$SUBMODULE_PIN"
git add "$SUBMODULE_PATH" .gitmodules

info "Staging all files..."
git add -A
STAGED_COUNT="$(git diff --cached --name-only | wc -l | tr -d ' ')"
info "Staged: $STAGED_COUNT files"

# ─── commit ─────────────────────────────────────────────────────────────────
DATE_ISO="$(date +%Y-%m-%d)"
git commit -q \
  -m "Initial public release: SDD 2026 London keynote companion repo" \
  -m "Multi-model local AI agent demo. Five small language models running locally (gemma3, qwen3.5, embeddinggemma, GLM-OCR, whisper/piper) collaborating on intent classification, multi-step planning, tool use, RAG, and voice round-trip." \
  -m "Architecture, training methodology, eval results, and SECURITY.md disclaimers are in the repo. See README.md to get started." \
  -m "Mirrors thinktecture-labs/local-multi-model-agent-slm at $LOCAL_HEAD ($DATE_ISO), with slides/ excluded. $SUBMODULE_PATH pinned at $SUBMODULE_PIN."

PUBLIC_SHA="$(git rev-parse HEAD)"
info "Local public commit: $PUBLIC_SHA"

# ─── push ───────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
  warn "DRY-RUN: skipping force-push. Built tree is at $TMP_DIR for inspection."
  exit 0
fi

if [[ "$ASSUME_YES" != "true" ]]; then
  echo
  warn "About to FORCE-PUSH to $PUBLIC_REPO main."
  warn "This OVERWRITES the entire public history with the new single orphan commit."
  read -r -p "Continue? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || err "Aborted."
fi

info "Force-pushing to $PUBLIC_REMOTE..."
git remote add origin "$PUBLIC_REMOTE"
git push -f -u origin "$SOURCE_BRANCH" 2>&1 | tail -5

# ─── verify ─────────────────────────────────────────────────────────────────
# Use gh api (not raw.githubusercontent.com) — the CDN caches for several
# minutes after a force-push and will lie about the new content.
info "Verifying public state via gh API..."
REMOTE_SHA="$(gh api "repos/$PUBLIC_REPO/commits/$SOURCE_BRANCH" --jq '.sha')"
LICENSE_SPDX="$(gh api "repos/$PUBLIC_REPO/license" --jq '.license.spdx_id' 2>/dev/null || echo 'unknown')"
VENDOR_SHA="$(gh api "repos/$PUBLIC_REPO/contents/$SUBMODULE_PATH" --jq '.sha' 2>/dev/null || echo 'unknown')"

echo
info "${BOLD}Publish complete.${RESET}"
echo "  Public commit:   $REMOTE_SHA"
echo "  Source HEAD:     $LOCAL_HEAD"
echo "  License (SPDX):  $LICENSE_SPDX"
echo "  Vendor pin:      $VENDOR_SHA"
echo "  URL:             https://github.com/$PUBLIC_REPO"

[[ "$LICENSE_SPDX" == "MIT" ]] || \
  warn "License is '$LICENSE_SPDX' (expected MIT). Likely licensee re-scan lag; re-check in ~5 min."
[[ "$VENDOR_SHA" == "$SUBMODULE_PIN" ]] || \
  warn "Vendor pin mismatch on remote — expected $SUBMODULE_PIN, got $VENDOR_SHA."
