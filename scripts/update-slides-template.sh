#!/usr/bin/env bash
#
# update-slides-template.sh — pull latest from upstream Slidev TT template
# into slides-v2/ without clobbering files we own.
#
# Files we own are listed in slides-v2/.template-sync-exclude
#
# Usage:
#   bash scripts/update-slides-template.sh
#
# What it does:
#   1. Fetches latest template into a /tmp staging clone
#   2. rsync into slides-v2/, excluding files we own
#   3. Shows you the diff for files that changed
#   4. Reminds you to re-run npm install if package.json changed
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLIDES_DIR="$REPO_ROOT/slides-v2"
TEMPLATE_URL="https://github.com/thinktecture-labs/slidev-tt-template.git"
STAGING="/tmp/slidev-tt-template-update-$(date +%s)"

if [[ ! -d "$SLIDES_DIR" ]]; then
  echo "ERROR: $SLIDES_DIR does not exist." >&2
  exit 1
fi

EXCLUDE_FILE="$SLIDES_DIR/.template-sync-exclude"
if [[ ! -f "$EXCLUDE_FILE" ]]; then
  echo "ERROR: $EXCLUDE_FILE missing. Cannot tell which files to preserve." >&2
  exit 1
fi

echo "→ Cloning latest template into $STAGING"
git clone --depth 1 "$TEMPLATE_URL" "$STAGING" 2>&1 | tail -3

cd "$STAGING"
LATEST_COMMIT=$(git log -1 --pretty=format:'%h %s (%ad)' --date=short)
echo "→ Latest upstream commit: $LATEST_COMMIT"

# Snapshot package.json before, to detect if deps changed.
PKG_BEFORE=$(sha1sum "$SLIDES_DIR/package.json" 2>/dev/null | cut -d' ' -f1 || echo "")

echo "→ Syncing into $SLIDES_DIR (preserving files in .template-sync-exclude)"
rsync -a --delete-excluded \
  --exclude-from="$EXCLUDE_FILE" \
  "$STAGING/" "$SLIDES_DIR/"

PKG_AFTER=$(sha1sum "$SLIDES_DIR/package.json" 2>/dev/null | cut -d' ' -f1 || echo "")

echo "✓ Sync complete."
echo ""

if [[ "$PKG_BEFORE" != "$PKG_AFTER" ]]; then
  echo "⚠ package.json changed — run:"
  echo "    cd slides-v2 && npm install"
fi

echo ""
echo "→ Files we preserved (your customizations):"
grep -v '^\s*$\|^\s*#' "$EXCLUDE_FILE" | sed 's/^/    /'

echo ""
echo "→ Cleaning up staging clone"
rm -rf "$STAGING"

echo ""
echo "Done. Review changes with: cd slides-v2 && git status"
