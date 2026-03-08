#!/bin/bash

# sync-from-yzma.sh
# Sync llama, loader, message, mtmd, template, utils packages from yzma upstream.
# The script creates a new branch, commits sync results, pushes, and opens a PR.
#
# Usage: ./scripts/sync-from-yzma.sh [yzma-path] [branch-name]
# Example: ./scripts/sync-from-yzma.sh ../yzma sync/yzma-20260308

set -euo pipefail

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found: $cmd"
        exit 1
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMALIB_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_YZMA_PATH="$LLAMALIB_ROOT/../yzma"
YZMA_PATH_INPUT="${1:-$DEFAULT_YZMA_PATH}"
YZMA_PATH="$(cd "$YZMA_PATH_INPUT" 2>/dev/null && pwd || true)"

if [ -z "$YZMA_PATH" ] || [ ! -d "$YZMA_PATH" ]; then
    echo "Error: yzma path not found: $YZMA_PATH_INPUT"
    echo "Usage: $0 [yzma-path] [branch-name]"
    echo "Example: $0 ../yzma sync/yzma-20260308"
    exit 1
fi

require_cmd git
require_cmd rsync
require_cmd sed
require_cmd gh

cd "$LLAMALIB_ROOT"

if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working tree is not clean in $LLAMALIB_ROOT"
    echo "Please commit/stash current changes before running this script."
    exit 1
fi

BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
YZMA_BRANCH="$(git -C "$YZMA_PATH" rev-parse --abbrev-ref HEAD)"
if [ -n "$(git -C "$YZMA_PATH" status --porcelain)" ]; then
    echo "Error: yzma working tree is not clean in $YZMA_PATH"
    echo "Please commit/stash yzma changes before running this script."
    exit 1
fi
echo "=== Updating yzma ==="
echo "yzma branch:  $YZMA_BRANCH"
git -C "$YZMA_PATH" fetch origin
git -C "$YZMA_PATH" pull --ff-only origin "$YZMA_BRANCH"
echo "  ✓ yzma updated"
echo ""

YZMA_COMMIT="$(git -C "$YZMA_PATH" rev-parse HEAD)"
YZMA_SHORT_COMMIT="$(git -C "$YZMA_PATH" rev-parse --short HEAD)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BRANCH_NAME="${2:-sync/yzma-$YZMA_SHORT_COMMIT-$TIMESTAMP}"

echo "=== Sync from yzma ==="
echo "llamalib:     $LLAMALIB_ROOT"
echo "yzma:         $YZMA_PATH"
echo "base branch:  $BASE_BRANCH"
echo "new branch:   $BRANCH_NAME"
echo "yzma commit:  $YZMA_COMMIT"
echo ""

git checkout -b "$BRANCH_NAME"

# Define packages to sync
SYNC_PACKAGES="llama loader message mtmd template utils"

# Sync each package
for pkg in $SYNC_PACKAGES; do
    echo "Syncing: pkg/$pkg -> $pkg"

    # Copy files from yzma to llamalib
    # Exclude: go.mod, go.sum, __pycache__, .git
    rsync -av --delete \
        --exclude='go.mod' \
        --exclude='go.sum' \
        --exclude='__pycache__' \
        --exclude='.git' \
        "$YZMA_PATH/pkg/$pkg/" \
        "$LLAMALIB_ROOT/$pkg/"

    echo "  ✓ Copied files"
done

echo ""
echo "=== Replacing import paths ==="

# Replace import paths in Go files
for pkg in $SYNC_PACKAGES; do
    echo "Processing: $pkg"

    while IFS= read -r -d '' file; do
        # Replace yzma pkg imports with llamalib root imports
        # github.com/hybridgroup/yzma/pkg/xxx -> github.com/getkawai/llamalib/xxx
        sed -i.bak 's|github.com/hybridgroup/yzma/pkg/|github.com/getkawai/llamalib/|g' "$file"
        rm -f "$file.bak"
    done < <(find "$LLAMALIB_ROOT/$pkg" -type f -name "*.go" -print0)

    echo "  ✓ Replaced imports"
done

if [ -z "$(git status --porcelain)" ]; then
    echo ""
    echo "No changes detected after sync."
    echo "Branch $BRANCH_NAME created with no file changes."
    exit 0
fi

COMMIT_MESSAGE="sync: update from yzma@$YZMA_COMMIT"
PR_TITLE="sync: update from yzma@$YZMA_SHORT_COMMIT"
PR_BODY=$(cat <<EOF
Sync packages from yzma commit \`$YZMA_COMMIT\`.

Synced packages:
- pkg/llama -> llama
- pkg/loader -> loader
- pkg/message -> message
- pkg/mtmd -> mtmd
- pkg/template -> template
- pkg/utils -> utils
EOF
)

git add llama loader message mtmd template utils
git commit -m "$COMMIT_MESSAGE"
git push -u origin "$BRANCH_NAME"

gh pr create \
    --base "$BASE_BRANCH" \
    --head "$BRANCH_NAME" \
    --title "$PR_TITLE" \
    --body "$PR_BODY"

echo ""
echo "=== Sync complete ==="
echo "Branch: $BRANCH_NAME"
echo "Commit: $COMMIT_MESSAGE"
echo "PR has been created via gh."
