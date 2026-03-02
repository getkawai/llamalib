#!/bin/bash

# sync-from-yzma.sh
# Sync llama, loader, message, mtmd, template, utils packages from yzma upstream
#
# Usage: ./scripts/sync-from-yzma.sh [yzma-path]
# Example: ./scripts/sync-from-yzma.sh ../yzma

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMALIB_ROOT="$(dirname "$SCRIPT_DIR")"

# Default yzma path
YZMA_PATH="${1:-../yzma}"

if [ ! -d "$YZMA_PATH" ]; then
    echo "Error: yzma path not found: $YZMA_PATH"
    echo "Usage: $0 [yzma-path]"
    echo "Example: $0 ../yzma"
    exit 1
fi

echo "=== Sync from yzma ==="
echo "llamalib: $LLAMALIB_ROOT"
echo "yzma:     $YZMA_PATH"
echo ""

# Get yzma commit hash for reference
YZMA_COMMIT=$(cd "$YZMA_PATH" && git rev-parse HEAD)
echo "yzma commit: $YZMA_COMMIT"
echo ""

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
    
    # Find all .go files and replace imports
    find "$LLAMALIB_ROOT/$pkg" -name "*.go" -type f | while read -r file; do
        # Replace yzma pkg imports with llamalib root imports
        # github.com/hybridgroup/yzma/pkg/xxx -> github.com/getkawai/llamalib/xxx
        sed -i.bak 's|github.com/hybridgroup/yzma/pkg/|github.com/getkawai/llamalib/|g' "$file"
        
        # Remove backup file
        rm -f "$file.bak"
    done
    
    echo "  ✓ Replaced imports"
done

echo ""
echo "=== Sync complete ==="
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Run tests: go test ./..."
echo "3. Commit with message:"
echo "   git add llama loader message mtmd template utils"
echo "   git commit -m \"sync: update from yzma@$YZMA_COMMIT\""
echo ""
echo "Packages synced:"
for pkg in $SYNC_PACKAGES; do
    echo "  - pkg/$pkg -> $pkg"
done
