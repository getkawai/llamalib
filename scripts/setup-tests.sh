#!/bin/bash
# Test setup script for llamalib CI
# This script replaces the yzma CLI functionality for CI workflows

set -e

LIB_DIR="${1:-$GITHUB_WORKSPACE/lib}"
MODELS_DIR="${2:-$GITHUB_WORKSPACE/models}"
LLAMA_VERSION="${3:-latest}"

echo "🔧 Setting up llamalib test environment..."
echo "   Library directory: $LIB_DIR"
echo "   Models directory: $MODELS_DIR"
echo "   llama.cpp version: $LLAMA_VERSION"

# Create directories
mkdir -p "$LIB_DIR"
mkdir -p "$MODELS_DIR"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo ""
echo "📦 Platform: $OS/$ARCH"

# Determine processor type for llama.cpp
PROCESSOR="cpu"
if [ "$OS" = "darwin" ]; then
    PROCESSOR="metal"
    echo "   Using Metal backend for macOS"
elif command -v nvidia-smi &> /dev/null; then
    PROCESSOR="cuda"
    echo "   Using CUDA backend (NVIDIA detected)"
elif command -v vulkaninfo &> /dev/null; then
    PROCESSOR="vulkan"
    echo "   Using Vulkan backend"
fi

# Create a simple Go program to install llama.cpp and download models
cat > /tmp/llamalib_setup.go << 'GOEOF'
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	llamalib "github.com/getkawai/llamalib"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: llamalib_setup <lib_dir> <models_dir> <processor>")
		os.Exit(1)
	}

	libDir := os.Args[1]
	modelsDir := os.Args[2]
	processor := os.Args[3]

	// Create installer
	installer := llamalib.NewLlamaCppInstaller()
	installer.BinaryPath = libDir
	installer.ModelsDir = modelsDir

	// Install llama.cpp
	log.Println("Installing llama.cpp...")
	if err := installer.InstallLlamaCpp(); err != nil {
		log.Fatalf("Failed to install llama.cpp: %v", err)
	}
	log.Println("✅ llama.cpp installed")

	// Download test models
	log.Println("Downloading test models...")

	// SmolLM for chat/inference tests
	chatModels := installer.GetAvailableChatModels()
	if len(chatModels) == 0 {
		log.Println("No chat models found, would download test model...")
		// Model download logic would go here
	}

	log.Println("✅ Setup complete")
}
GOEOF

# For now, use direct download approach
echo ""
echo "📥 Downloading llama.cpp binaries..."

# Get latest version if not specified
if [ "$LLAMA_VERSION" = "latest" ]; then
    LLAMA_VERSION=$(gh api repos/ggml-org/llama.cpp/releases/latest --jq '.tag_name')
    echo "   Latest version: $LLAMA_VERSION"
fi

# Build download URL based on platform (new format: llama-VERSION-bin-OS-ARCH.tar.gz)
case "$OS-$PROCESSOR" in
    linux-cpu)
        LIB_NAME="libllama.so"
        FILE_NAME="llama-$LLAMA_VERSION-bin-ubuntu-x64.tar.gz"
        ;;
    linux-cuda)
        LIB_NAME="libllama.so"
        FILE_NAME="llama-$LLAMA_VERSION-bin-ubuntu-cuda-x64.tar.gz"
        ;;
    darwin-metal)
        LIB_NAME="libllama.dylib"
        FILE_NAME="llama-$LLAMA_VERSION-bin-macos-arm64.tar.gz"
        ;;
    *)
        LIB_NAME="libllama.so"
        FILE_NAME="llama-$LLAMA_VERSION-bin-ubuntu-x64.tar.gz"
        ;;
esac

URL_BASE="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_VERSION/$FILE_NAME"

echo "   Downloading: $FILE_NAME"

# Download and extract llama.cpp
TEMP_ARCHIVE="/tmp/llama-cpp.tar.gz"
curl -L -o "$TEMP_ARCHIVE" "$URL_BASE"
tar -xzf "$TEMP_ARCHIVE" -C "$LIB_DIR"

# List extracted contents for debugging
echo "   Extracted contents:"
ls -la "$LIB_DIR"

# Find and move all required libraries to the root of LIB_DIR
# llama.cpp now uses libllama.so as the main library
for lib in libllama.so libggml.so libggml-base.so libggml-cpu.so; do
    if [ ! -f "$LIB_DIR/$lib" ]; then
        found=$(find "$LIB_DIR" -name "$lib" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            mv "$found" "$LIB_DIR/"
            echo "   Moved $lib to $LIB_DIR/"
        fi
    fi
done

# Verify main library exists
if [ ! -f "$LIB_DIR/libllama.so" ]; then
    echo "   ❌ Error: libllama.so not found"
    exit 1
fi

# Create symlinks for backward compatibility (llamalib expects ggml, ggml-base, llama)
# New llama.cpp releases combine all into libllama.so
for link in libggml.so libggml-base.so; do
    if [ ! -f "$LIB_DIR/$link" ] && [ ! -L "$LIB_DIR/$link" ]; then
        ln -sf libllama.so "$LIB_DIR/$link"
        echo "   Created symlink: $link -> libllama.so"
    fi
done

echo "✅ llama.cpp binaries installed"

# Download test models
echo ""
echo "📥 Downloading test models..."

download_model() {
    local url="$1"
    local output="$2"
    local filename=$(basename "$output")
    
    if [ -f "$output" ]; then
        echo "   ✓ $filename already exists"
        return
    fi
    
    echo "   Downloading $filename..."
    curl -L -o "$output" "$url"
}

# SmolLM - main chat model
download_model \
    "https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q2_K.gguf" \
    "$MODELS_DIR/SmolLM-135M.Q2_K.gguf"

# SmolVLM - vision-language model
download_model \
    "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf" \
    "$MODELS_DIR/SmolVLM-256M-Instruct-Q8_0.gguf"

download_model \
    "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" \
    "$MODELS_DIR/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

# Embedding model
download_model \
    "https://huggingface.co/ggml-org/models-moved/resolve/main/jina-reranker-v1-tiny-en/ggml-model-f16.gguf" \
    "$MODELS_DIR/ggml-model-f16.gguf"

# Encoder model
download_model \
    "https://huggingface.co/callgg/t5-base-encoder-f32/resolve/main/t5base-encoder-q4_0.gguf" \
    "$MODELS_DIR/t5base-encoder-q4_0.gguf"

# LoRA test models
download_model \
    "https://huggingface.co/deadprogram/yzma-tests/resolve/main/Gemma2-Base-F32.gguf" \
    "$MODELS_DIR/Gemma2-Base-F32.gguf"

download_model \
    "https://huggingface.co/deadprogram/yzma-tests/resolve/main/Gemma2-Lora-F32-LoRA.gguf" \
    "$MODELS_DIR/Gemma2-Lora-F32-LoRA.gguf"

# Split model test files
download_model \
    "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf" \
    "$MODELS_DIR/stories15M-q8_0-00001-of-00003.gguf"

download_model \
    "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/split/stories15M-q8_0-00002-of-00003.gguf" \
    "$MODELS_DIR/stories15M-q8_0-00002-of-00003.gguf"

download_model \
    "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/split/stories15M-q8_0-00003-of-00003.gguf" \
    "$MODELS_DIR/stories15M-q8_0-00003-of-00003.gguf"

echo ""
echo "✅ All test models downloaded"

# Output environment variables
echo ""
echo "📋 Set the following environment variables:"
echo "   export YZMA_LIB=$LIB_DIR"
echo "   export YZMA_TEST_MODEL=$MODELS_DIR/SmolLM-135M.Q2_K.gguf"
echo "   export YZMA_TEST_MMMODEL=$MODELS_DIR/SmolVLM-256M-Instruct-Q8_0.gguf"
echo "   export YZMA_TEST_MMPROJ=$MODELS_DIR/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
echo "   export YZMA_TEST_QUANTIZE_MODEL=$MODELS_DIR/ggml-model-f16.gguf"
echo "   export YZMA_TEST_ENCODER_MODEL=$MODELS_DIR/t5base-encoder-q4_0.gguf"
echo "   export YZMA_TEST_LORA_MODEL=$MODELS_DIR/Gemma2-Base-F32.gguf"
echo "   export YZMA_TEST_LORA_ADAPTER=$MODELS_DIR/Gemma2-Lora-F32-LoRA.gguf"
echo "   export YZMA_TEST_SPLIT_MODELS=\"$MODELS_DIR/stories15M-q8_0-00001-of-00003.gguf,$MODELS_DIR/stories15M-q8_0-00002-of-00003.gguf,$MODELS_DIR/stories15M-q8_0-00003-of-00003.gguf\""
echo ""
echo "🎉 Setup complete!"
