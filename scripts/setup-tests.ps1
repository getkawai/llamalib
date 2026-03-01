# Test setup script for llamalib CI (Windows)
# This script replaces the yzma CLI functionality for CI workflows

param(
    [string]$LibDir = "$env:GITHUB_WORKSPACE\lib",
    [string]$ModelsDir = "$env:GITHUB_WORKSPACE\models",
    [string]$LlamaVersion = "latest"
)

Write-Host "🔧 Setting up llamalib test environment..."
Write-Host "   Library directory: $LibDir"
Write-Host "   Models directory: $ModelsDir"
Write-Host "   llama.cpp version: $LlamaVersion"

# Create directories
New-Item -ItemType Directory -Force -Path $LibDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host ""
Write-Host "📦 Platform: Windows"

# Get latest version if not specified
if ($LlamaVersion -eq "latest") {
    $response = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    $LlamaVersion = $response.tag_name
    Write-Host "   Latest version: $LlamaVersion"
}

# Download and extract llama.cpp (new format: llama-VERSION-bin-win-cpu-x64.zip)
Write-Host ""
Write-Host "📥 Downloading llama.cpp binaries..."

$FileName = "llama-$LlamaVersion-bin-win-cpu-x64.zip"
$UrlBase = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaVersion/$FileName"
$TempZip = "$env:TEMP\llama-cpp.zip"

Write-Host "   Downloading: $FileName"

Invoke-WebRequest -Uri $UrlBase -OutFile $TempZip
Expand-Archive -Path $TempZip -DestinationPath $LibDir -Force

# Verify library exists
$LibName = "llama.dll"
if (!(Test-Path "$LibDir\$LibName")) {
    # Try to find and move the library
    $found = Get-ChildItem -Recurse -Filter $LibName -Path $LibDir | Select-Object -First 1
    if ($found) {
        Move-Item -Path $found.FullName -Destination "$LibDir\$LibName" -Force
    }
}

Write-Host "✅ llama.cpp binaries installed"

# Download test models
Write-Host ""
Write-Host "📥 Downloading test models..."

function Download-Model {
    param(
        [string]$Url,
        [string]$Output
    )
    
    $filename = Split-Path -Path $Output -Leaf
    
    if (Test-Path $Output) {
        Write-Host "   ✓ $filename already exists"
        return
    }
    
    Write-Host "   Downloading $filename..."
    Invoke-WebRequest -Uri $Url -OutFile $Output
}

# SmolLM - main chat model
Download-Model `
    -Url "https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q2_K.gguf" `
    -Output "$ModelsDir\SmolLM-135M.Q2_K.gguf"

# SmolVLM - vision-language model
Download-Model `
    -Url "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf" `
    -Output "$ModelsDir\SmolVLM-256M-Instruct-Q8_0.gguf"

Download-Model `
    -Url "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" `
    -Output "$ModelsDir\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

# Embedding model
Download-Model `
    -Url "https://huggingface.co/ggml-org/models-moved/resolve/main/jina-reranker-v1-tiny-en/ggml-model-f16.gguf" `
    -Output "$ModelsDir\ggml-model-f16.gguf"

# Encoder model
Download-Model `
    -Url "https://huggingface.co/callgg/t5-base-encoder-f32/resolve/main/t5base-encoder-q4_0.gguf" `
    -Output "$ModelsDir\t5base-encoder-q4_0.gguf"

# LoRA test models
Download-Model `
    -Url "https://huggingface.co/deadprogram/yzma-tests/resolve/main/Gemma2-Base-F32.gguf" `
    -Output "$ModelsDir\Gemma2-Base-F32.gguf"

Download-Model `
    -Url "https://huggingface.co/deadprogram/yzma-tests/resolve/main/Gemma2-Lora-F32-LoRA.gguf" `
    -Output "$ModelsDir\Gemma2-Lora-F32-LoRA.gguf"

# Split model test files
Download-Model `
    -Url "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf" `
    -Output "$ModelsDir\stories15M-q8_0-00001-of-00003.gguf"

Download-Model `
    -Url "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/split/stories15M-q8_0-00002-of-00003.gguf" `
    -Output "$ModelsDir\stories15M-q8_0-00002-of-00003.gguf"

Download-Model `
    -Url "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/split/stories15M-q8_0-00003-of-00003.gguf" `
    -Output "$ModelsDir\stories15M-q8_0-00003-of-00003.gguf"

Write-Host ""
Write-Host "✅ All test models downloaded"

# Output environment variables
Write-Host ""
Write-Host "📋 Set the following environment variables:"
Write-Host "   `$env:YZMA_LIB=$LibDir"
Write-Host "   `$env:YZMA_TEST_MODEL=$ModelsDir\SmolLM-135M.Q2_K.gguf"
Write-Host "   `$env:YZMA_TEST_MMMODEL=$ModelsDir\SmolVLM-256M-Instruct-Q8_0.gguf"
Write-Host "   `$env:YZMA_TEST_MMPROJ=$ModelsDir\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
Write-Host "   `$env:YZMA_TEST_QUANTIZE_MODEL=$ModelsDir\ggml-model-f16.gguf"
Write-Host "   `$env:YZMA_TEST_ENCODER_MODEL=$ModelsDir\t5base-encoder-q4_0.gguf"
Write-Host "   `$env:YZMA_TEST_LORA_MODEL=$ModelsDir\Gemma2-Base-F32.gguf"
Write-Host "   `$env:YZMA_TEST_LORA_ADAPTER=$ModelsDir\Gemma2-Lora-F32-LoRA.gguf"
Write-Host "   `$env:YZMA_TEST_SPLIT_MODELS=`"$ModelsDir\stories15M-q8_0-00001-of-00003.gguf,$ModelsDir\stories15M-q8_0-00002-of-00003.gguf,$ModelsDir\stories15M-q8_0-00003-of-00003.gguf`""
Write-Host ""
Write-Host "🎉 Setup complete!"
