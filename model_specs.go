package llamalib

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

// Constants for GGUF file validation
const (
	ggufMagicNumber     = "GGUF"
	ggufMagicNumberSize = 4
	ggufVersionSize     = 4
	ggufTensorCountSize = 8
	ggufMetadataSize    = 8
	maxGGUFVersion      = 10
	maxTensorCount      = 100000
	maxMetadataCount    = 10000
)

// QwenModelSpec represents a Qwen model specification for direct download
type QwenModelSpec struct {
	Name         string // Model name (e.g., "qwen2.5-0.5b-instruct-q4_k_m")
	URL          string // Direct download URL
	Quantization string // Quantization type (Q4_K_M, Q5_K_M, etc.)
	Parameters   string // Parameter size (0.5b, 1.5b, etc.)
	MinRAM       int64  // Minimum RAM required in GB
	SHA256       string // Expected SHA256 checksum (optional)
	Description  string // Model description
	ProjectorURL string // URL for the multimodal projector file (for VL models)
}

// Note: Service struct has been removed. All functions are now pure functions
// that accept modelsDir as a parameter or use GetModelsDirectory() directly.

// GetRecommendedVLModels returns recommended Qwen models for direct download
// Models are ordered from smallest to largest by MinRAM requirement
func GetRecommendedVLModels() []QwenModelSpec {
	return []QwenModelSpec{
		{
			Name:         "qwen3-vl-4b-instruct-q4_k_m",
			URL:          "https://huggingface.co/bartowski/Qwen_Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
			Quantization: "Q4_K_M",
			Parameters:   "4b",
			MinRAM:       6,
			Description:  "Qwen3-VL 4B - Multimodal vision-language model, perfect for low-end hardware",
			ProjectorURL: "https://huggingface.co/bartowski/Qwen_Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf",
		},
		{
			Name:         "qwen3-vl-8b-instruct-q4_k_m",
			URL:          "https://huggingface.co/bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen_Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
			Quantization: "Q4_K_M",
			Parameters:   "8b",
			MinRAM:       12,
			Description:  "Qwen3-VL 8B - Advanced multimodal model with excellent vision capabilities",
			ProjectorURL: "https://huggingface.co/bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen_Qwen3-VL-8B-Instruct-f16.gguf",
		},
		{
			Name:         "qwen3-vl-32b-instruct-q4_k_m",
			URL:          "https://huggingface.co/bartowski/Qwen_Qwen3-VL-32B-Instruct-GGUF/resolve/main/Qwen_Qwen3-VL-32B-Instruct-Q4_K_M.gguf",
			Quantization: "Q4_K_M",
			Parameters:   "32b",
			MinRAM:       24,
			Description:  "Qwen3-VL 32B - High-quality multimodal model for advanced vision-language tasks",
			ProjectorURL: "https://huggingface.co/bartowski/Qwen_Qwen3-VL-32B-Instruct-GGUF/resolve/main/mmproj-Qwen_Qwen3-VL-32B-Instruct-f16.gguf",
		},
	}
}

// ============================================================================
// Function Calling Models (FunctionGemma & Nemotron-3-Nano)
// ============================================================================

// GetRecommendedFunctionCallingModels returns recommended models for function/tool calling
// Models are ordered from smallest to largest by MinRAM requirement
// Selection logic:
//   - RAM >= 24GB: Use Nemotron-3-Nano-30B (MoE, 30B params, 3B active) - Best overall
//   - RAM < 24GB: Use FunctionGemma 270M - Tiny but specialized for function calling
//
// References:
//   - FunctionGemma: https://docs.unsloth.ai/models/functiongemma
//   - Nemotron-3-Nano: https://docs.unsloth.ai/models/nemotron-3
func GetRecommendedFunctionCallingModels() []QwenModelSpec {
	return []QwenModelSpec{
		// FunctionGemma 270M - Ultra-small model specialized for function calling
		// Based on Gemma 3 270M, trained specifically for tool/function calling
		// Can run on ~550MB RAM, perfect for low-end systems
		// Recommended settings: top_k=64, top_p=0.95, temperature=1.0, ctx=32768
		{
			Name:         "functiongemma-270m-it-bf16",
			URL:          "https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-BF16.gguf",
			Quantization: "BF16",
			Parameters:   "270m",
			MinRAM:       1, // ~550MB, runs on almost any system
			Description:  "FunctionGemma 270M - Ultra-small model specialized for function/tool calling",
		},
		// FunctionGemma 270M Q8_0 - Slightly smaller quantized version
		{
			Name:         "functiongemma-270m-it-q8_0",
			URL:          "https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-Q8_0.gguf",
			Quantization: "Q8_0",
			Parameters:   "270m",
			MinRAM:       1, // ~300MB
			Description:  "FunctionGemma 270M Q8 - Quantized version for even lower RAM",
		},
		// Nemotron-3-Nano-30B-A3B - NVIDIA's flagship small model
		// 30B total parameters, 3B active (MoE architecture)
		// Trained from scratch by NVIDIA, designed for both reasoning and non-reasoning
		// Supports <think> tags for hybrid reasoning mode
		// Recommended settings: temp=0.6, top_p=0.95, ctx=32768
		{
			Name:         "Nemotron-3-Nano-30B-A3B-Q4_K_M",
			URL:          "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
			Quantization: "Q4_K_M",
			Parameters:   "30b-a3b",
			MinRAM:       24, // ~24.6 GB file
			Description:  "Nemotron-3-Nano 30B (3B active) - NVIDIA's best small model for reasoning & tools",
		},
		// Nemotron-3-Nano-30B-A3B UD-Q4_K_XL - Unsloth Dynamic quantization (recommended)
		{
			Name:         "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL",
			URL:          "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf",
			Quantization: "UD-Q4_K_XL",
			Parameters:   "30b-a3b",
			MinRAM:       24, // ~22.8 GB file
			Description:  "Nemotron-3-Nano 30B UD-Q4 - Dynamic quantized, slightly smaller",
		},
		// Nemotron-3-Nano-30B-A3B Q6_K - Higher quality for systems with more RAM
		{
			Name:         "Nemotron-3-Nano-30B-A3B-Q6_K",
			URL:          "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-Q6_K.gguf",
			Quantization: "Q6_K",
			Parameters:   "30b-a3b",
			MinRAM:       32, // ~32 GB file
			Description:  "Nemotron-3-Nano 30B Q6 - Higher quality for high-RAM systems",
		},
	}
}

// SelectOptimalFunctionCallingModel selects the best function calling model based on available RAM
// Logic:
//   - RAM >= 24GB: Use Nemotron-3-Nano-30B (best quality, MoE architecture)
//   - RAM < 24GB: Use FunctionGemma 270M (tiny but specialized)
func SelectOptimalFunctionCallingModel(availableRAM int64) QwenModelSpec {
	models := GetRecommendedFunctionCallingModels()

	// Threshold for Nemotron-3-Nano
	const nemotronMinRAM int64 = 24

	if availableRAM >= nemotronMinRAM {
		// Select Nemotron-3-Nano (first one that fits)
		for _, model := range models {
			if strings.Contains(strings.ToLower(model.Name), "nemotron") && model.MinRAM <= availableRAM {
				log.Printf("📦 Selected function calling model: %s (%s, %s) - requires %dGB RAM (system has %dGB)",
					model.Name, model.Parameters, model.Quantization,
					model.MinRAM, availableRAM)
				return model
			}
		}
	}

	// Fallback to FunctionGemma (RAM < 24GB or no Nemotron fits)
	for _, model := range models {
		if strings.Contains(strings.ToLower(model.Name), "functiongemma") {
			log.Printf("📦 Selected function calling model: %s (%s, %s) - requires %dGB RAM (system has %dGB)",
				model.Name, model.Parameters, model.Quantization,
				model.MinRAM, availableRAM)
			return model
		}
	}

	// Ultimate fallback - return first model (FunctionGemma BF16)
	log.Printf("⚠️  No optimal function calling model found, using default: %s", models[0].Name)
	return models[0]
}

// SelectOptimalQwenModel selects the best VL model based on available RAM.
// It returns the largest model that fits within the available RAM.
// If no model fits, it returns the smallest model as a fallback.
// Note: Currently excludes 7B model as it requires multi-file download support
func SelectOptimalQwenModel(availableRAM int64) QwenModelSpec {
	models := GetRecommendedVLModels()

	// Select the largest model that fits in available RAM
	// But skip 7B model for now (multi-file download not yet supported)
	var selectedModel QwenModelSpec
	for _, model := range models {
		// Skip 7B model (requires multi-file download)
		if model.Parameters == "7b" {
			continue
		}

		if model.MinRAM <= availableRAM {
			selectedModel = model
		} else {
			break // Models are ordered, so we can stop here
		}
	}

	// If no model fits, use the smallest one as fallback
	if selectedModel.Name == "" {
		selectedModel = models[0]
		log.Printf("⚠️  System has low RAM (%dGB), using smallest model", availableRAM)
	}

	log.Printf("📦 Selected VL model: %s (%s, %s) - requires %dGB RAM (system has %dGB)",
		selectedModel.Name, selectedModel.Parameters, selectedModel.Quantization,
		selectedModel.MinRAM, availableRAM)

	return selectedModel
}

// SelectOptimalQwenTextModel selects the best text model based on available RAM.
// NEW LOGIC (per user request):
//   - RAM >= 24GB: Use Nemotron-3-Nano-30B (MoE, 30B params, 3B active) - Best for high-end systems
//   - RAM < 24GB: Use FunctionGemma 270M - Tiny but specialized for function calling
//
// This uses function calling models (FunctionGemma or Nemotron-3-Nano) based on RAM.
//
// References:
//   - Nemotron-3-Nano: https://docs.unsloth.ai/models/nemotron-3
//   - FunctionGemma: https://docs.unsloth.ai/models/functiongemma
func SelectOptimalQwenTextModel(availableRAM int64) QwenModelSpec {
	// Use the new function calling model selection logic
	// This implements the user's requirement:
	// - RAM >= 24GB → Nemotron-3-Nano-30B
	// - RAM < 24GB → FunctionGemma 270M
	return SelectOptimalFunctionCallingModel(availableRAM)
}

// Note: DownloadModel, CleanupStaleTempFiles, and GetAvailableModels have been moved to installer.go
// Use LlamaCppInstaller methods instead:
//   - installer.DownloadChatModel(modelSpec)
//   - installer.CleanupStaleTempFiles()
//   - installer.GetAvailableChatModels()

// validateDownloadedFile performs all validation checks on a downloaded file
// Used by installer.go for model validation
func validateDownloadedFile(filePath string, modelSpec QwenModelSpec) error {
	// Verify checksum if provided
	if modelSpec.SHA256 != "" {
		log.Printf("🔒 Verifying model integrity...")
		if err := verifyModelChecksum(filePath, modelSpec.SHA256); err != nil {
			return fmt.Errorf("model integrity check failed: %w", err)
		}
		log.Printf("✅ Model integrity verified")
	}

	// Verify it's a valid GGUF file
	if err := validateGGUFFile(filePath); err != nil {
		return fmt.Errorf("invalid GGUF file: %w", err)
	}

	return nil
}

// verifyModelChecksum verifies the SHA256 checksum of a file
// Used by installer.go for model validation
func verifyModelChecksum(filePath, expectedChecksum string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file for checksum: %w", err)
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return fmt.Errorf("failed to compute checksum: %w", err)
	}

	actualChecksum := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(actualChecksum, expectedChecksum) {
		return fmt.Errorf("checksum mismatch: got %s, expected %s", actualChecksum, expectedChecksum)
	}

	return nil
}

// validateGGUFFile performs basic validation on a GGUF file by checking the magic number
// Used by installer.go for model validation
func validateGGUFFile(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Read GGUF magic number (first 4 bytes should be "GGUF")
	magic := make([]byte, ggufMagicNumberSize)
	if _, err := io.ReadFull(file, magic); err != nil {
		return fmt.Errorf("failed to read file header: %w", err)
	}

	if string(magic) != ggufMagicNumber {
		return fmt.Errorf("invalid GGUF magic number: got %q, expected %q", string(magic), ggufMagicNumber)
	}

	return nil
}

// ============================================================================
// Embedding Model Functions
// ============================================================================

// EmbeddingModel represents an embedding model configuration
type EmbeddingModel struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	URL          string   `json:"url"`
	Filename     string   `json:"filename"`
	Size         int64    `json:"size"`
	SHA256       string   `json:"sha256"`
	Quantization string   `json:"quantization"`
	Dimensions   int      `json:"dimensions"`
	Languages    []string `json:"languages"`
	Type         string   `json:"type"` // "embedding"
}

var embeddingModelsCatalog = map[string]*EmbeddingModel{
	"granite-embedding-107m-multilingual-q6_k_l": {
		Name:         "granite-embedding-107m-multilingual-q6_k_l",
		Description:  "Granite Embedding 107M Multilingual (Q6_K_L) - Very high quality, recommended",
		URL:          "https://huggingface.co/bartowski/granite-embedding-107m-multilingual-GGUF/resolve/main/granite-embedding-107m-multilingual-Q6_K_L.gguf",
		Filename:     "granite-embedding-107m-multilingual-Q6_K_L.gguf",
		Size:         120163008,
		Quantization: "Q6_K_L",
		Dimensions:   384,
		Languages:    []string{"en", "de", "fr", "it", "pt", "es", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "th", "vi", "id"},
		Type:         "embedding",
	},
	"granite-embedding-107m-multilingual-q4_k_m": {
		Name:         "granite-embedding-107m-multilingual-q4_k_m",
		Description:  "Granite Embedding 107M Multilingual (Q4_K_M) - Good quality, smaller size",
		URL:          "https://huggingface.co/bartowski/granite-embedding-107m-multilingual-GGUF/resolve/main/granite-embedding-107m-multilingual-Q4_K_M.gguf",
		Filename:     "granite-embedding-107m-multilingual-Q4_K_M.gguf",
		Size:         123000000,
		Quantization: "Q4_K_M",
		Dimensions:   384,
		Languages:    []string{"en", "de", "fr", "it", "pt", "es", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "th", "vi", "id"},
		Type:         "embedding",
	},
	"granite-embedding-107m-multilingual-f16": {
		Name:         "granite-embedding-107m-multilingual-f16",
		Description:  "Granite Embedding 107M Multilingual (F16) - Full precision, highest quality",
		URL:          "https://huggingface.co/bartowski/granite-embedding-107m-multilingual-GGUF/resolve/main/granite-embedding-107m-multilingual-f16.gguf",
		Filename:     "granite-embedding-107m-multilingual-f16.gguf",
		Size:         236000000,
		Quantization: "F16",
		Dimensions:   384,
		Languages:    []string{"en", "de", "fr", "it", "pt", "es", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "th", "vi", "id"},
		Type:         "embedding",
	},
	"paraphrase-multilingual-minilm-l12-v2-f16": {
		Name:         "paraphrase-multilingual-minilm-l12-v2-f16",
		Description:  "Paraphrase Multilingual MiniLM L12 v2 (F16) - Alternative multilingual model",
		URL:          "https://huggingface.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf/resolve/main/paraphrase-multilingual-MiniLM-L12-118M-v2-F16.gguf",
		Filename:     "paraphrase-multilingual-MiniLM-L12-118M-v2-F16.gguf",
		Size:         242358176,
		Quantization: "F16",
		Dimensions:   384,
		Languages:    []string{"en", "de", "fr", "it", "pt", "es", "nl", "pl", "ru", "zh", "ja", "ko"},
		Type:         "embedding",
	},
	"nomic-embed-text-v1.5-q4_k_m": {
		Name:         "nomic-embed-text-v1.5-q4_k_m",
		Description:  "Nomic Embed Text v1.5 (Q4_K_M) - Popular, high-quality embeddings",
		URL:          "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
		Filename:     "nomic-embed-text-v1.5.Q4_K_M.gguf",
		Size:         550000000,
		Quantization: "Q4_K_M",
		Dimensions:   768,
		Languages:    []string{"en", "de", "fr", "it", "pt", "es", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "th", "vi", "id"},
		Type:         "embedding",
	},
}

// GetAvailableEmbeddingModels returns the catalog of available embedding models
func GetAvailableEmbeddingModels() map[string]*EmbeddingModel {
	return embeddingModelsCatalog
}

// GetEmbeddingModel returns a specific embedding model by name
func GetEmbeddingModel(name string) (*EmbeddingModel, bool) {
	model, exists := embeddingModelsCatalog[name]
	return model, exists
}

// GetRecommendedEmbeddingModel returns the recommended embedding model name
func GetRecommendedEmbeddingModel() string {
	return "granite-embedding-107m-multilingual-q6_k_l"
}

// Embedding model catalog - contains verified embedding models with correct URLs
//   - GetEmbeddingModelsDirectory() → installer.GetModelsDirectory()
//   - IsEmbeddingModelDownloaded() → used internally by installer
//   - GetDownloadedEmbeddingModels() → installer.GetDownloadedEmbeddingModels()
//   - DownloadEmbeddingModel() → installer.DownloadEmbeddingModel()
//   - AutoDownloadRecommendedEmbeddingModel() → installer.AutoDownloadRecommendedEmbeddingModel()
//
// This file now only contains:
//   - Model specifications and catalogs (data layer)
//   - Validation helpers (used by installer)

// validateEmbeddingGGUFFile validates a GGUF file by checking its header structure.
// It performs more thorough validation than validateGGUFFile by checking version, tensor count, and metadata count.
func validateEmbeddingGGUFFile(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// GGUF files start with a magic number: "GGUF" (4 bytes)
	magic := make([]byte, ggufMagicNumberSize)
	if _, err := io.ReadFull(file, magic); err != nil {
		return fmt.Errorf("failed to read magic bytes: %w", err)
	}

	if string(magic) != ggufMagicNumber {
		return fmt.Errorf("invalid GGUF file: wrong magic bytes (expected %q, got %q)", ggufMagicNumber, string(magic))
	}

	// Read version (4 bytes, little-endian uint32)
	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}

	// Check if version is reasonable
	if version == 0 || version > maxGGUFVersion {
		return fmt.Errorf("invalid GGUF file: unreasonable version %d (expected 1-%d)", version, maxGGUFVersion)
	}

	// Read tensor count (8 bytes, little-endian uint64)
	var tensorCount uint64
	if err := binary.Read(file, binary.LittleEndian, &tensorCount); err != nil {
		return fmt.Errorf("failed to read tensor count: %w", err)
	}

	// Read metadata key-value count (8 bytes, little-endian uint64)
	var metadataCount uint64
	if err := binary.Read(file, binary.LittleEndian, &metadataCount); err != nil {
		return fmt.Errorf("failed to read metadata count: %w", err)
	}

	// Basic sanity checks
	if tensorCount == 0 {
		return fmt.Errorf("invalid GGUF file: no tensors found")
	}

	if tensorCount > maxTensorCount {
		return fmt.Errorf("invalid GGUF file: unreasonable tensor count %d (max %d)", tensorCount, maxTensorCount)
	}

	if metadataCount > maxMetadataCount {
		return fmt.Errorf("invalid GGUF file: unreasonable metadata count %d (max %d)", metadataCount, maxMetadataCount)
	}

	log.Printf("GGUF file validation passed: version=%d, tensors=%d, metadata_entries=%d",
		version, tensorCount, metadataCount)
	return nil
}
