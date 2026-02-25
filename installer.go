package llamalib

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/getkawai/llamalib/download"
	"github.com/kawai-network/y/hardware"
)

// Release represents a GitHub release
type Release struct {
	Version string  `json:"tag_name"`
	Name    string  `json:"name"`
	Body    string  `json:"body"`
	Assets  []Asset `json:"assets"`
	URL     string  `json:"html_url"`
}

// Asset represents a release asset
type Asset struct {
	Name               string `json:"name"`
	BrowserDownloadURL string `json:"browser_download_url"`
	Size               int64  `json:"size"`
}

// LlamaCppInstaller handles llama.cpp installation and model downloads
type LlamaCppInstaller struct {
	// BinaryPath is where the llama.cpp library is stored locally
	BinaryPath string
	// MetadataPath is where version metadata and cache are stored
	MetadataPath string
	// ModelsDir is where GGUF models (chat & embedding) are stored
	ModelsDir string
	// HardwareSpecs caches the detected hardware specifications
	HardwareSpecs *hardware.HardwareSpecs
}

// NewLlamaCppInstaller creates a new llama.cpp installer
func NewLlamaCppInstaller() *LlamaCppInstaller {
	homeDir, _ := os.UserHomeDir()
	basePath := filepath.Join(homeDir, ".llama-cpp")
	binaryPath := filepath.Join(basePath, "bin")
	metadataPath := filepath.Join(basePath, "metadata")
	modelsDir := filepath.Join(basePath, "models")

	installer := &LlamaCppInstaller{
		BinaryPath:    binaryPath,
		MetadataPath:  metadataPath,
		ModelsDir:     modelsDir,
		HardwareSpecs: hardware.DetectHardwareSpecs(),
	}

	return installer
}

// InstallLlamaCpp installs llama.cpp via direct download
// Works on all platforms (macOS, Linux, Windows)
// Uses download.InstallLibraries for automatic version management
func (lcm *LlamaCppInstaller) InstallLlamaCpp() error {
	// Detect processor type
	processor := lcm.detectProcessor()
	log.Printf("Auto-installing llama.cpp for %s/%s", runtime.GOOS, processor)

	// Convert string processor to download.Processor type
	var proc download.Processor
	switch processor {
	case "cpu":
		proc = download.CPU
	case "cuda":
		proc = download.CUDA
	case "vulkan":
		proc = download.Vulkan
	case "metal":
		proc = download.Metal
	default:
		proc = download.CPU
	}

	// Use InstallLibraries which handles:
	// - Version checking and management (version.json)
	// - Auto-upgrade support (if allowUpgrade=true)
	// - Automatic fallback to previous versions if latest fails
	// - OS/Processor validation
	if err := download.InstallLibraries(lcm.BinaryPath, proc, true); err != nil {
		return fmt.Errorf("failed to install llama.cpp: %w", err)
	}

	log.Println("✅ llama.cpp installed successfully")
	return nil
}

// detectProcessor detects the best processor type for this system
func (lcm *LlamaCppInstaller) detectProcessor() string {
	hardware := lcm.detectHardwareCapabilities()

	// Priority: CUDA > Vulkan > Metal > CPU
	if hardware.HasCUDA {
		return "cuda"
	}
	if hardware.HasVulkan {
		return "vulkan"
	}
	if runtime.GOOS == "darwin" {
		return "metal" // macOS always supports Metal
	}
	return "cpu"
}

// HardwareCapabilities represents detected hardware capabilities
type HardwareCapabilities struct {
	OS        string
	Arch      string
	HasNVIDIA bool
	HasAMD    bool
	HasIntel  bool
	HasCUDA   bool
	HasVulkan bool
	HasOpenCL bool
	HasAVX2   bool
}

// IsLlamaCppInstalled checks if llama.cpp library is installed
// Verifies that the main llama library exists (libllama.so/dylib/dll)
func (lcm *LlamaCppInstaller) IsLlamaCppInstalled() bool {
	libraryName := download.LibraryName(runtime.GOOS)
	// Guard against unknown or empty library name
	if libraryName == "unknown" || libraryName == "" {
		return false
	}

	libPath := filepath.Join(lcm.BinaryPath, libraryName)
	_, err := os.Stat(libPath)
	return err == nil
}

// VerifyInstalledBinary verifies the main llama library is installed
// Returns an error if the library is missing
func (lcm *LlamaCppInstaller) VerifyInstalledBinary() error {
	libraryName := download.LibraryName(runtime.GOOS)
	if libraryName == "unknown" || libraryName == "" {
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}

	libPath := filepath.Join(lcm.BinaryPath, libraryName)
	if _, err := os.Stat(libPath); err != nil {
		// Distinguish between NotExist and other errors
		if os.IsNotExist(err) {
			return fmt.Errorf("llama library not found: %s", libraryName)
		}
		return fmt.Errorf("failed to stat llama library %s: %w", libraryName, err)
	}

	return nil
}

// GetModelsDirectory returns the directory where models are stored
func (lcm *LlamaCppInstaller) GetModelsDirectory() string {
	return lcm.ModelsDir
}

// GetLibraryPath returns the path to the llama.cpp library directory
// This is the directory containing all required libraries (libggml, libggml-base, libllama)
// This path should be passed to llama.Load() for library-based usage
func (lcm *LlamaCppInstaller) GetLibraryPath() string {
	return lcm.BinaryPath
}

// GetLibraryFilePath returns the full path to the main llama.cpp library file
// Returns the platform-specific library file (libllama.so, libllama.dylib, or llama.dll)
func (lcm *LlamaCppInstaller) GetLibraryFilePath() string {
	libraryName := download.LibraryName(runtime.GOOS)
	return filepath.Join(lcm.BinaryPath, libraryName)
}

// ============================================================================
// Model Download Methods
// ============================================================================

// DownloadChatModel downloads a chat model (Qwen) using model specs from model_specs.go
// Features:
// - Downloads to temporary file first (.tmp) to prevent corruption
// - Supports resume on app restart (temp files preserved)
// - Retries up to 3 times on network failure with exponential backoff (2s, 4s, 6s)
// - Validates file size, checksum (if provided), and GGUF format
// - Automatically cleans up partial downloads on validation failure
// - Only moves to final destination after successful validation
// - Skips download if model already exists and is valid
func (lcm *LlamaCppInstaller) DownloadChatModel(modelSpec QwenModelSpec) error {
	// Ensure models directory exists
	if err := os.MkdirAll(lcm.ModelsDir, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	// Build model filename and paths
	modelFileName := fmt.Sprintf("%s.gguf", modelSpec.Name)
	destModelPath := filepath.Join(lcm.ModelsDir, modelFileName)
	tempModelPath := destModelPath + ".tmp"

	// Check if model already exists
	if _, err := os.Stat(destModelPath); err == nil {
		log.Printf("✅ Model already exists: %s", modelFileName)
		// Verify existing model integrity if checksum is provided
		if modelSpec.SHA256 != "" {
			if err := verifyModelChecksum(destModelPath, modelSpec.SHA256); err != nil {
				log.Printf("⚠️  Existing model checksum invalid, re-downloading...")
				if removeErr := os.Remove(destModelPath); removeErr != nil {
					log.Printf("⚠️  Failed to remove invalid model: %v", removeErr)
				}
			} else {
				return nil // Model exists and is valid
			}
		} else {
			return nil // Model exists, no checksum to verify
		}
	}

	log.Printf("📥 Downloading chat model: %s", modelSpec.Name)
	log.Printf("   URL: %s", modelSpec.URL)
	log.Printf("   This may take several minutes depending on network speed...")

	// Check if we're resuming
	if info, err := os.Stat(tempModelPath); err == nil {
		// Validate temp file before resume
		if info.IsDir() {
			log.Printf("⚠️  Temp path is a directory, removing: %s", tempModelPath)
			os.RemoveAll(tempModelPath)
		} else if info.Size() > 0 {
			log.Printf("🔄 Resuming download from %.1f MB", float64(info.Size())/(1024*1024))
		}
	}

	// Download using GetModelWithProgress with automatic retry, resume, and progress tracking
	if err := download.GetModelWithProgress(modelSpec.URL, tempModelPath, download.ProgressTracker); err != nil {
		lcm.cleanupTempFile(tempModelPath)
		return fmt.Errorf("failed to download model: %w", err)
	}

	// Verify downloaded file
	if err := validateDownloadedFile(tempModelPath, modelSpec); err != nil {
		lcm.cleanupTempFile(tempModelPath)
		return err
	}

	// Move temporary file to final destination
	if err := os.Rename(tempModelPath, destModelPath); err != nil {
		lcm.cleanupTempFile(tempModelPath)
		return fmt.Errorf("failed to move downloaded file: %w", err)
	}

	fileInfo, _ := os.Stat(destModelPath)
	sizeMB := float64(fileInfo.Size()) / (1024 * 1024)
	log.Printf("✅ Model downloaded successfully: %s (%.1f MB)", modelFileName, sizeMB)

	// Download projector file if specified (for VL models)
	if modelSpec.ProjectorURL != "" {
		projectorFileName := filepath.Base(modelSpec.ProjectorURL)
		destProjectorPath := filepath.Join(lcm.ModelsDir, projectorFileName)
		tempProjectorPath := destProjectorPath + ".tmp"

		// Check if projector already exists
		if _, err := os.Stat(destProjectorPath); err == nil {
			log.Printf("✅ Projector already exists: %s", projectorFileName)
		} else {
			log.Printf("📥 Downloading projector file: %s", projectorFileName)
			log.Printf("   URL: %s", modelSpec.ProjectorURL)

			if err := download.GetModelWithProgress(modelSpec.ProjectorURL, tempProjectorPath, download.ProgressTracker); err != nil {
				lcm.cleanupTempFile(tempProjectorPath)
				// Don't fail the whole process if projector fails, but warn
				log.Printf("⚠️  Failed to download projector: %v", err)
			} else {
				// Move temporary file to final destination
				if err := os.Rename(tempProjectorPath, destProjectorPath); err != nil {
					lcm.cleanupTempFile(tempProjectorPath)
					log.Printf("⚠️  Failed to move downloaded projector: %v", err)
				} else {
					log.Printf("✅ Projector downloaded successfully: %s", projectorFileName)
				}
			}
		}
	}

	return nil
}

// DownloadEmbeddingModel downloads an embedding model with automatic retry and cleanup
// Features:
// - Downloads to temporary file first (.tmp) to prevent corruption
// - Supports resume on app restart (temp files preserved)
// - Retries up to 3 times on network failure with exponential backoff (2s, 4s, 6s)
// - Validates GGUF file structure after download
// - Automatically cleans up partial downloads on validation failure
// - Only moves to final destination after successful validation
// - Skips download if model already exists
func (lcm *LlamaCppInstaller) DownloadEmbeddingModel(model *EmbeddingModel) error {
	// Ensure models directory exists
	if err := os.MkdirAll(lcm.ModelsDir, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	finalPath := filepath.Join(lcm.ModelsDir, model.Filename)

	// Check if already downloaded
	if _, err := os.Stat(finalPath); err == nil {
		log.Printf("✅ Embedding model already exists: %s", model.Name)
		return nil
	}

	log.Printf("📥 Downloading embedding model: %s", model.Name)
	log.Printf("   URL: %s", model.URL)
	log.Printf("   Size: %.2f MB", float64(model.Size)/1024/1024)

	// Download using GetModelWithProgress with automatic retry, resume, and progress tracking
	tempPath := finalPath + ".tmp"

	if err := download.GetModelWithProgress(model.URL, tempPath, download.ProgressTracker); err != nil {
		lcm.cleanupTempFile(tempPath)
		return fmt.Errorf("failed to download model: %w", err)
	}

	// Move temp file to final destination
	if err := os.Rename(tempPath, finalPath); err != nil {
		lcm.cleanupTempFile(tempPath)
		return fmt.Errorf("failed to move downloaded file: %w", err)
	}

	// Validate GGUF file structure
	if err := validateEmbeddingGGUFFile(finalPath); err != nil {
		lcm.cleanupTempFile(finalPath)
		return fmt.Errorf("downloaded file failed GGUF validation: %w", err)
	}

	log.Printf("✅ Embedding model downloaded successfully: %s", model.Name)
	return nil
}

// AutoDownloadRecommendedChatModel automatically downloads the best chat model for the system
func (lcm *LlamaCppInstaller) AutoDownloadRecommendedChatModel() error {
	// Check if any models already exist
	models, err := lcm.GetAvailableChatModels()
	if err != nil {
		return fmt.Errorf("failed to check existing models: %w", err)
	}

	if len(models) > 0 {
		log.Printf("✅ Chat models already available (%d found), skipping auto-download", len(models))
		return nil
	}

	log.Println("📦 No chat models found, starting auto-download...")

	// Select optimal model based on available RAM
	modelSpec := SelectOptimalQwenModel(lcm.HardwareSpecs.AvailableRAM)

	// Download the model
	if err := lcm.DownloadChatModel(modelSpec); err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	log.Println("🎉 Chat model download completed successfully!")
	return nil
}

// AutoDownloadRecommendedVLModel automatically downloads the best VL model for the system
func (lcm *LlamaCppInstaller) AutoDownloadRecommendedVLModel() error {
	// Check if any VL models already exist
	models, err := lcm.GetAvailableVLModels()
	if err != nil {
		return fmt.Errorf("failed to check existing models: %w", err)
	}

	var modelSpec QwenModelSpec
	if len(models) > 0 {
		log.Printf("✅ VL models already available (%d found), skipping auto-download", len(models))

		// Model exists, but check if projector exists
		// Get the model spec for the existing model
		modelSpec = SelectOptimalQwenModel(lcm.HardwareSpecs.AvailableRAM)

		// Check if projector file exists
		if modelSpec.ProjectorURL != "" {
			projectorFileName := filepath.Base(modelSpec.ProjectorURL)
			destProjectorPath := filepath.Join(lcm.ModelsDir, projectorFileName)

			if _, err := os.Stat(destProjectorPath); os.IsNotExist(err) {
				// Projector doesn't exist, download it
				log.Printf("📥 VL model exists but projector missing, downloading projector...")
				log.Printf("   Projector: %s", projectorFileName)
				log.Printf("   URL: %s", modelSpec.ProjectorURL)

				tempProjectorPath := destProjectorPath + ".tmp"

				if err := download.GetModelWithProgress(modelSpec.ProjectorURL, tempProjectorPath, download.ProgressTracker); err != nil {
					lcm.cleanupTempFile(tempProjectorPath)
					log.Printf("⚠️  Failed to download projector: %v", err)
				} else {
					if err := os.Rename(tempProjectorPath, destProjectorPath); err != nil {
						lcm.cleanupTempFile(tempProjectorPath)
						log.Printf("⚠️  Failed to move downloaded projector: %v", err)
					} else {
						log.Printf("✅ Projector downloaded successfully: %s", projectorFileName)
					}
				}
			} else {
				log.Printf("✅ Projector file already exists: %s", projectorFileName)
			}
		}

		return nil
	}

	log.Println("📦 No VL models found, starting auto-download...")

	// Select optimal model based on available RAM
	modelSpec = SelectOptimalQwenModel(lcm.HardwareSpecs.AvailableRAM)

	// Download the model (this will also download projector via DownloadChatModel)
	if err := lcm.DownloadChatModel(modelSpec); err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	log.Println("🎉 VL model download completed successfully!")
	return nil
}

// AutoDownloadRecommendedTextModel automatically downloads the best text model for the system
// NEW LOGIC (per user request):
//   - RAM >= 24GB: Downloads Nemotron-3-Nano-30B (MoE, 30B params, 3B active)
//   - RAM < 24GB: Downloads FunctionGemma 270M (tiny but specialized for function calling)
//
// References:
//   - Nemotron-3-Nano: https://docs.unsloth.ai/models/nemotron-3
//   - FunctionGemma: https://docs.unsloth.ai/models/functiongemma
func (lcm *LlamaCppInstaller) AutoDownloadRecommendedTextModel() error {
	// Select optimal model based on available RAM
	// Uses SelectOptimalFunctionCallingModel internally:
	// - RAM >= 24GB → Nemotron-3-Nano-30B
	// - RAM < 24GB → FunctionGemma 270M
	modelSpec := SelectOptimalQwenTextModel(lcm.HardwareSpecs.AvailableRAM)

	// Check if the preferred model already exists
	expectedFileName := filepath.Base(modelSpec.URL)
	expectedPath := filepath.Join(lcm.ModelsDir, expectedFileName)

	if _, err := os.Stat(expectedPath); err == nil {
		log.Printf("✅ Preferred function calling model already available: %s", modelSpec.Name)
		return nil
	}

	// Check if any function calling models exist (for logging purposes)
	models, _ := lcm.GetAvailableFunctionCallingModels()
	if len(models) > 0 {
		log.Printf("📦 Found %d other function calling model(s), but downloading preferred: %s", len(models), modelSpec.Name)
	} else {
		log.Printf("📦 No function calling models found, downloading: %s", modelSpec.Name)
	}

	// Log the selection rationale
	if lcm.HardwareSpecs.AvailableRAM >= 24 {
		log.Printf("💪 High RAM detected (%dGB >= 24GB), using Nemotron-3-Nano-30B", lcm.HardwareSpecs.AvailableRAM)
	} else {
		log.Printf("💡 Low/Medium RAM detected (%dGB < 24GB), using FunctionGemma 270M", lcm.HardwareSpecs.AvailableRAM)
	}

	// Download the preferred model
	if err := lcm.DownloadChatModel(modelSpec); err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	log.Println("🎉 Function calling model download completed successfully!")
	log.Printf("   Model: %s (%s)", modelSpec.Name, modelSpec.Description)
	return nil
}

// AutoDownloadRecommendedEmbeddingModel automatically downloads the best embedding model for the system based on hardware detection
func (lcm *LlamaCppInstaller) AutoDownloadRecommendedEmbeddingModel() error {
	downloaded := lcm.GetDownloadedEmbeddingModels()
	if len(downloaded) > 0 {
		log.Printf("✅ Embedding models already available (%d found), skipping auto-download", len(downloaded))
		return nil
	}

	log.Println("📦 No embedding models found, starting auto-download...")

	// Get recommended embedding model
	recommendedName := GetRecommendedEmbeddingModel()
	model, exists := GetEmbeddingModel(recommendedName)
	if !exists {
		return fmt.Errorf("recommended embedding model not found in catalog: %s", recommendedName)
	}

	log.Printf("📦 Selected recommended embedding model: %s (%s, %d dims, %.1f MB)",
		model.Name, model.Quantization, model.Dimensions, float64(model.Size)/(1024*1024))

	if err := lcm.DownloadEmbeddingModel(model); err != nil {
		return fmt.Errorf("failed to download embedding model: %w", err)
	}

	log.Println("🎉 Embedding model download completed successfully!")
	return nil
}

// AutoDownloadAllRecommendedModels automatically downloads all recommended model types for the system
// Downloads text, VL, and embedding models based on hardware detection
// Note: Utility models removed - main model handles all tasks (summary/title use same model)
func (lcm *LlamaCppInstaller) AutoDownloadAllRecommendedModels() error {
	log.Println("🚀 Starting automatic download of all recommended models...")

	// Download text model (FunctionGemma or Nemotron-3-Nano based on RAM)
	if err := lcm.AutoDownloadRecommendedTextModel(); err != nil {
		return fmt.Errorf("failed to download text model: %w", err)
	}

	// Download VL model
	if err := lcm.AutoDownloadRecommendedVLModel(); err != nil {
		return fmt.Errorf("failed to download VL model: %w", err)
	}

	// Download embedding model
	if err := lcm.AutoDownloadRecommendedEmbeddingModel(); err != nil {
		return fmt.Errorf("failed to download embedding model: %w", err)
	}

	log.Println("🎉 All recommended models downloaded successfully!")
	log.Println("📊 Model download summary:")
	log.Println("   ✅ Text model: Ready for chat, summaries, and titles")
	log.Println("   ✅ VL model: Ready for vision-language tasks")
	log.Println("   ✅ Embedding model: Ready for text embedding and similarity")

	return nil
}

// GetAvailableChatModels returns a list of available chat model file paths
func (lcm *LlamaCppInstaller) GetAvailableChatModels() ([]string, error) {
	if err := os.MkdirAll(lcm.ModelsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create models directory: %w", err)
	}

	entries, err := os.ReadDir(lcm.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read models directory: %w", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") {
			// Check if it's a chat model (not embedding model)
			if !lcm.isEmbeddingModel(name) {
				models = append(models, filepath.Join(lcm.ModelsDir, name))
			}
		}
	}

	return models, nil
}

// GetAvailableVLModels returns a list of available VL (vision-language) model file paths
func (lcm *LlamaCppInstaller) GetAvailableVLModels() ([]string, error) {
	if err := os.MkdirAll(lcm.ModelsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create models directory: %w", err)
	}

	entries, err := os.ReadDir(lcm.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read models directory: %w", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") &&
			strings.HasPrefix(name, "qwen3-vl-") &&
			!lcm.isEmbeddingModel(name) {
			models = append(models, filepath.Join(lcm.ModelsDir, name))
		}
	}

	return models, nil
}

// GetAvailableFunctionCallingModels returns a list of available function calling model file paths
// These are specialized models for tool/function calling (FunctionGemma, Nemotron-3-Nano)
func (lcm *LlamaCppInstaller) GetAvailableFunctionCallingModels() ([]string, error) {
	if err := os.MkdirAll(lcm.ModelsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create models directory: %w", err)
	}

	entries, err := os.ReadDir(lcm.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read models directory: %w", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		nameLower := strings.ToLower(name)

		// Must be a GGUF file
		if !strings.HasSuffix(nameLower, ".gguf") {
			continue
		}

		// Skip embedding models
		if lcm.isEmbeddingModel(name) {
			continue
		}

		// Check for function calling model patterns
		isFunctionCallingModel := strings.Contains(nameLower, "functiongemma") ||
			strings.Contains(nameLower, "nemotron-3-nano") ||
			strings.Contains(nameLower, "nemotron_3_nano")

		if isFunctionCallingModel {
			models = append(models, filepath.Join(lcm.ModelsDir, name))
		}
	}

	return models, nil
}

// GetAvailableTextModels returns a list of available text model file paths
// Text models are chat models including function calling models (FunctionGemma, Nemotron-3-Nano)
// Also includes: Llama, Mistral, Gemma, Qwen (non-VL)
// Excludes: VL models, embedding models, projector files
func (lcm *LlamaCppInstaller) GetAvailableTextModels() ([]string, error) {
	if err := os.MkdirAll(lcm.ModelsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create models directory: %w", err)
	}

	entries, err := os.ReadDir(lcm.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read models directory: %w", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		nameLower := strings.ToLower(name)

		// Must be a GGUF file
		if !strings.HasSuffix(nameLower, ".gguf") {
			continue
		}

		// Skip embedding models
		if lcm.isEmbeddingModel(name) {
			continue
		}

		// Skip VL models (vision-language)
		if strings.Contains(nameLower, "-vl-") || strings.Contains(nameLower, "_vl_") {
			continue
		}

		// Skip projector files (mmproj-*)
		if strings.HasPrefix(nameLower, "mmproj") {
			continue
		}

		// Check if it's a known text model prefix (including function calling models)
		isTextModel := strings.HasPrefix(nameLower, "llama-") ||
			strings.HasPrefix(nameLower, "mistral-") ||
			strings.HasPrefix(nameLower, "gemma-") ||
			strings.HasPrefix(nameLower, "phi-") ||
			strings.Contains(nameLower, "functiongemma") ||
			strings.Contains(nameLower, "nemotron") ||
			(strings.HasPrefix(nameLower, "qwen") && !strings.Contains(nameLower, "-vl-"))

		if isTextModel {
			models = append(models, filepath.Join(lcm.ModelsDir, name))
		}
	}

	return models, nil
}

// GetDownloadedEmbeddingModels returns a list of downloaded embedding models
func (lcm *LlamaCppInstaller) GetDownloadedEmbeddingModels() []*EmbeddingModel {
	var downloaded []*EmbeddingModel
	catalog := GetAvailableEmbeddingModels()

	for _, model := range catalog {
		modelPath := filepath.Join(lcm.ModelsDir, model.Filename)
		if _, err := os.Stat(modelPath); err == nil {
			// Validate the GGUF file structure
			if err := validateEmbeddingGGUFFile(modelPath); err == nil {
				downloaded = append(downloaded, model)
			}
		}
	}

	return downloaded
}

// CleanupStaleTempFiles is a no-op function kept for backward compatibility
// Temp files are now preserved to support resume functionality
// They will be overwritten on next download attempt or can be manually deleted by user
func (lcm *LlamaCppInstaller) CleanupStaleTempFiles() error {
	return nil
}

// Helper methods for model downloads

func (lcm *LlamaCppInstaller) cleanupTempFile(filePath string) error {
	if err := os.Remove(filePath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (lcm *LlamaCppInstaller) isEmbeddingModel(filename string) bool {
	catalog := GetAvailableEmbeddingModels()
	for _, model := range catalog {
		if model.Filename == filename {
			return true
		}
	}
	return false
}
