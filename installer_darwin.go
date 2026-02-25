//go:build darwin

package llamalib

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// detectHardwareCapabilities detects macOS system hardware capabilities
func (lcm *LlamaCppInstaller) detectHardwareCapabilities() *HardwareCapabilities {
	caps := &HardwareCapabilities{
		OS:        "darwin",
		Arch:      runtime.GOARCH,
		HasNVIDIA: false, // Modern Macs don't have NVIDIA GPUs
		HasAMD:    false, // Apple Silicon or Intel integrated
		HasIntel:  false, // Could be detected if needed
		HasCUDA:   false, // macOS doesn't support CUDA
		HasVulkan: false, // macOS uses Metal instead
		HasOpenCL: false, // macOS deprecated OpenCL
		HasAVX2:   true,  // Modern Macs support AVX2 (Intel) or equivalent (Apple Silicon)
	}

	log.Printf("Hardware capabilities: OS=%s, Arch=%s, NVIDIA=%v, CUDA=%v, Vulkan=%v",
		caps.OS, caps.Arch, caps.HasNVIDIA, caps.HasCUDA, caps.HasVulkan)

	return caps
}

// GetBinaryPath returns the path to a specific llama.cpp binary on macOS
// Priority: 1) Local binary path, 2) Homebrew paths, 3) System PATH
func (lcm *LlamaCppInstaller) GetBinaryPath(binaryName string) string {
	// First check local binary path
	localPath := filepath.Join(lcm.BinaryPath, binaryName)
	if _, err := os.Stat(localPath); err == nil {
		return localPath
	}

	// Check Homebrew paths (Apple Silicon and Intel)
	homebrewPaths := []string{
		"/opt/homebrew/bin/" + binaryName, // Apple Silicon
		"/usr/local/bin/" + binaryName,    // Intel Mac
	}

	for _, path := range homebrewPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// Fallback to system PATH
	if systemPath, err := exec.LookPath(binaryName); err == nil {
		return systemPath
	}

	// Return local path as default (even if not exists) for error messages
	return localPath
}

// GetLlamaCLICacheDirectory returns the llama-cli cache directory on macOS
func GetLlamaCLICacheDirectory() string {
	homeDir, _ := os.UserHomeDir()
	return filepath.Join(homeDir, "Library", "Caches", "llama.cpp")
}

// IsLlamaCppInstalled and VerifyInstalledBinary are now in installer.go (unified)
