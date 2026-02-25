//go:build !darwin && !linux && !windows

package llamalib

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// detectHardwareCapabilities stub for unsupported platforms
func (lcm *LlamaCppInstaller) detectHardwareCapabilities() *HardwareCapabilities {
	return &HardwareCapabilities{
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
		HasNVIDIA: false,
		HasAMD:    false,
		HasIntel:  false,
		HasCUDA:   false,
		HasVulkan: false,
		HasOpenCL: false,
		HasAVX2:   false,
	}
}

// getAssetPatterns stub for unsupported platforms
func (lcm *LlamaCppInstaller) getAssetPatterns(hardware *HardwareCapabilities) []string {
	// Return empty patterns for unsupported platforms
	return []string{}
}

// GetBinaryPath returns the path to a specific llama.cpp binary on unsupported platforms
// Priority: 1) Local binary path, 2) System PATH
func (lcm *LlamaCppInstaller) GetBinaryPath(binaryName string) string {
	// First check local binary path
	localPath := filepath.Join(lcm.BinaryPath, binaryName)
	if _, err := os.Stat(localPath); err == nil {
		return localPath
	}

	// Try system PATH as fallback
	if systemPath, err := exec.LookPath(binaryName); err == nil {
		return systemPath
	}

	// Return local path as default (even if not exists) for error messages
	return localPath
}

// GetLlamaCLICacheDirectory returns the llama-cli cache directory on unsupported platforms
func GetLlamaCLICacheDirectory() string {
	homeDir, _ := os.UserHomeDir()
	return filepath.Join(homeDir, ".cache", "llama.cpp")
}
