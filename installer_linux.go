//go:build linux

package llamalib

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// hasNVIDIAGPU checks if system has NVIDIA GPU on Linux
func (lcm *LlamaCppInstaller) hasNVIDIAGPU() bool {
	// Check lspci for NVIDIA
	if out, err := exec.Command("lspci").Output(); err == nil {
		output := strings.ToLower(string(out))
		if strings.Contains(output, "nvidia") {
			return true
		}
	}

	// Check /proc/driver/nvidia
	if _, err := os.Stat("/proc/driver/nvidia"); err == nil {
		return true
	}

	return false
}

// detectLinuxCUDA detects CUDA availability on Linux
func (lcm *LlamaCppInstaller) detectLinuxCUDA() bool {
	// Check for nvidia-smi
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		if err := exec.Command("nvidia-smi").Run(); err == nil {
			return true
		}
	}

	// Check for CUDA libraries
	cudaPaths := []string{
		"/usr/local/cuda",
		"/opt/cuda",
		"/usr/lib/x86_64-linux-gnu/libcuda.so",
		"/usr/lib64/libcuda.so",
	}

	for _, path := range cudaPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}

	return false
}

// detectHardwareCapabilities detects Linux system hardware capabilities
func (lcm *LlamaCppInstaller) detectHardwareCapabilities() *HardwareCapabilities {
	caps := &HardwareCapabilities{
		OS:        "linux",
		Arch:      runtime.GOARCH,
		HasNVIDIA: lcm.hasNVIDIAGPU(),
		HasAMD:    false, // Could be detected via lspci if needed
		HasIntel:  false, // Could be detected via lspci if needed
		HasCUDA:   lcm.detectLinuxCUDA(),
		HasVulkan: false, // Could be detected if needed
		HasOpenCL: false, // Could be detected if needed
		HasAVX2:   true,  // Assume modern Linux systems
	}

	log.Printf("Hardware capabilities: OS=%s, Arch=%s, NVIDIA=%v, CUDA=%v, Vulkan=%v",
		caps.OS, caps.Arch, caps.HasNVIDIA, caps.HasCUDA, caps.HasVulkan)

	return caps
}

// getAssetPatterns returns Linux-specific asset patterns in priority order
func (lcm *LlamaCppInstaller) getAssetPatterns(hardware *HardwareCapabilities) []string {
	var patterns []string

	// Linux priority: CUDA > Ubuntu > Generic
	if hardware.HasNVIDIA && hardware.HasCUDA {
		patterns = append(patterns, "cudart.*linux.*cuda.*x64")
	}
	patterns = append(patterns, ".*ubuntu.*x64")
	patterns = append(patterns, ".*linux.*x64")

	return patterns
}

// GetBinaryPath returns the path to a specific llama.cpp binary on Linux
// Priority: 1) Local binary path, 2) Common system paths, 3) System PATH
func (lcm *LlamaCppInstaller) GetBinaryPath(binaryName string) string {
	// First check local binary path
	localPath := filepath.Join(lcm.BinaryPath, binaryName)
	if _, err := os.Stat(localPath); err == nil {
		return localPath
	}

	// Check common Linux paths
	systemPaths := []string{
		"/usr/local/bin/" + binaryName,
		"/usr/bin/" + binaryName,
		"/opt/llama.cpp/bin/" + binaryName,
	}

	for _, path := range systemPaths {
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

// GetLlamaCLICacheDirectory returns the llama-cli cache directory on Linux
func GetLlamaCLICacheDirectory() string {
	homeDir, _ := os.UserHomeDir()
	return filepath.Join(homeDir, ".cache", "llama.cpp")
}
