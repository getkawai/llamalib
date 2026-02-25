//go:build windows

package llamalib

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// WindowsHardware represents detected Windows hardware capabilities
type WindowsHardware struct {
	GPU       string
	IsNVIDIA  bool
	IsAMD     bool
	IsIntel   bool
	HasCUDA   bool
	HasVulkan bool
	HasOpenCL bool
	HasAVX2   bool
}

// detectWindowsHardware detects Windows hardware capabilities
func (lcm *LlamaCppInstaller) detectWindowsHardware() *WindowsHardware {
	hardware := &WindowsHardware{}

	// Detect GPU
	hardware.GPU = lcm.detectWindowsGPU()
	hardware.IsNVIDIA = strings.Contains(strings.ToLower(hardware.GPU), "nvidia") || strings.Contains(strings.ToLower(hardware.GPU), "geforce") || strings.Contains(strings.ToLower(hardware.GPU), "rtx") || strings.Contains(strings.ToLower(hardware.GPU), "gtx")
	hardware.IsAMD = strings.Contains(strings.ToLower(hardware.GPU), "amd") || strings.Contains(strings.ToLower(hardware.GPU), "radeon")
	hardware.IsIntel = strings.Contains(strings.ToLower(hardware.GPU), "intel")

	// Detect CUDA (NVIDIA GPUs only)
	if hardware.IsNVIDIA {
		hardware.HasCUDA = lcm.detectCUDA()
	}

	// Detect Vulkan
	hardware.HasVulkan = lcm.detectVulkan()

	// Detect OpenCL
	hardware.HasOpenCL = lcm.detectOpenCL()

	// Detect AVX2 (assume modern CPUs have it)
	hardware.HasAVX2 = lcm.detectAVX2()

	return hardware
}

// detectWindowsGPU detects GPU on Windows
func (lcm *LlamaCppInstaller) detectWindowsGPU() string {
	// Try wmic first
	if out, err := exec.Command("wmic", "path", "win32_VideoController", "get", "name", "/value").Output(); err == nil {
		lines := strings.Split(string(out), "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "Name=") {
				gpu := strings.TrimPrefix(line, "Name=")
				gpu = strings.TrimSpace(gpu)
				if gpu != "" && !strings.Contains(gpu, "Microsoft") {
					return gpu
				}
			}
		}
	}

	// Try PowerShell as fallback
	if out, err := exec.Command("powershell", "-Command", "Get-WmiObject -Class Win32_VideoController | Select-Object -ExpandProperty Name").Output(); err == nil {
		lines := strings.Split(string(out), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" && !strings.Contains(line, "Microsoft") {
				return line
			}
		}
	}

	return "Unknown GPU"
}

// detectCUDA detects CUDA availability
func (lcm *LlamaCppInstaller) detectCUDA() bool {
	// Check for nvidia-smi
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		// Try to run nvidia-smi to verify CUDA is working
		if err := exec.Command("nvidia-smi").Run(); err == nil {
			return true
		}
	}

	// Check for CUDA installation paths
	cudaPaths := []string{
		"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
		"C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA",
	}

	for _, path := range cudaPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}

	return false
}

// detectVulkan detects Vulkan availability
func (lcm *LlamaCppInstaller) detectVulkan() bool {
	// Check Windows registry for Vulkan
	if out, err := exec.Command("reg", "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Khronos\\Vulkan\\Drivers").Output(); err == nil {
		return len(strings.TrimSpace(string(out))) > 0
	}

	// Check for Vulkan DLLs in system
	vulkanPaths := []string{
		"C:\\Windows\\System32\\vulkan-1.dll",
		"C:\\Windows\\SysWOW64\\vulkan-1.dll",
	}

	for _, path := range vulkanPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}

	return false
}

// detectOpenCL detects OpenCL availability
func (lcm *LlamaCppInstaller) detectOpenCL() bool {
	// Check for OpenCL DLLs
	openclPaths := []string{
		"C:\\Windows\\System32\\OpenCL.dll",
		"C:\\Windows\\SysWOW64\\OpenCL.dll",
	}

	for _, path := range openclPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}

	return false
}

// detectAVX2 detects AVX2 CPU instruction support
func (lcm *LlamaCppInstaller) detectAVX2() bool {
	// Try to get CPU info via wmic
	if out, err := exec.Command("wmic", "cpu", "get", "name", "/value").Output(); err == nil {
		cpuInfo := strings.ToLower(string(out))
		// Most modern CPUs support AVX2, but we can check for known CPU families
		if strings.Contains(cpuInfo, "intel") || strings.Contains(cpuInfo, "amd") {
			// Assume modern Intel/AMD CPUs have AVX2 (safer assumption)
			return true
		}
	}

	// Default to true for modern systems
	return true
}

// detectHardwareCapabilities detects Windows system hardware capabilities
func (lcm *LlamaCppInstaller) detectHardwareCapabilities() *HardwareCapabilities {
	winHw := lcm.detectWindowsHardware()

	caps := &HardwareCapabilities{
		OS:        "windows",
		Arch:      runtime.GOARCH,
		HasNVIDIA: winHw.IsNVIDIA,
		HasAMD:    winHw.IsAMD,
		HasIntel:  winHw.IsIntel,
		HasCUDA:   winHw.HasCUDA,
		HasVulkan: winHw.HasVulkan,
		HasOpenCL: winHw.HasOpenCL,
		HasAVX2:   winHw.HasAVX2,
	}

	log.Printf("Hardware capabilities: OS=%s, Arch=%s, NVIDIA=%v, CUDA=%v, Vulkan=%v",
		caps.OS, caps.Arch, caps.HasNVIDIA, caps.HasCUDA, caps.HasVulkan)

	return caps
}

// getAssetPatterns returns Windows-specific asset patterns in priority order
func (lcm *LlamaCppInstaller) getAssetPatterns(hardware *HardwareCapabilities) []string {
	var patterns []string

	// Windows priority: CUDA > Vulkan > OpenCL > AVX2 > Generic
	if hardware.HasNVIDIA && hardware.HasCUDA {
		patterns = append(patterns, "cudart.*win.*cuda.*x64")
	}
	if hardware.HasVulkan {
		patterns = append(patterns, ".*win.*vulkan.*x64")
	}
	if hardware.HasOpenCL {
		patterns = append(patterns, ".*win.*opencl.*x64")
	}
	patterns = append(patterns, ".*win.*avx2.*x64")
	patterns = append(patterns, ".*win.*x64")

	return patterns
}

// GetBinaryPath returns the path to a specific llama.cpp binary on Windows
// Priority: 1) Local binary path, 2) System PATH
func (lcm *LlamaCppInstaller) GetBinaryPath(binaryName string) string {
	// Add .exe extension if not present
	if !strings.HasSuffix(binaryName, ".exe") {
		binaryName += ".exe"
	}

	// First check local binary path
	localPath := filepath.Join(lcm.BinaryPath, binaryName)
	if _, err := os.Stat(localPath); err == nil {
		return localPath
	}

	// Fallback to system PATH
	if systemPath, err := exec.LookPath(binaryName); err == nil {
		return systemPath
	}

	// Return local path as default (even if not exists) for error messages
	return localPath
}

// GetLlamaCLICacheDirectory returns the llama-cli cache directory on Windows
func GetLlamaCLICacheDirectory() string {
	// Use LOCALAPPDATA environment variable
	localAppData := os.Getenv("LOCALAPPDATA")
	if localAppData == "" {
		// Fallback to user home directory
		homeDir, _ := os.UserHomeDir()
		localAppData = filepath.Join(homeDir, "AppData", "Local")
	}
	return filepath.Join(localAppData, "llama.cpp", "cache")
}
