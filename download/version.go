package download

import "fmt"

// ResolveVersion returns the explicit version if provided, otherwise
// uses LLAMA_CPP_VERSION override, and falls back to GitHub latest lookup.
func ResolveVersion(version string) (string, error) {
	if version != "" {
		if err := VersionIsValid(version); err != nil {
			return "", ErrInvalidVersion
		}
		return version, nil
	}

	if envVersion := llamaCppVersionOverride(); envVersion != "" {
		if err := VersionIsValid(envVersion); err != nil {
			return "", fmt.Errorf("%s is set but invalid: %w", EnvLlamaCppVersion, ErrInvalidVersion)
		}
		return envVersion, nil
	}

	return LlamaLatestVersion()
}
