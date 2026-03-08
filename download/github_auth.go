package download

import (
	"os"
	"strings"
)

const EnvLlamaCppVersion = "LLAMA_CPP_VERSION"

func githubToken() string {
	if v := strings.TrimSpace(os.Getenv("GH_TOKEN")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("GITHUB_TOKEN")); v != "" {
		return v
	}
	return ""
}

func llamaCppVersionOverride() string {
	return strings.TrimSpace(os.Getenv(EnvLlamaCppVersion))
}
