package download

import (
	"os"
	"strings"
)

func githubToken() string {
	if v := strings.TrimSpace(os.Getenv("GH_TOKEN")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("GITHUB_TOKEN")); v != "" {
		return v
	}
	return ""
}
