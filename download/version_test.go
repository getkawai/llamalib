package download

import (
	"os"
	"testing"
)

func TestResolveVersion_WithExplicitVersion(t *testing.T) {
	t.Setenv(EnvLlamaCppVersion, "b9999")
	v, err := ResolveVersion("b1234")
	if err != nil {
		t.Fatalf("ResolveVersion should accept explicit version: %v", err)
	}
	if v != "b1234" {
		t.Fatalf("ResolveVersion returned %q, expected %q", v, "b1234")
	}
}

func TestResolveVersion_WithEnvOverride(t *testing.T) {
	t.Setenv(EnvLlamaCppVersion, "b7777")
	v, err := ResolveVersion("")
	if err != nil {
		t.Fatalf("ResolveVersion should accept env override: %v", err)
	}
	if v != "b7777" {
		t.Fatalf("ResolveVersion returned %q, expected %q", v, "b7777")
	}
}

func TestResolveVersion_WithInvalidEnvOverride(t *testing.T) {
	t.Setenv(EnvLlamaCppVersion, "latest")
	_, err := ResolveVersion("")
	if err == nil {
		t.Fatal("ResolveVersion should fail for invalid env override")
	}
}

func TestLlamaCppVersionOverride_EmptyWhenUnset(t *testing.T) {
	if err := os.Unsetenv(EnvLlamaCppVersion); err != nil {
		t.Fatalf("Unsetenv failed: %v", err)
	}
	if v := llamaCppVersionOverride(); v != "" {
		t.Fatalf("llamaCppVersionOverride returned %q, expected empty", v)
	}
}
