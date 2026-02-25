// Package llamalib provides the core infrastructure for llama.cpp integration.
//
// This package contains:
//   - Service: Main service for managing llama.cpp models (chat, VL, embedding)
//   - Installer: Cross-platform llama.cpp binary installer
//   - Model specs: Definitions for supported models
//   - Templates: Chat templates for various model architectures
//
// Usage:
//
//	service, err := llamalib.NewService()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer service.Cleanup()
//
//	// Wait for initialization
//	ctx := context.Background()
//	if err := service.WaitForInitialization(ctx); err != nil {
//	    log.Fatal(err)
//	}
//
//	// Load and use models
//	service.LoadChatModel("")  // Auto-select best model
//
// This package is used by fantasy/providers/llama and fantasy/providers/llama-vl
// to provide text generation and vision-language capabilities respectively.
package llamalib
