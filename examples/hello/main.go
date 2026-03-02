package main

import (
	"fmt"
	"os"

	"github.com/getkawai/llamalib/llama"
)

func main() {
	// Get library path from environment
	libPath := os.Getenv("YZMA_LIB")
	if libPath == "" {
		fmt.Println("Error: YZMA_LIB environment variable not set")
		os.Exit(1)
	}

	// Get model path from environment
	modelPath := os.Getenv("YZMA_TEST_MODEL")
	if modelPath == "" {
		fmt.Println("Error: YZMA_TEST_MODEL environment variable not set")
		os.Exit(1)
	}

	// Load the llama.cpp library
	if err := llama.Load(libPath); err != nil {
		fmt.Printf("Error loading library: %v\n", err)
		os.Exit(1)
	}
	defer llama.Close()

	// Initialize llama.cpp
	llama.Init()

	// Load the model
	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer llama.ModelFree(model)

	// Create context with default params
	params := llama.ContextDefaultParams()
	params.NCtx = 2048
	ctx, err := llama.InitFromModel(model, params)
	if err != nil {
		fmt.Printf("Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer llama.Free(ctx)

	// Create a chat message
	chat := []llama.ChatMessage{llama.NewChatMessage("user", "Hello, how are you?")}

	// Apply chat template
	buf := make([]byte, 4096)
	sz := llama.ChatApplyTemplate("", chat, false, buf)
	
	var prompt string
	if sz > 0 {
		prompt = string(buf[:sz])
		fmt.Printf("Formatted prompt: %s\n", prompt)
	} else {
		// Fallback to simple prompt if template fails
		prompt = "Hello, how are you?"
		fmt.Printf("Using simple prompt: %s\n", prompt)
	}

	// Setup samplers
	samplers := []llama.SamplerType{
		llama.SamplerTypeTopK,
		llama.SamplerTypeTopP,
		llama.SamplerTypeTemperature,
	}
	samplerParams := llama.DefaultSamplerParams()
	samplerParams.TopK = 40
	samplerParams.TopP = 0.9
	samplerParams.Temp = 0.8

	sampler := llama.NewSampler(model, samplers, samplerParams)
	defer llama.SamplerFree(sampler)

	// Tokenize the prompt
	vocab := llama.ModelGetVocab(model)
	tokens := llama.Tokenize(vocab, prompt, true, true)

	fmt.Printf("Input tokens: %d\n", len(tokens))

	// Decode the prompt
	batch := llama.BatchGetOne(tokens)
	if _, err := llama.Decode(ctx, batch); err != nil {
		fmt.Printf("Error decoding: %v\n", err)
		os.Exit(1)
	}

	// Generate response
	fmt.Println("\nGenerating response...")
	maxTokens := 128
	for i := 0; i < maxTokens; i++ {
		// Sample the next token
		tokenID := llama.SamplerSample(sampler, ctx, -1)

		// Check for end of generation
		if llama.VocabIsEOG(vocab, tokenID) {
			break
		}

		// Convert token to text
		buf := make([]byte, 256)
		n := llama.TokenToPiece(vocab, tokenID, buf, 0, false)
		if n <= 0 {
			break
		}
		text := string(buf[:n])

		fmt.Print(text)

		// Accept the token and continue
		llama.SamplerAccept(sampler, tokenID)

		// Decode the token
		batch = llama.BatchGetOne([]llama.Token{tokenID})
		if _, err := llama.Decode(ctx, batch); err != nil {
			break
		}
	}

	fmt.Println("\n\nGeneration complete!")
}
