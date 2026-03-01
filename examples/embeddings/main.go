package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/getkawai/llamalib/llama"
)

func main() {
	// Parse command line flags
	modelPath := flag.String("model", "", "Path to the embedding model")
	prompt := flag.String("p", "Hello World", "Prompt to generate embeddings for")
	flag.Parse()

	// Get library path from environment
	libPath := os.Getenv("YZMA_LIB")
	if libPath == "" {
		fmt.Println("Error: YZMA_LIB environment variable not set")
		os.Exit(1)
	}

	// Use model from flag or environment
	if *modelPath == "" {
		*modelPath = os.Getenv("YZMA_TEST_MODEL")
	}

	if *modelPath == "" {
		fmt.Println("Error: No model specified. Use -model flag or YZMA_TEST_MODEL environment variable")
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
	model, err := llama.ModelLoadFromFile(*modelPath, llama.ModelDefaultParams())
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer llama.ModelFree(model)

	// Create context with embedding mode enabled
	params := llama.ContextDefaultParams()
	params.NCtx = 2048
	params.Embeddings = 1 // Enable embeddings
	params.PoolingType = llama.PoolingTypeNone

	ctx, err := llama.InitFromModel(model, params)
	if err != nil {
		fmt.Printf("Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer llama.Free(ctx)

	// Tokenize the prompt
	vocab := llama.ModelGetVocab(model)
	tokens := llama.Tokenize(vocab, *prompt, true, true)

	fmt.Printf("Input: %s\n", *prompt)
	fmt.Printf("Tokens: %d\n", len(tokens))

	// Create batch and decode
	batch := llama.BatchGetOne(tokens)
	if _, err := llama.Decode(ctx, batch); err != nil {
		fmt.Printf("Error decoding: %v\n", err)
		os.Exit(1)
	}

	// Get embeddings
	nOutputs := len(tokens)
	nEmbeddings := int(llama.ModelNEmbd(model))
	embeddings, err := llama.GetEmbeddings(ctx, nOutputs, nEmbeddings)
	if err != nil {
		fmt.Printf("Error getting embeddings: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Embedding dimensions per token: %d\n", nEmbeddings)
	fmt.Printf("Total embedding values: %d\n", len(embeddings))
	
	// Safe slice access for first 10 values
	if len(embeddings) > 0 {
		safeEnd := 10
		if len(embeddings) < safeEnd {
			safeEnd = len(embeddings)
		}
		fmt.Printf("Embedding (first %d values): %v\n", safeEnd, embeddings[:safeEnd])
	} else {
		fmt.Println("Embedding (first 10 values): (empty)")
	}

	// Calculate some basic statistics
	var sum, min, max float32
	if len(embeddings) > 0 {
		min = embeddings[0]
		max = embeddings[0]
		for _, v := range embeddings {
			sum += v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		mean := sum / float32(len(embeddings))

		fmt.Printf("\nEmbedding statistics:\n")
		fmt.Printf("  Mean: %.6f\n", mean)
		fmt.Printf("  Min:  %.6f\n", min)
		fmt.Printf("  Max:  %.6f\n", max)
	} else {
		fmt.Println("\nEmbedding statistics: (no data)")
	}

	fmt.Println("\nEmbedding generation complete!")
}
