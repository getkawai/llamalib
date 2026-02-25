package download

import (
	"context"
)

// GetModel downloads a model from the specified URL to the destination path.
func GetModel(url, dest string) error {
	return getModel(context.Background(), url, dest, ProgressTracker)
}

// GetModelWithProgress downloads a model from the specified URL to the destination path
// using the provided progress callback.
func GetModelWithProgress(url, dest string, progress ProgressCallback) error {
	return getModel(context.Background(), url, dest, progress)
}

// GetModelWithContext downloads a model from the specified URL to the destination path
// using the provided context and progress callback.
func GetModelWithContext(ctx context.Context, url, dest string, progress ProgressCallback) error {
	return getModel(ctx, url, dest, progress)
}

func getModel(ctx context.Context, url, dest string, progress ProgressCallback) error {
	return getFunc(ctx, url, dest, progress)
}
