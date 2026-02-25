package download

import (
	"fmt"
)

// ProgressCallback is a function that receives download progress updates.
// src: source URL
// currentSize: bytes downloaded so far
// totalSize: total bytes to download (-1 if unknown)
// mibPerSec: download speed in MiB/s
// complete: true when download is finished
type ProgressCallback func(src string, currentSize int64, totalSize int64, mibPerSec float64, complete bool)

// ProgressTracker is the default progress callback that prints download progress to stdout.
var ProgressTracker = DefaultProgressTracker()

// DefaultProgressTracker returns the default ProgressCallback that prints download progress to stdout.
func DefaultProgressTracker() ProgressCallback {
	return func(src string, currentSize int64, totalSize int64, mibPerSec float64, complete bool) {
		if totalSize > 0 {
			fmt.Printf("\r\x1b[Kdownloading %s... %d MiB of %d MiB (%.2f MiB/s)",
				src, currentSize/(1024*1024), totalSize/(1024*1024), mibPerSec)
		} else {
			fmt.Printf("\r\x1b[Kdownloading %s... %d MiB (%.2f MiB/s)",
				src, currentSize/(1024*1024), mibPerSec)
		}
		if complete {
			fmt.Println()
		}
	}
}
