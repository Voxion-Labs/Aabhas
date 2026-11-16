//go:build !linux

package cgo_bridge

import (
	"fmt"
	"sync"
)

// StatusError reports a non-zero native engine status.
type StatusError struct {
	Code int
}

func (e StatusError) Error() string {
	return fmt.Sprintf("native engine returned status %d", e.Code)
}

var (
	nativeCallMu sync.Mutex
	stubStarted  bool
)

const maxModelPathBytes = 4096

// Start simulates starting the native engine on non-Linux development environments.
func Start(modelPath string) error {
	if modelPath == "" || len(modelPath) > maxModelPathBytes {
		return StatusError{Code: -1}
	}
	for i := 0; i < len(modelPath); i++ {
		if modelPath[i] == 0 {
			return StatusError{Code: -1}
		}
	}

	nativeCallMu.Lock()
	defer nativeCallMu.Unlock()
	if stubStarted {
		return StatusError{Code: -2}
	}
	stubStarted = true
	return nil
}

// Stop simulates releasing the native engine on non-Linux development environments.
func Stop() error {
	nativeCallMu.Lock()
	defer nativeCallMu.Unlock()
	stubStarted = false
	return nil
}
