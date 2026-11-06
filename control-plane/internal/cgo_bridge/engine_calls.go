// Package cgo_bridge contains the only Go-to-native boundary for the engine.
package cgo_bridge

/*
#cgo linux CFLAGS: -I${SRCDIR}/../../../ffi-bridge
#cgo linux LDFLAGS: -L${SRCDIR}/../../../core-engine/rust_trap/target/release -L${SRCDIR}/../../../core-engine/rust_trap/target/debug -laabhas
#include "rust_exports.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// StatusError reports a non-zero native engine status.
type StatusError struct {
	Code int
}

func (e StatusError) Error() string {
	return fmt.Sprintf("native engine returned status %d", e.Code)
}

var nativeCallMu sync.Mutex

// Start validates and starts one native model instance.
// The model path is copied into C-owned memory for the duration of the call.
func Start(modelPath string) error {
	if modelPath == "" {
		return StatusError{Code: int(C.AABHAS_STATUS_INVALID_ARGUMENT)}
	}
	for i := 0; i < len(modelPath); i++ {
		if modelPath[i] == 0 {
			return StatusError{Code: int(C.AABHAS_STATUS_INVALID_ARGUMENT)}
		}
	}

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	nativeCallMu.Lock()
	defer nativeCallMu.Unlock()
	status := int(C.start_aabhas_engine(cPath))
	if status != int(C.AABHAS_STATUS_OK) {
		return StatusError{Code: status}
	}
	return nil
}

// Stop releases the active native engine instance.
func Stop() error {
	nativeCallMu.Lock()
	defer nativeCallMu.Unlock()
	status := int(C.stop_aabhas_engine())
	if status != int(C.AABHAS_STATUS_OK) && status != int(C.AABHAS_STATUS_NOT_STARTED) {
		return StatusError{Code: status}
	}
	return nil
}
