package orchestrator

import (
	"errors"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	ErrEngineBusy       = errors.New("an Aabhas model is already running")
	ErrEngineNotRunning = errors.New("no Aabhas model is running")
	ErrEmptyModelPath   = errors.New("model path is empty")
	ErrLeaseReleased    = errors.New("execution lease has already been released")
)

// ExecutionStats is a point-in-time view of the single-model gate.
type ExecutionStats struct {
	Active         bool
	ModelPath      string
	Generation     uint64
	Starts         uint64
	Stops          uint64
	RejectedStarts uint64
}

// ExecutionController enforces one active native model per daemon process.
type ExecutionController struct {
	mu             sync.Mutex
	active         atomic.Bool
	starts         atomic.Uint64
	stops          atomic.Uint64
	rejectedStarts atomic.Uint64
	generation     atomic.Uint64
	modelPath      string
}

// ExecutionLease represents ownership of the native execution slot.
type ExecutionLease struct {
	controller *ExecutionController
	once       sync.Once
	released   atomic.Bool
}

// NewExecutionController constructs an idle single-model controller.
func NewExecutionController() *ExecutionController { return &ExecutionController{} }

// Acquire admits exactly one non-empty model path and returns its release
// lease. The lease is idempotent and safe to release from a defer.
func (c *ExecutionController) Acquire(modelPath string) (*ExecutionLease, error) {
	if strings.TrimSpace(modelPath) == "" || strings.IndexByte(modelPath, 0) >= 0 {
		return nil, ErrEmptyModelPath
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.active.Load() {
		c.rejectedStarts.Add(1)
		return nil, ErrEngineBusy
	}
	c.modelPath = modelPath
	c.active.Store(true)
	c.starts.Add(1)
	c.generation.Add(1)
	return &ExecutionLease{controller: c}, nil
}

// Release relinquishes the native execution slot. Releasing twice is safe and
// returns ErrLeaseReleased only on the second and later calls.
func (l *ExecutionLease) Release() error {
	if l == nil || l.controller == nil {
		return ErrLeaseReleased
	}
	if l.released.Swap(true) {
		return ErrLeaseReleased
	}
	l.once.Do(func() {
		l.controller.mu.Lock()
		l.controller.modelPath = ""
		l.controller.active.Store(false)
		l.controller.stops.Add(1)
		l.controller.mu.Unlock()
	})
	return nil
}

// Cancel relinquishes a lease after native start failed without counting a
// stop for an engine that never became active.
func (l *ExecutionLease) Cancel() error {
	if l == nil || l.controller == nil {
		return ErrLeaseReleased
	}
	if l.released.Swap(true) {
		return ErrLeaseReleased
	}
	l.once.Do(func() {
		l.controller.mu.Lock()
		l.controller.modelPath = ""
		l.controller.active.Store(false)
		l.controller.mu.Unlock()
	})
	return nil
}

// Release stops the currently active model when the caller does not retain a
// lease. It is intended for shutdown paths and returns an idle error.
func (c *ExecutionController) Release() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active.Load() {
		return ErrEngineNotRunning
	}
	c.modelPath = ""
	c.active.Store(false)
	c.stops.Add(1)
	return nil
}

// Stats returns a consistent ownership snapshot and lock-free counters.
func (c *ExecutionController) Stats() ExecutionStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return ExecutionStats{
		Active:         c.active.Load(),
		ModelPath:      c.modelPath,
		Generation:     c.generation.Load(),
		Starts:         c.starts.Load(),
		Stops:          c.stops.Load(),
		RejectedStarts: c.rejectedStarts.Load(),
	}
}

// Active reports whether a model currently owns the execution slot.
func (c *ExecutionController) Active() bool { return c.active.Load() }
