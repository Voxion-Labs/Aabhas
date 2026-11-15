package orchestrator

import (
	"errors"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrMultiModelPoolExhausted = errors.New("multi-model pool capacity exhausted; all loaded slots are actively leased")
	ErrMultiModelNotFound      = errors.New("model not registered in pool")
	ErrMultiModelInvalidID     = errors.New("model ID cannot be empty")
	ErrMultiModelInvalidPath   = errors.New("model path cannot be empty")
	ErrMultiModelLeaseReleased = errors.New("multi-model lease has already been released")
)

// MultiModelConfig configures bounded multi-model scheduling.
type MultiModelConfig struct {
	MaxLoadedModels int   `json:"max_loaded_models"`
	MaxMemoryBytes  int64 `json:"max_memory_bytes"`
}

// MultiModelStats reports point-in-time multi-model pool metrics.
type MultiModelStats struct {
	TotalRegistered int    `json:"total_registered"`
	LoadedModels    int    `json:"loaded_models"`
	ActiveLeases    int    `json:"active_leases"`
	TotalAcquires   uint64 `json:"total_acquires"`
	TotalEvictions  uint64 `json:"total_evictions"`
	MemoryUsedBytes int64  `json:"memory_used_bytes"`
}

// MultiModelRegistration holds model configuration for pool scheduling.
type MultiModelRegistration struct {
	ModelID     string `json:"model_id"`
	ModelPath   string `json:"model_path"`
	MemoryBytes int64  `json:"memory_bytes"`
}

type modelSlot struct {
	registration MultiModelRegistration
	activeLeases int
	lastAccessed time.Time
}

// MultiModelScheduler coordinates concurrent model slot leasing with LRU eviction.
type MultiModelScheduler struct {
	mu             sync.Mutex
	cfg            MultiModelConfig
	registered     map[string]MultiModelRegistration
	loaded         map[string]*modelSlot
	totalAcquires  atomic.Uint64
	totalEvictions atomic.Uint64
}

// MultiModelLease represents an active lease on a model slot in the scheduler.
type MultiModelLease struct {
	scheduler *MultiModelScheduler
	modelID   string
	once      sync.Once
	released  atomic.Bool
}

// NewMultiModelScheduler constructs a new scheduler with the given constraints.
func NewMultiModelScheduler(cfg MultiModelConfig) *MultiModelScheduler {
	if cfg.MaxLoadedModels <= 0 {
		cfg.MaxLoadedModels = 4
	}
	if cfg.MaxMemoryBytes <= 0 {
		cfg.MaxMemoryBytes = 4 * 1024 * 1024 * 1024 // 4 GB default
	}
	return &MultiModelScheduler{
		cfg:        cfg,
		registered: make(map[string]MultiModelRegistration),
		loaded:     make(map[string]*modelSlot),
	}
}

// Register registers a model specification with the scheduler.
func (s *MultiModelScheduler) Register(reg MultiModelRegistration) error {
	id := strings.TrimSpace(reg.ModelID)
	if id == "" {
		return ErrMultiModelInvalidID
	}
	path := strings.TrimSpace(reg.ModelPath)
	if path == "" {
		return ErrMultiModelInvalidPath
	}
	reg.ModelID = id
	reg.ModelPath = path
	if reg.MemoryBytes <= 0 {
		reg.MemoryBytes = 64 * 1024 * 1024 // 64 MB default baseline
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.registered[id]; exists {
		return errors.New("model already registered: " + id)
	}
	s.registered[id] = reg
	return nil
}

// Unregister removes a model from the scheduler. Fails if model is currently loaded with active leases.
func (s *MultiModelScheduler) Unregister(modelID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	slot, loaded := s.loaded[modelID]
	if loaded && slot.activeLeases > 0 {
		return errors.New("cannot unregister model with active leases: " + modelID)
	}

	if loaded {
		delete(s.loaded, modelID)
	}
	if _, exists := s.registered[modelID]; !exists {
		return ErrMultiModelNotFound
	}
	delete(s.registered, modelID)
	return nil
}

// Acquire acquires a lease for a registered model, loading it into a slot and evicting
// the least recently used idle model if necessary.
func (s *MultiModelScheduler) Acquire(modelID string) (*MultiModelLease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	reg, exists := s.registered[modelID]
	if !exists {
		return nil, ErrMultiModelNotFound
	}

	slot, loaded := s.loaded[modelID]
	if loaded {
		slot.activeLeases++
		slot.lastAccessed = time.Now().UTC()
		s.totalAcquires.Add(1)
		return &MultiModelLease{
			scheduler: s,
			modelID:   modelID,
		}, nil
	}

	// Evict idle models if capacity or memory is constrained
	for {
		currentMemory := int64(0)
		for _, sl := range s.loaded {
			currentMemory += sl.registration.MemoryBytes
		}

		if len(s.loaded) < s.cfg.MaxLoadedModels && (currentMemory+reg.MemoryBytes) <= s.cfg.MaxMemoryBytes {
			break
		}

		// Find least recently used idle model (activeLeases == 0)
		var oldestID string
		var oldestTime time.Time

		for id, sl := range s.loaded {
			if sl.activeLeases == 0 {
				if oldestID == "" || sl.lastAccessed.Before(oldestTime) {
					oldestID = id
					oldestTime = sl.lastAccessed
				}
			}
		}

		if oldestID == "" {
			return nil, ErrMultiModelPoolExhausted
		}

		delete(s.loaded, oldestID)
		s.totalEvictions.Add(1)
	}

	// Load model into slot
	newSlot := &modelSlot{
		registration: reg,
		activeLeases: 1,
		lastAccessed: time.Now().UTC(),
	}
	s.loaded[modelID] = newSlot
	s.totalAcquires.Add(1)

	return &MultiModelLease{
		scheduler: s,
		modelID:   modelID,
	}, nil
}

// Release relinquishes the lease on the model slot.
func (l *MultiModelLease) Release() error {
	if l == nil || l.scheduler == nil {
		return ErrMultiModelLeaseReleased
	}
	if l.released.Swap(true) {
		return ErrMultiModelLeaseReleased
	}

	l.once.Do(func() {
		l.scheduler.mu.Lock()
		defer l.scheduler.mu.Unlock()

		if slot, ok := l.scheduler.loaded[l.modelID]; ok {
			if slot.activeLeases > 0 {
				slot.activeLeases--
			}
			slot.lastAccessed = time.Now().UTC()
		}
	})
	return nil
}

// ModelID returns the model ID associated with this lease.
func (l *MultiModelLease) ModelID() string {
	if l == nil {
		return ""
	}
	return l.modelID
}

// Evict unloads an idle model from memory.
func (s *MultiModelScheduler) Evict(modelID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	slot, loaded := s.loaded[modelID]
	if !loaded {
		return nil
	}
	if slot.activeLeases > 0 {
		return errors.New("cannot evict actively leased model: " + modelID)
	}
	delete(s.loaded, modelID)
	s.totalEvictions.Add(1)
	return nil
}

// IsLoaded checks whether a model is currently loaded in memory.
func (s *MultiModelScheduler) IsLoaded(modelID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, loaded := s.loaded[modelID]
	return loaded
}

// Stats returns point-in-time statistics for the multi-model scheduler.
func (s *MultiModelScheduler) Stats() MultiModelStats {
	s.mu.Lock()
	defer s.mu.Unlock()

	activeLeases := 0
	memoryUsed := int64(0)
	for _, slot := range s.loaded {
		activeLeases += slot.activeLeases
		memoryUsed += slot.registration.MemoryBytes
	}

	return MultiModelStats{
		TotalRegistered: len(s.registered),
		LoadedModels:    len(s.loaded),
		ActiveLeases:    activeLeases,
		TotalAcquires:   s.totalAcquires.Load(),
		TotalEvictions:  s.totalEvictions.Load(),
		MemoryUsedBytes: memoryUsed,
	}
}

// ListRegistered returns a sorted slice of registered model IDs.
func (s *MultiModelScheduler) ListRegistered() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	res := make([]string, 0, len(s.registered))
	for id := range s.registered {
		res = append(res, id)
	}
	sort.Strings(res)
	return res
}
