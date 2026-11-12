package orchestrator

import (
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrModelNotFound = errors.New("model is not registered")
	ErrModelInvalid  = errors.New("model path is not a readable non-empty file")
)

// ModelRecord is metadata about a model path known to the control plane.
type ModelRecord struct {
	ModelPath  string    `json:"model_path"`
	SizeBytes  int64     `json:"size_bytes"`
	Registered time.Time `json:"registered_at"`
	Validated  time.Time `json:"validated_at"`
}

// ModelRegistry stores model metadata only; it never copies model payloads.
type ModelRegistry struct {
	mu     sync.RWMutex
	models map[string]ModelRecord
}

// NewModelRegistry constructs an empty registry.
func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{models: make(map[string]ModelRecord)}
}

// Register validates and records a regular, non-empty model file.
func (r *ModelRegistry) Register(modelPath string) (ModelRecord, error) {
	cleaned := filepath.Clean(strings.TrimSpace(modelPath))
	if cleaned == "." || strings.IndexByte(cleaned, 0) >= 0 {
		return ModelRecord{}, ErrModelInvalid
	}
	metadata, err := os.Stat(cleaned)
	if err != nil || !metadata.Mode().IsRegular() || metadata.Size() <= 0 {
		return ModelRecord{}, ErrModelInvalid
	}
	now := time.Now().UTC()
	record := ModelRecord{
		ModelPath:  cleaned,
		SizeBytes:  metadata.Size(),
		Registered: now,
		Validated:  now,
	}
	r.mu.Lock()
	r.models[cleaned] = record
	r.mu.Unlock()
	return record, nil
}

// Get returns a point-in-time record for a registered model.
func (r *ModelRegistry) Get(modelPath string) (ModelRecord, bool) {
	cleaned := filepath.Clean(strings.TrimSpace(modelPath))
	r.mu.RLock()
	record, ok := r.models[cleaned]
	r.mu.RUnlock()
	return record, ok
}

// List returns records sorted by model path for deterministic API responses.
func (r *ModelRegistry) List() []ModelRecord {
	r.mu.RLock()
	result := make([]ModelRecord, 0, len(r.models))
	for _, record := range r.models {
		result = append(result, record)
	}
	r.mu.RUnlock()
	sort.Slice(result, func(i, j int) bool { return result[i].ModelPath < result[j].ModelPath })
	return result
}
