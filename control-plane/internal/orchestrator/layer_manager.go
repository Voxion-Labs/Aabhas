package orchestrator

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// LayerState is the control-plane state of one tensor layer.
type LayerState uint8

const (
	LayerQueued LayerState = iota
	LayerStreaming
	LayerLoaded
	LayerDropped
	LayerFailed
)

func (s LayerState) String() string {
	switch s {
	case LayerQueued:
		return "queued"
	case LayerStreaming:
		return "streaming"
	case LayerLoaded:
		return "loaded"
	case LayerDropped:
		return "dropped"
	case LayerFailed:
		return "failed"
	default:
		return "unknown"
	}
}

var (
	ErrNoLayerAvailable  = errors.New("no queued tensor layer is available")
	ErrLayerNotFound     = errors.New("tensor layer was not found")
	ErrLayerBusy         = errors.New("tensor layer manager has an active layer")
	ErrInvalidTransition = errors.New("invalid tensor layer state transition")
	ErrInvalidLayer      = errors.New("invalid tensor layer descriptor")
)

// LayerSpec describes the byte range owned by a tensor layer.
type LayerSpec struct {
	ID         uint64
	FileOffset uint64
	ByteSize   uint64
}

// Layer is an immutable snapshot returned by LayerManager methods.
type Layer struct {
	Spec      LayerSpec
	State     LayerState
	Attempts  uint32
	LastError string
	UpdatedAt time.Time
}

type layerRecord struct{ layer Layer }

// LayerManager serializes layer ownership and keeps the queue bounded to the
// caller-provided layer descriptors. It does not copy tensor payloads.
type LayerManager struct {
	mu        sync.Mutex
	layers    map[uint64]*layerRecord
	order     []uint64
	queue     []uint64
	queueHead int
	activeID  uint64
	active    bool
}

// NewLayerManager constructs a manager with every descriptor initially queued.
func NewLayerManager(specs []LayerSpec) (*LayerManager, error) {
	m := &LayerManager{
		layers: make(map[uint64]*layerRecord, len(specs)),
		order:  make([]uint64, 0, len(specs)),
		queue:  make([]uint64, 0, len(specs)),
	}
	for _, spec := range specs {
		if spec.ByteSize == 0 || spec.FileOffset > math.MaxUint64-spec.ByteSize {
			return nil, fmt.Errorf("layer %d: %w", spec.ID, ErrInvalidLayer)
		}
		if _, exists := m.layers[spec.ID]; exists {
			return nil, fmt.Errorf("layer %d: duplicate descriptor: %w", spec.ID, ErrInvalidLayer)
		}
		m.layers[spec.ID] = &layerRecord{layer: Layer{
			Spec:      spec,
			State:     LayerQueued,
			UpdatedAt: time.Now().UTC(),
		}}
		m.order = append(m.order, spec.ID)
		m.queue = append(m.queue, spec.ID)
	}
	return m, nil
}

// Enqueue adds a layer for streaming. A loaded or streaming layer cannot be
// queued again until it has been explicitly dropped.
func (m *LayerManager) Enqueue(id uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	record, ok := m.layers[id]
	if !ok {
		return ErrLayerNotFound
	}
	switch record.layer.State {
	case LayerQueued:
		return nil
	case LayerDropped, LayerFailed:
		record.layer.State = LayerQueued
		record.layer.LastError = ""
		record.layer.UpdatedAt = time.Now().UTC()
		m.queue = append(m.queue, id)
		return nil
	default:
		return ErrInvalidTransition
	}
}

// Queue is an alias with terminology convenient for callers managing work.
func (m *LayerManager) Queue(id uint64) error { return m.Enqueue(id) }

// Next transfers the oldest queued layer into the streaming state.
func (m *LayerManager) Next() (Layer, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.active {
		return Layer{}, ErrLayerBusy
	}
	for m.queueHead < len(m.queue) {
		id := m.queue[m.queueHead]
		m.queueHead++
		record, ok := m.layers[id]
		if !ok || record.layer.State != LayerQueued {
			continue
		}
		record.layer.State = LayerStreaming
		record.layer.Attempts++
		record.layer.UpdatedAt = time.Now().UTC()
		m.activeID = id
		m.active = true
		m.compactQueueLocked()
		return record.layer, nil
	}
	m.compactQueueLocked()
	return Layer{}, ErrNoLayerAvailable
}

// NextLayer is an explicit alias for Next.
func (m *LayerManager) NextLayer() (Layer, error) { return m.Next() }

// MarkLoaded completes the active layer and releases manager ownership.
func (m *LayerManager) MarkLoaded(id uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.active || m.activeID != id {
		return ErrInvalidTransition
	}
	record := m.layers[id]
	record.layer.State = LayerLoaded
	record.layer.LastError = ""
	record.layer.UpdatedAt = time.Now().UTC()
	m.active = false
	return nil
}

// Drop marks a loaded or active layer as releasable by the native data plane.
func (m *LayerManager) Drop(id uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	record, ok := m.layers[id]
	if !ok {
		return ErrLayerNotFound
	}
	if record.layer.State != LayerLoaded &&
		(!m.active || m.activeID != id || record.layer.State != LayerStreaming) {
		return ErrInvalidTransition
	}
	record.layer.State = LayerDropped
	record.layer.LastError = ""
	record.layer.UpdatedAt = time.Now().UTC()
	if m.active && m.activeID == id {
		m.active = false
	}
	return nil
}

// MarkFailed records a terminal attempt failure and releases active ownership.
func (m *LayerManager) MarkFailed(id uint64, cause error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	record, ok := m.layers[id]
	if !ok {
		return ErrLayerNotFound
	}
	if !m.active || m.activeID != id || record.layer.State != LayerStreaming {
		return ErrInvalidTransition
	}
	record.layer.State = LayerFailed
	if cause != nil {
		record.layer.LastError = cause.Error()
	} else {
		record.layer.LastError = "layer streaming failed"
	}
	record.layer.UpdatedAt = time.Now().UTC()
	m.active = false
	return nil
}

// Get returns a point-in-time copy of a layer descriptor and its state.
func (m *LayerManager) Get(id uint64) (Layer, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	record, ok := m.layers[id]
	if !ok {
		return Layer{}, false
	}
	return record.layer, true
}

// Snapshot returns layers in their original descriptor order.
func (m *LayerManager) Snapshot() []Layer {
	m.mu.Lock()
	defer m.mu.Unlock()
	snapshot := make([]Layer, 0, len(m.order))
	for _, id := range m.order {
		if record, ok := m.layers[id]; ok {
			snapshot = append(snapshot, record.layer)
		}
	}
	return snapshot
}

// Pending returns the queued layer count at the instant of the call.
func (m *LayerManager) Pending() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.queue) - m.queueHead
}

func (m *LayerManager) compactQueueLocked() {
	if m.queueHead == 0 {
		return
	}
	if m.queueHead == len(m.queue) {
		m.queue = m.queue[:0]
		m.queueHead = 0
		return
	}
	if m.queueHead >= 64 && m.queueHead*2 >= len(m.queue) {
		copy(m.queue, m.queue[m.queueHead:])
		m.queue = m.queue[:len(m.queue)-m.queueHead]
		m.queueHead = 0
	}
}
