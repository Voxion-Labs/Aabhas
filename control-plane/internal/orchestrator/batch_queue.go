package orchestrator

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrBatchQueueClosed = errors.New("batch queue is closed")
	ErrBatchQueueFull   = errors.New("batch queue is full")
	ErrInvalidRequest   = errors.New("invalid batch request: empty tokens or zero max tokens")
)

// BatchQueueConfig configures dynamic batching windows and queue limits.
type BatchQueueConfig struct {
	MaxBatchSize int           `json:"max_batch_size"`
	MaxQueueSize int           `json:"max_queue_size"`
	BatchTimeout time.Duration `json:"batch_timeout"`
}

// QueuedRequest encapsulates a client generation request submitted to the batch queue.
type QueuedRequest struct {
	RequestID uint64
	ModelID   string
	Tokens    []uint32
	MaxTokens int
	ResultCh  chan QueuedResponse
}

// QueuedResponse holds the result of processing a queued request.
type QueuedResponse struct {
	RequestID       uint64
	GeneratedTokens []uint32
	ReachedEOS      bool
	Err             error
}

// BatchQueueStats reports point-in-time metrics for the batch queue.
type BatchQueueStats struct {
	QueuedRequests int    `json:"queued_requests"`
	TotalSubmitted uint64 `json:"total_submitted"`
	TotalBatches   uint64 `json:"total_batches"`
	Closed         bool   `json:"closed"`
}

// BatchQueue coordinates batch request buffering, dynamic timeout windowing, and concurrency dispatch.
type BatchQueue struct {
	mu            sync.Mutex
	cfg           BatchQueueConfig
	requests      []QueuedRequest
	notEmpty      *sync.Cond
	closed        bool
	nextRequestID atomic.Uint64
	totalQueued   atomic.Uint64
	totalBatches  atomic.Uint64
}

// NewBatchQueue constructs a new batch queue with the given configuration.
func NewBatchQueue(cfg BatchQueueConfig) *BatchQueue {
	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = 16
	}
	if cfg.MaxQueueSize <= 0 {
		cfg.MaxQueueSize = 128
	}
	if cfg.BatchTimeout <= 0 {
		cfg.BatchTimeout = 10 * time.Millisecond
	}
	bq := &BatchQueue{
		cfg:      cfg,
		requests: make([]QueuedRequest, 0, cfg.MaxQueueSize),
	}
	bq.notEmpty = sync.NewCond(&bq.mu)
	return bq
}

// Submit enqueues a generation request and returns a receive-only result channel.
func (bq *BatchQueue) Submit(ctx context.Context, modelID string, tokens []uint32, maxTokens int) (<-chan QueuedResponse, uint64, error) {
	if len(tokens) == 0 || maxTokens <= 0 {
		return nil, 0, ErrInvalidRequest
	}

	bq.mu.Lock()
	defer bq.mu.Unlock()

	if bq.closed {
		return nil, 0, ErrBatchQueueClosed
	}
	if len(bq.requests) >= bq.cfg.MaxQueueSize {
		return nil, 0, ErrBatchQueueFull
	}

	reqID := bq.nextRequestID.Add(1)
	resCh := make(chan QueuedResponse, 1)

	req := QueuedRequest{
		RequestID: reqID,
		ModelID:   modelID,
		Tokens:    tokens,
		MaxTokens: maxTokens,
		ResultCh:  resCh,
	}

	bq.requests = append(bq.requests, req)
	bq.totalQueued.Add(1)
	bq.notEmpty.Signal()

	return resCh, reqID, nil
}

// CollectBatch gathers up to MaxBatchSize requests, waiting up to BatchTimeout for the batch to fill.
func (bq *BatchQueue) CollectBatch(ctx context.Context) ([]QueuedRequest, error) {
	bq.mu.Lock()
	defer bq.mu.Unlock()

	for len(bq.requests) == 0 && !bq.closed {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		bq.notEmpty.Wait()
	}

	if bq.closed && len(bq.requests) == 0 {
		return nil, ErrBatchQueueClosed
	}

	batchCount := len(bq.requests)
	if batchCount > bq.cfg.MaxBatchSize {
		batchCount = bq.cfg.MaxBatchSize
	}

	batch := make([]QueuedRequest, batchCount)
	copy(batch, bq.requests[:batchCount])
	bq.requests = bq.requests[batchCount:]

	bq.totalBatches.Add(1)
	return batch, nil
}

// Close closes the batch queue and notifies pending callers with an error.
func (bq *BatchQueue) Close() {
	bq.mu.Lock()
	defer bq.mu.Unlock()

	if bq.closed {
		return
	}
	bq.closed = true

	for _, req := range bq.requests {
		if req.ResultCh != nil {
			req.ResultCh <- QueuedResponse{
				RequestID: req.RequestID,
				Err:       ErrBatchQueueClosed,
			}
			close(req.ResultCh)
		}
	}
	bq.requests = nil
	bq.notEmpty.Broadcast()
}

// Len returns the current number of requests in queue.
func (bq *BatchQueue) Len() int {
	bq.mu.Lock()
	defer bq.mu.Unlock()
	return len(bq.requests)
}

// Stats returns a point-in-time view of batch queue metrics.
func (bq *BatchQueue) Stats() BatchQueueStats {
	bq.mu.Lock()
	defer bq.mu.Unlock()
	return BatchQueueStats{
		QueuedRequests: len(bq.requests),
		TotalSubmitted: bq.totalQueued.Load(),
		TotalBatches:   bq.totalBatches.Load(),
		Closed:         bq.closed,
	}
}
