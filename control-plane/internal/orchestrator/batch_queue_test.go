package orchestrator

import (
	"context"
	"testing"
	"time"
)

func TestBatchQueueSubmitAndCollect(t *testing.T) {
	cfg := BatchQueueConfig{
		MaxBatchSize: 2,
		MaxQueueSize: 10,
		BatchTimeout: 50 * time.Millisecond,
	}
	bq := NewBatchQueue(cfg)
	defer bq.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	// Submit 3 requests
	resCh1, id1, err := bq.Submit(ctx, "model-a", []uint32{1, 2}, 4)
	if err != nil {
		t.Fatalf("unexpected submit error: %v", err)
	}
	resCh2, id2, err := bq.Submit(ctx, "model-a", []uint32{3}, 4)
	if err != nil {
		t.Fatalf("unexpected submit error: %v", err)
	}
	resCh3, id3, err := bq.Submit(ctx, "model-a", []uint32{4, 5, 6}, 4)
	if err != nil {
		t.Fatalf("unexpected submit error: %v", err)
	}

	if id1 == 0 || id2 == 0 || id3 == 0 {
		t.Fatalf("expected non-zero IDs")
	}

	// First batch should collect max 2 requests
	batch1, err := bq.CollectBatch(ctx)
	if err != nil {
		t.Fatalf("unexpected collect batch error: %v", err)
	}
	if len(batch1) != 2 {
		t.Fatalf("expected batch size 2, got %d", len(batch1))
	}
	if batch1[0].RequestID != id1 || batch1[1].RequestID != id2 {
		t.Fatalf("unexpected request IDs in batch 1")
	}

	// Reply to batch 1
	batch1[0].ResultCh <- QueuedResponse{RequestID: id1, GeneratedTokens: []uint32{10, 11}}
	batch1[1].ResultCh <- QueuedResponse{RequestID: id2, GeneratedTokens: []uint32{12, 13}}

	resp1 := <-resCh1
	resp2 := <-resCh2
	if len(resp1.GeneratedTokens) != 2 || len(resp2.GeneratedTokens) != 2 {
		t.Fatalf("unexpected response payloads")
	}

	// Second batch should collect remaining 1 request
	batch2, err := bq.CollectBatch(ctx)
	if err != nil {
		t.Fatalf("unexpected collect batch 2 error: %v", err)
	}
	if len(batch2) != 1 || batch2[0].RequestID != id3 {
		t.Fatalf("expected request id3 in batch 2")
	}

	batch2[0].ResultCh <- QueuedResponse{RequestID: id3, GeneratedTokens: []uint32{99}}
	resp3 := <-resCh3
	if len(resp3.GeneratedTokens) != 1 || resp3.GeneratedTokens[0] != 99 {
		t.Fatalf("unexpected response for request 3")
	}

	stats := bq.Stats()
	if stats.TotalSubmitted != 3 || stats.TotalBatches != 2 || stats.QueuedRequests != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestBatchQueueCloseDrainsPending(t *testing.T) {
	cfg := BatchQueueConfig{
		MaxBatchSize: 5,
		MaxQueueSize: 5,
		BatchTimeout: 10 * time.Millisecond,
	}
	bq := NewBatchQueue(cfg)

	ctx := context.Background()
	resCh, _, err := bq.Submit(ctx, "model-x", []uint32{100}, 2)
	if err != nil {
		t.Fatalf("unexpected submit error: %v", err)
	}

	// Close queue
	bq.Close()

	// Pending request should receive ErrBatchQueueClosed
	select {
	case resp, ok := <-resCh:
		if !ok || resp.Err != ErrBatchQueueClosed {
			t.Fatalf("expected ErrBatchQueueClosed, got %+v", resp)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("timeout waiting for closed response")
	}

	// Further submit should fail
	_, _, err = bq.Submit(ctx, "model-x", []uint32{100}, 2)
	if err != ErrBatchQueueClosed {
		t.Fatalf("expected ErrBatchQueueClosed, got %v", err)
	}
}
