package orchestrator

import (
	"testing"
	"time"
)

func TestMultiModelRegistration(t *testing.T) {
	sched := NewMultiModelScheduler(MultiModelConfig{
		MaxLoadedModels: 2,
		MaxMemoryBytes:  1024 * 1024 * 1024,
	})

	err := sched.Register(MultiModelRegistration{
		ModelID:     "model-a",
		ModelPath:   "/models/a.gguf",
		MemoryBytes: 100 * 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("unexpected register error: %v", err)
	}

	// Duplicate registration must fail
	if err := sched.Register(MultiModelRegistration{ModelID: "model-a", ModelPath: "/models/a.gguf"}); err == nil {
		t.Fatalf("expected duplicate register error")
	}

	// Empty ID must fail
	if err := sched.Register(MultiModelRegistration{ModelID: "", ModelPath: "/models/x.gguf"}); err != ErrMultiModelInvalidID {
		t.Fatalf("expected invalid id error, got %v", err)
	}

	stats := sched.Stats()
	if stats.TotalRegistered != 1 || stats.LoadedModels != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	if err := sched.Unregister("model-a"); err != nil {
		t.Fatalf("unexpected unregister error: %v", err)
	}
	if sched.Stats().TotalRegistered != 0 {
		t.Fatalf("expected 0 registered models")
	}
}

func TestMultiModelAcquireAndRelease(t *testing.T) {
	sched := NewMultiModelScheduler(MultiModelConfig{
		MaxLoadedModels: 2,
		MaxMemoryBytes:  1024 * 1024 * 1024,
	})

	_ = sched.Register(MultiModelRegistration{
		ModelID:     "model-1",
		ModelPath:   "/models/m1.gguf",
		MemoryBytes: 200 * 1024 * 1024,
	})

	lease, err := sched.Acquire("model-1")
	if err != nil {
		t.Fatalf("unexpected acquire error: %v", err)
	}
	if lease.ModelID() != "model-1" {
		t.Fatalf("unexpected model id: %s", lease.ModelID())
	}

	if !sched.IsLoaded("model-1") {
		t.Fatalf("expected model-1 to be loaded")
	}

	stats := sched.Stats()
	if stats.ActiveLeases != 1 || stats.LoadedModels != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	// Release lease
	if err := lease.Release(); err != nil {
		t.Fatalf("unexpected release error: %v", err)
	}
	// Releasing twice returns ErrMultiModelLeaseReleased
	if err := lease.Release(); err != ErrMultiModelLeaseReleased {
		t.Fatalf("expected ErrMultiModelLeaseReleased, got %v", err)
	}

	stats = sched.Stats()
	if stats.ActiveLeases != 0 || stats.LoadedModels != 1 {
		t.Fatalf("expected 0 active leases, model remains loaded: %+v", stats)
	}
}

func TestMultiModelLRUEviction(t *testing.T) {
	// Bounded capacity of 2 models
	sched := NewMultiModelScheduler(MultiModelConfig{
		MaxLoadedModels: 2,
		MaxMemoryBytes:  1024 * 1024 * 1024,
	})

	_ = sched.Register(MultiModelRegistration{ModelID: "m1", ModelPath: "/m1", MemoryBytes: 100})
	_ = sched.Register(MultiModelRegistration{ModelID: "m2", ModelPath: "/m2", MemoryBytes: 100})
	_ = sched.Register(MultiModelRegistration{ModelID: "m3", ModelPath: "/m3", MemoryBytes: 100})

	l1, err := sched.Acquire("m1")
	if err != nil {
		t.Fatal(err)
	}
	_ = l1.Release()
	time.Sleep(5 * time.Millisecond)

	l2, err := sched.Acquire("m2")
	if err != nil {
		t.Fatal(err)
	}
	_ = l2.Release()
	time.Sleep(5 * time.Millisecond)

	// Re-acquire m2 so m1 is the LRU
	l2_again, err := sched.Acquire("m2")
	if err != nil {
		t.Fatal(err)
	}
	_ = l2_again.Release()

	// Acquire m3: should evict m1
	l3, err := sched.Acquire("m3")
	if err != nil {
		t.Fatal(err)
	}
	_ = l3.Release()

	if sched.IsLoaded("m1") {
		t.Fatalf("expected m1 to be evicted")
	}
	if !sched.IsLoaded("m2") || !sched.IsLoaded("m3") {
		t.Fatalf("expected m2 and m3 to be loaded")
	}

	stats := sched.Stats()
	if stats.TotalEvictions != 1 || stats.LoadedModels != 2 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestMultiModelActiveLeasePreventsEviction(t *testing.T) {
	// Bounded capacity of 1 model
	sched := NewMultiModelScheduler(MultiModelConfig{
		MaxLoadedModels: 1,
		MaxMemoryBytes:  1024 * 1024 * 1024,
	})

	_ = sched.Register(MultiModelRegistration{ModelID: "m1", ModelPath: "/m1", MemoryBytes: 100})
	_ = sched.Register(MultiModelRegistration{ModelID: "m2", ModelPath: "/m2", MemoryBytes: 100})

	l1, err := sched.Acquire("m1")
	if err != nil {
		t.Fatal(err)
	}

	// Try to acquire m2 while m1 is actively held: should return ErrMultiModelPoolExhausted
	_, err = sched.Acquire("m2")
	if err != ErrMultiModelPoolExhausted {
		t.Fatalf("expected ErrMultiModelPoolExhausted, got %v", err)
	}

	// Release l1
	_ = l1.Release()

	// Now acquiring m2 should succeed, evicting m1
	l2, err := sched.Acquire("m2")
	if err != nil {
		t.Fatalf("unexpected acquire error: %v", err)
	}
	defer l2.Release()

	if sched.IsLoaded("m1") {
		t.Fatalf("expected m1 to be evicted")
	}
	if !sched.IsLoaded("m2") {
		t.Fatalf("expected m2 to be loaded")
	}
}
