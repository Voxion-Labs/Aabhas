package orchestrator

import "testing"

func TestExecutionLeaseCancelDoesNotCountStop(t *testing.T) {
	controller := NewExecutionController()
	lease, err := controller.Acquire("model.bin")
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.Cancel(); err != nil {
		t.Fatal(err)
	}
	stats := controller.Stats()
	if stats.Active || stats.Stops != 0 || stats.Starts != 1 {
		t.Fatalf("unexpected stats after cancel: %+v", stats)
	}
}

func TestExecutionControllerRejectsConcurrentStart(t *testing.T) {
	controller := NewExecutionController()
	lease, err := controller.Acquire("first.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if _, err := controller.Acquire("second.bin"); err != ErrEngineBusy {
		t.Fatalf("busy error = %v, want %v", err, ErrEngineBusy)
	}
}
