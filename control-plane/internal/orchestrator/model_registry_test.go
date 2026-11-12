package orchestrator

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelRegistryRegisterAndList(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(path, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}

	registry := NewModelRegistry()
	record, err := registry.Register(path)
	if err != nil {
		t.Fatal(err)
	}
	if record.SizeBytes != int64(len("model")) {
		t.Fatalf("size = %d, want %d", record.SizeBytes, len("model"))
	}
	if len(registry.List()) != 1 {
		t.Fatalf("registry length = %d, want 1", len(registry.List()))
	}
	if _, ok := registry.Get(path); !ok {
		t.Fatal("registered model was not found")
	}
}

func TestModelRegistryRejectsEmptyAndDirectory(t *testing.T) {
	t.Parallel()
	registry := NewModelRegistry()
	if _, err := registry.Register(""); err != ErrModelInvalid {
		t.Fatalf("empty path error = %v, want %v", err, ErrModelInvalid)
	}
	if _, err := registry.Register(t.TempDir()); err != ErrModelInvalid {
		t.Fatalf("directory error = %v, want %v", err, ErrModelInvalid)
	}
}
