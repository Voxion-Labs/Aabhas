package orchestrator

import (
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestModelIntegrityAndManifestSigning(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "synthetic-model.bin")

	// 1. Write synthetic model binary
	content := []byte("Aabhas Enterprise Model Binary v1.0.0 - Production Signed Manifest")
	if err := os.WriteFile(modelPath, content, 0644); err != nil {
		t.Fatalf("write synthetic model: %v", err)
	}

	// 2. Compute checksum
	checksum, size, err := ComputeFileSHA256(modelPath)
	if err != nil {
		t.Fatalf("compute checksum: %v", err)
	}
	if size != int64(len(content)) || checksum == "" {
		t.Fatalf("invalid checksum or size")
	}

	// 3. Verify clean model file
	if err := VerifyModelFile(modelPath, checksum); err != nil {
		t.Fatalf("verify model file: %v", err)
	}

	// 4. Verify tampering detection
	badChecksum := "0000000000000000000000000000000000000000000000000000000000000000"
	if err := VerifyModelFile(modelPath, badChecksum); err == nil {
		t.Fatalf("expected error for corrupted checksum, got nil")
	}

	// 5. Generate and sign manifest
	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	manifest, err := GenerateModelManifest("synthetic-model", modelPath)
	if err != nil {
		t.Fatalf("generate manifest: %v", err)
	}
	if err := SignModelManifest(manifest, privKey); err != nil {
		t.Fatalf("sign manifest: %v", err)
	}

	// 6. Verify signed manifest
	if err := VerifyModelManifest(manifest, pubKey, true); err != nil {
		t.Fatalf("verify signed manifest: %v", err)
	}

	// 7. Reject untrusted key
	untrustedPub, _, _ := ed25519.GenerateKey(rand.Reader)
	if err := VerifyModelManifest(manifest, untrustedPub, true); err != ErrManifestSignature {
		t.Fatalf("expected ErrManifestSignature, got %v", err)
	}

	// 8. Reject tampered manifest data
	tamperedManifest := *manifest
	tamperedManifest.ModelName = "tampered-model-name"
	if err := VerifyModelManifest(&tamperedManifest, pubKey, false); err != ErrManifestSignature {
		t.Fatalf("expected ErrManifestSignature for tampered data, got %v", err)
	}
}
