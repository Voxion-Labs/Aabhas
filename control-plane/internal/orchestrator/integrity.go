package orchestrator

import (
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

var (
	ErrChecksumMismatch   = errors.New("model file cryptographic SHA-256 checksum mismatch")
	ErrManifestSignature  = errors.New("model manifest cryptographic signature verification failed")
	ErrFileNotFound       = errors.New("model file not found or inaccessible")
	ErrInvalidManifest    = errors.New("invalid or incomplete model manifest")
)

// ModelManifest records cryptographic provenance and content verification digests.
type ModelManifest struct {
	ModelName     string    `json:"model_name"`
	FilePath      string    `json:"file_path,omitempty"`
	FileSHA256    string    `json:"file_sha256"`
	FileSizeBytes int64     `json:"file_size_bytes"`
	GeneratedAt   time.Time `json:"generated_at"`
	SignatureHex  string    `json:"signature_hex,omitempty"`
	PublicKeyHex  string    `json:"public_key_hex,omitempty"`
}

// ComputeFileSHA256 streams the file in 64 KiB chunks and returns the lowercase hex SHA-256 digest.
func ComputeFileSHA256(filepath string) (string, int64, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return "", 0, fmt.Errorf("%w: %v", ErrFileNotFound, err)
	}
	defer file.Close()

	hasher := sha256.New()
	buf := make([]byte, 64*1024)
	var totalBytes int64

	for {
		n, readErr := file.Read(buf)
		if n > 0 {
			hasher.Write(buf[:n])
			totalBytes += int64(n)
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return "", 0, fmt.Errorf("read model file: %w", readErr)
		}
	}

	return hex.EncodeToString(hasher.Sum(nil)), totalBytes, nil
}

// VerifyModelFile verifies that the target model file matches the expected SHA-256 checksum.
func VerifyModelFile(filepath string, expectedSHA256 string) error {
	computed, _, err := ComputeFileSHA256(filepath)
	if err != nil {
		return err
	}
	if subtle.ConstantTimeCompare([]byte(computed), []byte(expectedSHA256)) != 1 {
		return fmt.Errorf("%w: computed=%s expected=%s", ErrChecksumMismatch, computed, expectedSHA256)
	}
	return nil
}

// GenerateModelManifest creates a cryptographic provenance manifest for a local model file.
func GenerateModelManifest(name, filepath string) (*ModelManifest, error) {
	digest, size, err := ComputeFileSHA256(filepath)
	if err != nil {
		return nil, err
	}
	return &ModelManifest{
		ModelName:     name,
		FilePath:      filepath,
		FileSHA256:    digest,
		FileSizeBytes: size,
		GeneratedAt:   time.Now().UTC(),
	}, nil
}

// SignModelManifest signs the manifest contents with an Ed25519 private key.
func SignModelManifest(manifest *ModelManifest, privKey ed25519.PrivateKey) error {
	if manifest == nil || len(privKey) != ed25519.PrivateKeySize {
		return errors.New("invalid manifest or private key")
	}

	digestBytes := manifestContentBytes(manifest)
	sig := ed25519.Sign(privKey, digestBytes)
	pubKey := privKey.Public().(ed25519.PublicKey)

	manifest.SignatureHex = hex.EncodeToString(sig)
	manifest.PublicKeyHex = hex.EncodeToString(pubKey)
	return nil
}

// VerifyModelManifest validates the Ed25519 signature and optionally checks the local model file.
func VerifyModelManifest(manifest *ModelManifest, trustedPubKey ed25519.PublicKey, verifyFile bool) error {
	if manifest == nil || manifest.SignatureHex == "" {
		return ErrInvalidManifest
	}

	sigBytes, err := hex.DecodeString(manifest.SignatureHex)
	if err != nil || len(sigBytes) != ed25519.SignatureSize {
		return ErrManifestSignature
	}

	var pubBytes []byte
	if len(trustedPubKey) == ed25519.PublicKeySize {
		pubBytes = trustedPubKey
	} else {
		pubBytes, err = hex.DecodeString(manifest.PublicKeyHex)
		if err != nil || len(pubBytes) != ed25519.PublicKeySize {
			return ErrManifestSignature
		}
	}

	digestBytes := manifestContentBytes(manifest)
	if !ed25519.Verify(pubBytes, digestBytes, sigBytes) {
		return ErrManifestSignature
	}

	if verifyFile && manifest.FilePath != "" {
		if err := VerifyModelFile(manifest.FilePath, manifest.FileSHA256); err != nil {
			return err
		}
	}

	return nil
}

func manifestContentBytes(m *ModelManifest) []byte {
	// Deterministic serialization of invariant fields
	type Invariant struct {
		ModelName     string `json:"model_name"`
		FileSHA256    string `json:"file_sha256"`
		FileSizeBytes int64  `json:"file_size_bytes"`
	}
	inv := Invariant{
		ModelName:     m.ModelName,
		FileSHA256:    m.FileSHA256,
		FileSizeBytes: m.FileSizeBytes,
	}
	bytes, _ := json.Marshal(inv)
	return bytes
}
