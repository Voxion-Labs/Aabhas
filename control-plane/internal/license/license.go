// Package license provides hardware-bound enterprise license generation,
// Ed25519 cryptographic signing, and constant-time hardware verification.
package license

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

var (
	ErrInvalidHardwareUUID = errors.New("invalid or malformed hardware UUID (must be 32 hex digits)")
	ErrHardwareMismatch    = errors.New("license hardware UUID does not match target host machine")
	ErrSignatureRejected   = errors.New("license Ed25519 signature verification failed")
	ErrLicenseExpired      = errors.New("enterprise license has expired")
	ErrInvalidPublicKey    = errors.New("invalid Ed25519 public key length (must be 32 bytes)")
	ErrInvalidSignature    = errors.New("invalid Ed25519 signature length (must be 64 bytes)")
	ErrEmptyPayload        = errors.New("license payload is empty")
)

// LicenseTier specifies the operational capability tier.
type LicenseTier string

const (
	TierCommunity  LicenseTier = "Community"
	TierPro        LicenseTier = "Pro"
	TierEnterprise LicenseTier = "Enterprise"
)

// LicensePayload defines the enterprise contract bound to host hardware.
type LicensePayload struct {
	LicensedHardwareUUID    string      `json:"licensed_hardware_uuid"`
	CustomerName            string      `json:"customer_name"`
	Tier                    LicenseTier `json:"tier"`
	MaxConcurrentModels     int         `json:"max_concurrent_models"`
	MaxVramBytes            uint64      `json:"max_vram_bytes"`
	MaxContextLength        int         `json:"max_context_length"`
	IssuedAt                time.Time   `json:"issued_at"`
	ExpiresAt               time.Time   `json:"expires_at"`
	FeatureGpuAcceleration  bool        `json:"feature_gpu_acceleration"`
	FeatureDirectNvmeStream bool        `json:"feature_direct_nvme_stream"`
}

// SignedLicense represents an exportable, cryptographically signed hardware license envelope.
type SignedLicense struct {
	LicensedHardwareUUID string         `json:"licensed_hardware_uuid"`
	PayloadHex           string         `json:"payload_hex"`
	PublicKeyHex         string         `json:"public_key_hex"`
	SignatureHex         string         `json:"signature_hex"`
	Payload              LicensePayload `json:"payload"`
}

// NormalizeHardwareUUID standardizes UUID strings: removes braces, dashes, whitespace and lowercases.
// Exactly mirrors the Rust safety-grid hw_binder normalization logic.
func NormalizeHardwareUUID(input string) (string, error) {
	var sb strings.Builder
	sb.Grow(32)

	for _, r := range input {
		if r == '{' || r == '}' || r == '-' || r == ' ' || r == '\t' || r == '\r' || r == '\n' {
			continue
		}
		if (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F') {
			if r >= 'A' && r <= 'F' {
				sb.WriteRune(r + ('a' - 'A'))
			} else {
				sb.WriteRune(r)
			}
		} else {
			return "", ErrInvalidHardwareUUID
		}
	}

	norm := sb.String()
	if len(norm) != 32 {
		return "", fmt.Errorf("%w: length %d expected 32", ErrInvalidHardwareUUID, len(norm))
	}
	return norm, nil
}

// GenerateKeyPair generates a new Ed25519 signing keypair for license issuance.
func GenerateKeyPair() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("generate ed25519 keypair: %w", err)
	}
	return pub, priv, nil
}

// IssueLicense signs a LicensePayload with the private key and returns the complete signed envelope.
func IssueLicense(payload LicensePayload, privKey ed25519.PrivateKey) (*SignedLicense, error) {
	if len(privKey) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid ed25519 private key size")
	}

	normUUID, err := NormalizeHardwareUUID(payload.LicensedHardwareUUID)
	if err != nil {
		return nil, fmt.Errorf("normalize payload uuid: %w", err)
	}
	payload.LicensedHardwareUUID = normUUID

	if payload.IssuedAt.IsZero() {
		payload.IssuedAt = time.Now().UTC()
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal license payload: %w", err)
	}

	signature := ed25519.Sign(privKey, payloadBytes)
	pubKey := privKey.Public().(ed25519.PublicKey)

	return &SignedLicense{
		LicensedHardwareUUID: normUUID,
		PayloadHex:           hex.EncodeToString(payloadBytes),
		PublicKeyHex:         hex.EncodeToString(pubKey),
		SignatureHex:         hex.EncodeToString(signature),
		Payload:              payload,
	}, nil
}

// ValidateLicense validates constant-time hardware binding, Ed25519 signature, and expiration.
func ValidateLicense(signed *SignedLicense, actualHardwareUUID string, trustedPubKey ed25519.PublicKey) error {
	if signed == nil {
		return errors.New("signed license is nil")
	}

	// 1. Hardware Binding validation in constant time
	normActual, err := NormalizeHardwareUUID(actualHardwareUUID)
	if err != nil {
		return fmt.Errorf("actual hardware uuid: %w", err)
	}

	normLicensed, err := NormalizeHardwareUUID(signed.LicensedHardwareUUID)
	if err != nil {
		return fmt.Errorf("licensed hardware uuid: %w", err)
	}

	if subtle.ConstantTimeCompare([]byte(normActual), []byte(normLicensed)) != 1 {
		return ErrHardwareMismatch
	}

	// 2. Decode raw materials
	payloadBytes, err := hex.DecodeString(signed.PayloadHex)
	if err != nil || len(payloadBytes) == 0 {
		return ErrEmptyPayload
	}

	sigBytes, err := hex.DecodeString(signed.SignatureHex)
	if err != nil || len(sigBytes) != ed25519.SignatureSize {
		return ErrInvalidSignature
	}

	var pubKeyBytes []byte
	if len(trustedPubKey) == ed25519.PublicKeySize {
		pubKeyBytes = trustedPubKey
	} else {
		pubKeyBytes, err = hex.DecodeString(signed.PublicKeyHex)
		if err != nil || len(pubKeyBytes) != ed25519.PublicKeySize {
			return ErrInvalidPublicKey
		}
	}

	// 3. Ed25519 Signature Verification
	if !ed25519.Verify(pubKeyBytes, payloadBytes, sigBytes) {
		return ErrSignatureRejected
	}

	// 4. Parse payload and verify expiry
	var parsed LicensePayload
	if err := json.Unmarshal(payloadBytes, &parsed); err != nil {
		return fmt.Errorf("unmarshal payload: %w", err)
	}

	if !parsed.ExpiresAt.IsZero() && time.Now().UTC().After(parsed.ExpiresAt) {
		return ErrLicenseExpired
	}

	// Ensure payload hardware matches
	normPayloadUUID, err := NormalizeHardwareUUID(parsed.LicensedHardwareUUID)
	if err != nil || subtle.ConstantTimeCompare([]byte(normActual), []byte(normPayloadUUID)) != 1 {
		return ErrHardwareMismatch
	}

	return nil
}

// ToEnvMap formats the signed license into the environment variables required by Aabhas Rust safety grid.
func (sl *SignedLicense) ToEnvMap() map[string]string {
	return map[string]string{
		"AABHAS_LICENSE_REQUIRED":        "1",
		"AABHAS_LICENSED_HARDWARE_UUID":  sl.LicensedHardwareUUID,
		"AABHAS_LICENSE_PAYLOAD_HEX":     sl.PayloadHex,
		"AABHAS_LICENSE_PUBLIC_KEY_HEX":  sl.PublicKeyHex,
		"AABHAS_LICENSE_SIGNATURE_HEX":   sl.SignatureHex,
	}
}

// ToJSON serializes the signed license to a formatted JSON string.
func (sl *SignedLicense) ToJSON() string {
	bytes, err := json.MarshalIndent(sl, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(bytes)
}
