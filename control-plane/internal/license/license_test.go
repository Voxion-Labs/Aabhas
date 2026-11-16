package license

import (
	"crypto/ed25519"
	"testing"
	"time"
)

func TestNormalizeHardwareUUID(t *testing.T) {
	raw := "{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"
	expected := "a1b2c3d4e5f67890abcdef1234567890"

	norm, err := NormalizeHardwareUUID(raw)
	if err != nil {
		t.Fatalf("normalize valid uuid: %v", err)
	}
	if norm != expected {
		t.Fatalf("expected %s, got %s", expected, norm)
	}

	// Reject bad chars
	if _, err := NormalizeHardwareUUID("invalid-uuid-with-xyz-1234"); err == nil {
		t.Fatalf("expected error for invalid hex chars")
	}

	// Reject short
	if _, err := NormalizeHardwareUUID("a1b2c3"); err == nil {
		t.Fatalf("expected error for short uuid")
	}
}

func TestIssueAndValidateLicense(t *testing.T) {
	pubKey, privKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}

	targetHardware := "4A2B9F8C-1234-5678-9ABC-DEF012345678"
	payload := LicensePayload{
		LicensedHardwareUUID:    targetHardware,
		CustomerName:            "Voxion Labs Enterprise",
		Tier:                    TierEnterprise,
		MaxConcurrentModels:     8,
		MaxVramBytes:            64 * 1024 * 1024 * 1024,
		MaxContextLength:        8192,
		IssuedAt:                time.Now().UTC(),
		ExpiresAt:               time.Now().UTC().Add(365 * 24 * time.Hour),
		FeatureGpuAcceleration:  true,
		FeatureDirectNvmeStream: true,
	}

	signed, err := IssueLicense(payload, privKey)
	if err != nil {
		t.Fatalf("issue license: %v", err)
	}

	// 1. Valid matching hardware verification
	if err := ValidateLicense(signed, targetHardware, pubKey); err != nil {
		t.Fatalf("validate matching license: %v", err)
	}

	// 2. Different format of same UUID (lowercase with braces)
	diffFormat := "{4a2b9f8c-1234-5678-9abc-def012345678}"
	if err := ValidateLicense(signed, diffFormat, pubKey); err != nil {
		t.Fatalf("validate same uuid with different casing/braces: %v", err)
	}

	// 3. Different hardware UUID must fail closed
	wrongHardware := "11111111-2222-3333-4444-555555555555"
	if err := ValidateLicense(signed, wrongHardware, pubKey); err != ErrHardwareMismatch {
		t.Fatalf("expected ErrHardwareMismatch, got %v", err)
	}

	// 4. Tampered payload must fail
	tampered := *signed
	tampered.PayloadHex = tampered.PayloadHex[:len(tampered.PayloadHex)-2] + "ff"
	if err := ValidateLicense(&tampered, targetHardware, pubKey); err == nil {
		t.Fatalf("expected error for tampered payload, got nil")
	}

	// 5. Expired license must fail
	expiredPayload := payload
	expiredPayload.ExpiresAt = time.Now().UTC().Add(-1 * time.Hour)
	expiredSigned, err := IssueLicense(expiredPayload, privKey)
	if err != nil {
		t.Fatalf("issue expired license: %v", err)
	}
	if err := ValidateLicense(expiredSigned, targetHardware, pubKey); err != ErrLicenseExpired {
		t.Fatalf("expected ErrLicenseExpired, got %v", err)
	}

	// 6. Untrusted public key
	untrustedPub, _, _ := ed25519.GenerateKey(nil)
	if err := ValidateLicense(signed, targetHardware, untrustedPub); err != ErrSignatureRejected {
		t.Fatalf("expected ErrSignatureRejected for untrusted key, got %v", err)
	}

	// 7. Environment map export
	envMap := signed.ToEnvMap()
	if envMap["AABHAS_LICENSE_REQUIRED"] != "1" || envMap["AABHAS_LICENSED_HARDWARE_UUID"] == "" ||
		envMap["AABHAS_LICENSE_PAYLOAD_HEX"] == "" || envMap["AABHAS_LICENSE_PUBLIC_KEY_HEX"] == "" ||
		envMap["AABHAS_LICENSE_SIGNATURE_HEX"] == "" {
		t.Fatalf("incomplete environment map export: %+v", envMap)
	}
}
