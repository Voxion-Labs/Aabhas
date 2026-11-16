# Product Phase 10: Production Identity, Key Management, and Final Release Validation

## Objective

Deliver production-grade cryptographic identity, role-based access control (RBAC),
hardware-bound enterprise licensing, model integrity verification, and automated
release validation across both the C++ data plane and the Go control plane.

## Delivered Architecture

### 1. Cryptographic Identity & RBAC (`control-plane/internal/auth`)

- **Role-Based Access Control (RBAC)**:
  - `RoleAdmin`: Unrestricted privileges (`/api/v1/engine/*`, `/api/v1/auth/tokens`, `/api/v1/license/*`).
  - `RoleOperator`: Operational operations (`/api/v1/hardware/*`, `/api/v1/benchmark`, `/api/v1/models/verify`).
  - `RoleInference`: Model querying and execution (`/api/v1/models`, generation pipelines).
  - `RoleAuditor`: Read-only diagnostics and health monitoring (`/metrics`, `/healthz`, `/readyz`, `/api/v1/engine/status`).
  - Hierarchical role evaluation (`Identity.HasRole`): Admin satisfies all lower tiers; Operator satisfies Inference and Auditor; Inference satisfies Auditor.
- **Cryptographic Token Lifecycle (`auth.TokenManager`)**:
  - Tokens use role-prefixed formatting (`abh_adm_...`, `abh_ops_...`, `abh_inf_...`, `abh_aud_...`) with 32 bytes of CSPRNG entropy.
  - Zero raw token persistence: tokens are hashed with SHA-256 for storage and lookup.
  - Constant-time verification using `crypto/subtle.ConstantTimeCompare` against timing side-channel attacks.
  - Expiration tracking (`ExpiresAt`) and explicit revocation list (`Revoke`).
  - Dynamic token rotation (`Rotate`): invalidates the target token and atomically generates a successor.
  - Backward compatibility: environment-provided `AABHAS_API_TOKEN` is automatically bound as a static admin identity.
- **Secure Transport (TLS 1.3 / mTLS)**:
  - `BuildTLSConfig`: Configures modern TLS 1.3/1.2 cipher suites (`TLS_AES_128_GCM_SHA256`, `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`).
  - Optional mutual TLS client certificate verification (`ClientAuth: tls.RequireAndVerifyClientCert`).

### 2. Hardware-Bound Enterprise Licensing (`control-plane/internal/license`)

- **Hardware Binding & Ed25519 Signatures**:
  - `LicensePayload`: Customer details, tier (`Enterprise`, `Pro`, `Community`), concurrency limits, VRAM caps, context limits, and hardware binding.
  - Hardware UUID normalization (`NormalizeHardwareUUID`): removes braces, hyphens, and whitespace; validates 32 hex character constraint in lowercase ASCII, strictly matching the Rust safety-grid `hw_binder` specification.
  - Cryptographic envelope signing via Ed25519 (`crypto/ed25519`) and constant-time hardware verification (`subtle.ConstantTimeCompare`).
  - Fail-closed enforcement on expired licenses, tampered payloads, or mismatched machine IDs.
  - Interoperable environment map export (`ToEnvMap`) matching `AABHAS_LICENSE_REQUIRED`, `AABHAS_LICENSED_HARDWARE_UUID`, `AABHAS_LICENSE_PAYLOAD_HEX`, `AABHAS_LICENSE_PUBLIC_KEY_HEX`, and `AABHAS_LICENSE_SIGNATURE_HEX`.

### 3. Native Model Integrity & Provenance (`core-engine` & `control-plane`)

- **C++ Native Integrity (`core-engine/include/model/model_integrity.hpp`)**:
  - FIPS 180-4 compliant streaming SHA-256 state machine (`Sha256Context`) with zero third-party dependencies.
  - `ModelManifest`: Records structural tensor digest, file content hash, byte size, and tensor count.
  - `ModelIntegrityVerifier::compute_manifest_digest`: Computes deterministic SHA-256 digest over sorted tensor descriptors, offsets, sizes, dtypes, and shapes.
  - Detects corruption, missing tensors, altered offsets, or poisoned weight files before streaming.
- **Go Integrity Verifier (`control-plane/internal/orchestrator/integrity.go`)**:
  - Chunked 64 KiB streaming file hash computation (`ComputeFileSHA256`).
  - Manifest generation and Ed25519 provenance signing (`SignModelManifest` / `VerifyModelManifest`).

### 4. Control-Plane REST API & CLI Tooling

- **REST Endpoints (`control-plane/internal/api/rest_server.go`)**:
  - `POST /api/v1/auth/tokens`: Generate new RBAC tokens (Admin required).
  - `GET /api/v1/auth/tokens`: List token inventory with redacted secrets (Admin required).
  - `DELETE /api/v1/auth/tokens`: Revoke active token (Admin required).
  - `POST /api/v1/license/verify`: Validate hardware-bound enterprise license envelopes.
  - `POST /api/v1/models/verify`: Verify file checksum against expected digest.
  - Route-level RBAC middleware enforcing least-privilege access.
- **CLI Subcommands (`control-plane/cmd/aabhas-cli/main.go`)**:
  - `aabhas-cli keygen`: Generates cryptographically secure bearer tokens for services or operators.
  - `aabhas-cli license generate`: Issues signed enterprise licenses for target hardware UUIDs.
  - `aabhas-cli license verify`: Validates local license files against machine hardware.
  - `aabhas-cli verify-model`: Computes and checks model file SHA-256 checksums.

### 5. Release Validation Automation (`scripts/verify_release.sh`)

- Validates deterministic release archives (`dist/aabhas-*.tar.gz`).
- Cryptographic SHA-256 sidecar checksum validation.
- Verifies required binaries (`aabhas-daemon`, `aabhas-cli`, `libaabhas.so`) and license presence.
- Verifies stripped debugging symbol status.

## Verification

1. **CTest (`core-engine/tests/model_integrity_test.cpp`)**:
   - `test_sha256_standard_vectors`: Validates against NIST standard test vectors.
   - `test_sha256_file_hashing`: Verifies chunked file hashing consistency.
   - `test_model_manifest_and_tampering_detection`: Verifies detection of altered tensor offsets, shapes, dtypes, and counts.
   - `test_end_to_end_file_manifest_verification`: Verifies file-level content corruption detection.

2. **Go Test Suite (`control-plane/internal/...`)**:
   - `auth_test.go`: Role hierarchy, token issuance, expiration, revocation, rotation, and constant-time validation.
   - `license_test.go`: Hardware UUID normalization, Ed25519 signing, constant-time binding, and expiry checks.
   - `integrity_test.go`: Checksum calculation, manifest signing, and tamper rejection.
   - `rest_server_auth_test.go`: RBAC enforcement across start, stop, metrics, hardware compatibility, and token administration.

## Status

Phase 10 completes the full 10-phase production foundation for Aabhas. All components pass automated quality gates with zero compiler warnings under `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`.
