# Phase 8: Security and Licensing Hardening

## Objective

Add real cryptographic license verification, hardware binding, and an explicit
control-plane authentication mechanism without exposing secrets in responses
or logs.

## Delivered contract

1. Rust verifies license payloads with Ed25519 instead of the previous shape-
   only placeholder.
2. Hardware UUID normalization requires exactly 32 hexadecimal characters after
   separators are removed and compares the normalized values in constant time.
3. `AABHAS_LICENSE_REQUIRED=1` enables a fail-closed license gate using the
   documented hex-encoded key, signature, and payload variables. Development
   mode remains compatible when no license material is configured.
4. License rejection returns native status `-8` and maps to HTTP `403`.
5. `AABHAS_API_TOKEN` enables constant-time bearer-token authentication for all
   API endpoints; deployment manifests reference a secret without embedding a
   credential.
6. The standalone poison test keeps a local crypto stub because it compiles a
   narrow hardware-binding harness directly with `rustc`, outside Cargo.

## Explicit non-goals

This phase does not add a license issuance service, key-management system, or
network identity provider. Private signing keys remain outside the repository.
