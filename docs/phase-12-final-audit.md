# Phase 12: Final Audit and Release Readiness

Phase 12 closes the project implementation loop with a release-content audit,
final API contract tests, and a single safe local verification target.

## Release contract

The deterministic release script now copies and strips both control-plane
binaries (`aabhas-daemon` and `aabhas-cli`) alongside `libaabhas.so`, LICENSE,
and the SHA-256 sidecar. It checks for every host utility it needs before
starting the isolated Docker build.

## Final verification

`make safe-test` now runs the complete repository gate: native CTest, Rust
tests, race-enabled Go tests and vet, and shell syntax checks. The Go API test
contract verifies that health probes remain public and that metrics remains
protected when `AABHAS_API_TOKEN` is configured.

The local Windows environment cannot execute the native or Bash portions. CI
remains the authoritative Linux validation environment; no local toolchain or
software installation is required by this phase.

## Completion boundary

Aabhas is now a buildable, test-gated out-of-core inference foundation with a
bounded C++/Rust data plane, Go lifecycle control plane, licensing/authentication,
observability, and hardened container deployment assets. Full production
model-format coverage, GPU kernels, identity-provider integration, and cluster
policy remain explicit product extensions rather than hidden assumptions in the
current release.
