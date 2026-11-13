# Phase 10: CI and Integration Quality Gates

Phase 10 makes the repository’s existing safety and lifecycle checks enforceable
on every main-branch push and pull request.

## Automated gates

The core matrix now runs:

- C++ CTest coverage through the existing debug test build;
- Rust safety-grid unit tests;
- Go control-plane tests with the race detector;
- Go static analysis with `go vet`;
- Bash syntax validation for release, e2e, security-audit, and VM scripts.

The `Makefile` exposes the same checks through `make test`, with focused
`test-core`, `test-control`, and `lint-shell` targets for local development.

## End-to-end contract

The lifecycle e2e test now verifies the observability contract as well as
start/stop/restart behavior. It checks that `/metrics` responds successfully,
returns a bounded request counter, reports the stopped-engine gauge, and emits
an `X-Request-ID` response header.

These checks do not install software. They consume toolchains already supplied
by the developer environment or the CI runner.
