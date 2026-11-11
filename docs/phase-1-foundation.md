# Phase 1: Foundation Build Contract

## Objective

Make the repository's existing native layers observable and reproducibly
buildable before adding model execution behavior.

## Delivered contract

1. The C++17 data plane is configured and compiled by `make build-core`.
2. CI configures and builds `core-engine` independently of the Rust crate.
3. The production image build fails if the C++ static library is not produced.
4. The expected C++ artifact is checked explicitly as
   `libaabhas_data_plane.a`.
5. The architecture documentation records the build boundary and the limited
   Phase 2 lifecycle integration, so a green Rust/Go build cannot be mistaken
   for end-to-end inference readiness.

## Acceptance criteria

On a Linux development host with the repository's documented prerequisites:

```bash
make build-core
```

must compile both the C++ data plane and the Rust safety library. The CI and
container build paths must enforce the same C++ artifact check.

## Explicit non-goals

This phase does not add a model-format parser, tensor kernels, inference
runtime, or a new system dependency. Those belong to later phases and must be
integrated only after this build boundary is stable.
