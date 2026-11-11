# Phase 2: Native Data-Plane Integration

## Objective

Connect the existing C++ layer streamer to the Rust safety boundary used by
the Go control plane, while keeping the FFI surface small and lifecycle-safe.

## Delivered contract

1. C++ exposes a narrow `extern "C"` lifecycle boundary.
2. A successful Rust start performs RAM and storage admission checks before
   starting the C++ engine.
3. The native engine validates a regular, non-empty model file, opens it with
   the existing NVMe reader, and maps one bounded initial layer window.
4. Native stop releases the mapped pages, closes the file descriptor, and
   clears the active engine state.
5. The Rust build links the C++ static library and its required C++ runtime
   explicitly; missing native artifacts fail the build with an actionable
   message.
6. The end-to-end lifecycle test verifies a stop followed by a second start and
   stop, exercising native teardown and reinitialization.

## Explicit non-goals

This phase does not parse a model format, expose tensor metadata, execute
tensor kernels, or provide a complete inference request API. Those capabilities
remain in later phases.
