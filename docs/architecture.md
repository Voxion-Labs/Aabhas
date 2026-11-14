# Aabhas Architecture

## Scope

Aabhas is an out-of-core inference engine. The control plane accepts a small
REST command surface, the Rust safety layer performs host admission checks, and
the C++ data plane maps or streams tensor storage in bounded windows. The
system is designed for model files that cannot be resident in host RAM.

The implementation is split into three failure domains:

1. Go owns external requests, lifecycle serialization, and operational state.
2. Rust owns the narrow C ABI and health gates that must complete before native
   execution starts.
3. C++ owns POSIX file descriptors, virtual mappings, NVMe-oriented reads, and
   resident-page release.

## Data flow

```text
CLI or REST client
        |
        v
Go control plane ---> single-model execution gate
        |
        v
Rust C ABI ---> RAM cap and filesystem reserve checks
        |
        v
C++ data plane ---> fd/mmap window ---> tensor consumer
        |
        +---- madvise(MADV_SEQUENTIAL)
        +---- madvise(MADV_DONTNEED) on layer drop
```

The Go process never copies tensor payloads across the control-plane boundary.
The C ABI receives only a NUL-terminated model path and returns a small integer
status. Native calls are serialized in the Go bridge and the Rust engine state
also rejects a second active model.

## C++ NVMe data plane

The C++ data plane is currently built as the `aabhas_data_plane` static
library. The Phase 1 build contract makes that artifact mandatory in local,
CI, and container builds; a successful Rust or Go build must not be treated as
proof that the C++ layer was compiled. The Phase 2 Rust safety boundary now
invokes the C++ lifecycle boundary after RAM and storage admission checks.
Phase 2 starts one bounded file-backed layer window and releases it during
stop. Phase 3 adds bounded GGUF and SafeTensors metadata inspection so
recognized tensor ranges become individual streamer layers; tensor decoding
and execution remain later phases. Phase 4 wraps the streamer in a serialized
layer scheduler: one resident window is active at a time, explicit release is
required before advancement, and bounded `POSIX_FADV_WILLNEED` hints provide
read-ahead without a second user-space tensor buffer. Phase 5 adds typed tensor
metadata, a streamed F32 matrix-vector kernel, and a bounded KV-cache as the
first compute-facing runtime boundary.

`NvmeDirectReader` owns a raw POSIX file descriptor. On Linux it attempts
`O_DIRECT` and falls back to buffered `O_RDONLY` when the filesystem does not
support direct I/O. The fallback is explicit and observable through reader
state; direct-I/O callers must honor the platform alignment reported by the
reader.

`MemoryIllusionist` maps a file range with `mmap`. Unaligned file offsets are
rounded down to the host page size and the public view is adjusted to the
requested byte range. Mappings are RAII-owned and are unmapped on replacement or
destruction. The layer streamer maps at most one bounded chunk at a time, so a
large layer is consumed through repeated `fetch_next_chunk` calls rather than
materialized as a heap allocation.

Before consumption, the mapped range receives `MADV_SEQUENTIAL`. When a layer
or chunk is dropped, the page-aligned mapping base receives `MADV_DONTNEED`
before it is unmapped. These are kernel residency hints, not a guarantee that
the operating system will reclaim a page at a specific instant.

## Rust safety grid

The Rust library is built as a `cdylib` and exposes only
`start_aabhas_engine(const char*) -> i32` and `stop_aabhas_engine() -> i32`.
Each entry point catches panics so Rust unwinding cannot cross the C ABI.

The RAM capper reads host physical memory through `sysinfo` and rejects a start
when reported used memory is at or above 70% of total memory. When cgroup v2
memory files are available, the same strict boundary is evaluated against the
container limit and current usage as well. The SSD reserve probe uses `statvfs`
on the model path and requires at least 20 GiB of blocks available to the
calling process; `AABHAS_MIN_FREE_BYTES` can only increase that reserve. A
failed probe is a health-check failure, not an implicit permission to continue.

The C ABI bounds model paths to 4096 bytes. Direct-I/O reads validate the
destination address, offset, and length against the file's probed transfer
alignment before issuing a syscall.

These checks are admission controls. They do not replace cgroup or Kubernetes
memory limits, filesystem quotas, or operating-system monitoring. Production
deployments must keep those independent controls enabled.

## Go control plane

The daemon exposes:

* `POST /api/v1/engine/start` with `{ "model_path": "..." }`.
* `POST /api/v1/engine/stop` with an empty body.
* `GET /api/v1/engine/status` with lifecycle counters and current model state.
* `POST` and `GET /api/v1/models` for validated model metadata registration.
* `GET /metrics` for protected Prometheus-compatible operational metrics.
* `GET /healthz` and `GET /readyz` for non-sensitive orchestrator probes.

Handlers enforce method checks, bounded request bodies, strict JSON decoding,
request cancellation checks, and bounded server timeouts. `ExecutionController`
provides a mutex-protected single-model lease and atomic lifecycle counters;
failed starts cancel their lease without inflating stop counts. `ModelRegistry`
stores only validated file metadata. `LayerManager` tracks layer descriptors and
transitions them through queued, streaming, loaded, dropped, and failed states
without owning tensor bytes.

The product-format boundary additionally derives checked tensor storage layouts
for common GGUF quantized blocks and SafeTensors scalar dtypes. Known ranges
must match their shape-derived byte size; unsupported dtypes remain bounded
metadata only and are never silently reinterpreted by a kernel.

The daemon handles SIGINT and SIGTERM, shuts down HTTP acceptance, and calls
the native stop function on every normal exit path. Native status codes are
translated into stable HTTP responses, including conflict, throttling, and
storage-reserve failures.

## Deployment and storage

The production container runs as UID/GID 65532, drops all capabilities except
the explicitly requested `IPC_LOCK`, disables privilege escalation, and uses a
read-only root filesystem. The Kubernetes deployment mounts model and tensor
directories as read-only `hostPath` volumes. Operators must provision those
directories on the intended SSD on every eligible node and must review the
security implications of hostPath before enabling the deployment.

The data-plane memory cap is separate from container limits. Kubernetes memory
limits remain the final host-protection boundary; the Rust 70% check prevents
admission when the host is already under pressure.

## Testing and operations

The end-to-end script starts a prebuilt daemon, creates a temporary mock model,
asserts HTTP 200 for start and stop, and kills the process through a cleanup
trap if the test fails. The Valgrind audit compiles the native layer streamer
test harness with debug symbols and runs it with full leak checking and a
non-zero error exit code.

The dev container contains only the compilers and diagnostics required for
these tests. It is non-root by default and should receive the repository as a
workspace mount rather than baking credentials or model data into an image.

## Security boundaries

When `AABHAS_API_TOKEN` is configured, the Go server enforces bearer
authentication with constant-time comparison before dispatching any API
handler. When `AABHAS_LICENSE_REQUIRED=1` is configured, the Rust gate
requires a valid Ed25519 signature and matching normalized hardware UUID before
native startup.

The Go server also emits dependency-free Prometheus metrics at `/metrics` and
assigns every response an `X-Request-ID`. Metrics use normalized, low-cardinality
routes; access logs contain request metadata only and never model payloads or
credentials.

The repository release workflow is the only workflow granted `contents: write`.
It runs after successful main-branch CI, uses the GitHub Actions bot identity,
and writes generated release metadata in addition to publishing release
artifacts. Branch protection, required owner review, fork policy, and the
Actions-bot bypass list remain GitHub account settings and must be configured
by the repository owner.
