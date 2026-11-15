# Product Phase 8: Multi-Model Scheduling and Batching

## Objective

Deliver production-grade multi-model pooling and continuous sequence batching across
both the C++ data plane and the Go control plane: dynamic batch scheduling with
interleaved prompt prefill and multi-token decode across heterogeneous sequence lengths,
per-sequence KV-cache and sampler isolation, thread-safe multi-model slot leasing,
least-recently-used (LRU) model eviction under bounded memory constraints, and
asynchronous request batching with per-sequence streaming deltas.

## Delivered contract

1. Continuous / Dynamic Sequence Batching (`inference/batch_scheduler.hpp`, `inference/batch_scheduler.cpp`):
   - `BatchSchedulerConfig`: Configures maximum concurrent batch size (`max_batch_size`), queue depth (`max_queue_depth`), and sequence length limits (`max_sequence_length`).
   - `BatchSequenceState`:
     - Tracks individual sequence lifecycle through `SequenceStatus` (`Pending`, `Prefilling`, `Decoding`, `Finished`, `Failed`).
     - Supports individual termination conditions via `FinishReason` (`EosToken`, `StopToken`, `MaxNewTokens`, `ContextLengthExceeded`, `Aborted`).
     - Dedicated per-sequence `KvCache` instance preventing cross-sequence attention state contamination.
     - Dedicated per-sequence `Sampler` with independent temperatures, top-k/top-p parameters, and repetition penalty tracking.
   - Pipelined continuous execution (`BatchScheduler::step`):
     - Dynamic admission of pending requests into available batch slots.
     - Prefill phase executes full prompt forward passes and initial token sampling.
     - Decode phase steps single-token forward evaluation across all active decoding sequences.
     - Immediate retirement of finished sequences into completed results while active sequences continue decoding without stalls.
     - Incremental per-sequence `StreamCallback` invocations delivering live token IDs and text deltas.
   - High-level synchronous batch API (`BatchScheduler::generate_batch` / `TokenGenerator::generate_batch`).

2. Multi-Model Execution Pool (`inference/model_pool.hpp`, `inference/model_pool.cpp`):
   - `ModelPoolConfig`: Bounded constraints on loaded model instances (`max_loaded_models`), total host memory consumption (`max_host_memory_bytes`), and device VRAM budget (`max_vram_bytes`).
   - `ModelRegistration`: Complete model specification (model path, `ModelIndex`, `TransformerModelConfig`, `InferenceConfig`, and memory estimations).
   - `ModelLease`: RAII lease handle granting thread-safe execution access.
     - Holds an active reference preventing the underlying model from being evicted while in use.
     - Safe move semantics and automatic lease return upon destruction or explicit release.
   - LRU model eviction policy:
     - Automatically evicts least-recently-used idle model slots (`active_leases == 0`) when loaded model capacity or memory limits are reached.
     - Rejects new acquisitions with graceful exhaustion errors when all loaded slots are actively leased.
     - Explicit model eviction (`evict_model`) and preloading (`preload_model`).
     - Point-in-time metrics: active leases, total acquisitions, total evictions, and memory footprint.

3. Engine Runtime Integration (`inference/token_generator.hpp`, `inference/inference_runtime.hpp`):
   - `TokenGenerator::forward_step` overload accepting an explicit caller-owned `KvCache&`: allows external schedulers and multi-sequence runners to execute forward passes without mutating internal single-sequence state.
   - `TokenGenerator::generate_batch` and `InferenceRuntime::generate_batch`: Native batch generation convenience interfaces.

4. Control-Plane Coordination (`control-plane/internal/orchestrator/multi_model.go`, `batch_queue.go`):
   - `MultiModelScheduler`: Manages multi-model concurrency, LRU eviction, and ref-counted leasing in Go.
   - `BatchQueue`: Dynamic request windowing, timeout-based batch aggregation, and asynchronous response channels with thread-safe cancellation and shutdown.

## Verification

1. CTest (`tests/batch_scheduler_test.cpp`):
   - Idle and empty batch step invariants.
   - Multi-sequence batch execution with heterogeneous prompt lengths and token budgets.
   - Independent early stopping: verification that sequence A stopping on a designated stop token does not halt or affect sequence B.
   - Continuous admission: verifies that with `max_batch_size = 1`, completing request 1 immediately admits request 2 into the pipeline on the subsequent step.
   - Per-sequence token and text streaming callbacks.
   - `InferenceRuntime::generate_batch` end-to-end integration.

2. CTest (`tests/model_pool_test.cpp`):
   - Model registration, query, and unregistration lifecycle.
   - Model acquisition, lease lifetime, and autoregressive generation through acquired leases.
   - Concurrent leases sharing a model slot and preventing premature eviction.
   - Multi-model LRU eviction: verifies oldest idle model is evicted when slot limit is reached.
   - Active lease eviction protection: verifies busy model slots cannot be evicted under pressure.

3. Go Test Suite (`control-plane/internal/orchestrator`):
   - `multi_model_test.go`: Registration, leasing, LRU eviction, and active lease protection.
   - `batch_queue_test.go`: Request buffering, batch formation, response dispatch, and graceful drain on close.
   - 100% pass under `go test -race -count=1 ./...` and `go vet ./...`.

## Boundary

This phase delivers multi-model pooling, dynamic sequence batching, and control-plane scheduling.
Hardware performance benchmarking, target hardware compatibility matrices, and production key
management remain subsequent product phases.
