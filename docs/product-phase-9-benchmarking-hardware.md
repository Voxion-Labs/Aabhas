# Product Phase 9: Benchmarking and Hardware Compatibility

## Objective

Deliver production-grade hardware profiling, automatic execution tier evaluation, and high-precision inference benchmarking across both the C++ native data plane and the Go control plane:
1. Instruction-set and hardware capability discovery (SIMD extensions including AVX2, AVX-512, VNNI, AMX, ARM-NEON, cache hierarchy, NUMA nodes, storage backing types, and GPU accelerators).
2. Deep model resource estimation (parameter counting, weight bytes across quantized and unquantized dtypes, and linear KV-cache scaling).
3. Automated execution tier classification (`FullGpu`, `HybridGpuOffload`, `CpuRam`, `NvmeOutofCore`, `Incompatible`) with primary bottleneck identification.
4. Comprehensive benchmarking engine measuring quantized kernel throughput (GEMV/MatVec TOPS/GFLOPS), host memory copy/read bandwidth (GB/s), prefill latency (Time-to-First-Token / TTFT), and autoregressive decode latency (Time-per-Output-Token / TPOT) with full P50/P90/P99 percentile statistics.
5. Control-plane HTTP API endpoints (`/api/v1/hardware/compatibility`, `/api/v1/benchmark`) and `aabhas-cli` diagnostic commands (`hwcheck`, `benchmark`).

## Delivered contract

### 1. Hardware Compatibility Engine (`hardware/hw_compatibility.hpp`, `hardware/hw_compatibility.cpp`)

- **Instruction Set & SIMD Profiling (`CpuFeatures`)**:
  - Dynamically inspects CPU features via native CPUID leaves and kernel telemetry:
    - Base vector & floating-point: `SSE4.2`, `AVX`, `AVX2`, `FMA3`, `F16C`.
    - 512-bit vector extensions: `AVX-512F`, `AVX-512CD`, `AVX-512BW`, `AVX-512DQ`, `AVX-512VL`.
    - Deep learning vector acceleration: `AVX-512-VNNI` (Vector Neural Network Instructions).
    - Advanced Matrix Extensions: `AMX-TILE`, `AMX-INT8`, `AMX-BF16`.
    - ARM architectures: `ARM-NEON`, `ARM-SVE`.
- **System Profile Assembly (`HardwareProfile`)**:
  - Aggregates host OS specifications, CPU cores, cache sizes (L1d, L1i, L2, L3), NUMA topology, storage device specs (NVMe, SSD, HDD), and discovered hardware/virtual accelerators.
- **Model Resource Estimation (`ModelResourceEstimate`)**:
  - Accurate parameter counting:
    $$\text{Params} = V \cdot d + L \left[ d(h \cdot d_k + 2h_{kv} \cdot d_k + d) + d + 3d \cdot d_{ff} + d \right] + d + (V \cdot d \text{ if untied})$$
  - Weight memory accounting based on `TensorDType`:
    - `Q4_0`: 0.5625 bytes/param
    - `Q4_1`: 0.6250 bytes/param
    - `Q5_0`: 0.6875 bytes/param
    - `Q5_1`: 0.7500 bytes/param
    - `Q8_0`: 1.0625 bytes/param
    - `F16` / `BF16`: 2.0000 bytes/param
    - `F32`: 4.0000 bytes/param
  - KV-cache scaling: $2 \times L \times h_{kv} \times d_k \times 4\text{ bytes/token}$ (e.g. 1 MiB/token for LLaMA-7B, 2 GiB at 2,048 tokens).
- **Execution Tier Classification (`HardwareCompatibilityEvaluator`)**:
  - `FullGpu`: VRAM accommodates weights, KV cache, and working activations (Score: 95).
  - `HybridGpuOffload`: VRAM accommodates at least half the weights; remainder streamed via host memory (Score: 80).
  - `CpuRam`: System RAM fits weights, KV cache, and activations; vectorized execution via AVX2/AVX-512 (Score: 75-85).
  - `NvmeOutofCore`: RAM cannot hold full weights, but storage has space; layer drops streamed on-demand from NVMe (Score: 45-65, Recommended batch size: 1).
  - `Incompatible`: Neither memory nor storage can host the model.
  - Primary bottleneck diagnostics: `None`, `VramCapacity`, `SystemMemory`, `StorageIops`, `CpuVectorWidth`, `HostBandwidth`.

### 2. High-Precision Benchmarking Engine (`inference/benchmark_engine.hpp`, `inference/benchmark_engine.cpp`)

- **Kernel Micro-benchmarks (`BenchmarkEngine::benchmark_kernels`)**:
  - Measures latency ($\mu\text{s}$), computational throughput (GFLOPS/TOPS), and memory bandwidth (GB/s) for dot products across `FP32`, `FP16`, `Q4_0`, `Q4_1`, and `Q8_0` representations at realistic matrix dimensions ($D = 4096$).
- **Memory Bandwidth Micro-benchmarks (`BenchmarkEngine::benchmark_memory`)**:
  - Evaluates host memory sequential copy bandwidth (GB/s), sequential read bandwidth (GB/s), and sequential write bandwidth (GB/s).
- **Inference Prefill TTFT Benchmark (`BenchmarkEngine::benchmark_prefill`)**:
  - Times Time-to-First-Token across warm-up and measured iterations.
  - Computes prompt processing throughput (tokens/second), mean TTFT, min/max, and P50, P90, P99 percentiles.
- **Inference Decode TPOT Benchmark (`BenchmarkEngine::benchmark_decode`)**:
  - Times Time-per-Output-Token per token step using streaming callbacks.
  - Computes generation throughput (tokens/second), mean TPOT, and inter-token latency distribution percentiles (P50, P90, P99).
- **Batch Concurrency Scaling (`BenchmarkEngine::benchmark_batching`)**:
  - Evaluates continuous batch scheduling throughput across batch sizes $B \in \{1, 2, 4\}$.
  - Computes scaling speedup ratio and hardware efficiency percentage.
- **Reporting & Serialization (`BenchmarkSuiteReport`)**:
  - JSON serialization (`to_json()`) and terminal-friendly ASCII summary tables (`summary_table()`).

### 3. Control-Plane Integration (`control-plane/internal/orchestrator/`, `control-plane/internal/api/`, `cmd/aabhas-cli/`)

- **Go Hardware & Compatibility Evaluator (`hardware_eval.go`)**:
  - Re-implements hardware evaluation and tier recommendation in Go for distributed control planes.
- **Go Synthetic Benchmark Suite (`benchmark.go`)**:
  - Executes quick memory bandwidth tests, synthetic kernel benchmarks, and percentile statistics.
- **REST API Endpoints (`rest_server.go`)**:
  - `GET /api/v1/hardware/compatibility`: Returns system hardware capabilities and model execution feasibility report.
  - `POST /api/v1/benchmark`: Triggers on-demand benchmark suite and returns structured performance metrics.
- **CLI Subcommands (`aabhas-cli`)**:
  - `aabhas-cli hwcheck`: Queries daemon for hardware profile and compatibility assessment.
  - `aabhas-cli benchmark`: Triggers performance benchmarking suite.

## Verification

1. **CTest (`core-engine/tests/hw_compatibility_test.cpp`)**:
   - `test_cpu_features_detection`: CPUID leaf queries and feature string generation.
   - `test_system_profile_assembly`: Host, cache, storage, and accelerator enumeration.
   - `test_model_resource_estimation`: Accurate parameter counts and memory footprint for LLaMA-7B across Q4_0 and F16.
   - `test_compatibility_evaluation_tiers`: Verifies selection of `FullGpu`, `HybridGpuOffload`, `CpuRam`, `NvmeOutofCore`, and `Incompatible` tiers under simulated constraint profiles.
   - `test_json_and_string_serialization`: Validates JSON output formatting.

2. **CTest (`core-engine/tests/benchmark_test.cpp`)**:
   - `test_percentile_calculation`: Validates nearest-rank percentile calculation for P0, P50, P90, P100.
   - `test_kernel_benchmarks`: Verifies positive throughput and latency for FP32, FP16, Q4_0, Q4_1, and Q8_0 kernels.
   - `test_memory_benchmarks`: Verifies copy, read, and write memory bandwidth measurements.
   - `test_inference_prefill_and_decode_benchmarks`: Prefill TTFT and decode TPOT against synthetic model runtime.
   - `test_batch_scaling_benchmark`: Concurrency scaling across batch sizes 1, 2, and 4.
   - `test_full_suite_report`: JSON and table formatting validation.

3. **Go Test Suite (`control-plane/internal/orchestrator/`)**:
   - `hardware_eval_test.go`: Parameter estimation, tier selection, and JSON formatting.
   - `benchmark_test.go`: Percentile calculation and synthetic benchmark execution.
   - Verified 100% pass under `go test -race -count=1 ./...` and `go vet ./...`.

## Boundary

This phase establishes benchmarking and hardware compatibility matrices. Production identity, cryptographic key management, and final release validation are deferred to Phase 10.
