# Product Phase 7: GPU/VRAM Backend

## Objective

Deliver a high-performance GPU/VRAM hardware abstraction and execution backend
capable of device enumeration, strict VRAM memory budget enforcement, unified device
allocator management, accelerated compute kernels, and hybrid CPU/GPU layer offloading
with full headless CI/CD fallback support.

## Delivered contract

1. Hardware Discovery & Device Registry (`hardware/gpu_device.hpp`, `hardware/gpu_device.cpp`):
   - `GpuBackendType`: Supports `CUDA`, `Vulkan`, `ROCm`, and `Emulated` backends.
   - `GpuDeviceInfo`: Reports device ID, vendor name, device name, total VRAM, available VRAM, compute units, maximum workgroup size, and FP16/BF16 support flags.
   - Dynamic host discovery:
     - Linux DRM subsystem inspection (`/sys/class/drm/card*/device`) with PCI vendor detection (NVIDIA `0x10de`, AMD `0x1002`, Intel `0x8086`).
     - Discrete NVIDIA driver detection via `/dev/nvidia*` and `/proc/driver/nvidia/version`.
     - Always guarantees a deterministic `Emulated` compute device for headless CI/CD environments and machines without dedicated GPU silicon.

2. Unified VRAM Allocator (`inference/vram_allocator.hpp`, `inference/vram_allocator.cpp`):
   - Strict budget enforcement (`VramConfig`): Rejects allocations that would exceed configured VRAM budgets, preventing out-of-memory host crashes.
   - `DeviceBuffer`: RAII handle representing device-allocated memory with explicit byte tracking and safe move semantics.
   - Coalesced memory alignment (64-byte alignment default).
   - High-throughput synchronous and asynchronous Host-to-Device (H2D) and Device-to-Host (D2H) memory streaming with bounds checking.
   - Fast device zeroing (`DeviceBuffer::zero`).
   - Detailed telemetry tracking allocated bytes, peak watermark, active allocation count, and total allocation operations.

3. Accelerated GPU Backend & Kernels (`inference/gpu_backend.hpp`, `inference/gpu_backend.cpp`):
   - Device compute kernels:
     - Accelerated GEMV (matrix-vector multiplication): Vectorized loop accumulation across weight matrices and input activation vectors.
     - Accelerated RMSNorm: Numerically stable sum-of-squares reduction and reciprocal square root scaling.
     - Accelerated SwiGLU: Fused gate-up elementwise activation.
     - Accelerated Rotary Position Embedding (RoPE): In-place complex rotation across head dimensions.
   - Full transformer layer execution on device: `execute_layer_forward` stages and executes complete decoder blocks on device VRAM.

4. Hybrid Layer Offloading in Engine Runtime:
   - `InferenceConfig` extended with `GpuConfig`:
     - `gpu_layers`: Configures how many transformer layers are offloaded into device VRAM (0 = pure CPU, $N$ = hybrid, all = pure GPU).
     - `vram.budget_bytes`: Restricts maximum VRAM consumption.
   - `TokenGenerator` automatically streams and uploads designated layers to VRAM during `initialize()`.
   - `forward_step` routes offloaded layers to `GpuBackend` while streaming non-offloaded layers through the CPU scheduler, seamlessly transferring intermediate activation states.

## Verification

CTest (`tests/gpu_backend_test.cpp`) verifies:
- Device enumeration and fallback default device retrieval.
- VRAM allocator bounded budget enforcement, out-of-memory rejection, and memory reclamation.
- Bidirectional Host-to-Device (H2D) and Device-to-Host (D2H) roundtrip data integrity with bit-exact float verification.
- Device compute kernels: GEMV, RMSNorm, and SwiGLU outputs match mathematical specifications.
- End-to-end hybrid layer execution: 2-layer model with Layer 0 executing in device VRAM and Layer 1 executing on CPU, verifying forward logit calculation and autoregressive multi-token generation.
- Full resource teardown and zero memory leakage upon session close.

## Boundary

This phase establishes the GPU/VRAM hardware abstraction, device memory management, and
hybrid layer offloading. Multi-model scheduling and concurrent batching, hardware benchmarking,
and final release identity validation remain subsequent product phases.
