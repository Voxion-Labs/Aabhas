# Product Phase 4: Quantized CPU Kernels

## Objective

Deliver high-performance, finite-checked quantized CPU kernels and streaming
dequantization paths for standard GGML and float-compressed weight formats,
allowing full decoder layers and matrix-vector operations to execute directly
from quantized model indices without materializing redundant model-scale buffers.

## Delivered contract

1. IEEE 754 half-precision float (`F16`) and BFloat16 (`BF16`) scalar conversion
   primitives support normalized numbers, subnormals, zeros, infinities, and
   NaN values without standard library undefined behavior.
2. Standard GGML block layout specifications and dequantization kernels:
   - `Q4_0`: 32 values per 18-byte block (fp16 delta, 16 4-bit nibbles).
   - `Q4_1`: 32 values per 20-byte block (fp16 delta, fp16 min, 16 4-bit nibbles).
   - `Q5_0`: 32 values per 22-byte block (fp16 delta, 32-bit high-mask `qh`, 16 4-bit nibbles).
   - `Q5_1`: 32 values per 24-byte block (fp16 delta, fp16 min, 32-bit high-mask `qh`, 16 4-bit nibbles).
   - `Q8_0`: 32 values per 34-byte block (fp16 delta, 32 signed int8 values).
   - `Q8_1`: 32 values per 40-byte block (float32 delta, float32 sum, 32 signed int8 values).
3. Direct vector dot-product (`vec_dot`) for every supported data type accumulates
   intermediate block products directly into double-precision registers, avoiding
   full intermediate float vector allocation.
4. Matrix-vector kernel `gemv_quantized` multiplies a quantized row-major matrix
   $[rows \times columns]$ with an input float vector in $O(1)$ scratch memory.
5. `QuantizedStreamDecoder` provides a bounded stream-fed decoder that consumes
   scheduler chunks of arbitrary size (including unaligned chunk boundaries) and
   dequantizes complete blocks directly into destination memory or invokes a block
   callback for streamed dot-product accumulation.
6. `InferenceRuntime::run_matvec` streams and computes matrix-vector multiplication
   for any supported data type (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`,
   `Q5_1`, `Q8_0`, `Q8_1`). `run_f32_matvec` remains for backward compatibility.
7. `InferenceRuntime::stream_tensor` streams and dequantizes any supported weight
   format into a caller-owned float32 arena.
8. `build_transformer_layer_plan` resolves transformer block tensor bindings for
   both full-precision and quantized weights, and `load_transformer_layer`
   streams and dequantizes them directly into the layer's bounded execution arena.
   Downstream `transformer_block_forward` executes without changes.

## Verification

CTest verifies:
- FP16 and BF16 scalar round-trip and subnormal conversion accuracy.
- Analytical block dequantization for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and Q8_1 against known reference tables.
- Symmetric and asymmetric quantization round-trip error bounds.
- Direct `vec_dot` and `gemv_quantized` against reference float products.
- `QuantizedStreamDecoder` streaming across arbitrary and unaligned chunk boundaries (e.g., 1, 3, 7, 17, 34 bytes).
- Streamed runtime matrix-vector multiplication through `InferenceRuntime`.
- Full end-to-end quantized transformer layer loading (mixed Q8_0, Q4_0, and F32) and forward execution through `transformer_block_forward` and `KvCache`.

## Boundary

This phase delivers quantized CPU kernels and streamed layer dequantization into
the decoder execution arena. Dynamic KV-cache quantization, token generation loop,
sampling strategies (temperature, top-p, min-p, top-k), streaming HTTP responses,
and GPU/VRAM backends remain subsequent product phases.