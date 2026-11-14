# Product Phase 3: Transformer Layer Kernels

## Objective

Establish the numerically safe CPU primitives needed by a transformer block
without allocating a full model or bypassing the out-of-core scheduler.

## Delivered contract

1. Transformer shape validation checks head geometry and bounded float32 KV
   cache requirements with overflow protection.
2. RMSNorm validates finite inputs and computes in double precision before
   writing the bounded float32 output.
3. SiLU and SwiGLU feed-forward primitives are available with finite-value
   rejection.
4. RoPE rotates query and optional key heads in-place with checked even head
   dimensions and bounded numeric parameters.
5. Stable single-head causal attention uses a max-subtracted softmax and
   consumes an existing contiguous KV prefix without allocating logits.
6. CTest coverage verifies shape rejection and deterministic numeric outputs.

## Boundary

These kernels are allocation-free building blocks. Weight streaming, tensor
layout-specific dequantization, multi-head/grouped-query dispatch, batching,
and GPU execution remain subsequent product phases.
