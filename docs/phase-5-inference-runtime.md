# Phase 5: Streamed CPU Inference Runtime Foundation

## Objective

Provide a bounded compute layer that consumes tensor windows from the
out-of-core scheduler instead of materializing model weights in host memory.

## Delivered contract

1. Tensor descriptors carry shape and dtype metadata for GGUF and SafeTensors.
2. `InferenceRuntime` executes streamed row-major F32 matrix-vector tensors
   across multiple bounded windows.
3. The runtime owns a bounded float32 KV-cache with checked allocation size,
   indexed append, and read access for autoregressive attention state.
4. Runtime tests verify a multi-window matrix-vector result and KV-cache bounds.
5. Unsupported dtypes remain explicit and are rejected by the F32 kernel rather
   than silently reinterpreted.

## Explicit non-goals

This phase does not claim a complete transformer graph, tokenizer, quantized
kernel suite, or production model compatibility matrix. Those require the
architecture-specific execution work in later phases.
