# Product Phase 3: Transformer Layer Execution

## Objective

Turn the bounded tokenizer and scalar kernels into an executable decoder
transformer block without materializing a whole model or bypassing the
out-of-core scheduler.

## Delivered contract

1. `TransformerBlockConfig` validates head geometry, grouped-query divisibility,
   RoPE parameters, RMSNorm epsilon, attention scale, and the exact bounded
   scratch requirement before execution begins.
2. RMSNorm, SiLU, SwiGLU, and RoPE remain finite-checked, allocation-free
   primitives. RMSNorm accumulates in double precision before writing float32.
3. `linear_f32` provides a checked row-major matrix-vector primitive, and
   `causal_attention_strided` performs stable max-subtracted softmax over
   strided key/value rows so a packed KV cache can be consumed without copying.
4. Multi-head causal attention dispatches each query head to its grouped-query
   key/value head, consumes the complete causal prefix, and appends the new key
   and value rows to the bounded `KvCache` at the requested layer and position.
5. `transformer_block_forward` executes the complete pre-norm decoder block:
   attention norm, Q/K/V projections, RoPE, KV append, grouped-query attention,
   output projection and residual, feed-forward norm, gate/up projections,
   SwiGLU, down projection and residual. Multi-token prefill and single-token
   decode use the same code path.
6. `llama_transformer_layer_naming` maps standard GGUF (`blk.N.*`) and
   SafeTensors/Hugging Face (`model.layers.N.*`) tensor names.
   `build_transformer_layer_plan` verifies that every tensor is float32 with the
   exact required shape, rejects duplicate or malformed bindings, and computes a
   checked non-overlapping arena layout.
7. `InferenceRuntime::stream_f32_tensor` copies a scheduled tensor through
   bounded layer windows into a caller-owned arena without a second full-model
   allocation. `load_transformer_layer` uses it to build non-owning weight views,
   and `rewind_stream` restarts the sequence for another pass.
8. `KvCache` is a separate bounded component so execution code can share cache
   state without depending on the whole runtime header.

## Verification

CTest covers primitive numerics, strided attention equivalence, grouped-query
block output against a fixed reference, KV-cache contents after prefill and
decode, batch-versus-sequential equivalence, invalid cache geometry, scratch
under-allocation, GGUF and SafeTensors plan resolution, malformed shape and
duplicate-name rejection, streamed layer loading, and scheduler rewind.

## Boundary

Weights must still be float32 in this phase. Quantized dequantization kernels,
fallback CPU variants, sampling, token-loop generation, streaming HTTP
responses, and GPU/VRAM execution remain later product phases. The block
implementation assumes standard Llama-style bias-free projections and does not
yet support fused QKV tensors, sliding-window attention, or mixture-of-experts
routing.
