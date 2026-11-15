# Product Phase 5: KV-Cache and Token Generation

## Objective

Deliver an autoregressive token generation pipeline and enhanced KV-cache
foundation capable of prompt prefill, iterative single-token decoding, sequence
tracking, and greedy logit selection with zero dynamic memory allocation during
generation loops.

## Delivered contract

1. Enhanced `KvCache` management:
   - Sequence length and position tracking (`sequence_length()`, `set_sequence_length()`, `advance_sequence()`).
   - Contiguous layer-wide key and value pointers (`layer_keys()`, `layer_values()`) with fixed token stride for zero-copy attention passes.
   - Batch sequence appending (`append_sequence`) allowing multi-token prompt prefill to write directly to cache buffers.
   - Sequence truncation and rollback (`truncate()`, `reset_sequence()`) for prefix caching and prompt context reuse.
   - Strict bounded allocation enforcing `max_bytes` resource caps.
2. `TransformerModelConfig` and `TransformerModelPlan`:
   - Full model architectural configuration (vocabulary size, hidden dimension, layer count, block geometry, BOS/EOS token IDs).
   - `llama_transformer_model_naming` maps GGUF (`token_embd.weight`, `output_norm.weight`, `output.weight`, `blk.N.*`) and SafeTensors (`model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`, `model.layers.N.*`) conventions.
   - `build_transformer_model_plan` validates tensor shapes, layouts, and data types across the entire model and handles tied embedding weights when explicit LM heads are omitted.
3. Pre-allocated `TokenGenerator` execution engine:
   - Pre-allocates hidden states, layer scratch, block scratch, and logit buffers during `initialize()`, guaranteeing zero heap allocations during autoregressive generation loops.
   - Pre-loads static embedding tables and output normalization weights.
   - Caches layer weight arenas in-memory when permitted by host limits, while seamlessly streaming layer windows when memory is constrained.
4. Prefill and decode execution:
   - `forward_step` performs embedding lookup, sequentially executes all transformer decoder layers via `transformer_block_forward` with KV-cache appends, applies final RMSNorm, and projects to unnormalized logits via `linear_f32`.
   - `generate` executes the full prompt prefill pass followed by an autoregressive single-token decode loop up to `max_new_tokens`.
   - Supports early stopping upon EOS token generation or custom stop token IDs.
   - Optional repetition penalty downweights previously generated token logits.
5. High-level runtime integration:
   - `InferenceRuntime::configure_model`, `generate`, and `generate_text` provide clean, idiomatic entry points linking tokenization, generation, and detokenization.

## Verification

CTest verifies:
- KV-cache sequence appending, layer-wide contiguous slice retrieval, truncation, and boundary violation rejection.
- Model plan resolution for GGUF and SafeTensors naming conventions, including tied embedding weights.
- Single-step forward logit calculation and numerical finiteness.
- End-to-end multi-step autoregressive generation matching prompt and target token sequences.
- End-to-end string prompt tokenization to detokenized text generation.
- Early stopping upon EOS or custom stop token generation.
- Repetition penalty logit dampening.

## Boundary

This phase establishes the autoregressive generation pipeline and greedy token
selection. Advanced stochastic sampling strategies (temperature, top-k, top-p,
min-p), streaming chunked HTTP responses, and multi-sequence concurrent batching
remain subsequent product phases.