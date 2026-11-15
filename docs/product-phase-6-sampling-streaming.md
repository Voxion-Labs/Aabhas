# Product Phase 6: Sampling and Streaming Responses

## Objective

Deliver production-grade stochastic sampling strategies and an asynchronous token
streaming pipeline capable of progressive UTF-8 response emission, client cancellation,
reproducible seed control, and multi-tier logit filtering (temperature, Top-K, Top-P,
Min-P, repetition, frequency, presence penalties) with zero dynamic memory allocation
during autoregressive decode loops.

## Delivered contract

1. Zero-Allocation `Sampler` (`sampler.hpp`, `sampler.cpp`):
   - **Temperature Scaling**: `logits[i] / T` with fast greedy argmax short-circuit when $T \le 1.0\times 10^{-4}$ or $\text{top\_k} = 1$.
   - **Multi-Penalty Application**:
     - *Repetition penalty*: Logits $> 0$ are divided by penalty; logits $< 0$ are multiplied by penalty.
     - *Presence penalty*: Flat penalty subtracted from logits of tokens appearing in context history.
     - *Frequency penalty*: Scaled penalty subtracted proportionally to token occurrence frequency in context history.
     - Zero heap allocations using pre-allocated index tracker vectors with $O(|\text{history}|)$ reset cost.
   - **Top-K Truncation**: Bounded partial sorting via `std::partial_sort` retaining top $K$ largest candidates in descending order with zero allocations.
   - **Top-P (Nucleus) Filtering**: Dynamic cumulative probability mass cutoff retaining the minimal set of candidates with cumulative probability $\ge P$, with probability re-normalization.
   - **Min-P Adaptive Thresholding**: Dynamic cutoff discarding any candidate whose probability falls below $\text{min\_p} \times p_{\max}$.
   - **Numerically Stable Softmax**: Max-subtracted exponential summation robust against dynamic range overflow or $-\infty$ values.
   - **Deterministic Categorical PRNG**: Seedable 64-bit Mersenne Twister engine (`std::mt19937_64`) for reproducible stochastic token selection.

2. Streaming Response Pipeline (`StreamCallback`, `StreamChunk`):
   - `StreamChunk` captures `token_id`, `text_delta`, `token_index`, and `is_eos`.
   - `StreamCallback` (`std::function<bool(const StreamChunk&)>`) invoked synchronously per generated token during both prompt prefill and decode iterations.
   - Robust incremental UTF-8 detokenization computing textual diffs between consecutive token sequence states, handling multi-byte code points and byte-fallback tokens cleanly.
   - Immediate early cancellation: returning `false` from `StreamCallback` terminates generation immediately without memory leaks or cache corruption.

3. Engine and Runtime Integration:
   - `TokenGenerator::generate` and `TokenGenerator::generate_text` accept optional `StreamCallback`.
   - `InferenceRuntime::generate` and `InferenceRuntime::generate_text` expose full streaming capabilities.
   - Full backward compatibility: omitting `stream_callback` defaults to standard synchronous generation.

## Verification

CTest (`tests/sampling_test.cpp`) verifies:
- Greedy argmax selection across positive, negative, and edge-case logits.
- Repetition, presence, and frequency penalty logit modifications.
- Softmax numerical stability and probability normalization under extreme logit ranges (1000.0F+).
- Top-K truncation strictly restricting candidate tokens to top $K$ positions.
- Top-P nucleus filtering concentrating sampling mass on dominant candidates.
- Min-P adaptive thresholding filtering low-confidence tail candidates.
- PRNG reproducibility: identical seeds produce byte-identical token sequences.
- End-to-end streaming token generation: progressive `text_delta` concatenation strictly equals full detokenized response text.
- Early client cancellation: returning `false` from callback halts generation on the requested token.

## Boundary

This phase completes stochastic sampling and response streaming on the data plane.
Hardware acceleration (GPU/VRAM backend with Vulkan/CUDA compute shaders), multi-sequence
concurrent batching, and hardware compatibility checks remain subsequent product phases.
