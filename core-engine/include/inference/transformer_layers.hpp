#ifndef AABHAS_INFERENCE_TRANSFORMER_LAYERS_HPP
#define AABHAS_INFERENCE_TRANSFORMER_LAYERS_HPP

#include <cstddef>
#include <cstdint>

namespace aabhas::inference {

struct TransformerShape {
    std::size_t model_dimension{0};
    std::size_t head_count{0};
    std::size_t head_dimension{0};
    std::size_t kv_head_count{0};
    std::size_t intermediate_dimension{0};
    std::size_t max_sequence_length{0};
};

/** Validate a transformer shape and its bounded float32 KV-cache footprint. */
bool validate_transformer_shape(const TransformerShape& shape,
                               std::size_t max_kv_bytes,
                               std::size_t& required_kv_bytes) noexcept;

namespace kernels {

/** Apply RMSNorm to one hidden-state vector. Input and output may alias. */
bool rms_norm(const float* input, const float* weight, std::size_t width,
              float epsilon, float* output) noexcept;

/** Apply element-wise SiLU. Input and output may alias. */
bool silu(const float* input, std::size_t width, float* output) noexcept;

/** Apply the gated SiLU feed-forward operation: SiLU(gate) * up. */
bool swiglu(const float* gate, const float* up, std::size_t width,
            float* output) noexcept;

/** Rotate one query and optionally one key head in-place using RoPE. */
bool apply_rope(float* query, float* key, std::size_t head_dimension,
                std::size_t position, float theta) noexcept;

/** Compute stable single-head causal attention over an existing KV prefix. */
bool causal_attention(const float* query, const float* keys, const float* values,
                      std::size_t token_count, std::size_t head_dimension,
                      float scale, float* output) noexcept;

} // namespace kernels
} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_TRANSFORMER_LAYERS_HPP
