#ifndef AABHAS_INFERENCE_TRANSFORMER_LAYERS_HPP
#define AABHAS_INFERENCE_TRANSFORMER_LAYERS_HPP

#include "inference/kv_cache.hpp"
#include "model/model_index.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

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

/** Runtime parameters for one decoder transformer block. */
struct TransformerBlockConfig {
    std::size_t model_dimension{0};
    std::size_t head_count{0};
    std::size_t kv_head_count{0};
    std::size_t head_dimension{0};
    std::size_t intermediate_dimension{0};
    float rms_epsilon{1.0e-5F};
    float rope_theta{10000.0F};
    /** Zero selects 1 / sqrt(head_dimension). */
    float attention_scale{0.0F};
};

/** Non-owning row-major float32 views of one decoder block's weights. */
struct TransformerLayerWeights {
    const float* attention_norm{nullptr};
    const float* attention_query{nullptr};
    const float* attention_key{nullptr};
    const float* attention_value{nullptr};
    const float* attention_output{nullptr};
    const float* feed_forward_norm{nullptr};
    const float* feed_forward_gate{nullptr};
    const float* feed_forward_up{nullptr};
    const float* feed_forward_down{nullptr};
};

/** Tensor names for a standard Llama-style decoder block. */
struct TransformerLayerNaming {
    std::string attention_norm;
    std::string attention_query;
    std::string attention_key;
    std::string attention_value;
    std::string attention_output;
    std::string feed_forward_norm;
    std::string feed_forward_gate;
    std::string feed_forward_up;
    std::string feed_forward_down;
};

/**
 * Populate GGUF (`blk.N.*`) or SafeTensors/Hugging Face (`model.layers.N.*`)
 * tensor names. Raw binary models have no naming convention and are rejected.
 */
bool llama_transformer_layer_naming(aabhas::model::ModelFormat format,
                                    std::size_t layer_index,
                                    TransformerLayerNaming& naming);

struct TransformerTensorBinding {
    std::size_t tensor_index{0};
    std::size_t offset_floats{0};
    std::size_t float_count{0};
};

/** Checked float32 arena layout for every tensor in one decoder block. */
struct TransformerLayerPlan {
    TransformerLayerNaming naming;
    TransformerTensorBinding attention_norm;
    TransformerTensorBinding attention_query;
    TransformerTensorBinding attention_key;
    TransformerTensorBinding attention_value;
    TransformerTensorBinding attention_output;
    TransformerTensorBinding feed_forward_norm;
    TransformerTensorBinding feed_forward_gate;
    TransformerTensorBinding feed_forward_up;
    TransformerTensorBinding feed_forward_down;
    std::size_t total_floats{0};
};

/**
 * Resolve a naming convention against a model index and verify that every
 * tensor is float32 with the exact shape required by the block configuration.
 */
bool build_transformer_layer_plan(const aabhas::model::ModelIndex& model,
                                  const TransformerLayerNaming& naming,
                                  const TransformerBlockConfig& config,
                                  TransformerLayerPlan& plan);

/** Bind non-owning weight views into a plan-sized arena. */
TransformerLayerWeights bind_transformer_layer_plan(const TransformerLayerPlan& plan,
                                                    const float* arena) noexcept;

/** Exact scratch requirement for one forward call, or zero on invalid shape. */
std::size_t transformer_block_scratch_floats(const TransformerBlockConfig& config) noexcept;

/**
 * Execute one pre-norm decoder block for `token_count` consecutive tokens.
 *
 * Inputs and outputs are row-major `[token_count, model_dimension]`. Keys and
 * values for `layer_index` are appended to `cache` at `start_position` and
 * attention consumes the complete causal prefix already present in the cache.
 * The function is allocation-free and returns false without partial success.
 */
bool transformer_block_forward(const TransformerBlockConfig& config,
                               const TransformerLayerWeights& weights,
                               std::size_t layer_index,
                               std::size_t start_position,
                               const float* input,
                               std::size_t token_count,
                               KvCache& cache,
                               float* output,
                               float* scratch,
                               std::size_t scratch_floats) noexcept;

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

/** Multiply a row-major `[rows, columns]` float32 matrix by one vector. */
bool linear_f32(const float* weights, std::size_t rows, std::size_t columns,
                const float* input, float* output) noexcept;

/** Causal attention over strided key/value rows. */
bool causal_attention_strided(const float* query, const float* keys,
                              const float* values, std::size_t token_count,
                              std::size_t head_dimension, std::size_t key_stride,
                              std::size_t value_stride, float scale,
                              float* output) noexcept;

/** Compute stable single-head causal attention over an existing KV prefix. */
bool causal_attention(const float* query, const float* keys, const float* values,
                      std::size_t token_count, std::size_t head_dimension,
                      float scale, float* output) noexcept;

} // namespace kernels
} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_TRANSFORMER_LAYERS_HPP
