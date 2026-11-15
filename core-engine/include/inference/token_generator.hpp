#ifndef AABHAS_INFERENCE_TOKEN_GENERATOR_HPP
#define AABHAS_INFERENCE_TOKEN_GENERATOR_HPP

#include "inference/kv_cache.hpp"
#include "inference/sampler.hpp"
#include "inference/transformer_layers.hpp"
#include "model/model_index.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aabhas::inference {

class InferenceRuntime;

/** Complete decoder transformer model architectural configuration. */
struct TransformerModelConfig {
    std::size_t vocab_size{0};
    std::size_t model_dimension{0};
    std::size_t layer_count{0};
    TransformerBlockConfig block{};
    std::int64_t bos_token_id{-1};
    std::int64_t eos_token_id{-1};
};

/** Tensor naming convention across full model (embedding, layers, norm, lm_head). */
struct TransformerModelNaming {
    std::string token_embedding;
    std::string output_norm;
    std::string lm_head;
    std::vector<TransformerLayerNaming> layers;
};

/** Populate standard LLaMA-style GGUF or SafeTensors tensor names for full model. */
bool llama_transformer_model_naming(aabhas::model::ModelFormat format,
                                    std::size_t layer_count,
                                    TransformerModelNaming& naming);

/** Checked full model execution plan resolving all tensors and layer sub-plans. */
struct TransformerModelPlan {
    TransformerTensorBinding token_embedding;
    TransformerTensorBinding output_norm;
    TransformerTensorBinding lm_head;
    bool lm_head_tied_to_embedding{false};
    std::vector<TransformerLayerPlan> layers;
    std::size_t layer_arena_floats{0};
};

/**
 * Resolve tensor names against model metadata and verify layout for full model.
 */
bool build_transformer_model_plan(const aabhas::model::ModelIndex& model,
                                  const TransformerModelNaming& naming,
                                  const TransformerModelConfig& config,
                                  TransformerModelPlan& plan);

/** Overload using format-derived LLaMA naming. */
bool build_transformer_model_plan(const aabhas::model::ModelIndex& model,
                                  aabhas::model::ModelFormat format,
                                  const TransformerModelConfig& config,
                                  TransformerModelPlan& plan);

/** Generation parameters controlling stop tokens, penalties, and limits. */
struct GenerationOptions {
    std::size_t max_new_tokens{128};
    bool early_stopping{true};
    float repetition_penalty{1.0F};
    SamplerConfig sampler{};
    std::vector<std::uint32_t> stop_token_ids{};
};

/** Comprehensive result of token generation run. */
struct GenerationResult {
    std::vector<std::uint32_t> prompt_tokens;
    std::vector<std::uint32_t> generated_tokens;
    std::vector<std::uint32_t> all_tokens;
    std::string text;
    bool reached_eos{false};
    std::size_t prompt_token_count{0};
    std::size_t generated_token_count{0};
};

/**
 * High-performance autoregressive token generation engine.
 *
 * Implements batched prompt prefill, single-token decode loop, KV-cache
 * management, embedding lookup, layer forward passes, final normalization,
 * logit calculation, and greedy token selection.
 *
 * All execution buffers are pre-allocated during initialize() so that
 * the autoregressive generation loop performs zero heap allocations.
 */
class TokenGenerator final {
public:
    TokenGenerator() noexcept = default;
    ~TokenGenerator() noexcept = default;
    TokenGenerator(const TokenGenerator&) = delete;
    TokenGenerator& operator=(const TokenGenerator&) = delete;

    /**
     * Initialize execution buffers, pre-load static weights (embedding, norm, lm_head,
     * and layers if memory allows), and configure the bounded KV-cache.
     */
    bool initialize(InferenceRuntime* runtime,
                    const TransformerModelPlan& plan,
                    const TransformerModelConfig& config,
                    std::size_t max_sequence_length = 2048);

    /**
     * Run prompt prefill and iterative decode steps up to options.max_new_tokens.
     * Optionally delivers incremental chunks via stream_callback.
     */
    bool generate(const std::vector<std::uint32_t>& prompt_tokens,
                  const GenerationOptions& options,
                  GenerationResult& result,
                  StreamCallback stream_callback = nullptr);

    /**
     * Batch generation across multiple prompt token sequences concurrently.
     * Evaluates prompts and steps decodes across all sequences in parallel.
     */
    bool generate_batch(const std::vector<std::vector<std::uint32_t>>& prompt_batch,
                        const std::vector<GenerationOptions>& options_batch,
                        std::vector<GenerationResult>& results_batch,
                        std::vector<StreamCallback> stream_callbacks = {});

    /**
     * End-to-end string prompt tokenization, generation, and text detokenization.
     * Optionally delivers incremental chunks via stream_callback.
     */
    bool generate_text(std::string_view prompt,
                       const GenerationOptions& options,
                       GenerationResult& result,
                       StreamCallback stream_callback = nullptr);

    /**
     * Execute a forward pass for a sequence of tokens at the given start position using
     * the generator's internal KV-cache.
     * Computes logits for the final token in the slice into `logits_out` ([vocab_size]).
     */
    bool forward_step(const std::uint32_t* tokens, std::size_t token_count,
                      std::size_t start_position, float* logits_out);

    /**
     * Execute a forward pass for a sequence of tokens with an explicit caller-supplied KV-cache.
     * Enables multi-sequence execution and dynamic batching without mutating internal generator KV-cache.
     */
    bool forward_step(const std::uint32_t* tokens, std::size_t token_count,
                      std::size_t start_position, float* logits_out,
                      KvCache& kv_cache);

    /** Clear KV-cache and reset current sequence position to 0. */
    void reset() noexcept;

    const KvCache& kv_cache() const noexcept { return kv_cache_; }
    KvCache& kv_cache() noexcept { return kv_cache_; }
    std::size_t current_position() const noexcept { return current_position_; }
    const TransformerModelConfig& config() const noexcept { return config_; }
    const TransformerModelPlan& plan() const noexcept { return plan_; }
    Sampler& sampler() noexcept { return sampler_; }
    const Sampler& sampler() const noexcept { return sampler_; }
    bool is_initialized() const noexcept { return runtime_ != nullptr; }

private:
    InferenceRuntime* runtime_{nullptr};
    TransformerModelPlan plan_{};
    TransformerModelConfig config_{};
    KvCache kv_cache_{};
    Sampler sampler_{};
    std::size_t max_sequence_length_{0};
    std::size_t current_position_{0};

    // Pre-allocated buffers ensuring zero heap allocations in generation loop
    std::vector<float> token_embedding_table_{};
    std::vector<float> output_norm_table_{};
    std::vector<float> lm_head_table_{};
    std::vector<float> hidden_states_{};
    std::vector<float> hidden_scratch_{};
    std::vector<float> layer_arena_{};
    std::vector<float> block_scratch_{};
    std::vector<float> logits_{};
    std::vector<std::vector<float>> layer_weight_arenas_{};
    bool weights_cached_{false};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_TOKEN_GENERATOR_HPP
