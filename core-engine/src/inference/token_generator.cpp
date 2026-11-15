#include "inference/token_generator.hpp"
#include "inference/batch_scheduler.hpp"
#include "inference/inference_runtime.hpp"
#include "inference/quantized_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace aabhas::inference {
namespace {

bool shape_is(const std::vector<std::uint64_t>& shape,
              std::initializer_list<std::uint64_t> expected) noexcept {
    if (shape.size() != expected.size()) return false;
    std::size_t index = 0;
    for (const std::uint64_t extent : expected) {
        if (shape[index] != extent) return false;
        ++index;
    }
    return true;
}

bool resolve_named_tensor(const aabhas::model::ModelIndex& model,
                          const std::string& name,
                          std::initializer_list<std::uint64_t> expected_shape,
                          TransformerTensorBinding& binding) noexcept {
    if (name.empty()) return false;
    const aabhas::model::TensorDescriptor* found = nullptr;
    for (const auto& tensor : model.tensors) {
        if (tensor.name == name) {
            found = &tensor;
            break;
        }
    }
    if (found == nullptr || !kernels::is_supported_weight_dtype(found->dtype) ||
        !shape_is(found->shape, expected_shape)) {
        return false;
    }
    aabhas::model::TensorLayout layout;
    if (!aabhas::model::describe_tensor_layout(found->dtype, found->shape, layout) ||
        layout.element_count == 0 ||
        layout.element_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        layout.byte_size > found->byte_size) {
        return false;
    }
    binding.tensor_index = static_cast<std::size_t>(found - model.tensors.data());
    binding.offset_floats = 0;
    binding.float_count = static_cast<std::size_t>(layout.element_count);
    binding.dtype = found->dtype;
    return true;
}

} // namespace

bool llama_transformer_model_naming(aabhas::model::ModelFormat format,
                                    std::size_t layer_count,
                                    TransformerModelNaming& naming) {
    if (layer_count == 0) return false;
    naming = {};
    naming.layers.resize(layer_count);

    switch (format) {
    case aabhas::model::ModelFormat::GGUF: {
        naming.token_embedding = "token_embd.weight";
        naming.output_norm = "output_norm.weight";
        naming.lm_head = "output.weight";
        for (std::size_t i = 0; i < layer_count; ++i) {
            if (!llama_transformer_layer_naming(format, i, naming.layers[i])) {
                naming = {};
                return false;
            }
        }
        return true;
    }
    case aabhas::model::ModelFormat::SafeTensors: {
        naming.token_embedding = "model.embed_tokens.weight";
        naming.output_norm = "model.norm.weight";
        naming.lm_head = "lm_head.weight";
        for (std::size_t i = 0; i < layer_count; ++i) {
            if (!llama_transformer_layer_naming(format, i, naming.layers[i])) {
                naming = {};
                return false;
            }
        }
        return true;
    }
    case aabhas::model::ModelFormat::RawBinary:
    default:
        return false;
    }
}

bool build_transformer_model_plan(const aabhas::model::ModelIndex& model,
                                  const TransformerModelNaming& naming,
                                  const TransformerModelConfig& config,
                                  TransformerModelPlan& plan) {
    if (config.vocab_size == 0 || config.model_dimension == 0 ||
        config.layer_count == 0 || naming.layers.size() != config.layer_count ||
        config.block.model_dimension != config.model_dimension) {
        return false;
    }

    plan = {};
    if (!resolve_named_tensor(model, naming.token_embedding,
                              {config.vocab_size, config.model_dimension},
                              plan.token_embedding) ||
        !resolve_named_tensor(model, naming.output_norm, {config.model_dimension},
                              plan.output_norm)) {
        plan = {};
        return false;
    }

    // Attempt to resolve explicit LM head; fall back to tied embedding weights
    if (!resolve_named_tensor(model, naming.lm_head,
                              {config.vocab_size, config.model_dimension},
                              plan.lm_head)) {
        plan.lm_head = plan.token_embedding;
        plan.lm_head_tied_to_embedding = true;
    } else {
        plan.lm_head_tied_to_embedding = false;
    }

    plan.layers.reserve(config.layer_count);
    plan.layer_arena_floats = 0;
    for (std::size_t i = 0; i < config.layer_count; ++i) {
        TransformerLayerPlan layer_plan;
        if (!build_transformer_layer_plan(model, naming.layers[i], config.block,
                                          layer_plan)) {
            plan = {};
            return false;
        }
        plan.layer_arena_floats =
            std::max(plan.layer_arena_floats, layer_plan.total_floats);
        plan.layers.push_back(std::move(layer_plan));
    }
    return plan.layer_arena_floats > 0;
}

bool build_transformer_model_plan(const aabhas::model::ModelIndex& model,
                                  aabhas::model::ModelFormat format,
                                  const TransformerModelConfig& config,
                                  TransformerModelPlan& plan) {
    TransformerModelNaming naming;
    if (!llama_transformer_model_naming(format, config.layer_count, naming)) {
        return false;
    }
    return build_transformer_model_plan(model, naming, config, plan);
}

bool TokenGenerator::initialize(InferenceRuntime* runtime,
                                const TransformerModelPlan& plan,
                                const TransformerModelConfig& config,
                                std::size_t max_sequence_length) {
    if (runtime == nullptr || plan.layers.size() != config.layer_count ||
        config.vocab_size == 0 || config.model_dimension == 0 ||
        config.layer_count == 0 || max_sequence_length == 0) {
        return false;
    }

    runtime_ = runtime;
    plan_ = plan;
    config_ = config;
    max_sequence_length_ = max_sequence_length;
    current_position_ = 0;

    const std::size_t kv_head_count = config.block.kv_head_count;
    const std::size_t head_dimension = config.block.head_dimension;
    if (!kv_cache_.configure(config.layer_count, kv_head_count, head_dimension,
                             max_sequence_length, runtime->max_kv_bytes())) {
        reset();
        return false;
    }

    const std::size_t block_scratch =
        transformer_block_scratch_floats(config.block);
    if (block_scratch == 0) {
        reset();
        return false;
    }

    if (!sampler_.configure(config.vocab_size)) {
        reset();
        return false;
    }

    try {
        block_scratch_.assign(block_scratch, 0.0F);
        logits_.assign(config.vocab_size, 0.0F);
        hidden_states_.assign(max_sequence_length * config.model_dimension, 0.0F);
        hidden_scratch_.assign(max_sequence_length * config.model_dimension, 0.0F);
        layer_arena_.assign(plan.layer_arena_floats, 0.0F);

        token_embedding_table_.assign(plan.token_embedding.float_count, 0.0F);
        output_norm_table_.assign(plan.output_norm.float_count, 0.0F);
        if (!plan.lm_head_tied_to_embedding) {
            lm_head_table_.assign(plan.lm_head.float_count, 0.0F);
        }
    } catch (...) {
        reset();
        return false;
    }

    // Stream and dequantize embedding and normalization tables
    if (!runtime->seek_stream(plan.token_embedding.tensor_index) ||
        !runtime->stream_tensor(plan.token_embedding.tensor_index,
                                token_embedding_table_.data(),
                                token_embedding_table_.size()) ||
        !runtime->seek_stream(plan.output_norm.tensor_index) ||
        !runtime->stream_tensor(plan.output_norm.tensor_index,
                                output_norm_table_.data(),
                                output_norm_table_.size())) {
        reset();
        return false;
    }

    if (!plan.lm_head_tied_to_embedding) {
        if (!runtime->seek_stream(plan.lm_head.tensor_index) ||
            !runtime->stream_tensor(plan.lm_head.tensor_index,
                                    lm_head_table_.data(),
                                    lm_head_table_.size())) {
            reset();
            return false;
        }
    }

    // Attempt to preload all layer weight arenas in-memory for zero-I/O generation
    try {
        layer_weight_arenas_.resize(config.layer_count);
        weights_cached_ = true;
        for (std::size_t i = 0; i < config.layer_count; ++i) {
            TransformerLayerWeights weights;
            if (!runtime->seek_stream(plan.layers[i].attention_norm.tensor_index) ||
                !runtime->load_transformer_layer(plan.layers[i],
                                                 layer_weight_arenas_[i], weights)) {
                weights_cached_ = false;
                layer_weight_arenas_.clear();
                break;
            }
            if (runtime->gpu_backend().is_layer_offloaded(i)) {
                (void)runtime->gpu_backend().upload_layer_weights(i, weights, plan.layers[i]);
            }
        }
    } catch (...) {
        weights_cached_ = false;
        layer_weight_arenas_.clear();
    }

    return true;
}

void TokenGenerator::reset() noexcept {
    runtime_ = nullptr;
    plan_ = {};
    config_ = {};
    kv_cache_.clear();
    sampler_.reset();
    max_sequence_length_ = 0;
    current_position_ = 0;
    token_embedding_table_.clear();
    output_norm_table_.clear();
    lm_head_table_.clear();
    hidden_states_.clear();
    hidden_scratch_.clear();
    layer_arena_.clear();
    block_scratch_.clear();
    logits_.clear();
    layer_weight_arenas_.clear();
    weights_cached_ = false;
}

bool TokenGenerator::forward_step(const std::uint32_t* tokens, std::size_t token_count,
                                  std::size_t start_position, float* logits_out) {
    return forward_step(tokens, token_count, start_position, logits_out, kv_cache_);
}

bool TokenGenerator::forward_step(const std::uint32_t* tokens, std::size_t token_count,
                                  std::size_t start_position, float* logits_out,
                                  KvCache& kv_cache) {
    if (!is_initialized() || tokens == nullptr || token_count == 0 ||
        logits_out == nullptr) {
        return false;
    }
    if (start_position > max_sequence_length_ ||
        token_count > max_sequence_length_ - start_position) {
        return false;
    }

    const std::size_t model_dim = config_.model_dimension;

    // 1. Embedding lookup for prompt or current token
    for (std::size_t i = 0; i < token_count; ++i) {
        const std::uint32_t token_id = tokens[i];
        if (token_id >= config_.vocab_size) return false;
        const float* src = token_embedding_table_.data() + token_id * model_dim;
        float* dst = hidden_states_.data() + i * model_dim;
        std::memcpy(dst, src, model_dim * sizeof(float));
    }

    // 2. Transformer block forward through every layer
    for (std::size_t l = 0; l < config_.layer_count; ++l) {
        if (runtime_->gpu_backend().is_layer_offloaded(l)) {
            if (!runtime_->gpu_backend().execute_layer_forward(
                    l, config_.block, start_position, hidden_states_.data(),
                    token_count, kv_cache, hidden_scratch_.data())) {
                return false;
            }
        } else {
            TransformerLayerWeights weights;
            if (weights_cached_) {
                weights = bind_transformer_layer_plan(plan_.layers[l],
                                                      layer_weight_arenas_[l].data());
            } else {
                runtime_->seek_stream(plan_.layers[l].attention_norm.tensor_index);
                if (!runtime_->load_transformer_layer(plan_.layers[l], layer_arena_,
                                                      weights)) {
                    return false;
                }
            }

            if (!transformer_block_forward(
                    config_.block, weights, l, start_position, hidden_states_.data(),
                    token_count, kv_cache, hidden_scratch_.data(),
                    block_scratch_.data(), block_scratch_.size())) {
                return false;
            }
        }

        // Copy layer output hidden states back to active buffer
        std::memcpy(hidden_states_.data(), hidden_scratch_.data(),
                    token_count * model_dim * sizeof(float));
    }

    // 3. Final layer normalization on the last token's hidden state
    const float* last_hidden =
        hidden_states_.data() + (token_count - 1) * model_dim;
    float* normed_last = hidden_scratch_.data();
    if (!kernels::rms_norm(last_hidden, output_norm_table_.data(), model_dim,
                           config_.block.rms_epsilon, normed_last)) {
        return false;
    }

    // 4. Compute unnormalized logits via LM head matrix-vector projection
    const float* lm_weights = plan_.lm_head_tied_to_embedding
                                  ? token_embedding_table_.data()
                                  : lm_head_table_.data();
    if (!kernels::linear_f32(lm_weights, config_.vocab_size, model_dim, normed_last,
                             logits_out)) {
        return false;
    }

    return true;
}

bool TokenGenerator::generate(const std::vector<std::uint32_t>& prompt_tokens,
                              const GenerationOptions& options,
                              GenerationResult& result,
                              StreamCallback stream_callback) {
    if (!is_initialized() || prompt_tokens.empty() || options.max_new_tokens == 0) {
        return false;
    }
    if (prompt_tokens.size() > max_sequence_length_) return false;

    result = {};
    result.prompt_tokens = prompt_tokens;
    result.prompt_token_count = prompt_tokens.size();
    result.all_tokens = prompt_tokens;
    result.reached_eos = false;

    SamplerConfig sampler_cfg = options.sampler;
    if (options.repetition_penalty > 1.0F && sampler_cfg.repetition_penalty == 1.0F) {
        sampler_cfg.repetition_penalty = options.repetition_penalty;
    }

    auto emit_chunk = [&](std::uint32_t token_id, bool is_eos) -> bool {
        if (!stream_callback) return true;
        StreamChunk chunk;
        chunk.token_id = token_id;
        chunk.token_index = result.generated_tokens.size() - 1;
        chunk.is_eos = is_eos;
        if (runtime_->tokenizer().configured()) {
            std::string current_full_text;
            runtime_->detokenize(result.generated_tokens, current_full_text);
            if (current_full_text.size() >= result.text.size()) {
                chunk.text_delta = current_full_text.substr(result.text.size());
                result.text = std::move(current_full_text);
            } else {
                chunk.text_delta = current_full_text;
                result.text = std::move(current_full_text);
            }
        }
        return stream_callback(chunk);
    };

    // Reset KV-cache sequence tracking for new generation session
    kv_cache_.reset_sequence();
    current_position_ = 0;

    // Prefill phase: evaluate entire prompt sequence
    if (!forward_step(prompt_tokens.data(), prompt_tokens.size(), 0, logits_.data())) {
        return false;
    }
    current_position_ = prompt_tokens.size();

    std::uint32_t next_token = sampler_.sample(logits_.data(), config_.vocab_size,
                                               result.all_tokens, sampler_cfg);

    if (config_.eos_token_id >= 0 &&
        next_token == static_cast<std::uint32_t>(config_.eos_token_id)) {
        result.reached_eos = true;
    }
    for (const std::uint32_t stop_id : options.stop_token_ids) {
        if (next_token == stop_id) {
            result.reached_eos = true;
            break;
        }
    }

    result.generated_tokens.push_back(next_token);
    result.all_tokens.push_back(next_token);

    if (!emit_chunk(next_token, result.reached_eos)) {
        result.generated_token_count = result.generated_tokens.size();
        return true;
    }

    if (result.reached_eos && options.early_stopping) {
        result.generated_token_count = result.generated_tokens.size();
        if (runtime_->tokenizer().configured() && result.text.empty()) {
            runtime_->detokenize(result.generated_tokens, result.text);
        }
        return true;
    }

    // Autoregressive decode loop: single token per iteration
    while (result.generated_tokens.size() < options.max_new_tokens &&
           current_position_ < max_sequence_length_) {
        const std::uint32_t current_token = result.generated_tokens.back();
        if (!forward_step(&current_token, 1, current_position_, logits_.data())) {
            return false;
        }
        current_position_ += 1;

        next_token = sampler_.sample(logits_.data(), config_.vocab_size,
                                     result.all_tokens, sampler_cfg);

        if (config_.eos_token_id >= 0 &&
            next_token == static_cast<std::uint32_t>(config_.eos_token_id)) {
            result.reached_eos = true;
        }
        for (const std::uint32_t stop_id : options.stop_token_ids) {
            if (next_token == stop_id) {
                result.reached_eos = true;
                break;
            }
        }

        result.generated_tokens.push_back(next_token);
        result.all_tokens.push_back(next_token);

        if (!emit_chunk(next_token, result.reached_eos)) {
            break;
        }

        if (result.reached_eos && options.early_stopping) {
            break;
        }
    }

    result.generated_token_count = result.generated_tokens.size();
    if (runtime_->tokenizer().configured() && result.text.empty() && !result.generated_tokens.empty()) {
        runtime_->detokenize(result.generated_tokens, result.text);
    }
    return true;
}

bool TokenGenerator::generate_batch(
    const std::vector<std::vector<std::uint32_t>>& prompt_batch,
    const std::vector<GenerationOptions>& options_batch,
    std::vector<GenerationResult>& results_batch,
    std::vector<StreamCallback> stream_callbacks) {
    if (!is_initialized() || prompt_batch.empty() ||
        prompt_batch.size() != options_batch.size()) {
        return false;
    }
    BatchSchedulerConfig sched_cfg;
    sched_cfg.max_batch_size = std::max<std::size_t>(prompt_batch.size(), 16);
    sched_cfg.max_sequence_length = max_sequence_length_;
    sched_cfg.max_queue_depth = prompt_batch.size() + 16;

    BatchScheduler scheduler(sched_cfg);
    if (!scheduler.initialize(runtime_, config_)) {
        return false;
    }
    results_batch = scheduler.generate_batch(prompt_batch, options_batch,
                                             std::move(stream_callbacks));
    return results_batch.size() == prompt_batch.size();
}

bool TokenGenerator::generate_text(std::string_view prompt,
                                   const GenerationOptions& options,
                                   GenerationResult& result,
                                   StreamCallback stream_callback) {
    if (!is_initialized() || runtime_ == nullptr) return false;
    const auto tokenized = runtime_->tokenize(prompt);
    if (!tokenized || tokenized.tokens.empty()) return false;

    std::vector<std::uint32_t> prompt_tokens;
    prompt_tokens.reserve(tokenized.tokens.size() + 1);
    if (config_.bos_token_id >= 0) {
        prompt_tokens.push_back(static_cast<std::uint32_t>(config_.bos_token_id));
    }
    for (const auto& tok : tokenized.tokens) {
        prompt_tokens.push_back(tok.id);
    }

    return generate(prompt_tokens, options, result, std::move(stream_callback));
}

} // namespace aabhas::inference
