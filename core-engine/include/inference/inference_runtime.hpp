#ifndef AABHAS_INFERENCE_INFERENCE_RUNTIME_HPP
#define AABHAS_INFERENCE_INFERENCE_RUNTIME_HPP

#include "inference/kv_cache.hpp"
#include "inference/transformer_layers.hpp"
#include "model/model_index.hpp"
#include "scheduler/layer_scheduler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aabhas::inference {

struct InferenceConfig {
    aabhas::scheduler::SchedulerConfig scheduler{};
    std::size_t max_kv_bytes{256ULL * 1024ULL * 1024ULL};
    std::size_t max_layer_buffer_bytes{512ULL * 1024ULL * 1024ULL};
};

/** Executes bounded CPU kernels over tensor windows supplied by the scheduler. */
class InferenceRuntime final {
public:
    InferenceRuntime() noexcept = default;
    ~InferenceRuntime() noexcept { close(); }
    InferenceRuntime(const InferenceRuntime&) = delete;
    InferenceRuntime& operator=(const InferenceRuntime&) = delete;

    bool open(const char* model_path, const aabhas::model::ModelIndex& model,
              InferenceConfig config = {});
    void close() noexcept;

    /** Run one streamed row-major matrix-vector tensor (supports F32, F16, BF16, and quantized types). */
    bool run_matvec(std::size_t tensor_index, const float* input,
                    std::size_t input_size, float* output,
                    std::size_t output_size) noexcept;

    /** Run one streamed row-major F32 matrix-vector tensor. */
    bool run_f32_matvec(std::size_t tensor_index, const float* input,
                        std::size_t input_size, float* output,
                        std::size_t output_size) noexcept;

    /**
     * Stream one row-major tensor of any supported dtype, dequantizing into a
     * caller-owned float32 arena.
     *
     * Tensors must be consumed in scheduler order; call rewind_stream() before
     * starting another full pass over the model.
     */
    bool stream_tensor(std::size_t tensor_index, float* destination,
                       std::size_t capacity_floats) noexcept;

    /**
     * Stream one row-major F32 tensor into a caller-owned arena.
     *
     * Tensors must be consumed in scheduler order; call rewind_stream() before
     * starting another full pass over the model.
     */
    bool stream_f32_tensor(std::size_t tensor_index, float* destination,
                           std::size_t capacity_floats) noexcept;

    /**
     * Stream every tensor referenced by a transformer-layer plan into a bounded
     * arena and bind non-owning weight views. The arena must stay alive and
     * unmodified for the lifetime of the returned views.
     */
    bool load_transformer_layer(const TransformerLayerPlan& plan,
                                std::vector<float>& arena,
                                TransformerLayerWeights& weights);

    /** Reset the scheduler sequence so the model can be streamed again. */
    bool rewind_stream() noexcept;

    /** Configure the bounded cache. `head_count` must be the number of cached
     *  key/value heads; grouped-query models pass their kv head count. */
    bool configure_kv_cache(std::size_t layer_count, std::size_t head_count,
                            std::size_t head_dimension, std::size_t max_tokens);

    bool configure_tokenizer(aabhas::tokenizer::TokenizerConfig config = {});
    aabhas::tokenizer::TokenizeResult tokenize(std::string_view input) const;
    bool detokenize(const std::vector<std::uint32_t>& tokens, std::string& output) const;

    KvCache& kv_cache() noexcept { return kv_cache_; }
    const KvCache& kv_cache() const noexcept { return kv_cache_; }
    const aabhas::model::ModelIndex& model() const noexcept { return model_; }

private:
    aabhas::model::ModelIndex model_;
    aabhas::scheduler::LayerScheduler scheduler_;
    KvCache kv_cache_;
    aabhas::tokenizer::GreedyTokenizer tokenizer_;
    std::size_t max_kv_bytes_{InferenceConfig{}.max_kv_bytes};
    std::size_t max_layer_buffer_bytes_{InferenceConfig{}.max_layer_buffer_bytes};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_INFERENCE_RUNTIME_HPP
