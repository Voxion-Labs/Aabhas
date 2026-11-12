#ifndef AABHAS_INFERENCE_INFERENCE_RUNTIME_HPP
#define AABHAS_INFERENCE_INFERENCE_RUNTIME_HPP

#include "model/model_index.hpp"
#include "scheduler/layer_scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aabhas::inference {

struct InferenceConfig {
    aabhas::scheduler::SchedulerConfig scheduler{};
    std::size_t max_kv_bytes{256ULL * 1024ULL * 1024ULL};
};

/** Bounded float32 key/value storage for autoregressive attention state. */
class KvCache final {
public:
    bool configure(std::size_t layer_count, std::size_t head_count,
                   std::size_t head_dimension, std::size_t max_tokens,
                   std::size_t max_bytes);
    void clear() noexcept;

    bool append(std::size_t layer, std::size_t token,
                const float* key, const float* value) noexcept;
    const float* key(std::size_t layer, std::size_t token) const noexcept;
    const float* value(std::size_t layer, std::size_t token) const noexcept;

    std::size_t token_width() const noexcept { return token_width_; }
    std::size_t max_tokens() const noexcept { return max_tokens_; }
    std::size_t bytes() const noexcept;

private:
    std::size_t offset(std::size_t layer, std::size_t token) const noexcept;

    std::size_t layer_count_{0};
    std::size_t token_width_{0};
    std::size_t max_tokens_{0};
    std::vector<float> keys_;
    std::vector<float> values_;
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

    /** Run one streamed row-major F32 matrix-vector tensor. */
    bool run_f32_matvec(std::size_t tensor_index, const float* input,
                        std::size_t input_size, float* output,
                        std::size_t output_size) noexcept;

    bool configure_kv_cache(std::size_t layer_count, std::size_t head_count,
                            std::size_t head_dimension, std::size_t max_tokens);

    KvCache& kv_cache() noexcept { return kv_cache_; }
    const KvCache& kv_cache() const noexcept { return kv_cache_; }
    const aabhas::model::ModelIndex& model() const noexcept { return model_; }

private:
    aabhas::model::ModelIndex model_;
    aabhas::scheduler::LayerScheduler scheduler_;
    KvCache kv_cache_;
    std::size_t max_kv_bytes_{InferenceConfig{}.max_kv_bytes};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_INFERENCE_RUNTIME_HPP
