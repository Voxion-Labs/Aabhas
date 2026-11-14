#ifndef AABHAS_INFERENCE_KV_CACHE_HPP
#define AABHAS_INFERENCE_KV_CACHE_HPP

#include <cstddef>
#include <vector>

namespace aabhas::inference {

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
    std::size_t layer_count() const noexcept { return layer_count_; }
    std::size_t bytes() const noexcept;

private:
    std::size_t offset(std::size_t layer, std::size_t token) const noexcept;

    std::size_t layer_count_{0};
    std::size_t token_width_{0};
    std::size_t max_tokens_{0};
    std::vector<float> keys_;
    std::vector<float> values_;
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_KV_CACHE_HPP
