#ifndef AABHAS_INFERENCE_KV_CACHE_HPP
#define AABHAS_INFERENCE_KV_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aabhas::inference {

enum class KvCacheDType : std::uint8_t {
    F32,
    F16,
};

/** Bounded key/value storage for autoregressive attention state. */
class KvCache final {
public:
    KvCache() noexcept = default;

    /**
     * Configure key/value cache dimensions and maximum capacity.
     * Guaranteed bounded memory allocation strictly within max_bytes limit.
     */
    bool configure(std::size_t layer_count, std::size_t head_count,
                   std::size_t head_dimension, std::size_t max_tokens,
                   std::size_t max_bytes,
                   KvCacheDType dtype = KvCacheDType::F32);

    /** Free all memory and reset dimensions. */
    void clear() noexcept;

    /**
     * Append one token's key and value vectors at `token` position.
     * Bounds-checked and allocation-free.
     */
    bool append(std::size_t layer, std::size_t token,
                const float* key_data, const float* value_data) noexcept;

    /**
     * Append a sequence of `token_count` tokens starting at `start_token`.
     * Used for prompt prefill in a single contiguous memory copy.
     */
    bool append_sequence(std::size_t layer, std::size_t start_token,
                         std::size_t token_count,
                         const float* key_data, const float* value_data) noexcept;

    /** Read a single token's key vector. */
    const float* key(std::size_t layer, std::size_t token) const noexcept;

    /** Read a single token's value vector. */
    const float* value(std::size_t layer, std::size_t token) const noexcept;

    /** Pointer to all cached keys for a layer starting at token 0. */
    const float* layer_keys(std::size_t layer) const noexcept;

    /** Pointer to all cached values for a layer starting at token 0. */
    const float* layer_values(std::size_t layer) const noexcept;

    /** Current sequence length in the cache. */
    std::size_t sequence_length() const noexcept { return sequence_length_; }
    void set_sequence_length(std::size_t len) noexcept { sequence_length_ = len; }
    void advance_sequence(std::size_t delta) noexcept { sequence_length_ += delta; }

    /** Truncate sequence to a prefix for context reuse or rollback. */
    bool truncate(std::size_t new_length) noexcept;

    /** Reset sequence length to 0 without reallocating buffers. */
    void reset_sequence() noexcept { sequence_length_ = 0; }

    std::size_t token_width() const noexcept { return token_width_; }
    std::size_t head_count() const noexcept { return head_count_; }
    std::size_t head_dimension() const noexcept { return head_dimension_; }
    std::size_t max_tokens() const noexcept { return max_tokens_; }
    std::size_t layer_count() const noexcept { return layer_count_; }
    KvCacheDType dtype() const noexcept { return dtype_; }
    std::size_t bytes() const noexcept;

private:
    std::size_t offset(std::size_t layer, std::size_t token) const noexcept;

    std::size_t layer_count_{0};
    std::size_t head_count_{0};
    std::size_t head_dimension_{0};
    std::size_t token_width_{0};
    std::size_t max_tokens_{0};
    std::size_t sequence_length_{0};
    KvCacheDType dtype_{KvCacheDType::F32};
    std::vector<float> keys_{};
    std::vector<float> values_{};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_KV_CACHE_HPP