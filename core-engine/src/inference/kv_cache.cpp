#include "inference/kv_cache.hpp"

#include <cstring>
#include <limits>

namespace aabhas::inference {
namespace {

bool multiply_checked(std::size_t lhs, std::size_t rhs, std::size_t& result) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    result = lhs * rhs;
    return true;
}

} // namespace

bool KvCache::configure(std::size_t layer_count, std::size_t head_count,
                        std::size_t head_dimension, std::size_t max_tokens,
                        std::size_t max_bytes) {
    try {
        std::size_t width = 0;
        std::size_t elements = 0;
        if (layer_count == 0 || head_count == 0 || head_dimension == 0 ||
            max_tokens == 0 || max_bytes == 0 ||
            !multiply_checked(head_count, head_dimension, width) ||
            !multiply_checked(layer_count, max_tokens, elements) ||
            !multiply_checked(elements, width, elements) ||
            !multiply_checked(elements, sizeof(float) * 2, elements) ||
            elements > max_bytes) {
            return false;
        }
        layer_count_ = layer_count;
        token_width_ = width;
        max_tokens_ = max_tokens;
        keys_.assign(elements / (sizeof(float) * 2), 0.0F);
        values_.assign(keys_.size(), 0.0F);
        return true;
    } catch (...) {
        clear();
        return false;
    }
}

void KvCache::clear() noexcept {
    layer_count_ = 0;
    token_width_ = 0;
    max_tokens_ = 0;
    keys_.clear();
    values_.clear();
}

std::size_t KvCache::offset(std::size_t layer, std::size_t token) const noexcept {
    return (layer * max_tokens_ + token) * token_width_;
}

bool KvCache::append(std::size_t layer, std::size_t token,
                     const float* key_data, const float* value_data) noexcept {
    if (key_data == nullptr || value_data == nullptr || layer >= layer_count_ ||
        token >= max_tokens_ || token_width_ == 0) {
        return false;
    }
    const std::size_t start = offset(layer, token);
    std::memcpy(keys_.data() + start, key_data, token_width_ * sizeof(float));
    std::memcpy(values_.data() + start, value_data, token_width_ * sizeof(float));
    return true;
}

const float* KvCache::key(std::size_t layer, std::size_t token) const noexcept {
    if (layer >= layer_count_ || token >= max_tokens_ || token_width_ == 0) return nullptr;
    return keys_.data() + offset(layer, token);
}

const float* KvCache::value(std::size_t layer, std::size_t token) const noexcept {
    if (layer >= layer_count_ || token >= max_tokens_ || token_width_ == 0) return nullptr;
    return values_.data() + offset(layer, token);
}

std::size_t KvCache::bytes() const noexcept {
    return keys_.size() * sizeof(float) * 2;
}

} // namespace aabhas::inference
