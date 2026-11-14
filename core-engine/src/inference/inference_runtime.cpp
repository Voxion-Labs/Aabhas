#include "inference/inference_runtime.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

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

bool InferenceRuntime::open(const char* model_path,
                            const aabhas::model::ModelIndex& model,
                            InferenceConfig config) {
    try {
        close();
        if (model_path == nullptr || model.tensors.empty() ||
            config.scheduler.chunk_bytes == 0 ||
            config.scheduler.chunk_bytes % sizeof(float) != 0) {
            return false;
        }
        std::vector<aabhas::tensor_ops::LayerDescriptor> layers;
        layers.reserve(model.tensors.size());
        for (const auto& tensor : model.tensors) {
            if (tensor.byte_size == 0 ||
                tensor.byte_size > std::numeric_limits<std::size_t>::max()) {
                return false;
            }
            layers.push_back({tensor.file_offset, static_cast<std::size_t>(tensor.byte_size)});
        }
        if (!scheduler_.open(model_path, std::move(layers), config.scheduler)) return false;
        model_ = model;
        max_kv_bytes_ = config.max_kv_bytes;
        return true;
    } catch (...) {
        close();
        return false;
    }
}

void InferenceRuntime::close() noexcept {
    scheduler_.close();
    model_ = {};
    kv_cache_.clear();
    tokenizer_.clear();
    max_kv_bytes_ = InferenceConfig{}.max_kv_bytes;
}

bool InferenceRuntime::configure_kv_cache(std::size_t layer_count,
                                          std::size_t head_count,
                                          std::size_t head_dimension,
                                          std::size_t max_tokens) {
    return kv_cache_.configure(layer_count, head_count, head_dimension, max_tokens,
                               max_kv_bytes_);
}

bool InferenceRuntime::configure_tokenizer(aabhas::tokenizer::TokenizerConfig config) {
    return tokenizer_.configure(model_.tokenizer, config);
}

aabhas::tokenizer::TokenizeResult InferenceRuntime::tokenize(std::string_view input) const {
    return tokenizer_.encode(input);
}

bool InferenceRuntime::detokenize(const std::vector<std::uint32_t>& tokens,
                                  std::string& output) const {
    return tokenizer_.decode(tokens, output);
}

bool InferenceRuntime::run_f32_matvec(std::size_t tensor_index, const float* input,
                                      std::size_t input_size, float* output,
                                      std::size_t output_size) noexcept {
    if (input == nullptr || output == nullptr || tensor_index >= model_.tensors.size() ||
        scheduler_.has_current() ||
        scheduler_.stats().next_layer_index != tensor_index) {
        return false;
    }
    const auto& tensor = model_.tensors[tensor_index];
    if (tensor.dtype != aabhas::model::TensorDType::F32 || tensor.shape.size() != 2) {
        return false;
    }
    if (tensor.shape[0] == 0 || tensor.shape[1] == 0 ||
        tensor.shape[0] > std::numeric_limits<std::size_t>::max() ||
        tensor.shape[1] > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const std::size_t rows = static_cast<std::size_t>(tensor.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(tensor.shape[1]);
    std::size_t elements = 0;
    if (!multiply_checked(rows, columns, elements) ||
        !multiply_checked(elements, sizeof(float), elements) ||
        input_size != columns || output_size != rows || tensor.byte_size < elements) {
        return false;
    }

    std::fill(output, output + rows, 0.0F);
    if (!scheduler_.next_layer()) return false;

    std::size_t processed = 0;
    bool complete = false;
    while (scheduler_.has_current()) {
        const auto view = scheduler_.current();
        if (!view || view.size % sizeof(float) != 0) break;
        const std::size_t values = view.size / sizeof(float);
        for (std::size_t index = 0; index < values && processed < elements / sizeof(float);
             ++index, ++processed) {
            float weight = 0.0F;
            std::memcpy(&weight, view.data + index * sizeof(float), sizeof(float));
            output[processed / columns] += weight * input[processed % columns];
        }
        if (processed == elements / sizeof(float)) {
            complete = true;
            break;
        }
        if (!scheduler_.next_chunk()) break;
    }
    scheduler_.release();
    return complete;
}

} // namespace aabhas::inference
