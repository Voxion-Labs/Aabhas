#include "inference/transformer_layers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aabhas::inference {
namespace {

bool multiply_checked(std::size_t lhs, std::size_t rhs, std::size_t& result) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    result = lhs * rhs;
    return true;
}

bool finite_vector(const float* values, std::size_t count) noexcept {
    if (values == nullptr) return false;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) return false;
    }
    return true;
}

} // namespace

bool validate_transformer_shape(const TransformerShape& shape,
                                std::size_t max_kv_bytes,
                                std::size_t& required_kv_bytes) noexcept {
    required_kv_bytes = 0;
    if (shape.model_dimension == 0 || shape.head_count == 0 ||
        shape.head_dimension == 0 || shape.kv_head_count == 0 ||
        shape.intermediate_dimension == 0 || shape.max_sequence_length == 0 ||
        max_kv_bytes == 0 || shape.kv_head_count > shape.head_count ||
        shape.head_count % shape.kv_head_count != 0) {
        return false;
    }

    std::size_t expected_model_dimension = 0;
    if (!multiply_checked(shape.head_count, shape.head_dimension,
                          expected_model_dimension) ||
        expected_model_dimension != shape.model_dimension) {
        return false;
    }

    std::size_t elements = 0;
    if (!multiply_checked(shape.kv_head_count, shape.max_sequence_length, elements) ||
        !multiply_checked(elements, shape.head_dimension, elements) ||
        !multiply_checked(elements, sizeof(float) * 2, required_kv_bytes) ||
        required_kv_bytes > max_kv_bytes) {
        required_kv_bytes = 0;
        return false;
    }
    return true;
}

namespace kernels {

bool rms_norm(const float* input, const float* weight, std::size_t width,
              float epsilon, float* output) noexcept {
    if (width == 0 || output == nullptr || !finite_vector(input, width) ||
        !finite_vector(weight, width) || !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return false;
    }
    double sum = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
        sum += static_cast<double>(input[index]) * static_cast<double>(input[index]);
    }
    const double mean_square = sum / static_cast<double>(width);
    const double inverse_root = 1.0 / std::sqrt(mean_square + static_cast<double>(epsilon));
    if (!std::isfinite(inverse_root)) return false;
    for (std::size_t index = 0; index < width; ++index) {
        const double value = static_cast<double>(input[index]) * inverse_root *
                             static_cast<double>(weight[index]);
        if (!std::isfinite(value)) return false;
        output[index] = static_cast<float>(value);
    }
    return true;
}

bool silu(const float* input, std::size_t width, float* output) noexcept {
    if (width == 0 || output == nullptr || !finite_vector(input, width)) return false;
    for (std::size_t index = 0; index < width; ++index) {
        const double value = static_cast<double>(input[index]);
        const double sigmoid = 1.0 / (1.0 + std::exp(-value));
        const double result = value * sigmoid;
        if (!std::isfinite(result)) return false;
        output[index] = static_cast<float>(result);
    }
    return true;
}

bool swiglu(const float* gate, const float* up, std::size_t width,
            float* output) noexcept {
    if (width == 0 || output == nullptr || !finite_vector(gate, width) ||
        !finite_vector(up, width)) {
        return false;
    }
    for (std::size_t index = 0; index < width; ++index) {
        const double gate_value = static_cast<double>(gate[index]);
        const double sigmoid = 1.0 / (1.0 + std::exp(-gate_value));
        const double result = gate_value * sigmoid * static_cast<double>(up[index]);
        if (!std::isfinite(result)) return false;
        output[index] = static_cast<float>(result);
    }
    return true;
}

bool apply_rope(float* query, float* key, std::size_t head_dimension,
                std::size_t position, float theta) noexcept {
    if (query == nullptr || head_dimension == 0 || head_dimension % 2 != 0 ||
        !std::isfinite(theta) || theta <= 1.0F || !finite_vector(query, head_dimension) ||
        (key != nullptr && !finite_vector(key, head_dimension))) {
        return false;
    }
    for (std::size_t index = 0; index < head_dimension; index += 2) {
        const double exponent = -static_cast<double>(index) /
                                static_cast<double>(head_dimension);
        const double angle = static_cast<double>(position) *
                             std::pow(static_cast<double>(theta), exponent);
        const float cosine = static_cast<float>(std::cos(angle));
        const float sine = static_cast<float>(std::sin(angle));
        const float query_even = query[index];
        const float query_odd = query[index + 1];
        query[index] = query_even * cosine - query_odd * sine;
        query[index + 1] = query_even * sine + query_odd * cosine;
        if (key != nullptr) {
            const float key_even = key[index];
            const float key_odd = key[index + 1];
            key[index] = key_even * cosine - key_odd * sine;
            key[index + 1] = key_even * sine + key_odd * cosine;
        }
    }
    return finite_vector(query, head_dimension) &&
           (key == nullptr || finite_vector(key, head_dimension));
}

bool causal_attention(const float* query, const float* keys, const float* values,
                      std::size_t token_count, std::size_t head_dimension,
                      float scale, float* output) noexcept {
    std::size_t elements = 0;
    if (query == nullptr || keys == nullptr || values == nullptr || output == nullptr ||
        token_count == 0 || head_dimension == 0 || !std::isfinite(scale) || scale <= 0.0F ||
        !multiply_checked(token_count, head_dimension, elements) ||
        !finite_vector(query, head_dimension) || !finite_vector(keys, elements) ||
        !finite_vector(values, elements)) {
        return false;
    }

    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < token_count; ++token) {
        double score = 0.0;
        const float* key = keys + token * head_dimension;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            score += static_cast<double>(query[index]) * static_cast<double>(key[index]);
        }
        score *= static_cast<double>(scale);
        if (!std::isfinite(score)) return false;
        maximum = std::max(maximum, score);
    }

    double denominator = 0.0;
    for (std::size_t token = 0; token < token_count; ++token) {
        double score = 0.0;
        const float* key = keys + token * head_dimension;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            score += static_cast<double>(query[index]) * static_cast<double>(key[index]);
        }
        denominator += std::exp(score * static_cast<double>(scale) - maximum);
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) return false;

    for (std::size_t index = 0; index < head_dimension; ++index) output[index] = 0.0F;
    for (std::size_t token = 0; token < token_count; ++token) {
        double score = 0.0;
        const float* key = keys + token * head_dimension;
        const float* value = values + token * head_dimension;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            score += static_cast<double>(query[index]) * static_cast<double>(key[index]);
        }
        const double probability =
            std::exp(score * static_cast<double>(scale) - maximum) / denominator;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            output[index] += static_cast<float>(probability * static_cast<double>(value[index]));
        }
    }
    return finite_vector(output, head_dimension);
}

} // namespace kernels
} // namespace aabhas::inference
