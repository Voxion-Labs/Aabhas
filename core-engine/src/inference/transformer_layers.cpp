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

bool add_checked(std::size_t lhs, std::size_t rhs, std::size_t& result) noexcept {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

bool finite_vector(const float* values, std::size_t count) noexcept {
    if (values == nullptr) return false;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) return false;
    }
    return true;
}

bool block_dimensions(const TransformerBlockConfig& config, std::size_t& model_dimension,
                      std::size_t& kv_width, std::size_t& group_size) noexcept {
    model_dimension = 0;
    kv_width = 0;
    group_size = 0;
    if (config.model_dimension == 0 || config.head_count == 0 ||
        config.kv_head_count == 0 || config.head_dimension == 0 ||
        config.intermediate_dimension == 0 || config.kv_head_count > config.head_count ||
        config.head_count % config.kv_head_count != 0) {
        return false;
    }
    if (!multiply_checked(config.head_count, config.head_dimension, model_dimension) ||
        model_dimension != config.model_dimension) {
        return false;
    }
    if (!multiply_checked(config.kv_head_count, config.head_dimension, kv_width) ||
        kv_width == 0) {
        return false;
    }
    group_size = config.head_count / config.kv_head_count;
    return true;
}

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

bool resolve_tensor(const aabhas::model::ModelIndex& model, const std::string& name,
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
    if (found == nullptr || found->dtype != aabhas::model::TensorDType::F32 ||
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

bool llama_transformer_layer_naming(aabhas::model::ModelFormat format,
                                    std::size_t layer_index,
                                    TransformerLayerNaming& naming) {
    const std::string layer = std::to_string(layer_index);
    switch (format) {
    case aabhas::model::ModelFormat::GGUF: {
        const std::string prefix = "blk." + layer + ".";
        naming = {prefix + "attn_norm.weight",  prefix + "attn_q.weight",
                  prefix + "attn_k.weight",     prefix + "attn_v.weight",
                  prefix + "attn_output.weight", prefix + "ffn_norm.weight",
                  prefix + "ffn_gate.weight",   prefix + "ffn_up.weight",
                  prefix + "ffn_down.weight"};
        return true;
    }
    case aabhas::model::ModelFormat::SafeTensors: {
        const std::string prefix = "model.layers." + layer + ".";
        naming = {prefix + "input_layernorm.weight",
                  prefix + "self_attn.q_proj.weight",
                  prefix + "self_attn.k_proj.weight",
                  prefix + "self_attn.v_proj.weight",
                  prefix + "self_attn.o_proj.weight",
                  prefix + "post_attention_layernorm.weight",
                  prefix + "mlp.gate_proj.weight",
                  prefix + "mlp.up_proj.weight",
                  prefix + "mlp.down_proj.weight"};
        return true;
    }
    case aabhas::model::ModelFormat::RawBinary:
    default:
        return false;
    }
}

bool build_transformer_layer_plan(const aabhas::model::ModelIndex& model,
                                  const TransformerLayerNaming& naming,
                                  const TransformerBlockConfig& config,
                                  TransformerLayerPlan& plan) {
    std::size_t model_dimension = 0;
    std::size_t kv_width = 0;
    std::size_t group_size = 0;
    if (!block_dimensions(config, model_dimension, kv_width, group_size)) return false;

    plan = {};
    plan.naming = naming;
    if (!resolve_tensor(model, naming.attention_norm, {model_dimension},
                        plan.attention_norm) ||
        !resolve_tensor(model, naming.attention_query, {model_dimension, model_dimension},
                        plan.attention_query) ||
        !resolve_tensor(model, naming.attention_key, {kv_width, model_dimension},
                        plan.attention_key) ||
        !resolve_tensor(model, naming.attention_value, {kv_width, model_dimension},
                        plan.attention_value) ||
        !resolve_tensor(model, naming.attention_output, {model_dimension, model_dimension},
                        plan.attention_output) ||
        !resolve_tensor(model, naming.feed_forward_norm, {model_dimension},
                        plan.feed_forward_norm) ||
        !resolve_tensor(model, naming.feed_forward_gate,
                        {config.intermediate_dimension, model_dimension},
                        plan.feed_forward_gate) ||
        !resolve_tensor(model, naming.feed_forward_up,
                        {config.intermediate_dimension, model_dimension},
                        plan.feed_forward_up) ||
        !resolve_tensor(model, naming.feed_forward_down,
                        {model_dimension, config.intermediate_dimension},
                        plan.feed_forward_down)) {
        plan = {};
        return false;
    }

    struct Slot {
        TransformerTensorBinding* binding;
    };
    Slot slots[] = {{&plan.attention_norm},    {&plan.attention_query},
                    {&plan.attention_key},     {&plan.attention_value},
                    {&plan.attention_output},  {&plan.feed_forward_norm},
                    {&plan.feed_forward_gate}, {&plan.feed_forward_up},
                    {&plan.feed_forward_down}};
    constexpr std::size_t slot_count = sizeof(slots) / sizeof(slots[0]);
    std::sort(slots, slots + slot_count, [](const Slot& lhs, const Slot& rhs) {
        return lhs.binding->tensor_index < rhs.binding->tensor_index;
    });
    std::size_t offset = 0;
    for (std::size_t index = 0; index < slot_count; ++index) {
        if (index > 0 &&
            slots[index].binding->tensor_index == slots[index - 1].binding->tensor_index) {
            plan = {};
            return false;
        }
        if (!add_checked(offset, slots[index].binding->float_count, offset)) {
            plan = {};
            return false;
        }
        slots[index].binding->offset_floats =
            offset - slots[index].binding->float_count;
    }
    plan.total_floats = offset;
    return plan.total_floats != 0;
}

TransformerLayerWeights bind_transformer_layer_plan(const TransformerLayerPlan& plan,
                                                    const float* arena) noexcept {
    TransformerLayerWeights weights;
    if (arena == nullptr) return weights;
    weights.attention_norm = arena + plan.attention_norm.offset_floats;
    weights.attention_query = arena + plan.attention_query.offset_floats;
    weights.attention_key = arena + plan.attention_key.offset_floats;
    weights.attention_value = arena + plan.attention_value.offset_floats;
    weights.attention_output = arena + plan.attention_output.offset_floats;
    weights.feed_forward_norm = arena + plan.feed_forward_norm.offset_floats;
    weights.feed_forward_gate = arena + plan.feed_forward_gate.offset_floats;
    weights.feed_forward_up = arena + plan.feed_forward_up.offset_floats;
    weights.feed_forward_down = arena + plan.feed_forward_down.offset_floats;
    return weights;
}

std::size_t transformer_block_scratch_floats(const TransformerBlockConfig& config) noexcept {
    std::size_t model_dimension = 0;
    std::size_t kv_width = 0;
    std::size_t group_size = 0;
    if (!block_dimensions(config, model_dimension, kv_width, group_size)) return 0;

    std::size_t total = 0;
    std::size_t auxiliary = 0;
    if (!multiply_checked(model_dimension, 4, total) ||
        !multiply_checked(kv_width, 2, auxiliary) ||
        !add_checked(total, auxiliary, total) ||
        !multiply_checked(config.intermediate_dimension, 2, auxiliary) ||
        !add_checked(total, auxiliary, total)) {
        return 0;
    }
    return total;
}

bool transformer_block_forward(const TransformerBlockConfig& config,
                               const TransformerLayerWeights& weights,
                               std::size_t layer_index, std::size_t start_position,
                               const float* input, std::size_t token_count,
                               KvCache& cache, float* output, float* scratch,
                               std::size_t scratch_floats) noexcept {
    std::size_t model_dimension = 0;
    std::size_t kv_width = 0;
    std::size_t group_size = 0;
    if (!block_dimensions(config, model_dimension, kv_width, group_size)) return false;
    if (weights.attention_norm == nullptr || weights.attention_query == nullptr ||
        weights.attention_key == nullptr || weights.attention_value == nullptr ||
        weights.attention_output == nullptr || weights.feed_forward_norm == nullptr ||
        weights.feed_forward_gate == nullptr || weights.feed_forward_up == nullptr ||
        weights.feed_forward_down == nullptr) {
        return false;
    }
    if (!std::isfinite(config.rms_epsilon) || config.rms_epsilon <= 0.0F ||
        !std::isfinite(config.rope_theta) || config.rope_theta <= 1.0F ||
        !std::isfinite(config.attention_scale) || config.attention_scale < 0.0F) {
        return false;
    }
    const std::size_t required_scratch = transformer_block_scratch_floats(config);
    if (required_scratch == 0 || scratch == nullptr || scratch_floats < required_scratch ||
        input == nullptr || output == nullptr || token_count == 0 ||
        cache.token_width() != kv_width || cache.layer_count() <= layer_index ||
        cache.max_tokens() == 0 || start_position > cache.max_tokens() ||
        token_count > cache.max_tokens() - start_position) {
        return false;
    }

    const double scale_value = config.attention_scale == 0.0F
        ? 1.0 / std::sqrt(static_cast<double>(config.head_dimension))
        : static_cast<double>(config.attention_scale);
    if (!std::isfinite(scale_value) || scale_value <= 0.0) return false;
    const float scale = static_cast<float>(scale_value);
    if (!std::isfinite(scale) || scale <= 0.0F) return false;

    float* residual = scratch;
    float* normed = residual + model_dimension;
    float* query = normed + model_dimension;
    float* key = query + model_dimension;
    float* value = key + kv_width;
    float* attention = value + kv_width;
    float* gate = attention + model_dimension;
    float* up = gate + config.intermediate_dimension;

    for (std::size_t token = 0; token < token_count; ++token) {
        const float* source = input + token * model_dimension;
        float* destination = output + token * model_dimension;
        const std::size_t position = start_position + token;

        for (std::size_t index = 0; index < model_dimension; ++index) {
            residual[index] = source[index];
        }
        if (!kernels::rms_norm(residual, weights.attention_norm, model_dimension,
                               config.rms_epsilon, normed) ||
            !kernels::linear_f32(weights.attention_query, model_dimension,
                                 model_dimension, normed, query) ||
            !kernels::linear_f32(weights.attention_key, kv_width, model_dimension,
                                 normed, key) ||
            !kernels::linear_f32(weights.attention_value, kv_width, model_dimension,
                                 normed, value)) {
            return false;
        }

        for (std::size_t head = 0; head < config.head_count; ++head) {
            if (!kernels::apply_rope(query + head * config.head_dimension, nullptr,
                                     config.head_dimension, position,
                                     config.rope_theta)) {
                return false;
            }
        }
        for (std::size_t head = 0; head < config.kv_head_count; ++head) {
            if (!kernels::apply_rope(key + head * config.head_dimension, nullptr,
                                     config.head_dimension, position,
                                     config.rope_theta)) {
                return false;
            }
        }
        if (!cache.append(layer_index, position, key, value)) return false;

        const float* cached_keys = cache.key(layer_index, 0);
        const float* cached_values = cache.value(layer_index, 0);
        if (cached_keys == nullptr || cached_values == nullptr) return false;
        for (std::size_t head = 0; head < config.head_count; ++head) {
            const std::size_t kv_head = head / group_size;
            if (!kernels::causal_attention_strided(
                    query + head * config.head_dimension,
                    cached_keys + kv_head * config.head_dimension,
                    cached_values + kv_head * config.head_dimension, position + 1,
                    config.head_dimension, kv_width, kv_width, scale,
                    attention + head * config.head_dimension)) {
                return false;
            }
        }

        if (!kernels::linear_f32(weights.attention_output, model_dimension,
                                 model_dimension, attention, normed)) {
            return false;
        }
        for (std::size_t index = 0; index < model_dimension; ++index) {
            destination[index] = residual[index] + normed[index];
        }
        if (!kernels::rms_norm(destination, weights.feed_forward_norm, model_dimension,
                               config.rms_epsilon, normed) ||
            !kernels::linear_f32(weights.feed_forward_gate,
                                 config.intermediate_dimension, model_dimension, normed,
                                 gate) ||
            !kernels::linear_f32(weights.feed_forward_up,
                                 config.intermediate_dimension, model_dimension, normed,
                                 up) ||
            !kernels::swiglu(gate, up, config.intermediate_dimension, gate) ||
            !kernels::linear_f32(weights.feed_forward_down, model_dimension,
                                 config.intermediate_dimension, gate, normed)) {
            return false;
        }
        for (std::size_t index = 0; index < model_dimension; ++index) {
            destination[index] += normed[index];
        }
        if (!finite_vector(destination, model_dimension)) return false;
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

bool linear_f32(const float* weights, std::size_t rows, std::size_t columns,
                const float* input, float* output) noexcept {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 ||
        columns == 0 || output == input ||
        rows > std::numeric_limits<std::size_t>::max() / columns ||
        !finite_vector(input, columns)) {
        return false;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        const float* row_weights = weights + row * columns;
        double sum = 0.0;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += static_cast<double>(row_weights[column]) *
                   static_cast<double>(input[column]);
        }
        if (!std::isfinite(sum)) return false;
        output[row] = static_cast<float>(sum);
    }
    return true;
}

bool causal_attention_strided(const float* query, const float* keys,
                              const float* values, std::size_t token_count,
                              std::size_t head_dimension, std::size_t key_stride,
                              std::size_t value_stride, float scale,
                              float* output) noexcept {
    if (query == nullptr || keys == nullptr || values == nullptr || output == nullptr ||
        token_count == 0 || head_dimension == 0 || key_stride < head_dimension ||
        value_stride < head_dimension || !std::isfinite(scale) || scale <= 0.0F ||
        (key_stride != 0 &&
         token_count > std::numeric_limits<std::size_t>::max() / key_stride) ||
        (value_stride != 0 &&
         token_count > std::numeric_limits<std::size_t>::max() / value_stride) ||
        !finite_vector(query, head_dimension)) {
        return false;
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (!finite_vector(keys + token * key_stride, head_dimension) ||
            !finite_vector(values + token * value_stride, head_dimension)) {
            return false;
        }
    }

    const double scale_value = static_cast<double>(scale);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < token_count; ++token) {
        double score = 0.0;
        const float* key = keys + token * key_stride;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            score += static_cast<double>(query[index]) * static_cast<double>(key[index]);
        }
        score *= scale_value;
        if (!std::isfinite(score)) return false;
        maximum = std::max(maximum, score);
    }

    double denominator = 0.0;
    for (std::size_t token = 0; token < token_count; ++token) {
        double score = 0.0;
        const float* key = keys + token * key_stride;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            score += static_cast<double>(query[index]) * static_cast<double>(key[index]);
        }
        denominator += std::exp(score * scale_value - maximum);
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) return false;

    for (std::size_t index = 0; index < head_dimension; ++index) output[index] = 0.0F;
    for (std::size_t token = 0; token < token_count; ++token) {
        double score = 0.0;
        const float* key = keys + token * key_stride;
        const float* value = values + token * value_stride;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            score += static_cast<double>(query[index]) * static_cast<double>(key[index]);
        }
        const double probability =
            std::exp(score * scale_value - maximum) / denominator;
        if (!std::isfinite(probability)) return false;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            output[index] = static_cast<float>(
                static_cast<double>(output[index]) +
                probability * static_cast<double>(value[index]));
        }
    }
    return finite_vector(output, head_dimension);
}

bool causal_attention(const float* query, const float* keys, const float* values,
                      std::size_t token_count, std::size_t head_dimension,
                      float scale, float* output) noexcept {
    return causal_attention_strided(query, keys, values, token_count, head_dimension,
                                    head_dimension, head_dimension, scale, output);
}

} // namespace kernels
} // namespace aabhas::inference
