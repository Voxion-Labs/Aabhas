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

bool InferenceRuntime::open(const char* model_path,
                            const aabhas::model::ModelIndex& model,
                            InferenceConfig config) {
    try {
        close();
        if (model_path == nullptr || model.tensors.empty() ||
            config.scheduler.chunk_bytes == 0 ||
            config.scheduler.chunk_bytes % sizeof(float) != 0 ||
            config.max_layer_buffer_bytes == 0) {
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
        max_layer_buffer_bytes_ = config.max_layer_buffer_bytes;
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
    max_layer_buffer_bytes_ = InferenceConfig{}.max_layer_buffer_bytes;
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

bool InferenceRuntime::stream_f32_tensor(std::size_t tensor_index, float* destination,
                                         std::size_t capacity_floats) noexcept {
    if (destination == nullptr || capacity_floats == 0 ||
        tensor_index >= model_.tensors.size() || scheduler_.has_current() ||
        scheduler_.stats().next_layer_index != tensor_index) {
        return false;
    }
    const auto& tensor = model_.tensors[tensor_index];
    if (tensor.dtype != aabhas::model::TensorDType::F32 || tensor.shape.empty()) {
        return false;
    }
    aabhas::model::TensorLayout layout;
    if (!aabhas::model::describe_tensor_layout(tensor.dtype, tensor.shape, layout) ||
        layout.element_count == 0 ||
        layout.element_count > static_cast<std::uint64_t>(capacity_floats) ||
        layout.element_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        layout.byte_size > tensor.byte_size) {
        return false;
    }
    const std::size_t elements = static_cast<std::size_t>(layout.element_count);

    if (!scheduler_.next_layer()) return false;
    std::size_t processed = 0;
    while (scheduler_.has_current()) {
        const auto view = scheduler_.current();
        if (!view || view.size % sizeof(float) != 0) break;
        const std::size_t values = view.size / sizeof(float);
        if (processed > elements || values > elements - processed) break;
        std::memcpy(destination + processed, view.data, values * sizeof(float));
        processed += values;
        if (processed == elements) break;
        if (!scheduler_.next_chunk()) break;
    }
    scheduler_.release();
    return processed == elements;
}

bool InferenceRuntime::load_transformer_layer(const TransformerLayerPlan& plan,
                                              std::vector<float>& arena,
                                              TransformerLayerWeights& weights) {
    struct Item {
        std::size_t tensor_index{0};
        std::size_t offset_floats{0};
        std::size_t float_count{0};
    };
    Item items[] = {
        {plan.attention_norm.tensor_index, plan.attention_norm.offset_floats,
         plan.attention_norm.float_count},
        {plan.attention_query.tensor_index, plan.attention_query.offset_floats,
         plan.attention_query.float_count},
        {plan.attention_key.tensor_index, plan.attention_key.offset_floats,
         plan.attention_key.float_count},
        {plan.attention_value.tensor_index, plan.attention_value.offset_floats,
         plan.attention_value.float_count},
        {plan.attention_output.tensor_index, plan.attention_output.offset_floats,
         plan.attention_output.float_count},
        {plan.feed_forward_norm.tensor_index, plan.feed_forward_norm.offset_floats,
         plan.feed_forward_norm.float_count},
        {plan.feed_forward_gate.tensor_index, plan.feed_forward_gate.offset_floats,
         plan.feed_forward_gate.float_count},
        {plan.feed_forward_up.tensor_index, plan.feed_forward_up.offset_floats,
         plan.feed_forward_up.float_count},
        {plan.feed_forward_down.tensor_index, plan.feed_forward_down.offset_floats,
         plan.feed_forward_down.float_count},
    };
    constexpr std::size_t item_count = sizeof(items) / sizeof(items[0]);
    if (plan.total_floats == 0 || plan.total_floats > max_layer_buffer_bytes_ / sizeof(float)) {
        return false;
    }
    try {
        std::sort(items, items + item_count,
                  [](const Item& lhs, const Item& rhs) {
                      return lhs.tensor_index < rhs.tensor_index;
                  });
        arena.assign(plan.total_floats, 0.0F);
    } catch (...) {
        return false;
    }
    for (std::size_t index = 0; index < item_count; ++index) {
        const Item& item = items[index];
        if (item.float_count == 0 || item.offset_floats > plan.total_floats ||
            item.float_count > plan.total_floats - item.offset_floats) {
            arena.clear();
            return false;
        }
        if (!stream_f32_tensor(item.tensor_index,
                               arena.data() + item.offset_floats,
                               item.float_count)) {
            arena.clear();
            scheduler_.rewind();
            return false;
        }
    }
    weights = bind_transformer_layer_plan(plan, arena.data());
    return true;
}

bool InferenceRuntime::rewind_stream() noexcept {
    return scheduler_.rewind();
}

} // namespace aabhas::inference
