#include "inference/inference_runtime.hpp"
#include "inference/transformer_layers.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t model_dimension = 4;
constexpr std::size_t head_count = 2;
constexpr std::size_t kv_head_count = 1;
constexpr std::size_t head_dimension = 2;
constexpr std::size_t intermediate_dimension = 4;
constexpr std::size_t layer_data_floats = 104;

bool near(float lhs, float rhs, float tolerance = 1.0e-4F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

struct LayerFixture {
    float attention_norm[model_dimension] = {1.0F, 1.0F, 1.0F, 1.0F};
    float attention_query[model_dimension * model_dimension] = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    float attention_key[kv_head_count * head_dimension * model_dimension] = {
        0.5F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.5F, 0.0F, 0.0F,
    };
    float attention_value[kv_head_count * head_dimension * model_dimension] = {
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    float attention_output[model_dimension * model_dimension] = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    float feed_forward_norm[model_dimension] = {1.0F, 1.0F, 1.0F, 1.0F};
    float feed_forward_gate[intermediate_dimension * model_dimension] = {
        0.25F, -0.5F, 0.0F, 0.0F,
        0.0F, 0.5F, 0.25F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
    };
    float feed_forward_up[intermediate_dimension * model_dimension] = {
        1.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
    };
    float feed_forward_down[model_dimension * intermediate_dimension] = {
        0.5F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.5F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.5F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.5F,
    };
};

aabhas::inference::TransformerBlockConfig block_config() {
    aabhas::inference::TransformerBlockConfig config;
    config.model_dimension = model_dimension;
    config.head_count = head_count;
    config.kv_head_count = kv_head_count;
    config.head_dimension = head_dimension;
    config.intermediate_dimension = intermediate_dimension;
    return config;
}

aabhas::inference::TransformerLayerWeights bind_fixture(const LayerFixture& fixture) {
    aabhas::inference::TransformerLayerWeights weights;
    weights.attention_norm = fixture.attention_norm;
    weights.attention_query = fixture.attention_query;
    weights.attention_key = fixture.attention_key;
    weights.attention_value = fixture.attention_value;
    weights.attention_output = fixture.attention_output;
    weights.feed_forward_norm = fixture.feed_forward_norm;
    weights.feed_forward_gate = fixture.feed_forward_gate;
    weights.feed_forward_up = fixture.feed_forward_up;
    weights.feed_forward_down = fixture.feed_forward_down;
    return weights;
}

aabhas::model::ModelIndex make_model(aabhas::model::ModelFormat format,
                                     std::size_t layer_index) {
    aabhas::inference::TransformerLayerNaming naming;
    assert(aabhas::inference::llama_transformer_layer_naming(format, layer_index, naming));
    aabhas::model::ModelIndex model;
    model.format = format;
    std::uint64_t offset = 0;
    const auto add = [&](const std::string& name,
                         std::initializer_list<std::uint64_t> shape) {
        std::vector<std::uint64_t> extents(shape);
        std::uint64_t elements = 1;
        for (const std::uint64_t extent : extents) elements *= extent;
        model.tensors.push_back({name, offset,
                                 elements * static_cast<std::uint64_t>(sizeof(float)),
                                 aabhas::model::TensorDType::F32, std::move(extents)});
        offset += elements * static_cast<std::uint64_t>(sizeof(float));
    };
    add(naming.attention_norm, {model_dimension});
    add(naming.attention_query, {model_dimension, model_dimension});
    add(naming.attention_key, {kv_head_count * head_dimension, model_dimension});
    add(naming.attention_value, {kv_head_count * head_dimension, model_dimension});
    add(naming.attention_output, {model_dimension, model_dimension});
    add(naming.feed_forward_norm, {model_dimension});
    add(naming.feed_forward_gate, {intermediate_dimension, model_dimension});
    add(naming.feed_forward_up, {intermediate_dimension, model_dimension});
    add(naming.feed_forward_down, {model_dimension, intermediate_dimension});
    model.file_size = offset;
    return model;
}

void test_direct_block_forward() {
    const LayerFixture fixture;
    const aabhas::inference::TransformerLayerWeights weights = bind_fixture(fixture);
    const aabhas::inference::TransformerBlockConfig config = block_config();

    aabhas::inference::KvCache cache;
    assert(cache.configure(1, kv_head_count, head_dimension, 8, 1ULL << 20));

    const float input[2 * model_dimension] = {1.0F, 2.0F, 3.0F, 4.0F,
                                              0.5F, -1.0F, 2.0F, 0.0F};
    float output[2 * model_dimension]{};
    std::vector<float> scratch(
        aabhas::inference::transformer_block_scratch_floats(config));
    assert(scratch.size() == 28);
    assert(!aabhas::inference::transformer_block_forward(
        config, weights, 0, 0, input, 2, cache, output, scratch.data(),
        scratch.size() - 1));
    assert(aabhas::inference::transformer_block_forward(
        config, weights, 0, 0, input, 2, cache, output, scratch.data(),
        scratch.size()));
    assert(near(output[0], 2.0050404F));
    assert(near(output[1], 3.9014821F));
    assert(near(output[2], 4.0954447F));
    assert(near(output[3], 5.4605923F));
    assert(near(output[4], 2.0367658F));
    assert(near(output[5], -0.2098939F));
    assert(near(output[6], 3.3830919F));
    assert(near(output[7], 0.8145204F));

    assert(cache.key(0, 0) != nullptr && cache.value(0, 0) != nullptr);
    assert(near(cache.key(0, 0)[0], 0.1825741F));
    assert(near(cache.key(0, 1)[0], 0.4851498F));
    assert(near(cache.value(0, 0)[0], 1.0954444F));
    assert(near(cache.value(0, 1)[0], 1.7457365F));
    assert(near(cache.value(0, 1)[1], 0.0F));

    const float decode[model_dimension] = {0.0F, 1.0F, 2.0F, 1.0F};
    float decoded[model_dimension]{};
    assert(aabhas::inference::transformer_block_forward(
        config, weights, 0, 2, decode, 1, cache, decoded, scratch.data(),
        scratch.size()));
    assert(near(decoded[0], 1.4370259F));
    assert(near(decoded[1], 2.3775706F));
    assert(near(decoded[2], 3.4530530F));
    assert(near(decoded[3], 1.8971994F));
    assert(near(cache.key(0, 2)[0], -0.3712179F));
    assert(near(cache.value(0, 2)[1], 0.8164939F));

    aabhas::inference::KvCache sequential_cache;
    assert(sequential_cache.configure(1, kv_head_count, head_dimension, 8, 1ULL << 20));
    float sequential[2 * model_dimension]{};
    assert(aabhas::inference::transformer_block_forward(
        config, weights, 0, 0, input, 1, sequential_cache, sequential,
        scratch.data(), scratch.size()));
    assert(aabhas::inference::transformer_block_forward(
        config, weights, 0, 1, input + model_dimension, 1, sequential_cache,
        sequential + model_dimension, scratch.data(), scratch.size()));
    for (std::size_t index = 0; index < 2 * model_dimension; ++index) {
        assert(near(sequential[index], output[index]));
    }

    aabhas::inference::KvCache narrow_cache;
    assert(narrow_cache.configure(1, head_count, head_dimension, 8, 1ULL << 20));
    assert(!aabhas::inference::transformer_block_forward(
        config, weights, 0, 0, input, 1, narrow_cache, output, scratch.data(),
        scratch.size()));

    aabhas::inference::TransformerBlockConfig invalid = config;
    invalid.head_count = 3;
    assert(aabhas::inference::transformer_block_scratch_floats(invalid) == 0);
    assert(!aabhas::inference::transformer_block_forward(
        invalid, weights, 0, 0, input, 1, cache, output, scratch.data(),
        scratch.size()));
}

void test_plan_resolution() {
    const aabhas::inference::TransformerBlockConfig config = block_config();
    for (const aabhas::model::ModelFormat format :
         {aabhas::model::ModelFormat::GGUF, aabhas::model::ModelFormat::SafeTensors}) {
        aabhas::model::ModelIndex model = make_model(format, 3);
        aabhas::inference::TransformerLayerNaming naming;
        assert(aabhas::inference::llama_transformer_layer_naming(format, 3, naming));
        aabhas::inference::TransformerLayerPlan plan;
        assert(aabhas::inference::build_transformer_layer_plan(model, naming, config,
                                                               plan));
        assert(plan.total_floats == layer_data_floats);
        assert(plan.naming.attention_norm == naming.attention_norm);
        assert(plan.attention_norm.float_count == model_dimension);
        assert(plan.attention_query.float_count ==
               model_dimension * model_dimension);
        assert(plan.attention_key.float_count == kv_head_count * head_dimension *
                                                 model_dimension);
        assert(plan.feed_forward_gate.float_count ==
               intermediate_dimension * model_dimension);
        assert(plan.attention_norm.offset_floats == 0);
        assert(plan.attention_query.offset_floats == model_dimension);
        assert(plan.attention_key.offset_floats ==
               model_dimension + model_dimension * model_dimension);
        assert(plan.feed_forward_down.offset_floats ==
               layer_data_floats -
                   model_dimension * intermediate_dimension);
        assert(plan.feed_forward_down.offset_floats +
                   plan.feed_forward_down.float_count ==
               layer_data_floats);

        std::vector<float> arena(plan.total_floats, 0.0F);
        const aabhas::inference::TransformerLayerWeights weights =
            aabhas::inference::bind_transformer_layer_plan(plan, arena.data());
        assert(weights.attention_norm == arena.data() + plan.attention_norm.offset_floats);
        assert(weights.feed_forward_down ==
               arena.data() + plan.feed_forward_down.offset_floats);
        assert(aabhas::inference::bind_transformer_layer_plan(plan, nullptr)
                   .attention_norm == nullptr);
    }

    aabhas::inference::TransformerLayerNaming naming;
    assert(!aabhas::inference::llama_transformer_layer_naming(
        aabhas::model::ModelFormat::RawBinary, 0, naming));

    aabhas::model::ModelIndex malformed = make_model(
        aabhas::model::ModelFormat::GGUF, 0);
    assert(aabhas::inference::llama_transformer_layer_naming(
        aabhas::model::ModelFormat::GGUF, 0, naming));
    malformed.tensors[1].shape[0] = 3;
    aabhas::inference::TransformerLayerPlan rejected;
    assert(!aabhas::inference::build_transformer_layer_plan(malformed, naming, config,
                                                            rejected));

    aabhas::model::ModelIndex duplicate = make_model(
        aabhas::model::ModelFormat::GGUF, 0);
    duplicate.tensors[5].name = naming.attention_norm;
    assert(!aabhas::inference::build_transformer_layer_plan(duplicate, naming, config,
                                                            rejected));
}

void test_streamed_block_forward() {
    const auto path =
        std::filesystem::temp_directory_path() / "aabhas-transformer-block-test.bin";
    const LayerFixture fixture;
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        assert(output.is_open());
        const auto write = [&](const float* values, std::size_t count) {
            output.write(reinterpret_cast<const char*>(values),
                         static_cast<std::streamsize>(count * sizeof(float)));
        };
        write(fixture.attention_norm, model_dimension);
        write(fixture.attention_query, model_dimension * model_dimension);
        write(fixture.attention_key, kv_head_count * head_dimension * model_dimension);
        write(fixture.attention_value, kv_head_count * head_dimension * model_dimension);
        write(fixture.attention_output, model_dimension * model_dimension);
        write(fixture.feed_forward_norm, model_dimension);
        write(fixture.feed_forward_gate, intermediate_dimension * model_dimension);
        write(fixture.feed_forward_up, intermediate_dimension * model_dimension);
        write(fixture.feed_forward_down, model_dimension * intermediate_dimension);
        output.close();
    }

    aabhas::model::ModelIndex model = make_model(aabhas::model::ModelFormat::GGUF, 0);
    aabhas::inference::InferenceConfig config;
    config.scheduler.memory_cap_bytes = 64;
    config.scheduler.chunk_bytes = 16;
    config.scheduler.lookahead_layers = 0;
    aabhas::inference::InferenceRuntime runtime;
    assert(runtime.open(path.string().c_str(), model, config));

    aabhas::inference::TransformerLayerNaming naming;
    assert(aabhas::inference::llama_transformer_layer_naming(
        aabhas::model::ModelFormat::GGUF, 0, naming));
    const aabhas::inference::TransformerBlockConfig block = block_config();
    aabhas::inference::TransformerLayerPlan plan;
    assert(aabhas::inference::build_transformer_layer_plan(model, naming, block, plan));
    std::vector<float> arena;
    aabhas::inference::TransformerLayerWeights weights;
    assert(runtime.load_transformer_layer(plan, arena, weights));
    assert(arena.size() == layer_data_floats);
    assert(weights.attention_query == arena.data() + plan.attention_query.offset_floats);

    aabhas::inference::KvCache cache;
    assert(cache.configure(1, kv_head_count, head_dimension, 8, 1ULL << 20));
    const float input[model_dimension] = {1.0F, 2.0F, 3.0F, 4.0F};
    float output[model_dimension]{};
    std::vector<float> scratch(
        aabhas::inference::transformer_block_scratch_floats(block));
    assert(aabhas::inference::transformer_block_forward(
        block, weights, 0, 0, input, 1, cache, output, scratch.data(),
        scratch.size()));
    assert(near(output[0], 2.0050404F));
    assert(near(output[3], 5.4605923F));

    assert(runtime.rewind_stream());
    arena.clear();
    assert(runtime.load_transformer_layer(plan, arena, weights));
    assert(arena.size() == layer_data_floats);
    assert(!runtime.load_transformer_layer(plan, arena, weights));
    assert(runtime.load_transformer_layer(plan, arena, weights));

    runtime.close();
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_direct_block_forward();
    test_plan_resolution();
    test_streamed_block_forward();
    return 0;
}
