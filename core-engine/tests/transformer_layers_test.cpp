#include "inference/transformer_layers.hpp"

#include <cassert>
#include <cmath>

namespace {

bool near(float lhs, float rhs, float tolerance = 1.0e-4F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

} // namespace

int main() {
    aabhas::inference::TransformerShape shape;
    shape.model_dimension = 8;
    shape.head_count = 2;
    shape.head_dimension = 4;
    shape.kv_head_count = 1;
    shape.intermediate_dimension = 16;
    shape.max_sequence_length = 32;
    std::size_t required_kv_bytes = 0;
    assert(aabhas::inference::validate_transformer_shape(shape, 4096, required_kv_bytes));
    assert(required_kv_bytes == 1 * 32 * 4 * sizeof(float) * 2);
    shape.model_dimension = 7;
    assert(!aabhas::inference::validate_transformer_shape(shape, 4096, required_kv_bytes));

    const float input[] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float norm_weight[] = {1.0F, 1.0F, 1.0F, 1.0F};
    float normalized[4]{};
    assert(aabhas::inference::kernels::rms_norm(
        input, norm_weight, 4, 1.0e-5F, normalized));
    assert(near(normalized[0], 0.365148F));
    assert(near(normalized[3], 1.460592F));

    const float gate[] = {0.0F, 1.0F};
    const float up[] = {2.0F, 2.0F};
    float gated[2]{};
    assert(aabhas::inference::kernels::swiglu(gate, up, 2, gated));
    assert(near(gated[0], 0.0F));
    assert(near(gated[1], 1.462117F));

    float query[] = {1.0F, 0.0F};
    float key[] = {1.0F, 0.0F};
    assert(aabhas::inference::kernels::apply_rope(query, key, 2, 1, 10000.0F));
    assert(near(query[0], std::cos(1.0F)));
    assert(near(query[1], std::sin(1.0F)));

    const float attention_query[] = {1.0F, 0.0F};
    const float attention_keys[] = {1.0F, 0.0F, 0.0F, 1.0F};
    const float attention_values[] = {2.0F, 0.0F, 0.0F, 4.0F};
    float attention_output[2]{};
    assert(aabhas::inference::kernels::causal_attention(
        attention_query, attention_keys, attention_values, 2, 2, 1.0F,
        attention_output));
    const float weight = std::exp(1.0F) / (std::exp(1.0F) + 1.0F);
    assert(near(attention_output[0], 2.0F * weight));
    assert(near(attention_output[1], 4.0F * (1.0F - weight)));
    return 0;
}
