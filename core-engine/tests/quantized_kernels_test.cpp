#include "inference/inference_runtime.hpp"
#include "inference/quantized_kernels.hpp"
#include "inference/transformer_layers.hpp"
#include "model/model_index.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

bool near(float a, float b, float tol = 1.0e-3F) {
    return std::fabs(a - b) <= tol;
}

void test_fp16_bf16() {
    const float fp16_vals[] = {0.0F, -0.0F, 1.0F, -1.0F, 0.5F, 2.0F, -4.0F, 65504.0F};
    for (float val : fp16_vals) {
        const std::uint16_t h = aabhas::inference::kernels::fp32_to_fp16(val);
        const float out = aabhas::inference::kernels::fp16_to_fp32(h);
        assert(near(val, out, 1.0e-4F));
    }

    const float bf16_vals[] = {0.0F, -0.0F, 1.0F, -1.0F, 0.5F, 2.0F, -4.0F, 65280.0F};
    for (float val : bf16_vals) {
        const std::uint16_t b = aabhas::inference::kernels::fp32_to_bf16(val);
        const float b_out = aabhas::inference::kernels::bf16_to_fp32(b);
        assert(near(val, b_out, 1.0e-2F));
    }

    // Relative precision check for arbitrary values in BF16 (7-bit mantissa -> ~0.8% error)
    const float arbitrary = 65504.0F;
    const std::uint16_t b_arb = aabhas::inference::kernels::fp32_to_bf16(arbitrary);
    const float arb_out = aabhas::inference::kernels::bf16_to_fp32(b_arb);
    const float rel_err = std::fabs(arbitrary - arb_out) / arbitrary;
    assert(rel_err < 0.01F);

    // Subnormal fp16 test: smallest positive subnormal is 2^-24
    const std::uint16_t subnormal = 0x0001U;
    const float subnormal_f = aabhas::inference::kernels::fp16_to_fp32(subnormal);
    assert(subnormal_f > 0.0F);
    assert(near(subnormal_f, 5.9604645e-8F, 1.0e-12F));
}

void test_q4_0_block() {
    aabhas::inference::kernels::BlockQ4_0 block{};
    block.d = aabhas::inference::kernels::fp32_to_fp16(0.5F);
    // qs[0] = 0x98: low nibble 8 -> 8-8=0; high nibble 9 -> 9-8=+1
    block.qs[0] = 0x98;
    // qs[1] = 0x0F: low nibble 15 -> 15-8=+7; high nibble 0 -> 0-8=-8
    block.qs[1] = 0x0F;
    for (std::size_t i = 2; i < 16; ++i) {
        block.qs[i] = 0x88; // both 8 -> 0
    }

    float out[32] = {0};
    aabhas::inference::kernels::dequantize_q4_0_block(&block, out);
    assert(near(out[0], 0.0F));
    assert(near(out[16], 0.5F)); // high nibble of qs[0] is at index 16
    assert(near(out[1], 3.5F));  // low nibble of qs[1] is 15-8=7 * 0.5
    assert(near(out[17], -4.0F)); // high nibble of qs[1] is 0-8=-8 * 0.5
}

void test_q4_1_block() {
    aabhas::inference::kernels::BlockQ4_1 block{};
    block.d = aabhas::inference::kernels::fp32_to_fp16(0.25F);
    block.m = aabhas::inference::kernels::fp32_to_fp16(-2.0F);
    // qs[0] = 0x24: low 4 -> 4*0.25 - 2 = -1; high 2 -> 2*0.25 - 2 = -1.5
    block.qs[0] = 0x24;
    for (std::size_t i = 1; i < 16; ++i) {
        block.qs[i] = 0x00;
    }

    float out[32] = {0};
    aabhas::inference::kernels::dequantize_q4_1_block(&block, out);
    assert(near(out[0], -1.0F));
    assert(near(out[16], -1.5F));
}

void test_q5_0_block() {
    aabhas::inference::kernels::BlockQ5_0 block{};
    block.d = aabhas::inference::kernels::fp32_to_fp16(0.5F);
    // qh: bit 0 = 1, bit 16 = 0
    const std::uint32_t qh = 0x00000001U;
    std::memcpy(block.qh, &qh, sizeof(qh));
    // qs[0] = 0x00: low 0 | (1<<4) = 16 -> 16-16 = 0; high 0 | (0<<4) = 0 -> 0-16 = -16
    block.qs[0] = 0x00;
    for (std::size_t i = 1; i < 16; ++i) {
        block.qs[i] = 0x00;
    }

    float out[32] = {0};
    aabhas::inference::kernels::dequantize_q5_0_block(&block, out);
    assert(near(out[0], 0.0F));
    assert(near(out[16], -8.0F)); // -16 * 0.5
}

void test_q5_1_block() {
    aabhas::inference::kernels::BlockQ5_1 block{};
    block.d = aabhas::inference::kernels::fp32_to_fp16(0.5F);
    block.m = aabhas::inference::kernels::fp32_to_fp16(1.0F);
    const std::uint32_t qh = 0x00000001U; // bit 0 is 1
    std::memcpy(block.qh, &qh, sizeof(qh));
    block.qs[0] = 0x04; // low 4 | (1<<4) = 20 -> 20*0.5 + 1 = 11; high 0 -> 0*0.5 + 1 = 1
    for (std::size_t i = 1; i < 16; ++i) {
        block.qs[i] = 0x00;
    }

    float out[32] = {0};
    aabhas::inference::kernels::dequantize_q5_1_block(&block, out);
    assert(near(out[0], 11.0F));
    assert(near(out[16], 1.0F));
}

void test_q8_0_block() {
    aabhas::inference::kernels::BlockQ8_0 block{};
    block.d = aabhas::inference::kernels::fp32_to_fp16(0.125F);
    block.qs[0] = 10;
    block.qs[1] = -20;
    for (std::size_t i = 2; i < 32; ++i) {
        block.qs[i] = 0;
    }

    float out[32] = {0};
    aabhas::inference::kernels::dequantize_q8_0_block(&block, out);
    assert(near(out[0], 1.25F));
    assert(near(out[1], -2.5F));
    assert(near(out[2], 0.0F));
}

void test_q8_1_block() {
    aabhas::inference::kernels::BlockQ8_1 block{};
    block.d = 0.5F;
    block.s = 2.0F;
    block.qs[0] = 4;
    block.qs[1] = -6;
    for (std::size_t i = 2; i < 32; ++i) {
        block.qs[i] = 0;
    }

    float out[32] = {0};
    aabhas::inference::kernels::dequantize_q8_1_block(&block, out);
    assert(near(out[0], 2.0F));
    assert(near(out[1], -3.0F));
}

void test_roundtrip_quantization() {
    float src[64];
    for (std::size_t i = 0; i < 64; ++i) {
        src[i] = static_cast<float>(i) * 0.1F - 3.2F;
    }

    // Test Q8_0 roundtrip
    {
        std::uint8_t buffer[68]; // 2 blocks of 34 bytes
        assert(aabhas::inference::kernels::quantize_row(
            aabhas::model::TensorDType::Q8_0, src, 64, buffer, sizeof(buffer)));
        float dequant[64];
        assert(aabhas::inference::kernels::dequantize_row(
            aabhas::model::TensorDType::Q8_0, buffer, sizeof(buffer), dequant, 64));
        for (std::size_t i = 0; i < 64; ++i) {
            assert(near(src[i], dequant[i], 0.05F));
        }
    }

    // Test Q4_0 roundtrip
    {
        std::uint8_t buffer[36]; // 2 blocks of 18 bytes
        assert(aabhas::inference::kernels::quantize_row(
            aabhas::model::TensorDType::Q4_0, src, 64, buffer, sizeof(buffer)));
        float dequant[64];
        assert(aabhas::inference::kernels::dequantize_row(
            aabhas::model::TensorDType::Q4_0, buffer, sizeof(buffer), dequant, 64));
        for (std::size_t i = 0; i < 64; ++i) {
            assert(near(src[i], dequant[i], 0.5F));
        }
    }

    // Test Q4_1 roundtrip
    {
        std::uint8_t buffer[40]; // 2 blocks of 20 bytes
        assert(aabhas::inference::kernels::quantize_row(
            aabhas::model::TensorDType::Q4_1, src, 64, buffer, sizeof(buffer)));
        float dequant[64];
        assert(aabhas::inference::kernels::dequantize_row(
            aabhas::model::TensorDType::Q4_1, buffer, sizeof(buffer), dequant, 64));
        for (std::size_t i = 0; i < 64; ++i) {
            assert(near(src[i], dequant[i], 0.5F));
        }
    }
}

void test_vec_dot_and_gemv() {
    constexpr std::size_t k = 64;
    float x[k];
    float w_f32[k];
    for (std::size_t i = 0; i < k; ++i) {
        x[i] = 0.5F + static_cast<float>(i % 3) * 0.25F;
        w_f32[i] = static_cast<float>(i % 5) - 2.0F;
    }

    // Compute float32 reference dot product
    float ref_dot = 0.0F;
    assert(aabhas::inference::kernels::vec_dot(
        aabhas::model::TensorDType::F32, w_f32, x, k, ref_dot));

    // Quantize to Q8_0 and compare vec_dot with dequantized dot
    std::uint8_t q8_buf[68];
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, w_f32, k, q8_buf, sizeof(q8_buf)));
    float q8_dot = 0.0F;
    assert(aabhas::inference::kernels::vec_dot(
        aabhas::model::TensorDType::Q8_0, q8_buf, x, k, q8_dot));
    float dequant_q8[k];
    assert(aabhas::inference::kernels::dequantize_row(
        aabhas::model::TensorDType::Q8_0, q8_buf, sizeof(q8_buf), dequant_q8, k));
    float expected_q8_dot = 0.0F;
    for (std::size_t i = 0; i < k; ++i) expected_q8_dot += dequant_q8[i] * x[i];
    assert(near(q8_dot, expected_q8_dot, 1.0e-3F));

    // Quantize to Q4_0 and compare vec_dot with dequantized dot
    std::uint8_t q4_buf[36];
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q4_0, w_f32, k, q4_buf, sizeof(q4_buf)));
    float q4_dot = 0.0F;
    assert(aabhas::inference::kernels::vec_dot(
        aabhas::model::TensorDType::Q4_0, q4_buf, x, k, q4_dot));
    float dequant_q4[k];
    assert(aabhas::inference::kernels::dequantize_row(
        aabhas::model::TensorDType::Q4_0, q4_buf, sizeof(q4_buf), dequant_q4, k));
    float expected_q4_dot = 0.0F;
    for (std::size_t i = 0; i < k; ++i) expected_q4_dot += dequant_q4[i] * x[i];
    assert(near(q4_dot, expected_q4_dot, 1.0e-3F));

    // Test gemv_quantized (2 rows x 64 columns)
    std::uint8_t q8_matrix[136]; // 2 rows * 68 bytes
    std::memcpy(q8_matrix, q8_buf, 68);
    std::memcpy(q8_matrix + 68, q8_buf, 68);
    float y[2] = {0.0F, 0.0F};
    assert(aabhas::inference::kernels::gemv_quantized(
        aabhas::model::TensorDType::Q8_0, q8_matrix, 2, k, x, y));
    assert(near(y[0], q8_dot, 1.0e-3F));
    assert(near(y[1], q8_dot, 1.0e-3F));
}

void test_stream_decoder_arbitrary_chunks() {
    float src[64];
    for (std::size_t i = 0; i < 64; ++i) {
        src[i] = static_cast<float>(i) * 0.25F - 8.0F;
    }
    std::uint8_t raw_bytes[68];
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, src, 64, raw_bytes, sizeof(raw_bytes)));

    float ref_dequant[64];
    assert(aabhas::inference::kernels::dequantize_row(
        aabhas::model::TensorDType::Q8_0, raw_bytes, sizeof(raw_bytes), ref_dequant, 64));

    // Feed in strange prime chunk sizes (e.g. 1, 3, 7, 13 bytes)
    const std::size_t chunk_sizes[] = {1, 2, 3, 5, 7, 11, 17, 34, 68};
    for (std::size_t c_size : chunk_sizes) {
        float decoded[64] = {0};
        aabhas::inference::kernels::QuantizedStreamDecoder decoder;
        assert(decoder.init(aabhas::model::TensorDType::Q8_0, decoded, 64));

        std::size_t fed = 0;
        while (fed < sizeof(raw_bytes)) {
            const std::size_t to_feed = std::min(c_size, sizeof(raw_bytes) - fed);
            assert(decoder.feed(raw_bytes + fed, to_feed));
            fed += to_feed;
        }
        assert(decoder.finish());
        assert(decoder.elements_decoded() == 64);
        for (std::size_t i = 0; i < 64; ++i) {
            assert(near(decoded[i], ref_dequant[i], 1.0e-4F));
        }
    }
}

void test_inference_runtime_quantized_stream() {
    const auto path =
        std::filesystem::temp_directory_path() / "aabhas-quantized-runtime-test.bin";

    constexpr std::size_t cols = 32;
    constexpr std::size_t rows = 2;
    float row0[cols];
    float row1[cols];
    for (std::size_t i = 0; i < cols; ++i) {
        row0[i] = 1.0F;
        row1[i] = 2.0F;
    }

    // Write a Q8_0 matrix (2 rows * 32 cols = 2 * 34 = 68 bytes)
    std::uint8_t q8_matrix[68];
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, row0, cols, q8_matrix, 34));
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, row1, cols, q8_matrix + 34, 34));

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        out.write(reinterpret_cast<const char*>(q8_matrix), sizeof(q8_matrix));
        assert(out.good());
    }

    aabhas::model::ModelIndex model;
    model.file_size = sizeof(q8_matrix);
    model.tensors.push_back({"quantized_weight", 0, sizeof(q8_matrix),
                             aabhas::model::TensorDType::Q8_0, {rows, cols}});

    aabhas::inference::InferenceRuntime runtime;
    aabhas::inference::InferenceConfig config;
    config.scheduler.memory_cap_bytes = 17; // small memory cap
    config.scheduler.chunk_bytes = 17;      // unaligned chunk size
    config.scheduler.lookahead_layers = 0;
    assert(runtime.open(path.string().c_str(), model, config));

    float input[cols];
    for (std::size_t i = 0; i < cols; ++i) input[i] = 1.0F;

    float result[rows] = {0};
    assert(runtime.run_matvec(0, input, cols, result, rows));
    assert(near(result[0], 32.0F, 0.5F));
    assert(near(result[1], 64.0F, 1.0F));

    assert(runtime.rewind_stream());
    float stream_dest[rows * cols] = {0};
    assert(runtime.stream_tensor(0, stream_dest, rows * cols));
    assert(near(stream_dest[0], 1.0F, 0.05F));
    assert(near(stream_dest[cols], 2.0F, 0.05F));

    runtime.close();
    std::filesystem::remove(path);
}

void test_quantized_transformer_layer_execution() {
    // Test that a transformer layer plan accepts quantized weights and executes
    // seamlessly through load_transformer_layer and transformer_block_forward!
    constexpr std::size_t dim = 32;
    constexpr std::size_t heads = 2;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t head_dim = 16;
    constexpr std::size_t ffn_dim = 64;

    const auto path =
        std::filesystem::temp_directory_path() / "aabhas-quantized-block-test.bin";

    // 9 tensors:
    // 1. attention_norm: [dim] (F32) = 32 * 4 = 128 bytes
    // 2. attention_query: [dim, dim] (Q8_0) = 32 * 34 = 1088 bytes
    // 3. attention_key: [kv_heads * head_dim, dim] = [16, 32] (Q8_0) = 16 * 34 = 544 bytes
    // 4. attention_value: [16, 32] (Q8_0) = 544 bytes
    // 5. attention_output: [dim, dim] (Q8_0) = 1088 bytes
    // 6. feed_forward_norm: [dim] (F32) = 128 bytes
    // 7. feed_forward_gate: [ffn_dim, dim] = [64, 32] (Q4_0) = 64 * 18 = 1152 bytes
    // 8. feed_forward_up: [64, 32] (Q4_0) = 1152 bytes
    // 9. feed_forward_down: [dim, ffn_dim] = [32, 64] (Q4_0) = 32 * 36 = 1152 bytes

    std::vector<float> f32_norm(dim, 1.0F);
    std::vector<float> f32_eye(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) f32_eye[i * dim + i] = 1.0F;

    std::vector<float> f32_kv(16 * dim, 0.0F);
    for (std::size_t i = 0; i < 16; ++i) f32_kv[i * dim + i] = 0.5F;

    std::vector<float> f32_gate(ffn_dim * dim, 0.0F);
    for (std::size_t i = 0; i < 16; ++i) f32_gate[i * dim + i] = 0.25F;

    std::vector<float> f32_down(dim * ffn_dim, 0.0F);
    for (std::size_t i = 0; i < 16; ++i) f32_down[i * ffn_dim + i] = 0.5F;

    std::vector<std::uint8_t> q_query(32 * 34);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, f32_eye.data(), dim * dim,
        q_query.data(), q_query.size()));

    std::vector<std::uint8_t> q_key(16 * 34);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, f32_kv.data(), 16 * dim,
        q_key.data(), q_key.size()));

    std::vector<std::uint8_t> q_val(16 * 34);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, f32_kv.data(), 16 * dim,
        q_val.data(), q_val.size()));

    std::vector<std::uint8_t> q_out(32 * 34);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q8_0, f32_eye.data(), dim * dim,
        q_out.data(), q_out.size()));

    std::vector<std::uint8_t> q_gate(64 * 18);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q4_0, f32_gate.data(), ffn_dim * dim,
        q_gate.data(), q_gate.size()));

    std::vector<std::uint8_t> q_up(64 * 18);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q4_0, f32_gate.data(), ffn_dim * dim,
        q_up.data(), q_up.size()));

    std::vector<std::uint8_t> q_down(32 * 36);
    assert(aabhas::inference::kernels::quantize_row(
        aabhas::model::TensorDType::Q4_0, f32_down.data(), dim * ffn_dim,
        q_down.data(), q_down.size()));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out.is_open());
    const auto write_buf = [&](const void* ptr, std::size_t sz) {
        out.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(sz));
    };

    std::uint64_t offset = 0;
    aabhas::model::ModelIndex model;

    const auto add_t = [&](const std::string& name, aabhas::model::TensorDType dt,
                           const std::vector<std::uint64_t>& shape,
                           const void* data, std::size_t sz) {
        write_buf(data, sz);
        model.tensors.push_back({name, offset, sz, dt, shape});
        offset += sz;
    };

    add_t("blk.0.attn_norm.weight", aabhas::model::TensorDType::F32, {dim},
          f32_norm.data(), dim * sizeof(float));
    add_t("blk.0.attn_q.weight", aabhas::model::TensorDType::Q8_0, {dim, dim},
          q_query.data(), q_query.size());
    add_t("blk.0.attn_k.weight", aabhas::model::TensorDType::Q8_0, {16, dim},
          q_key.data(), q_key.size());
    add_t("blk.0.attn_v.weight", aabhas::model::TensorDType::Q8_0, {16, dim},
          q_val.data(), q_val.size());
    add_t("blk.0.attn_output.weight", aabhas::model::TensorDType::Q8_0, {dim, dim},
          q_out.data(), q_out.size());
    add_t("blk.0.ffn_norm.weight", aabhas::model::TensorDType::F32, {dim},
          f32_norm.data(), dim * sizeof(float));
    add_t("blk.0.ffn_gate.weight", aabhas::model::TensorDType::Q4_0, {ffn_dim, dim},
          q_gate.data(), q_gate.size());
    add_t("blk.0.ffn_up.weight", aabhas::model::TensorDType::Q4_0, {ffn_dim, dim},
          q_up.data(), q_up.size());
    add_t("blk.0.ffn_down.weight", aabhas::model::TensorDType::Q4_0, {dim, ffn_dim},
          q_down.data(), q_down.size());
    out.close();

    model.file_size = offset;

    aabhas::inference::TransformerBlockConfig block_cfg{};
    block_cfg.model_dimension = dim;
    block_cfg.head_count = heads;
    block_cfg.kv_head_count = kv_heads;
    block_cfg.head_dimension = head_dim;
    block_cfg.intermediate_dimension = ffn_dim;
    block_cfg.rms_epsilon = 1.0e-5F;
    block_cfg.rope_theta = 10000.0F;

    aabhas::inference::TransformerLayerNaming naming;
    assert(aabhas::inference::llama_transformer_layer_naming(
        aabhas::model::ModelFormat::GGUF, 0, naming));

    aabhas::inference::TransformerLayerPlan plan;
    assert(aabhas::inference::build_transformer_layer_plan(model, naming, block_cfg, plan));

    aabhas::inference::InferenceRuntime runtime;
    aabhas::inference::InferenceConfig config;
    config.scheduler.memory_cap_bytes = 256;
    config.scheduler.chunk_bytes = 64;
    config.scheduler.lookahead_layers = 0;
    assert(runtime.open(path.string().c_str(), model, config));

    std::vector<float> arena;
    aabhas::inference::TransformerLayerWeights weights;
    assert(runtime.load_transformer_layer(plan, arena, weights));
    assert(arena.size() == plan.total_floats);

    aabhas::inference::KvCache cache;
    assert(cache.configure(1, kv_heads, head_dim, 8, 1ULL << 20));

    std::vector<float> input(dim, 1.0F);
    std::vector<float> output(dim, 0.0F);
    const std::size_t scratch_size =
        aabhas::inference::transformer_block_scratch_floats(block_cfg);
    std::vector<float> scratch(scratch_size, 0.0F);

    assert(aabhas::inference::transformer_block_forward(
        block_cfg, weights, 0, 0, input.data(), 1, cache, output.data(),
        scratch.data(), scratch.size()));

    for (std::size_t i = 0; i < dim; ++i) {
        assert(std::isfinite(output[i]));
    }

    runtime.close();
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_fp16_bf16();
    test_q4_0_block();
    test_q4_1_block();
    test_q5_0_block();
    test_q5_1_block();
    test_q8_0_block();
    test_q8_1_block();
    test_roundtrip_quantization();
    test_vec_dot_and_gemv();
    test_stream_decoder_arbitrary_chunks();
    test_inference_runtime_quantized_stream();
    test_quantized_transformer_layer_execution();
    return 0;
}