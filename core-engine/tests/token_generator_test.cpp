#include "inference/inference_runtime.hpp"
#include "inference/kv_cache.hpp"
#include "inference/token_generator.hpp"
#include "inference/transformer_layers.hpp"
#include "model/model_index.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool near(float a, float b, float tol = 1.0e-3F) {
    return std::fabs(a - b) <= tol;
}

void test_kv_cache_extensions() {
    aabhas::inference::KvCache cache;
    // 2 layers, 2 heads, head_dimension 4 -> token_width = 8
    // max_tokens = 16
    assert(cache.configure(2, 2, 4, 16, 1ULL << 20));
    assert(cache.token_width() == 8);
    assert(cache.max_tokens() == 16);
    assert(cache.layer_count() == 2);
    assert(cache.sequence_length() == 0);

    // Test append single token
    float key_t0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float val_t0[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    assert(cache.append(0, 0, key_t0, val_t0));
    assert(cache.sequence_length() == 1);
    assert(cache.key(0, 0)[0] == 1.0F);
    assert(cache.value(0, 0)[0] == 8.0F);

    // Test append_sequence with 3 tokens (positions 1, 2, 3)
    float keys_seq[24];
    float vals_seq[24];
    for (std::size_t i = 0; i < 24; ++i) {
        keys_seq[i] = static_cast<float>(i + 10);
        vals_seq[i] = static_cast<float>(i + 50);
    }
    assert(cache.append_sequence(0, 1, 3, keys_seq, vals_seq));
    assert(cache.sequence_length() == 4);
    assert(cache.key(0, 1)[0] == 10.0F);
    assert(cache.key(0, 3)[7] == 33.0F);

    // Test contiguous layer pointers
    const float* layer0_keys = cache.layer_keys(0);
    const float* layer0_vals = cache.layer_values(0);
    assert(layer0_keys != nullptr);
    assert(layer0_vals != nullptr);
    assert(layer0_keys[0] == 1.0F);
    assert(layer0_keys[8] == 10.0F); // token 1 starts at offset 8

    // Test truncate to 2 tokens
    assert(cache.truncate(2));
    assert(cache.sequence_length() == 2);

    // Test reset sequence
    cache.reset_sequence();
    assert(cache.sequence_length() == 0);

    // Test invalid bounds
    assert(!cache.truncate(20)); // exceeds max_tokens
    assert(!cache.append(3, 0, key_t0, val_t0)); // invalid layer
    assert(!cache.append(0, 16, key_t0, val_t0)); // invalid token index
    assert(!cache.append_sequence(0, 14, 5, keys_seq, vals_seq)); // overflow
}

void test_build_model_plan() {
    aabhas::model::ModelIndex model;
    model.format = aabhas::model::ModelFormat::GGUF;

    constexpr std::size_t vocab = 8;
    constexpr std::size_t dim = 4;
    constexpr std::size_t layers = 1;

    // Add embedding, norm, and lm_head tensors
    model.tensors.push_back({"token_embd.weight", 0, vocab * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {vocab, dim}});
    model.tensors.push_back({"output_norm.weight", 100, dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {dim}});
    model.tensors.push_back({"output.weight", 200, vocab * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {vocab, dim}});

    // Add layer 0 tensors
    const std::string prefix = "blk.0.";
    model.tensors.push_back({prefix + "attn_norm.weight", 300, dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {dim}});
    model.tensors.push_back({prefix + "attn_q.weight", 400, dim * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {dim, dim}});
    model.tensors.push_back({prefix + "attn_k.weight", 500, 2 * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {2, dim}});
    model.tensors.push_back({prefix + "attn_v.weight", 600, 2 * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {2, dim}});
    model.tensors.push_back({prefix + "attn_output.weight", 700, dim * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {dim, dim}});
    model.tensors.push_back({prefix + "ffn_norm.weight", 800, dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {dim}});
    model.tensors.push_back({prefix + "ffn_gate.weight", 900, 4 * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {4, dim}});
    model.tensors.push_back({prefix + "ffn_up.weight", 1000, 4 * dim * sizeof(float),
                             aabhas::model::TensorDType::F32, {4, dim}});
    model.tensors.push_back({prefix + "ffn_down.weight", 1100, dim * 4 * sizeof(float),
                             aabhas::model::TensorDType::F32, {dim, 4}});

    aabhas::inference::TransformerModelConfig config{};
    config.vocab_size = vocab;
    config.model_dimension = dim;
    config.layer_count = layers;
    config.block.model_dimension = dim;
    config.block.head_count = 2;
    config.block.kv_head_count = 1;
    config.block.head_dimension = 2;
    config.block.intermediate_dimension = 4;
    config.block.rms_epsilon = 1.0e-5F;
    config.block.rope_theta = 10000.0F;

    aabhas::inference::TransformerModelPlan plan;
    assert(aabhas::inference::build_transformer_model_plan(model, model.format, config, plan));
    assert(plan.layers.size() == 1);
    assert(plan.token_embedding.float_count == vocab * dim);
    assert(plan.output_norm.float_count == dim);
    assert(plan.lm_head.float_count == vocab * dim);
    assert(!plan.lm_head_tied_to_embedding);
}

void test_end_to_end_token_generation() {
    const auto path =
        std::filesystem::temp_directory_path() / "aabhas-token-generator-test.bin";

    constexpr std::size_t vocab = 8;
    constexpr std::size_t dim = 4;
    constexpr std::size_t layers = 1;
    constexpr std::size_t heads = 2;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t ffn_dim = 4;

    // 1. Prepare synthetic weights:
    // Embedding: token i has 1.0 at index (i % dim)
    std::vector<float> embd(vocab * dim, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i) {
        embd[i * dim + (i % dim)] = 1.0F;
    }

    std::vector<float> norm(dim, 1.0F);

    // Identity matrices for attention query and output
    std::vector<float> eye_dim(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) eye_dim[i * dim + i] = 1.0F;

    std::vector<float> key_val(kv_heads * head_dim * dim, 0.0F);
    for (std::size_t i = 0; i < kv_heads * head_dim; ++i) key_val[i * dim + i] = 0.5F;

    std::vector<float> ffn_gate_up(ffn_dim * dim, 0.0F);
    for (std::size_t i = 0; i < ffn_dim; ++i) ffn_gate_up[i * dim + (i % dim)] = 0.25F;

    std::vector<float> ffn_down(dim * ffn_dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) ffn_down[i * ffn_dim + (i % ffn_dim)] = 0.5F;

    // LM head: token i has highest dot product with hidden state with 1.0 at (i % dim)
    std::vector<float> lm_head(vocab * dim, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i) {
        // Bias token (i + 1) % vocab to be next token
        const std::size_t target_tok = (i + 1) % vocab;
        lm_head[target_tok * dim + (i % dim)] = 5.0F;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out.is_open());
    const auto write_v = [&](const std::vector<float>& v) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(float)));
    };

    std::uint64_t offset = 0;
    aabhas::model::ModelIndex model;
    model.format = aabhas::model::ModelFormat::GGUF;

    const auto add_t = [&](const std::string& name, const std::vector<float>& data,
                           const std::vector<std::uint64_t>& shape) {
        write_v(data);
        const std::size_t sz = data.size() * sizeof(float);
        model.tensors.push_back({name, offset, sz, aabhas::model::TensorDType::F32, shape});
        offset += sz;
    };

    add_t("token_embd.weight", embd, {vocab, dim});
    add_t("blk.0.attn_norm.weight", norm, {dim});
    add_t("blk.0.attn_q.weight", eye_dim, {dim, dim});
    add_t("blk.0.attn_k.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.0.attn_v.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.0.attn_output.weight", eye_dim, {dim, dim});
    add_t("blk.0.ffn_norm.weight", norm, {dim});
    add_t("blk.0.ffn_gate.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.0.ffn_up.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.0.ffn_down.weight", ffn_down, {dim, ffn_dim});
    add_t("output_norm.weight", norm, {dim});
    add_t("output.weight", lm_head, {vocab, dim});
    out.close();

    model.file_size = offset;

    // Tokenizer setup
    model.tokenizer.vocabulary = {"<unk>", "<bos>", "<eos>", "alpha", "beta", "gamma", "delta", "epsilon"};
    model.tokenizer.bos_token_id = 1;
    model.tokenizer.eos_token_id = 2;
    model.tokenizer.unk_token_id = 0;

    aabhas::inference::InferenceRuntime runtime;
    aabhas::inference::InferenceConfig config;
    config.scheduler.memory_cap_bytes = 4096;
    config.scheduler.chunk_bytes = 128;
    assert(runtime.open(path.string().c_str(), model, config));
    assert(runtime.configure_tokenizer());

    aabhas::inference::TransformerModelConfig model_cfg{};
    model_cfg.vocab_size = vocab;
    model_cfg.model_dimension = dim;
    model_cfg.layer_count = layers;
    model_cfg.block.model_dimension = dim;
    model_cfg.block.head_count = heads;
    model_cfg.block.kv_head_count = kv_heads;
    model_cfg.block.head_dimension = head_dim;
    model_cfg.block.intermediate_dimension = ffn_dim;
    model_cfg.block.rms_epsilon = 1.0e-5F;
    model_cfg.block.rope_theta = 10000.0F;
    model_cfg.bos_token_id = 1;
    model_cfg.eos_token_id = 2;

    assert(runtime.configure_model(model_cfg, 32));

    // Test forward_step for single prompt token
    float logits[vocab] = {0};
    const std::uint32_t prompt_token = 3; // "alpha"
    assert(runtime.generator().forward_step(&prompt_token, 1, 0, logits));
    for (std::size_t i = 0; i < vocab; ++i) {
        assert(std::isfinite(logits[i]));
    }

    // Test generate with prompt tokens [3, 4]
    aabhas::inference::GenerationOptions options;
    options.max_new_tokens = 3;
    options.early_stopping = true;

    const std::vector<std::uint32_t> prompt = {3, 4};
    aabhas::inference::GenerationResult result;
    assert(runtime.generate(prompt, options, result));

    assert(result.prompt_token_count == 2);
    assert(result.generated_token_count <= 3);
    assert(result.all_tokens.size() == result.prompt_token_count + result.generated_token_count);

    // Test text generation with string prompt
    aabhas::inference::GenerationResult text_result;
    assert(runtime.generate_text("alpha", options, text_result));
    assert(!text_result.generated_tokens.empty());
    assert(text_result.prompt_tokens[0] == 1); // <bos> automatically prepended

    // Test early stopping with EOS token
    options.stop_token_ids = {result.generated_tokens[0]};
    aabhas::inference::GenerationResult stop_result;
    assert(runtime.generate(prompt, options, stop_result));
    assert(stop_result.reached_eos);
    assert(stop_result.generated_token_count == 1);

    runtime.close();
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_kv_cache_extensions();
    test_build_model_plan();
    test_end_to_end_token_generation();
    return 0;
}