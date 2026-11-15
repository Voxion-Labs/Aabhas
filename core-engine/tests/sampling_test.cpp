#include "inference/inference_runtime.hpp"
#include "inference/sampler.hpp"
#include "inference/token_generator.hpp"
#include "model/model_index.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool near(float a, float b, float tol = 1.0e-4F) {
    return std::fabs(a - b) <= tol;
}

void test_greedy_argmax() {
    float logits[5] = {1.0F, 5.5F, -2.0F, 3.2F, 0.0F};
    assert(aabhas::inference::Sampler::argmax(logits, 5) == 1);

    logits[4] = 10.0F;
    assert(aabhas::inference::Sampler::argmax(logits, 5) == 4);

    assert(aabhas::inference::Sampler::argmax(nullptr, 0) == 0);
}

void test_penalties() {
    std::vector<float> logits = {2.0F, -2.0F, 1.0F};
    const std::vector<std::uint32_t> history = {0, 1, 0};

    // Repetition penalty: 2.0 -> 2.0 / 2.0 = 1.0, -2.0 -> -2.0 * 2.0 = -4.0
    aabhas::inference::Sampler::apply_penalties(logits.data(), logits.size(), history,
                                               2.0F, 0.0F, 0.0F);
    assert(near(logits[0], 0.5F)); // token 0 appeared twice in history, penalized twice
    assert(near(logits[1], -4.0F));
    assert(near(logits[2], 1.0F));  // unpenalized
}

void test_softmax_stability() {
    std::vector<float> logits = {1000.0F, 1001.0F, 999.0F};
    aabhas::inference::Sampler::softmax(logits.data(), logits.size());

    float sum = 0.0F;
    for (const float p : logits) {
        assert(std::isfinite(p));
        assert(p >= 0.0F && p <= 1.0F);
        sum += p;
    }
    assert(near(sum, 1.0F));
    assert(logits[1] > logits[0]);
    assert(logits[0] > logits[2]);
}

void test_top_k_filtering() {
    aabhas::inference::Sampler sampler(42);
    assert(sampler.configure(4));

    aabhas::inference::SamplerConfig cfg;
    cfg.temperature = 1.0F;
    cfg.top_k = 2; // only tokens 1 and 3 should be eligible

    for (std::size_t trial = 0; trial < 50; ++trial) {
        std::vector<float> logits = {0.0F, 10.0F, 1.0F, 8.0F};
        const std::vector<std::uint32_t> history;
        const std::uint32_t chosen = sampler.sample(logits.data(), logits.size(), history, cfg);
        assert(chosen == 1 || chosen == 3);
    }
}

void test_top_p_nucleus() {
    aabhas::inference::Sampler sampler(101);
    assert(sampler.configure(4));

    aabhas::inference::SamplerConfig cfg;
    cfg.temperature = 1.0F;
    cfg.top_p = 0.95F;

    for (std::size_t trial = 0; trial < 50; ++trial) {
        // Token 0 dominates overwhelmingly (>99.9% mass)
        std::vector<float> logits = {20.0F, 0.0F, 0.0F, 0.0F};
        const std::vector<std::uint32_t> history;
        const std::uint32_t chosen = sampler.sample(logits.data(), logits.size(), history, cfg);
        assert(chosen == 0);
    }
}

void test_min_p_filtering() {
    aabhas::inference::Sampler sampler(202);
    assert(sampler.configure(3));

    aabhas::inference::SamplerConfig cfg;
    cfg.temperature = 1.0F;
    cfg.min_p = 0.5F; // Must be at least 50% of top candidate prob

    for (std::size_t trial = 0; trial < 50; ++trial) {
        // Token 0 is highest, Token 1 is close, Token 2 is tiny
        std::vector<float> logits = {5.0F, 4.8F, -10.0F};
        const std::vector<std::uint32_t> history;
        const std::uint32_t chosen = sampler.sample(logits.data(), logits.size(), history, cfg);
        assert(chosen == 0 || chosen == 1);
        assert(chosen != 2);
    }
}

void test_prng_reproducibility() {
    aabhas::inference::Sampler s1(777);
    aabhas::inference::Sampler s2(777);
    assert(s1.configure(8));
    assert(s2.configure(8));

    aabhas::inference::SamplerConfig cfg;
    cfg.temperature = 0.8F;
    cfg.top_k = 4;

    const std::vector<std::uint32_t> history;
    for (std::size_t i = 0; i < 20; ++i) {
        std::vector<float> l1 = {1.0F, 2.0F, 3.0F, 4.0F, 1.5F, 2.5F, 3.5F, 0.5F};
        std::vector<float> l2 = l1;
        const std::uint32_t tok1 = s1.sample(l1.data(), l1.size(), history, cfg);
        const std::uint32_t tok2 = s2.sample(l2.data(), l2.size(), history, cfg);
        assert(tok1 == tok2);
    }
}

void test_streaming_token_generation() {
    const auto path =
        std::filesystem::temp_directory_path() / "aabhas-streaming-test.bin";

    constexpr std::size_t vocab = 8;
    constexpr std::size_t dim = 4;
    constexpr std::size_t layers = 1;
    constexpr std::size_t heads = 2;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t ffn_dim = 4;

    std::vector<float> embd(vocab * dim, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i) {
        embd[i * dim + (i % dim)] = 1.0F;
    }
    std::vector<float> norm(dim, 1.0F);
    std::vector<float> eye_dim(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) eye_dim[i * dim + i] = 1.0F;

    std::vector<float> key_val(kv_heads * head_dim * dim, 0.0F);
    for (std::size_t i = 0; i < kv_heads * head_dim; ++i) key_val[i * dim + i] = 0.5F;

    std::vector<float> ffn_gate_up(ffn_dim * dim, 0.0F);
    for (std::size_t i = 0; i < ffn_dim; ++i) ffn_gate_up[i * dim + (i % dim)] = 0.25F;

    std::vector<float> ffn_down(dim * ffn_dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) ffn_down[i * ffn_dim + (i % ffn_dim)] = 0.5F;

    std::vector<float> lm_head(vocab * dim, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i) {
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
    model.tokenizer.vocabulary = {"<unk>", "<bos>", "<eos>", "foo", "bar", "baz", "qux", "quux"};
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

    // Test 1: Full streaming response assembly
    aabhas::inference::GenerationOptions options;
    options.max_new_tokens = 4;
    options.early_stopping = false;
    options.sampler.temperature = 0.0F; // deterministic greedy

    std::string streamed_concatenation;
    std::vector<std::uint32_t> streamed_tokens;

    const auto callback = [&](const aabhas::inference::StreamChunk& chunk) -> bool {
        streamed_tokens.push_back(chunk.token_id);
        streamed_concatenation += chunk.text_delta;
        return true;
    };

    aabhas::inference::GenerationResult stream_result;
    const std::vector<std::uint32_t> prompt = {3}; // "foo"
    assert(runtime.generate(prompt, options, stream_result, callback));

    assert(streamed_tokens.size() == stream_result.generated_tokens.size());
    assert(streamed_concatenation == stream_result.text);

    // Test 2: Early stream cancellation by client callback returning false
    std::size_t received_chunks = 0;
    const auto abort_callback = [&](const aabhas::inference::StreamChunk&) -> bool {
        ++received_chunks;
        return false; // abort immediately after first token
    };

    aabhas::inference::GenerationResult abort_result;
    assert(runtime.generate(prompt, options, abort_result, abort_callback));
    assert(received_chunks == 1);
    assert(abort_result.generated_token_count == 1);

    runtime.close();
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_greedy_argmax();
    test_penalties();
    test_softmax_stability();
    test_top_k_filtering();
    test_top_p_nucleus();
    test_min_p_filtering();
    test_prng_reproducibility();
    test_streaming_token_generation();
    return 0;
}
