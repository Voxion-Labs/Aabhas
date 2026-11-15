#include "inference/batch_scheduler.hpp"
#include "inference/inference_runtime.hpp"
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

struct SyntheticModelContext {
    std::filesystem::path path;
    aabhas::model::ModelIndex model_index;
    aabhas::inference::TransformerModelConfig model_config;
};

SyntheticModelContext create_synthetic_model(const std::string& filename) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::filesystem::path path = dir / filename;

    constexpr std::size_t vocab = 8;
    constexpr std::size_t dim = 4;
    constexpr std::size_t layers = 1;
    constexpr std::size_t heads = 1;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t head_dim = 4;
    constexpr std::size_t ffn_dim = 8;

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
    model.tokenizer.vocabulary = {"<unk>", "<bos>", "<eos>", "alpha", "beta", "gamma", "delta", "epsilon"};
    model.tokenizer.bos_token_id = 1;
    model.tokenizer.eos_token_id = 2;
    model.tokenizer.unk_token_id = 0;

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

    return {path, model, model_cfg};
}

void test_batch_scheduler_empty_and_idle() {
    using namespace aabhas::inference;
    BatchSchedulerConfig cfg;
    cfg.max_batch_size = 4;
    BatchScheduler scheduler(cfg);

    assert(scheduler.is_idle());
    assert(scheduler.active_count() == 0);
    assert(scheduler.pending_count() == 0);
    assert(scheduler.completed_count() == 0);

    BatchStepMetrics metrics;
    assert(scheduler.step(metrics));
    assert(metrics.active_sequences == 0);
    assert(metrics.tokens_generated_in_step == 0);

    auto ctx = create_synthetic_model("test_batch_sched_idle.bin");
    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_model(ctx.model_config, 32));
    assert(scheduler.initialize(&runtime, ctx.model_config));

    assert(scheduler.is_idle());
    assert(scheduler.step(metrics));
    assert(metrics.active_sequences == 0);

    runtime.close();
    std::filesystem::remove(ctx.path);
}

void test_batch_scheduler_multi_sequence() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_batch_sched_multi.bin");

    InferenceRuntime runtime;
    InferenceConfig config;
    config.scheduler.memory_cap_bytes = 4096;
    config.scheduler.chunk_bytes = 128;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index, config));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    BatchSchedulerConfig sched_cfg;
    sched_cfg.max_batch_size = 4;
    sched_cfg.max_sequence_length = 32;

    BatchScheduler scheduler(sched_cfg);
    assert(scheduler.initialize(&runtime, ctx.model_config));

    // Submit 3 sequences with different prompt lengths and limits
    BatchRequest req1;
    req1.prompt_tokens = {3}; // "alpha"
    req1.options.max_new_tokens = 2;
    req1.options.early_stopping = true;

    BatchRequest req2;
    req2.prompt_tokens = {3, 4}; // "alpha", "beta"
    req2.options.max_new_tokens = 3;
    req2.options.early_stopping = true;

    BatchRequest req3;
    req3.prompt_tokens = {4, 5, 6}; // "beta", "gamma", "delta"
    req3.options.max_new_tokens = 2;
    req3.options.early_stopping = true;

    const std::uint64_t id1 = scheduler.enqueue(std::move(req1));
    const std::uint64_t id2 = scheduler.enqueue(std::move(req2));
    const std::uint64_t id3 = scheduler.enqueue(std::move(req3));

    assert(id1 > 0 && id2 > 0 && id3 > 0);
    assert(scheduler.pending_count() == 3);
    assert(!scheduler.is_idle());

    BatchStepMetrics metrics;
    std::size_t step_count = 0;
    while (!scheduler.is_idle() && step_count < 20) {
        assert(scheduler.step(metrics));
        step_count++;
    }

    assert(scheduler.is_idle());
    assert(scheduler.completed_count() == 3);

    GenerationResult res1, res2, res3;
    assert(scheduler.get_result(id1, res1));
    assert(scheduler.get_result(id2, res2));
    assert(scheduler.get_result(id3, res3));

    assert(res1.generated_tokens.size() == 2);
    assert(res2.generated_tokens.size() == 3);
    assert(res3.generated_tokens.size() == 2);

    assert(res1.prompt_tokens.size() == 1);
    assert(res2.prompt_tokens.size() == 2);
    assert(res3.prompt_tokens.size() == 3);

    runtime.close();
    std::filesystem::remove(ctx.path);
}

void test_batch_scheduler_early_stopping_and_stop_tokens() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_batch_sched_stop.bin");

    InferenceRuntime runtime;
    InferenceConfig config;
    config.scheduler.memory_cap_bytes = 4096;
    config.scheduler.chunk_bytes = 128;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index, config));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    BatchScheduler scheduler(BatchSchedulerConfig{4, 16, 32});
    assert(scheduler.initialize(&runtime, ctx.model_config));

    // Request A stops on token 5 ("gamma")
    BatchRequest req_a;
    req_a.prompt_tokens = {3}; // "alpha" -> next will be 4 ("beta"), then 5 ("gamma")
    req_a.options.max_new_tokens = 10;
    req_a.options.early_stopping = true;
    req_a.options.stop_token_ids = {5};

    // Request B has no stop token, generates 5 tokens
    BatchRequest req_b;
    req_b.prompt_tokens = {3};
    req_b.options.max_new_tokens = 5;
    req_b.options.early_stopping = true;

    const std::uint64_t id_a = scheduler.enqueue(std::move(req_a));
    const std::uint64_t id_b = scheduler.enqueue(std::move(req_b));

    BatchStepMetrics metrics;
    while (!scheduler.is_idle()) {
        assert(scheduler.step(metrics));
    }

    GenerationResult res_a, res_b;
    assert(scheduler.get_result(id_a, res_a));
    assert(scheduler.get_result(id_b, res_b));

    assert(res_a.reached_eos);
    assert(res_a.generated_tokens.size() == 2); // 4 ("beta"), then 5 ("gamma" -> stop)
    assert(res_a.generated_tokens.back() == 5);

    assert(!res_b.reached_eos);
    assert(res_b.generated_tokens.size() == 5);

    runtime.close();
    std::filesystem::remove(ctx.path);
}

void test_batch_scheduler_continuous_admission() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_batch_sched_continuous.bin");

    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    // Bounded batch size of 1 to test pipelined continuous admission
    BatchScheduler scheduler(BatchSchedulerConfig{1, 16, 32});
    assert(scheduler.initialize(&runtime, ctx.model_config));

    BatchRequest req1;
    req1.prompt_tokens = {3};
    req1.options.max_new_tokens = 2;

    BatchRequest req2;
    req2.prompt_tokens = {4};
    req2.options.max_new_tokens = 2;

    const std::uint64_t id1 = scheduler.enqueue(std::move(req1));
    const std::uint64_t id2 = scheduler.enqueue(std::move(req2));

    // Before step 1: active 0, pending 2
    assert(scheduler.active_count() == 0);
    assert(scheduler.pending_count() == 2);

    // Step 1: req1 admitted (prefill)
    BatchStepMetrics m1;
    assert(scheduler.step(m1));
    assert(scheduler.active_count() == 1);
    assert(scheduler.pending_count() == 1);

    // Step 2: req1 decode step 2 -> completes!
    BatchStepMetrics m2;
    assert(scheduler.step(m2));
    assert(m2.completed_sequences == 1);

    // Step 3: req1 was retired, req2 is immediately admitted (prefill)!
    BatchStepMetrics m3;
    assert(scheduler.step(m3));
    assert(m3.prefilling_sequences == 1);
    assert(scheduler.active_count() == 1);
    assert(scheduler.pending_count() == 0);

    // Step 4: req2 completes!
    BatchStepMetrics m4;
    assert(scheduler.step(m4));
    assert(m4.completed_sequences == 1);
    assert(scheduler.is_idle());

    GenerationResult r1, r2;
    assert(scheduler.get_result(id1, r1));
    assert(scheduler.get_result(id2, r2));
    assert(r1.generated_tokens.size() == 2);
    assert(r2.generated_tokens.size() == 2);

    runtime.close();
    std::filesystem::remove(ctx.path);
}

void test_batch_scheduler_streaming() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_batch_sched_stream.bin");

    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    BatchScheduler scheduler(BatchSchedulerConfig{2, 16, 32});
    assert(scheduler.initialize(&runtime, ctx.model_config));

    std::vector<std::uint32_t> streamed_tokens;
    std::vector<std::string> streamed_text;

    BatchRequest req;
    req.prompt_tokens = {3}; // "alpha"
    req.options.max_new_tokens = 3;
    req.options.early_stopping = true;
    req.stream_callback = [&](const StreamChunk& chunk) -> bool {
        streamed_tokens.push_back(chunk.token_id);
        if (!chunk.text_delta.empty()) {
            streamed_text.push_back(chunk.text_delta);
        }
        return true;
    };

    const std::uint64_t id = scheduler.enqueue(std::move(req));
    BatchStepMetrics metrics;
    while (!scheduler.is_idle()) {
        assert(scheduler.step(metrics));
    }

    GenerationResult result;
    assert(scheduler.get_result(id, result));
    assert(result.generated_tokens.size() == 3);
    assert(streamed_tokens.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        assert(streamed_tokens[i] == result.generated_tokens[i]);
    }
    assert(!streamed_text.empty());

    runtime.close();
    std::filesystem::remove(ctx.path);
}

void test_runtime_generate_batch_api() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_runtime_batch_api.bin");

    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    std::vector<std::vector<std::uint32_t>> prompts = {{3}, {3, 4}, {4}};
    std::vector<GenerationOptions> opts(3);
    opts[0].max_new_tokens = 2;
    opts[1].max_new_tokens = 3;
    opts[2].max_new_tokens = 1;

    std::vector<GenerationResult> batch_results;
    assert(runtime.generate_batch(prompts, opts, batch_results));

    assert(batch_results.size() == 3);
    assert(batch_results[0].generated_tokens.size() == 2);
    assert(batch_results[1].generated_tokens.size() == 3);
    assert(batch_results[2].generated_tokens.size() == 1);

    runtime.close();
    std::filesystem::remove(ctx.path);
}

} // namespace

int main() {
    test_batch_scheduler_empty_and_idle();
    test_batch_scheduler_multi_sequence();
    test_batch_scheduler_early_stopping_and_stop_tokens();
    test_batch_scheduler_continuous_admission();
    test_batch_scheduler_streaming();
    test_runtime_generate_batch_api();
    return 0;
}
