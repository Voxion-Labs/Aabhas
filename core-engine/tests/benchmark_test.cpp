#include "inference/benchmark_engine.hpp"
#include "inference/inference_runtime.hpp"
#include "inference/token_generator.hpp"
#include "model/model_index.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    model_cfg.bos_token_id = 1;
    model_cfg.eos_token_id = 2;

    return {path, model, model_cfg};
}

void test_percentile_calculation() {
    using namespace aabhas::inference;

    assert(BenchmarkEngine::calculate_percentile({}, 50.0) == 0.0);
    assert(BenchmarkEngine::calculate_percentile({42.0}, 50.0) == 42.0);

    std::vector<double> samples;
    for (int i = 1; i <= 100; ++i) {
        samples.push_back(static_cast<double>(i));
    }

    const double p0 = BenchmarkEngine::calculate_percentile(samples, 0.0);
    const double p50 = BenchmarkEngine::calculate_percentile(samples, 50.0);
    const double p90 = BenchmarkEngine::calculate_percentile(samples, 90.0);
    const double p100 = BenchmarkEngine::calculate_percentile(samples, 100.0);

    assert(p0 == 1.0);
    assert(std::fabs(p50 - 50.5) < 1e-4);
    assert(std::fabs(p90 - 90.1) < 1e-4);
    assert(p100 == 100.0);

    std::cout << "[PASS] test_percentile_calculation\n";
}

void test_kernel_benchmarks() {
    using namespace aabhas::inference;

    auto results = BenchmarkEngine::benchmark_kernels(1024, 20);
    assert(results.size() >= 5); // FP32, FP16, Q4_0, Q4_1, Q8_0

    for (const auto& res : results) {
        assert(!res.dtype_name.empty());
        assert(res.dimension == 1024);
        assert(res.iterations == 20);
        assert(res.total_time_ms > 0.0);
        assert(res.latency_us > 0.0);
        assert(res.gflops > 0.0);
        assert(res.gigabytes_per_sec > 0.0);
    }

    std::cout << "[PASS] test_kernel_benchmarks: evaluated " << results.size() << " kernel formats\n";
}

void test_memory_benchmarks() {
    using namespace aabhas::inference;

    auto res = BenchmarkEngine::benchmark_memory(4ULL * 1024ULL * 1024ULL, 10);
    assert(res.buffer_bytes == 4ULL * 1024ULL * 1024ULL);
    assert(res.iterations == 10);
    assert(res.copy_bandwidth_gbps > 0.0);
    assert(res.sequential_read_gbps > 0.0);
    assert(res.sequential_write_gbps > 0.0);

    std::cout << "[PASS] test_memory_benchmarks: copy=" << res.copy_bandwidth_gbps
              << " GB/s, read=" << res.sequential_read_gbps
              << " GB/s, write=" << res.sequential_write_gbps << " GB/s\n";
}

void test_inference_prefill_and_decode_benchmarks() {
    using namespace aabhas::inference;

    auto ctx = create_synthetic_model("test_bench_synthetic.bin");
    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    // Prefill benchmark
    auto prefill_res = BenchmarkEngine::benchmark_prefill(runtime, 8, 3);
    assert(prefill_res.prompt_tokens == 8);
    assert(prefill_res.iterations == 3);
    assert(prefill_res.ttft_mean_ms > 0.0);
    assert(prefill_res.tokens_per_second > 0.0);
    assert(prefill_res.run_latencies_ms.size() == 3);

    // Decode benchmark
    auto decode_res = BenchmarkEngine::benchmark_decode(runtime, 4, 2);
    assert(decode_res.generated_tokens == 4);
    assert(decode_res.iterations == 2);
    assert(decode_res.tpot_mean_ms > 0.0);
    assert(decode_res.tokens_per_second > 0.0);
    assert(decode_res.inter_token_latencies_ms.size() == 8); // 2 runs * 4 tokens

    runtime.close();
    std::filesystem::remove(ctx.path);

    std::cout << "[PASS] test_inference_prefill_and_decode_benchmarks: TTFT="
              << prefill_res.ttft_mean_ms << " ms, TPOT=" << decode_res.tpot_mean_ms << " ms\n";
}

void test_batch_scaling_benchmark() {
    using namespace aabhas::inference;

    auto ctx = create_synthetic_model("test_bench_scaling.bin");
    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    auto batch_res = BenchmarkEngine::benchmark_batching(runtime, ctx.model_config, 4, 2);
    assert(batch_res.points.size() == 3); // Batch 1, 2, 4
    assert(batch_res.points[0].batch_size == 1);
    assert(batch_res.points[0].speedup_ratio == 1.0);
    assert(batch_res.points[1].batch_size == 2);
    assert(batch_res.points[2].batch_size == 4);

    runtime.close();
    std::filesystem::remove(ctx.path);

    std::cout << "[PASS] test_batch_scaling_benchmark\n";
}

void test_full_suite_report() {
    using namespace aabhas::inference;

    auto ctx = create_synthetic_model("test_bench_full_suite.bin");
    InferenceRuntime runtime;
    assert(runtime.open(ctx.path.string().c_str(), ctx.model_index));
    assert(runtime.configure_tokenizer());
    assert(runtime.configure_model(ctx.model_config, 32));

    BenchmarkSuiteReport report = BenchmarkEngine::run_full_suite(&runtime, &ctx.model_config);
    assert(!report.hardware.host.system.empty());
    assert(!report.kernels.empty());
    assert(report.memory.copy_bandwidth_gbps > 0.0);
    assert(report.prefill.tokens_per_second > 0.0);
    assert(report.decode.tokens_per_second > 0.0);

    std::string json = report.to_json();
    assert(json.find("\"hardware\"") != std::string::npos);
    assert(json.find("\"quantized_kernels\"") != std::string::npos);
    assert(json.find("\"prefill\"") != std::string::npos);
    assert(json.find("\"decode\"") != std::string::npos);

    std::string table = report.summary_table();
    assert(table.find("AABHAS HARDWARE & INFERENCE BENCHMARK") != std::string::npos);
    assert(table.find("QUANTIZED KERNEL PERFORMANCE") != std::string::npos);

    runtime.close();
    std::filesystem::remove(ctx.path);

    std::cout << "[PASS] test_full_suite_report\n";
}

} // namespace

int main() {
    test_percentile_calculation();
    test_kernel_benchmarks();
    test_memory_benchmarks();
    test_inference_prefill_and_decode_benchmarks();
    test_batch_scaling_benchmark();
    test_full_suite_report();
    std::cout << "All Benchmark Engine tests passed successfully.\n";
    return 0;
}
