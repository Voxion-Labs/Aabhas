#ifndef AABHAS_INFERENCE_BENCHMARK_ENGINE_HPP
#define AABHAS_INFERENCE_BENCHMARK_ENGINE_HPP

#include "hardware/hw_compatibility.hpp"
#include "inference/batch_scheduler.hpp"
#include "inference/inference_runtime.hpp"
#include "inference/quantized_kernels.hpp"
#include "model/model_index.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aabhas::inference {

/** Quantized kernel micro-benchmark result for a single dtype and matrix dimension. */
struct KernelBenchmarkResult {
    aabhas::model::TensorDType dtype{aabhas::model::TensorDType::Unknown};
    std::string dtype_name;
    std::size_t dimension{4096};
    std::size_t iterations{100};
    double total_time_ms{0.0};
    double latency_us{0.0};
    double gflops{0.0};
    double gigabytes_per_sec{0.0};
};

/** Memory bandwidth micro-benchmark result. */
struct MemoryBenchmarkResult {
    std::size_t buffer_bytes{64ULL * 1024ULL * 1024ULL}; // 64 MiB
    std::size_t iterations{20};
    double sequential_read_gbps{0.0};
    double sequential_write_gbps{0.0};
    double copy_bandwidth_gbps{0.0};
};

/** Prefill latency and prompt-processing throughput statistics. */
struct PrefillBenchmarkResult {
    std::size_t prompt_tokens{128};
    std::size_t iterations{10};
    double ttft_mean_ms{0.0};
    double ttft_min_ms{0.0};
    double ttft_max_ms{0.0};
    double ttft_p50_ms{0.0};
    double ttft_p90_ms{0.0};
    double ttft_p99_ms{0.0};
    double tokens_per_second{0.0};
    std::vector<double> run_latencies_ms;
};

/** Autoregressive decode token latency and throughput statistics. */
struct DecodeBenchmarkResult {
    std::size_t generated_tokens{32};
    std::size_t iterations{5};
    double tpot_mean_ms{0.0};
    double tpot_min_ms{0.0};
    double tpot_max_ms{0.0};
    double tpot_p50_ms{0.0};
    double tpot_p90_ms{0.0};
    double tpot_p99_ms{0.0};
    double tokens_per_second{0.0};
    std::vector<double> inter_token_latencies_ms;
};

/** Concurrent batch scaling benchmark metrics. */
struct BatchScalePoint {
    std::size_t batch_size{1};
    double total_time_ms{0.0};
    double aggregate_tokens_per_second{0.0};
    double speedup_ratio{1.0};
    double efficiency_percent{100.0};
};

struct BatchScalingBenchmarkResult {
    std::vector<BatchScalePoint> points;
};

/** Complete performance and hardware benchmark report. */
struct BenchmarkSuiteReport {
    aabhas::hardware::HardwareProfile hardware;
    std::vector<KernelBenchmarkResult> kernels;
    MemoryBenchmarkResult memory;
    PrefillBenchmarkResult prefill;
    DecodeBenchmarkResult decode;
    BatchScalingBenchmarkResult batching;

    std::string to_json() const;
    std::string summary_table() const;
};

/**
 * High-performance inference and micro-kernel benchmarking engine.
 */
class BenchmarkEngine final {
public:
    BenchmarkEngine() = default;
    ~BenchmarkEngine() = default;

    /** Run standalone quantized kernel micro-benchmarks across Q4_0, Q4_1, Q8_0, FP16, and FP32. */
    static std::vector<KernelBenchmarkResult> benchmark_kernels(
        std::size_t dimension = 4096,
        std::size_t iterations = 50);

    /** Run host memory copy and streaming bandwidth benchmark. */
    static MemoryBenchmarkResult benchmark_memory(
        std::size_t buffer_bytes = 32ULL * 1024ULL * 1024ULL,
        std::size_t iterations = 20);

    /** Benchmark prefill (TTFT) on an initialized runtime with given prompt length. */
    static PrefillBenchmarkResult benchmark_prefill(
        InferenceRuntime& runtime,
        std::size_t prompt_tokens = 64,
        std::size_t iterations = 5);

    /** Benchmark autoregressive decode (TPOT) on an initialized runtime. */
    static DecodeBenchmarkResult benchmark_decode(
        InferenceRuntime& runtime,
        std::size_t max_new_tokens = 32,
        std::size_t iterations = 3);

    /** Benchmark continuous batch scaling across batch sizes 1, 2, 4. */
    static BatchScalingBenchmarkResult benchmark_batching(
        InferenceRuntime& runtime,
        const TransformerModelConfig& config,
        std::size_t prompt_tokens = 16,
        std::size_t tokens_to_generate = 8);

    /** Run comprehensive benchmark suite combining hardware profile, kernels, memory, and inference. */
    static BenchmarkSuiteReport run_full_suite(
        InferenceRuntime* runtime = nullptr,
        const TransformerModelConfig* model_config = nullptr);

    /** Helper: Calculate nearest-rank percentile from a set of numeric measurements. */
    static double calculate_percentile(std::vector<double> samples, double percentile);
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_BENCHMARK_ENGINE_HPP
