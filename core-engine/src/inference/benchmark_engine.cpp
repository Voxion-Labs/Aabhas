#include "inference/benchmark_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

namespace aabhas::inference {

double BenchmarkEngine::calculate_percentile(std::vector<double> samples, double percentile) {
    if (samples.empty()) return 0.0;
    if (samples.size() == 1) return samples.front();
    std::sort(samples.begin(), samples.end());

    if (percentile <= 0.0) return samples.front();
    if (percentile >= 100.0) return samples.back();

    const double rank = (percentile / 100.0) * static_cast<double>(samples.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    const double weight = rank - static_cast<double>(lower);

    return samples[lower] * (1.0 - weight) + samples[upper] * weight;
}

std::vector<KernelBenchmarkResult> BenchmarkEngine::benchmark_kernels(
    std::size_t dimension,
    std::size_t iterations) {

    std::vector<KernelBenchmarkResult> results;
    const std::size_t dim = (dimension / 32) * 32; // Align to 32
    if (dim == 0 || iterations == 0) return results;

    std::vector<float> vec_a(dim, 0.25F);
    std::vector<float> vec_b(dim, 0.75F);

    // 1. FP32 Dot Product
    {
        KernelBenchmarkResult res{};
        res.dtype = aabhas::model::TensorDType::F32;
        res.dtype_name = "FP32";
        res.dimension = dim;
        res.iterations = iterations;

        // Warm up
        float dummy = 0.0F;
        float val = 0.0F;
        for (std::size_t i = 0; i < 5; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::F32, vec_a.data(), vec_b.data(), dim, val);
            dummy += val;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::F32, vec_a.data(), vec_b.data(), dim, val);
            dummy += val;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Prevent optimizer from discarding dummy
        if (dummy == 123456.0F) vec_a[0] = dummy;

        res.total_time_ms = duration_ms;
        res.latency_us = (duration_ms * 1000.0) / static_cast<double>(iterations);
        const double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(iterations);
        res.gflops = (flops / 1e9) / (duration_ms / 1000.0);
        const double bytes = 2.0 * static_cast<double>(dim * sizeof(float)) * static_cast<double>(iterations);
        res.gigabytes_per_sec = (bytes / 1e9) / (duration_ms / 1000.0);
        results.push_back(res);
    }

    // 2. FP16 Dot Product
    {
        KernelBenchmarkResult res{};
        res.dtype = aabhas::model::TensorDType::F16;
        res.dtype_name = "FP16";
        res.dimension = dim;
        res.iterations = iterations;

        std::vector<std::uint16_t> fp16_a(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            fp16_a[i] = kernels::fp32_to_fp16(vec_a[i]);
        }

        float dummy = 0.0F;
        float val = 0.0F;
        for (std::size_t i = 0; i < 5; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::F16, fp16_a.data(), vec_b.data(), dim, val);
            dummy += val;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::F16, fp16_a.data(), vec_b.data(), dim, val);
            dummy += val;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (dummy == 123456.0F) vec_a[0] = dummy;

        res.total_time_ms = duration_ms;
        res.latency_us = (duration_ms * 1000.0) / static_cast<double>(iterations);
        const double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(iterations);
        res.gflops = (flops / 1e9) / (duration_ms / 1000.0);
        const double bytes = static_cast<double>(dim * sizeof(std::uint16_t) + dim * sizeof(float)) * static_cast<double>(iterations);
        res.gigabytes_per_sec = (bytes / 1e9) / (duration_ms / 1000.0);
        results.push_back(res);
    }

    // 3. Q4_0 Matrix-Vector Kernel
    {
        KernelBenchmarkResult res{};
        res.dtype = aabhas::model::TensorDType::Q4_0;
        res.dtype_name = "Q4_0";
        res.dimension = dim;
        res.iterations = iterations;

        const std::size_t num_blocks = dim / 32;
        std::vector<kernels::BlockQ4_0> blocks(num_blocks);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            blocks[b].d = kernels::fp32_to_fp16(0.05F);
            std::memset(blocks[b].qs, 0x44, sizeof(blocks[b].qs));
        }

        float dummy = 0.0F;
        float val = 0.0F;
        for (std::size_t i = 0; i < 5; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::Q4_0, blocks.data(), vec_b.data(), dim, val);
            dummy += val;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::Q4_0, blocks.data(), vec_b.data(), dim, val);
            dummy += val;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (dummy == 123456.0F) vec_a[0] = dummy;

        res.total_time_ms = duration_ms;
        res.latency_us = (duration_ms * 1000.0) / static_cast<double>(iterations);
        const double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(iterations);
        res.gflops = (flops / 1e9) / (duration_ms / 1000.0);
        const double bytes = static_cast<double>(num_blocks * sizeof(kernels::BlockQ4_0) + dim * sizeof(float)) * static_cast<double>(iterations);
        res.gigabytes_per_sec = (bytes / 1e9) / (duration_ms / 1000.0);
        results.push_back(res);
    }

    // 4. Q4_1 Matrix-Vector Kernel
    {
        KernelBenchmarkResult res{};
        res.dtype = aabhas::model::TensorDType::Q4_1;
        res.dtype_name = "Q4_1";
        res.dimension = dim;
        res.iterations = iterations;

        const std::size_t num_blocks = dim / 32;
        std::vector<kernels::BlockQ4_1> blocks(num_blocks);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            blocks[b].d = kernels::fp32_to_fp16(0.05F);
            blocks[b].m = kernels::fp32_to_fp16(-0.02F);
            std::memset(blocks[b].qs, 0x55, sizeof(blocks[b].qs));
        }

        float dummy = 0.0F;
        float val = 0.0F;
        for (std::size_t i = 0; i < 5; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::Q4_1, blocks.data(), vec_b.data(), dim, val);
            dummy += val;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::Q4_1, blocks.data(), vec_b.data(), dim, val);
            dummy += val;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (dummy == 123456.0F) vec_a[0] = dummy;

        res.total_time_ms = duration_ms;
        res.latency_us = (duration_ms * 1000.0) / static_cast<double>(iterations);
        const double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(iterations);
        res.gflops = (flops / 1e9) / (duration_ms / 1000.0);
        const double bytes = static_cast<double>(num_blocks * sizeof(kernels::BlockQ4_1) + dim * sizeof(float)) * static_cast<double>(iterations);
        res.gigabytes_per_sec = (bytes / 1e9) / (duration_ms / 1000.0);
        results.push_back(res);
    }

    // 5. Q8_0 Matrix-Vector Kernel
    {
        KernelBenchmarkResult res{};
        res.dtype = aabhas::model::TensorDType::Q8_0;
        res.dtype_name = "Q8_0";
        res.dimension = dim;
        res.iterations = iterations;

        const std::size_t num_blocks = dim / 32;
        std::vector<kernels::BlockQ8_0> blocks(num_blocks);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            blocks[b].d = kernels::fp32_to_fp16(0.02F);
            std::memset(blocks[b].qs, 2, sizeof(blocks[b].qs));
        }

        float dummy = 0.0F;
        float val = 0.0F;
        for (std::size_t i = 0; i < 5; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::Q8_0, blocks.data(), vec_b.data(), dim, val);
            dummy += val;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            kernels::vec_dot(aabhas::model::TensorDType::Q8_0, blocks.data(), vec_b.data(), dim, val);
            dummy += val;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (dummy == 123456.0F) vec_a[0] = dummy;

        res.total_time_ms = duration_ms;
        res.latency_us = (duration_ms * 1000.0) / static_cast<double>(iterations);
        const double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(iterations);
        res.gflops = (flops / 1e9) / (duration_ms / 1000.0);
        const double bytes = static_cast<double>(num_blocks * sizeof(kernels::BlockQ8_0) + dim * sizeof(float)) * static_cast<double>(iterations);
        res.gigabytes_per_sec = (bytes / 1e9) / (duration_ms / 1000.0);
        results.push_back(res);
    }

    return results;
}

MemoryBenchmarkResult BenchmarkEngine::benchmark_memory(
    std::size_t buffer_bytes,
    std::size_t iterations) {

    MemoryBenchmarkResult res{};
    const std::size_t bytes = (buffer_bytes > 0) ? buffer_bytes : (32ULL * 1024ULL * 1024ULL);
    const std::size_t iters = (iterations > 0) ? iterations : 20;

    res.buffer_bytes = bytes;
    res.iterations = iters;

    std::vector<std::uint8_t> src(bytes, 0xAA);
    std::vector<std::uint8_t> dst(bytes, 0x00);

    // Warmup
    std::memcpy(dst.data(), src.data(), bytes);

    // Copy Bandwidth
    {
        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iters; ++i) {
            std::memcpy(dst.data(), src.data(), bytes);
            // Invert to avoid compiler elision
            dst[0] ^= 0x01;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_s = std::chrono::duration<double>(end - start).count();
        // 2x bytes per copy (read src + write dst)
        res.copy_bandwidth_gbps = (2.0 * static_cast<double>(bytes) * static_cast<double>(iters) / 1e9) / std::max(duration_s, 1e-6);
    }

    // Sequential Read Bandwidth
    {
        std::uint64_t sum = 0;
        const auto* ptr = reinterpret_cast<const std::uint64_t*>(src.data());
        const std::size_t words = bytes / sizeof(std::uint64_t);

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iters; ++i) {
            for (std::size_t w = 0; w < words; ++w) {
                sum += ptr[w];
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_s = std::chrono::duration<double>(end - start).count();
        if (sum == 12345678ULL) dst[0] = static_cast<std::uint8_t>(sum);

        res.sequential_read_gbps = (static_cast<double>(bytes) * static_cast<double>(iters) / 1e9) / std::max(duration_s, 1e-6);
    }

    // Sequential Write Bandwidth
    {
        auto* ptr = reinterpret_cast<std::uint64_t*>(dst.data());
        const std::size_t words = bytes / sizeof(std::uint64_t);

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iters; ++i) {
            for (std::size_t w = 0; w < words; ++w) {
                ptr[w] = static_cast<std::uint64_t>(w + i);
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_s = std::chrono::duration<double>(end - start).count();

        res.sequential_write_gbps = (static_cast<double>(bytes) * static_cast<double>(iters) / 1e9) / std::max(duration_s, 1e-6);
    }

    return res;
}

PrefillBenchmarkResult BenchmarkEngine::benchmark_prefill(
    InferenceRuntime& runtime,
    std::size_t prompt_tokens,
    std::size_t iterations) {

    PrefillBenchmarkResult result{};
    const std::size_t p_tokens = (prompt_tokens > 0) ? prompt_tokens : 32;
    const std::size_t iters = (iterations > 0) ? iterations : 5;
    result.prompt_tokens = p_tokens;
    result.iterations = iters;

    std::vector<std::uint32_t> prompt(p_tokens, 3U);
    GenerationOptions options{};
    options.max_new_tokens = 1;
    options.early_stopping = false;

    // Warm up
    GenerationResult warm_res{};
    runtime.generate(prompt, options, warm_res);

    result.run_latencies_ms.reserve(iters);
    double total_ms = 0.0;

    for (std::size_t i = 0; i < iters; ++i) {
        GenerationResult run_res{};
        const auto start = std::chrono::high_resolution_clock::now();
        runtime.generate(prompt, options, run_res);
        const auto end = std::chrono::high_resolution_clock::now();
        const double latency = std::chrono::duration<double, std::milli>(end - start).count();
        result.run_latencies_ms.push_back(latency);
        total_ms += latency;
    }

    if (!result.run_latencies_ms.empty()) {
        std::vector<double> sorted = result.run_latencies_ms;
        std::sort(sorted.begin(), sorted.end());
        result.ttft_min_ms = sorted.front();
        result.ttft_max_ms = sorted.back();
        result.ttft_mean_ms = total_ms / static_cast<double>(iters);
        result.ttft_p50_ms = calculate_percentile(sorted, 50.0);
        result.ttft_p90_ms = calculate_percentile(sorted, 90.0);
        result.ttft_p99_ms = calculate_percentile(sorted, 99.0);
        const double total_s = total_ms / 1000.0;
        result.tokens_per_second = (static_cast<double>(p_tokens * iters)) / std::max(total_s, 1e-6);
    }

    return result;
}

DecodeBenchmarkResult BenchmarkEngine::benchmark_decode(
    InferenceRuntime& runtime,
    std::size_t max_new_tokens,
    std::size_t iterations) {

    DecodeBenchmarkResult result{};
    const std::size_t gen_tokens = (max_new_tokens > 0) ? max_new_tokens : 16;
    const std::size_t iters = (iterations > 0) ? iterations : 3;
    result.generated_tokens = gen_tokens;
    result.iterations = iters;

    std::vector<std::uint32_t> prompt = {3U};
    GenerationOptions options{};
    options.max_new_tokens = gen_tokens;
    options.early_stopping = false;

    // Warm up
    GenerationResult warm_res{};
    runtime.generate(prompt, options, warm_res);

    result.inter_token_latencies_ms.reserve(iters * gen_tokens);
    double total_ms = 0.0;

    for (std::size_t i = 0; i < iters; ++i) {
        std::vector<double> step_times;
        step_times.reserve(gen_tokens);
        auto last_time = std::chrono::high_resolution_clock::now();

        StreamCallback stream_callback = [&](const StreamChunk&) -> bool {
            const auto now = std::chrono::high_resolution_clock::now();
            const double delta = std::chrono::duration<double, std::milli>(now - last_time).count();
            step_times.push_back(delta);
            last_time = now;
            return true;
        };

        GenerationResult run_res{};
        const auto run_start = std::chrono::high_resolution_clock::now();
        last_time = run_start;
        runtime.generate(prompt, options, run_res, stream_callback);
        const auto run_end = std::chrono::high_resolution_clock::now();
        const double duration = std::chrono::duration<double, std::milli>(run_end - run_start).count();
        total_ms += duration;

        for (const double st : step_times) {
            result.inter_token_latencies_ms.push_back(st);
        }
    }

    if (!result.inter_token_latencies_ms.empty()) {
        std::vector<double> sorted = result.inter_token_latencies_ms;
        std::sort(sorted.begin(), sorted.end());
        result.tpot_min_ms = sorted.front();
        result.tpot_max_ms = sorted.back();
        result.tpot_mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
        result.tpot_p50_ms = calculate_percentile(sorted, 50.0);
        result.tpot_p90_ms = calculate_percentile(sorted, 90.0);
        result.tpot_p99_ms = calculate_percentile(sorted, 99.0);
        const double total_s = total_ms / 1000.0;
        result.tokens_per_second = static_cast<double>(result.inter_token_latencies_ms.size()) / std::max(total_s, 1e-6);
    }

    return result;
}

BatchScalingBenchmarkResult BenchmarkEngine::benchmark_batching(
    InferenceRuntime& runtime,
    const TransformerModelConfig& config,
    std::size_t prompt_tokens,
    std::size_t tokens_to_generate) {

    BatchScalingBenchmarkResult result;
    const std::vector<std::size_t> batch_sizes = {1, 2, 4};
    const std::size_t p_toks = (prompt_tokens > 0) ? prompt_tokens : 8;
    const std::size_t gen_toks = (tokens_to_generate > 0) ? tokens_to_generate : 4;

    double baseline_throughput = 0.0;

    for (const std::size_t b_size : batch_sizes) {
        BatchScheduler scheduler(BatchSchedulerConfig{b_size, 32, 64});
        if (!scheduler.initialize(&runtime, config)) {
            continue;
        }

        std::vector<std::vector<std::uint32_t>> prompts(b_size, std::vector<std::uint32_t>(p_toks, 3U));
        std::vector<GenerationOptions> opts(b_size);
        for (auto& o : opts) {
            o.max_new_tokens = gen_toks;
            o.early_stopping = false;
        }

        // Warm up
        scheduler.generate_batch(prompts, opts);

        // Timed run
        const auto start = std::chrono::high_resolution_clock::now();
        auto batch_results = scheduler.generate_batch(prompts, opts);
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::size_t total_generated = 0;
        for (const auto& br : batch_results) {
            total_generated += br.generated_token_count;
        }

        BatchScalePoint point{};
        point.batch_size = b_size;
        point.total_time_ms = duration_ms;
        const double duration_s = duration_ms / 1000.0;
        point.aggregate_tokens_per_second = static_cast<double>(total_generated) / std::max(duration_s, 1e-6);

        if (b_size == 1 || baseline_throughput <= 0.0) {
            baseline_throughput = point.aggregate_tokens_per_second;
            point.speedup_ratio = 1.0;
            point.efficiency_percent = 100.0;
        } else {
            point.speedup_ratio = point.aggregate_tokens_per_second / std::max(baseline_throughput, 1e-6);
            point.efficiency_percent = (point.speedup_ratio / static_cast<double>(b_size)) * 100.0;
        }
        result.points.push_back(point);
    }

    return result;
}

BenchmarkSuiteReport BenchmarkEngine::run_full_suite(
    InferenceRuntime* runtime,
    const TransformerModelConfig* model_config) {

    BenchmarkSuiteReport report{};
    report.hardware = aabhas::hardware::HardwareCompatibilityEvaluator::detect_system_profile();
    report.kernels = benchmark_kernels(4096, 50);
    report.memory = benchmark_memory(32ULL * 1024ULL * 1024ULL, 20);

    if (runtime != nullptr && runtime->is_open()) {
        report.prefill = benchmark_prefill(*runtime, 32, 5);
        report.decode = benchmark_decode(*runtime, 16, 3);
        if (model_config != nullptr) {
            report.batching = benchmark_batching(*runtime, *model_config, 8, 4);
        }
    }

    return report;
}

std::string BenchmarkSuiteReport::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "{\n";
    oss << "  \"hardware\": {\n";
    oss << "    \"system\": \"" << hardware.host.system << "\",\n";
    oss << "    \"cpu_cores\": " << hardware.host.logical_cpu_count << ",\n";
    oss << "    \"cpu_features\": \"" << hardware.cpu.to_string() << "\",\n";
    oss << "    \"physical_memory_bytes\": " << hardware.host.physical_memory_bytes << ",\n";
    oss << "    \"primary_gpu\": \"" << hardware.primary_gpu().name << "\"\n";
    oss << "  },\n";
    oss << "  \"memory_bandwidth\": {\n";
    oss << "    \"copy_gbps\": " << memory.copy_bandwidth_gbps << ",\n";
    oss << "    \"read_gbps\": " << memory.sequential_read_gbps << ",\n";
    oss << "    \"write_gbps\": " << memory.sequential_write_gbps << "\n";
    oss << "  },\n";
    oss << "  \"quantized_kernels\": [\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        oss << "    {\n";
        oss << "      \"dtype\": \"" << kernels[i].dtype_name << "\",\n";
        oss << "      \"latency_us\": " << kernels[i].latency_us << ",\n";
        oss << "      \"gflops\": " << kernels[i].gflops << ",\n";
        oss << "      \"gbps\": " << kernels[i].gigabytes_per_sec << "\n";
        oss << "    }" << (i + 1 < kernels.size() ? "," : "") << "\n";
    }
    oss << "  ],\n";
    oss << "  \"prefill\": {\n";
    oss << "    \"ttft_mean_ms\": " << prefill.ttft_mean_ms << ",\n";
    oss << "    \"ttft_p50_ms\": " << prefill.ttft_p50_ms << ",\n";
    oss << "    \"ttft_p90_ms\": " << prefill.ttft_p90_ms << ",\n";
    oss << "    \"tokens_per_second\": " << prefill.tokens_per_second << "\n";
    oss << "  },\n";
    oss << "  \"decode\": {\n";
    oss << "    \"tpot_mean_ms\": " << decode.tpot_mean_ms << ",\n";
    oss << "    \"tpot_p50_ms\": " << decode.tpot_p50_ms << ",\n";
    oss << "    \"tpot_p90_ms\": " << decode.tpot_p90_ms << ",\n";
    oss << "    \"tokens_per_second\": " << decode.tokens_per_second << "\n";
    oss << "  }\n";
    oss << "}";
    return oss.str();
}

std::string BenchmarkSuiteReport::summary_table() const {
    std::ostringstream oss;
    oss << "================================================================================\n";
    oss << "                        AABHAS HARDWARE & INFERENCE BENCHMARK                  \n";
    oss << "================================================================================\n";
    oss << "System:       " << hardware.host.system << " (" << hardware.host.logical_cpu_count << " cores)\n";
    oss << "CPU Flags:    " << hardware.cpu.to_string() << "\n";
    oss << "Host Memory:  " << (hardware.host.physical_memory_bytes / (1024ULL * 1024ULL)) << " MiB physical\n";
    oss << "Accelerator:  " << hardware.primary_gpu().name << " (" << hardware.primary_gpu().vendor << ")\n";
    oss << "--------------------------------------------------------------------------------\n";
    oss << "MEMORY BANDWIDTH\n";
    oss << "  Copy: " << memory.copy_bandwidth_gbps << " GB/s | Read: "
        << memory.sequential_read_gbps << " GB/s | Write: " << memory.sequential_write_gbps << " GB/s\n";
    oss << "--------------------------------------------------------------------------------\n";
    oss << "QUANTIZED KERNEL PERFORMANCE (M=1, K=4096)\n";
    oss << "  Format    Latency (us)    Throughput (GFLOPS)    Bandwidth (GB/s)\n";
    for (const auto& k : kernels) {
        oss << "  " << std::left << std::setw(10) << k.dtype_name
            << std::right << std::setw(12) << std::fixed << std::setprecision(2) << k.latency_us
            << std::setw(23) << k.gflops
            << std::setw(20) << k.gigabytes_per_sec << "\n";
    }
    oss << "--------------------------------------------------------------------------------\n";
    if (prefill.tokens_per_second > 0.0 || decode.tokens_per_second > 0.0) {
        oss << "INFERENCE LATENCY & THROUGHPUT\n";
        oss << "  Prefill TTFT Mean: " << prefill.ttft_mean_ms << " ms | P50: "
            << prefill.ttft_p50_ms << " ms | P90: " << prefill.ttft_p90_ms << " ms ("
            << prefill.tokens_per_second << " tok/s)\n";
        oss << "  Decode  TPOT Mean: " << decode.tpot_mean_ms << " ms | P50: "
            << decode.tpot_p50_ms << " ms | P90: " << decode.tpot_p90_ms << " ms ("
            << decode.tokens_per_second << " tok/s)\n";
        oss << "--------------------------------------------------------------------------------\n";
    }
    return oss.str();
}

} // namespace aabhas::inference
