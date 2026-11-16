#include "hardware/hw_compatibility.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

void test_cpu_features_detection() {
    using namespace aabhas::hardware;
    CpuFeatures feat = HardwareCompatibilityEvaluator::detect_cpu_features();
    std::string str = feat.to_string();
    assert(!str.empty());
    // On any modern x86 or ARM CPU, features string is valid
    std::cout << "[PASS] test_cpu_features_detection: " << str << "\n";
}

void test_system_profile_assembly() {
    using namespace aabhas::hardware;
    HardwareProfile profile = HardwareCompatibilityEvaluator::detect_system_profile(".");
    assert(!profile.host.system.empty());
    assert(profile.host.logical_cpu_count > 0);
    assert(profile.cache.l1d_bytes > 0);
    assert(profile.cache.l2_bytes > 0);
    assert(!profile.gpus.empty());

    GpuDeviceInfo primary = profile.primary_gpu();
    assert(!primary.name.empty());
    assert(!primary.vendor.empty());
    std::cout << "[PASS] test_system_profile_assembly: host=" << profile.host.system
              << ", cpus=" << profile.host.logical_cpu_count
              << ", primary_gpu=" << primary.name << "\n";
}

void test_model_resource_estimation() {
    using namespace aabhas::hardware;
    using aabhas::model::TensorDType;

    // LLaMA-7B architecture: vocab 32000, dim 4096, 32 layers, 32 heads, 32 kv_heads, head_dim 128, ff 11008
    ModelResourceEstimate est_q4 = HardwareCompatibilityEvaluator::estimate_model_resources(
        32000, 4096, 32, 32, 32, 128, 11008, TensorDType::Q4_0, false);

    // Param count should be ~6.7 Billion parameters
    assert(est_q4.parameter_count > 6000000000ULL);
    assert(est_q4.parameter_count < 7500000000ULL);

    // Q4_0 weight bytes (~3.8 GB)
    assert(est_q4.weight_bytes > 3000000000ULL);
    assert(est_q4.weight_bytes < 4500000000ULL);

    // KV Cache: 2 * 32 * 32 * 128 * 4 = 1,048,576 bytes per token (1 MiB/token)
    assert(est_q4.kv_cache_bytes_per_token == 1048576ULL);
    assert(est_q4.kv_cache_bytes_at_context(2048) == 2048ULL * 1048576ULL); // 2 GiB

    // F16 estimation (~13.4 GB)
    ModelResourceEstimate est_f16 = HardwareCompatibilityEvaluator::estimate_model_resources(
        32000, 4096, 32, 32, 32, 128, 11008, TensorDType::F16, false);
    assert(est_f16.weight_bytes > 12000000000ULL);
    assert(est_f16.weight_bytes < 1500000000ULL * 10ULL);

    std::cout << "[PASS] test_model_resource_estimation: params=" << est_q4.parameter_count
              << ", q4_bytes=" << est_q4.weight_bytes
              << ", f16_bytes=" << est_f16.weight_bytes << "\n";
}

void test_compatibility_evaluation_tiers() {
    using namespace aabhas::hardware;
    using aabhas::model::TensorDType;

    ModelResourceEstimate est = HardwareCompatibilityEvaluator::estimate_model_resources(
        32000, 4096, 32, 32, 32, 128, 11008, TensorDType::Q4_0, false);

    // 1. Full GPU Tier (32 GB VRAM)
    {
        HardwareProfile prof{};
        prof.host.physical_memory_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.host.available_memory_bytes = 48ULL * 1024ULL * 1024ULL * 1024ULL;
        GpuDeviceInfo gpu{};
        gpu.backend_type = GpuBackendType::CUDA;
        gpu.vendor = "NVIDIA";
        gpu.name = "NVIDIA RTX 4090";
        gpu.vram_total_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
        gpu.vram_free_bytes  = 22ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.gpus.push_back(gpu);

        HardwareCompatibilityEvaluator eval(prof);
        CompatibilityAssessment assess = eval.evaluate(est, 2048, 1);
        assert(assess.can_run);
        assert(assess.recommended_tier == ExecutionTier::FullGpu);
        assert(assess.compatibility_score == 95);
        assert(assess.primary_bottleneck == BottleneckType::None);
    }

    // 2. Hybrid GPU Offload (4 GB VRAM for 3.8 GB model)
    {
        HardwareProfile prof{};
        prof.host.physical_memory_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.host.available_memory_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        GpuDeviceInfo gpu{};
        gpu.backend_type = GpuBackendType::ROCm;
        gpu.vendor = "AMD";
        gpu.name = "AMD Radeon RX 6600";
        gpu.vram_total_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        gpu.vram_free_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2 GB free < 3.8 GB
        prof.gpus.push_back(gpu);

        HardwareCompatibilityEvaluator eval(prof);
        CompatibilityAssessment assess = eval.evaluate(est, 2048, 1);
        assert(assess.can_run);
        assert(assess.recommended_tier == ExecutionTier::HybridGpuOffload);
        assert(assess.primary_bottleneck == BottleneckType::VramCapacity);
    }

    // 3. CPU RAM Resident (No GPU, 32 GB RAM)
    {
        HardwareProfile prof{};
        prof.host.physical_memory_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.host.available_memory_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.cpu.avx2 = true;
        // Primary GPU is emulated
        GpuDeviceInfo emu{};
        emu.backend_type = GpuBackendType::Emulated;
        prof.gpus.push_back(emu);

        HardwareCompatibilityEvaluator eval(prof);
        CompatibilityAssessment assess = eval.evaluate(est, 2048, 1);
        assert(assess.can_run);
        assert(assess.recommended_tier == ExecutionTier::CpuRam);
    }

    // 4. NVMe Out-of-Core Streaming (RAM 4 GB, Weights 3.8 GB + KV 2 GB = 5.8 GB, 500 GB NVMe)
    {
        HardwareProfile prof{};
        prof.host.physical_memory_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.host.available_memory_bytes = 3ULL * 1024ULL * 1024ULL * 1024ULL; // 3 GB free < 5.8 GB total
        prof.drive.type = DriveType::NVMe;
        prof.drive.capacity_bytes = 500ULL * 1024ULL * 1024ULL * 1024ULL;
        prof.drive.available_bytes = 200ULL * 1024ULL * 1024ULL * 1024ULL;
        GpuDeviceInfo emu{};
        emu.backend_type = GpuBackendType::Emulated;
        prof.gpus.push_back(emu);

        HardwareCompatibilityEvaluator eval(prof);
        CompatibilityAssessment assess = eval.evaluate(est, 1024, 1);
        assert(assess.can_run);
        assert(assess.recommended_tier == ExecutionTier::NvmeOutofCore);
        assert(assess.max_recommended_batch_size == 1);
    }

    // 5. Incompatible (RAM 512 MB, Drive full)
    {
        HardwareProfile prof{};
        prof.host.physical_memory_bytes = 512ULL * 1024ULL * 1024ULL;
        prof.host.available_memory_bytes = 256ULL * 1024ULL * 1024ULL;
        prof.drive.type = DriveType::HDD;
        prof.drive.capacity_bytes = 100ULL * 1024ULL * 1024ULL;
        prof.drive.available_bytes = 50ULL * 1024ULL * 1024ULL;

        HardwareCompatibilityEvaluator eval(prof);
        CompatibilityAssessment assess = eval.evaluate(est, 2048, 1);
        assert(!assess.can_run);
        assert(assess.recommended_tier == ExecutionTier::Incompatible);
    }

    std::cout << "[PASS] test_compatibility_evaluation_tiers\n";
}

void test_json_and_string_serialization() {
    using namespace aabhas::hardware;
    CompatibilityAssessment assess{};
    assess.can_run = true;
    assess.recommended_tier = ExecutionTier::CpuRam;
    assess.primary_bottleneck = BottleneckType::HostBandwidth;
    assess.compatibility_score = 75;
    assess.max_recommended_batch_size = 4;
    assess.max_recommended_context_length = 4096;
    assess.diagnostics.emplace_back("Model fits in system memory.");

    std::string json = assess.to_json();
    assert(json.find("\"can_run\": true") != std::string::npos);
    assert(json.find("\"recommended_tier\": \"CpuRam\"") != std::string::npos);
    assert(json.find("\"primary_bottleneck\": \"HostBandwidth\"") != std::string::npos);
    assert(json.find("\"compatibility_score\": 75") != std::string::npos);

    std::cout << "[PASS] test_json_and_string_serialization\n";
}

} // namespace

int main() {
    test_cpu_features_detection();
    test_system_profile_assembly();
    test_model_resource_estimation();
    test_compatibility_evaluation_tiers();
    test_json_and_string_serialization();
    std::cout << "All Hardware Compatibility tests passed successfully.\n";
    return 0;
}
