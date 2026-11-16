#include "hardware/hw_compatibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <unistd.h>
#endif

namespace aabhas::hardware {

std::string CpuFeatures::to_string() const {
    std::vector<std::string> active;
    if (sse42) active.emplace_back("SSE4.2");
    if (avx) active.emplace_back("AVX");
    if (avx2) active.emplace_back("AVX2");
    if (fma3) active.emplace_back("FMA3");
    if (f16c) active.emplace_back("F16C");
    if (avx512f) active.emplace_back("AVX-512F");
    if (avx512cd) active.emplace_back("AVX-512CD");
    if (avx512bw) active.emplace_back("AVX-512BW");
    if (avx512dq) active.emplace_back("AVX-512DQ");
    if (avx512vl) active.emplace_back("AVX-512VL");
    if (avx512_vnni) active.emplace_back("AVX-512-VNNI");
    if (amx_tile) active.emplace_back("AMX-TILE");
    if (amx_int8) active.emplace_back("AMX-INT8");
    if (amx_bf16) active.emplace_back("AMX-BF16");
    if (arm_neon) active.emplace_back("ARM-NEON");
    if (arm_sve) active.emplace_back("ARM-SVE");

    if (active.empty()) return "None";

    std::ostringstream oss;
    for (std::size_t i = 0; i < active.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << active[i];
    }
    return oss.str();
}

GpuDeviceInfo HardwareProfile::primary_gpu() const {
    if (gpus.empty()) {
        GpuDeviceInfo fallback{};
        fallback.backend_type = GpuBackendType::Emulated;
        fallback.name = "Aabhas Emulated Hardware";
        fallback.vendor = "Voxion-Labs";
        return fallback;
    }
    for (const auto& dev : gpus) {
        if (dev.backend_type != GpuBackendType::Emulated) {
            return dev;
        }
    }
    return gpus.front();
}

std::uint64_t ModelResourceEstimate::kv_cache_bytes_at_context(std::size_t context_length) const noexcept {
    return kv_cache_bytes_per_token * static_cast<std::uint64_t>(context_length);
}

std::uint64_t ModelResourceEstimate::total_memory_required(std::size_t context_length, std::size_t batch_size) const noexcept {
    const std::uint64_t total_kv = kv_cache_bytes_at_context(context_length) * static_cast<std::uint64_t>(batch_size);
    return weight_bytes + total_kv + activation_scratch_bytes;
}

const char* HardwareCompatibilityEvaluator::tier_name(ExecutionTier tier) noexcept {
    switch (tier) {
    case ExecutionTier::FullGpu: return "FullGpu";
    case ExecutionTier::HybridGpuOffload: return "HybridGpuOffload";
    case ExecutionTier::CpuRam: return "CpuRam";
    case ExecutionTier::NvmeOutofCore: return "NvmeOutofCore";
    case ExecutionTier::Incompatible: return "Incompatible";
    }
    return "Unknown";
}

const char* HardwareCompatibilityEvaluator::bottleneck_name(BottleneckType bottleneck) noexcept {
    switch (bottleneck) {
    case BottleneckType::None: return "None";
    case BottleneckType::VramCapacity: return "VramCapacity";
    case BottleneckType::SystemMemory: return "SystemMemory";
    case BottleneckType::StorageIops: return "StorageIops";
    case BottleneckType::CpuVectorWidth: return "CpuVectorWidth";
    case BottleneckType::HostBandwidth: return "HostBandwidth";
    }
    return "Unknown";
}

std::string CompatibilityAssessment::tier_name() const {
    return HardwareCompatibilityEvaluator::tier_name(recommended_tier);
}

std::string CompatibilityAssessment::bottleneck_name() const {
    return HardwareCompatibilityEvaluator::bottleneck_name(primary_bottleneck);
}

std::string CompatibilityAssessment::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"can_run\": " << (can_run ? "true" : "false") << ",\n";
    oss << "  \"recommended_tier\": \"" << tier_name() << "\",\n";
    oss << "  \"primary_bottleneck\": \"" << bottleneck_name() << "\",\n";
    oss << "  \"compatibility_score\": " << compatibility_score << ",\n";
    oss << "  \"max_recommended_batch_size\": " << max_recommended_batch_size << ",\n";
    oss << "  \"max_recommended_context_length\": " << max_recommended_context_length << ",\n";
    oss << "  \"diagnostics\": [\n";
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        oss << "    \"" << diagnostics[i] << "\"";
        if (i + 1 < diagnostics.size()) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}";
    return oss.str();
}

HardwareCompatibilityEvaluator::HardwareCompatibilityEvaluator(HardwareProfile profile)
    : profile_(std::move(profile)) {}

CpuFeatures HardwareCompatibilityEvaluator::detect_cpu_features() {
    CpuFeatures feat{};

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4]{};
    __cpuid(info, 0);
    const int num_ids = info[0];

    if (num_ids >= 1) {
        __cpuid(info, 1);
        feat.sse42 = (info[2] & (1 << 20)) != 0;
        feat.f16c  = (info[2] & (1 << 29)) != 0;
        feat.fma3  = (info[2] & (1 << 12)) != 0;
        feat.avx   = (info[2] & (1 << 28)) != 0;
    }

    if (num_ids >= 7) {
        __cpuidex(info, 7, 0);
        feat.avx2        = (info[1] & (1 << 5)) != 0;
        feat.avx512f     = (info[1] & (1 << 16)) != 0;
        feat.avx512dq    = (info[1] & (1 << 17)) != 0;
        feat.avx512cd    = (info[1] & (1 << 28)) != 0;
        feat.avx512bw    = (info[1] & (1 << 30)) != 0;
        feat.avx512vl    = (info[1] & (1 << 31)) != 0;
        feat.avx512_vnni = (info[2] & (1 << 11)) != 0;
        feat.amx_bf16    = (info[3] & (1 << 22)) != 0;
        feat.amx_tile    = (info[3] & (1 << 24)) != 0;
        feat.amx_int8    = (info[3] & (1 << 25)) != 0;
    }
#elif (defined(__x86_64__) || defined(__i386__))
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        const unsigned int max_leaf = eax;
        if (max_leaf >= 1) {
            __cpuid(1, eax, ebx, ecx, edx);
            feat.sse42 = (ecx & (1 << 20)) != 0;
            feat.f16c  = (ecx & (1 << 29)) != 0;
            feat.fma3  = (ecx & (1 << 12)) != 0;
            feat.avx   = (ecx & (1 << 28)) != 0;
        }
        if (max_leaf >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            feat.avx2        = (ebx & (1 << 5)) != 0;
            feat.avx512f     = (ebx & (1 << 16)) != 0;
            feat.avx512dq    = (ebx & (1 << 17)) != 0;
            feat.avx512cd    = (ebx & (1 << 28)) != 0;
            feat.avx512bw    = (ebx & (1 << 30)) != 0;
            feat.avx512vl    = (ebx & (1 << 31)) != 0;
            feat.avx512_vnni = (ecx & (1 << 11)) != 0;
            feat.amx_bf16    = (edx & (1 << 22)) != 0;
            feat.amx_tile    = (edx & (1 << 24)) != 0;
            feat.amx_int8    = (edx & (1 << 25)) != 0;
        }
    }
#endif

#if defined(__linux__)
    // Fallback or ARM detection via /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("flags", 0) == 0 || line.rfind("Features", 0) == 0) {
                if (line.find(" sse4_2") != std::string::npos) feat.sse42 = true;
                if (line.find(" avx ") != std::string::npos) feat.avx = true;
                if (line.find(" avx2 ") != std::string::npos) feat.avx2 = true;
                if (line.find(" fma ") != std::string::npos) feat.fma3 = true;
                if (line.find(" f16c ") != std::string::npos) feat.f16c = true;
                if (line.find(" avx512f ") != std::string::npos) feat.avx512f = true;
                if (line.find(" avx512cd ") != std::string::npos) feat.avx512cd = true;
                if (line.find(" avx512bw ") != std::string::npos) feat.avx512bw = true;
                if (line.find(" avx512dq ") != std::string::npos) feat.avx512dq = true;
                if (line.find(" avx512vl ") != std::string::npos) feat.avx512vl = true;
                if (line.find(" avx512_vnni ") != std::string::npos) feat.avx512_vnni = true;
                if (line.find(" asimd") != std::string::npos || line.find(" neon") != std::string::npos) feat.arm_neon = true;
                if (line.find(" sve") != std::string::npos) feat.arm_sve = true;
                break;
            }
        }
    }
#endif

    return feat;
}

HardwareProfile HardwareCompatibilityEvaluator::detect_system_profile(const char* storage_path) {
    HardwareProfile profile{};
    HardwareIdentifier ident;
    profile.host = ident.host();
    profile.cpu = detect_cpu_features();
    profile.drive = ident.drive(storage_path != nullptr ? storage_path : ".");
    profile.gpus = GpuDeviceRegistry::enumerate_devices();

    // Cache geometry defaults and detection
    profile.cache.l1d_bytes = 32ULL * 1024ULL;
    profile.cache.l1i_bytes = 32ULL * 1024ULL;
    profile.cache.l2_bytes  = 512ULL * 1024ULL;
    profile.cache.l3_bytes  = (profile.host.logical_cpu_count > 0)
                                  ? (profile.host.logical_cpu_count * 2ULL * 1024ULL * 1024ULL)
                                  : (16ULL * 1024ULL * 1024ULL);
    profile.cache.numa_nodes = 1;
    profile.cache.estimated_mem_bandwidth_gbps = 34ULL; // Default baseline DDR4

#if defined(__linux__)
    std::ifstream numa_nodes_file("/sys/devices/system/node/online");
    if (numa_nodes_file.is_open()) {
        std::string numa_range;
        numa_nodes_file >> numa_range;
        if (!numa_range.empty()) {
            const auto dash = numa_range.find('-');
            if (dash != std::string::npos) {
                profile.cache.numa_nodes = static_cast<std::uint32_t>(
                    std::stoul(numa_range.substr(dash + 1)) + 1);
            }
        }
    }
#endif

    return profile;
}

ModelResourceEstimate HardwareCompatibilityEvaluator::estimate_model_resources(
    std::size_t vocab_size,
    std::size_t model_dimension,
    std::size_t layer_count,
    std::size_t head_count,
    std::size_t kv_head_count,
    std::size_t head_dimension,
    std::size_t intermediate_dimension,
    aabhas::model::TensorDType weight_dtype,
    bool tied_lm_head) noexcept {

    ModelResourceEstimate est{};
    if (vocab_size == 0 || model_dimension == 0 || layer_count == 0) {
        return est;
    }

    const std::uint64_t d = static_cast<std::uint64_t>(model_dimension);
    const std::uint64_t h = static_cast<std::uint64_t>(head_count > 0 ? head_count : 1);
    const std::uint64_t h_kv = static_cast<std::uint64_t>(kv_head_count > 0 ? kv_head_count : h);
    const std::uint64_t d_k = static_cast<std::uint64_t>(head_dimension > 0 ? head_dimension : (d / h));
    const std::uint64_t d_ff = static_cast<std::uint64_t>(intermediate_dimension > 0 ? intermediate_dimension : (4 * d));
    const std::uint64_t v = static_cast<std::uint64_t>(vocab_size);
    const std::uint64_t layers = static_cast<std::uint64_t>(layer_count);

    // 1. Embedding parameters: vocab * d
    const std::uint64_t embed_params = v * d;

    // 2. Per-layer parameters:
    // Attention: Q (d * h * d_k), K (d * h_kv * d_k), V (d * h_kv * d_k), O (h * d_k * d)
    const std::uint64_t q_params = d * (h * d_k);
    const std::uint64_t k_params = d * (h_kv * d_k);
    const std::uint64_t v_params = d * (h_kv * d_k);
    const std::uint64_t o_params = (h * d_k) * d;
    const std::uint64_t attn_norm = d;
    const std::uint64_t attn_params = q_params + k_params + v_params + o_params + attn_norm;

    // Feedforward: Gate (d * d_ff), Up (d * d_ff), Down (d_ff * d), Norm (d)
    const std::uint64_t ffn_gate = d * d_ff;
    const std::uint64_t ffn_up   = d * d_ff;
    const std::uint64_t ffn_down = d_ff * d;
    const std::uint64_t ffn_norm = d;
    const std::uint64_t ffn_params = ffn_gate + ffn_up + ffn_down + ffn_norm;

    const std::uint64_t per_layer_params = attn_params + ffn_params;

    // 3. Final norm and LM head
    const std::uint64_t final_norm = d;
    const std::uint64_t lm_head_params = tied_lm_head ? 0 : (d * v);

    est.parameter_count = embed_params + (layers * per_layer_params) + final_norm + lm_head_params;

    // Bytes per parameter for weights
    double bytes_per_param = 4.0; // Default FP32
    switch (weight_dtype) {
    case aabhas::model::TensorDType::Q4_0:
        bytes_per_param = 18.0 / 32.0; // 0.5625 bytes/weight
        break;
    case aabhas::model::TensorDType::Q4_1:
        bytes_per_param = 20.0 / 32.0; // 0.625 bytes/weight
        break;
    case aabhas::model::TensorDType::Q5_0:
        bytes_per_param = 22.0 / 32.0; // 0.6875 bytes/weight
        break;
    case aabhas::model::TensorDType::Q5_1:
        bytes_per_param = 24.0 / 32.0; // 0.75 bytes/weight
        break;
    case aabhas::model::TensorDType::Q8_0:
        bytes_per_param = 34.0 / 32.0; // 1.0625 bytes/weight
        break;
    case aabhas::model::TensorDType::F16:
    case aabhas::model::TensorDType::BF16:
        bytes_per_param = 2.0;
        break;
    case aabhas::model::TensorDType::F32:
    default:
        bytes_per_param = 4.0;
        break;
    }

    est.weight_bytes = static_cast<std::uint64_t>(std::ceil(static_cast<double>(est.parameter_count) * bytes_per_param));

    // KV Cache: 2 (K & V) * layer_count * kv_head_count * head_dimension * sizeof(float)
    est.kv_cache_bytes_per_token = 2ULL * layers * h_kv * d_k * sizeof(float);

    // Scratchpad activation requirement for 1 forward pass
    const std::uint64_t max_intermediate = std::max(d, d_ff);
    est.activation_scratch_bytes = 4ULL * max_intermediate * sizeof(float) + (h * d_k * sizeof(float));

    return est;
}

CompatibilityAssessment HardwareCompatibilityEvaluator::evaluate(
    const ModelResourceEstimate& estimate,
    std::size_t target_context,
    std::size_t target_batch) const {

    CompatibilityAssessment result{};
    if (estimate.parameter_count == 0) {
        result.can_run = false;
        result.recommended_tier = ExecutionTier::Incompatible;
        result.primary_bottleneck = BottleneckType::SystemMemory;
        result.diagnostics.emplace_back("Model parameter count is zero or invalid.");
        return result;
    }

    const std::uint64_t target_ctx = static_cast<std::uint64_t>(target_context > 0 ? target_context : 2048);
    const std::uint64_t target_b = static_cast<std::uint64_t>(target_batch > 0 ? target_batch : 1);
    const std::uint64_t total_kv_bytes = estimate.kv_cache_bytes_at_context(target_ctx) * target_b;
    const std::uint64_t total_resident_memory = estimate.weight_bytes + total_kv_bytes + estimate.activation_scratch_bytes;

    const std::uint64_t host_phys_ram = profile_.host.physical_memory_bytes;
    const std::uint64_t host_avail_ram = profile_.host.available_memory_bytes > 0
                                             ? profile_.host.available_memory_bytes
                                             : (host_phys_ram * 8ULL / 10ULL);

    const auto primary_gpu = profile_.primary_gpu();
    const bool has_hardware_gpu = (primary_gpu.backend_type != GpuBackendType::Emulated &&
                                   primary_gpu.backend_type != GpuBackendType::None);
    const std::uint64_t vram_bytes = primary_gpu.vram_free_bytes > 0
                                         ? primary_gpu.vram_free_bytes
                                         : primary_gpu.vram_total_bytes;

    // Minimum RAM required when streaming out-of-core from NVMe:
    // RAM must hold KV cache + active layer window (~10% of weights) + activation scratchpad
    const std::uint64_t layer_window_ram = std::max<std::uint64_t>(estimate.weight_bytes / 8ULL, 32ULL * 1024ULL * 1024ULL);
    const std::uint64_t min_streamed_ram_required = total_kv_bytes + layer_window_ram + estimate.activation_scratch_bytes;

    // 1. Tier: Full GPU
    if (has_hardware_gpu && vram_bytes >= total_resident_memory) {
        result.can_run = true;
        result.recommended_tier = ExecutionTier::FullGpu;
        result.primary_bottleneck = BottleneckType::None;
        result.compatibility_score = 95;
        result.max_recommended_batch_size = std::max<std::size_t>(1, static_cast<std::size_t>(vram_bytes / std::max<std::uint64_t>(1, estimate.weight_bytes + estimate.activation_scratch_bytes)));
        result.max_recommended_context_length = static_cast<std::size_t>(target_ctx);
        result.diagnostics.emplace_back("Model fits entirely within accelerator VRAM for maximum throughput.");
        return result;
    }

    // 2. Tier: Hybrid GPU Offload
    if (has_hardware_gpu && vram_bytes >= (estimate.weight_bytes / 2ULL)) {
        result.can_run = true;
        result.recommended_tier = ExecutionTier::HybridGpuOffload;
        result.primary_bottleneck = BottleneckType::VramCapacity;
        result.compatibility_score = 80;
        result.max_recommended_batch_size = std::min<std::size_t>(static_cast<std::size_t>(target_b), 4);
        result.max_recommended_context_length = static_cast<std::size_t>(target_ctx);
        result.diagnostics.emplace_back("Partial GPU offload active; memory bandwidth across PCIe will dictate token latency.");
        return result;
    }

    // 3. Tier: CPU RAM Resident
    if (host_avail_ram >= total_resident_memory) {
        result.can_run = true;
        result.recommended_tier = ExecutionTier::CpuRam;
        result.primary_bottleneck = profile_.cpu.avx2 ? BottleneckType::HostBandwidth : BottleneckType::CpuVectorWidth;
        result.compatibility_score = profile_.cpu.avx512f ? 85 : (profile_.cpu.avx2 ? 75 : 55);
        result.max_recommended_batch_size = std::max<std::size_t>(1, static_cast<std::size_t>((host_avail_ram - estimate.weight_bytes) / std::max<std::uint64_t>(1, estimate.kv_cache_bytes_at_context(target_ctx))));
        result.max_recommended_context_length = static_cast<std::size_t>(target_ctx);

        if (profile_.cpu.avx512_vnni) {
            result.diagnostics.emplace_back("AVX-512 VNNI vector acceleration enabled for quantized inference.");
        } else if (profile_.cpu.avx2) {
            result.diagnostics.emplace_back("AVX2 vector acceleration active; model resides fully in system RAM.");
        } else {
            result.diagnostics.emplace_back("CPU lacks AVX2/AVX-512 vector units; execution will rely on scalar kernels.");
        }
        return result;
    }

    // 4. Tier: NVMe Out-of-Core Layer Dropping
    const bool storage_has_capacity = (profile_.drive.available_bytes >= estimate.weight_bytes ||
                                       profile_.drive.capacity_bytes >= estimate.weight_bytes);
    if (host_avail_ram >= min_streamed_ram_required && storage_has_capacity) {
        result.can_run = true;
        result.recommended_tier = ExecutionTier::NvmeOutofCore;
        result.primary_bottleneck = (profile_.drive.type == DriveType::NVMe) ? BottleneckType::HostBandwidth : BottleneckType::StorageIops;
        result.compatibility_score = (profile_.drive.type == DriveType::NVMe) ? 65 : 45;
        result.max_recommended_batch_size = 1; // Out-of-core streaming is optimal with batch size 1
        result.max_recommended_context_length = std::min<std::size_t>(static_cast<std::size_t>(target_ctx), 4096);

        std::string drive_info = profile_.drive.path.empty() ? "storage" : profile_.drive.path;
        if (profile_.drive.type == DriveType::NVMe) {
            result.diagnostics.emplace_back("Direct NVMe-to-CPU tensor streaming active. Out-of-core execution bypasses host page table overhead.");
        } else {
            result.diagnostics.emplace_back("Storage backing is SSD/HDD; NVMe is recommended for reduced layer streaming latency.");
        }
        return result;
    }

    // 5. Incompatible
    result.can_run = false;
    result.recommended_tier = ExecutionTier::Incompatible;
    result.primary_bottleneck = BottleneckType::SystemMemory;
    result.compatibility_score = 10;
    result.max_recommended_batch_size = 0;
    result.max_recommended_context_length = 0;

    std::ostringstream err;
    err << "Insufficient system RAM (" << (host_avail_ram / (1024ULL * 1024ULL))
        << " MiB available) and storage for model resource footprint ("
        << (estimate.weight_bytes / (1024ULL * 1024ULL)) << " MiB weights).";
    result.diagnostics.push_back(err.str());
    return result;
}

} // namespace aabhas::hardware
