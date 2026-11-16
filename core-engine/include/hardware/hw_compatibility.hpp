#ifndef AABHAS_HARDWARE_HW_COMPATIBILITY_HPP
#define AABHAS_HARDWARE_HW_COMPATIBILITY_HPP

#include "hardware/gpu_device.hpp"
#include "hardware/hw_identifier.hpp"
#include "model/model_index.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aabhas::hardware {

/** CPU SIMD vector extensions and acceleration capabilities. */
struct CpuFeatures {
    bool sse42{false};
    bool avx{false};
    bool avx2{false};
    bool fma3{false};
    bool f16c{false};
    bool avx512f{false};
    bool avx512cd{false};
    bool avx512bw{false};
    bool avx512dq{false};
    bool avx512vl{false};
    bool avx512_vnni{false};
    bool amx_tile{false};
    bool amx_int8{false};
    bool amx_bf16{false};
    bool arm_neon{false};
    bool arm_sve{false};

    /** Convert active flags to comma-separated feature names. */
    std::string to_string() const;
};

/** Hardware cache geometry and NUMA configuration. */
struct CpuCacheGeometry {
    std::uint64_t l1d_bytes{32ULL * 1024ULL};
    std::uint64_t l1i_bytes{32ULL * 1024ULL};
    std::uint64_t l2_bytes{512ULL * 1024ULL};
    std::uint64_t l3_bytes{16ULL * 1024ULL * 1024ULL};
    std::uint32_t numa_nodes{1};
    std::uint64_t estimated_mem_bandwidth_gbps{34}; // Default DDR4 dual-channel baseline
};

/** Complete hardware snapshot combining CPU, RAM, Storage, and Accelerators. */
struct HardwareProfile {
    HostSpecifications host;
    CpuFeatures cpu;
    CpuCacheGeometry cache;
    DriveSpecifications drive;
    std::vector<GpuDeviceInfo> gpus;

    /** Primary GPU or default emulated accelerator. */
    GpuDeviceInfo primary_gpu() const;
};

/** Theoretical memory and parameter requirements for a transformer model. */
struct ModelResourceEstimate {
    std::uint64_t parameter_count{0};
    std::uint64_t weight_bytes{0};
    std::uint64_t kv_cache_bytes_per_token{0};
    std::uint64_t activation_scratch_bytes{0};
    std::uint64_t kv_cache_bytes_at_context(std::size_t context_length) const noexcept;
    std::uint64_t total_memory_required(std::size_t context_length, std::size_t batch_size = 1) const noexcept;
};

/** Execution deployment tier based on available hardware hierarchy. */
enum class ExecutionTier : std::uint32_t {
    FullGpu = 0,         // Fits entirely in GPU VRAM (highest throughput)
    HybridGpuOffload = 1,// Critical layers on GPU, remainder on CPU
    CpuRam = 2,          // Complete model resident in host system RAM
    NvmeOutofCore = 3,   // System RAM < model weights; layer drops streamed from NVMe
    Incompatible = 4     // System lacks minimum storage or memory to execute model
};

/** Primary hardware bottleneck identified during compatibility analysis. */
enum class BottleneckType : std::uint32_t {
    None = 0,
    VramCapacity = 1,
    SystemMemory = 2,
    StorageIops = 3,
    CpuVectorWidth = 4,
    HostBandwidth = 5
};

/** Structured compatibility assessment result for model deployment. */
struct CompatibilityAssessment {
    bool can_run{false};
    ExecutionTier recommended_tier{ExecutionTier::Incompatible};
    BottleneckType primary_bottleneck{BottleneckType::None};
    std::uint32_t compatibility_score{0}; // 0 - 100
    std::size_t max_recommended_batch_size{1};
    std::size_t max_recommended_context_length{2048};
    std::vector<std::string> diagnostics;

    std::string tier_name() const;
    std::string bottleneck_name() const;
    std::string to_json() const;
};

/**
 * Analyzes system hardware specifications and evaluates model execution feasibility.
 */
class HardwareCompatibilityEvaluator final {
public:
    HardwareCompatibilityEvaluator() = default;
    explicit HardwareCompatibilityEvaluator(HardwareProfile profile);

    /** Query and assemble a complete hardware profile from the active host. */
    static HardwareProfile detect_system_profile(const char* storage_path = ".");

    /** Query CPU SIMD features via CPUID or OS-level telemetry. */
    static CpuFeatures detect_cpu_features();

    /** Estimate parameters and memory footprint for a transformer architecture. */
    static ModelResourceEstimate estimate_model_resources(
        std::size_t vocab_size,
        std::size_t model_dimension,
        std::size_t layer_count,
        std::size_t head_count,
        std::size_t kv_head_count,
        std::size_t head_dimension,
        std::size_t intermediate_dimension,
        aabhas::model::TensorDType weight_dtype,
        bool tied_lm_head = false) noexcept;

    /** Evaluate execution feasibility and recommend optimal deployment tier. */
    CompatibilityAssessment evaluate(
        const ModelResourceEstimate& estimate,
        std::size_t target_context = 2048,
        std::size_t target_batch = 1) const;

    /** Access current hardware profile. */
    const HardwareProfile& profile() const noexcept { return profile_; }

    /** String representation of execution tier. */
    static const char* tier_name(ExecutionTier tier) noexcept;

    /** String representation of bottleneck type. */
    static const char* bottleneck_name(BottleneckType bottleneck) noexcept;

private:
    HardwareProfile profile_{};
};

} // namespace aabhas::hardware

#endif // AABHAS_HARDWARE_HW_COMPATIBILITY_HPP
