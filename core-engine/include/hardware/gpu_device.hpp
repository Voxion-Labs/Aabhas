#ifndef AABHAS_HARDWARE_GPU_DEVICE_HPP
#define AABHAS_HARDWARE_GPU_DEVICE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aabhas::hardware {

/** Accelerator compute backend technology. */
enum class GpuBackendType : std::uint32_t {
    None = 0,
    Emulated = 1,
    Vulkan = 2,
    CUDA = 3,
    ROCm = 4
};

/** Hardware specifications and runtime limits of a compute device. */
struct GpuDeviceInfo {
    std::uint32_t device_id{0};
    GpuBackendType backend_type{GpuBackendType::None};
    std::string name;
    std::string vendor;
    std::uint64_t vram_total_bytes{0};
    std::uint64_t vram_free_bytes{0};
    std::size_t compute_units{0};
    std::size_t max_workgroup_size{0};
    bool supports_fp16{false};
    bool supports_bf16{false};
    bool unified_memory{false};
};

/**
 * Discovers and manages host GPU/accelerator hardware.
 *
 * Inspects Linux DRM subsystem, NVIDIA procfs/devfs entries, and system devices.
 * Always guarantees at least an Emulated backend is available for headless CI/CD
 * and CPU fallback environments.
 */
class GpuDeviceRegistry final {
public:
    GpuDeviceRegistry() = default;
    ~GpuDeviceRegistry() = default;

    /** Enumerate all compute accelerators available on the host system. */
    static std::vector<GpuDeviceInfo> enumerate_devices();

    /** Get the primary recommended device (or Emulated if no hardware GPU exists). */
    static GpuDeviceInfo default_device();

    /** String representation of backend technology. */
    static const char* backend_name(GpuBackendType type) noexcept;
};

} // namespace aabhas::hardware

#endif // AABHAS_HARDWARE_GPU_DEVICE_HPP
