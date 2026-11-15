#include "hardware/gpu_device.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aabhas::hardware {

const char* GpuDeviceRegistry::backend_name(GpuBackendType type) noexcept {
    switch (type) {
    case GpuBackendType::Emulated:
        return "Emulated";
    case GpuBackendType::Vulkan:
        return "Vulkan";
    case GpuBackendType::CUDA:
        return "CUDA";
    case GpuBackendType::ROCm:
        return "ROCm";
    case GpuBackendType::None:
    default:
        return "None";
    }
}

std::vector<GpuDeviceInfo> GpuDeviceRegistry::enumerate_devices() {
    std::vector<GpuDeviceInfo> devices;

#if defined(__linux__)
    // 1. Check for NVIDIA discrete devices via /proc or /dev
    struct stat st {};
    if (::stat("/dev/nvidia0", &st) == 0 || ::stat("/proc/driver/nvidia/version", &st) == 0) {
        GpuDeviceInfo nv_device{};
        nv_device.device_id = static_cast<std::uint32_t>(devices.size());
        nv_device.backend_type = GpuBackendType::CUDA;
        nv_device.vendor = "NVIDIA";
        nv_device.name = "NVIDIA Hardware Accelerator";
        nv_device.vram_total_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB default estimate
        nv_device.vram_free_bytes = 14ULL * 1024ULL * 1024ULL * 1024ULL;
        nv_device.compute_units = 128;
        nv_device.max_workgroup_size = 1024;
        nv_device.supports_fp16 = true;
        nv_device.supports_bf16 = true;
        nv_device.unified_memory = false;
        devices.push_back(std::move(nv_device));
    }

    // 2. Check Linux DRM card devices
    for (int card_idx = 0; card_idx < 4; ++card_idx) {
        std::string base_path = "/sys/class/drm/card" + std::to_string(card_idx) + "/device";
        std::ifstream vendor_file(base_path + "/vendor");
        if (!vendor_file.is_open()) continue;

        std::string vendor_id_str;
        vendor_file >> vendor_id_str;
        if (vendor_id_str.empty()) continue;

        std::string vendor_name = "Unknown Vendor";
        GpuBackendType backend = GpuBackendType::Vulkan;
        if (vendor_id_str.find("0x10de") != std::string::npos) {
            vendor_name = "NVIDIA";
            backend = GpuBackendType::CUDA;
        } else if (vendor_id_str.find("0x1002") != std::string::npos) {
            vendor_name = "AMD";
            backend = GpuBackendType::ROCm;
        } else if (vendor_id_str.find("0x8086") != std::string::npos) {
            vendor_name = "Intel";
            backend = GpuBackendType::Vulkan;
        }

        // Avoid duplicate entry if NVIDIA was already discovered via /dev/nvidia
        bool already_present = false;
        for (const auto& existing : devices) {
            if (existing.vendor == vendor_name && existing.backend_type == backend) {
                already_present = true;
                break;
            }
        }
        if (already_present) continue;

        GpuDeviceInfo drm_device{};
        drm_device.device_id = static_cast<std::uint32_t>(devices.size());
        drm_device.backend_type = backend;
        drm_device.vendor = vendor_name;
        drm_device.name = vendor_name + " GPU (card" + std::to_string(card_idx) + ")";
        drm_device.vram_total_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        drm_device.vram_free_bytes = 7ULL * 1024ULL * 1024ULL * 1024ULL;
        drm_device.compute_units = 64;
        drm_device.max_workgroup_size = 512;
        drm_device.supports_fp16 = true;
        drm_device.supports_bf16 = true;
        drm_device.unified_memory = false;
        devices.push_back(std::move(drm_device));
    }
#endif

    // Always provide an Emulated accelerator for deterministic execution and headless CI
    GpuDeviceInfo emu_device{};
    emu_device.device_id = static_cast<std::uint32_t>(devices.size());
    emu_device.backend_type = GpuBackendType::Emulated;
    emu_device.vendor = "Voxion-Labs";
    emu_device.name = "Aabhas High-Throughput Matrix Engine (Virtual)";
    emu_device.vram_total_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL; // 8 GiB virtual VRAM
    emu_device.vram_free_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    emu_device.compute_units = 64;
    emu_device.max_workgroup_size = 1024;
    emu_device.supports_fp16 = true;
    emu_device.supports_bf16 = true;
    emu_device.unified_memory = true;
    devices.push_back(std::move(emu_device));

    return devices;
}

GpuDeviceInfo GpuDeviceRegistry::default_device() {
    auto devices = enumerate_devices();
    if (devices.empty()) {
        GpuDeviceInfo fallback{};
        fallback.backend_type = GpuBackendType::Emulated;
        fallback.name = "Aabhas Virtual Accelerator Fallback";
        fallback.vendor = "Voxion-Labs";
        fallback.vram_total_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        fallback.vram_free_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        fallback.supports_fp16 = true;
        fallback.supports_bf16 = true;
        return fallback;
    }
    // Prefer discrete GPU if present, otherwise return emulated
    for (const auto& dev : devices) {
        if (dev.backend_type != GpuBackendType::Emulated) {
            return dev;
        }
    }
    return devices.front();
}

} // namespace aabhas::hardware
