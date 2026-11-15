#ifndef AABHAS_INFERENCE_VRAM_ALLOCATOR_HPP
#define AABHAS_INFERENCE_VRAM_ALLOCATOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace aabhas::inference {

class VramAllocator;

/** Configuration parameters for the VRAM memory manager. */
struct VramConfig {
    std::size_t budget_bytes{1024ULL * 1024ULL * 1024ULL}; // 1 GiB default VRAM budget
    std::size_t alignment{64};                            // Coalesced memory alignment
};

/** Tracking telemetry for device memory allocation. */
struct VramStats {
    std::size_t budget_bytes{0};
    std::size_t allocated_bytes{0};
    std::size_t peak_allocated_bytes{0};
    std::size_t allocation_count{0};
    std::uint64_t total_allocations{0};
};

/**
 * RAII handle representing a contiguous slice of device memory (VRAM).
 */
class DeviceBuffer final {
public:
    DeviceBuffer() noexcept = default;
    ~DeviceBuffer() noexcept { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    /** Synchronous Host-to-Device (H2D) memory transfer. */
    bool copy_from_host(const void* host_src, std::size_t bytes, std::size_t offset = 0) noexcept;

    /** Synchronous Device-to-Host (D2H) memory transfer. */
    bool copy_to_host(void* host_dst, std::size_t bytes, std::size_t offset = 0) const noexcept;

    /** In-place memset on device memory. */
    bool zero(std::size_t bytes = 0, std::size_t offset = 0) noexcept;

    void release() noexcept;

    void* raw_ptr() noexcept { return ptr_; }
    const void* raw_ptr() const noexcept { return ptr_; }

    template <typename T>
    T* as() noexcept {
        return static_cast<T*>(ptr_);
    }

    template <typename T>
    const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

    std::size_t size_bytes() const noexcept { return size_bytes_; }
    bool valid() const noexcept { return ptr_ != nullptr && size_bytes_ > 0; }
    explicit operator bool() const noexcept { return valid(); }

private:
    friend class VramAllocator;
    DeviceBuffer(VramAllocator* allocator, void* ptr, std::size_t size_bytes) noexcept;

    VramAllocator* allocator_{nullptr};
    void* ptr_{nullptr};
    std::size_t size_bytes_{0};
};

/**
 * High-performance, zero-fragmentation VRAM / device memory allocator.
 *
 * Enforces strict upper memory bounds on VRAM usage, rejects out-of-budget
 * allocations, and provides alignment for high-throughput SIMD/GPU tensor operations.
 */
class VramAllocator final {
public:
    explicit VramAllocator(VramConfig config = {}) noexcept;
    ~VramAllocator() noexcept;

    VramAllocator(const VramAllocator&) = delete;
    VramAllocator& operator=(const VramAllocator&) = delete;

    /** Reconfigure VRAM allocator bounds. Fails if existing allocations exceed new budget. */
    bool configure(VramConfig config) noexcept;

    /** Allocate a contiguous, aligned device buffer within the VRAM budget. */
    DeviceBuffer allocate(std::size_t bytes) noexcept;

    /** Current telemetry metrics. */
    VramStats stats() const noexcept;

    /** Check if the allocator currently has live allocations. */
    bool has_active_allocations() const noexcept { return stats_.allocated_bytes > 0; }

private:
    friend class DeviceBuffer;
    void deallocate(void* ptr, std::size_t bytes) noexcept;

    VramConfig config_{};
    VramStats stats_{};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_VRAM_ALLOCATOR_HPP
