#include "inference/vram_allocator.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace aabhas::inference {
namespace {

void* allocate_aligned(std::size_t bytes, std::size_t alignment) noexcept {
    if (bytes == 0) return nullptr;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);

#if defined(_WIN32)
    return ::_aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, alignment, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void free_aligned(void* ptr) noexcept {
    if (ptr == nullptr) return;
#if defined(_WIN32)
    ::_aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace

DeviceBuffer::DeviceBuffer(VramAllocator* allocator, void* ptr, std::size_t size_bytes) noexcept
    : allocator_(allocator), ptr_(ptr), size_bytes_(size_bytes) {}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : allocator_(other.allocator_), ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
    other.allocator_ = nullptr;
    other.ptr_ = nullptr;
    other.size_bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        ptr_ = other.ptr_;
        size_bytes_ = other.size_bytes_;
        other.allocator_ = nullptr;
        other.ptr_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

void DeviceBuffer::release() noexcept {
    if (ptr_ != nullptr && allocator_ != nullptr) {
        allocator_->deallocate(ptr_, size_bytes_);
    } else if (ptr_ != nullptr) {
        free_aligned(ptr_);
    }
    allocator_ = nullptr;
    ptr_ = nullptr;
    size_bytes_ = 0;
}

bool DeviceBuffer::copy_from_host(const void* host_src, std::size_t bytes,
                                  std::size_t offset) noexcept {
    if (host_src == nullptr || !valid()) return false;
    if (offset > size_bytes_ || bytes > size_bytes_ - offset) return false;
    std::memcpy(static_cast<char*>(ptr_) + offset, host_src, bytes);
    return true;
}

bool DeviceBuffer::copy_to_host(void* host_dst, std::size_t bytes,
                                std::size_t offset) const noexcept {
    if (host_dst == nullptr || !valid()) return false;
    if (offset > size_bytes_ || bytes > size_bytes_ - offset) return false;
    std::memcpy(host_dst, static_cast<const char*>(ptr_) + offset, bytes);
    return true;
}

bool DeviceBuffer::zero(std::size_t bytes, std::size_t offset) noexcept {
    if (!valid()) return false;
    if (bytes == 0) bytes = size_bytes_ - offset;
    if (offset > size_bytes_ || bytes > size_bytes_ - offset) return false;
    std::memset(static_cast<char*>(ptr_) + offset, 0, bytes);
    return true;
}

VramAllocator::VramAllocator(VramConfig config) noexcept
    : config_(config) {
    if (config_.alignment == 0) config_.alignment = 64;
    stats_.budget_bytes = config_.budget_bytes;
}

VramAllocator::~VramAllocator() noexcept = default;

bool VramAllocator::configure(VramConfig config) noexcept {
    if (config.budget_bytes < stats_.allocated_bytes) {
        return false;
    }
    config_ = config;
    if (config_.alignment == 0) config_.alignment = 64;
    stats_.budget_bytes = config_.budget_bytes;
    return true;
}

DeviceBuffer VramAllocator::allocate(std::size_t bytes) noexcept {
    if (bytes == 0) return {};
    if (stats_.allocated_bytes > config_.budget_bytes ||
        bytes > config_.budget_bytes - stats_.allocated_bytes) {
        return {}; // Out of VRAM budget
    }

    void* ptr = allocate_aligned(bytes, config_.alignment);
    if (ptr == nullptr) return {};

    stats_.allocated_bytes += bytes;
    stats_.peak_allocated_bytes =
        std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
    stats_.allocation_count += 1;
    stats_.total_allocations += 1;

    return DeviceBuffer(this, ptr, bytes);
}

void VramAllocator::deallocate(void* ptr, std::size_t bytes) noexcept {
    if (ptr == nullptr) return;
    free_aligned(ptr);
    if (stats_.allocated_bytes >= bytes) {
        stats_.allocated_bytes -= bytes;
    } else {
        stats_.allocated_bytes = 0;
    }
    if (stats_.allocation_count > 0) {
        stats_.allocation_count -= 1;
    }
}

VramStats VramAllocator::stats() const noexcept {
    return stats_;
}

} // namespace aabhas::inference
