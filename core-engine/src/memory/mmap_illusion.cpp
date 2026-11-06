#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "memory/mmap_illusion.hpp"

#include <cerrno>
#include <cstdint>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

namespace aabhas::memory {

MemoryIllusionist::~MemoryIllusionist() noexcept { unmap(); }

MemoryIllusionist::MemoryIllusionist(MemoryIllusionist&& other) noexcept
    : mapping_base_(other.mapping_base_), view_(other.view_),
      mapping_size_(other.mapping_size_), view_size_(other.view_size_) {
    other.mapping_base_ = nullptr;
    other.view_ = nullptr;
    other.mapping_size_ = 0;
    other.view_size_ = 0;
}

MemoryIllusionist& MemoryIllusionist::operator=(MemoryIllusionist&& other) noexcept {
    if (this != &other) {
        unmap();
        mapping_base_ = other.mapping_base_;
        view_ = other.view_;
        mapping_size_ = other.mapping_size_;
        view_size_ = other.view_size_;
        other.mapping_base_ = nullptr;
        other.view_ = nullptr;
        other.mapping_size_ = 0;
        other.view_size_ = 0;
    }
    return *this;
}

bool MemoryIllusionist::map(int fd, std::uint64_t file_offset, std::size_t length,
                            int protection, int flags) noexcept {
    if (fd < 0 || length == 0) {
        errno = EINVAL;
        return false;
    }
    long page_size = 0;
#if defined(_SC_PAGESIZE)
    page_size = ::sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
    page_size = ::sysconf(_SC_PAGE_SIZE);
#endif
    if (page_size <= 0) {
        errno = EINVAL;
        return false;
    }

    const std::uint64_t page = static_cast<std::uint64_t>(page_size);
    const std::uint64_t aligned_offset = file_offset - (file_offset % page);
    const std::uint64_t delta = file_offset - aligned_offset;
    if (delta > std::numeric_limits<std::size_t>::max() ||
        length > std::numeric_limits<std::size_t>::max() - static_cast<std::size_t>(delta) ||
        aligned_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        errno = EOVERFLOW;
        return false;
    }

    const std::size_t mapping_size = length + static_cast<std::size_t>(delta);
    void* base = ::mmap(nullptr, mapping_size, protection, flags, fd,
                        static_cast<off_t>(aligned_offset));
    if (base == MAP_FAILED) {
        return false;
    }
    unmap();
    mapping_base_ = base;
    view_ = static_cast<char*>(base) + static_cast<std::size_t>(delta);
    mapping_size_ = mapping_size;
    view_size_ = length;
    return true;
}

void MemoryIllusionist::unmap() noexcept {
    if (mapping_base_ != nullptr) {
        (void)::munmap(mapping_base_, mapping_size_);
    }
    mapping_base_ = nullptr;
    view_ = nullptr;
    mapping_size_ = 0;
    view_size_ = 0;
}

bool MemoryIllusionist::release_ram() noexcept {
    if (mapping_base_ == nullptr || mapping_size_ == 0) {
        return true;
    }
#if defined(MADV_DONTNEED)
    return ::madvise(mapping_base_, mapping_size_, MADV_DONTNEED) == 0;
#else
    errno = ENOTSUP;
    return false;
#endif
}

} // namespace aabhas::memory
