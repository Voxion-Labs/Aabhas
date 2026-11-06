#ifndef AABHAS_MEMORY_MMAP_ILLUSION_HPP
#define AABHAS_MEMORY_MMAP_ILLUSION_HPP

#include <cstddef>
#include <cstdint>
#include <sys/mman.h>

namespace aabhas::memory {

/** Owns one file-backed virtual-memory mapping. */
class MemoryIllusionist final {
public:
    MemoryIllusionist() noexcept = default;
    ~MemoryIllusionist() noexcept;
    MemoryIllusionist(const MemoryIllusionist&) = delete;
    MemoryIllusionist& operator=(const MemoryIllusionist&) = delete;
    MemoryIllusionist(MemoryIllusionist&& other) noexcept;
    MemoryIllusionist& operator=(MemoryIllusionist&& other) noexcept;

    /** Map [file_offset, file_offset + length) from fd. */
    bool map(int fd, std::uint64_t file_offset, std::size_t length,
             int protection = PROT_READ, int flags = MAP_PRIVATE) noexcept;
    void unmap() noexcept;
    /** Discard resident pages while retaining the virtual address range. */
    bool release_ram() noexcept;

    void* data() noexcept { return view_; }
    const void* data() const noexcept { return view_; }
    void* mapped_base() noexcept { return mapping_base_; }
    const void* mapped_base() const noexcept { return mapping_base_; }
    std::size_t size() const noexcept { return view_size_; }
    std::size_t mapped_size() const noexcept { return mapping_size_; }
    bool mapped() const noexcept { return view_ != nullptr; }

private:
    void* mapping_base_{nullptr};
    void* view_{nullptr};
    std::size_t mapping_size_{0};
    std::size_t view_size_{0};
};

} // namespace aabhas::memory

#endif // AABHAS_MEMORY_MMAP_ILLUSION_HPP
