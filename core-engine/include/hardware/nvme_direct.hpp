#ifndef AABHAS_HARDWARE_NVME_DIRECT_HPP
#define AABHAS_HARDWARE_NVME_DIRECT_HPP

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace aabhas::hardware {

/** Owns a sequential, optionally O_DIRECT-backed tensor file descriptor. */
class NvmeDirectReader final {
public:
    NvmeDirectReader() noexcept = default;
    ~NvmeDirectReader() noexcept;

    NvmeDirectReader(const NvmeDirectReader&) = delete;
    NvmeDirectReader& operator=(const NvmeDirectReader&) = delete;
    NvmeDirectReader(NvmeDirectReader&& other) noexcept;
    NvmeDirectReader& operator=(NvmeDirectReader&& other) noexcept;

    bool open(const char* path) noexcept;
    void close() noexcept;

    /** Read into an O_DIRECT-compatible buffer, returning 0 at EOF. */
    ssize_t read(void* destination, std::size_t bytes) noexcept;
    /** Positional read; callers must satisfy direct-I/O alignment requirements. */
    ssize_t read_at(void* destination, std::size_t bytes,
                    std::uint64_t offset) const noexcept;
    /** Ask the kernel to read a file range ahead without allocating a user buffer. */
    bool prefetch(std::uint64_t offset, std::size_t bytes) const noexcept;
    bool seek(std::uint64_t offset) noexcept;

    int fd() const noexcept { return fd_; }
    std::uint64_t file_size() const noexcept { return file_size_; }
    std::uint64_t position() const noexcept { return position_; }
    std::size_t alignment() const noexcept { return alignment_; }
    bool direct_io_enabled() const noexcept { return direct_io_enabled_; }
    bool is_open() const noexcept { return fd_ >= 0; }

private:
    int fd_{-1};
    std::uint64_t file_size_{0};
    std::uint64_t position_{0};
    std::size_t alignment_{4096};
    bool direct_io_enabled_{false};
};

} // namespace aabhas::hardware

#endif // AABHAS_HARDWARE_NVME_DIRECT_HPP
