#ifndef AABHAS_HARDWARE_NVME_STREAMER_HPP
#define AABHAS_HARDWARE_NVME_STREAMER_HPP

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace aabhas::hardware {

/** Sequential reader for large regular files. */
class NvmeStreamer final {
public:
    NvmeStreamer() noexcept = default;
    ~NvmeStreamer() noexcept;
    NvmeStreamer(const NvmeStreamer&) = delete;
    NvmeStreamer& operator=(const NvmeStreamer&) = delete;
    NvmeStreamer(NvmeStreamer&& other) noexcept;
    NvmeStreamer& operator=(NvmeStreamer&& other) noexcept;

    bool open(const char* path) noexcept;
    void close() noexcept;
    /** Read the next bytes, returning 0 at EOF or -1 on error. */
    ssize_t read(void* destination, std::size_t bytes) noexcept;
    /** Positional read that does not change the sequential cursor. */
    ssize_t read_at(void* destination, std::size_t bytes,
                    std::uint64_t offset) const noexcept;
    bool seek(std::uint64_t offset) noexcept;
    bool rewind() noexcept { return seek(0); }

    int fd() const noexcept { return fd_; }
    std::uint64_t file_size() const noexcept { return file_size_; }
    std::uint64_t position() const noexcept { return position_; }
    bool is_open() const noexcept { return fd_ >= 0; }

private:
    int fd_{-1};
    std::uint64_t file_size_{0};
    std::uint64_t position_{0};
};

} // namespace aabhas::hardware

#endif // AABHAS_HARDWARE_NVME_STREAMER_HPP
