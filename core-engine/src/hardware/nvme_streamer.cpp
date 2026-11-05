#include "hardware/nvme_streamer.hpp"

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace aabhas::hardware {

NvmeStreamer::~NvmeStreamer() noexcept { close(); }

NvmeStreamer::NvmeStreamer(NvmeStreamer&& other) noexcept
    : fd_(other.fd_), file_size_(other.file_size_), position_(other.position_) {
    other.fd_ = -1;
    other.file_size_ = 0;
    other.position_ = 0;
}

NvmeStreamer& NvmeStreamer::operator=(NvmeStreamer&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        file_size_ = other.file_size_;
        position_ = other.position_;
        other.fd_ = -1;
        other.file_size_ = 0;
        other.position_ = 0;
    }
    return *this;
}

bool NvmeStreamer::open(const char* path) noexcept {
    if (path == nullptr) {
        errno = EINVAL;
        return false;
    }
    close();
    int flags = O_RDONLY;
#if defined(O_CLOEXEC)
    flags |= O_CLOEXEC;
#endif
    const int opened = ::open(path, flags);
    if (opened < 0) return false;

    struct stat metadata {};
    if (::fstat(opened, &metadata) != 0) {
        const int saved_errno = errno;
        (void)::close(opened);
        errno = saved_errno;
        return false;
    }
    if (metadata.st_size < 0) {
        (void)::close(opened);
        errno = EINVAL;
        return false;
    }
    fd_ = opened;
    file_size_ = static_cast<std::uint64_t>(metadata.st_size);
    position_ = 0;
    (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
    return true;
}

void NvmeStreamer::close() noexcept {
    if (fd_ >= 0) (void)::close(fd_);
    fd_ = -1;
    file_size_ = 0;
    position_ = 0;
}

ssize_t NvmeStreamer::read(void* destination, std::size_t bytes) noexcept {
    if (fd_ < 0 || (destination == nullptr && bytes != 0)) {
        errno = EINVAL;
        return -1;
    }
    if (bytes == 0) return 0;
    const std::size_t limit = static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
    const std::size_t request = bytes > limit ? limit : bytes;
    for (;;) {
        const ssize_t result = ::read(fd_, destination, request);
        if (result >= 0) {
            position_ += static_cast<std::uint64_t>(result);
            return result;
        }
        if (errno != EINTR) return -1;
    }
}

ssize_t NvmeStreamer::read_at(void* destination, std::size_t bytes,
                              std::uint64_t offset) const noexcept {
    if (fd_ < 0 || (destination == nullptr && bytes != 0)) {
        errno = EINVAL;
        return -1;
    }
    if (bytes == 0) return 0;
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        errno = EOVERFLOW;
        return -1;
    }
    const std::size_t limit = static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
    const std::size_t request = bytes > limit ? limit : bytes;
    for (;;) {
        const ssize_t result = ::pread(fd_, destination, request, static_cast<off_t>(offset));
        if (result >= 0 || errno != EINTR) return result;
    }
}

bool NvmeStreamer::seek(std::uint64_t offset) noexcept {
    if (fd_ < 0) {
        errno = EBADF;
        return false;
    }
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        errno = EOVERFLOW;
        return false;
    }
    if (::lseek(fd_, static_cast<off_t>(offset), SEEK_SET) < 0) return false;
    position_ = offset;
    return true;
}

} // namespace aabhas::hardware
