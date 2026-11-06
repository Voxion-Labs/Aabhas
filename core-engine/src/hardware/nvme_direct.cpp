#include "hardware/nvme_direct.hpp"

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace aabhas::hardware {

NvmeDirectReader::~NvmeDirectReader() noexcept { close(); }

NvmeDirectReader::NvmeDirectReader(NvmeDirectReader&& other) noexcept
    : fd_(other.fd_), file_size_(other.file_size_), position_(other.position_),
      alignment_(other.alignment_), direct_io_enabled_(other.direct_io_enabled_) {
    other.fd_ = -1;
    other.file_size_ = 0;
    other.position_ = 0;
    other.direct_io_enabled_ = false;
}

NvmeDirectReader& NvmeDirectReader::operator=(NvmeDirectReader&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        file_size_ = other.file_size_;
        position_ = other.position_;
        alignment_ = other.alignment_;
        direct_io_enabled_ = other.direct_io_enabled_;
        other.fd_ = -1;
        other.file_size_ = 0;
        other.position_ = 0;
        other.direct_io_enabled_ = false;
    }
    return *this;
}

bool NvmeDirectReader::open(const char* path) noexcept {
    if (path == nullptr) {
        errno = EINVAL;
        return false;
    }
    close();

    int flags = O_RDONLY;
#if defined(O_CLOEXEC)
    flags |= O_CLOEXEC;
#endif

    int opened = -1;
#if defined(O_DIRECT)
    opened = ::open(path, flags | O_DIRECT);
    if (opened >= 0) {
        direct_io_enabled_ = true;
    } else if (errno == EINVAL || errno == ENOTSUP || errno == EOPNOTSUPP) {
        opened = ::open(path, flags);
    }
#else
    opened = ::open(path, flags);
#endif
    if (opened < 0) return false;

    struct stat metadata {};
    if (::fstat(opened, &metadata) != 0 || metadata.st_size < 0) {
        const int saved_errno = errno == 0 ? EINVAL : errno;
        (void)::close(opened);
        errno = saved_errno;
        return false;
    }

    fd_ = opened;
    file_size_ = static_cast<std::uint64_t>(metadata.st_size);
    position_ = 0;
    (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#if defined(POSIX_FADV_NOREUSE)
    (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_NOREUSE);
#endif
    return true;
}

void NvmeDirectReader::close() noexcept {
    if (fd_ >= 0) (void)::close(fd_);
    fd_ = -1;
    file_size_ = 0;
    position_ = 0;
    direct_io_enabled_ = false;
}

ssize_t NvmeDirectReader::read(void* destination, std::size_t bytes) noexcept {
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

ssize_t NvmeDirectReader::read_at(void* destination, std::size_t bytes,
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

bool NvmeDirectReader::seek(std::uint64_t offset) noexcept {
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
