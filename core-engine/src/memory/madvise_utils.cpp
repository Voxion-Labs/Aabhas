#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "memory/madvise_utils.hpp"

#include <cerrno>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>

namespace aabhas::memory {
namespace {

bool advise(void* address, std::size_t length, int advice) noexcept {
    if (address == nullptr || length == 0) {
        errno = EINVAL;
        return false;
    }
    long page_size = 0;
#if defined(_SC_PAGESIZE)
    page_size = ::sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
    page_size = ::sysconf(_SC_PAGE_SIZE);
#endif
    if (page_size <= 0 ||
        (reinterpret_cast<std::uintptr_t>(address) %
         static_cast<std::uintptr_t>(page_size)) != 0U) {
        errno = EINVAL;
        return false;
    }
#if defined(MADV_DONTNEED) || defined(MADV_SEQUENTIAL)
    return ::madvise(address, length, advice) == 0;
#else
    (void)advice;
    errno = ENOTSUP;
    return false;
#endif
}

} // namespace

bool release_ram(void* page_aligned_address, std::size_t length) noexcept {
#if defined(MADV_DONTNEED)
    return advise(page_aligned_address, length, MADV_DONTNEED);
#else
    (void)page_aligned_address;
    (void)length;
    errno = ENOTSUP;
    return false;
#endif
}

bool advise_sequential(void* page_aligned_address, std::size_t length) noexcept {
#if defined(MADV_SEQUENTIAL)
    return advise(page_aligned_address, length, MADV_SEQUENTIAL);
#else
    (void)page_aligned_address;
    (void)length;
    errno = ENOTSUP;
    return false;
#endif
}

} // namespace aabhas::memory
