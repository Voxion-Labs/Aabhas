#ifndef AABHAS_MEMORY_MADVISE_UTILS_HPP
#define AABHAS_MEMORY_MADVISE_UTILS_HPP

#include <cstddef>

namespace aabhas::memory {

/** Advise the kernel that mapped pages are no longer needed. */
bool release_ram(void* page_aligned_address, std::size_t length) noexcept;

/** Advise the kernel that a mapped range will be consumed sequentially. */
bool advise_sequential(void* page_aligned_address, std::size_t length) noexcept;

} // namespace aabhas::memory

#endif // AABHAS_MEMORY_MADVISE_UTILS_HPP
