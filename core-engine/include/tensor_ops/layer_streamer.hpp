#ifndef AABHAS_TENSOR_OPS_LAYER_STREAMER_HPP
#define AABHAS_TENSOR_OPS_LAYER_STREAMER_HPP

#include "hardware/nvme_direct.hpp"
#include "memory/mmap_illusion.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aabhas::tensor_ops {

struct LayerDescriptor {
    std::uint64_t file_offset{0};
    std::size_t byte_size{0};
};

struct LayerView {
    const std::byte* data{nullptr};
    std::size_t size{0};
    std::size_t index{static_cast<std::size_t>(-1)};
    explicit operator bool() const noexcept { return data != nullptr && size != 0; }
};

/** Presents one file-backed tensor layer at a time without a heap copy. */
class LayerStreamer final {
public:
    static constexpr std::size_t default_memory_cap = 512ULL * 1024ULL * 1024ULL;
    static constexpr std::size_t default_chunk_size = 64ULL * 1024ULL * 1024ULL;

    LayerStreamer() noexcept = default;
    ~LayerStreamer() noexcept;
    LayerStreamer(const LayerStreamer&) = delete;
    LayerStreamer& operator=(const LayerStreamer&) = delete;

    bool open(const char* model_path, std::vector<LayerDescriptor> layers,
              std::size_t memory_cap_bytes = default_memory_cap);
    void close() noexcept;
    bool fetch(std::size_t layer_index) noexcept;
    bool fetch_chunk(std::size_t layer_index, std::uint64_t chunk_offset,
                     std::size_t chunk_bytes) noexcept;
    bool fetch_next_chunk(std::size_t chunk_bytes = default_chunk_size) noexcept;
    bool fetch_next() noexcept;
    /** Issue a bounded kernel read-ahead hint for a future layer. */
    bool prefetch(std::size_t layer_index) const noexcept;
    void drop() noexcept;

    /** The returned view remains valid until the next fetch(), drop(), or close(). */
    LayerView current() const noexcept;
    std::size_t current_index() const noexcept { return current_index_; }
    std::size_t layer_count() const noexcept { return layers_.size(); }
    std::size_t memory_cap() const noexcept { return memory_cap_bytes_; }
    std::uint64_t model_size() const noexcept { return source_.file_size(); }
    bool is_open() const noexcept { return source_.is_open(); }

private:
    static constexpr std::size_t no_layer = static_cast<std::size_t>(-1);
    aabhas::hardware::NvmeDirectReader source_;
    aabhas::memory::MemoryIllusionist mapping_;
    std::vector<LayerDescriptor> layers_;
    std::size_t current_index_{no_layer};
    std::size_t memory_cap_bytes_{default_memory_cap};
    std::uint64_t current_chunk_offset_{0};
};

} // namespace aabhas::tensor_ops

#endif // AABHAS_TENSOR_OPS_LAYER_STREAMER_HPP
