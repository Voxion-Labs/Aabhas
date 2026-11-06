#include "tensor_ops/layer_streamer.hpp"

#include "memory/madvise_utils.hpp"

#include <algorithm>
#include <cerrno>
#include <utility>

namespace aabhas::tensor_ops {

LayerStreamer::~LayerStreamer() noexcept { close(); }

bool LayerStreamer::open(const char* model_path, std::vector<LayerDescriptor> layers,
                         std::size_t memory_cap_bytes) {
    close();
    if (memory_cap_bytes == 0) {
        errno = EINVAL;
        return false;
    }
    if (!source_.open(model_path)) {
        return false;
    }
    layers_ = std::move(layers);
    memory_cap_bytes_ = memory_cap_bytes;
    return true;
}

void LayerStreamer::close() noexcept {
    drop();
    source_.close();
    layers_.clear();
    memory_cap_bytes_ = default_memory_cap;
    current_chunk_offset_ = 0;
}

bool LayerStreamer::fetch(std::size_t layer_index) noexcept {
    if (!source_.is_open() || layer_index >= layers_.size()) {
        errno = EINVAL;
        return false;
    }
    const LayerDescriptor& layer = layers_[layer_index];
    const std::uint64_t model_size = source_.file_size();
    if (layer.byte_size == 0 ||
        layer.file_offset > model_size ||
        static_cast<std::uint64_t>(layer.byte_size) > model_size - layer.file_offset) {
        errno = EINVAL;
        return false;
    }

    const std::size_t bytes = std::min(
        layer.byte_size, std::min(default_chunk_size, memory_cap_bytes_));
    return fetch_chunk(layer_index, 0, bytes);
}

bool LayerStreamer::fetch_chunk(std::size_t layer_index, std::uint64_t chunk_offset,
                                std::size_t chunk_bytes) noexcept {
    if (!source_.is_open() || layer_index >= layers_.size() || chunk_bytes == 0 ||
        chunk_bytes > memory_cap_bytes_) {
        errno = EINVAL;
        return false;
    }
    const LayerDescriptor& layer = layers_[layer_index];
    const std::uint64_t model_size = source_.file_size();
    if (layer.file_offset > model_size ||
        static_cast<std::uint64_t>(layer.byte_size) >
            model_size - layer.file_offset ||
        chunk_offset > static_cast<std::uint64_t>(layer.byte_size) ||
        static_cast<std::uint64_t>(chunk_bytes) >
            static_cast<std::uint64_t>(layer.byte_size) - chunk_offset) {
        errno = EINVAL;
        return false;
    }

    drop();
    const std::uint64_t absolute_offset = layer.file_offset + chunk_offset;
    if (!mapping_.map(source_.fd(), absolute_offset, chunk_bytes)) return false;
    (void)aabhas::memory::advise_sequential(mapping_.mapped_base(), mapping_.mapped_size());
    current_index_ = layer_index;
    current_chunk_offset_ = chunk_offset;
    return true;
}

bool LayerStreamer::fetch_next_chunk(std::size_t chunk_bytes) noexcept {
    if (current_index_ == no_layer) {
        errno = ENOENT;
        return false;
    }
    const LayerDescriptor& layer = layers_[current_index_];
    const std::uint64_t next_offset = current_chunk_offset_ + mapping_.size();
    if (next_offset >= static_cast<std::uint64_t>(layer.byte_size)) {
        errno = ENOENT;
        return false;
    }
    const std::size_t remaining = layer.byte_size - static_cast<std::size_t>(next_offset);
    const std::size_t next_bytes = remaining < chunk_bytes ? remaining : chunk_bytes;
    return fetch_chunk(current_index_, next_offset, next_bytes);
}

bool LayerStreamer::fetch_next() noexcept {
    if (layers_.empty()) {
        errno = ENOENT;
        return false;
    }
    const std::size_t next = current_index_ == no_layer ? 0 : current_index_ + 1;
    if (next >= layers_.size()) {
        errno = ENOENT;
        return false;
    }
    return fetch(next);
}

void LayerStreamer::drop() noexcept {
    (void)aabhas::memory::release_ram(mapping_.mapped_base(), mapping_.mapped_size());
    mapping_.unmap();
    current_index_ = no_layer;
    current_chunk_offset_ = 0;
}

LayerView LayerStreamer::current() const noexcept {
    if (current_index_ == no_layer || !mapping_.mapped()) {
        return {};
    }
    return {static_cast<const std::byte*>(mapping_.data()), mapping_.size(), current_index_};
}

} // namespace aabhas::tensor_ops
