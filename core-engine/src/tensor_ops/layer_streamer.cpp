#include "tensor_ops/layer_streamer.hpp"

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
}

bool LayerStreamer::fetch(std::size_t layer_index) noexcept {
    if (!source_.is_open() || layer_index >= layers_.size()) {
        errno = EINVAL;
        return false;
    }
    const LayerDescriptor& layer = layers_[layer_index];
    const std::uint64_t model_size = source_.file_size();
    if (layer.byte_size == 0 || layer.byte_size > memory_cap_bytes_ ||
        layer.file_offset > model_size ||
        static_cast<std::uint64_t>(layer.byte_size) > model_size - layer.file_offset) {
        errno = EINVAL;
        return false;
    }

    drop();
    if (!mapping_.map(source_.fd(), layer.file_offset, layer.byte_size)) {
        return false;
    }
    current_index_ = layer_index;
    return true;
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
    (void)mapping_.release_ram();
    mapping_.unmap();
    current_index_ = no_layer;
}

LayerView LayerStreamer::current() const noexcept {
    if (current_index_ == no_layer || !mapping_.mapped()) {
        return {};
    }
    return {static_cast<const std::byte*>(mapping_.data()), mapping_.size(), current_index_};
}

} // namespace aabhas::tensor_ops
