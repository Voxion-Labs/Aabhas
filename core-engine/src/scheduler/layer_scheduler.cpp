#include "scheduler/layer_scheduler.hpp"

#include <algorithm>
#include <cerrno>
#include <utility>

namespace aabhas::scheduler {

bool LayerScheduler::open(const char* model_path,
                          std::vector<aabhas::tensor_ops::LayerDescriptor> layers,
                          SchedulerConfig config) {
    close();
    if (config.memory_cap_bytes == 0 || config.chunk_bytes == 0 ||
        config.chunk_bytes > config.memory_cap_bytes || config.lookahead_layers > 64 ||
        layers.empty()) {
        errno = EINVAL;
        return false;
    }
    if (!streamer_.open(model_path, layers, config.memory_cap_bytes)) return false;
    layers_ = std::move(layers);
    config_ = config;
    stats_ = {};
    next_layer_index_ = 0;
    current_active_ = false;
    prefetch_lookahead();
    return true;
}

void LayerScheduler::close() noexcept {
    streamer_.close();
    layers_.clear();
    config_ = {};
    stats_ = {};
    next_layer_index_ = 0;
    current_active_ = false;
}

bool LayerScheduler::next_layer() noexcept {
    if (current_active_) {
        errno = EBUSY;
        return false;
    }
    if (next_layer_index_ >= layers_.size()) {
        errno = ENOENT;
        return false;
    }
    if (!streamer_.fetch(next_layer_index_)) return false;
    current_active_ = true;
    ++stats_.layers_started;
    ++next_layer_index_;
    stats_.next_layer_index = next_layer_index_;
    stats_.resident_bytes = streamer_.current().size;
    prefetch_lookahead();
    return true;
}

bool LayerScheduler::next_chunk() noexcept {
    if (!current_active_) {
        errno = ENOENT;
        return false;
    }
    if (!streamer_.fetch_next_chunk(config_.chunk_bytes)) return false;
    ++stats_.chunks_started;
    stats_.resident_bytes = streamer_.current().size;
    return true;
}

bool LayerScheduler::release() noexcept {
    if (!current_active_) return false;
    streamer_.drop();
    current_active_ = false;
    stats_.resident_bytes = 0;
    return true;
}

std::size_t LayerScheduler::pending_layers() const noexcept {
    return layers_.size() - std::min(next_layer_index_, layers_.size());
}

void LayerScheduler::prefetch_lookahead() noexcept {
    const std::size_t remaining = layers_.size() - std::min(next_layer_index_, layers_.size());
    const std::size_t count = std::min(remaining, config_.lookahead_layers);
    const std::size_t end = next_layer_index_ + count;
    for (std::size_t index = next_layer_index_; index < end; ++index) {
        ++stats_.prefetch_attempts;
        if (streamer_.prefetch(index)) ++stats_.prefetch_successes;
    }
}

} // namespace aabhas::scheduler
