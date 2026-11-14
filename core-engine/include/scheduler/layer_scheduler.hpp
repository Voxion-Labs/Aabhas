#ifndef AABHAS_SCHEDULER_LAYER_SCHEDULER_HPP
#define AABHAS_SCHEDULER_LAYER_SCHEDULER_HPP

#include "tensor_ops/layer_streamer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aabhas::scheduler {

struct SchedulerConfig {
    std::size_t memory_cap_bytes{aabhas::tensor_ops::LayerStreamer::default_memory_cap};
    std::size_t chunk_bytes{aabhas::tensor_ops::LayerStreamer::default_chunk_size};
    std::size_t lookahead_layers{1};
};

struct SchedulerStats {
    std::size_t next_layer_index{0};
    std::size_t resident_bytes{0};
    std::uint64_t layers_started{0};
    std::uint64_t chunks_started{0};
    std::uint64_t prefetch_attempts{0};
    std::uint64_t prefetch_successes{0};
};

/** Serializes bounded layer windows and non-resident kernel read-ahead hints. */
class LayerScheduler final {
public:
    LayerScheduler() noexcept = default;
    ~LayerScheduler() noexcept = default;
    LayerScheduler(const LayerScheduler&) = delete;
    LayerScheduler& operator=(const LayerScheduler&) = delete;

    bool open(const char* model_path, std::vector<aabhas::tensor_ops::LayerDescriptor> layers,
              SchedulerConfig config = {});
    void close() noexcept;

    /** Start the next layer; the previous window must have been released. */
    bool next_layer() noexcept;
    /** Advance the active layer by one bounded chunk. */
    bool next_chunk() noexcept;
    /** Release the active mapping and its resident pages. */
    bool release() noexcept;
    /** Restart the layer sequence after the active window has been released. */
    bool rewind() noexcept;

    aabhas::tensor_ops::LayerView current() const noexcept { return streamer_.current(); }
    bool has_current() const noexcept { return current_active_; }
    std::size_t layer_count() const noexcept { return layers_.size(); }
    std::size_t pending_layers() const noexcept;
    SchedulerStats stats() const noexcept { return stats_; }

private:
    void prefetch_lookahead() noexcept;

    aabhas::tensor_ops::LayerStreamer streamer_;
    std::vector<aabhas::tensor_ops::LayerDescriptor> layers_;
    SchedulerConfig config_{};
    SchedulerStats stats_{};
    std::size_t next_layer_index_{0};
    bool current_active_{false};
};

} // namespace aabhas::scheduler

#endif // AABHAS_SCHEDULER_LAYER_SCHEDULER_HPP
