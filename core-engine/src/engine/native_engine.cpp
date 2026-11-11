#include "engine/native_engine.hpp"

#include "tensor_ops/layer_streamer.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace aabhas::engine {
namespace {

std::mutex lifecycle_mutex;
std::unique_ptr<aabhas::tensor_ops::LayerStreamer> active_streamer;

bool model_size(const char* path, std::size_t& size) noexcept {
    if (path == nullptr) {
        errno = EINVAL;
        return false;
    }
    struct stat metadata {};
    if (::stat(path, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
        metadata.st_size <= 0) {
        errno = EINVAL;
        return false;
    }
    const auto unsigned_size = static_cast<std::uintmax_t>(metadata.st_size);
    if (unsigned_size > std::numeric_limits<std::size_t>::max()) {
        errno = EOVERFLOW;
        return false;
    }
    size = static_cast<std::size_t>(unsigned_size);
    return true;
}

} // namespace

NativeStatus start(const char* model_path) noexcept {
    try {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (model_path == nullptr || model_path[0] == '\0') {
            errno = EINVAL;
            return NativeStatus::InvalidArgument;
        }
        if (active_streamer != nullptr) return NativeStatus::AlreadyStarted;

        std::size_t file_size = 0;
        if (!model_size(model_path, file_size)) {
            return NativeStatus::StorageOpenFailed;
        }

        auto streamer = std::make_unique<aabhas::tensor_ops::LayerStreamer>();
        std::vector<aabhas::tensor_ops::LayerDescriptor> layers;
        layers.push_back({0, file_size});
        if (!streamer->open(model_path, std::move(layers))) {
            return NativeStatus::StorageOpenFailed;
        }
        if (!streamer->fetch(0)) {
            streamer->close();
            return NativeStatus::MappingFailed;
        }
        active_streamer = std::move(streamer);
        return NativeStatus::Ok;
    } catch (...) {
        return NativeStatus::InternalFailure;
    }
}

NativeStatus stop() noexcept {
    try {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (active_streamer == nullptr) return NativeStatus::NotStarted;
        active_streamer->close();
        active_streamer.reset();
        return NativeStatus::Ok;
    } catch (...) {
        return NativeStatus::InternalFailure;
    }
}

bool is_started() noexcept {
    try {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        return active_streamer != nullptr;
    } catch (...) {
        return false;
    }
}

} // namespace aabhas::engine

extern "C" std::int32_t aabhas_native_start(const char* model_path) {
    return static_cast<std::int32_t>(aabhas::engine::start(model_path));
}

extern "C" std::int32_t aabhas_native_stop() {
    return static_cast<std::int32_t>(aabhas::engine::stop());
}

extern "C" std::int32_t aabhas_native_is_started() {
    return aabhas::engine::is_started() ? 1 : 0;
}
