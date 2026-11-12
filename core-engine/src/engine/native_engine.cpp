#include "engine/native_engine.hpp"

#include "model/model_index.hpp"
#include "tensor_ops/layer_streamer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace aabhas::engine {
namespace {

std::mutex lifecycle_mutex;
std::unique_ptr<aabhas::tensor_ops::LayerStreamer> active_streamer;

} // namespace

NativeStatus start(const char* model_path) noexcept {
    try {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (model_path == nullptr || model_path[0] == '\0') {
            return NativeStatus::InvalidArgument;
        }
        if (active_streamer != nullptr) return NativeStatus::AlreadyStarted;

        const aabhas::model::IndexResult model = aabhas::model::inspect(model_path);
        if (!model) {
            return NativeStatus::StorageOpenFailed;
        }

        auto streamer = std::make_unique<aabhas::tensor_ops::LayerStreamer>();
        std::vector<aabhas::tensor_ops::LayerDescriptor> layers;
        layers.reserve(model.index.tensors.size());
        for (const aabhas::model::TensorDescriptor& tensor : model.index.tensors) {
            if (tensor.byte_size == 0 ||
                tensor.byte_size > std::numeric_limits<std::size_t>::max()) {
                return NativeStatus::StorageOpenFailed;
            }
            layers.push_back({tensor.file_offset, static_cast<std::size_t>(tensor.byte_size)});
        }
        if (layers.empty()) return NativeStatus::StorageOpenFailed;
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
