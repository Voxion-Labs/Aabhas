#ifndef AABHAS_ENGINE_NATIVE_ENGINE_HPP
#define AABHAS_ENGINE_NATIVE_ENGINE_HPP

#include <cstdint>

namespace aabhas::engine {

/** Stable status values returned by the C++ lifecycle boundary. */
enum class NativeStatus : std::int32_t {
    Ok = 0,
    InvalidArgument = -1,
    AlreadyStarted = -2,
    NotStarted = -3,
    StorageOpenFailed = -4,
    MappingFailed = -5,
    InternalFailure = -6,
};

/** Starts and owns one bounded native layer-streaming session. */
NativeStatus start(const char* model_path) noexcept;

/** Stops the active native session and releases its mapping. */
NativeStatus stop() noexcept;

/** Returns whether a native session currently owns the data plane. */
bool is_started() noexcept;

} // namespace aabhas::engine

extern "C" {

std::int32_t aabhas_native_start(const char* model_path);
std::int32_t aabhas_native_stop();
std::int32_t aabhas_native_is_started();

}

#endif // AABHAS_ENGINE_NATIVE_ENGINE_HPP
