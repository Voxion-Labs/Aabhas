#ifndef AABHAS_MODEL_MODEL_INDEX_HPP
#define AABHAS_MODEL_MODEL_INDEX_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace aabhas::model {

enum class ModelFormat : std::uint8_t {
    RawBinary,
    GGUF,
    SafeTensors,
};

enum class IndexError : std::uint8_t {
    None,
    InvalidArgument,
    IoFailure,
    Malformed,
};

struct TensorDescriptor {
    std::string name;
    std::uint64_t file_offset{0};
    std::uint64_t byte_size{0};
};

struct ModelIndex {
    ModelFormat format{ModelFormat::RawBinary};
    std::uint64_t file_size{0};
    std::vector<TensorDescriptor> tensors;
};

struct IndexResult {
    ModelIndex index;
    IndexError error{IndexError::None};

    explicit operator bool() const noexcept { return error == IndexError::None; }
};

/** Inspect model metadata without reading tensor payloads into memory. */
IndexResult inspect(const char* model_path);

} // namespace aabhas::model

#endif // AABHAS_MODEL_MODEL_INDEX_HPP
