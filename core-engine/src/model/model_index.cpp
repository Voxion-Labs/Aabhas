#include "model/model_index.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>
#include <utility>

namespace aabhas::model {
namespace {

constexpr std::uint64_t max_header_bytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t max_entries = 1ULL << 20;

bool add_overflow(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t& result) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) return true;
    result = lhs + rhs;
    return false;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, bool& overflow) {
    if (alignment == 0) {
        overflow = true;
        return 0;
    }
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) return value;
    const std::uint64_t increment = alignment - remainder;
    if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
        overflow = true;
        return 0;
    }
    return value + increment;
}

class BinaryReader final {
public:
    explicit BinaryReader(std::ifstream& stream) : stream_(stream) {}

    std::uint64_t position() const noexcept {
        const std::streampos position = stream_.tellg();
        return position == std::streampos(-1) ? 0 : static_cast<std::uint64_t>(position);
    }

    bool read_bytes(char* destination, std::size_t length) {
        stream_.read(destination, static_cast<std::streamsize>(length));
        return stream_.good();
    }

    bool read_u8(std::uint8_t& value) {
        char byte = '\0';
        if (!read_bytes(&byte, 1)) return false;
        value = static_cast<std::uint8_t>(static_cast<unsigned char>(byte));
        return true;
    }

    bool read_u32(std::uint32_t& value) {
        std::uint8_t bytes[4]{};
        if (!read_bytes(reinterpret_cast<char*>(bytes), sizeof(bytes))) return false;
        value = static_cast<std::uint32_t>(bytes[0]) |
                (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                (static_cast<std::uint32_t>(bytes[3]) << 24U);
        return true;
    }

    bool read_u64(std::uint64_t& value) {
        std::uint8_t bytes[8]{};
        if (!read_bytes(reinterpret_cast<char*>(bytes), sizeof(bytes))) return false;
        value = 0;
        for (std::size_t index = 0; index < sizeof(bytes); ++index) {
            value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
        }
        return true;
    }

    bool skip(std::uint64_t length) {
        if (length > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            return false;
        }
        stream_.seekg(static_cast<std::streamoff>(length), std::ios::cur);
        return stream_.good();
    }

private:
    std::ifstream& stream_;
};

bool read_gguf_string(BinaryReader& reader, std::string& value) {
    std::uint64_t length = 0;
    if (!reader.read_u64(length) || length > max_header_bytes) return false;
    value.resize(static_cast<std::size_t>(length));
    return length == 0 || reader.read_bytes(value.data(), value.size());
}

bool skip_gguf_value(BinaryReader& reader, std::uint32_t type,
                     std::uint64_t& alignment) {
    std::uint64_t length = 0;
    std::uint32_t nested_type = 0;
    switch (type) {
    case 0:
    case 1:
    case 7:
        return reader.skip(1);
    case 2:
    case 3:
        return reader.skip(2);
    case 4:
    case 5:
    case 6:
        return reader.skip(4);
    case 8:
        if (!reader.read_u64(length)) return false;
        return length <= max_header_bytes && reader.skip(length);
    case 9:
        if (!reader.read_u32(nested_type) || !reader.read_u64(length) ||
            length > max_entries) {
            return false;
        }
        for (std::uint64_t index = 0; index < length; ++index) {
            if (!skip_gguf_value(reader, nested_type, alignment)) return false;
        }
        return true;
    case 10:
    case 11:
    case 12:
        return reader.skip(8);
    default:
        return false;
    }
}

TensorDType gguf_dtype(std::uint32_t type) {
    switch (type) {
    case 0: return TensorDType::F32;
    case 1: return TensorDType::F16;
    case 2: return TensorDType::Q4_0;
    case 8: return TensorDType::Q8_0;
    default: return TensorDType::Unknown;
    }
}

IndexResult inspect_gguf(std::ifstream& stream, std::uint64_t file_size) {
    BinaryReader reader(stream);
    char magic[4]{};
    if (!reader.read_bytes(magic, sizeof(magic)) || std::memcmp(magic, "GGUF", 4) != 0) {
        return {{}, IndexError::Malformed};
    }

    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (!reader.read_u32(version) || !reader.read_u64(tensor_count) ||
        !reader.read_u64(metadata_count) || (version != 2 && version != 3) ||
        tensor_count == 0 || tensor_count > max_entries || metadata_count > max_entries) {
        return {{}, IndexError::Malformed};
    }

    std::uint64_t alignment = 32;
    for (std::uint64_t index = 0; index < metadata_count; ++index) {
        std::string key;
        std::uint32_t type = 0;
        if (!read_gguf_string(reader, key) || !reader.read_u32(type)) {
            return {{}, IndexError::Malformed};
        }
        if (key == "general.alignment" && (type == 4 || type == 10)) {
            if (type == 4) {
                std::uint32_t value = 0;
                if (!reader.read_u32(value)) return {{}, IndexError::Malformed};
                alignment = value;
            } else {
                if (!reader.read_u64(alignment)) return {{}, IndexError::Malformed};
            }
            if (alignment == 0 || alignment > (1ULL << 20) ||
                (alignment & (alignment - 1)) != 0) {
                return {{}, IndexError::Malformed};
            }
        } else if (!skip_gguf_value(reader, type, alignment)) {
            return {{}, IndexError::Malformed};
        }
    }

    struct PendingTensor {
        std::string name;
        std::uint64_t offset{0};
        std::uint32_t type{0};
        std::vector<std::uint64_t> shape;
    };
    std::vector<PendingTensor> pending;
    pending.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        PendingTensor tensor;
        std::uint32_t dimensions = 0;
        std::uint32_t tensor_type = 0;
        if (!read_gguf_string(reader, tensor.name) || !reader.read_u32(dimensions) ||
            dimensions > 32) {
            return {{}, IndexError::Malformed};
        }
        tensor.shape.reserve(dimensions);
        for (std::uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            std::uint64_t extent = 0;
            if (!reader.read_u64(extent) || extent == 0) return {{}, IndexError::Malformed};
            tensor.shape.push_back(extent);
        }
        if (!reader.read_u32(tensor_type) || tensor_type > 100 ||
            !reader.read_u64(tensor.offset)) {
            return {{}, IndexError::Malformed};
        }
        tensor.type = tensor_type;
        pending.push_back(std::move(tensor));
    }

    bool alignment_overflow = false;
    const std::uint64_t data_start = align_up(reader.position(), alignment, alignment_overflow);
    if (alignment_overflow || data_start > file_size) return {{}, IndexError::Malformed};

    std::sort(pending.begin(), pending.end(),
              [](const PendingTensor& lhs, const PendingTensor& rhs) {
                  return lhs.offset < rhs.offset;
              });
    ModelIndex index;
    index.format = ModelFormat::GGUF;
    index.file_size = file_size;
    index.tensors.reserve(pending.size());
    for (std::size_t item = 0; item < pending.size(); ++item) {
        const std::uint64_t relative_offset = pending[item].offset;
        std::uint64_t absolute_offset = 0;
        if (add_overflow(data_start, relative_offset, absolute_offset) ||
            absolute_offset >= file_size) {
            return {{}, IndexError::Malformed};
        }
        const std::uint64_t end = item + 1 < pending.size()
            ? pending[item + 1].offset
            : file_size - data_start;
        if (end <= relative_offset || end > file_size - data_start) {
            return {{}, IndexError::Malformed};
        }
        index.tensors.push_back({pending[item].name, absolute_offset,
                                 end - relative_offset, gguf_dtype(pending[item].type),
                                 std::move(pending[item].shape)});
    }
    return {std::move(index), IndexError::None};
}

class JsonReader final {
public:
    explicit JsonReader(std::string_view source) : source_(source) {}

    void whitespace() noexcept {
        while (position_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[position_])) != 0) {
            ++position_;
        }
    }

    bool consume(char expected) noexcept {
        whitespace();
        if (position_ >= source_.size() || source_[position_] != expected) return false;
        ++position_;
        return true;
    }

    bool string(std::string& value) {
        whitespace();
        if (position_ >= source_.size() || source_[position_] != '"') return false;
        ++position_;
        value.clear();
        while (position_ < source_.size()) {
            const char character = source_[position_++];
            if (character == '"') return true;
            if (character == '\\') {
                if (position_ >= source_.size()) return false;
                const char escaped = source_[position_++];
                switch (escaped) {
                case '"':
                case '\\':
                case '/': value.push_back(escaped); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: return false;
                }
            } else {
                if (static_cast<unsigned char>(character) < 0x20U) return false;
                value.push_back(character);
            }
        }
        return false;
    }

    bool unsigned_number(std::uint64_t& value) {
        whitespace();
        const std::size_t begin = position_;
        while (position_ < source_.size() &&
               std::isdigit(static_cast<unsigned char>(source_[position_])) != 0) {
            ++position_;
        }
        if (begin == position_) return false;
        value = 0;
        for (std::size_t index = begin; index < position_; ++index) {
            const std::uint64_t digit = static_cast<std::uint64_t>(source_[index] - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) return false;
            value = value * 10 + digit;
        }
        return true;
    }

    bool skip_value() {
        whitespace();
        if (position_ >= source_.size()) return false;
        if (source_[position_] == '"') {
            std::string ignored;
            return string(ignored);
        }
        if (source_[position_] == '{') {
            ++position_;
            whitespace();
            if (consume('}')) return true;
            for (;;) {
                std::string key;
                if (!string(key) || !consume(':') || !skip_value()) return false;
                whitespace();
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (source_[position_] == '[') {
            ++position_;
            whitespace();
            if (consume(']')) return true;
            for (;;) {
                if (!skip_value()) return false;
                whitespace();
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        const std::size_t begin = position_;
        while (position_ < source_.size() && source_[position_] != ',' &&
               source_[position_] != '}' && source_[position_] != ']') {
            ++position_;
        }
        if (begin == position_) return false;
        std::size_t literal_end = position_;
        while (literal_end > begin &&
               std::isspace(static_cast<unsigned char>(source_[literal_end - 1])) != 0) {
            --literal_end;
        }
        if (literal_end == begin) return false;
        const std::string_view literal = source_.substr(begin, literal_end - begin);
        return literal == "true" || literal == "false" || literal == "null" ||
               literal.find_first_not_of("-+.0123456789eE") == std::string_view::npos;
    }

    bool data_offsets(std::uint64_t& begin, std::uint64_t& end) {
        if (!consume('[') || !unsigned_number(begin) || !consume(',') ||
            !unsigned_number(end) || !consume(']')) {
            return false;
        }
        return end >= begin;
    }

    bool unsigned_array(std::vector<std::uint64_t>& values) {
        if (!consume('[')) return false;
        values.clear();
        whitespace();
        if (consume(']')) return true;
        for (;;) {
            std::uint64_t value = 0;
            if (!unsigned_number(value) || value == 0) return false;
            values.push_back(value);
            if (consume(']')) return true;
            if (!consume(',')) return false;
        }
    }

    bool at_end() noexcept {
        whitespace();
        return position_ == source_.size();
    }

private:
    std::string_view source_;
    std::size_t position_{0};
};

TensorDType safetensors_dtype(const std::string& value) {
    if (value == "F32") return TensorDType::F32;
    if (value == "F16") return TensorDType::F16;
    if (value == "BF16") return TensorDType::BF16;
    if (value == "I8") return TensorDType::I8;
    if (value == "U8") return TensorDType::U8;
    return TensorDType::Unknown;
}

bool parse_safetensors_tensor(JsonReader& reader, const std::string& name,
                              std::uint64_t data_base, std::uint64_t file_size,
                              TensorDescriptor& descriptor) {
    if (!reader.consume('{')) return false;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    bool found_offsets = false;
    bool found_dtype = false;
    bool found_shape = false;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    reader.whitespace();
    if (!reader.consume('}')) {
        for (;;) {
            std::string key;
            if (!reader.string(key) || !reader.consume(':')) return false;
            if (key == "data_offsets") {
                if (!reader.data_offsets(begin, end)) return false;
                found_offsets = true;
            } else if (key == "dtype") {
                if (!reader.string(dtype)) return false;
                found_dtype = true;
            } else if (key == "shape") {
                if (!reader.unsigned_array(shape)) return false;
                found_shape = true;
            } else if (!reader.skip_value()) {
                return false;
            }
            if (reader.consume('}')) break;
            if (!reader.consume(',')) return false;
        }
    }
    std::uint64_t absolute_begin = 0;
    std::uint64_t absolute_end = 0;
    if (!found_offsets || !found_dtype || !found_shape || shape.empty() ||
        add_overflow(data_base, begin, absolute_begin) ||
        add_overflow(data_base, end, absolute_end) || absolute_end > file_size ||
        absolute_begin >= absolute_end) {
        return false;
    }
    descriptor = {name, absolute_begin, absolute_end - absolute_begin,
                  safetensors_dtype(dtype), std::move(shape)};
    return true;
}

IndexResult inspect_safetensors(std::ifstream& stream, std::uint64_t file_size) {
    BinaryReader reader(stream);
    std::uint64_t header_size = 0;
    if (!reader.read_u64(header_size) || file_size < 8 || header_size == 0 ||
        header_size > max_header_bytes || header_size > file_size - 8) {
        return {{}, IndexError::Malformed};
    }
    std::string header(static_cast<std::size_t>(header_size), '\0');
    if (!reader.read_bytes(header.data(), header.size())) return {{}, IndexError::Malformed};
    const std::uint64_t data_base = 8 + header_size;
    JsonReader json(header);
    if (!json.consume('{')) return {{}, IndexError::Malformed};

    ModelIndex index;
    index.format = ModelFormat::SafeTensors;
    index.file_size = file_size;
    json.whitespace();
    if (!json.consume('}')) {
        for (;;) {
            std::string name;
            if (!json.string(name) || !json.consume(':')) return {{}, IndexError::Malformed};
            if (name != "__metadata__") {
                TensorDescriptor descriptor;
                if (!parse_safetensors_tensor(json, name, data_base, file_size, descriptor)) {
                    return {{}, IndexError::Malformed};
                }
                index.tensors.push_back(std::move(descriptor));
            } else if (!json.skip_value()) {
                return {{}, IndexError::Malformed};
            }
            if (json.consume('}')) break;
            if (!json.consume(',')) return {{}, IndexError::Malformed};
        }
    }
    if (index.tensors.empty() || !json.at_end()) return {{}, IndexError::Malformed};
    std::sort(index.tensors.begin(), index.tensors.end(),
              [](const TensorDescriptor& lhs, const TensorDescriptor& rhs) {
                  return lhs.file_offset < rhs.file_offset;
              });
    return {std::move(index), IndexError::None};
}

} // namespace

IndexResult inspect(const char* model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return {{}, IndexError::InvalidArgument};
    }
    std::ifstream stream(model_path, std::ios::binary);
    if (!stream.is_open()) return {{}, IndexError::IoFailure};
    stream.seekg(0, std::ios::end);
    const std::streampos end = stream.tellg();
    if (end <= 0) return {{}, IndexError::Malformed};
    const std::uint64_t file_size = static_cast<std::uint64_t>(end);
    stream.seekg(0, std::ios::beg);

    char magic[8]{};
    stream.read(magic, sizeof(magic));
    if (stream.gcount() >= 4 && std::memcmp(magic, "GGUF", 4) == 0) {
        stream.seekg(0, std::ios::beg);
        return inspect_gguf(stream, file_size);
    }
    stream.clear();
    stream.seekg(0, std::ios::beg);
    if (file_size >= 8) {
        std::uint64_t header_size = 0;
        for (std::size_t index = 0; index < sizeof(header_size); ++index) {
            header_size |= static_cast<std::uint64_t>(
                static_cast<unsigned char>(magic[index])) << (index * 8U);
        }
        if (header_size > 0 && header_size <= max_header_bytes &&
            header_size <= file_size - 8) {
            char first_header_byte = '\0';
            stream.clear();
            stream.seekg(8, std::ios::beg);
            stream.read(&first_header_byte, 1);
            if (first_header_byte != '{' && first_header_byte != ' ' &&
                first_header_byte != '\n' && first_header_byte != '\r' &&
                first_header_byte != '\t') {
                header_size = 0;
            }
        }
        if (header_size > 0 && header_size <= max_header_bytes &&
            header_size <= file_size - 8) {
            stream.seekg(0, std::ios::beg);
            auto result = inspect_safetensors(stream, file_size);
            if (result || result.error == IndexError::Malformed) return result;
        }
    }

    ModelIndex raw;
    raw.format = ModelFormat::RawBinary;
    raw.file_size = file_size;
    raw.tensors.push_back({"raw_model", 0, file_size});
    return {std::move(raw), IndexError::None};
}

} // namespace aabhas::model
