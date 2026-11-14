#include "model/model_index.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void append_u32(std::vector<char>& bytes, std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        bytes.push_back(static_cast<char>((value >> (index * 8U)) & 0xffU));
    }
}

void append_u64(std::vector<char>& bytes, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        bytes.push_back(static_cast<char>((value >> (index * 8U)) & 0xffU));
    }
}

void append_string(std::vector<char>& bytes, const std::string& value) {
    append_u64(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void write_file(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output.is_open());
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    assert(output.good());
}

void test_safetensors(const std::filesystem::path& path) {
    const std::string header =
        R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    std::vector<char> bytes;
    append_u64(bytes, static_cast<std::uint64_t>(header.size()));
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), {0, 0, 0, 0});
    write_file(path, bytes);

    const auto result = aabhas::model::inspect(path.string().c_str());
    assert(result);
    assert(result.index.format == aabhas::model::ModelFormat::SafeTensors);
    assert(result.index.tensors.size() == 1);
    assert(result.index.tensors[0].name == "tensor");
    assert(result.index.tensors[0].file_offset == 8 + header.size());
    assert(result.index.tensors[0].byte_size == 4);
    assert(result.index.tensors[0].dtype == aabhas::model::TensorDType::F32);
    assert(result.index.tensors[0].shape == std::vector<std::uint64_t>{1});
}

void test_tensor_layouts() {
    aabhas::model::TensorLayout layout;
    assert(aabhas::model::describe_tensor_layout(
        aabhas::model::TensorDType::F32, {2, 3}, layout));
    assert(layout.element_count == 6);
    assert(layout.byte_size == 24);

    assert(aabhas::model::describe_tensor_layout(
        aabhas::model::TensorDType::Q4_0, {32}, layout));
    assert(layout.element_count == 32);
    assert(layout.byte_size == 18);
    assert(!aabhas::model::describe_tensor_layout(
        aabhas::model::TensorDType::Q4_0, {31}, layout));
}

void test_safetensors_rejects_invalid_ranges(const std::filesystem::path& path) {
    const std::string header =
        R"({"first":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"second":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})";
    std::vector<char> bytes;
    append_u64(bytes, static_cast<std::uint64_t>(header.size()));
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), 12, 0);
    write_file(path, bytes);

    const auto result = aabhas::model::inspect(path.string().c_str());
    assert(!result);
    assert(result.error == aabhas::model::IndexError::Malformed);
}

void test_gguf(const std::filesystem::path& path) {
    std::vector<char> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    append_u32(bytes, 3);
    append_u64(bytes, 1);
    append_u64(bytes, 4);
    append_string(bytes, "general.alignment");
    append_u32(bytes, 4);
    append_u32(bytes, 32);
    append_string(bytes, "tokenizer.ggml.tokens");
    append_u32(bytes, 9);
    append_u32(bytes, 8);
    append_u64(bytes, 2);
    append_string(bytes, "hello");
    append_string(bytes, "world");
    append_string(bytes, "tokenizer.ggml.bos_token_id");
    append_u32(bytes, 4);
    append_u32(bytes, 0);
    append_string(bytes, "tokenizer.ggml.eos_token_id");
    append_u32(bytes, 4);
    append_u32(bytes, 1);
    append_string(bytes, "tensor");
    append_u32(bytes, 1);
    append_u64(bytes, 1);
    append_u32(bytes, 0);
    append_u64(bytes, 0);
    while (bytes.size() % 32 != 0) bytes.push_back(0);
    const std::size_t data_offset = bytes.size();
    bytes.insert(bytes.end(), {0, 0, 0, 0});
    write_file(path, bytes);

    const auto result = aabhas::model::inspect(path.string().c_str());
    assert(result);
    assert(result.index.format == aabhas::model::ModelFormat::GGUF);
    assert(result.index.tensors.size() == 1);
    assert(result.index.tensors[0].name == "tensor");
    assert(result.index.tensors[0].file_offset == data_offset);
    assert(result.index.tensors[0].byte_size == 4);
    assert(result.index.tensors[0].dtype == aabhas::model::TensorDType::F32);
    assert(result.index.tensors[0].shape == std::vector<std::uint64_t>{1});
    assert(result.index.tokenizer.vocabulary == std::vector<std::string>{"hello", "world"});
    assert(result.index.tokenizer.bos_token_id == 0);
    assert(result.index.tokenizer.eos_token_id == 1);
}

} // namespace

int main() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / "aabhas-model-index-test";
    const auto safe_path = base.string() + ".safetensors";
    const auto gguf_path = base.string() + ".gguf";
    const auto invalid_safe_path = base.string() + ".invalid.safetensors";
    test_tensor_layouts();
    test_safetensors(safe_path);
    test_safetensors_rejects_invalid_ranges(invalid_safe_path);
    test_gguf(gguf_path);
    std::filesystem::remove(safe_path);
    std::filesystem::remove(gguf_path);
    std::filesystem::remove(invalid_safe_path);
    return 0;
}
