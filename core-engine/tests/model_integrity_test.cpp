#include "model/model_integrity.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void test_sha256_standard_vectors() {
    using namespace aabhas::model;

    // 1. Empty string test vector
    // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    const Sha256Digest empty_d = Sha256Context::hash_string("");
    assert(empty_d.to_hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    // 2. "abc" NIST test vector
    // SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    const Sha256Digest abc_d = Sha256Context::hash_string("abc");
    assert(abc_d.to_hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // 3. Multi-block message test vector
    // "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    // SHA256 = 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
    const std::string multi_block = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const Sha256Digest multi_d = Sha256Context::hash_string(multi_block);
    assert(multi_d.to_hex() == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // 4. Hex roundtrip
    const Sha256Digest parsed = Sha256Digest::from_hex(abc_d.to_hex());
    assert(parsed == abc_d);
    assert(!parsed.is_zero());

    std::cout << "[PASS] test_sha256_standard_vectors\n";
}

void test_sha256_file_hashing() {
    using namespace aabhas::model;

    const std::filesystem::path temp_file = std::filesystem::temp_directory_path() / "test_aabhas_sha.bin";
    std::ofstream out(temp_file, std::ios::binary | std::ios::trunc);
    assert(out.is_open());
    const std::string content = "Aabhas Enterprise Out-of-Core Tensor Streaming Engine - Phase 10";
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.close();

    const Sha256Digest file_d = Sha256Context::hash_file(temp_file.string().c_str());
    const Sha256Digest string_d = Sha256Context::hash_string(content);

    assert(file_d == string_d);
    assert(!file_d.is_zero());

    std::filesystem::remove(temp_file);
    std::cout << "[PASS] test_sha256_file_hashing: hash=" << file_d.to_hex() << "\n";
}

void test_model_manifest_and_tampering_detection() {
    using namespace aabhas::model;

    // Create synthetic ModelIndex
    ModelIndex index{};
    index.format = ModelFormat::GGUF;
    index.file_size = 1048576;
    index.tensors.push_back({"blk.0.attn_q.weight", 0, 4096, TensorDType::F32, {64, 64}});
    index.tensors.push_back({"blk.0.attn_k.weight", 4096, 4096, TensorDType::F32, {64, 64}});
    index.tensors.push_back({"output_norm.weight", 8192, 256, TensorDType::F32, {64}});

    // 1. Generate Manifest
    ModelManifest manifest = ModelIntegrityVerifier::generate_manifest(index, nullptr);
    assert(!manifest.manifest_digest.empty());
    assert(manifest.tensor_count == 3);
    assert(manifest.file_size_bytes == 1048576);

    // 2. Verification passes on untampered model
    assert(ModelIntegrityVerifier::verify(index, manifest, nullptr));

    // 3. Verification fails if tensor offset is tampered
    ModelIndex tampered_offset = index;
    tampered_offset.tensors[0].file_offset = 128;
    assert(!ModelIntegrityVerifier::verify(tampered_offset, manifest, nullptr));

    // 4. Verification fails if tensor dtype is tampered
    ModelIndex tampered_dtype = index;
    tampered_dtype.tensors[1].dtype = TensorDType::Q4_0;
    assert(!ModelIntegrityVerifier::verify(tampered_dtype, manifest, nullptr));

    // 5. Verification fails if tensor shape is tampered
    ModelIndex tampered_shape = index;
    tampered_shape.tensors[2].shape = {128};
    assert(!ModelIntegrityVerifier::verify(tampered_shape, manifest, nullptr));

    // 6. Verification fails if tensor is missing
    ModelIndex tampered_missing = index;
    tampered_missing.tensors.pop_back();
    assert(!ModelIntegrityVerifier::verify(tampered_missing, manifest, nullptr));

    // 7. Verification fails if format is changed
    ModelIndex tampered_fmt = index;
    tampered_fmt.format = ModelFormat::SafeTensors;
    assert(!ModelIntegrityVerifier::verify(tampered_fmt, manifest, nullptr));

    // 8. Manifest JSON serialization
    std::string json = manifest.to_json();
    assert(json.find("\"manifest_digest\"") != std::string::npos);
    assert(json.find("\"tensor_count\": 3") != std::string::npos);

    std::cout << "[PASS] test_model_manifest_and_tampering_detection\n";
}

void test_end_to_end_file_manifest_verification() {
    using namespace aabhas::model;

    const std::filesystem::path temp_file = std::filesystem::temp_directory_path() / "test_model_manifest_e2e.bin";
    std::ofstream out(temp_file, std::ios::binary | std::ios::trunc);
    assert(out.is_open());
    const std::vector<float> data(1024, 1.23F);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    out.close();

    ModelIndex index{};
    index.format = ModelFormat::GGUF;
    index.file_size = data.size() * sizeof(float);
    index.tensors.push_back({"token_embd.weight", 0, index.file_size, TensorDType::F32, {1024}});

    ModelManifest manifest = ModelIntegrityVerifier::generate_manifest(index, temp_file.string().c_str());
    assert(!manifest.file_sha256.empty());

    // Clean verification
    assert(ModelIntegrityVerifier::verify(index, manifest, temp_file.string().c_str()));

    // Tamper the file content
    std::ofstream tamper_out(temp_file, std::ios::binary | std::ios::in | std::ios::out);
    assert(tamper_out.is_open());
    tamper_out.seekp(0);
    const float bad_val = 999.0F;
    tamper_out.write(reinterpret_cast<const char*>(&bad_val), sizeof(bad_val));
    tamper_out.close();

    // Verification must fail due to corrupted SHA-256
    assert(!ModelIntegrityVerifier::verify(index, manifest, temp_file.string().c_str()));

    std::filesystem::remove(temp_file);
    std::cout << "[PASS] test_end_to_end_file_manifest_verification\n";
}

} // namespace

int main() {
    test_sha256_standard_vectors();
    test_sha256_file_hashing();
    test_model_manifest_and_tampering_detection();
    test_end_to_end_file_manifest_verification();
    std::cout << "All Model Integrity tests passed successfully.\n";
    return 0;
}
