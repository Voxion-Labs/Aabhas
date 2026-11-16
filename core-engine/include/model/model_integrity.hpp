#ifndef AABHAS_MODEL_MODEL_INTEGRITY_HPP
#define AABHAS_MODEL_MODEL_INTEGRITY_HPP

#include "model/model_index.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aabhas::model {

/** SHA-256 digest container (32 bytes raw / 64 hex characters). */
struct Sha256Digest {
    std::uint8_t bytes[32]{};

    std::string to_hex() const;
    static Sha256Digest from_hex(std::string_view hex);
    bool is_zero() const noexcept;
    bool operator==(const Sha256Digest& other) const noexcept;
    bool operator!=(const Sha256Digest& other) const noexcept;
};

/** Streaming SHA-256 state machine conforming to FIPS 180-4. */
class Sha256Context final {
public:
    Sha256Context() noexcept;
    void reset() noexcept;
    void update(const void* data, std::size_t length) noexcept;
    Sha256Digest finalize() noexcept;

    /** Convenience one-shot hashing. */
    static Sha256Digest hash(const void* data, std::size_t length) noexcept;
    static Sha256Digest hash_string(std::string_view text) noexcept;
    static Sha256Digest hash_file(const char* filepath);

private:
    void transform(const std::uint8_t chunk[64]) noexcept;

    std::uint32_t state_[8]{};
    std::uint64_t count_{0};
    std::uint8_t buffer_[64]{};
};

/** Cryptographic model verification manifest. */
struct ModelManifest {
    std::string model_name;
    std::string file_sha256;
    std::uint64_t file_size_bytes{0};
    std::size_t tensor_count{0};
    std::string manifest_digest;
    bool verified{false};

    std::string to_json() const;
};

/** Model integrity verifier ensuring tamper resistance and corruption prevention. */
class ModelIntegrityVerifier final {
public:
    /** Compute structural digest across all tensors in a parsed ModelIndex. */
    static Sha256Digest compute_manifest_digest(const ModelIndex& index) noexcept;

    /** Generate complete ModelManifest including optional file-content SHA-256. */
    static ModelManifest generate_manifest(const ModelIndex& index,
                                           const char* filepath = nullptr);

    /** Verify that a parsed model matches expected structural and content checksums. */
    static bool verify(const ModelIndex& index,
                       const ModelManifest& expected,
                       const char* filepath = nullptr);
};

} // namespace aabhas::model

#endif // AABHAS_MODEL_MODEL_INTEGRITY_HPP
