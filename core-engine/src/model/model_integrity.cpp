#include "model/model_integrity.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aabhas::model {

namespace {

// FIPS 180-4 initial hash values
constexpr std::array<std::uint32_t, 8> SHA256_INIT = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
};

// FIPS 180-4 round constants
constexpr std::array<std::uint32_t, 64> SHA256_K = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept {
    return (x >> n) | (x << (32U - n));
}

inline std::uint32_t choose(std::uint32_t e, std::uint32_t f, std::uint32_t g) noexcept {
    return (e & f) ^ (~e & g);
}

inline std::uint32_t majority(std::uint32_t a, std::uint32_t b, std::uint32_t c) noexcept {
    return (a & b) ^ (a & c) ^ (b & c);
}

inline std::uint32_t sigma0(std::uint32_t x) noexcept {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline std::uint32_t sigma1(std::uint32_t x) noexcept {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline std::uint32_t gamma0(std::uint32_t x) noexcept {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline std::uint32_t gamma1(std::uint32_t x) noexcept {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

inline std::uint8_t hex_char_to_nibble(char c) noexcept {
    if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(c - 'A' + 10);
    return 0xFFU;
}

} // namespace

std::string Sha256Digest::to_hex() const {
    static constexpr char HEX_DIGITS[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (std::size_t i = 0; i < 32; ++i) {
        hex.push_back(HEX_DIGITS[(bytes[i] >> 4) & 0x0FU]);
        hex.push_back(HEX_DIGITS[bytes[i] & 0x0FU]);
    }
    return hex;
}

Sha256Digest Sha256Digest::from_hex(std::string_view hex) {
    Sha256Digest digest{};
    if (hex.size() != 64) return digest;
    for (std::size_t i = 0; i < 32; ++i) {
        const std::uint8_t hi = hex_char_to_nibble(hex[i * 2]);
        const std::uint8_t lo = hex_char_to_nibble(hex[i * 2 + 1]);
        if (hi > 0x0FU || lo > 0x0FU) {
            std::memset(digest.bytes, 0, sizeof(digest.bytes));
            return digest;
        }
        digest.bytes[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return digest;
}

bool Sha256Digest::is_zero() const noexcept {
    for (const std::uint8_t b : bytes) {
        if (b != 0) return false;
    }
    return true;
}

bool Sha256Digest::operator==(const Sha256Digest& other) const noexcept {
    return std::memcmp(bytes, other.bytes, sizeof(bytes)) == 0;
}

bool Sha256Digest::operator!=(const Sha256Digest& other) const noexcept {
    return !(*this == other);
}

Sha256Context::Sha256Context() noexcept {
    reset();
}

void Sha256Context::reset() noexcept {
    std::memcpy(state_, SHA256_INIT.data(), sizeof(state_));
    count_ = 0;
    std::memset(buffer_, 0, sizeof(buffer_));
}

void Sha256Context::transform(const std::uint8_t chunk[64]) noexcept {
    std::uint32_t w[64];
    for (std::size_t i = 0; i < 16; ++i) {
        const std::size_t idx = i * 4;
        w[i] = (static_cast<std::uint32_t>(chunk[idx]) << 24) |
               (static_cast<std::uint32_t>(chunk[idx + 1]) << 16) |
               (static_cast<std::uint32_t>(chunk[idx + 2]) << 8) |
               static_cast<std::uint32_t>(chunk[idx + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + sigma1(e) + choose(e, f, g) + SHA256_K[i] + w[i];
        const std::uint32_t t2 = sigma0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256Context::update(const void* data, std::size_t length) noexcept {
    if (data == nullptr || length == 0) return;
    const auto* ptr = static_cast<const std::uint8_t*>(data);

    std::size_t buffer_index = static_cast<std::size_t>(count_ & 0x3FU);
    count_ += static_cast<std::uint64_t>(length);

    // If existing buffer has data, fill it first
    if (buffer_index > 0) {
        const std::size_t to_fill = 64 - buffer_index;
        if (length < to_fill) {
            std::memcpy(buffer_ + buffer_index, ptr, length);
            return;
        }
        std::memcpy(buffer_ + buffer_index, ptr, to_fill);
        transform(buffer_);
        ptr += to_fill;
        length -= to_fill;
        buffer_index = 0;
    }

    // Process 64-byte chunks directly
    while (length >= 64) {
        transform(ptr);
        ptr += 64;
        length -= 64;
    }

    // Remaining residual bytes
    if (length > 0) {
        std::memcpy(buffer_, ptr, length);
    }
}

Sha256Digest Sha256Context::finalize() noexcept {
    const std::uint64_t bit_count = count_ * 8ULL;
    const std::size_t buffer_index = static_cast<std::size_t>(count_ & 0x3FU);

    // Append 0x80 bit
    buffer_[buffer_index] = 0x80U;
    const std::size_t padded = buffer_index + 1;

    if (padded > 56) {
        std::memset(buffer_ + padded, 0, 64 - padded);
        transform(buffer_);
        std::memset(buffer_, 0, 56);
    } else {
        std::memset(buffer_ + padded, 0, 56 - padded);
    }

    // Append 64-bit big-endian length
    for (std::size_t i = 0; i < 8; ++i) {
        const std::size_t shift = (7 - i) * 8;
        buffer_[56 + i] = static_cast<std::uint8_t>((bit_count >> shift) & 0xFFU);
    }
    transform(buffer_);

    Sha256Digest digest{};
    for (std::size_t i = 0; i < 8; ++i) {
        const std::size_t out_idx = i * 4;
        digest.bytes[out_idx]     = static_cast<std::uint8_t>((state_[i] >> 24) & 0xFFU);
        digest.bytes[out_idx + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xFFU);
        digest.bytes[out_idx + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xFFU);
        digest.bytes[out_idx + 3] = static_cast<std::uint8_t>(state_[i] & 0xFFU);
    }
    reset();
    return digest;
}

Sha256Digest Sha256Context::hash(const void* data, std::size_t length) noexcept {
    Sha256Context ctx;
    ctx.update(data, length);
    return ctx.finalize();
}

Sha256Digest Sha256Context::hash_string(std::string_view text) noexcept {
    return hash(text.data(), text.size());
}

Sha256Digest Sha256Context::hash_file(const char* filepath) {
    Sha256Digest zero{};
    if (filepath == nullptr || filepath[0] == '\0') return zero;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return zero;

    Sha256Context ctx;
    constexpr std::size_t CHUNK_SIZE = 64ULL * 1024ULL; // 64 KiB
    std::vector<char> buffer(CHUNK_SIZE);

    while (file.good()) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize bytes_read = file.gcount();
        if (bytes_read > 0) {
            ctx.update(buffer.data(), static_cast<std::size_t>(bytes_read));
        }
    }
    return ctx.finalize();
}

std::string ModelManifest::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"model_name\": \"" << model_name << "\",\n";
    oss << "  \"file_sha256\": \"" << file_sha256 << "\",\n";
    oss << "  \"file_size_bytes\": " << file_size_bytes << ",\n";
    oss << "  \"tensor_count\": " << tensor_count << ",\n";
    oss << "  \"manifest_digest\": \"" << manifest_digest << "\",\n";
    oss << "  \"verified\": " << (verified ? "true" : "false") << "\n";
    oss << "}";
    return oss.str();
}

Sha256Digest ModelIntegrityVerifier::compute_manifest_digest(const ModelIndex& index) noexcept {
    Sha256Context ctx;

    // Stream format identifier
    const auto fmt = static_cast<std::uint32_t>(index.format);
    ctx.update(&fmt, sizeof(fmt));

    // Sort tensor descriptors deterministically by name
    std::vector<TensorDescriptor> sorted = index.tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorDescriptor& a, const TensorDescriptor& b) {
        return a.name < b.name;
    });

    for (const auto& t : sorted) {
        ctx.update(t.name.data(), t.name.size());
        ctx.update(&t.file_offset, sizeof(t.file_offset));
        ctx.update(&t.byte_size, sizeof(t.byte_size));
        const auto dt = static_cast<std::uint32_t>(t.dtype);
        ctx.update(&dt, sizeof(dt));
        for (const auto dim : t.shape) {
            ctx.update(&dim, sizeof(dim));
        }
    }

    return ctx.finalize();
}

ModelManifest ModelIntegrityVerifier::generate_manifest(const ModelIndex& index,
                                                       const char* filepath) {
    ModelManifest manifest{};
    manifest.model_name = filepath != nullptr ? filepath : "in-memory-model";
    manifest.file_size_bytes = index.file_size;
    manifest.tensor_count = index.tensors.size();
    manifest.manifest_digest = compute_manifest_digest(index).to_hex();

    if (filepath != nullptr && filepath[0] != '\0') {
        const Sha256Digest file_hash = Sha256Context::hash_file(filepath);
        manifest.file_sha256 = file_hash.to_hex();
    }
    manifest.verified = true;
    return manifest;
}

bool ModelIntegrityVerifier::verify(const ModelIndex& index,
                                   const ModelManifest& expected,
                                   const char* filepath) {
    if (expected.manifest_digest.empty()) return false;

    // 1. Verify structural manifest digest
    const std::string computed_manifest = compute_manifest_digest(index).to_hex();
    if (computed_manifest != expected.manifest_digest) {
        return false;
    }

    // 2. Verify tensor count and file size if recorded
    if (expected.tensor_count > 0 && index.tensors.size() != expected.tensor_count) {
        return false;
    }
    if (expected.file_size_bytes > 0 && index.file_size != expected.file_size_bytes) {
        return false;
    }

    // 3. Verify file hash if present and filepath provided
    if (!expected.file_sha256.empty() && filepath != nullptr && filepath[0] != '\0') {
        const Sha256Digest file_hash = Sha256Context::hash_file(filepath);
        if (file_hash.to_hex() != expected.file_sha256) {
            return false;
        }
    }

    return true;
}

} // namespace aabhas::model
