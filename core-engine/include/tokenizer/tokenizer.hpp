#ifndef AABHAS_TOKENIZER_TOKENIZER_HPP
#define AABHAS_TOKENIZER_TOKENIZER_HPP

#include "model/model_index.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aabhas::tokenizer {

enum class TokenizeError : std::uint8_t {
    None,
    NotConfigured,
    InvalidUtf8,
    InputTooLarge,
    TokenLimit,
    UnknownToken,
    InvalidVocabulary,
};

struct TokenizerConfig {
    std::size_t max_input_bytes{1ULL * 1024ULL * 1024ULL};
    std::size_t max_tokens{1ULL * 1024ULL * 1024ULL};
};

struct TokenSpan {
    std::uint32_t id{0};
    std::size_t byte_offset{0};
    std::size_t byte_length{0};
};

struct TokenizeResult {
    std::vector<TokenSpan> tokens;
    TokenizeError error{TokenizeError::None};

    explicit operator bool() const noexcept { return error == TokenizeError::None; }
};

/** Bounded longest-match tokenizer over a model-provided vocabulary. */
class GreedyTokenizer final {
public:
    GreedyTokenizer() noexcept = default;

    bool configure(const aabhas::model::ModelIndex::TokenizerInfo& vocabulary,
                   TokenizerConfig config = {});
    void clear() noexcept;

    TokenizeResult encode(std::string_view input) const;
    bool decode(const std::vector<std::uint32_t>& tokens, std::string& output) const;

    bool configured() const noexcept { return configured_; }
    const TokenizerConfig& config() const noexcept { return config_; }

private:
    struct TrieNode {
        std::unordered_map<unsigned char, std::size_t> children;
        std::int64_t token_id{-1};
    };

    static bool valid_utf8(std::string_view input) noexcept;
    static std::size_t codepoint_length(std::string_view input,
                                        std::size_t offset) noexcept;
    bool append_token(std::vector<TokenSpan>& output, std::uint32_t id,
                      std::size_t offset, std::size_t length) const;

    TokenizerConfig config_{};
    std::vector<TrieNode> trie_;
    std::unordered_map<std::uint32_t, std::string> pieces_by_id_;
    std::int64_t bos_token_id_{-1};
    std::int64_t eos_token_id_{-1};
    std::int64_t unk_token_id_{-1};
    bool configured_{false};
};

} // namespace aabhas::tokenizer

#endif // AABHAS_TOKENIZER_TOKENIZER_HPP
