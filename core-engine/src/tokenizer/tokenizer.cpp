#include "tokenizer/tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace aabhas::tokenizer {
namespace {

bool is_continuation(unsigned char byte) noexcept {
    return (byte & 0xc0U) == 0x80U;
}

} // namespace

bool GreedyTokenizer::configure(
    const aabhas::model::ModelIndex::TokenizerInfo& vocabulary,
    TokenizerConfig config) {
    try {
        clear();
        if (config.max_input_bytes == 0 || config.max_tokens == 0 ||
            vocabulary.vocabulary.empty() ||
            vocabulary.vocabulary.size() > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }

        config_ = config;
        trie_.emplace_back();
        pieces_by_id_.reserve(vocabulary.vocabulary.size());
        for (std::size_t index = 0; index < vocabulary.vocabulary.size(); ++index) {
            const std::string& piece = vocabulary.vocabulary[index];
            if (piece.empty() || index > std::numeric_limits<std::uint32_t>::max()) {
                clear();
                return false;
            }
            const std::uint32_t token_id = static_cast<std::uint32_t>(index);
            if (!pieces_by_id_.emplace(token_id, piece).second) {
                clear();
                return false;
            }

            std::size_t node_index = 0;
            for (const char character : piece) {
                const unsigned char byte = static_cast<unsigned char>(character);
                auto& children = trie_[node_index].children;
                const auto found = children.find(byte);
                if (found != children.end()) {
                    node_index = found->second;
                    continue;
                }
                const std::size_t new_node = trie_.size();
                children.emplace(byte, new_node);
                trie_.emplace_back();
                node_index = new_node;
            }
            if (trie_[node_index].token_id >= 0) {
                clear();
                return false;
            }
            trie_[node_index].token_id = static_cast<std::int64_t>(token_id);
        }

        bos_token_id_ = vocabulary.bos_token_id;
        eos_token_id_ = vocabulary.eos_token_id;
        unk_token_id_ = vocabulary.unk_token_id;
        for (const std::int64_t special_id : {bos_token_id_, eos_token_id_, unk_token_id_}) {
            if (special_id >= 0 &&
                (static_cast<std::uint64_t>(special_id) >= vocabulary.vocabulary.size() ||
                 pieces_by_id_.find(static_cast<std::uint32_t>(special_id)) == pieces_by_id_.end())) {
                clear();
                return false;
            }
        }
        configured_ = true;
        return true;
    } catch (...) {
        clear();
        return false;
    }
}

void GreedyTokenizer::clear() noexcept {
    config_ = {};
    trie_.clear();
    pieces_by_id_.clear();
    bos_token_id_ = -1;
    eos_token_id_ = -1;
    unk_token_id_ = -1;
    configured_ = false;
}

bool GreedyTokenizer::valid_utf8(std::string_view input) noexcept {
    std::size_t offset = 0;
    while (offset < input.size()) {
        const unsigned char first = static_cast<unsigned char>(input[offset]);
        std::size_t length = 0;
        if (first <= 0x7fU) {
            length = 1;
        } else if (first >= 0xc2U && first <= 0xdfU) {
            length = 2;
        } else if (first >= 0xe0U && first <= 0xefU) {
            length = 3;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            length = 4;
        } else {
            return false;
        }
        if (length > input.size() - offset) return false;
        for (std::size_t index = 1; index < length; ++index) {
            if (!is_continuation(static_cast<unsigned char>(input[offset + index]))) {
                return false;
            }
        }
        if (length >= 2) {
            const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
            if ((first == 0xe0U && second < 0xa0U) ||
                (first == 0xedU && second >= 0xa0U) ||
                (first == 0xf0U && second < 0x90U) ||
                (first == 0xf4U && second >= 0x90U)) {
                return false;
            }
        }
        offset += length;
    }
    return true;
}

std::size_t GreedyTokenizer::codepoint_length(std::string_view input,
                                              std::size_t offset) noexcept {
    if (offset >= input.size()) return 0;
    const unsigned char first = static_cast<unsigned char>(input[offset]);
    if (first <= 0x7fU) return 1;
    if (first <= 0xdfU) return 2;
    if (first <= 0xefU) return 3;
    return 4;
}

bool GreedyTokenizer::append_token(std::vector<TokenSpan>& output, std::uint32_t id,
                                   std::size_t offset, std::size_t length) const {
    if (output.size() >= config_.max_tokens) return false;
    output.push_back({id, offset, length});
    return true;
}

TokenizeResult GreedyTokenizer::encode(std::string_view input) const {
    TokenizeResult result;
    if (!configured_) {
        result.error = TokenizeError::NotConfigured;
        return result;
    }
    if (input.size() > config_.max_input_bytes) {
        result.error = TokenizeError::InputTooLarge;
        return result;
    }
    if (!valid_utf8(input)) {
        result.error = TokenizeError::InvalidUtf8;
        return result;
    }

    try {
        const std::size_t reserve_limit =
            input.size() > std::numeric_limits<std::size_t>::max() - 2
                ? std::numeric_limits<std::size_t>::max()
                : input.size() + 2;
        result.tokens.reserve(std::min(config_.max_tokens, reserve_limit));
        if (bos_token_id_ >= 0 && !append_token(result.tokens,
                                                   static_cast<std::uint32_t>(bos_token_id_), 0, 0)) {
            result.error = TokenizeError::TokenLimit;
            return result;
        }

        std::size_t offset = 0;
        while (offset < input.size()) {
            std::size_t node_index = 0;
            std::size_t best_end = offset;
            std::int64_t best_id = -1;
            for (std::size_t cursor = offset; cursor < input.size(); ++cursor) {
                const unsigned char byte = static_cast<unsigned char>(input[cursor]);
                const auto child = trie_[node_index].children.find(byte);
                if (child == trie_[node_index].children.end()) break;
                node_index = child->second;
                if (trie_[node_index].token_id >= 0) {
                    best_id = trie_[node_index].token_id;
                    best_end = cursor + 1;
                }
            }
            if (best_id < 0) {
                if (unk_token_id_ < 0) {
                    result.error = TokenizeError::UnknownToken;
                    result.tokens.clear();
                    return result;
                }
                const std::size_t length = codepoint_length(input, offset);
                if (!append_token(result.tokens, static_cast<std::uint32_t>(unk_token_id_),
                                  offset, length)) {
                    result.error = TokenizeError::TokenLimit;
                    result.tokens.clear();
                    return result;
                }
                offset += length;
                continue;
            }
            if (!append_token(result.tokens, static_cast<std::uint32_t>(best_id), offset,
                              best_end - offset)) {
                result.error = TokenizeError::TokenLimit;
                result.tokens.clear();
                return result;
            }
            offset = best_end;
        }

        if (eos_token_id_ >= 0 && !append_token(result.tokens,
                                                   static_cast<std::uint32_t>(eos_token_id_),
                                                   input.size(), 0)) {
            result.error = TokenizeError::TokenLimit;
            result.tokens.clear();
        }
        return result;
    } catch (...) {
        result.tokens.clear();
        result.error = TokenizeError::InvalidVocabulary;
        return result;
    }
}

bool GreedyTokenizer::decode(const std::vector<std::uint32_t>& tokens,
                             std::string& output) const {
    if (!configured_ || tokens.size() > config_.max_tokens) return false;
    try {
        output.clear();
        for (const std::uint32_t token : tokens) {
            const auto piece = pieces_by_id_.find(token);
            if (piece == pieces_by_id_.end() ||
                piece->second.size() > config_.max_input_bytes -
                                          std::min(output.size(), config_.max_input_bytes)) {
                output.clear();
                return false;
            }
            output += piece->second;
        }
        return true;
    } catch (...) {
        output.clear();
        return false;
    }
}

} // namespace aabhas::tokenizer
