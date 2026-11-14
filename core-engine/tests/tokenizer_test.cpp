#include "tokenizer/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    aabhas::model::ModelIndex::TokenizerInfo info;
    info.vocabulary = {"<bos>", "<eos>", "<unk>", "he", "hello", " ", "world"};
    info.bos_token_id = 0;
    info.eos_token_id = 1;
    info.unk_token_id = 2;

    aabhas::tokenizer::GreedyTokenizer tokenizer;
    aabhas::tokenizer::TokenizerConfig config;
    config.max_input_bytes = 64;
    config.max_tokens = 16;
    assert(tokenizer.configure(info, config));

    const auto result = tokenizer.encode("hello world");
    assert(result);
    assert(result.tokens.size() == 5);
    assert(result.tokens[0].id == 0);
    assert(result.tokens[1].id == 4);
    assert(result.tokens[1].byte_offset == 0);
    assert(result.tokens[1].byte_length == 5);
    assert(result.tokens[2].id == 5);
    assert(result.tokens[3].id == 6);
    assert(result.tokens.back().id == 1);

    std::vector<std::uint32_t> ids;
    for (const auto& token : result.tokens) ids.push_back(token.id);
    std::string decoded;
    assert(tokenizer.decode(ids, decoded));
    assert(decoded == "<bos>hello world<eos>");

    const auto unknown = tokenizer.encode("!");
    assert(unknown);
    assert(unknown.tokens.size() == 3);
    assert(unknown.tokens[1].id == 2);
    assert(unknown.tokens[1].byte_length == 1);

    const std::string invalid_utf8("\xc3\x28", 2);
    assert(tokenizer.encode(invalid_utf8).error ==
           aabhas::tokenizer::TokenizeError::InvalidUtf8);

    config.max_tokens = 2;
    assert(tokenizer.configure(info, config));
    assert(tokenizer.encode("hello").error ==
           aabhas::tokenizer::TokenizeError::TokenLimit);
    return 0;
}
