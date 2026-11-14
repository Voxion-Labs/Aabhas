#include "inference/inference_runtime.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "aabhas-inference-runtime-test.bin";
    const float weights[] = {1.0F, 2.0F, 3.0F, 4.0F};
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output.is_open());
    output.write(reinterpret_cast<const char*>(weights), sizeof(weights));
    output.close();

    aabhas::model::ModelIndex model;
    model.file_size = sizeof(weights);
    model.tensors.push_back({"linear", 0, sizeof(weights),
                             aabhas::model::TensorDType::F32, {2, 2}});
    model.tokenizer.vocabulary = {"<unk>", "hello"};
    model.tokenizer.unk_token_id = 0;

    aabhas::inference::InferenceRuntime runtime;
    aabhas::inference::InferenceConfig config;
    config.scheduler.memory_cap_bytes = sizeof(float) * 2;
    config.scheduler.chunk_bytes = sizeof(float) * 2;
    config.scheduler.lookahead_layers = 0;
    assert(runtime.open(path.string().c_str(), model, config));

    const float input[] = {1.0F, 1.0F};
    float result[] = {0.0F, 0.0F};
    assert(runtime.run_f32_matvec(0, input, 2, result, 2));
    assert(result[0] == 3.0F);
    assert(result[1] == 7.0F);

    assert(runtime.configure_kv_cache(1, 1, 2, 4));
    const float key[] = {5.0F, 6.0F};
    const float value[] = {7.0F, 8.0F};
    assert(runtime.kv_cache().append(0, 2, key, value));
    assert(runtime.kv_cache().key(0, 2)[1] == 6.0F);
    assert(runtime.kv_cache().value(0, 2)[0] == 7.0F);
    assert(runtime.kv_cache().bytes() == 64);

    aabhas::tokenizer::TokenizerConfig tokenizer_config;
    tokenizer_config.max_input_bytes = 32;
    tokenizer_config.max_tokens = 8;
    assert(runtime.configure_tokenizer(tokenizer_config));
    const auto tokenized = runtime.tokenize("hello");
    assert(tokenized);
    assert(tokenized.tokens.size() == 1);
    assert(tokenized.tokens[0].id == 1);
    std::vector<std::uint32_t> token_ids{1};
    std::string detokenized;
    assert(runtime.detokenize(token_ids, detokenized));
    assert(detokenized == "hello");

    runtime.close();
    std::filesystem::remove(path);
    return 0;
}
