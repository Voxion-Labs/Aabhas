#include "inference/inference_runtime.hpp"
#include "inference/model_pool.hpp"
#include "model/model_index.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct SyntheticModelContext {
    std::filesystem::path path;
    aabhas::model::ModelIndex model_index;
    aabhas::inference::TransformerModelConfig model_config;
};

SyntheticModelContext create_synthetic_model(const std::string& filename) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::filesystem::path path = dir / filename;

    constexpr std::size_t vocab = 8;
    constexpr std::size_t dim = 4;
    constexpr std::size_t layers = 1;
    constexpr std::size_t heads = 1;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t head_dim = 4;
    constexpr std::size_t ffn_dim = 8;

    std::vector<float> embd(vocab * dim, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i) {
        embd[i * dim + (i % dim)] = 1.0F;
    }
    std::vector<float> norm(dim, 1.0F);

    std::vector<float> eye_dim(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) eye_dim[i * dim + i] = 1.0F;

    std::vector<float> key_val(kv_heads * head_dim * dim, 0.0F);
    for (std::size_t i = 0; i < kv_heads * head_dim; ++i) key_val[i * dim + i] = 0.5F;

    std::vector<float> ffn_gate_up(ffn_dim * dim, 0.0F);
    for (std::size_t i = 0; i < ffn_dim; ++i) ffn_gate_up[i * dim + (i % dim)] = 0.25F;

    std::vector<float> ffn_down(dim * ffn_dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) ffn_down[i * ffn_dim + (i % ffn_dim)] = 0.5F;

    std::vector<float> lm_head(vocab * dim, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i) {
        const std::size_t target_tok = (i + 1) % vocab;
        lm_head[target_tok * dim + (i % dim)] = 5.0F;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out.is_open());
    const auto write_v = [&](const std::vector<float>& v) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(float)));
    };

    std::uint64_t offset = 0;
    aabhas::model::ModelIndex model;
    model.format = aabhas::model::ModelFormat::GGUF;

    const auto add_t = [&](const std::string& name, const std::vector<float>& data,
                           const std::vector<std::uint64_t>& shape) {
        write_v(data);
        const std::size_t sz = data.size() * sizeof(float);
        model.tensors.push_back({name, offset, sz, aabhas::model::TensorDType::F32, shape});
        offset += sz;
    };

    add_t("token_embd.weight", embd, {vocab, dim});
    add_t("blk.0.attn_norm.weight", norm, {dim});
    add_t("blk.0.attn_q.weight", eye_dim, {dim, dim});
    add_t("blk.0.attn_k.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.0.attn_v.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.0.attn_output.weight", eye_dim, {dim, dim});
    add_t("blk.0.ffn_norm.weight", norm, {dim});
    add_t("blk.0.ffn_gate.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.0.ffn_up.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.0.ffn_down.weight", ffn_down, {dim, ffn_dim});
    add_t("output_norm.weight", norm, {dim});
    add_t("output.weight", lm_head, {vocab, dim});
    out.close();

    model.file_size = offset;
    model.tokenizer.vocabulary = {"<unk>", "<bos>", "<eos>", "alpha", "beta", "gamma", "delta", "epsilon"};
    model.tokenizer.bos_token_id = 1;
    model.tokenizer.eos_token_id = 2;
    model.tokenizer.unk_token_id = 0;

    aabhas::inference::TransformerModelConfig model_cfg{};
    model_cfg.vocab_size = vocab;
    model_cfg.model_dimension = dim;
    model_cfg.layer_count = layers;
    model_cfg.block.model_dimension = dim;
    model_cfg.block.head_count = heads;
    model_cfg.block.kv_head_count = kv_heads;
    model_cfg.block.head_dimension = head_dim;
    model_cfg.block.intermediate_dimension = ffn_dim;
    model_cfg.block.rms_epsilon = 1.0e-5F;
    model_cfg.block.rope_theta = 10000.0F;
    model_cfg.bos_token_id = 1;
    model_cfg.eos_token_id = 2;

    return {path, model, model_cfg};
}

void test_model_pool_registration_and_queries() {
    using namespace aabhas::inference;
    auto ctx1 = create_synthetic_model("test_pool_reg_1.bin");
    auto ctx2 = create_synthetic_model("test_pool_reg_2.bin");

    ModelPoolConfig pool_cfg;
    pool_cfg.max_loaded_models = 2;
    ModelPool pool(pool_cfg);

    ModelRegistration reg1;
    reg1.model_id = "model_1";
    reg1.model_path = ctx1.path.string();
    reg1.model_index = ctx1.model_index;
    reg1.model_config = ctx1.model_config;

    ModelRegistration reg2;
    reg2.model_id = "model_2";
    reg2.model_path = ctx2.path.string();
    reg2.model_index = ctx2.model_index;
    reg2.model_config = ctx2.model_config;

    assert(pool.register_model(reg1));
    assert(pool.register_model(reg2));

    // Reject duplicate registration
    assert(!pool.register_model(reg1));

    assert(pool.has_model("model_1"));
    assert(pool.has_model("model_2"));
    assert(!pool.has_model("model_3"));

    assert(!pool.is_model_loaded("model_1"));
    assert(!pool.is_model_loaded("model_2"));

    const auto stats = pool.stats();
    assert(stats.total_registered_models == 2);
    assert(stats.loaded_models == 0);
    assert(stats.active_leases == 0);

    // Unregister non-loaded model
    assert(pool.unregister_model("model_2"));
    assert(!pool.has_model("model_2"));
    assert(!pool.unregister_model("unknown"));

    std::filesystem::remove(ctx1.path);
    std::filesystem::remove(ctx2.path);
}

void test_model_pool_acquire_and_lease() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_pool_acquire.bin");

    ModelPool pool(ModelPoolConfig{2, 1024ULL * 1024ULL * 1024ULL, 512ULL * 1024ULL * 1024ULL});

    ModelRegistration reg;
    reg.model_id = "demo_model";
    reg.model_path = ctx.path.string();
    reg.model_index = ctx.model_index;
    reg.model_config = ctx.model_config;

    assert(pool.register_model(reg));

    {
        ModelLease lease = pool.acquire_model("demo_model");
        assert(lease.valid());
        assert(lease.model_id() == "demo_model");
        assert(lease.runtime() != nullptr);
        assert(pool.is_model_loaded("demo_model"));

        auto stats = pool.stats();
        assert(stats.loaded_models == 1);
        assert(stats.active_leases == 1);
        assert(stats.total_acquisitions == 1);

        // Run generation through acquired model lease
        GenerationOptions opts;
        opts.max_new_tokens = 2;
        GenerationResult res;
        std::vector<std::uint32_t> prompt = {3};
        assert(lease->generate(prompt, opts, res));
        assert(res.generated_tokens.size() == 2);
    }

    // Lease went out of scope: active leases must now be 0, model still cached
    auto stats = pool.stats();
    assert(stats.active_leases == 0);
    assert(stats.loaded_models == 1);
    assert(pool.is_model_loaded("demo_model"));

    // Explicit eviction
    assert(pool.evict_model("demo_model"));
    assert(!pool.is_model_loaded("demo_model"));
    stats = pool.stats();
    assert(stats.loaded_models == 0);
    assert(stats.total_evictions == 1);

    std::filesystem::remove(ctx.path);
}

void test_model_pool_multiple_leases() {
    using namespace aabhas::inference;
    auto ctx = create_synthetic_model("test_pool_multi_lease.bin");

    ModelPool pool(ModelPoolConfig{2, 1024ULL * 1024ULL * 1024ULL, 512ULL * 1024ULL * 1024ULL});

    ModelRegistration reg;
    reg.model_id = "shared_model";
    reg.model_path = ctx.path.string();
    reg.model_index = ctx.model_index;
    reg.model_config = ctx.model_config;
    assert(pool.register_model(reg));

    ModelLease lease1 = pool.acquire_model("shared_model");
    ModelLease lease2 = pool.acquire_model("shared_model");
    assert(lease1.valid() && lease2.valid());
    assert(pool.stats().active_leases == 2);

    // Eviction must fail because active leases > 0
    assert(!pool.evict_model("shared_model"));
    assert(!pool.unregister_model("shared_model"));

    lease1.release();
    assert(!lease1.valid());
    assert(pool.stats().active_leases == 1);
    assert(!pool.evict_model("shared_model")); // Still 1 active lease

    lease2.release();
    assert(!lease2.valid());
    assert(pool.stats().active_leases == 0);

    // Now eviction succeeds
    assert(pool.evict_model("shared_model"));
    assert(!pool.is_model_loaded("shared_model"));

    std::filesystem::remove(ctx.path);
}

void test_model_pool_lru_eviction() {
    using namespace aabhas::inference;
    auto ctx1 = create_synthetic_model("test_pool_lru_1.bin");
    auto ctx2 = create_synthetic_model("test_pool_lru_2.bin");
    auto ctx3 = create_synthetic_model("test_pool_lru_3.bin");

    // Capacity of 2 models max
    ModelPoolConfig cfg;
    cfg.max_loaded_models = 2;
    ModelPool pool(cfg);

    ModelRegistration r1, r2, r3;
    r1.model_id = "m1";
    r1.model_path = ctx1.path.string();
    r1.model_index = ctx1.model_index;
    r1.model_config = ctx1.model_config;

    r2.model_id = "m2";
    r2.model_path = ctx2.path.string();
    r2.model_index = ctx2.model_index;
    r2.model_config = ctx2.model_config;

    r3.model_id = "m3";
    r3.model_path = ctx3.path.string();
    r3.model_index = ctx3.model_index;
    r3.model_config = ctx3.model_config;

    assert(pool.register_model(r1));
    assert(pool.register_model(r2));
    assert(pool.register_model(r3));

    // Load m1
    {
        ModelLease l1 = pool.acquire_model("m1");
        assert(l1.valid());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Load m2
    {
        ModelLease l2 = pool.acquire_model("m2");
        assert(l2.valid());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    assert(pool.is_model_loaded("m1"));
    assert(pool.is_model_loaded("m2"));
    assert(!pool.is_model_loaded("m3"));
    assert(pool.stats().loaded_models == 2);

    // Re-access m2 so m1 is the oldest accessed (LRU)
    {
        ModelLease l2 = pool.acquire_model("m2");
        assert(l2.valid());
    }

    // Now acquire m3: pool must evict m1 and keep m2 and m3
    {
        ModelLease l3 = pool.acquire_model("m3");
        assert(l3.valid());
    }

    assert(!pool.is_model_loaded("m1")); // m1 was evicted
    assert(pool.is_model_loaded("m2"));
    assert(pool.is_model_loaded("m3"));
    assert(pool.stats().loaded_models == 2);
    assert(pool.stats().total_evictions == 1);

    std::filesystem::remove(ctx1.path);
    std::filesystem::remove(ctx2.path);
    std::filesystem::remove(ctx3.path);
}

void test_model_pool_active_lease_protects_against_eviction() {
    using namespace aabhas::inference;
    auto ctx1 = create_synthetic_model("test_pool_protect_1.bin");
    auto ctx2 = create_synthetic_model("test_pool_protect_2.bin");

    // Capacity of 1 model
    ModelPoolConfig cfg;
    cfg.max_loaded_models = 1;
    ModelPool pool(cfg);

    ModelRegistration r1, r2;
    r1.model_id = "m1";
    r1.model_path = ctx1.path.string();
    r1.model_index = ctx1.model_index;
    r1.model_config = ctx1.model_config;

    r2.model_id = "m2";
    r2.model_path = ctx2.path.string();
    r2.model_index = ctx2.model_index;
    r2.model_config = ctx2.model_config;

    assert(pool.register_model(r1));
    assert(pool.register_model(r2));

    ModelLease l1 = pool.acquire_model("m1");
    assert(l1.valid());
    assert(pool.is_model_loaded("m1"));

    // Attempting to acquire m2 while m1 is actively held must fail (cannot evict active lease)
    ModelLease l2 = pool.acquire_model("m2");
    assert(!l2.valid());
    assert(pool.is_model_loaded("m1"));
    assert(!pool.is_model_loaded("m2"));

    // Release l1
    l1.release();
    assert(!l1.valid());

    // Now acquiring m2 must succeed, evicting m1
    l2 = pool.acquire_model("m2");
    assert(l2.valid());
    assert(!pool.is_model_loaded("m1"));
    assert(pool.is_model_loaded("m2"));
    assert(pool.stats().total_evictions == 1);

    std::filesystem::remove(ctx1.path);
    std::filesystem::remove(ctx2.path);
}

} // namespace

int main() {
    test_model_pool_registration_and_queries();
    test_model_pool_acquire_and_lease();
    test_model_pool_multiple_leases();
    test_model_pool_lru_eviction();
    test_model_pool_active_lease_protects_against_eviction();
    return 0;
}
