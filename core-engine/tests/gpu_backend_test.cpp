#include "hardware/gpu_device.hpp"
#include "inference/gpu_backend.hpp"
#include "inference/inference_runtime.hpp"
#include "inference/vram_allocator.hpp"
#include "model/model_index.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

bool near(float a, float b, float tol = 1.0e-3F) {
    return std::fabs(a - b) <= tol;
}

void test_gpu_device_enumeration() {
    using namespace aabhas::hardware;
    auto devices = GpuDeviceRegistry::enumerate_devices();
    assert(!devices.empty());

    const auto default_dev = GpuDeviceRegistry::default_device();
    assert(default_dev.backend_type != GpuBackendType::None);
    assert(default_dev.vram_total_bytes > 0);
    assert(!default_dev.name.empty());
    assert(!default_dev.vendor.empty());

    const char* emu_name = GpuDeviceRegistry::backend_name(GpuBackendType::Emulated);
    assert(std::string(emu_name) == "Emulated");
}

void test_vram_allocator_budget_and_oom() {
    using namespace aabhas::inference;
    VramConfig config;
    config.budget_bytes = 1024;
    config.alignment = 64;

    VramAllocator allocator(config);
    assert(allocator.stats().budget_bytes == 1024);
    assert(allocator.stats().allocated_bytes == 0);

    // Allocate 512 bytes -> succeeds
    DeviceBuffer buf1 = allocator.allocate(512);
    assert(buf1.valid());
    assert(buf1.size_bytes() == 512);
    assert(allocator.stats().allocated_bytes == 512);
    assert(allocator.stats().allocation_count == 1);

    // Allocate 600 bytes -> exceeds remaining 512 bytes, must fail
    DeviceBuffer buf2 = allocator.allocate(600);
    assert(!buf2.valid());
    assert(allocator.stats().allocated_bytes == 512);

    // Release buf1
    buf1.release();
    assert(!buf1.valid());
    assert(allocator.stats().allocated_bytes == 0);
    assert(allocator.stats().allocation_count == 0);

    // Now 600 bytes must succeed
    buf2 = allocator.allocate(600);
    assert(buf2.valid());
    assert(allocator.stats().allocated_bytes == 600);
    assert(allocator.stats().peak_allocated_bytes >= 600);

    buf2.release();
    assert(allocator.stats().allocated_bytes == 0);
}

void test_device_buffer_transfers() {
    using namespace aabhas::inference;
    VramAllocator allocator(VramConfig{4096, 64});

    constexpr std::size_t count = 16;
    constexpr std::size_t bytes = count * sizeof(float);

    std::vector<float> host_src(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_src[i] = static_cast<float>(i * 3.14F);
    }

    DeviceBuffer dev_buf = allocator.allocate(bytes);
    assert(dev_buf.valid());

    // Host to Device
    assert(dev_buf.copy_from_host(host_src.data(), bytes));

    // Device to Host
    std::vector<float> host_dst(count, 0.0F);
    assert(dev_buf.copy_to_host(host_dst.data(), bytes));

    for (std::size_t i = 0; i < count; ++i) {
        assert(near(host_src[i], host_dst[i], 1.0e-5F));
    }

    // Zero memory
    assert(dev_buf.zero());
    assert(dev_buf.copy_to_host(host_dst.data(), bytes));
    for (std::size_t i = 0; i < count; ++i) {
        assert(host_dst[i] == 0.0F);
    }
}

void test_gpu_accelerated_kernels() {
    using namespace aabhas::inference;
    GpuConfig gpu_cfg;
    gpu_cfg.backend_type = aabhas::hardware::GpuBackendType::Emulated;
    gpu_cfg.vram.budget_bytes = 1ULL << 20; // 1 MiB

    GpuBackend backend;
    assert(backend.initialize(gpu_cfg));
    assert(backend.is_active());

    // 1. GEMV test: W (2x4) * X (4) -> Y (2)
    std::vector<float> W = {
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };
    std::vector<float> X = {0.5F, 1.0F, -0.5F, 2.0F};
    // Expected:
    // Y[0] = 1*0.5 + 2*1 + 3*(-0.5) + 4*2 = 0.5 + 2 - 1.5 + 8 = 9.0
    // Y[1] = 5*0.5 + 6*1 + 7*(-0.5) + 8*2 = 2.5 + 6 - 3.5 + 16 = 21.0

    DeviceBuffer w_buf = backend.allocator().allocate(W.size() * sizeof(float));
    DeviceBuffer x_buf = backend.allocator().allocate(X.size() * sizeof(float));
    DeviceBuffer y_buf = backend.allocator().allocate(2 * sizeof(float));
    assert(w_buf.copy_from_host(W.data(), W.size() * sizeof(float)));
    assert(x_buf.copy_from_host(X.data(), X.size() * sizeof(float)));

    assert(backend.gemv(w_buf, x_buf, y_buf, 2, 4));

    std::vector<float> Y(2, 0.0F);
    assert(y_buf.copy_to_host(Y.data(), 2 * sizeof(float)));
    assert(near(Y[0], 9.0F));
    assert(near(Y[1], 21.0F));

    // 2. RMSNorm test
    std::vector<float> norm_in = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> gamma = {1.0F, 1.0F, 1.0F, 1.0F};
    DeviceBuffer norm_in_buf = backend.allocator().allocate(4 * sizeof(float));
    DeviceBuffer gamma_buf = backend.allocator().allocate(4 * sizeof(float));
    DeviceBuffer norm_out_buf = backend.allocator().allocate(4 * sizeof(float));
    assert(norm_in_buf.copy_from_host(norm_in.data(), 4 * sizeof(float)));
    assert(gamma_buf.copy_from_host(gamma.data(), 4 * sizeof(float)));

    assert(backend.rms_norm(norm_in_buf, gamma_buf, norm_out_buf, 4, 1.0e-5F));
    std::vector<float> norm_out(4, 0.0F);
    assert(norm_out_buf.copy_to_host(norm_out.data(), 4 * sizeof(float)));
    assert(near(norm_out[0], 0.365148F));
    assert(near(norm_out[3], 1.460592F));

    // 3. SwiGLU test
    std::vector<float> gate = {0.0F, 1.0F};
    std::vector<float> up = {2.0F, 2.0F};
    DeviceBuffer gate_buf = backend.allocator().allocate(2 * sizeof(float));
    DeviceBuffer up_buf = backend.allocator().allocate(2 * sizeof(float));
    DeviceBuffer glu_out_buf = backend.allocator().allocate(2 * sizeof(float));
    assert(gate_buf.copy_from_host(gate.data(), 2 * sizeof(float)));
    assert(up_buf.copy_from_host(up.data(), 2 * sizeof(float)));

    assert(backend.swiglu(gate_buf, up_buf, glu_out_buf, 2));
    std::vector<float> glu_out(2, 0.0F);
    assert(glu_out_buf.copy_to_host(glu_out.data(), 2 * sizeof(float)));
    assert(near(glu_out[0], 0.0F));
    assert(near(glu_out[1], 1.462117F));

    backend.shutdown();
    assert(!backend.is_active());
}

void test_hybrid_gpu_layer_offloading() {
    const auto path =
        std::filesystem::temp_directory_path() / "aabhas-gpu-hybrid-test.bin";

    constexpr std::size_t vocab = 8;
    constexpr std::size_t dim = 4;
    constexpr std::size_t layers = 2; // 2 layers: layer 0 on GPU, layer 1 on CPU
    constexpr std::size_t heads = 2;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t ffn_dim = 4;

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

    // Layer 0 tensors (will be offloaded to GPU)
    add_t("blk.0.attn_norm.weight", norm, {dim});
    add_t("blk.0.attn_q.weight", eye_dim, {dim, dim});
    add_t("blk.0.attn_k.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.0.attn_v.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.0.attn_output.weight", eye_dim, {dim, dim});
    add_t("blk.0.ffn_norm.weight", norm, {dim});
    add_t("blk.0.ffn_gate.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.0.ffn_up.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.0.ffn_down.weight", ffn_down, {dim, ffn_dim});

    // Layer 1 tensors (will run on CPU)
    add_t("blk.1.attn_norm.weight", norm, {dim});
    add_t("blk.1.attn_q.weight", eye_dim, {dim, dim});
    add_t("blk.1.attn_k.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.1.attn_v.weight", key_val, {kv_heads * head_dim, dim});
    add_t("blk.1.attn_output.weight", eye_dim, {dim, dim});
    add_t("blk.1.ffn_norm.weight", norm, {dim});
    add_t("blk.1.ffn_gate.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.1.ffn_up.weight", ffn_gate_up, {ffn_dim, dim});
    add_t("blk.1.ffn_down.weight", ffn_down, {dim, ffn_dim});

    add_t("output_norm.weight", norm, {dim});
    add_t("output.weight", lm_head, {vocab, dim});
    out.close();

    model.file_size = offset;
    model.tokenizer.vocabulary = {"<unk>", "<bos>", "<eos>", "alpha", "beta", "gamma", "delta", "epsilon"};
    model.tokenizer.bos_token_id = 1;
    model.tokenizer.eos_token_id = 2;
    model.tokenizer.unk_token_id = 0;

    aabhas::inference::InferenceRuntime runtime;
    aabhas::inference::InferenceConfig config;
    config.scheduler.memory_cap_bytes = 8192;
    config.scheduler.chunk_bytes = 128;
    // Configure hybrid GPU offload: 1 layer on GPU VRAM
    config.gpu.backend_type = aabhas::hardware::GpuBackendType::Emulated;
    config.gpu.gpu_layers = 1;
    config.gpu.vram.budget_bytes = 1ULL << 20;

    assert(runtime.open(path.string().c_str(), model, config));
    assert(runtime.gpu_backend().is_active());
    assert(runtime.gpu_backend().is_layer_offloaded(0));
    assert(!runtime.gpu_backend().is_layer_offloaded(1));
    assert(runtime.configure_tokenizer());

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

    assert(runtime.configure_model(model_cfg, 32));
    assert(runtime.gpu_backend().vram_stats().allocated_bytes > 0);

    // Test forward step on single token with hybrid GPU/CPU pipeline
    std::vector<float> logits(vocab, 0.0F);
    const std::uint32_t prompt_token = 3; // "alpha"
    assert(runtime.generator().forward_step(&prompt_token, 1, 0, logits.data()));
    for (std::size_t i = 0; i < vocab; ++i) {
        assert(std::isfinite(logits[i]));
    }

    // Test end-to-end generation with hybrid GPU offload
    aabhas::inference::GenerationOptions options;
    options.max_new_tokens = 3;
    options.early_stopping = true;

    const std::vector<std::uint32_t> prompt = {3, 4};
    aabhas::inference::GenerationResult result;
    assert(runtime.generate(prompt, options, result));
    assert(result.prompt_token_count == 2);
    assert(result.generated_token_count <= 3);

    runtime.close();
    assert(!runtime.gpu_backend().is_active());
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_gpu_device_enumeration();
    test_vram_allocator_budget_and_oom();
    test_device_buffer_transfers();
    test_gpu_accelerated_kernels();
    test_hybrid_gpu_layer_offloading();
    return 0;
}
