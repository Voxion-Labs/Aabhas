#ifndef AABHAS_INFERENCE_GPU_BACKEND_HPP
#define AABHAS_INFERENCE_GPU_BACKEND_HPP

#include "hardware/gpu_device.hpp"
#include "inference/kv_cache.hpp"
#include "inference/transformer_layers.hpp"
#include "inference/vram_allocator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aabhas::inference {

class InferenceRuntime;

/** Configuration for GPU layer offloading and VRAM placement. */
struct GpuConfig {
    hardware::GpuBackendType backend_type{hardware::GpuBackendType::Emulated};
    std::size_t gpu_layers{0};       // 0 = all CPU; > 0 = first N layers offloaded to VRAM
    bool offload_kv_cache{false};    // Store active KV cache in device memory
    bool offload_embeddings{false};  // Store token embedding table in VRAM
    bool offload_lm_head{false};     // Store LM head in VRAM
    VramConfig vram{};               // VRAM budget and alignment parameters
};

/**
 * High-performance GPU/VRAM acceleration backend.
 *
 * Provides:
 * - Dynamic accelerator discovery and capability detection (CUDA, Vulkan, ROCm, Emulated)
 * - Strict VRAM budget enforcement via VramAllocator
 * - Hybrid CPU/GPU transformer layer execution
 * - Pinned Host-to-Device (H2D) and Device-to-Host (D2H) memory streaming
 * - Accelerated device kernels: GEMV, RMSNorm, SwiGLU, RoPE, and Causal Attention
 */
class GpuBackend final {
public:
    GpuBackend() noexcept;
    ~GpuBackend() noexcept;

    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;

    GpuBackend(GpuBackend&&) noexcept = default;
    GpuBackend& operator=(GpuBackend&&) noexcept = default;

    /** Initialize compute device and allocate VRAM structures based on configuration. */
    bool initialize(const GpuConfig& config);

    /** Release all device buffers and shutdown accelerator session. */
    void shutdown() noexcept;

    /** Whether an accelerator backend is initialized and active. */
    bool is_active() const noexcept { return is_initialized_; }

    /** Number of layers configured to execute on the accelerator. */
    std::size_t offloaded_layer_count() const noexcept { return config_.gpu_layers; }

    /** Whether the given layer index is offloaded to GPU. */
    bool is_layer_offloaded(std::size_t layer_index) const noexcept {
        return is_initialized_ && layer_index < config_.gpu_layers;
    }

    /** Pre-upload layer weights into device VRAM. */
    bool upload_layer_weights(std::size_t layer_index,
                              const TransformerLayerWeights& host_weights,
                              const TransformerLayerPlan& plan);

    /**
     * Execute a forward pass for an offloaded transformer layer on the accelerator.
     * Takes host input activations, transfers to device if needed, computes block
     * forward on device, and writes outputs to host_output.
     */
    bool execute_layer_forward(std::size_t layer_index,
                               const TransformerBlockConfig& config,
                               std::size_t start_position,
                               const float* host_input,
                               std::size_t token_count,
                               KvCache& kv_cache,
                               float* host_output);

    /** Accelerated device matrix-vector multiplication (Y = W * X). */
    bool gemv(const DeviceBuffer& weights,
              const DeviceBuffer& input,
              DeviceBuffer& output,
              std::size_t rows,
              std::size_t columns) noexcept;

    /** Accelerated device RMSNorm. */
    bool rms_norm(const DeviceBuffer& input,
                  const DeviceBuffer& gamma,
                  DeviceBuffer& output,
                  std::size_t dimension,
                  float epsilon) noexcept;

    /** Accelerated device SwiGLU feed-forward activation. */
    bool swiglu(const DeviceBuffer& gate,
                const DeviceBuffer& up,
                DeviceBuffer& output,
                std::size_t dimension) noexcept;

    /** Accelerated device Rotary Position Embedding (RoPE). */
    bool apply_rope(DeviceBuffer& query,
                    DeviceBuffer& key,
                    std::size_t head_dimension,
                    std::size_t position,
                    float theta) noexcept;

    const hardware::GpuDeviceInfo& device_info() const noexcept { return device_info_; }
    const GpuConfig& config() const noexcept { return config_; }
    VramAllocator& allocator() noexcept { return allocator_; }
    const VramAllocator& allocator() const noexcept { return allocator_; }
    VramStats vram_stats() const noexcept { return allocator_.stats(); }

private:
    struct OffloadedLayer {
        DeviceBuffer weights_buffer;
        TransformerLayerPlan plan;
        bool is_loaded{false};
    };

    bool is_initialized_{false};
    GpuConfig config_{};
    hardware::GpuDeviceInfo device_info_{};
    VramAllocator allocator_{};
    std::vector<OffloadedLayer> layers_{};

    // Device scratch buffers for intermediate activations
    DeviceBuffer act_in_;
    DeviceBuffer act_out_;
    DeviceBuffer q_buf_;
    DeviceBuffer k_buf_;
    DeviceBuffer v_buf_;
    DeviceBuffer attn_out_;
    DeviceBuffer ffn_norm_;
    DeviceBuffer ffn_gate_;
    DeviceBuffer ffn_up_;
    DeviceBuffer ffn_mid_;
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_GPU_BACKEND_HPP
