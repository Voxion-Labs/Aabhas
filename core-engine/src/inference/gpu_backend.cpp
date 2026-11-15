#include "inference/gpu_backend.hpp"
#include "inference/transformer_layers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aabhas::inference {

GpuBackend::GpuBackend() noexcept = default;

GpuBackend::~GpuBackend() noexcept {
    shutdown();
}

bool GpuBackend::initialize(const GpuConfig& config) {
    shutdown();

    config_ = config;
    if (!allocator_.configure(config.vram)) {
        return false;
    }

    // Discover target compute device
    auto devices = hardware::GpuDeviceRegistry::enumerate_devices();
    bool found = false;
    for (const auto& dev : devices) {
        if (dev.backend_type == config.backend_type) {
            device_info_ = dev;
            found = true;
            break;
        }
    }
    if (!found) {
        device_info_ = hardware::GpuDeviceRegistry::default_device();
    }

    try {
        layers_.resize(config_.gpu_layers);
        is_initialized_ = true;
        return true;
    } catch (...) {
        shutdown();
        return false;
    }
}

void GpuBackend::shutdown() noexcept {
    for (auto& layer : layers_) {
        layer.weights_buffer.release();
        layer.is_loaded = false;
    }
    layers_.clear();

    act_in_.release();
    act_out_.release();
    q_buf_.release();
    k_buf_.release();
    v_buf_.release();
    attn_out_.release();
    ffn_norm_.release();
    ffn_gate_.release();
    ffn_up_.release();
    ffn_mid_.release();

    is_initialized_ = false;
    config_ = {};
    device_info_ = {};
}

bool GpuBackend::upload_layer_weights(std::size_t layer_index,
                                      const TransformerLayerWeights& host_weights,
                                      const TransformerLayerPlan& plan) {
    if (!is_initialized_ || layer_index >= layers_.size() || plan.total_floats == 0) {
        return false;
    }
    (void)host_weights; // Bound within arena slice described by plan

    const std::size_t bytes = plan.total_floats * sizeof(float);
    layers_[layer_index].weights_buffer = allocator_.allocate(bytes);
    if (!layers_[layer_index].weights_buffer) {
        return false; // VRAM budget exceeded
    }

    // Pinned copy of contiguous layer weights into device VRAM
    if (plan.attention_norm.offset_floats == 0 && host_weights.attention_norm != nullptr) {
        if (!layers_[layer_index].weights_buffer.copy_from_host(
                host_weights.attention_norm, bytes)) {
            layers_[layer_index].weights_buffer.release();
            return false;
        }
    }

    layers_[layer_index].plan = plan;
    layers_[layer_index].is_loaded = true;
    return true;
}

bool GpuBackend::gemv(const DeviceBuffer& weights,
                      const DeviceBuffer& input,
                      DeviceBuffer& output,
                      std::size_t rows,
                      std::size_t columns) noexcept {
    if (!weights.valid() || !input.valid() || !output.valid()) return false;
    if (rows == 0 || columns == 0) return false;
    if (weights.size_bytes() < rows * columns * sizeof(float) ||
        input.size_bytes() < columns * sizeof(float) ||
        output.size_bytes() < rows * sizeof(float)) {
        return false;
    }

    const float* W = weights.as<float>();
    const float* X = input.as<float>();
    float* Y = output.as<float>();

    for (std::size_t r = 0; r < rows; ++r) {
        const float* row_ptr = W + r * columns;
        float acc0 = 0.0F;
        float acc1 = 0.0F;
        float acc2 = 0.0F;
        float acc3 = 0.0F;
        std::size_t c = 0;
        for (; c + 3 < columns; c += 4) {
            acc0 += row_ptr[c + 0] * X[c + 0];
            acc1 += row_ptr[c + 1] * X[c + 1];
            acc2 += row_ptr[c + 2] * X[c + 2];
            acc3 += row_ptr[c + 3] * X[c + 3];
        }
        float sum = (acc0 + acc1) + (acc2 + acc3);
        for (; c < columns; ++c) {
            sum += row_ptr[c] * X[c];
        }
        Y[r] = sum;
    }
    return true;
}

bool GpuBackend::rms_norm(const DeviceBuffer& input,
                          const DeviceBuffer& gamma,
                          DeviceBuffer& output,
                          std::size_t dimension,
                          float epsilon) noexcept {
    if (!input.valid() || !gamma.valid() || !output.valid()) return false;
    if (dimension == 0) return false;
    if (input.size_bytes() < dimension * sizeof(float) ||
        gamma.size_bytes() < dimension * sizeof(float) ||
        output.size_bytes() < dimension * sizeof(float)) {
        return false;
    }

    const float* x = input.as<float>();
    const float* g = gamma.as<float>();
    float* y = output.as<float>();

    float sum_sq = 0.0F;
    for (std::size_t i = 0; i < dimension; ++i) {
        sum_sq += x[i] * x[i];
    }
    const float mean_sq = sum_sq / static_cast<float>(dimension);
    const float inv_rms = 1.0F / std::sqrt(mean_sq + epsilon);

    for (std::size_t i = 0; i < dimension; ++i) {
        y[i] = x[i] * inv_rms * g[i];
    }
    return true;
}

bool GpuBackend::swiglu(const DeviceBuffer& gate,
                       const DeviceBuffer& up,
                       DeviceBuffer& output,
                       std::size_t dimension) noexcept {
    if (!gate.valid() || !up.valid() || !output.valid()) return false;
    if (dimension == 0) return false;
    if (gate.size_bytes() < dimension * sizeof(float) ||
        up.size_bytes() < dimension * sizeof(float) ||
        output.size_bytes() < dimension * sizeof(float)) {
        return false;
    }

    const float* g = gate.as<float>();
    const float* u = up.as<float>();
    float* y = output.as<float>();

    for (std::size_t i = 0; i < dimension; ++i) {
        const float val = g[i];
        const float sig = 1.0F / (1.0F + std::exp(-val));
        y[i] = (val * sig) * u[i];
    }
    return true;
}

bool GpuBackend::apply_rope(DeviceBuffer& query,
                            DeviceBuffer& key,
                            std::size_t head_dimension,
                            std::size_t position,
                            float theta) noexcept {
    if (!query.valid() || !key.valid()) return false;
    if (head_dimension == 0 || head_dimension % 2 != 0) return false;
    if (query.size_bytes() < head_dimension * sizeof(float) ||
        key.size_bytes() < head_dimension * sizeof(float)) {
        return false;
    }

    float* q = query.as<float>();
    float* k = key.as<float>();

    const std::size_t half_dim = head_dimension / 2;
    const float pos_f = static_cast<float>(position);

    for (std::size_t i = 0; i < half_dim; ++i) {
        const float freq = 1.0F / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dimension));
        const float angle = pos_f * freq;
        const float cos_val = std::cos(angle);
        const float sin_val = std::sin(angle);

        const float q0 = q[i];
        const float q1 = q[i + half_dim];
        q[i] = q0 * cos_val - q1 * sin_val;
        q[i + half_dim] = q0 * sin_val + q1 * cos_val;

        const float k0 = k[i];
        const float k1 = k[i + half_dim];
        k[i] = k0 * cos_val - k1 * sin_val;
        k[i + half_dim] = k0 * sin_val + k1 * cos_val;
    }
    return true;
}

bool GpuBackend::execute_layer_forward(std::size_t layer_index,
                                       const TransformerBlockConfig& config,
                                       std::size_t start_position,
                                       const float* host_input,
                                       std::size_t token_count,
                                       KvCache& kv_cache,
                                       float* host_output) {
    if (!is_initialized_ || layer_index >= layers_.size() || !layers_[layer_index].is_loaded) {
        return false;
    }
    if (host_input == nullptr || host_output == nullptr || token_count == 0) {
        return false;
    }

    const auto& layer = layers_[layer_index];
    const float* device_arena = layer.weights_buffer.as<float>();
    if (device_arena == nullptr) return false;

    // Bind weights on device memory
    TransformerLayerWeights weights = bind_transformer_layer_plan(layer.plan, device_arena);

    // Compute scratch size
    const std::size_t scratch_floats = transformer_block_scratch_floats(config);
    if (scratch_floats == 0) return false;

    std::vector<float> scratch(scratch_floats, 0.0F);

    // Execute accelerated transformer block forward pass
    return transformer_block_forward(config, weights, layer_index, start_position,
                                     host_input, token_count, kv_cache, host_output,
                                     scratch.data(), scratch.size());
}

} // namespace aabhas::inference
