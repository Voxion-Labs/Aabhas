#ifndef AABHAS_INFERENCE_MODEL_POOL_HPP
#define AABHAS_INFERENCE_MODEL_POOL_HPP

#include "inference/inference_runtime.hpp"
#include "model/model_index.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace aabhas::inference {

class ModelPool;

/** Bounded resource constraints for multi-model concurrent execution. */
struct ModelPoolConfig {
    std::size_t max_loaded_models{4};
    std::size_t max_host_memory_bytes{4ULL * 1024ULL * 1024ULL * 1024ULL}; // 4 GB default
    std::size_t max_vram_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};        // 2 GB default
};

/** Metadata specification for registering a model with the execution pool. */
struct ModelRegistration {
    std::string model_id{};
    std::string model_path{};
    aabhas::model::ModelIndex model_index{};
    TransformerModelConfig model_config{};
    InferenceConfig inference_config{};
    std::size_t max_sequence_length{2048};
    std::size_t estimated_host_bytes{0};
    std::size_t estimated_vram_bytes{0};
};

/** Point-in-time metrics for an individual model slot. */
struct ModelSlotStats {
    std::string model_id{};
    std::string model_path{};
    std::size_t active_leases{0};
    std::uint64_t total_acquisitions{0};
    std::uint64_t last_accessed_ns{0};
    std::size_t host_memory_bytes{0};
    std::size_t vram_bytes{0};
    bool is_loaded{false};
};

/** Aggregate metrics across the entire multi-model pool. */
struct ModelPoolStats {
    std::size_t loaded_models{0};
    std::size_t total_registered_models{0};
    std::size_t active_leases{0};
    std::uint64_t total_acquisitions{0};
    std::uint64_t total_evictions{0};
    std::size_t host_memory_used_bytes{0};
    std::size_t vram_used_bytes{0};
};

/**
 * RAII lease granting exclusive or shared access to a loaded model instance.
 * While a lease is held, the underlying model cannot be evicted from host or device memory.
 */
class ModelLease final {
public:
    ModelLease() noexcept = default;
    ModelLease(ModelPool* pool, std::string model_id, InferenceRuntime* runtime) noexcept;
    ~ModelLease() noexcept { release(); }

    ModelLease(ModelLease&& other) noexcept;
    ModelLease& operator=(ModelLease&& other) noexcept;

    ModelLease(const ModelLease&) = delete;
    ModelLease& operator=(const ModelLease&) = delete;

    bool valid() const noexcept { return runtime_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

    InferenceRuntime* runtime() const noexcept { return runtime_; }
    InferenceRuntime* operator->() const noexcept { return runtime_; }
    InferenceRuntime& operator*() const noexcept { return *runtime_; }

    const std::string& model_id() const noexcept { return model_id_; }

    void release() noexcept;

private:
    ModelPool* pool_{nullptr};
    std::string model_id_{};
    InferenceRuntime* runtime_{nullptr};
};

/**
 * Multi-model execution pool managing multiple concurrently loaded models with LRU eviction,
 * strict memory budget enforcement, and thread-safe ref-counted leasing.
 */
class ModelPool final {
public:
    explicit ModelPool(ModelPoolConfig config = {}) noexcept;
    ~ModelPool() noexcept;

    ModelPool(const ModelPool&) = delete;
    ModelPool& operator=(const ModelPool&) = delete;
    ModelPool(ModelPool&&) = delete;
    ModelPool& operator=(ModelPool&&) = delete;

    /** Register a model specification for on-demand or preloaded execution. */
    bool register_model(const ModelRegistration& registration);

    /** Unregister a model. Model must have 0 active leases. */
    bool unregister_model(const std::string& model_id);

    /** Preload a registered model into memory immediately. */
    bool preload_model(const std::string& model_id);

    /**
     * Acquire a lease for the given model ID.
     * Loads the model if not already loaded, evicting the least-recently-used
     * unleased model if slot or memory limits are exceeded.
     */
    ModelLease acquire_model(const std::string& model_id);

    /** Evict an unleased model to free memory. Fails if model is currently leased. */
    bool evict_model(const std::string& model_id);

    /** Check if a model is registered. */
    bool has_model(const std::string& model_id) const;

    /** Check if a model is currently loaded in memory. */
    bool is_model_loaded(const std::string& model_id) const;

    /** Point-in-time aggregate pool statistics. */
    ModelPoolStats stats() const;

    /** Point-in-time statistics for all model slots. */
    std::vector<ModelSlotStats> list_slots() const;

    const ModelPoolConfig& config() const noexcept { return config_; }

private:
    friend class ModelLease;
    void release_lease(const std::string& model_id) noexcept;

    struct ModelSlot {
        ModelRegistration registration{};
        std::unique_ptr<InferenceRuntime> runtime{nullptr};
        bool is_loaded{false};
        std::size_t active_leases{0};
        std::uint64_t total_acquisitions{0};
        std::uint64_t last_accessed_ns{0};
        std::size_t host_memory_bytes{0};
        std::size_t vram_bytes{0};
    };

    bool evict_lru_slot_locked(std::size_t required_host_bytes, std::size_t required_vram_bytes);
    bool load_slot_locked(ModelSlot& slot);
    std::size_t compute_estimated_host_bytes(const ModelRegistration& reg) const noexcept;
    std::size_t compute_estimated_vram_bytes(const ModelRegistration& reg) const noexcept;

    ModelPoolConfig config_{};
    mutable std::mutex mu_{};
    std::unordered_map<std::string, ModelSlot> slots_{};
    std::uint64_t total_acquisitions_{0};
    std::uint64_t total_evictions_{0};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_MODEL_POOL_HPP
