#include "inference/model_pool.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace aabhas::inference {

// ----------------------------------------------------------------------------
// ModelLease implementation
// ----------------------------------------------------------------------------

ModelLease::ModelLease(ModelPool* pool, std::string model_id,
                       InferenceRuntime* runtime) noexcept
    : pool_(pool), model_id_(std::move(model_id)), runtime_(runtime) {}

ModelLease::ModelLease(ModelLease&& other) noexcept
    : pool_(other.pool_),
      model_id_(std::move(other.model_id_)),
      runtime_(other.runtime_) {
    other.pool_ = nullptr;
    other.runtime_ = nullptr;
}

ModelLease& ModelLease::operator=(ModelLease&& other) noexcept {
    if (this != &other) {
        release();
        pool_ = other.pool_;
        model_id_ = std::move(other.model_id_);
        runtime_ = other.runtime_;
        other.pool_ = nullptr;
        other.runtime_ = nullptr;
    }
    return *this;
}

void ModelLease::release() noexcept {
    if (pool_ != nullptr && !model_id_.empty()) {
        pool_->release_lease(model_id_);
        pool_ = nullptr;
        runtime_ = nullptr;
        model_id_.clear();
    }
}

// ----------------------------------------------------------------------------
// ModelPool implementation
// ----------------------------------------------------------------------------

ModelPool::ModelPool(ModelPoolConfig config) noexcept : config_(config) {
    if (config_.max_loaded_models == 0) config_.max_loaded_models = 1;
    if (config_.max_host_memory_bytes == 0) {
        config_.max_host_memory_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    }
    if (config_.max_vram_bytes == 0) {
        config_.max_vram_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    }
}

ModelPool::~ModelPool() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& pair : slots_) {
        if (pair.second.runtime) {
            pair.second.runtime->close();
            pair.second.runtime.reset();
        }
        pair.second.is_loaded = false;
    }
    slots_.clear();
}

std::size_t ModelPool::compute_estimated_host_bytes(const ModelRegistration& reg) const noexcept {
    if (reg.estimated_host_bytes > 0) return reg.estimated_host_bytes;
    std::size_t bytes = reg.model_index.file_size;
    if (bytes == 0) {
        for (const auto& t : reg.model_index.tensors) {
            bytes += t.size_bytes;
        }
    }
    bytes += reg.inference_config.max_kv_bytes;
    bytes += reg.inference_config.max_layer_buffer_bytes;
    return (bytes > 0) ? bytes : (64ULL * 1024ULL * 1024ULL);
}

std::size_t ModelPool::compute_estimated_vram_bytes(const ModelRegistration& reg) const noexcept {
    if (reg.estimated_vram_bytes > 0) return reg.estimated_vram_bytes;
    if (reg.inference_config.gpu.gpu_layers == 0) return 0;
    std::size_t bytes = 0;
    if (reg.model_config.layer_count > 0) {
        std::size_t total_weight_bytes = 0;
        for (const auto& t : reg.model_index.tensors) {
            total_weight_bytes += t.size_bytes;
        }
        const std::size_t offloaded = std::min(reg.inference_config.gpu.gpu_layers,
                                              reg.model_config.layer_count);
        bytes = (total_weight_bytes * offloaded) / reg.model_config.layer_count;
    }
    return bytes;
}

bool ModelPool::register_model(const ModelRegistration& registration) {
    if (registration.model_id.empty() || registration.model_path.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (slots_.find(registration.model_id) != slots_.end()) {
        return false;
    }
    ModelSlot slot;
    slot.registration = registration;
    slot.is_loaded = false;
    slot.active_leases = 0;
    slot.total_acquisitions = 0;
    slot.last_accessed_ns = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    slot.host_memory_bytes = 0;
    slot.vram_bytes = 0;

    slots_.emplace(registration.model_id, std::move(slot));
    return true;
}

bool ModelPool::unregister_model(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = slots_.find(model_id);
    if (it == slots_.end()) return false;
    if (it->second.active_leases > 0) return false;

    if (it->second.runtime) {
        it->second.runtime->close();
        it->second.runtime.reset();
    }
    slots_.erase(it);
    return true;
}

bool ModelPool::evict_lru_slot_locked(std::size_t required_host_bytes,
                                      std::size_t required_vram_bytes) {
    (void)required_host_bytes;
    (void)required_vram_bytes;
    std::string candidate_id{};
    std::uint64_t oldest_access = std::numeric_limits<std::uint64_t>::max();

    for (const auto& pair : slots_) {
        const auto& slot = pair.second;
        if (slot.is_loaded && slot.active_leases == 0) {
            if (slot.last_accessed_ns <= oldest_access) {
                oldest_access = slot.last_accessed_ns;
                candidate_id = pair.first;
            }
        }
    }

    if (candidate_id.empty()) {
        return false;
    }

    auto it = slots_.find(candidate_id);
    if (it != slots_.end()) {
        auto& slot = it->second;
        if (slot.runtime) {
            slot.runtime->close();
            slot.runtime.reset();
        }
        slot.is_loaded = false;
        slot.host_memory_bytes = 0;
        slot.vram_bytes = 0;
        total_evictions_++;
        return true;
    }
    return false;
}

bool ModelPool::load_slot_locked(ModelSlot& slot) {
    if (slot.is_loaded && slot.runtime != nullptr) {
        return true;
    }

    const std::size_t est_host = compute_estimated_host_bytes(slot.registration);
    const std::size_t est_vram = compute_estimated_vram_bytes(slot.registration);

    // Evict LRU models until limits are respected
    std::size_t loaded_cnt = 0;
    std::size_t current_host = 0;
    std::size_t current_vram = 0;
    for (const auto& pair : slots_) {
        if (pair.second.is_loaded) {
            loaded_cnt++;
            current_host += pair.second.host_memory_bytes;
            current_vram += pair.second.vram_bytes;
        }
    }

    while ((loaded_cnt >= config_.max_loaded_models) ||
           (current_host + est_host > config_.max_host_memory_bytes) ||
           (current_vram + est_vram > config_.max_vram_bytes)) {
        if (!evict_lru_slot_locked(est_host, est_vram)) {
            return false;
        }
        loaded_cnt = 0;
        current_host = 0;
        current_vram = 0;
        for (const auto& pair : slots_) {
            if (pair.second.is_loaded) {
                loaded_cnt++;
                current_host += pair.second.host_memory_bytes;
                current_vram += pair.second.vram_bytes;
            }
        }
    }

    auto rt = std::make_unique<InferenceRuntime>();
    if (!rt->open(slot.registration.model_path.c_str(), slot.registration.model_index,
                  slot.registration.inference_config)) {
        return false;
    }
    (void)rt->configure_tokenizer();
    if (!rt->configure_model(slot.registration.model_config,
                             slot.registration.max_sequence_length)) {
        rt->close();
        return false;
    }

    slot.runtime = std::move(rt);
    slot.is_loaded = true;
    slot.host_memory_bytes = est_host;
    slot.vram_bytes = est_vram;
    slot.last_accessed_ns = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return true;
}

bool ModelPool::preload_model(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = slots_.find(model_id);
    if (it == slots_.end()) return false;
    return load_slot_locked(it->second);
}

ModelLease ModelPool::acquire_model(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = slots_.find(model_id);
    if (it == slots_.end()) {
        return ModelLease{};
    }

    auto& slot = it->second;
    if (!slot.is_loaded) {
        if (!load_slot_locked(slot)) {
            return ModelLease{};
        }
    }

    slot.active_leases++;
    slot.total_acquisitions++;
    total_acquisitions_++;
    slot.last_accessed_ns = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    return ModelLease(this, model_id, slot.runtime.get());
}

bool ModelPool::evict_model(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = slots_.find(model_id);
    if (it == slots_.end() || !it->second.is_loaded) return false;
    if (it->second.active_leases > 0) return false;

    if (it->second.runtime) {
        it->second.runtime->close();
        it->second.runtime.reset();
    }
    it->second.is_loaded = false;
    it->second.host_memory_bytes = 0;
    it->second.vram_bytes = 0;
    total_evictions_++;
    return true;
}

bool ModelPool::has_model(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return slots_.find(model_id) != slots_.end();
}

bool ModelPool::is_model_loaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = slots_.find(model_id);
    if (it == slots_.end()) return false;
    return it->second.is_loaded;
}

ModelPoolStats ModelPool::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    ModelPoolStats pool_stats;
    pool_stats.total_registered_models = slots_.size();
    pool_stats.total_acquisitions = total_acquisitions_;
    pool_stats.total_evictions = total_evictions_;

    for (const auto& pair : slots_) {
        const auto& slot = pair.second;
        if (slot.is_loaded) {
            pool_stats.loaded_models++;
            pool_stats.active_leases += slot.active_leases;
            pool_stats.host_memory_used_bytes += slot.host_memory_bytes;
            pool_stats.vram_used_bytes += slot.vram_bytes;
        }
    }
    return pool_stats;
}

std::vector<ModelSlotStats> ModelPool::list_slots() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ModelSlotStats> list;
    list.reserve(slots_.size());
    for (const auto& pair : slots_) {
        const auto& slot = pair.second;
        ModelSlotStats slot_stats;
        slot_stats.model_id = pair.first;
        slot_stats.model_path = slot.registration.model_path;
        slot_stats.active_leases = slot.active_leases;
        slot_stats.total_acquisitions = slot.total_acquisitions;
        slot_stats.last_accessed_ns = slot.last_accessed_ns;
        slot_stats.host_memory_bytes = slot.host_memory_bytes;
        slot_stats.vram_bytes = slot.vram_bytes;
        slot_stats.is_loaded = slot.is_loaded;
        list.push_back(std::move(slot_stats));
    }
    return list;
}

void ModelPool::release_lease(const std::string& model_id) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = slots_.find(model_id);
    if (it != slots_.end()) {
        if (it->second.active_leases > 0) {
            it->second.active_leases--;
        }
        it->second.last_accessed_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
}

} // namespace aabhas::inference
