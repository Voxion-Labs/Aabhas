#ifndef AABHAS_INFERENCE_BATCH_SCHEDULER_HPP
#define AABHAS_INFERENCE_BATCH_SCHEDULER_HPP

#include "inference/kv_cache.hpp"
#include "inference/sampler.hpp"
#include "inference/token_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace aabhas::inference {

class InferenceRuntime;

/** Lifecycle status for an individual sequence inside the batch scheduler. */
enum class SequenceStatus : std::uint8_t {
    Pending,
    Prefilling,
    Decoding,
    Finished,
    Failed
};

/** Termination reason for a completed sequence. */
enum class FinishReason : std::uint8_t {
    None,
    EosToken,
    StopToken,
    MaxNewTokens,
    ContextLengthExceeded,
    Aborted
};

/** Runtime configuration parameters for continuous sequence batching. */
struct BatchSchedulerConfig {
    std::size_t max_batch_size{16};
    std::size_t max_queue_depth{128};
    std::size_t max_sequence_length{2048};
};

/** Client submission request for a single sequence generation. */
struct BatchRequest {
    std::uint64_t request_id{0};
    std::vector<std::uint32_t> prompt_tokens{};
    GenerationOptions options{};
    StreamCallback stream_callback{nullptr};
};

/** Internal state tracking an active sequence within the batch scheduler. */
struct BatchSequenceState {
    std::uint64_t request_id{0};
    std::vector<std::uint32_t> prompt_tokens{};
    std::vector<std::uint32_t> generated_tokens{};
    std::vector<std::uint32_t> all_tokens{};
    std::string text{};
    SequenceStatus status{SequenceStatus::Pending};
    FinishReason finish_reason{FinishReason::None};
    GenerationOptions options{};
    StreamCallback stream_callback{nullptr};
    std::size_t current_position{0};
    KvCache kv_cache{};
    Sampler sampler{};
    bool reached_eos{false};
};

/** Telemetry metrics reported per batch execution step. */
struct BatchStepMetrics {
    std::size_t active_sequences{0};
    std::size_t prefilling_sequences{0};
    std::size_t decoding_sequences{0};
    std::size_t completed_sequences{0};
    std::size_t tokens_generated_in_step{0};
};

/**
 * Continuous / dynamic batch scheduler for multi-sequence autoregressive inference.
 *
 * Implements interleaved prompt prefill and multi-token decoding across concurrent
 * sequences with heterogeneous sequence lengths, per-sequence independent KV caches,
 * per-sequence samplers, individual early-stopping, and fine-grained streaming deltas.
 */
class BatchScheduler final {
public:
    explicit BatchScheduler(BatchSchedulerConfig config = {}) noexcept;
    ~BatchScheduler() noexcept = default;

    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;
    BatchScheduler(BatchScheduler&&) noexcept = default;
    BatchScheduler& operator=(BatchScheduler&&) noexcept = default;

    /**
     * Initialize scheduler with runtime reference and model architecture.
     * Allocates scratch logit buffers sized to the model vocabulary.
     */
    bool initialize(InferenceRuntime* runtime, const TransformerModelConfig& config);

    /**
     * Enqueue a new generation request into the pending queue.
     * Returns assigned request_id, or 0 if queue capacity is exceeded or request is invalid.
     */
    std::uint64_t enqueue(BatchRequest request);

    /**
     * Execute one dynamic batch step across all active sequences:
     * - Admits pending requests into active batch slots up to max_batch_size.
     * - Evaluates prompt prefill for newly admitted sequences.
     * - Evaluates autoregressive decode step for all active decoding sequences.
     * - Retires completed sequences and invokes stream callbacks.
     */
    bool step(BatchStepMetrics& metrics);

    /**
     * Helper to run synchronous batch generation over a set of prompts.
     * Runs dynamic batch steps until all sequences have completed.
     */
    std::vector<GenerationResult> generate_batch(
        const std::vector<std::vector<std::uint32_t>>& prompts,
        const std::vector<GenerationOptions>& options,
        std::vector<StreamCallback> stream_callbacks = {});

    /** Retrieve completed generation result for a finished request. */
    bool get_result(std::uint64_t request_id, GenerationResult& result) const;

    /** Clear all pending queues, active sequences, and completed results. */
    void reset() noexcept;

    bool is_idle() const noexcept {
        return pending_requests_.empty() && active_sequences_.empty();
    }

    std::size_t active_count() const noexcept { return active_sequences_.size(); }
    std::size_t pending_count() const noexcept { return pending_requests_.size(); }
    std::size_t completed_count() const noexcept { return completed_results_.size(); }
    const BatchSchedulerConfig& config() const noexcept { return config_; }

private:
    bool step_sequence_prefill(BatchSequenceState& seq, BatchStepMetrics& metrics);
    bool step_sequence_decode(BatchSequenceState& seq, BatchStepMetrics& metrics);
    bool emit_chunk(BatchSequenceState& seq, std::uint32_t token_id, bool is_eos);

    BatchSchedulerConfig config_{};
    InferenceRuntime* runtime_{nullptr};
    TransformerModelConfig model_config_{};
    std::uint64_t next_request_id_{1};

    std::deque<BatchRequest> pending_requests_{};
    std::vector<BatchSequenceState> active_sequences_{};
    std::unordered_map<std::uint64_t, GenerationResult> completed_results_{};
    std::vector<float> logits_scratch_{};
};

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_BATCH_SCHEDULER_HPP
