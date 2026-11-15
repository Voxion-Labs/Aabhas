#include "inference/batch_scheduler.hpp"
#include "inference/inference_runtime.hpp"

#include <algorithm>
#include <utility>

namespace aabhas::inference {

BatchScheduler::BatchScheduler(BatchSchedulerConfig config) noexcept
    : config_(config) {
    if (config_.max_batch_size == 0) config_.max_batch_size = 1;
    if (config_.max_queue_depth == 0) config_.max_queue_depth = 1;
    if (config_.max_sequence_length == 0) config_.max_sequence_length = 2048;
}

bool BatchScheduler::initialize(InferenceRuntime* runtime,
                                const TransformerModelConfig& config) {
    if (runtime == nullptr || config.vocab_size == 0 ||
        config.model_dimension == 0 || config.layer_count == 0) {
        return false;
    }
    runtime_ = runtime;
    model_config_ = config;
    reset();

    try {
        logits_scratch_.assign(config.vocab_size, 0.0F);
    } catch (...) {
        reset();
        return false;
    }
    return true;
}

std::uint64_t BatchScheduler::enqueue(BatchRequest request) {
    if (runtime_ == nullptr || request.prompt_tokens.empty() ||
        request.options.max_new_tokens == 0) {
        return 0;
    }
    if (request.prompt_tokens.size() > config_.max_sequence_length) {
        return 0;
    }
    if (pending_requests_.size() >= config_.max_queue_depth) {
        return 0;
    }

    const std::uint64_t req_id = (request.request_id != 0) ? request.request_id : next_request_id_++;
    request.request_id = req_id;
    pending_requests_.push_back(std::move(request));
    return req_id;
}

bool BatchScheduler::emit_chunk(BatchSequenceState& seq, std::uint32_t token_id, bool is_eos) {
    if (!seq.stream_callback) return true;
    StreamChunk chunk;
    chunk.token_id = token_id;
    chunk.token_index = seq.generated_tokens.size() - 1;
    chunk.is_eos = is_eos;
    if (runtime_->tokenizer().configured()) {
        std::string current_full_text;
        runtime_->detokenize(seq.generated_tokens, current_full_text);
        if (current_full_text.size() >= seq.text.size()) {
            chunk.text_delta = current_full_text.substr(seq.text.size());
            seq.text = std::move(current_full_text);
        } else {
            chunk.text_delta = current_full_text;
            seq.text = std::move(current_full_text);
        }
    }
    return seq.stream_callback(chunk);
}

bool BatchScheduler::step_sequence_prefill(BatchSequenceState& seq, BatchStepMetrics& metrics) {
    metrics.prefilling_sequences++;
    if (!runtime_->generator().forward_step(seq.prompt_tokens.data(),
                                            seq.prompt_tokens.size(),
                                            0,
                                            logits_scratch_.data(),
                                            seq.kv_cache)) {
        seq.status = SequenceStatus::Failed;
        seq.finish_reason = FinishReason::Aborted;
        return false;
    }

    seq.current_position = seq.prompt_tokens.size();
    SamplerConfig sampler_cfg = seq.options.sampler;
    if (seq.options.repetition_penalty > 1.0F && sampler_cfg.repetition_penalty == 1.0F) {
        sampler_cfg.repetition_penalty = seq.options.repetition_penalty;
    }

    const std::uint32_t next_token = seq.sampler.sample(
        logits_scratch_.data(), model_config_.vocab_size, seq.all_tokens, sampler_cfg);
    metrics.tokens_generated_in_step++;

    if (model_config_.eos_token_id >= 0 &&
        next_token == static_cast<std::uint32_t>(model_config_.eos_token_id)) {
        seq.reached_eos = true;
        seq.finish_reason = FinishReason::EosToken;
    }
    for (const std::uint32_t stop_id : seq.options.stop_token_ids) {
        if (next_token == stop_id) {
            seq.reached_eos = true;
            seq.finish_reason = FinishReason::StopToken;
            break;
        }
    }

    seq.generated_tokens.push_back(next_token);
    seq.all_tokens.push_back(next_token);

    if (!emit_chunk(seq, next_token, seq.reached_eos)) {
        seq.status = SequenceStatus::Finished;
        seq.finish_reason = FinishReason::Aborted;
        return true;
    }

    if ((seq.reached_eos && seq.options.early_stopping) ||
        seq.generated_tokens.size() >= seq.options.max_new_tokens ||
        seq.current_position >= config_.max_sequence_length) {
        seq.status = SequenceStatus::Finished;
        if (seq.finish_reason == FinishReason::None) {
            seq.finish_reason = (seq.generated_tokens.size() >= seq.options.max_new_tokens)
                                    ? FinishReason::MaxNewTokens
                                    : FinishReason::ContextLengthExceeded;
        }
    } else {
        seq.status = SequenceStatus::Decoding;
    }
    return true;
}

bool BatchScheduler::step_sequence_decode(BatchSequenceState& seq, BatchStepMetrics& metrics) {
    metrics.decoding_sequences++;
    const std::uint32_t current_token = seq.generated_tokens.back();
    if (!runtime_->generator().forward_step(&current_token, 1, seq.current_position,
                                            logits_scratch_.data(), seq.kv_cache)) {
        seq.status = SequenceStatus::Failed;
        seq.finish_reason = FinishReason::Aborted;
        return false;
    }
    seq.current_position += 1;

    SamplerConfig sampler_cfg = seq.options.sampler;
    if (seq.options.repetition_penalty > 1.0F && sampler_cfg.repetition_penalty == 1.0F) {
        sampler_cfg.repetition_penalty = seq.options.repetition_penalty;
    }

    const std::uint32_t next_token = seq.sampler.sample(
        logits_scratch_.data(), model_config_.vocab_size, seq.all_tokens, sampler_cfg);
    metrics.tokens_generated_in_step++;

    if (model_config_.eos_token_id >= 0 &&
        next_token == static_cast<std::uint32_t>(model_config_.eos_token_id)) {
        seq.reached_eos = true;
        seq.finish_reason = FinishReason::EosToken;
    }
    for (const std::uint32_t stop_id : seq.options.stop_token_ids) {
        if (next_token == stop_id) {
            seq.reached_eos = true;
            seq.finish_reason = FinishReason::StopToken;
            break;
        }
    }

    seq.generated_tokens.push_back(next_token);
    seq.all_tokens.push_back(next_token);

    if (!emit_chunk(seq, next_token, seq.reached_eos)) {
        seq.status = SequenceStatus::Finished;
        seq.finish_reason = FinishReason::Aborted;
        return true;
    }

    if ((seq.reached_eos && seq.options.early_stopping) ||
        seq.generated_tokens.size() >= seq.options.max_new_tokens ||
        seq.current_position >= config_.max_sequence_length) {
        seq.status = SequenceStatus::Finished;
        if (seq.finish_reason == FinishReason::None) {
            seq.finish_reason = (seq.generated_tokens.size() >= seq.options.max_new_tokens)
                                    ? FinishReason::MaxNewTokens
                                    : FinishReason::ContextLengthExceeded;
        }
    }
    return true;
}

bool BatchScheduler::step(BatchStepMetrics& metrics) {
    metrics = {};
    if (active_sequences_.empty() && pending_requests_.empty()) {
        return true;
    }
    if (runtime_ == nullptr) return false;

    // 1. Admit new requests up to available batch slots
    const std::size_t kv_head_cnt = (model_config_.block.kv_head_count > 0)
                                        ? model_config_.block.kv_head_count
                                        : model_config_.block.head_count;

    while (active_sequences_.size() < config_.max_batch_size && !pending_requests_.empty()) {
        BatchRequest req = std::move(pending_requests_.front());
        pending_requests_.pop_front();

        BatchSequenceState seq;
        seq.request_id = req.request_id;
        seq.prompt_tokens = std::move(req.prompt_tokens);
        seq.all_tokens = seq.prompt_tokens;
        seq.options = req.options;
        seq.stream_callback = std::move(req.stream_callback);
        seq.status = SequenceStatus::Prefilling;
        seq.finish_reason = FinishReason::None;
        seq.current_position = 0;
        seq.reached_eos = false;

        if (!seq.kv_cache.configure(model_config_.layer_count, kv_head_cnt,
                                    model_config_.block.head_dimension,
                                    config_.max_sequence_length,
                                    runtime_->max_kv_bytes())) {
            seq.status = SequenceStatus::Failed;
            seq.finish_reason = FinishReason::Aborted;
        } else if (!seq.sampler.configure(model_config_.vocab_size)) {
            seq.status = SequenceStatus::Failed;
            seq.finish_reason = FinishReason::Aborted;
        }

        active_sequences_.push_back(std::move(seq));
    }

    if (active_sequences_.empty()) {
        return true;
    }

    // 2. Step prefilling sequences
    for (auto& seq : active_sequences_) {
        if (seq.status == SequenceStatus::Prefilling) {
            step_sequence_prefill(seq, metrics);
        }
    }

    // 3. Step decoding sequences
    for (auto& seq : active_sequences_) {
        if (seq.status == SequenceStatus::Decoding) {
            step_sequence_decode(seq, metrics);
        }
    }

    // 4. Retire completed or failed sequences
    std::vector<BatchSequenceState> remaining;
    remaining.reserve(active_sequences_.size());

    for (auto& seq : active_sequences_) {
        if (seq.status == SequenceStatus::Finished || seq.status == SequenceStatus::Failed) {
            if (runtime_->tokenizer().configured() && seq.text.empty() &&
                !seq.generated_tokens.empty()) {
                runtime_->detokenize(seq.generated_tokens, seq.text);
            }
            GenerationResult res;
            res.prompt_tokens = std::move(seq.prompt_tokens);
            res.generated_tokens = std::move(seq.generated_tokens);
            res.all_tokens = std::move(seq.all_tokens);
            res.text = std::move(seq.text);
            res.reached_eos = seq.reached_eos;
            res.prompt_token_count = res.prompt_tokens.size();
            res.generated_token_count = res.generated_tokens.size();
            completed_results_[seq.request_id] = std::move(res);
            metrics.completed_sequences++;
        } else {
            remaining.push_back(std::move(seq));
        }
    }

    active_sequences_ = std::move(remaining);
    metrics.active_sequences = active_sequences_.size();
    return true;
}

std::vector<GenerationResult> BatchScheduler::generate_batch(
    const std::vector<std::vector<std::uint32_t>>& prompts,
    const std::vector<GenerationOptions>& options,
    std::vector<StreamCallback> stream_callbacks) {
    std::vector<GenerationResult> results;
    if (runtime_ == nullptr || prompts.empty() || prompts.size() != options.size()) {
        return results;
    }

    std::vector<std::uint64_t> ids;
    ids.reserve(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        BatchRequest req;
        req.prompt_tokens = prompts[i];
        req.options = options[i];
        if (i < stream_callbacks.size()) {
            req.stream_callback = std::move(stream_callbacks[i]);
        }
        const std::uint64_t id = enqueue(std::move(req));
        if (id == 0) {
            return {};
        }
        ids.push_back(id);
    }

    BatchStepMetrics metrics;
    while (!is_idle()) {
        if (!step(metrics)) {
            break;
        }
    }

    results.reserve(ids.size());
    for (const std::uint64_t id : ids) {
        GenerationResult res;
        if (get_result(id, res)) {
            results.push_back(std::move(res));
        } else {
            results.push_back({});
        }
    }
    return results;
}

bool BatchScheduler::get_result(std::uint64_t request_id, GenerationResult& result) const {
    const auto it = completed_results_.find(request_id);
    if (it == completed_results_.end()) return false;
    result = it->second;
    return true;
}

void BatchScheduler::reset() noexcept {
    pending_requests_.clear();
    active_sequences_.clear();
    completed_results_.clear();
    next_request_id_ = 1;
}

} // namespace aabhas::inference
