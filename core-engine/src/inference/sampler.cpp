#include "inference/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aabhas::inference {

Sampler::Sampler(std::uint64_t seed) noexcept
    : rng_(seed) {}

bool Sampler::configure(std::size_t vocab_size) {
    if (vocab_size == 0) return false;
    try {
        vocab_size_ = vocab_size;
        candidates_.assign(vocab_size, {0.0F, 0});
        probs_.assign(vocab_size, 0.0F);
        token_counts_.assign(vocab_size, 0);
        return true;
    } catch (...) {
        vocab_size_ = 0;
        candidates_.clear();
        probs_.clear();
        token_counts_.clear();
        return false;
    }
}

void Sampler::set_seed(std::uint64_t seed) noexcept {
    rng_.seed(seed);
}

std::uint32_t Sampler::argmax(const float* logits, std::size_t count) noexcept {
    if (logits == nullptr || count == 0) return 0;
    std::size_t best_idx = 0;
    float best_val = logits[0];
    for (std::size_t i = 1; i < count; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return static_cast<std::uint32_t>(best_idx);
}

void Sampler::apply_penalties(float* logits,
                              std::size_t vocab_size,
                              const std::vector<std::uint32_t>& history,
                              float repetition_penalty,
                              float presence_penalty,
                              float frequency_penalty) noexcept {
    if (logits == nullptr || vocab_size == 0 || history.empty()) return;
    const bool has_rep = repetition_penalty > 1.0F;
    const bool has_pres = presence_penalty != 0.0F;
    const bool has_freq = frequency_penalty != 0.0F;
    if (!has_rep && !has_pres && !has_freq) return;

    // Direct iteration over history
    for (const std::uint32_t token : history) {
        if (token < vocab_size) {
            if (has_rep) {
                if (logits[token] > 0.0F) {
                    logits[token] /= repetition_penalty;
                } else {
                    logits[token] *= repetition_penalty;
                }
            }
            if (has_pres) {
                logits[token] -= presence_penalty;
            }
            if (has_freq) {
                logits[token] -= frequency_penalty;
            }
        }
    }
}

void Sampler::softmax(float* values, std::size_t count) noexcept {
    if (values == nullptr || count == 0) return;
    float max_val = values[0];
    for (std::size_t i = 1; i < count; ++i) {
        if (values[i] > max_val) max_val = values[i];
    }
    float sum = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::exp(values[i] - max_val);
        sum += values[i];
    }
    if (sum > 0.0F) {
        const float inv_sum = 1.0F / sum;
        for (std::size_t i = 0; i < count; ++i) {
            values[i] *= inv_sum;
        }
    }
}

std::uint32_t Sampler::sample(float* logits,
                             std::size_t vocab_size,
                             const std::vector<std::uint32_t>& history,
                             const SamplerConfig& config) {
    if (logits == nullptr || vocab_size == 0) return 0;
    if (vocab_size_ < vocab_size) {
        if (!configure(vocab_size)) {
            return argmax(logits, vocab_size);
        }
    }

    // 1. Penalties
    if (!history.empty() &&
        (config.repetition_penalty > 1.0F || config.presence_penalty != 0.0F ||
         config.frequency_penalty != 0.0F)) {
        if (config.frequency_penalty != 0.0F) {
            // Count frequencies using zero-allocation token_counts_
            for (const std::uint32_t token : history) {
                if (token < vocab_size) {
                    ++token_counts_[token];
                }
            }
            for (const std::uint32_t token : history) {
                if (token < vocab_size && token_counts_[token] > 0) {
                    const std::uint32_t count = token_counts_[token];
                    if (config.repetition_penalty > 1.0F) {
                        if (logits[token] > 0.0F) {
                            logits[token] /= config.repetition_penalty;
                        } else {
                            logits[token] *= config.repetition_penalty;
                        }
                    }
                    logits[token] -= config.presence_penalty;
                    logits[token] -= config.frequency_penalty * static_cast<float>(count);
                    token_counts_[token] = 0; // reset
                }
            }
        } else {
            apply_penalties(logits, vocab_size, history,
                            config.repetition_penalty,
                            config.presence_penalty,
                            config.frequency_penalty);
        }
    }

    // 2. Greedy fallback check
    if (config.temperature <= 1.0e-4F || config.top_k == 1) {
        return argmax(logits, vocab_size);
    }

    // 3. Temperature scaling
    const float inv_temp = 1.0F / config.temperature;
    for (std::size_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    // 4. Populate candidate array
    for (std::size_t i = 0; i < vocab_size; ++i) {
        candidates_[i] = {logits[i], static_cast<std::uint32_t>(i)};
    }

    // 5. Top-K truncation
    std::size_t active_count = vocab_size;
    if (config.top_k > 0 && config.top_k < active_count) {
        std::partial_sort(candidates_.begin(),
                          candidates_.begin() + config.top_k,
                          candidates_.begin() + active_count,
                          [](const std::pair<float, std::uint32_t>& a,
                             const std::pair<float, std::uint32_t>& b) {
                              return a.first > b.first;
                          });
        active_count = config.top_k;
    } else {
        std::sort(candidates_.begin(),
                  candidates_.begin() + active_count,
                  [](const std::pair<float, std::uint32_t>& a,
                     const std::pair<float, std::uint32_t>& b) {
                      return a.first > b.first;
                  });
    }

    // 6. Softmax over active candidates (first candidate is guaranteed maximum)
    const float max_val = candidates_[0].first;
    float sum_exp = 0.0F;
    for (std::size_t i = 0; i < active_count; ++i) {
        const float p = std::exp(candidates_[i].first - max_val);
        probs_[i] = p;
        sum_exp += p;
    }
    if (sum_exp <= 0.0F || !std::isfinite(sum_exp)) {
        return candidates_[0].second;
    }
    const float inv_sum = 1.0F / sum_exp;
    for (std::size_t i = 0; i < active_count; ++i) {
        probs_[i] *= inv_sum;
    }

    // 7. Min-P adaptive threshold filtering
    if (config.min_p > 0.0F && config.min_p < 1.0F) {
        const float threshold = probs_[0] * config.min_p;
        std::size_t kept = 0;
        float kept_sum = 0.0F;
        for (std::size_t i = 0; i < active_count; ++i) {
            if (probs_[i] >= threshold) {
                ++kept;
                kept_sum += probs_[i];
            } else {
                break;
            }
        }
        if (kept > 0 && kept_sum > 0.0F) {
            active_count = kept;
            const float inv_kept = 1.0F / kept_sum;
            for (std::size_t i = 0; i < active_count; ++i) {
                probs_[i] *= inv_kept;
            }
        }
    }

    // 8. Top-P (nucleus) cumulative mass filtering
    if (config.top_p > 0.0F && config.top_p < 1.0F) {
        std::size_t kept = 0;
        float cumsum = 0.0F;
        for (std::size_t i = 0; i < active_count; ++i) {
            cumsum += probs_[i];
            ++kept;
            if (cumsum >= config.top_p) {
                break;
            }
        }
        if (kept > 0 && cumsum > 0.0F) {
            active_count = kept;
            const float inv_cum = 1.0F / cumsum;
            for (std::size_t i = 0; i < active_count; ++i) {
                probs_[i] *= inv_cum;
            }
        }
    }

    // 9. Categorical sampling with PRNG
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const float u = dist(rng_);
    float cum = 0.0F;
    std::uint32_t selected = candidates_[active_count - 1].second;
    for (std::size_t i = 0; i < active_count; ++i) {
        cum += probs_[i];
        if (u <= cum) {
            selected = candidates_[i].second;
            break;
        }
    }

    return selected;
}

} // namespace aabhas::inference
