#ifndef AABHAS_INFERENCE_SAMPLER_HPP
#define AABHAS_INFERENCE_SAMPLER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace aabhas::inference {

/**
 * Stochastic and greedy sampling configuration.
 */
struct SamplerConfig {
    float temperature{1.0F};        // <= 1.0e-4F activates greedy argmax
    std::size_t top_k{0};           // 0 = disabled; > 0 keeps top K logits
    float top_p{1.0F};              // 1.0F = disabled; (0.0F, 1.0F) nucleus threshold
    float min_p{0.0F};              // 0.0F = disabled; (0.0F, 1.0F) min relative probability
    float repetition_penalty{1.0F}; // 1.0F = disabled; > 1.0 penalizes repeats
    float presence_penalty{0.0F};   // 0.0F = disabled; subtracted if present
    float frequency_penalty{0.0F};  // 0.0F = disabled; scaled by token occurrence count
    std::uint64_t seed{1337ULL};    // PRNG seed for reproducible sampling
};

/**
 * High-performance, zero-allocation token sampler.
 *
 * Implements:
 * - Repetition, frequency, and presence penalties
 * - Temperature scaling with fast greedy fallback
 * - Top-K truncation via bounded partial sorting
 * - Top-P (nucleus) cumulative mass filtering
 * - Min-P adaptive threshold filtering
 * - Numerically stable max-subtracted softmax
 * - Categorical PRNG distribution sampling
 */
class Sampler final {
public:
    explicit Sampler(std::uint64_t seed = 1337ULL) noexcept;
    ~Sampler() noexcept = default;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    /** Pre-allocate internal candidate and probability buffers for vocab_size. */
    bool configure(std::size_t vocab_size);

    /** Set or re-seed the pseudo-random generator. */
    void set_seed(std::uint64_t seed) noexcept;

    /**
     * Apply penalties, temperature, filtering (top-k, top-p, min-p), and sample a token.
     * Modifies the input logits buffer in-place.
     */
    std::uint32_t sample(float* logits,
                         std::size_t vocab_size,
                         const std::vector<std::uint32_t>& history,
                         const SamplerConfig& config);

    /** Fast greedy argmax lookup with zero state mutation. */
    static std::uint32_t argmax(const float* logits, std::size_t count) noexcept;

    /** Apply repetition, presence, and frequency penalties to logits in-place. */
    static void apply_penalties(float* logits,
                                std::size_t vocab_size,
                                const std::vector<std::uint32_t>& history,
                                float repetition_penalty,
                                float presence_penalty = 0.0F,
                                float frequency_penalty = 0.0F) noexcept;

    /** In-place numerically stable softmax over a slice of float values. */
    static void softmax(float* values, std::size_t count) noexcept;

    std::size_t vocab_size() const noexcept { return vocab_size_; }

private:
    std::size_t vocab_size_{0};
    std::mt19937_64 rng_;
    // Pre-allocated scratch buffers to ensure zero heap allocations during sample()
    std::vector<std::pair<float, std::uint32_t>> candidates_{};
    std::vector<float> probs_{};
    std::vector<std::uint32_t> token_counts_{};
};

/** Streaming token chunk delivered via StreamCallback. */
struct StreamChunk {
    std::uint32_t token_id{0};
    std::string text_delta{};
    std::size_t token_index{0};
    bool is_eos{false};
};

/** Callback invoked per generated token during streaming; return false to cancel. */
using StreamCallback = std::function<bool(const StreamChunk& chunk)>;

} // namespace aabhas::inference

#endif // AABHAS_INFERENCE_SAMPLER_HPP
