#ifndef AABHAS_INFERENCE_QUANTIZED_KERNELS_HPP
#define AABHAS_INFERENCE_QUANTIZED_KERNELS_HPP

#include "model/model_index.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aabhas::inference::kernels {

/** IEEE 754 half-precision float to float32. */
float fp16_to_fp32(std::uint16_t h) noexcept;

/** Float32 to IEEE 754 half-precision float. */
std::uint16_t fp32_to_fp16(float f) noexcept;

/** BFloat16 to float32. */
float bf16_to_fp32(std::uint16_t b) noexcept;

/** Float32 to BFloat16. */
std::uint16_t fp32_to_bf16(float f) noexcept;

/** Whether a dtype is a quantized block format. */
bool is_quantized_dtype(aabhas::model::TensorDType dtype) noexcept;

/** Whether a dtype can be consumed as weights by inference kernels. */
bool is_supported_weight_dtype(aabhas::model::TensorDType dtype) noexcept;

/** Number of logical elements per block (32 for Q4/Q5/Q8, 1 for scalars). */
std::size_t block_elements(aabhas::model::TensorDType dtype) noexcept;

/** Number of storage bytes per block. */
std::size_t block_bytes(aabhas::model::TensorDType dtype) noexcept;

#pragma pack(push, 1)

/** Standard GGML Q4_0 block (32 values, 18 bytes). */
struct BlockQ4_0 {
    std::uint16_t d;     // delta in fp16
    std::uint8_t qs[16]; // 32 4-bit nibbles
};

/** Standard GGML Q4_1 block (32 values, 20 bytes). */
struct BlockQ4_1 {
    std::uint16_t d;     // delta in fp16
    std::uint16_t m;     // min in fp16
    std::uint8_t qs[16]; // 32 4-bit nibbles
};

/** Standard GGML Q5_0 block (32 values, 22 bytes). */
struct BlockQ5_0 {
    std::uint16_t d;     // delta in fp16
    std::uint8_t qh[4];  // 32 high bits
    std::uint8_t qs[16]; // 32 low 4-bit nibbles
};

/** Standard GGML Q5_1 block (32 values, 24 bytes). */
struct BlockQ5_1 {
    std::uint16_t d;     // delta in fp16
    std::uint16_t m;     // min in fp16
    std::uint8_t qh[4];  // 32 high bits
    std::uint8_t qs[16]; // 32 low 4-bit nibbles
};

/** Standard GGML Q8_0 block (32 values, 34 bytes). */
struct BlockQ8_0 {
    std::uint16_t d;    // delta in fp16
    std::int8_t qs[32]; // 32 int8 values
};

/** Standard GGML Q8_1 block (32 values, 40 bytes). */
struct BlockQ8_1 {
    float d;            // delta in f32
    float s;            // sum in f32
    std::int8_t qs[32]; // 32 int8 values
};

#pragma pack(pop)

/** Dequantize one 32-element Q4_0 block into 32 float32 values. */
void dequantize_q4_0_block(const BlockQ4_0* block, float* output) noexcept;

/** Dequantize one 32-element Q4_1 block into 32 float32 values. */
void dequantize_q4_1_block(const BlockQ4_1* block, float* output) noexcept;

/** Dequantize one 32-element Q5_0 block into 32 float32 values. */
void dequantize_q5_0_block(const BlockQ5_0* block, float* output) noexcept;

/** Dequantize one 32-element Q5_1 block into 32 float32 values. */
void dequantize_q5_1_block(const BlockQ5_1* block, float* output) noexcept;

/** Dequantize one 32-element Q8_0 block into 32 float32 values. */
void dequantize_q8_0_block(const BlockQ8_0* block, float* output) noexcept;

/** Dequantize one 32-element Q8_1 block into 32 float32 values. */
void dequantize_q8_1_block(const BlockQ8_1* block, float* output) noexcept;

/**
 * Dequantize an entire contiguous row of quantized or half-precision weights.
 *
 * Verifies bounds, block alignment, and non-null pointers.
 */
bool dequantize_row(aabhas::model::TensorDType dtype,
                    const void* source,
                    std::size_t source_bytes,
                    float* destination,
                    std::size_t element_count) noexcept;

/**
 * Compute the dot product between one quantized/float row and a float input vector.
 *
 * Direct accumulation avoids allocating full dequantized vectors.
 */
bool vec_dot(aabhas::model::TensorDType dtype,
             const void* weights,
             const float* input,
             std::size_t count,
             float& result) noexcept;

/**
 * General row-major matrix-vector multiplication Y = W * X.
 *
 * W has dimensions [rows, columns] and data type `dtype`.
 * X has dimension [columns] and Y has dimension [rows].
 */
bool gemv_quantized(aabhas::model::TensorDType dtype,
                    const void* weights,
                    std::size_t rows,
                    std::size_t columns,
                    const float* input,
                    float* output) noexcept;

/**
 * Quantize a float array into a target format (used for synthetic test generation).
 */
bool quantize_row(aabhas::model::TensorDType dtype,
                  const float* source,
                  std::size_t count,
                  void* destination,
                  std::size_t destination_bytes) noexcept;

/**
 * Stream decoder that consumes raw chunks of arbitrary size and dequantizes
 * complete blocks directly into destination or yields them to a callback.
 */
class QuantizedStreamDecoder final {
public:
    QuantizedStreamDecoder() noexcept = default;

    bool init(aabhas::model::TensorDType dtype, float* destination,
              std::size_t target_elements) noexcept;

    bool feed(const void* chunk_data, std::size_t chunk_bytes) noexcept;

    /** Feed chunks and invoke a callback for every decoded block. */
    bool feed_with_callback(
        const void* chunk_data, std::size_t chunk_bytes,
        const std::function<void(const float* block_values,
                                 std::size_t block_elem_count,
                                 std::size_t element_offset)>& on_block) noexcept;

    bool finish() const noexcept;

    std::size_t elements_decoded() const noexcept { return elements_decoded_; }

private:
    bool process_single_block(
        const void* block_raw,
        const std::function<void(const float* block_values,
                                 std::size_t block_elem_count,
                                 std::size_t element_offset)>& on_block) noexcept;

    aabhas::model::TensorDType dtype_{aabhas::model::TensorDType::Unknown};
    float* destination_{nullptr};
    std::size_t target_elements_{0};
    std::size_t elements_decoded_{0};
    std::size_t block_elems_{0};
    std::size_t block_byte_size_{0};
    std::uint8_t buffer_[64]{};
    std::size_t buffer_len_{0};
};

} // namespace aabhas::inference::kernels

#endif // AABHAS_INFERENCE_QUANTIZED_KERNELS_HPP