#include "inference/quantized_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace aabhas::inference::kernels {

float fp16_to_fp32(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1FU;
    const std::uint32_t mant = h & 0x03FFU;
    std::uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            std::uint32_t m = mant;
            std::int32_t e = 127 - 15 + 1;
            while ((m & 0x0400U) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFU;
            f = sign | (static_cast<std::uint32_t>(e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000U | (mant << 13);
    } else {
        const std::uint32_t e = exp + (127 - 15);
        f = sign | (e << 23) | (mant << 13);
    }
    float result = 0.0F;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

std::uint16_t fp32_to_fp16(float f) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(float));
    const std::uint32_t sign = (u >> 16) & 0x8000U;
    const std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFFU) - 127 + 15;
    std::uint32_t mant = u & 0x007FFFFFU;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mant = (mant | 0x00800000U) >> (1 - exp);
        return static_cast<std::uint16_t>(sign | (mant >> 13));
    } else if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U | (mant != 0 ? 1 : 0));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                      (mant >> 13));
}

float bf16_to_fp32(std::uint16_t b) noexcept {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float result = 0.0F;
    std::memcpy(&result, &u, sizeof(float));
    return result;
}

std::uint16_t fp32_to_bf16(float f) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(float));
    return static_cast<std::uint16_t>(u >> 16);
}

bool is_quantized_dtype(aabhas::model::TensorDType dtype) noexcept {
    switch (dtype) {
    case aabhas::model::TensorDType::Q4_0:
    case aabhas::model::TensorDType::Q4_1:
    case aabhas::model::TensorDType::Q5_0:
    case aabhas::model::TensorDType::Q5_1:
    case aabhas::model::TensorDType::Q8_0:
    case aabhas::model::TensorDType::Q8_1:
        return true;
    default:
        return false;
    }
}

bool is_supported_weight_dtype(aabhas::model::TensorDType dtype) noexcept {
    switch (dtype) {
    case aabhas::model::TensorDType::F32:
    case aabhas::model::TensorDType::F16:
    case aabhas::model::TensorDType::BF16:
    case aabhas::model::TensorDType::Q4_0:
    case aabhas::model::TensorDType::Q4_1:
    case aabhas::model::TensorDType::Q5_0:
    case aabhas::model::TensorDType::Q5_1:
    case aabhas::model::TensorDType::Q8_0:
    case aabhas::model::TensorDType::Q8_1:
        return true;
    default:
        return false;
    }
}

std::size_t block_elements(aabhas::model::TensorDType dtype) noexcept {
    switch (dtype) {
    case aabhas::model::TensorDType::Q4_0:
    case aabhas::model::TensorDType::Q4_1:
    case aabhas::model::TensorDType::Q5_0:
    case aabhas::model::TensorDType::Q5_1:
    case aabhas::model::TensorDType::Q8_0:
    case aabhas::model::TensorDType::Q8_1:
        return 32;
    case aabhas::model::TensorDType::F32:
    case aabhas::model::TensorDType::F16:
    case aabhas::model::TensorDType::BF16:
        return 1;
    default:
        return 0;
    }
}

std::size_t block_bytes(aabhas::model::TensorDType dtype) noexcept {
    switch (dtype) {
    case aabhas::model::TensorDType::Q4_0:
        return 18;
    case aabhas::model::TensorDType::Q4_1:
        return 20;
    case aabhas::model::TensorDType::Q5_0:
        return 22;
    case aabhas::model::TensorDType::Q5_1:
        return 24;
    case aabhas::model::TensorDType::Q8_0:
        return 34;
    case aabhas::model::TensorDType::Q8_1:
        return 40;
    case aabhas::model::TensorDType::F32:
        return 4;
    case aabhas::model::TensorDType::F16:
    case aabhas::model::TensorDType::BF16:
        return 2;
    default:
        return 0;
    }
}

void dequantize_q4_0_block(const BlockQ4_0* block, float* output) noexcept {
    const float d = fp16_to_fp32(block->d);
    for (std::size_t i = 0; i < 16; ++i) {
        const std::uint8_t q = block->qs[i];
        const std::int8_t v0 = static_cast<std::int8_t>(q & 0x0FU) - 8;
        const std::int8_t v1 = static_cast<std::int8_t>(q >> 4) - 8;
        output[i] = static_cast<float>(v0) * d;
        output[i + 16] = static_cast<float>(v1) * d;
    }
}

void dequantize_q4_1_block(const BlockQ4_1* block, float* output) noexcept {
    const float d = fp16_to_fp32(block->d);
    const float m = fp16_to_fp32(block->m);
    for (std::size_t i = 0; i < 16; ++i) {
        const std::uint8_t q = block->qs[i];
        const float v0 = static_cast<float>(q & 0x0FU);
        const float v1 = static_cast<float>(q >> 4);
        output[i] = v0 * d + m;
        output[i + 16] = v1 * d + m;
    }
}

void dequantize_q5_0_block(const BlockQ5_0* block, float* output) noexcept {
    const float d = fp16_to_fp32(block->d);
    std::uint32_t qh = 0;
    std::memcpy(&qh, block->qh, sizeof(qh));
    for (std::size_t i = 0; i < 16; ++i) {
        const std::uint8_t q = block->qs[i];
        const std::uint8_t h0 = static_cast<std::uint8_t>((qh >> i) & 1U);
        const std::uint8_t h1 = static_cast<std::uint8_t>((qh >> (i + 16)) & 1U);
        const std::int8_t v0 = static_cast<std::int8_t>((q & 0x0FU) | (h0 << 4)) - 16;
        const std::int8_t v1 = static_cast<std::int8_t>((q >> 4) | (h1 << 4)) - 16;
        output[i] = static_cast<float>(v0) * d;
        output[i + 16] = static_cast<float>(v1) * d;
    }
}

void dequantize_q5_1_block(const BlockQ5_1* block, float* output) noexcept {
    const float d = fp16_to_fp32(block->d);
    const float m = fp16_to_fp32(block->m);
    std::uint32_t qh = 0;
    std::memcpy(&qh, block->qh, sizeof(qh));
    for (std::size_t i = 0; i < 16; ++i) {
        const std::uint8_t q = block->qs[i];
        const std::uint8_t h0 = static_cast<std::uint8_t>((qh >> i) & 1U);
        const std::uint8_t h1 = static_cast<std::uint8_t>((qh >> (i + 16)) & 1U);
        const float v0 = static_cast<float>((q & 0x0FU) | (h0 << 4));
        const float v1 = static_cast<float>((q >> 4) | (h1 << 4));
        output[i] = v0 * d + m;
        output[i + 16] = v1 * d + m;
    }
}

void dequantize_q8_0_block(const BlockQ8_0* block, float* output) noexcept {
    const float d = fp16_to_fp32(block->d);
    for (std::size_t i = 0; i < 32; ++i) {
        output[i] = static_cast<float>(block->qs[i]) * d;
    }
}

void dequantize_q8_1_block(const BlockQ8_1* block, float* output) noexcept {
    const float d = block->d;
    for (std::size_t i = 0; i < 32; ++i) {
        output[i] = static_cast<float>(block->qs[i]) * d;
    }
}

bool dequantize_row(aabhas::model::TensorDType dtype,
                    const void* source,
                    std::size_t source_bytes,
                    float* destination,
                    std::size_t element_count) noexcept {
    if (source == nullptr || destination == nullptr || element_count == 0) {
        return false;
    }
    if (!is_supported_weight_dtype(dtype)) return false;
    const std::size_t elems_per_blk = block_elements(dtype);
    const std::size_t bytes_per_blk = block_bytes(dtype);
    if (element_count % elems_per_blk != 0) return false;
    const std::size_t num_blocks = element_count / elems_per_blk;
    const std::size_t required_bytes = num_blocks * bytes_per_blk;
    if (source_bytes < required_bytes) return false;

    const std::uint8_t* src_ptr = static_cast<const std::uint8_t*>(source);
    switch (dtype) {
    case aabhas::model::TensorDType::F32:
        std::memcpy(destination, source, element_count * sizeof(float));
        return true;
    case aabhas::model::TensorDType::F16: {
        const std::uint16_t* h = static_cast<const std::uint16_t*>(source);
        for (std::size_t i = 0; i < element_count; ++i) {
            destination[i] = fp16_to_fp32(h[i]);
        }
        return true;
    }
    case aabhas::model::TensorDType::BF16: {
        const std::uint16_t* b = static_cast<const std::uint16_t*>(source);
        for (std::size_t i = 0; i < element_count; ++i) {
            destination[i] = bf16_to_fp32(b[i]);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q4_0: {
        const BlockQ4_0* blocks = static_cast<const BlockQ4_0*>(source);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            dequantize_q4_0_block(&blocks[b], destination + b * 32);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q4_1: {
        const BlockQ4_1* blocks = static_cast<const BlockQ4_1*>(source);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            dequantize_q4_1_block(&blocks[b], destination + b * 32);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q5_0: {
        const BlockQ5_0* blocks = static_cast<const BlockQ5_0*>(source);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            dequantize_q5_0_block(&blocks[b], destination + b * 32);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q5_1: {
        const BlockQ5_1* blocks = static_cast<const BlockQ5_1*>(source);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            dequantize_q5_1_block(&blocks[b], destination + b * 32);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q8_0: {
        const BlockQ8_0* blocks = static_cast<const BlockQ8_0*>(source);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            dequantize_q8_0_block(&blocks[b], destination + b * 32);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q8_1: {
        const BlockQ8_1* blocks = static_cast<const BlockQ8_1*>(source);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            dequantize_q8_1_block(&blocks[b], destination + b * 32);
        }
        return true;
    }
    default:
        return false;
    }
}

bool vec_dot(aabhas::model::TensorDType dtype,
             const void* weights,
             const float* input,
             std::size_t count,
             float& result) noexcept {
    if (weights == nullptr || input == nullptr || count == 0) {
        return false;
    }
    if (!is_supported_weight_dtype(dtype)) return false;
    const std::size_t elems_per_blk = block_elements(dtype);
    if (count % elems_per_blk != 0) return false;
    const std::size_t num_blocks = count / elems_per_blk;

    double acc = 0.0;
    switch (dtype) {
    case aabhas::model::TensorDType::F32: {
        const float* w = static_cast<const float*>(weights);
        for (std::size_t i = 0; i < count; ++i) {
            acc += static_cast<double>(w[i]) * static_cast<double>(input[i]);
        }
        break;
    }
    case aabhas::model::TensorDType::F16: {
        const std::uint16_t* h = static_cast<const std::uint16_t*>(weights);
        for (std::size_t i = 0; i < count; ++i) {
            acc += static_cast<double>(fp16_to_fp32(h[i])) * static_cast<double>(input[i]);
        }
        break;
    }
    case aabhas::model::TensorDType::BF16: {
        const std::uint16_t* b = static_cast<const std::uint16_t*>(weights);
        for (std::size_t i = 0; i < count; ++i) {
            acc += static_cast<double>(bf16_to_fp32(b[i])) * static_cast<double>(input[i]);
        }
        break;
    }
    case aabhas::model::TensorDType::Q4_0: {
        const BlockQ4_0* blocks = static_cast<const BlockQ4_0*>(weights);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float d = fp16_to_fp32(blocks[b].d);
            const float* xb = input + b * 32;
            double block_sum = 0.0;
            for (std::size_t i = 0; i < 16; ++i) {
                const std::uint8_t q = blocks[b].qs[i];
                const std::int8_t v0 = static_cast<std::int8_t>(q & 0x0FU) - 8;
                const std::int8_t v1 = static_cast<std::int8_t>(q >> 4) - 8;
                block_sum += static_cast<double>(v0) * static_cast<double>(xb[i]) +
                             static_cast<double>(v1) * static_cast<double>(xb[i + 16]);
            }
            acc += block_sum * static_cast<double>(d);
        }
        break;
    }
    case aabhas::model::TensorDType::Q4_1: {
        const BlockQ4_1* blocks = static_cast<const BlockQ4_1*>(weights);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float d = fp16_to_fp32(blocks[b].d);
            const float m = fp16_to_fp32(blocks[b].m);
            const float* xb = input + b * 32;
            double sum_v = 0.0;
            double sum_x = 0.0;
            for (std::size_t i = 0; i < 16; ++i) {
                const std::uint8_t q = blocks[b].qs[i];
                const double v0 = static_cast<double>(q & 0x0FU);
                const double v1 = static_cast<double>(q >> 4);
                sum_v += v0 * static_cast<double>(xb[i]) +
                         v1 * static_cast<double>(xb[i + 16]);
                sum_x += static_cast<double>(xb[i]) + static_cast<double>(xb[i + 16]);
            }
            acc += sum_v * static_cast<double>(d) + sum_x * static_cast<double>(m);
        }
        break;
    }
    case aabhas::model::TensorDType::Q5_0: {
        const BlockQ5_0* blocks = static_cast<const BlockQ5_0*>(weights);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float d = fp16_to_fp32(blocks[b].d);
            std::uint32_t qh = 0;
            std::memcpy(&qh, blocks[b].qh, sizeof(qh));
            const float* xb = input + b * 32;
            double block_sum = 0.0;
            for (std::size_t i = 0; i < 16; ++i) {
                const std::uint8_t q = blocks[b].qs[i];
                const std::uint8_t h0 = static_cast<std::uint8_t>((qh >> i) & 1U);
                const std::uint8_t h1 = static_cast<std::uint8_t>((qh >> (i + 16)) & 1U);
                const std::int8_t v0 = static_cast<std::int8_t>((q & 0x0FU) | (h0 << 4)) - 16;
                const std::int8_t v1 = static_cast<std::int8_t>((q >> 4) | (h1 << 4)) - 16;
                block_sum += static_cast<double>(v0) * static_cast<double>(xb[i]) +
                             static_cast<double>(v1) * static_cast<double>(xb[i + 16]);
            }
            acc += block_sum * static_cast<double>(d);
        }
        break;
    }
    case aabhas::model::TensorDType::Q5_1: {
        const BlockQ5_1* blocks = static_cast<const BlockQ5_1*>(weights);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float d = fp16_to_fp32(blocks[b].d);
            const float m = fp16_to_fp32(blocks[b].m);
            std::uint32_t qh = 0;
            std::memcpy(&qh, blocks[b].qh, sizeof(qh));
            const float* xb = input + b * 32;
            double sum_v = 0.0;
            double sum_x = 0.0;
            for (std::size_t i = 0; i < 16; ++i) {
                const std::uint8_t q = blocks[b].qs[i];
                const std::uint8_t h0 = static_cast<std::uint8_t>((qh >> i) & 1U);
                const std::uint8_t h1 = static_cast<std::uint8_t>((qh >> (i + 16)) & 1U);
                const double v0 = static_cast<double>((q & 0x0FU) | (h0 << 4));
                const double v1 = static_cast<double>((q >> 4) | (h1 << 4));
                sum_v += v0 * static_cast<double>(xb[i]) +
                         v1 * static_cast<double>(xb[i + 16]);
                sum_x += static_cast<double>(xb[i]) + static_cast<double>(xb[i + 16]);
            }
            acc += sum_v * static_cast<double>(d) + sum_x * static_cast<double>(m);
        }
        break;
    }
    case aabhas::model::TensorDType::Q8_0: {
        const BlockQ8_0* blocks = static_cast<const BlockQ8_0*>(weights);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float d = fp16_to_fp32(blocks[b].d);
            const float* xb = input + b * 32;
            double block_sum = 0.0;
            for (std::size_t i = 0; i < 32; ++i) {
                block_sum += static_cast<double>(blocks[b].qs[i]) * static_cast<double>(xb[i]);
            }
            acc += block_sum * static_cast<double>(d);
        }
        break;
    }
    case aabhas::model::TensorDType::Q8_1: {
        const BlockQ8_1* blocks = static_cast<const BlockQ8_1*>(weights);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float d = blocks[b].d;
            const float* xb = input + b * 32;
            double block_sum = 0.0;
            for (std::size_t i = 0; i < 32; ++i) {
                block_sum += static_cast<double>(blocks[b].qs[i]) * static_cast<double>(xb[i]);
            }
            acc += block_sum * static_cast<double>(d);
        }
        break;
    }
    default:
        return false;
    }

    result = static_cast<float>(acc);
    return true;
}

bool gemv_quantized(aabhas::model::TensorDType dtype,
                    const void* weights,
                    std::size_t rows,
                    std::size_t columns,
                    const float* input,
                    float* output) noexcept {
    if (weights == nullptr || input == nullptr || output == nullptr ||
        rows == 0 || columns == 0) {
        return false;
    }
    if (!is_supported_weight_dtype(dtype)) return false;
    const std::size_t elems_per_blk = block_elements(dtype);
    if (columns % elems_per_blk != 0) return false;
    const std::size_t bytes_per_blk = block_bytes(dtype);
    const std::size_t row_bytes = (columns / elems_per_blk) * bytes_per_blk;
    const std::uint8_t* row_ptr = static_cast<const std::uint8_t*>(weights);

    for (std::size_t r = 0; r < rows; ++r) {
        float row_result = 0.0F;
        if (!vec_dot(dtype, row_ptr + r * row_bytes, input, columns, row_result)) {
            return false;
        }
        output[r] = row_result;
    }
    return true;
}

bool quantize_row(aabhas::model::TensorDType dtype,
                  const float* source,
                  std::size_t count,
                  void* destination,
                  std::size_t destination_bytes) noexcept {
    if (source == nullptr || destination == nullptr || count == 0) {
        return false;
    }
    const std::size_t elems_per_blk = block_elements(dtype);
    const std::size_t bytes_per_blk = block_bytes(dtype);
    if (elems_per_blk == 0 || bytes_per_blk == 0 || count % elems_per_blk != 0) {
        return false;
    }
    const std::size_t num_blocks = count / elems_per_blk;
    const std::size_t required_bytes = num_blocks * bytes_per_blk;
    if (destination_bytes < required_bytes) return false;

    switch (dtype) {
    case aabhas::model::TensorDType::F32:
        std::memcpy(destination, source, count * sizeof(float));
        return true;
    case aabhas::model::TensorDType::F16: {
        std::uint16_t* h = static_cast<std::uint16_t*>(destination);
        for (std::size_t i = 0; i < count; ++i) {
            h[i] = fp32_to_fp16(source[i]);
        }
        return true;
    }
    case aabhas::model::TensorDType::BF16: {
        std::uint16_t* b = static_cast<std::uint16_t*>(destination);
        for (std::size_t i = 0; i < count; ++i) {
            b[i] = fp32_to_bf16(source[i]);
        }
        return true;
    }
    case aabhas::model::TensorDType::Q4_0: {
        BlockQ4_0* blocks = static_cast<BlockQ4_0*>(destination);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float* x = source + b * 32;
            float max_val = 0.0F;
            for (std::size_t i = 0; i < 32; ++i) {
                max_val = std::max(max_val, std::fabs(x[i]));
            }
            const float d = max_val / -8.0F;
            const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
            blocks[b].d = fp32_to_fp16(d);
            for (std::size_t i = 0; i < 16; ++i) {
                int x0 = static_cast<int>(std::round(x[i] * id)) + 8;
                int x1 = static_cast<int>(std::round(x[i + 16] * id)) + 8;
                x0 = std::max(0, std::min(15, x0));
                x1 = std::max(0, std::min(15, x1));
                blocks[b].qs[i] = static_cast<std::uint8_t>(x0 | (x1 << 4));
            }
        }
        return true;
    }
    case aabhas::model::TensorDType::Q4_1: {
        BlockQ4_1* blocks = static_cast<BlockQ4_1*>(destination);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float* x = source + b * 32;
            float min_val = x[0];
            float max_val = x[0];
            for (std::size_t i = 1; i < 32; ++i) {
                min_val = std::min(min_val, x[i]);
                max_val = std::max(max_val, x[i]);
            }
            const float d = (max_val - min_val) / 15.0F;
            const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
            blocks[b].d = fp32_to_fp16(d);
            blocks[b].m = fp32_to_fp16(min_val);
            for (std::size_t i = 0; i < 16; ++i) {
                int x0 = static_cast<int>(std::round((x[i] - min_val) * id));
                int x1 = static_cast<int>(std::round((x[i + 16] - min_val) * id));
                x0 = std::max(0, std::min(15, x0));
                x1 = std::max(0, std::min(15, x1));
                blocks[b].qs[i] = static_cast<std::uint8_t>(x0 | (x1 << 4));
            }
        }
        return true;
    }
    case aabhas::model::TensorDType::Q8_0: {
        BlockQ8_0* blocks = static_cast<BlockQ8_0*>(destination);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float* x = source + b * 32;
            float max_val = 0.0F;
            for (std::size_t i = 0; i < 32; ++i) {
                max_val = std::max(max_val, std::fabs(x[i]));
            }
            const float d = max_val / 127.0F;
            const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
            blocks[b].d = fp32_to_fp16(d);
            for (std::size_t i = 0; i < 32; ++i) {
                int v = static_cast<int>(std::round(x[i] * id));
                v = std::max(-128, std::min(127, v));
                blocks[b].qs[i] = static_cast<std::int8_t>(v);
            }
        }
        return true;
    }
    case aabhas::model::TensorDType::Q8_1: {
        BlockQ8_1* blocks = static_cast<BlockQ8_1*>(destination);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const float* x = source + b * 32;
            float max_val = 0.0F;
            for (std::size_t i = 0; i < 32; ++i) {
                max_val = std::max(max_val, std::fabs(x[i]));
            }
            const float d = max_val / 127.0F;
            const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
            blocks[b].d = d;
            float sum = 0.0F;
            for (std::size_t i = 0; i < 32; ++i) {
                int v = static_cast<int>(std::round(x[i] * id));
                v = std::max(-128, std::min(127, v));
                blocks[b].qs[i] = static_cast<std::int8_t>(v);
                sum += x[i];
            }
            blocks[b].s = sum;
        }
        return true;
    }
    default:
        return false;
    }
}

bool QuantizedStreamDecoder::init(aabhas::model::TensorDType dtype, float* destination,
                                  std::size_t target_elements) noexcept {
    if (!is_supported_weight_dtype(dtype) || target_elements == 0) {
        return false;
    }
    block_elems_ = block_elements(dtype);
    block_byte_size_ = block_bytes(dtype);
    if (block_elems_ == 0 || block_byte_size_ == 0 ||
        target_elements % block_elems_ != 0) {
        return false;
    }
    dtype_ = dtype;
    destination_ = destination;
    target_elements_ = target_elements;
    elements_decoded_ = 0;
    buffer_len_ = 0;
    return true;
}

bool QuantizedStreamDecoder::process_single_block(
    const void* block_raw,
    const std::function<void(const float* block_values,
                             std::size_t block_elem_count,
                             std::size_t element_offset)>& on_block) noexcept {
    float temp[32];
    float* out_target = temp;
    if (destination_ != nullptr) {
        out_target = destination_ + elements_decoded_;
    }
    switch (dtype_) {
    case aabhas::model::TensorDType::F32:
        std::memcpy(out_target, block_raw, sizeof(float));
        break;
    case aabhas::model::TensorDType::F16: {
        std::uint16_t h = 0;
        std::memcpy(&h, block_raw, sizeof(h));
        out_target[0] = fp16_to_fp32(h);
        break;
    }
    case aabhas::model::TensorDType::BF16: {
        std::uint16_t b = 0;
        std::memcpy(&b, block_raw, sizeof(b));
        out_target[0] = bf16_to_fp32(b);
        break;
    }
    case aabhas::model::TensorDType::Q4_0:
        dequantize_q4_0_block(static_cast<const BlockQ4_0*>(block_raw), out_target);
        break;
    case aabhas::model::TensorDType::Q4_1:
        dequantize_q4_1_block(static_cast<const BlockQ4_1*>(block_raw), out_target);
        break;
    case aabhas::model::TensorDType::Q5_0:
        dequantize_q5_0_block(static_cast<const BlockQ5_0*>(block_raw), out_target);
        break;
    case aabhas::model::TensorDType::Q5_1:
        dequantize_q5_1_block(static_cast<const BlockQ5_1*>(block_raw), out_target);
        break;
    case aabhas::model::TensorDType::Q8_0:
        dequantize_q8_0_block(static_cast<const BlockQ8_0*>(block_raw), out_target);
        break;
    case aabhas::model::TensorDType::Q8_1:
        dequantize_q8_1_block(static_cast<const BlockQ8_1*>(block_raw), out_target);
        break;
    default:
        return false;
    }

    if (on_block) {
        on_block(out_target, block_elems_, elements_decoded_);
    }
    elements_decoded_ += block_elems_;
    return true;
}

bool QuantizedStreamDecoder::feed(const void* chunk_data, std::size_t chunk_bytes) noexcept {
    return feed_with_callback(chunk_data, chunk_bytes, nullptr);
}

bool QuantizedStreamDecoder::feed_with_callback(
    const void* chunk_data, std::size_t chunk_bytes,
    const std::function<void(const float* block_values,
                             std::size_t block_elem_count,
                             std::size_t element_offset)>& on_block) noexcept {
    if (chunk_data == nullptr || chunk_bytes == 0) return true;
    const std::uint8_t* ptr = static_cast<const std::uint8_t*>(chunk_data);
    std::size_t rem = chunk_bytes;

    if (buffer_len_ > 0) {
        const std::size_t needed = block_byte_size_ - buffer_len_;
        const std::size_t take = std::min(needed, rem);
        std::memcpy(buffer_ + buffer_len_, ptr, take);
        buffer_len_ += take;
        ptr += take;
        rem -= take;
        if (buffer_len_ == block_byte_size_) {
            if (elements_decoded_ + block_elems_ > target_elements_) return false;
            if (!process_single_block(buffer_, on_block)) return false;
            buffer_len_ = 0;
        }
    }

    while (rem >= block_byte_size_ && elements_decoded_ < target_elements_) {
        if (!process_single_block(ptr, on_block)) return false;
        ptr += block_byte_size_;
        rem -= block_byte_size_;
    }

    if (rem > 0) {
        if (elements_decoded_ >= target_elements_) return false;
        if (rem > sizeof(buffer_)) return false;
        std::memcpy(buffer_, ptr, rem);
        buffer_len_ = rem;
    }

    return true;
}

bool QuantizedStreamDecoder::finish() const noexcept {
    return buffer_len_ == 0 && elements_decoded_ == target_elements_;
}

} // namespace aabhas::inference::kernels