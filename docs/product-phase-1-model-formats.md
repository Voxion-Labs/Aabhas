# Product Phase 1: Model-Format Execution Boundary

## Objective

Turn the existing GGUF and SafeTensors metadata index into a typed, safe
execution boundary. The parser must describe the exact tensor storage range
before later tokenizer, graph, or kernel phases consume any payload bytes.

## Delivered contract

1. Tensor layout calculation is overflow-checked for standard scalar dtypes and
   common GGML block formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, and
   `Q8_1`).
2. Common GGUF dtype identifiers and SafeTensors scalar dtype names are mapped
   into the shared model index.
3. Known SafeTensors ranges must exactly match `shape × dtype` storage size.
4. GGUF tensor ranges use the dtype-derived byte size instead of treating
   alignment padding before the next tensor as payload.
5. SafeTensors tensor ranges are sorted and rejected when they overlap.
6. Malformed layouts and impossible block shapes fail closed without reading
   tensor payloads into the metadata buffer.

## Boundary

Unknown dtypes remain indexable only when their file ranges are structurally
bounded; they are not silently decoded. Actual F16/BF16 conversion,
quantized dequantization kernels, tokenizer integration, and graph execution
are subsequent product phases.
