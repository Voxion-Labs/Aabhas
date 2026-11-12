# Phase 3: Model Metadata and Layer Indexing

## Objective

Discover tensor byte ranges without materializing model payloads in RAM, then
feed those ranges into the existing bounded C++ layer streamer.

## Delivered contract

1. GGUF v2/v3 metadata and tensor-info sections are parsed with bounded counts,
   validated lengths, alignment checks, and overflow protection.
2. SafeTensors headers are parsed with a bounded JSON reader and validated
   `data_offsets` ranges.
3. Unknown non-empty files retain a raw-binary fallback descriptor for the
   repository's existing development fixtures.
4. Native engine startup now uses discovered tensor ranges instead of treating
   every recognized model as one whole-file layer.
5. Model payload bytes are never copied into a metadata buffer; only the
   bounded header and descriptor metadata are retained.
6. Native CTest coverage exercises both GGUF and SafeTensors index creation.

## Safety boundaries

- Header allocations are capped at 64 MiB.
- Tensor and metadata entry counts are capped at 1,048,576.
- File offsets and ranges are checked for integer overflow and containment.
- Malformed recognized formats fail closed; only unrecognized files use the raw
  development fallback.

## Explicit non-goals

This phase does not decode tensor values, execute kernels, or implement the
inference scheduler. Those remain later phases.
