# Product Phase 2: Tokenizer and Vocabulary Pipeline

## Objective

Provide a bounded tokenizer boundary that consumes model vocabulary metadata
without copying model tensor payloads into memory.

## Delivered contract

1. GGUF tokenizer vocabulary plus BOS, EOS, and unknown-token IDs are retained
   during bounded model indexing.
2. `GreedyTokenizer` builds a trie and uses deterministic longest-match
   tokenization, so overlapping pieces do not depend on hash-map iteration order.
3. Input UTF-8 is validated, input bytes and output token count are capped, and
   unknown code points use the configured unknown token or fail explicitly.
4. Token spans retain byte offsets and lengths for downstream request tracing
   and embedding lookup.
5. Decode is bounded and rejects IDs that are not present in the vocabulary.

## Boundary

This phase establishes the vocabulary and UTF-8 token contract. SentencePiece
normalization, byte-level BPE merge tables, special-token policy per request,
and embedding/transformer execution remain subsequent product phases.
