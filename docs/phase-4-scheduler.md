# Phase 4: Bounded Out-of-Core Scheduler

## Objective

Keep resident tensor pages bounded while advancing through model layers and
provide non-blocking kernel read-ahead hints for upcoming ranges.

## Delivered contract

1. `LayerScheduler` enforces a non-zero memory cap and chunk size that never
   exceeds that cap.
2. Only one mapped layer window is active at a time; callers must explicitly
   release it before advancing to the next layer.
3. `POSIX_FADV_WILLNEED` read-ahead hints are issued for a bounded lookahead
   window without allocating a second user-space tensor buffer.
4. Scheduler statistics expose resident bytes, layer/chunk progress, and
   read-ahead attempts for later observability work.
5. The native engine lifecycle now owns the scheduler rather than directly
   owning a raw streamer.
6. Native CTest coverage verifies memory-cap enforcement and serialized layer
   progression.

## Explicit non-goals

This phase does not introduce a background worker, tensor decoding, or compute
kernels. Read-ahead remains an OS advisory; actual page residency is still
controlled by the operating system.
