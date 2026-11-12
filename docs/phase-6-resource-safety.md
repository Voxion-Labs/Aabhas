# Phase 6: Resource and FFI Safety Hardening

## Objective

Make admission and native I/O limits enforceable under host, container, and
FFI boundary conditions without relaxing the default safety policy.

## Delivered contract

1. Rust RAM admission considers both host memory and cgroup v2 memory usage
   when those files are available; either strict 70% boundary can throttle a
   start.
2. The 20 GiB SSD reserve remains the minimum. `AABHAS_MIN_FREE_BYTES` may
   raise the reserve but cannot lower it.
3. Model paths crossing the Rust C ABI are bounded to 4096 bytes, and the Go
   bridge applies the same bound before allocation.
4. O_DIRECT reads now validate buffer address, file offset, and byte-count
   alignment against the probed transfer alignment.
5. Alignment state resets correctly across reader moves and close/reopen
   cycles.

## Explicit non-goals

These checks remain admission controls and do not replace kernel cgroups,
filesystem quotas, SELinux/AppArmor policy, or node-level monitoring.
