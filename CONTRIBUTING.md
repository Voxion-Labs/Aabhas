# AABHAS Operational & Contribution Protocols

Aabhas operates under strict execution and memory management protocols. This repository is maintained for high-performance, out-of-core tensor streaming. We require minimal abstractions, optimized memory allocations, and asynchronous, non-blocking I/O.

If you intend to submit a Pull Request, please adhere to the following directives.

## 1. Architectural Standards
All submissions must meet our baseline performance and safety metrics:
* **Zero-Copy Data Plane (C++17):** Code introducing unnecessary page faults, buffer copying, or failing to utilize direct NVMe I/O (`O_DIRECT`, `mmap`) will not be accepted.
* **Memory Management:** Modifications must respect the `madvise(MADV_DONTNEED)` cache-eviction protocols to ensure host OS stability.
* **Safety Grid (Rust):** Unsafe blocks lacking explicit FFI justification, unresolved memory leaks, or bypasses to the physical RAM cap are prohibited. Memory safety must be validated via Valgrind.
* **Orchestration (Go):** The control-plane must remain lightweight. Mutexes should strictly guard single-model execution states without blocking incoming REST requests.

## 2. Pull Request (PR) Governance
Before initiating a merge request, ensure your PR adheres to this structure:
1. **[METRIC] Benchmark Data:** Provide before/after execution telemetry (e.g., SSD read latency, peak physical RAM usage, CPU context-switch overhead).
2. **[LOGIC] State Transition:** Explicitly document the architectural state transitions your code alters within the FFI boundary or the hardware I/O scheduler.
3. **[ISOLATION] Threat Model:** Demonstrate that your modifications do not introduce memory leaks across the C-ABI boundary or compromise host OS stability.

*Note: PRs failing to provide empirical benchmark data or Valgrind logs will be closed.*

## 3. Vulnerability Disclosure
**DO NOT** open public issues for memory leaks, system crashes, or hardware execution bypasses. 
* All safety and memory corruption reports must be routed internally.
* Contact the repository maintainers directly for secure transmission protocols.

## 4. Code of Conduct
Submissions are evaluated objectively based on hardware optimization, POSIX compliance, and memory frugality. Keep discussions clinical and exclusively focused on systems architecture.

By contributing to this repository, you agree to the intellectual property terms defined in the [LICENSE](LICENSE).
