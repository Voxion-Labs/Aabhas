//! Physical-memory admission control for the data plane.

use sysinfo::System;

/// Maximum permitted physical-memory utilization.
pub const MAX_UTILIZATION_PERCENT: u64 = 70;

/// Result of one physical-memory observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamSnapshot {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Memory reported as used by the host operating system, in bytes.
    pub used_bytes: u64,
    /// Absolute hard cap derived from `total_bytes`.
    pub cap_bytes: u64,
}

impl RamSnapshot {
    /// Returns `true` when the hard cap has been reached or exceeded.
    pub const fn throttled(self) -> bool {
        self.used_bytes >= self.cap_bytes
    }
}

/// Errors produced while obtaining a usable memory reading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RamError {
    /// The host did not provide a non-zero physical-memory reading.
    Unavailable,
}

/// Read host memory and calculate the strict 70% utilization boundary.
pub fn snapshot() -> Result<RamSnapshot, RamError> {
    let mut system = System::new();
    system.refresh_memory();

    let total_bytes = system.total_memory();
    if total_bytes == 0 {
        return Err(RamError::Unavailable);
    }

    let cap_bytes = ((total_bytes as u128 * MAX_UTILIZATION_PERCENT as u128) / 100)
        .min(u64::MAX as u128) as u64;
    Ok(RamSnapshot {
        total_bytes,
        used_bytes: system.used_memory(),
        cap_bytes,
    })
}

/// Return a strict throttle decision for the current host state.
pub fn should_throttle() -> Result<bool, RamError> {
    Ok(snapshot()?.throttled())
}
