//! Physical-memory admission control for the data plane.

use sysinfo::System;

const CGROUP_MEMORY_MAX: &str = "/sys/fs/cgroup/memory.max";
const CGROUP_MEMORY_CURRENT: &str = "/sys/fs/cgroup/memory.current";

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
    /// Optional cgroup memory limit in bytes.
    pub cgroup_limit_bytes: Option<u64>,
    /// Optional current cgroup memory usage in bytes.
    pub cgroup_used_bytes: Option<u64>,
    /// Optional cgroup cap derived from the same strict percentage.
    pub cgroup_cap_bytes: Option<u64>,
}

impl RamSnapshot {
    /// Returns `true` when the hard cap has been reached or exceeded.
    pub const fn throttled(self) -> bool {
        self.used_bytes >= self.cap_bytes || match (self.cgroup_used_bytes, self.cgroup_cap_bytes) {
            (Some(used), Some(cap)) => used >= cap,
            _ => false,
        }
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
    let cgroup_limit_bytes = read_cgroup_value(CGROUP_MEMORY_MAX);
    let cgroup_used_bytes = read_cgroup_value(CGROUP_MEMORY_CURRENT);
    let cgroup_cap_bytes = cgroup_limit_bytes.map(cap_for);
    Ok(RamSnapshot {
        total_bytes,
        used_bytes: system.used_memory(),
        cap_bytes,
        cgroup_limit_bytes,
        cgroup_used_bytes,
        cgroup_cap_bytes,
    })
}

/// Return a strict throttle decision for the current host state.
pub fn should_throttle() -> Result<bool, RamError> {
    Ok(snapshot()?.throttled())
}

fn cap_for(total_bytes: u64) -> u64 {
    ((total_bytes as u128 * MAX_UTILIZATION_PERCENT as u128) / 100)
        .min(u64::MAX as u128) as u64
}

fn read_cgroup_value(path: &str) -> Option<u64> {
    let value = std::fs::read_to_string(path).ok()?;
    let trimmed = value.trim();
    if trimmed == "max" {
        return None;
    }
    trimmed.parse::<u64>().ok().filter(|value| *value != 0)
}

#[cfg(test)]
mod tests {
    use super::{cap_for, MAX_UTILIZATION_PERCENT};

    #[test]
    fn cap_is_strictly_seventy_percent() {
        assert_eq!(cap_for(100), MAX_UTILIZATION_PERCENT);
        assert_eq!(cap_for(0), 0);
    }
}
