//! Pre-flight storage reserve check for model-backed operations.

use std::ffi::CStr;
use std::io;

/// Minimum free space required before the engine may start: 20 GiB.
pub const MIN_FREE_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Storage information for the filesystem containing a model path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageSnapshot {
    /// Free blocks available to the calling process, in bytes.
    pub available_bytes: u64,
    /// Required reserve, in bytes.
    pub reserve_bytes: u64,
}

impl StorageSnapshot {
    /// Returns `true` when the configured reserve is available.
    pub const fn has_reserve(self) -> bool {
        self.available_bytes >= self.reserve_bytes
    }
}

/// Read filesystem capacity for a NUL-terminated POSIX path.
pub fn snapshot(path: &CStr) -> io::Result<StorageSnapshot> {
    let mut stats = unsafe { std::mem::zeroed::<libc::statvfs>() };
    let result = unsafe { libc::statvfs(path.as_ptr(), &mut stats) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }

    let block_size = if stats.f_frsize != 0 {
        stats.f_frsize as u128
    } else {
        stats.f_bsize as u128
    };
    let available_blocks = stats.f_bavail as u128;
    let available_bytes = available_blocks
        .saturating_mul(block_size)
        .min(u64::MAX as u128) as u64;

    Ok(StorageSnapshot {
        available_bytes,
        reserve_bytes: MIN_FREE_BYTES,
    })
}

/// Return whether the filesystem containing `path` satisfies the reserve.
pub fn has_reserve(path: &CStr) -> io::Result<bool> {
    Ok(snapshot(path)?.has_reserve())
}
