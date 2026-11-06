//! Bounded DRM Enforcement response for unauthorized tensor buffers.
//!
//! This routine is intentionally limited to the caller-provided memory range.
//! It performs no allocation, system calls, or access outside `(buffer, len)`.

const WORD_BYTES: usize = core::mem::size_of::<u64>();
const SCRAMBLE_KEY: u64 = 0xAAB5_4A9D_731E_C2F7;

/// Scramble tensor bytes in place after a failed Enterprise IP Protection check.
///
/// Null and empty buffers are safe no-ops. Unaligned u64 chunks provide a
/// portable SIMD-like fast path without requiring target-specific intrinsics;
/// the tail is processed byte-wise. The caller must provide a valid writable
/// range of `len` bytes when `tensor_buffer` is non-null.
#[inline(never)]
pub fn execute_ip_protection(tensor_buffer: *mut u8, len: usize) {
    if tensor_buffer.is_null() || len == 0 {
        return;
    }

    let word_count = len / WORD_BYTES;
    let tail_start = word_count * WORD_BYTES;

    for word_index in 0..word_count {
        let offset = word_index * WORD_BYTES;
        let key = SCRAMBLE_KEY.rotate_left((word_index as u32) & 63);
        unsafe {
            let location = tensor_buffer.add(offset) as *mut u64;
            let value = core::ptr::read_unaligned(location);
            core::ptr::write_unaligned(location, value ^ key);
        }
    }

    let tail_key = SCRAMBLE_KEY.to_le_bytes();
    for offset in tail_start..len {
        let key_byte = tail_key[(offset - tail_start) & (tail_key.len() - 1)];
        unsafe {
            let location = tensor_buffer.add(offset);
            *location ^= key_byte;
        }
    }
}
