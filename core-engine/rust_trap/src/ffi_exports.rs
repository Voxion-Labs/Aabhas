//! Minimal C ABI for the Aabhas control plane.

use crate::health::{ram_capper, ssd_reserve};
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};

/// Start completed successfully.
pub const STATUS_OK: i32 = 0;
/// The model path pointer was null or the path was empty.
pub const STATUS_INVALID_ARGUMENT: i32 = -1;
/// An engine instance is already active.
pub const STATUS_ALREADY_STARTED: i32 = -2;
/// Host RAM is at or above the strict 70% cap.
pub const STATUS_RAM_THROTTLED: i32 = -3;
/// The filesystem has less than the required 20 GiB reserve.
pub const STATUS_SSD_RESERVE: i32 = -4;
/// A host health probe or filesystem operation failed.
pub const STATUS_HEALTH_CHECK: i32 = -5;
/// The FFI call recovered from a Rust panic.
pub const STATUS_INTERNAL_PANIC: i32 = -6;
/// Stop was requested while no engine instance was active.
pub const STATUS_NOT_STARTED: i32 = -7;

struct EngineState;

fn state() -> &'static Mutex<Option<EngineState>> {
    static STATE: OnceLock<Mutex<Option<EngineState>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

fn start_inner(model_path: *const c_char) -> i32 {
    if model_path.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    // The caller owns this memory and must provide a valid NUL-terminated C string.
    let path = unsafe { CStr::from_ptr(model_path) };
    if path.to_bytes().is_empty() {
        return STATUS_INVALID_ARGUMENT;
    }

    let mut guard = match state().lock() {
        Ok(value) => value,
        Err(_) => return STATUS_HEALTH_CHECK,
    };
    if guard.is_some() {
        return STATUS_ALREADY_STARTED;
    }

    match ram_capper::should_throttle() {
        Ok(true) => return STATUS_RAM_THROTTLED,
        Ok(false) => {}
        Err(_) => return STATUS_HEALTH_CHECK,
    }
    match ssd_reserve::has_reserve(path) {
        Ok(true) => {}
        Ok(false) => return STATUS_SSD_RESERVE,
        Err(_) => return STATUS_HEALTH_CHECK,
    }

    *guard = Some(EngineState);
    STATUS_OK
}

fn stop_inner() -> i32 {
    let mut guard = match state().lock() {
        Ok(value) => value,
        Err(_) => return STATUS_HEALTH_CHECK,
    };
    if guard.take().is_some() {
        STATUS_OK
    } else {
        STATUS_NOT_STARTED
    }
}

/// Start the safety-gated Aabhas engine for a NUL-terminated model path.
///
/// The caller must provide a valid, readable C string for `model_path`.
/// No Rust panic is allowed to cross this boundary.
#[no_mangle]
pub extern "C" fn start_aabhas_engine(model_path: *const c_char) -> i32 {
    catch_unwind(AssertUnwindSafe(|| start_inner(model_path)))
        .unwrap_or(STATUS_INTERNAL_PANIC)
}

/// Stop the active Aabhas engine. No Rust panic is allowed to cross this boundary.
#[no_mangle]
pub extern "C" fn stop_aabhas_engine() -> i32 {
    catch_unwind(AssertUnwindSafe(stop_inner)).unwrap_or(STATUS_INTERNAL_PANIC)
}
