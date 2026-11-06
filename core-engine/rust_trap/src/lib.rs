#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

//! Aabhas safety boundary and C-ABI entry points.

mod health;
mod security;
pub mod ffi_exports;
