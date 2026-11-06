//! Hardware-bound license validation for Enterprise IP Protection.

use super::crypto_verify::{self, SignatureError};

const NORMALIZED_UUID_CAPACITY: usize = 64;

/// License material consumed by the DRM enforcement gate.
#[derive(Clone, Copy, Debug)]
pub struct LicenseConfiguration<'a> {
    /// UUID recorded when the license was issued.
    pub licensed_hardware_uuid: &'a str,
    /// Signed license payload.
    pub payload: &'a [u8],
    /// Public verification key associated with the license issuer.
    pub public_key: &'a [u8],
    /// Signature over `payload`.
    pub signature: &'a [u8],
}

/// Hardware-binding failure. Enforcement must fail closed for every variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingError {
    /// One or both UUID values are absent or malformed.
    InvalidHardwareUuid,
    /// The current host does not match the licensed host.
    HardwareMismatch,
    /// The license signature is malformed or not cryptographically verified.
    SignatureRejected,
}

/// Validate a hardware UUID obtained from the native host-identification layer.
///
/// UUID comparison is normalized for case, braces, separators, and surrounding
/// whitespace, then compared in constant time over a fixed-size stack buffer.
pub fn validate_hardware_binding(
    actual_hardware_uuid: &str,
    licensed_hardware_uuid: &str,
) -> Result<(), BindingError> {
    let actual = normalize_uuid(actual_hardware_uuid)
        .ok_or(BindingError::InvalidHardwareUuid)?;
    let licensed = normalize_uuid(licensed_hardware_uuid)
        .ok_or(BindingError::InvalidHardwareUuid)?;

    let mut difference = (actual.len as u8) ^ (licensed.len as u8);
    for index in 0..NORMALIZED_UUID_CAPACITY {
        difference |= actual.bytes[index] ^ licensed.bytes[index];
    }
    if difference == 0 {
        Ok(())
    } else {
        Err(BindingError::HardwareMismatch)
    }
}

/// Validate both the hardware binding and the signature envelope.
///
/// The current verifier intentionally fails closed because it is a structural
/// verifier only. A real Ed25519/RSA backend must replace it before licenses
/// can be authorized in production.
pub fn validate_license(
    actual_hardware_uuid: &str,
    license: &LicenseConfiguration<'_>,
) -> Result<(), BindingError> {
    validate_hardware_binding(actual_hardware_uuid, license.licensed_hardware_uuid)?;
    crypto_verify::verify_license_signature(
        license.public_key,
        license.payload,
        license.signature,
    )
    .map_err(|error| match error {
        SignatureError::InvalidPublicKeyLength
        | SignatureError::InvalidSignatureLength
        | SignatureError::EmptyPayload
        | SignatureError::UnsupportedAlgorithm => BindingError::SignatureRejected,
    })
}

#[derive(Clone, Copy)]
struct NormalizedUuid {
    bytes: [u8; NORMALIZED_UUID_CAPACITY],
    len: usize,
}

fn normalize_uuid(value: &str) -> Option<NormalizedUuid> {
    let mut normalized = NormalizedUuid {
        bytes: [0; NORMALIZED_UUID_CAPACITY],
        len: 0,
    };
    for byte in value.bytes() {
        if byte == b'{' || byte == b'}' || byte == b'-' || byte.is_ascii_whitespace() {
            continue;
        }
        if !byte.is_ascii_hexdigit() || normalized.len == NORMALIZED_UUID_CAPACITY {
            return None;
        }
        normalized.bytes[normalized.len] = byte.to_ascii_lowercase();
        normalized.len += 1;
    }
    if normalized.len == 0 {
        None
    } else {
        Some(normalized)
    }
}
