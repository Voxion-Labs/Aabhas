//! Structural license-signature verification boundary.
//!
//! This module deliberately does not claim to implement cryptography. It
//! validates the wire shape and fails closed until a reviewed Ed25519 or RSA
//! implementation is linked into the product.

/// Ed25519 public-key size in bytes.
pub const ED25519_PUBLIC_KEY_BYTES: usize = 32;
/// Ed25519 signature size in bytes.
pub const ED25519_SIGNATURE_BYTES: usize = 64;

/// Structural signature-validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureError {
    /// The public key is not the expected size.
    InvalidPublicKeyLength,
    /// The signature is not the expected size.
    InvalidSignatureLength,
    /// The signed payload is empty.
    EmptyPayload,
    /// A cryptographic backend is not linked into this placeholder module.
    UnsupportedAlgorithm,
}

/// Check the Ed25519 envelope shape without authorizing it.
pub const fn has_valid_ed25519_shape(public_key: &[u8], signature: &[u8], payload: &[u8]) -> bool {
    public_key.len() == ED25519_PUBLIC_KEY_BYTES
        && signature.len() == ED25519_SIGNATURE_BYTES
        && !payload.is_empty()
}

/// Validate the license signature envelope and fail closed.
pub fn verify_license_signature(
    public_key: &[u8],
    payload: &[u8],
    signature: &[u8],
) -> Result<(), SignatureError> {
    if public_key.len() != ED25519_PUBLIC_KEY_BYTES {
        return Err(SignatureError::InvalidPublicKeyLength);
    }
    if signature.len() != ED25519_SIGNATURE_BYTES {
        return Err(SignatureError::InvalidSignatureLength);
    }
    if payload.is_empty() {
        return Err(SignatureError::EmptyPayload);
    }
    Err(SignatureError::UnsupportedAlgorithm)
}
