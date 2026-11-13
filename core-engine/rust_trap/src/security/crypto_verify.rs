//! Ed25519 license-signature verification.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

/// Ed25519 public-key size in bytes.
pub const ED25519_PUBLIC_KEY_BYTES: usize = 32;
/// Ed25519 signature size in bytes.
pub const ED25519_SIGNATURE_BYTES: usize = 64;

/// Signature-validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureError {
    /// The public key is not the expected size.
    InvalidPublicKeyLength,
    /// The signature is not the expected size.
    InvalidSignatureLength,
    /// The public key bytes do not encode a valid Ed25519 key.
    InvalidPublicKeyMaterial,
    /// The signature bytes do not encode a valid Ed25519 signature.
    InvalidSignatureMaterial,
    /// The signed payload is empty.
    EmptyPayload,
    /// The signature did not authenticate the payload.
    VerificationFailed,
}

/// Check the Ed25519 envelope shape before invoking the verifier.
pub const fn has_valid_ed25519_shape(public_key: &[u8], signature: &[u8], payload: &[u8]) -> bool {
    public_key.len() == ED25519_PUBLIC_KEY_BYTES
        && signature.len() == ED25519_SIGNATURE_BYTES
        && !payload.is_empty()
}

/// Verify a non-empty license payload with its Ed25519 public key.
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

    let key_bytes: &[u8; ED25519_PUBLIC_KEY_BYTES] = public_key
        .try_into()
        .map_err(|_| SignatureError::InvalidPublicKeyLength)?;
    let key = VerifyingKey::from_bytes(key_bytes)
        .map_err(|_| SignatureError::InvalidPublicKeyMaterial)?;
    let signature = Signature::from_slice(signature)
        .map_err(|_| SignatureError::InvalidSignatureMaterial)?;
    key.verify(payload, &signature)
        .map_err(|_| SignatureError::VerificationFailed)
}

#[cfg(test)]
mod tests {
    use super::{verify_license_signature, SignatureError};
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn verifies_and_rejects_ed25519_signatures() {
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let payload = b"aabhas-license-v1";
        let signature = signing_key.sign(payload);
        let public_key = signing_key.verifying_key().to_bytes();

        assert!(verify_license_signature(&public_key, payload, &signature.to_bytes()).is_ok());
        assert_eq!(
            verify_license_signature(&public_key, b"tampered", &signature.to_bytes()),
            Err(SignatureError::VerificationFailed)
        );
    }
}
