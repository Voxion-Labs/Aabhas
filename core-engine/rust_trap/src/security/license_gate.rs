//! Opt-in, fail-closed hardware-bound license enforcement.

use super::hw_binder::{validate_license, BindingError, LicenseConfiguration};

const REQUIRED: &str = "AABHAS_LICENSE_REQUIRED";
const LICENSED_UUID: &str = "AABHAS_LICENSED_HARDWARE_UUID";
const PAYLOAD: &str = "AABHAS_LICENSE_PAYLOAD_HEX";
const PUBLIC_KEY: &str = "AABHAS_LICENSE_PUBLIC_KEY_HEX";
const SIGNATURE: &str = "AABHAS_LICENSE_SIGNATURE_HEX";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LicenseGateError {
    MissingConfiguration,
    InvalidEncoding,
    Hardware(BindingError),
}

/// Enforce a configured license when enterprise mode is enabled.
///
/// Development mode remains compatible when no license environment is set.
/// Setting `AABHAS_LICENSE_REQUIRED=1` makes missing or invalid material fail
/// closed before the native data plane starts.
pub fn enforce_if_configured() -> Result<(), LicenseGateError> {
    let required = std::env::var(REQUIRED)
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !required {
        return Ok(());
    }

    let licensed_uuid = std::env::var(LICENSED_UUID)
        .map_err(|_| LicenseGateError::MissingConfiguration)?;
    let payload = decode_hex_env(PAYLOAD)?;
    let public_key = decode_hex_env(PUBLIC_KEY)?;
    let signature = decode_hex_env(SIGNATURE)?;
    let actual_uuid = actual_hardware_uuid().ok_or(LicenseGateError::MissingConfiguration)?;
    let license = LicenseConfiguration {
        licensed_hardware_uuid: &licensed_uuid,
        payload: &payload,
        public_key: &public_key,
        signature: &signature,
    };
    validate_license(&actual_uuid, &license).map_err(LicenseGateError::Hardware)
}

fn decode_hex_env(name: &str) -> Result<Vec<u8>, LicenseGateError> {
    let value = std::env::var(name).map_err(|_| LicenseGateError::MissingConfiguration)?;
    let max_hex_bytes = match name {
        PUBLIC_KEY => 64,
        SIGNATURE => 128,
        PAYLOAD => 2 * 1024 * 1024,
        _ => return Err(LicenseGateError::InvalidEncoding),
    };
    if value.trim().len() > max_hex_bytes {
        return Err(LicenseGateError::InvalidEncoding);
    }
    decode_hex(value.trim()).ok_or(LicenseGateError::InvalidEncoding)
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.is_empty() || value.len() % 2 != 0 {
        return None;
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for index in (0..bytes.len()).step_by(2) {
        let high = hex_nibble(bytes[index])?;
        let low = hex_nibble(bytes[index + 1])?;
        output.push((high << 4) | low);
    }
    Some(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn actual_hardware_uuid() -> Option<String> {
    if let Ok(value) = std::env::var("AABHAS_HARDWARE_UUID") {
        return Some(value);
    }
    [
        "/sys/class/dmi/id/product_uuid",
        "/sys/devices/virtual/dmi/id/product_uuid",
        "/etc/machine-id",
    ]
    .iter()
    .find_map(|path| std::fs::read_to_string(path).ok())
}
