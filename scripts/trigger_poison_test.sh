#!/usr/bin/env bash
set -Eeuo pipefail

# This is a bounded integration test. It never passes a production tensor
# address to the poison routine and never changes the host hardware UUID.

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON_BIN="${AABHAS_DAEMON_BIN:-${REPO_ROOT}/bin/aabhas-daemon}"
LISTEN_ADDR="${AABHAS_LISTEN_ADDR:-127.0.0.1:18081}"
BASE_URL="http://${LISTEN_ADDR}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aabhas-poison-test.XXXXXX")"
HARNESS="${WORK_DIR}/poison-test.rs"
HARNESS_BIN="${WORK_DIR}/poison-test"
MODEL_PATH="${WORK_DIR}/mock-model.bin"
DAEMON_LOG="${WORK_DIR}/daemon.log"
DAEMON_PID=""

cleanup() {
	set +e
	if [[ -n "${DAEMON_PID}" ]] && kill -0 "${DAEMON_PID}" 2>/dev/null; then
		curl --silent --show-error --max-time 3 -X POST "${BASE_URL}/api/v1/engine/stop" >/dev/null 2>&1 || true
		kill "${DAEMON_PID}" 2>/dev/null || true
		wait "${DAEMON_PID}" 2>/dev/null || true
	fi
	rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT INT TERM

die() { printf 'trigger_poison_test: %s\n' "$*" >&2; exit 1; }
command -v rustc >/dev/null 2>&1 || die "rustc is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
[[ -x "${DAEMON_BIN}" ]] || die "daemon not found: ${DAEMON_BIN}; build it first or set AABHAS_DAEMON_BIN"

export AABHAS_TEST_HARDWARE_UUID="00000000-0000-0000-0000-000000000001"
export AABHAS_LICENSED_HARDWARE_UUID="00000000-0000-0000-0000-000000000002"

cat >"${HARNESS}" <<'RS'
mod security {
    pub mod crypto_verify {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum SignatureError {
            InvalidPublicKeyLength,
            InvalidSignatureLength,
            InvalidPublicKeyMaterial,
            InvalidSignatureMaterial,
            EmptyPayload,
            VerificationFailed,
        }

        pub fn verify_license_signature(
            _public_key: &[u8],
            _payload: &[u8],
            _signature: &[u8],
        ) -> Result<(), SignatureError> {
            Err(SignatureError::VerificationFailed)
        }
    }
    pub mod hw_binder {
        include!(concat!(env!("AABHAS_REPO_ROOT"), "/core-engine/rust_trap/src/security/hw_binder.rs"));
    }
    pub mod poison_pill {
        include!(concat!(env!("AABHAS_REPO_ROOT"), "/core-engine/rust_trap/src/security/poison_pill.rs"));
    }
}

fn main() {
    let actual = std::env::var("AABHAS_TEST_HARDWARE_UUID").expect("mock actual UUID");
    let licensed = std::env::var("AABHAS_LICENSED_HARDWARE_UUID").expect("mock licensed UUID");
    assert!(matches!(
        security::hw_binder::validate_hardware_binding(&actual, &licensed),
        Err(security::hw_binder::BindingError::HardwareMismatch)
    ));

    let mut tensor = [0xA5_u8; 256];
    let original = tensor;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        security::poison_pill::execute_ip_protection(tensor.as_mut_ptr(), tensor.len());
    }));
    assert!(result.is_ok(), "poison routine panicked");
    assert_ne!(tensor, original, "unauthorized tensor was not scrambled");
    assert_eq!(tensor.len(), original.len());
    println!("unauthorized UUID rejected; bounded tensor buffer scrambled safely");
}
RS

export AABHAS_REPO_ROOT="${REPO_ROOT}"
rustc --edition=2021 -C opt-level=3 -C debuginfo=0 "${HARNESS}" -o "${HARNESS_BIN}"

dd if=/dev/zero of="${MODEL_PATH}" bs=4096 count=1 status=none
AABHAS_LISTEN_ADDR="${LISTEN_ADDR}" "${DAEMON_BIN}" >"${DAEMON_LOG}" 2>&1 &
DAEMON_PID=$!

ready=0
for _ in $(seq 1 60); do
	if curl --silent --output /dev/null --connect-timeout 1 --max-time 2 "${BASE_URL}/api/v1/engine/stop"; then
		ready=1
		break
	fi
	sleep 0.25
done
if [[ "${ready}" -ne 1 ]]; then
	cat "${DAEMON_LOG}" >&2
	die "daemon did not become reachable"
fi

start_code="$(curl --silent --show-error --output "${WORK_DIR}/start.response" --write-out '%{http_code}' \
	-H 'Content-Type: application/json' -X POST \
	-d "{\"model_path\":\"${MODEL_PATH}\"}" "${BASE_URL}/api/v1/engine/start")"
if [[ "${start_code}" != "200" ]]; then
	cat "${WORK_DIR}/start.response" >&2
	cat "${DAEMON_LOG}" >&2
	die "engine start returned HTTP ${start_code}"
fi

"${HARNESS_BIN}"
if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
	cat "${DAEMON_LOG}" >&2
	die "daemon exited during poison test"
fi

stop_code="$(curl --silent --show-error --output "${WORK_DIR}/stop.response" --write-out '%{http_code}' \
	-X POST "${BASE_URL}/api/v1/engine/stop")"
if [[ "${stop_code}" != "200" ]]; then
	cat "${WORK_DIR}/stop.response" >&2
	die "engine stop returned HTTP ${stop_code}"
fi

echo "Aabhas unauthorized-hardware poison test passed"
