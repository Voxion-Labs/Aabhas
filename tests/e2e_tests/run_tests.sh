#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DAEMON_BIN="${AABHAS_DAEMON_BIN:-${REPO_ROOT}/bin/aabhas-daemon}"
LISTEN_ADDR="${AABHAS_LISTEN_ADDR:-127.0.0.1:18080}"
BASE_URL="http://${LISTEN_ADDR}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aabhas-e2e.XXXXXX")"
MODEL_PATH="${WORK_DIR}/mock-model.bin"
LOG_PATH="${WORK_DIR}/daemon.log"
PID=""

cleanup() {
	set +e
	if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
		curl --silent --show-error --max-time 3 -X POST "${BASE_URL}/api/v1/engine/stop" >/dev/null 2>&1 || true
		kill "${PID}" 2>/dev/null || true
		wait "${PID}" 2>/dev/null || true
	fi
	rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT INT TERM

command -v curl >/dev/null 2>&1 || { echo "run_tests: curl is required" >&2; exit 1; }
[[ -x "${DAEMON_BIN}" ]] || {
	echo "run_tests: daemon not found or not executable: ${DAEMON_BIN}" >&2
	echo "Build it first or set AABHAS_DAEMON_BIN." >&2
	exit 1
}

dd if=/dev/zero of="${MODEL_PATH}" bs=4096 count=1 status=none
"${DAEMON_BIN}" >"${LOG_PATH}" 2>&1 &
PID=$!

ready=0
for _ in $(seq 1 60); do
	if curl --silent --output /dev/null --connect-timeout 1 --max-time 2 "${BASE_URL}/api/v1/engine/stop"; then
		ready=1
		break
	fi
	sleep 0.25
done
if [[ "${ready}" -ne 1 ]]; then
	echo "run_tests: daemon did not become reachable" >&2
	cat "${LOG_PATH}" >&2
	exit 1
fi

start_response="${WORK_DIR}/start.response"
start_code="$(curl --silent --show-error --output "${start_response}" --write-out '%{http_code}' \
	-H 'Content-Type: application/json' -X POST \
	-d "{\"model_path\":\"${MODEL_PATH}\"}" \
	"${BASE_URL}/api/v1/engine/start")"
if [[ "${start_code}" != "200" ]]; then
	echo "run_tests: expected start HTTP 200, got ${start_code}" >&2
	cat "${start_response}" >&2
	cat "${LOG_PATH}" >&2
	exit 1
fi

stop_code="$(curl --silent --show-error --output "${WORK_DIR}/stop.response" --write-out '%{http_code}' \
	-X POST "${BASE_URL}/api/v1/engine/stop")"
if [[ "${stop_code}" != "200" ]]; then
	echo "run_tests: expected stop HTTP 200, got ${stop_code}" >&2
	cat "${WORK_DIR}/stop.response" >&2
	exit 1
fi

echo "Aabhas REST end-to-end test passed"
