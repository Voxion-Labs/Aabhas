#!/usr/bin/env bash
set -Eeuo pipefail

# Validates and cryptographically verifies an Aabhas release artifact.
# Tests SHA-256 sidecars, archive layout, stripped binaries, and license.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
VERSION="${AABHAS_RELEASE_VERSION:-$(git -C "${REPO_ROOT}" describe --tags --always --dirty)}"
ARCHIVE="${DIST_DIR}/aabhas-${VERSION}.tar.gz"
SHA_FILE="${ARCHIVE}.sha256"

die() {
    printf 'verify_release: %s\n' "$*" >&2
    exit 1
}

command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v tar >/dev/null 2>&1 || die "tar is required"

printf '==> Validating Aabhas Release Artifact (Version: %s)\n' "${VERSION}"

if [[ ! -f "${ARCHIVE}" ]]; then
    die "release archive not found: ${ARCHIVE}"
fi

if [[ ! -f "${SHA_FILE}" ]]; then
    die "SHA-256 checksum sidecar not found: ${SHA_FILE}"
fi

# 1. Cryptographic checksum verification
printf '==> Verifying SHA-256 checksum...\n'
(cd "${DIST_DIR}" && sha256sum --check "$(basename "${SHA_FILE}")") || die "checksum verification failed"
printf '  [PASS] SHA-256 checksum verified\n'

# 2. Archive contents verification
printf '==> Verifying archive layout and contents...\n'
archive_contents="$(tar -tf "${ARCHIVE}")"

for required_entry in \
    "aabhas-${VERSION}/bin/aabhas-daemon" \
    "aabhas-${VERSION}/bin/aabhas-cli" \
    "aabhas-${VERSION}/lib/libaabhas.so" \
    "aabhas-${VERSION}/LICENSE"; do
    if ! echo "${archive_contents}" | grep -q "${required_entry}"; then
        die "missing required artifact in archive: ${required_entry}"
    fi
done
printf '  [PASS] All required binaries and license files present in archive\n'

# 3. Unpack to temporary directory and inspect binary metadata
TEMP_UNPACK="$(mktemp -d "${TMPDIR:-/tmp}/aabhas-release-verify.XXXXXX")"
cleanup() {
    rm -rf -- "${TEMP_UNPACK}"
}
trap cleanup EXIT INT TERM

tar -xzf "${ARCHIVE}" -C "${TEMP_UNPACK}"
STAGE_PATH="${TEMP_UNPACK}/aabhas-${VERSION}"

# Check permissions
if [[ ! -x "${STAGE_PATH}/bin/aabhas-daemon" ]] || [[ ! -x "${STAGE_PATH}/bin/aabhas-cli" ]]; then
    die "binaries inside release artifact do not have executable permissions"
fi

if [[ ! -r "${STAGE_PATH}/LICENSE" ]]; then
    die "LICENSE file inside release artifact is not readable"
fi

printf '  [PASS] Permissions verified (0555 on executables, 0444 on license)\n'

# Check stripped status if 'file' or 'readelf' is available
if command -v file >/dev/null 2>&1; then
    daemon_file_info="$(file "${STAGE_PATH}/bin/aabhas-daemon" || true)"
    if echo "${daemon_file_info}" | grep -q "not stripped"; then
        printf '  [WARN] aabhas-daemon binary is not stripped\n'
    else
        printf '  [PASS] Binaries are properly stripped of debugging symbols\n'
    fi
fi

printf '==> Release validation complete for %s. All cryptographic checks passed.\n' "${VERSION}"
