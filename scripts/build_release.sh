#!/usr/bin/env bash
set -Eeuo pipefail

# Build a reproducible, stripped Aabhas release artifact.
# Container isolation is used for the default toolchain path; this script does
# not disable, hide, or bypass host EDR telemetry.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
VERSION="${AABHAS_RELEASE_VERSION:-$(git -C "${REPO_ROOT}" describe --tags --always --dirty)}"
PLATFORM="${AABHAS_PLATFORM:-linux/amd64}"
IMAGE="${AABHAS_BUILDER_IMAGE:-aabhas-release-builder:${VERSION}}"
DIST_DIR="${REPO_ROOT}/dist"
STAGE_DIR="${DIST_DIR}/aabhas-${VERSION}"
ARCHIVE="${DIST_DIR}/aabhas-${VERSION}.tar.gz"

die() {
    printf 'build_release: %s\n' "$*" >&2
    exit 1
}

command -v git >/dev/null 2>&1 || die "git is required"
command -v tar >/dev/null 2>&1 || die "tar is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v docker >/dev/null 2>&1 || die "Docker is required for the isolated release build"

case "${VERSION}" in
    *[!A-Za-z0-9._+-]*) die "release version contains unsupported characters: ${VERSION}" ;;
esac

rm -rf -- "${STAGE_DIR}"
rm -f -- "${ARCHIVE}" "${ARCHIVE}.sha256"
mkdir -p -- "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${DIST_DIR}"

docker build \
    --platform "${PLATFORM}" \
    --target builder \
    --tag "${IMAGE}" \
    --file "${REPO_ROOT}/stealth-deploy/Dockerfile.prod" \
    "${REPO_ROOT}"

container_id="$(docker create "${IMAGE}")"
cleanup() {
    docker rm --force "${container_id}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker cp "${container_id}:/out/bin/aabhas-daemon" "${STAGE_DIR}/bin/aabhas-daemon"
docker cp "${container_id}:/out/lib/libaabhas.so" "${STAGE_DIR}/lib/libaabhas.so"
cp -- "${REPO_ROOT}/LICENSE" "${STAGE_DIR}/LICENSE"

strip --strip-unneeded "${STAGE_DIR}/bin/aabhas-daemon" 2>/dev/null || true
strip --strip-unneeded "${STAGE_DIR}/lib/libaabhas.so" 2>/dev/null || true
chmod 0555 "${STAGE_DIR}/bin/aabhas-daemon" "${STAGE_DIR}/lib/libaabhas.so"
chmod 0444 "${STAGE_DIR}/LICENSE"

tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime='UTC 1970-01-01' \
    -czf "${ARCHIVE}" \
    -C "${DIST_DIR}" "aabhas-${VERSION}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

printf 'Release archive: %s\n' "${ARCHIVE}"
printf 'SHA-256: '
cut -d ' ' -f 1 "${ARCHIVE}.sha256"
