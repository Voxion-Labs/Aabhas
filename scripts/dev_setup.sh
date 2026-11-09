#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
	readonly C_RESET=$'\033[0m'
	readonly C_BLUE=$'\033[1;34m'
	readonly C_GREEN=$'\033[1;32m'
	readonly C_RED=$'\033[1;31m'
	readonly C_YELLOW=$'\033[1;33m'
else
	readonly C_RESET=""
	readonly C_BLUE=""
	readonly C_GREEN=""
	readonly C_RED=""
	readonly C_YELLOW=""
fi

log() { printf '%s[dev-setup]%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"; }
ok() { printf '%s[dev-setup]%s %s\n' "${C_GREEN}" "${C_RESET}" "$*"; }
warn() { printf '%s[dev-setup]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
die() { printf '%s[dev-setup]%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2; exit 1; }

on_error() {
	local status=$?
	die "failed at line ${BASH_LINENO[0]} with exit status ${status}"
}
trap on_error ERR

[[ "${EUID}" -eq 0 || -n "$(command -v sudo || true)" ]] || die "run as root or install sudo"
command -v apt-get >/dev/null 2>&1 || die "this script requires Ubuntu/Debian apt-get"
command -v dpkg >/dev/null 2>&1 || die "this script requires dpkg"

if [[ "${EUID}" -eq 0 ]]; then
	APT=(apt-get)
else
	APT=(sudo apt-get)
fi

export DEBIAN_FRONTEND=noninteractive
log "updating package metadata"
"${APT[@]}" update

log "installing native, Rust, Go, and audit dependencies"
"${APT[@]}" install --no-install-recommends -y \
	build-essential \
	ca-certificates \
	cmake \
	curl \
	git \
	golang-go \
	pkg-config \
	valgrind

if ! command -v rustup >/dev/null 2>&1; then
	log "installing Rust through the official rustup installer"
	curl --proto '=https' --tlsv1.2 --fail --silent --show-error https://sh.rustup.rs \
		| sh -s -- -y --profile minimal --default-toolchain stable
else
	warn "rustup already exists; leaving the installed toolchain unchanged"
fi

export PATH="${HOME}/.cargo/bin:${PATH}"
command -v rustup >/dev/null 2>&1 || die "rustup is not available after installation"
rustup toolchain install stable --profile minimal --no-self-update
rustup default stable

for required in g++ cmake go rustc cargo valgrind; do
	command -v "${required}" >/dev/null 2>&1 || die "required command is missing: ${required}"
done

ok "development environment is ready"
printf '  g++:     %s\n' "$(g++ --version | head -n 1)"
printf '  cmake:   %s\n' "$(cmake --version | head -n 1)"
printf '  rustc:   %s\n' "$(rustc --version)"
printf '  cargo:   %s\n' "$(cargo --version)"
printf '  go:      %s\n' "$(go version)"
printf '  valgrind: %s\n' "$(valgrind --version)"
