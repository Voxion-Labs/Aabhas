#!/usr/bin/env bash
set -Eeuo pipefail

# Provision a disposable, network-isolated QEMU guest. This script does not
# mount host model directories into the guest and does not enable guest LAN.

VM_NAME="${AABHAS_VM_NAME:-aabhas-sandbox}"
VM_DIR="${AABHAS_VM_DIR:-${HOME}/.local/share/aabhas-vm/${VM_NAME}}"
IMAGE_NAME="${AABHAS_IMAGE_NAME:-noble-server-cloudimg-amd64.img}"
IMAGE_URL="${AABHAS_IMAGE_URL:-https://cloud-images.ubuntu.com/noble/current/${IMAGE_NAME}}"
CHECKSUMS_URL="${AABHAS_CHECKSUMS_URL:-https://cloud-images.ubuntu.com/noble/current/SHA256SUMS}"
DISK_SIZE="${AABHAS_VM_DISK_SIZE:-80G}"
MEMORY="${AABHAS_VM_MEMORY:-4096}"
VCPUS="${AABHAS_VM_VCPUS:-2}"

BASE_IMAGE="${VM_DIR}/${IMAGE_NAME}"
DISK_IMAGE="${VM_DIR}/${VM_NAME}.qcow2"
SEED_IMAGE="${VM_DIR}/${VM_NAME}-seed.iso"
RUNNER="${VM_DIR}/run-${VM_NAME}.sh"

die() { printf 'setup_vm: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"; }
trap 'die "failed at line ${BASH_LINENO[0]}"' ERR

[[ "${VM_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || die "AABHAS_VM_NAME contains unsupported characters"

if ! command -v qemu-img >/dev/null 2>&1 || \
	! command -v qemu-system-x86_64 >/dev/null 2>&1 || \
	! command -v cloud-localds >/dev/null 2>&1 || \
	! command -v curl >/dev/null 2>&1; then
	command -v apt-get >/dev/null 2>&1 || die "install QEMU and cloud-image-utils before running this script"
	if [[ "${EUID}" -eq 0 ]]; then
		APT=(apt-get)
	else
		command -v sudo >/dev/null 2>&1 || die "run as root or install sudo"
		APT=(sudo apt-get)
	fi
	"${APT[@]}" update
	"${APT[@]}" install --no-install-recommends -y \
		ca-certificates curl qemu-system-x86 qemu-utils cloud-image-utils
fi

need curl
need sha256sum
need qemu-img
need qemu-system-x86_64
need cloud-localds

if [[ ! -e /dev/kvm && "${AABHAS_ALLOW_TCG:-0}" != "1" ]]; then
	die "/dev/kvm is unavailable; set AABHAS_ALLOW_TCG=1 only for a slower software-emulation test"
fi

mkdir -p -- "${VM_DIR}"
chmod 0700 "${VM_DIR}"

if [[ ! -f "${BASE_IMAGE}" ]]; then
	printf 'setup_vm: downloading Ubuntu guest image\n'
	curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
		--output "${BASE_IMAGE}.partial" "${IMAGE_URL}"
	mv -- "${BASE_IMAGE}.partial" "${BASE_IMAGE}"
fi

if [[ "${AABHAS_SKIP_CHECKSUM:-0}" != "1" ]]; then
	curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
		--output "${VM_DIR}/SHA256SUMS" "${CHECKSUMS_URL}"
	expected="$(awk -v name="${IMAGE_NAME}" '$2 == name || $2 == "*" name { print $1; exit }' "${VM_DIR}/SHA256SUMS")"
	[[ -n "${expected}" ]] || die "no checksum found for ${IMAGE_NAME}"
	actual="$(sha256sum "${BASE_IMAGE}" | awk '{print $1}')"
	[[ "${actual}" == "${expected}" ]] || die "guest image checksum mismatch"
fi

if [[ ! -f "${DISK_IMAGE}" ]]; then
	qemu-img create -f qcow2 -F qcow2 -b "${BASE_IMAGE}" "${DISK_IMAGE}" "${DISK_SIZE}"
fi

cat >"${VM_DIR}/user-data" <<'EOF'
#cloud-config
disable_root: true
ssh_pwauth: false
preserve_hostname: false
users:
  - default
  - name: aabhas
    gecos: Aabhas Sandbox
    shell: /bin/bash
    lock_passwd: true
    sudo: []
packages: []
write_files:
  - path: /etc/aabhas-sandbox
    permissions: '0644'
    content: |
      This guest is intended for isolated model testing.
      No host directory is shared with the guest.
runcmd:
  - [systemctl, disable, --now, systemd-networkd-wait-online.service]
EOF
cat >"${VM_DIR}/meta-data" <<EOF
instance-id: ${VM_NAME}-$(date -u +%Y%m%dT%H%M%SZ)
local-hostname: ${VM_NAME}
EOF

cloud-localds --filesystem=ext4 "${SEED_IMAGE}" "${VM_DIR}/user-data" "${VM_DIR}/meta-data"

cat >"${RUNNER}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

DISK_MODE=()
if [[ "${AABHAS_VM_PERSIST:-0}" != "1" ]]; then
  DISK_MODE=(-snapshot)
fi
if [[ -e /dev/kvm ]]; then
  ACCEL=(-enable-kvm -machine q35,accel=kvm)
else
  ACCEL=(-accel tcg,thread=multi -machine q35)
fi

exec qemu-system-x86_64 \\
  "${ACCEL[@]}" \\
  -m "${MEMORY}" -smp "${VCPUS}" \\
  -nodefaults \\
  -drive "file=${DISK_IMAGE},if=virtio,format=qcow2,cache=none,aio=native" \\
  -drive "file=${SEED_IMAGE},if=virtio,format=raw,readonly=on" \\
  -nic none \\
  -display none \\
  -serial mon:stdio \\
  -no-reboot \\
  -sandbox on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny \\
  "${DISK_MODE[@]}"
EOF
chmod 0700 "${RUNNER}"

printf 'Aabhas VM prepared at %s\n' "${VM_DIR}"
printf 'Run disposable guest: %s\n' "${RUNNER}"
printf 'Persistent disk mode: AABHAS_VM_PERSIST=1 %s\n' "${RUNNER}"
