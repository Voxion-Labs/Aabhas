<p align="center">
  <img src="docs/assets/Aabhas.jpg" width="130" alt="Aabhas logo">
</p>

<h1 align="center">Aabhas</h1>

<p align="center">
  Enterprise out-of-core tensor streaming for large language model inference.
</p>

<p align="center">
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-safety%20grid-000000">
  <img alt="Go" src="https://img.shields.io/badge/Go-control%20plane-00ADD8">
  <img alt="License" src="https://img.shields.io/badge/license-see%20LICENSE-lightgrey">
</p>

## Overview

Aabhas is a modular inference-engine foundation for models whose tensor layers
are larger than the available host memory. It separates request orchestration,
health admission, and native data movement into independently auditable layers.

The engine is designed around four operational properties:

- bounded resident memory instead of whole-model materialization;
- sequential storage access for large binary tensor files;
- a narrow, panic-safe C ABI between the Go control plane and Rust safety grid;
- explicit container and virtual-machine isolation options for test workloads.

This repository contains the control and data-plane foundations. It is not a
complete model-format implementation or a replacement for production identity,
authorization, monitoring, or Kubernetes policy controls.

## Architecture

```text
                         +---------------------------+
                         | CLI / authenticated proxy |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | Go Control-Plane          |
                         | REST + lifecycle gate     |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | Rust Safety Grid          |
                         | RAM / SSD / FFI boundary  |
                         +-------------+-------------+
                                       |
                                       v
 +------------------+       +---------------------------+
 | NVMe SSD models  | ----> | C++17 Data-Plane          |
 | .bin tensor data |       | fd + mmap + layer windows |
 +------------------+       +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | Tensor consumer            |
                         | bounded resident pages     |
                         +---------------------------+
```

### Data plane

The C++17 data plane uses POSIX file descriptors, optional Linux `O_DIRECT`,
`mmap`, and `madvise`. `LayerStreamer` maps one bounded layer chunk at a time,
marks sequential access, and applies `MADV_DONTNEED` before dropping a previous
mapping. The kernel controls actual page residency; the hints do not promise an
instantaneous physical-memory reclaim.

### Safety grid

The Rust `cdylib` exposes only `start_aabhas_engine` and
`stop_aabhas_engine`. FFI entry points catch panics and return stable integer
status codes. Admission checks reject starts when host memory reaches the 70%
threshold or the model filesystem has less than 20 GiB available.

### Control plane

The Go daemon uses the standard `net/http` library. Its native bridge copies
model paths into C-owned memory for the duration of each call and serializes
native calls. The orchestrator package provides a thread-safe layer queue and a
single-model execution lease for higher-level scheduling.

## Repository layout

```text
core-engine/                  C++17 data plane and Rust safety grid
control-plane/                Go daemon, CLI, API, and orchestration
ffi-bridge/                   C ABI header and protobuf schema
stealth-deploy/               Container and Kubernetes deployment assets
sandbox/                      Isolated development and VM tooling
tests/                        End-to-end and Valgrind audit scripts
docs/                         API and architecture documentation
scripts/                      Setup and release automation
```

## Build prerequisites

Ubuntu/Debian development hosts require:

- GCC/G++, CMake, and POSIX development headers;
- Rust and Cargo;
- Go 1.22 or newer;
- Bash, `curl`, `valgrind`, and standard archive utilities;
- Docker for the isolated release pipeline.

The repository setup script installs the compiler and audit prerequisites:

```bash
sudo ./scripts/dev_setup.sh
```

Review the script before running it on a managed host. It installs packages
through the configured Ubuntu/Debian package manager and Rust through the
official `rustup` installer.

## Build

```bash
make build-core
make build-go
```

The Rust build produces `core-engine/rust_trap/target/{debug,release}/libaabhas.so`.
The Go build places `aabhas-cli` and `aabhas-daemon` in `bin/`.

For a deterministic stripped release archive:

```bash
AABHAS_RELEASE_VERSION=local-test ./scripts/build_release.sh
```

The archive contains both `aabhas-daemon` and `aabhas-cli`, the native
`libaabhas.so`, the license, and a SHA-256 sidecar.

## Run the daemon

The daemon defaults to `127.0.0.1:8080`:

```bash
./bin/aabhas-daemon
```

Set an explicit address when running behind a trusted service boundary:

```bash
AABHAS_LISTEN_ADDR=127.0.0.1:8080 ./bin/aabhas-daemon
```

The HTTP API is documented in [`docs/api_reference.md`](docs/api_reference.md).

## Control commands

Start a model:

```bash
./bin/aabhas-cli start \
  -addr 127.0.0.1:8080 \
  -model /models/example/model.bin
```

Stop the active model:

```bash
./bin/aabhas-cli stop -addr 127.0.0.1:8080
```

Equivalent HTTP requests:

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8080/api/v1/engine/start \
  -H 'Content-Type: application/json' \
  --data '{"model_path":"/models/example/model.bin"}'

curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8080/api/v1/engine/stop
```

## Tests and audits

Run the repository quality gates after the required toolchains are available:

```bash
make test
```

This runs the native CTest suite, Rust tests, race-enabled Go tests and vet,
plus shell syntax checks. The CI workflow runs the same categories on every
push and pull request.

End-to-end lifecycle test:

```bash
make build-go
AABHAS_DAEMON_BIN="$PWD/bin/aabhas-daemon" ./tests/e2e_tests/run_tests.sh
```

Native leak audit:

```bash
./tests/security_audits/run_valgrind.sh
```

Unauthorized-hardware protection test:

```bash
AABHAS_DAEMON_BIN="$PWD/bin/aabhas-daemon" ./scripts/trigger_poison_test.sh
```

The audit script compiles a temporary native layer-streaming harness and
removes all generated test files on exit. The poison-protection test uses a
temporary buffer and mock UUID values; it never scrambles model files or host
memory.

## Container and VM isolation

Build the production image:

```bash
docker build -f stealth-deploy/Dockerfile.prod -t aabhas:local .
```

The production image runs as a non-root UID, uses a read-only root filesystem
in Compose/Kubernetes configurations, and drops capabilities except the
explicitly requested `IPC_LOCK` capability. Host SSD paths should be mounted
read-only for model consumption.

Container health is exposed through `/healthz` and `/readyz`; the production
image and Compose configuration use the bundled `aabhas-cli health` probe.
Kubernetes deployments require the `aabhas-api-auth` Secret with a `token` key
for control-plane authentication.

For a non-root diagnostic environment:

```bash
docker build -f sandbox/docker-test-env/Dockerfile.dev -t aabhas-dev .
docker run --rm -it --mount type=bind,src="$PWD",dst=/workspace aabhas-dev
```

The VM helper creates a disposable QEMU/KVM guest with no default network and
no host-directory sharing. Review host virtualization policy before running it:

```bash
sudo ./sandbox/vm-scripts/setup_vm.sh
```

## Operational limits

- The control plane supports one active model per daemon process.
- RAM and storage checks are admission controls, not substitutes for cgroups,
  quotas, or node-level monitoring.
- The REST API has no authentication by itself. Keep it on loopback or place
  it behind an authenticated, mutually trusted proxy.
- Kubernetes `hostPath` volumes require operator review and pre-provisioned
  directories on eligible nodes.

## Security and repository policy

The release workflow is the only workflow granted `contents: write`, and its
job uses the GitHub Actions bot identity after successful main-branch CI. The
repository owner must still configure GitHub branch rulesets, required
Code Owner approval, push restrictions, and fork policy in repository settings.
Those account-level controls cannot be enforced by source files alone.

## License

See [`LICENSE`](LICENSE) for the repository license and distribution terms.
