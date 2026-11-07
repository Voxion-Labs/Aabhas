#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aabhas-valgrind.XXXXXX")"
MODEL_PATH="${BUILD_DIR}/model.bin"
TEST_BINARY="${BUILD_DIR}/layer-streamer-test"

cleanup() { rm -rf -- "${BUILD_DIR}"; }
trap cleanup EXIT INT TERM

command -v "${CXX}" >/dev/null 2>&1 || { echo "run_valgrind: CXX not found: ${CXX}" >&2; exit 1; }
command -v valgrind >/dev/null 2>&1 || { echo "run_valgrind: valgrind is required" >&2; exit 1; }

dd if=/dev/zero of="${MODEL_PATH}" bs=1048576 count=1 status=none

"${CXX}" -std=c++17 -O0 -g -Wall -Wextra -Wpedantic \
	-I"${REPO_ROOT}/core-engine/include" \
	"${REPO_ROOT}/core-engine/src/memory/madvise_utils.cpp" \
	"${REPO_ROOT}/core-engine/src/memory/mmap_illusion.cpp" \
	"${REPO_ROOT}/core-engine/src/hardware/nvme_direct.cpp" \
	"${REPO_ROOT}/core-engine/src/tensor_ops/layer_streamer.cpp" \
	-x c++ -o "${TEST_BINARY}" - <<'CPP'
#include "tensor_ops/layer_streamer.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int fd = ::open(argv[1], O_RDONLY);
    if (fd < 0) return 3;
    struct stat metadata {};
    const bool stat_ok = ::fstat(fd, &metadata) == 0 && metadata.st_size > 0;
    (void)::close(fd);
    if (!stat_ok) return 4;

    using aabhas::tensor_ops::LayerDescriptor;
    using aabhas::tensor_ops::LayerStreamer;
    LayerStreamer streamer;
    std::vector<LayerDescriptor> layers{{0, static_cast<std::size_t>(metadata.st_size)}};
    if (!streamer.open(argv[1], std::move(layers), 16U * 1024U * 1024U)) return 5;
    if (!streamer.fetch(0) || !streamer.current()) return 6;
    streamer.drop();
    streamer.close();
    return 0;
}
CPP

valgrind --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=all \
	--error-exitcode=1 --track-origins=yes "${TEST_BINARY}" "${MODEL_PATH}"
echo "Aabhas Valgrind audit passed"
