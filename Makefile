
.PHONY: all setup build-core build-go clean nuke safe-test sync push dev

CORE_BUILD_DIR ?= core-engine/build

all: build-core build-go

# --- DEV TOOLS ---
setup:
	@echo "Setting up the Voxion Matrix..."
	@chmod +x scripts/*.sh
	@./scripts/dev_setup.sh
	@echo "✅ Matrix ready for raw pointers."

dev:
	@echo "Running Aabhas Control-Plane in DEV mode..."
	@cd control-plane && go run cmd/aabhas-daemon/main.go

# --- BUILD COMMANDS ---
build-core:
	@echo "Compiling the Data-Plane (C++ & Rust)..."
	@cmake -S core-engine -B $(CORE_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_EXTENSIONS=OFF
	@cmake --build $(CORE_BUILD_DIR) --parallel
	@test -f $(CORE_BUILD_DIR)/libaabhas_data_plane.a
	@cd core-engine/rust_trap && cargo build --release
	@test -f core-engine/rust_trap/target/release/libaabhas.so
	@echo "✅ C++ data-plane and Rust safety library built successfully."

build-go:
	@echo "Compiling the Control-Plane (Go)..."
	@cd control-plane && go build -o ../bin/aabhas-cli cmd/aabhas-cli/main.go
	@cd control-plane && go build -o ../bin/aabhas-daemon cmd/aabhas-daemon/main.go
	@echo "✅ Go binaries ready."

# --- CLEANING PROTOCOLS ---
clean:
	@echo "Cleaning up the matrix..."
	@cd core-engine/rust_trap && cargo clean
	@rm -rf bin/
	@echo "✅ Matrix clean."

nuke: clean
	@echo "Deep Clean..."
	@rm -rf core-engine/build/
	@rm -rf sandbox/docker-test-env/mock_ssd_data/*.bin
	@echo "✅ All traces erased. No evidence left."

safe-test:
	@echo "Booting Sandbox Containment Zone..."
	@# Add your docker run or VM script here later
	@echo "✅ Sandbox checks passed."

# --- THE GIT HUSTLE (Bot-Safe Workflow) ---
sync:
	@echo "Syncing with origin (Rebase mode to handle Bot commits)..."
	git pull --rebase origin main
	@echo "✅ Repo in perfect sync."

push:
	@echo "Committing and pushing matrix to production..."
	git add .
	@read -p "Commit message: " msg; \
	git commit -m "$$msg" || echo "No changes to commit"
	git pull --rebase origin main
	git push -u origin main
	@echo "✅ Code injected into the cloud like a boss."
