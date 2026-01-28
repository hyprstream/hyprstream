# Hyprstream Development Guide

Instructions for building Hyprstream from source.

## Prerequisites

- Rust toolchain (1.75+)
- LibTorch (PyTorch C++ library)
- pkg-config, OpenSSL dev headers
- Cap'n Proto compiler (for RPC)

## Building from Source

### 1. Clone Repository

```bash
git clone https://github.com/hyprstream/hyprstream.git
cd hyprstream
```

### 2. Install LibTorch

**Option A: Automatic Download (Recommended)**
```bash
# tch-rs will automatically download libtorch during build
cargo build --release
```

**Option B: Download from PyTorch**
```bash
# CUDA 12.8
wget https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip
unzip libtorch-shared-with-deps-2.10.0+cu128.zip

# CUDA 13.0
wget https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.10.0%2Bcu130.zip
unzip libtorch-shared-with-deps-2.10.0+cu130.zip

# ROCm 7.1
wget https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-2.10.0%2Brocm7.1.zip
unzip libtorch-shared-with-deps-2.10.0+rocm7.1.zip

# CPU
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip
unzip libtorch-shared-with-deps-2.10.0+cpu.zip
```

**Option C: Use Existing PyTorch Installation**
```bash
export LIBTORCH_USE_PYTORCH=1
```

### 3. Set Environment Variables

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

### 4. Build

**CPU Backend (Default)**
```bash
cargo build --release
```

**CUDA Backend**
```bash
cargo build --release --no-default-features --features tch-cuda
```

**ROCm Backend**
```bash
cargo build --release --no-default-features --features tch-rocm
```

### 5. Run

```bash
./target/release/hyprstream --help
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `tch-cpu` | CPU inference (default) |
| `tch-cuda` | NVIDIA CUDA support |
| `tch-rocm` | AMD ROCm support |
| `otel` | OpenTelemetry tracing |
| `xet` | XET large file storage (experimental) |
| `gittorrent` | P2P git transport |
| `systemd` | Systemd service integration |

### Combining Features

```bash
# CUDA + OpenTelemetry
cargo build --release --no-default-features --features tch-cuda,otel

# ROCm + XET support
cargo build --release --no-default-features --features tch-rocm,xet

# Full featured CPU build
cargo build --release --features otel,gittorrent,xet
```

## Running Tests

```bash
# Run all tests
cargo test --workspace

# Test specific crate
cargo test -p hyprstream
cargo test -p git2db

# Test with logging
RUST_LOG=debug cargo test

# Run examples (GPU/inference testing)
cargo run --example test_cuda --release
cargo run --example qwen_chat --release
```

## Code Quality

```bash
# Format code
cargo fmt --all

# Lint
cargo clippy --all-targets --all-features

# Check without building
cargo check --workspace
```

## Development Environment

### Required System Packages (Debian/Ubuntu)

```bash
apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libsystemd-dev \
    git \
    capnproto \
    ca-certificates
```

### ROCm Development

```bash
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx90a  # or gfx1100, gfx1101, etc.
```

### Static Linking

```bash
export LIBTORCH_STATIC=1
cargo build --release
```

## Project Structure

```
hyprstream/
├── crates/
│   ├── hyprstream/          # Main application
│   ├── git2db/              # Git repository management
│   ├── git-xet-filter/      # XET large file storage
│   └── ...
├── docs/                    # Architecture documentation
├── Cargo.toml               # Workspace manifest
├── Dockerfile               # Multi-variant container build
└── docker-compose.yml       # Service orchestration
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.
