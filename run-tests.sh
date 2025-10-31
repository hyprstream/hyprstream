#!/bin/bash

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROCm/HIP environment setup
export LD_LIBRARY_PATH="$SCRIPT_DIR/libtorch/lib:$LD_LIBRARY_PATH"
export ROCM_PATH=/usr
export PYTORCH_ROCM_ARCH=gfx90a
export LIBTORCH="$SCRIPT_DIR/libtorch"
export LIBTORCH_STATIC=0
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Run cargo test with all passed arguments
cargo test --release --features otel "$@"
