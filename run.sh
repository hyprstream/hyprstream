#!/bin/bash

export HYPRSTREAM_SERVER_HOST=0.0.0.0
export HYPRSTREAM_CORS_ENABLED=true
export HYPRSTREAM_CORS_ORIGINS="*"

# ROCm/HIP environment setup
export LD_LIBRARY_PATH=./libtorch/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/usr
export PYTORCH_ROCM_ARCH=gfx90a
export LIBTORCH=./libtorch
export LIBTORCH_STATIC=0
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Run hyprstream with all passed arguments
./target/release/hyprstream "$@"
