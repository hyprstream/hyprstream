#!/bin/bash
# hyprstream.sh - Launch script with sensible defaults and ENV overrides

# Server configuration (all overridable via ENV)
export HYPRSTREAM_SERVER_HOST=${HYPRSTREAM_SERVER_HOST:-0.0.0.0}
export HYPRSTREAM_CORS_ENABLED=${HYPRSTREAM_CORS_ENABLED:-true}
export HYPRSTREAM_CORS_ORIGINS=${HYPRSTREAM_CORS_ORIGINS:-"*"}

# Paths - relative to script location by default, override with ENV
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR=${DEPS_DIR:-$SCRIPT_DIR/target-deps}

# ROCm path (for bitsandbytes compatibility with libtorch)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}

# Libtorch location
export LIBTORCH=${LIBTORCH:-$DEPS_DIR/libtorch}

# Bitsandbytes library directory (for LD_LIBRARY_PATH)
# If BITSANDBYTES_LIB_PATH is a file, use its directory; otherwise use as-is
if [ -n "$BITSANDBYTES_LIB_PATH" ] && [ -f "$BITSANDBYTES_LIB_PATH" ]; then
  BITSANDBYTES_LIB_DIR=$(dirname "$BITSANDBYTES_LIB_PATH")
else
  BITSANDBYTES_LIB_DIR=${BITSANDBYTES_LIB_PATH:-$DEPS_DIR/bitsandbytes/bitsandbytes}
fi

# GPU architecture
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a}

# Libtorch build settings
export LIBTORCH_STATIC=0
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Library paths - libtorch first (includes bundled ROCm), then bitsandbytes
export LD_LIBRARY_PATH=$LIBTORCH/lib:$BITSANDBYTES_LIB_DIR:$LD_LIBRARY_PATH

# Run hyprstream with all passed arguments
exec "$SCRIPT_DIR/target/release/hyprstream" "$@"
