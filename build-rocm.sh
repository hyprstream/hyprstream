#!/bin/bash

# Build script for hyprstream-torch with ROCm/HIP support
# Automates the environment setup and build process for AMD GPU acceleration
#
# Usage:
#   ./build-rocm.sh [debug|release|check|test]
#   LIBTORCH=/path/to/libtorch ./build-rocm.sh release
#
# Environment Variables:
#   LIBTORCH - Path to libtorch installation (default: ./libtorch)

set -e  # Exit on any error

echo "🚀 Building hyprstream-torch with ROCm/HIP support..."
echo ""

# Set LIBTORCH path - use environment variable if set, otherwise default to ./libtorch
if [ -z "$LIBTORCH" ]; then
    export LIBTORCH="./libtorch"
fi

# ROCm/HIP environment setup
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/usr
export PYTORCH_ROCM_ARCH=gfx90a
export LIBTORCH_STATIC=0
export LIBTORCH_BYPASS_VERSION_CHECK=1

echo "📋 Environment Configuration:"
echo "   LIBTORCH: $LIBTORCH"
echo "   ROCM_PATH: $ROCM_PATH"
echo "   PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
echo "   LIBTORCH_STATIC: $LIBTORCH_STATIC"
echo ""

# Verify libtorch installation
if [ ! -d "$LIBTORCH" ]; then
    echo "❌ Error: libtorch directory not found at $LIBTORCH"
    echo "   Please ensure ROCm-compatible libtorch is installed"
    exit 1
fi

if [ ! -f "$LIBTORCH/lib/libtorch_hip.so" ]; then
    echo "❌ Error: libtorch_hip.so not found"
    echo "   Please ensure you have ROCm-enabled PyTorch libtorch"
    exit 1
fi

echo "✅ libtorch installation verified"
echo ""

# Check for HIP/ROCm
if command -v rocm-smi &> /dev/null; then
    echo "📊 ROCm GPU Status:"
    rocm-smi --showproductname --showtemp --showuse
    echo ""
else
    echo "⚠️  rocm-smi not found - ROCm may not be properly installed"
fi

# Build options
BUILD_TYPE="${1:-release}"

case $BUILD_TYPE in
    "debug"|"dev")
        echo "🔧 Building debug version..."
        cargo build
        ;;
    "release"|"prod")
        echo "🔧 Building optimized release version..."
        cargo build --release
        ;;
    "check")
        echo "🔍 Checking code compilation..."
        cargo check
        ;;
    "test")
        echo "🧪 Running tests..."
        cargo test
        ;;
    *)
        echo "Usage: $0 [debug|release|check|test]"
        echo "  debug   - Build debug version (faster compilation)"
        echo "  release - Build optimized release version (default)"
        echo "  check   - Check compilation without building"
        echo "  test    - Run test suite"
        exit 1
        ;;
esac

echo ""
if [ "$BUILD_TYPE" = "release" ] || [ "$BUILD_TYPE" = "prod" ]; then
    if [ -f "./target/release/hyprstream" ]; then
        echo "✅ Build successful! Binary: ./target/release/hyprstream"
        echo "🚀 Run with: ./run.sh [args...]"
    else
        echo "❌ Build failed - binary not found"
        exit 1
    fi
elif [ "$BUILD_TYPE" = "debug" ] || [ "$BUILD_TYPE" = "dev" ]; then
    if [ -f "./target/debug/hyprstream" ]; then
        echo "✅ Debug build successful! Binary: ./target/debug/hyprstream"
        echo "🐛 Run with: ./target/debug/hyprstream [args...]"
    else
        echo "❌ Build failed - binary not found"
        exit 1
    fi
fi
