#!/bin/bash
# build-appimage.sh - Build hyprstream AppImages
#
# Usage:
#   ./build-appimage.sh <command> [variant] [options]
#
# Commands:
#   build [VARIANT]      Build and package AppImage (default: all variants + universal)
#   clean [VARIANT]      Clean libtorch cache and build artifacts
#   help                 Show this help message
#
# Variants:
#   cpu, cuda128, cuda130, rocm71, universal, all (default)
#
# Options:
#   --version VERSION    Set version string (default: dev)
#
# Examples:
#   ./build-appimage.sh build                    # Build all variants + universal
#   ./build-appimage.sh build cpu --version 1.0  # Build only CPU variant
#   ./build-appimage.sh build universal          # Build all variants into universal AppImage
#   ./build-appimage.sh clean                    # Clean everything
#   ./build-appimage.sh clean cuda128            # Clean only CUDA 12.8 libtorch
#
# Environment:
#   LIBTORCH_CACHE_DIR   Directory to cache libtorch downloads (default: ./libtorch-cache)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
VERSION="${VERSION:-dev}"
LIBTORCH_VERSION="2.10.0"
LIBTORCH_CACHE_DIR="${LIBTORCH_CACHE_DIR:-$SCRIPT_DIR/libtorch-cache}"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"

# All variants
ALL_VARIANTS=(cpu cuda128 cuda130 rocm71)

# libtorch download URLs
declare -A LIBTORCH_URLS=(
    [cpu]="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
    [cuda128]="https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip"
    [cuda130]="https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu130.zip"
    [rocm71]="https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Brocm7.1.zip"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

show_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Validate variant name
validate_variant() {
    local variant="$1"
    if [[ "$variant" == "all" ]] || [[ "$variant" == "universal" ]]; then
        return 0
    fi
    if [[ ! "${LIBTORCH_URLS[$variant]+isset}" ]]; then
        log_error "Invalid variant: $variant"
        log_error "Valid variants: ${ALL_VARIANTS[*]} universal all"
        exit 1
    fi
}

# Ensure appimagetool exists
ensure_appimagetool() {
    if command -v appimagetool &>/dev/null; then
        APPIMAGETOOL="appimagetool"
        return
    fi

    APPIMAGETOOL="$SCRIPT_DIR/appimagetool"
    if [[ -x "$APPIMAGETOOL" ]]; then
        return
    fi

    log_info "Downloading appimagetool..."
    curl -sSL -o "$APPIMAGETOOL" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$APPIMAGETOOL"
}

# Download libtorch for a variant
download_libtorch() {
    local variant="$1"
    local url="${LIBTORCH_URLS[$variant]}"
    local cache_file="$LIBTORCH_CACHE_DIR/libtorch-${LIBTORCH_VERSION}-${variant}.zip"
    local extract_dir="$LIBTORCH_CACHE_DIR/$variant"

    mkdir -p "$LIBTORCH_CACHE_DIR"

    if [[ ! -f "$cache_file" ]]; then
        log_info "Downloading libtorch for $variant..."
        curl -sSL -o "$cache_file" "$url"
    fi

    if [[ ! -d "$extract_dir/libtorch" ]]; then
        log_info "Extracting libtorch for $variant..."
        mkdir -p "$extract_dir"
        unzip -q "$cache_file" -d "$extract_dir"
    fi
}

# Build hyprstream binary for a variant
build_binary() {
    local variant="$1"
    local libtorch_dir="$LIBTORCH_CACHE_DIR/$variant/libtorch"

    log_info "Building hyprstream for $variant..."

    mkdir -p "$BUILD_DIR/bin"

    export LIBTORCH="$libtorch_dir"
    export LD_LIBRARY_PATH="$libtorch_dir/lib:${LD_LIBRARY_PATH:-}"
    export LIBTORCH_BYPASS_VERSION_CHECK=1

    (cd "$PROJECT_ROOT" && cargo build --release --features otel)

    cp "$PROJECT_ROOT/target/release/hyprstream" "$BUILD_DIR/bin/hyprstream-$variant"
    log_success "Built hyprstream-$variant"
}

# Create per-backend AppImage
create_appimage() {
    local variant="$1"
    local appdir="$BUILD_DIR/hyprstream-$variant.AppDir"
    local output="$OUTPUT_DIR/hyprstream-${VERSION}-${variant}-x86_64.AppImage"

    log_info "Creating AppImage for $variant..."

    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/libtorch/lib"

    cp "$BUILD_DIR/bin/hyprstream-$variant" "$appdir/usr/bin/hyprstream"
    # Copy only shared libraries (skip static libs, headers, cmake)
    cp "$LIBTORCH_CACHE_DIR/$variant/libtorch/lib/"*.so* "$appdir/usr/lib/libtorch/lib/"

    sed "s/HYPRSTREAM_VARIANT:-cpu/HYPRSTREAM_VARIANT:-$variant/" \
        "$SCRIPT_DIR/AppRun-single" > "$appdir/AppRun"
    chmod +x "$appdir/AppRun"

    cp "$SCRIPT_DIR/hyprstream.desktop" "$appdir/"
    cp "$SCRIPT_DIR/hyprstream.svg" "$appdir/"

    mkdir -p "$OUTPUT_DIR"
    ARCH=x86_64 "$APPIMAGETOOL" "$appdir" "$output"
    log_success "Created: $output"
}

# Create universal AppImage with all backends
create_universal_appimage() {
    local appdir="$BUILD_DIR/hyprstream-universal.AppDir"
    local output="$OUTPUT_DIR/hyprstream-${VERSION}-x86_64.AppImage"
    local staging="$BUILD_DIR/universal-staging"

    log_info "Creating universal AppImage..."

    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib"

    for variant in "${ALL_VARIANTS[@]}"; do
        # Use staged files if available (from stage command), otherwise use build dirs
        if [[ -f "$staging/bin/hyprstream-$variant" ]]; then
            cp "$staging/bin/hyprstream-$variant" "$appdir/usr/bin/"
            mkdir -p "$appdir/usr/lib/$variant/libtorch/lib"
            cp "$staging/lib/$variant/libtorch/lib/"*.so* "$appdir/usr/lib/$variant/libtorch/lib/"
        else
            cp "$BUILD_DIR/bin/hyprstream-$variant" "$appdir/usr/bin/"
            mkdir -p "$appdir/usr/lib/$variant/libtorch/lib"
            cp "$LIBTORCH_CACHE_DIR/$variant/libtorch/lib/"*.so* "$appdir/usr/lib/$variant/libtorch/lib/"
        fi
    done

    cp "$SCRIPT_DIR/AppRun" "$appdir/"
    chmod +x "$appdir/AppRun"
    cp "$SCRIPT_DIR/hyprstream.desktop" "$appdir/"
    cp "$SCRIPT_DIR/hyprstream.svg" "$appdir/"

    mkdir -p "$OUTPUT_DIR"
    ARCH=x86_64 "$APPIMAGETOOL" "$appdir" "$output"
    log_success "Created: $output"
}

# Command: build
cmd_build() {
    local variant="${1:-all}"
    validate_variant "$variant"

    log_info "Building hyprstream AppImage"
    log_info "Version: $VERSION"
    log_info "Variant: $variant"
    df -h | grep -E '^/dev|Filesystem'

    ensure_appimagetool

    if [[ "$variant" == "all" ]]; then
        # Build all per-backend AppImages + universal
        for v in "${ALL_VARIANTS[@]}"; do
            download_libtorch "$v"
            build_binary "$v"
            create_appimage "$v"
        done
        create_universal_appimage
    elif [[ "$variant" == "universal" ]]; then
        # Build universal only (requires all variants)
        for v in "${ALL_VARIANTS[@]}"; do
            download_libtorch "$v"
            build_binary "$v"
        done
        create_universal_appimage
    else
        # Build single variant
        download_libtorch "$variant"
        build_binary "$variant"
        create_appimage "$variant"
    fi

    log_success "Build complete!"
    ls -lh "$OUTPUT_DIR"/*.AppImage 2>/dev/null || true
}

# Command: stage - copy files needed for universal AppImage before cleaning
cmd_stage() {
    local variant="${1:-}"
    if [[ -z "$variant" ]] || [[ "$variant" == "all" ]] || [[ "$variant" == "universal" ]]; then
        log_error "stage requires a specific variant (cpu, cuda128, cuda130, rocm71)"
        exit 1
    fi
    validate_variant "$variant"

    local staging="$BUILD_DIR/universal-staging"
    log_info "Staging $variant for universal AppImage..."

    # Stage binary
    mkdir -p "$staging/bin"
    cp "$BUILD_DIR/bin/hyprstream-$variant" "$staging/bin/"

    # Stage only shared libraries (much smaller than full libtorch)
    mkdir -p "$staging/lib/$variant/libtorch/lib"
    cp "$LIBTORCH_CACHE_DIR/$variant/libtorch/lib/"*.so* "$staging/lib/$variant/libtorch/lib/"

    log_success "Staged $variant"
    du -sh "$staging"
}

# Command: clean
cmd_clean() {
    local variant="${1:-all}"

    if [[ "$variant" == "all" ]]; then
        log_info "Cleaning all build artifacts..."
        rm -rf "$BUILD_DIR" "$OUTPUT_DIR" "$LIBTORCH_CACHE_DIR"
        log_success "Cleaned everything"
    else
        validate_variant "$variant"
        log_info "Cleaning $variant..."
        rm -rf "$LIBTORCH_CACHE_DIR/$variant"
        rm -f "$LIBTORCH_CACHE_DIR/libtorch-${LIBTORCH_VERSION}-${variant}.zip"
        rm -f "$BUILD_DIR/bin/hyprstream-$variant"
        rm -rf "$BUILD_DIR/hyprstream-$variant.AppDir"
        log_success "Cleaned $variant"
    fi
    df -h | grep -E '^/dev|Filesystem'
}

# Main
main() {
    local cmd="${1:-help}"
    shift || true

    # Parse remaining args for options
    local variant=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                if [[ -z "$variant" ]]; then
                    variant="$1"
                fi
                shift
                ;;
        esac
    done

    case "$cmd" in
        build)
            cmd_build "$variant"
            ;;
        stage)
            cmd_stage "$variant"
            ;;
        clean)
            cmd_clean "$variant"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $cmd"
            log_error "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
