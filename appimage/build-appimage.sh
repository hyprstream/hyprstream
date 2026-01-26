#!/bin/bash
# build-appimage.sh - Build hyprstream AppImages locally
#
# Usage:
#   ./build-appimage.sh [OPTIONS]
#
# Options:
#   --version VERSION    Set version string (default: dev)
#   --variant VARIANT    Build only specific variant (cpu, cuda128, cuda130, rocm71, universal)
#   --skip-download      Skip downloading libtorch (use cached)
#   --skip-build         Skip building binaries (use cached)
#   --clean              Clean build artifacts before building
#   --help               Show this help message
#
# Environment:
#   LIBTORCH_CACHE_DIR   Directory to cache libtorch downloads (default: ./libtorch-cache)
#   CARGO_TARGET_DIR     Cargo target directory (default: ./target)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
VERSION="${VERSION:-dev}"
LIBTORCH_VERSION="2.10.0"
LIBTORCH_CACHE_DIR="${LIBTORCH_CACHE_DIR:-$SCRIPT_DIR/libtorch-cache}"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Variants to build
VARIANTS=(cpu cuda128 cuda130 rocm71)
BUILD_VARIANT=""
SKIP_DOWNLOAD=false
SKIP_BUILD=false
CLEAN=false

# libtorch download URLs (cxx11 ABI)
declare -A LIBTORCH_URLS=(
    [cpu]="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
    [cuda128]="https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip"
    [cuda130]="https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu130.zip"
    [rocm71]="https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Brocm7.1.zip"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

show_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --variant)
            BUILD_VARIANT="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate variant if specified
if [[ -n "$BUILD_VARIANT" ]] && [[ "$BUILD_VARIANT" != "universal" ]]; then
    if [[ ! "${LIBTORCH_URLS[$BUILD_VARIANT]+isset}" ]]; then
        log_error "Invalid variant: $BUILD_VARIANT"
        log_error "Valid variants: ${VARIANTS[*]} universal"
        exit 1
    fi
    VARIANTS=("$BUILD_VARIANT")
fi

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local missing=()

    command -v curl &>/dev/null || missing+=("curl")
    command -v unzip &>/dev/null || missing+=("unzip")
    command -v cargo &>/dev/null || missing+=("cargo (rust)")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi

    # Check for appimagetool
    if ! command -v appimagetool &>/dev/null; then
        if [[ ! -x "$SCRIPT_DIR/appimagetool" ]]; then
            log_warn "appimagetool not found, will download..."
        fi
    fi

    log_success "All dependencies satisfied"
}

# Download appimagetool if needed
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
    log_success "Downloaded appimagetool"
}

# Download libtorch for a variant
download_libtorch() {
    local variant="$1"
    local url="${LIBTORCH_URLS[$variant]}"
    local cache_file="$LIBTORCH_CACHE_DIR/libtorch-${LIBTORCH_VERSION}-${variant}.zip"
    local extract_dir="$LIBTORCH_CACHE_DIR/$variant"

    if [[ -d "$extract_dir/libtorch" ]] && $SKIP_DOWNLOAD; then
        log_info "Using cached libtorch for $variant"
        return
    fi

    mkdir -p "$LIBTORCH_CACHE_DIR"

    if [[ ! -f "$cache_file" ]]; then
        log_info "Downloading libtorch for $variant..."
        curl -sSL -o "$cache_file" "$url"
        log_success "Downloaded libtorch for $variant"
    else
        log_info "Using cached download for $variant"
    fi

    if [[ ! -d "$extract_dir/libtorch" ]]; then
        log_info "Extracting libtorch for $variant..."
        mkdir -p "$extract_dir"
        unzip -q "$cache_file" -d "$extract_dir"
        log_success "Extracted libtorch for $variant"
    fi
}

# Build hyprstream for a variant
build_variant() {
    local variant="$1"
    local libtorch_dir="$LIBTORCH_CACHE_DIR/$variant/libtorch"
    local output_binary="$BUILD_DIR/bin/hyprstream-$variant"

    if [[ -f "$output_binary" ]] && $SKIP_BUILD; then
        log_info "Using cached binary for $variant"
        return
    fi

    log_info "Building hyprstream for $variant..."

    mkdir -p "$BUILD_DIR/bin"

    # Set up environment for this variant
    export LIBTORCH="$libtorch_dir"
    export LD_LIBRARY_PATH="$libtorch_dir/lib:${LD_LIBRARY_PATH:-}"
    export LIBTORCH_BYPASS_VERSION_CHECK=1

    # Build with appropriate features
    local features="otel"

    # Note: The tch-rs backend is determined by the libtorch that's linked,
    # not by cargo features. The binary links against whichever libtorch
    # is specified by $LIBTORCH at build time.

    (
        cd "$PROJECT_ROOT"
        cargo build --release --features "$features"
    )

    # Copy binary with variant suffix
    cp "$PROJECT_ROOT/target/release/hyprstream" "$output_binary"
    log_success "Built hyprstream for $variant"
}

# Create per-backend AppImage
create_per_backend_appimage() {
    local variant="$1"
    local appdir="$BUILD_DIR/hyprstream-$variant.AppDir"
    local output="$OUTPUT_DIR/hyprstream-${VERSION}-${variant}-x86_64.AppImage"

    log_info "Creating AppImage for $variant..."

    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib"

    # Copy binary (without variant suffix for per-backend)
    cp "$BUILD_DIR/bin/hyprstream-$variant" "$appdir/usr/bin/hyprstream"

    # Copy libtorch libraries
    cp -r "$LIBTORCH_CACHE_DIR/$variant/libtorch" "$appdir/usr/lib/"

    # Copy AppRun (with variant embedded)
    sed "s/HYPRSTREAM_VARIANT:-cpu/HYPRSTREAM_VARIANT:-$variant/" \
        "$SCRIPT_DIR/AppRun-single" > "$appdir/AppRun"
    chmod +x "$appdir/AppRun"

    # Copy desktop and icon
    cp "$SCRIPT_DIR/hyprstream.desktop" "$appdir/"
    cp "$SCRIPT_DIR/hyprstream.svg" "$appdir/"

    # Create AppImage
    mkdir -p "$OUTPUT_DIR"
    ARCH=x86_64 "$APPIMAGETOOL" "$appdir" "$output"

    log_success "Created: $output"
}

# Create universal AppImage
create_universal_appimage() {
    local appdir="$BUILD_DIR/hyprstream-universal.AppDir"
    local output="$OUTPUT_DIR/hyprstream-${VERSION}-x86_64.AppImage"

    log_info "Creating universal AppImage..."

    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib"

    # Copy all variant binaries
    for variant in "${VARIANTS[@]}"; do
        cp "$BUILD_DIR/bin/hyprstream-$variant" "$appdir/usr/bin/"

        # Copy libtorch for each variant
        mkdir -p "$appdir/usr/lib/$variant"
        cp -r "$LIBTORCH_CACHE_DIR/$variant/libtorch" "$appdir/usr/lib/$variant/"
    done

    # Copy universal AppRun
    cp "$SCRIPT_DIR/AppRun" "$appdir/"
    chmod +x "$appdir/AppRun"

    # Copy desktop and icon
    cp "$SCRIPT_DIR/hyprstream.desktop" "$appdir/"
    cp "$SCRIPT_DIR/hyprstream.svg" "$appdir/"

    # Create AppImage
    mkdir -p "$OUTPUT_DIR"
    ARCH=x86_64 "$APPIMAGETOOL" "$appdir" "$output"

    log_success "Created: $output"
}

# Clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR" "$OUTPUT_DIR"
    log_success "Cleaned"
}

# Main execution
main() {
    log_info "Building hyprstream AppImage(s)"
    log_info "Version: $VERSION"
    log_info "Variants: ${VARIANTS[*]}"

    if $CLEAN; then
        clean
    fi

    check_dependencies
    ensure_appimagetool

    # Download all required libtorch variants
    if ! $SKIP_DOWNLOAD; then
        for variant in "${VARIANTS[@]}"; do
            download_libtorch "$variant"
        done
    fi

    # Build all variants
    if ! $SKIP_BUILD; then
        for variant in "${VARIANTS[@]}"; do
            build_variant "$variant"
        done
    fi

    # Create AppImages
    if [[ "$BUILD_VARIANT" == "universal" ]] || [[ -z "$BUILD_VARIANT" ]]; then
        # Build universal AppImage
        create_universal_appimage

        # Also build per-backend AppImages if building all
        if [[ -z "$BUILD_VARIANT" ]]; then
            for variant in "${VARIANTS[@]}"; do
                create_per_backend_appimage "$variant"
            done
        fi
    else
        # Build single variant AppImage
        create_per_backend_appimage "$BUILD_VARIANT"
    fi

    log_success "Build complete!"
    log_info "Output files:"
    ls -lh "$OUTPUT_DIR"/*.AppImage 2>/dev/null || true
}

main "$@"
