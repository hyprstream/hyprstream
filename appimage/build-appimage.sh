#!/bin/bash
# build-appimage.sh - Build hyprstream AppImages
#
# Usage:
#   ./build-appimage.sh <command> [variant] [options]
#
# Commands:
#   build [VARIANT]      Build and package AppImage (default: all variants + universal)
#   package-universal    Package universal AppImage from staged backend outputs
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
#   ./build-appimage.sh package-universal        # Package from prior stage commands
#   ./build-appimage.sh clean                    # Clean everything
#   ./build-appimage.sh clean cuda128            # Clean only CUDA 12.8 libtorch
#
# Environment:
#   LIBTORCH_CACHE_DIR   Directory to cache libtorch downloads (default: ./libtorch-cache)
#   CARGO_TARGET_DIR     Honored: the built binary is copied from Cargo's
#                        resolved target directory (see lib.sh cargo_target_dir),
#                        never a hard-coded target/release.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Shared packaging helpers (target resolution, ELF arch verification,
# checksums, phase timing). Sourced from the script's own directory.
source "$SCRIPT_DIR/lib.sh"

# Configuration
VERSION="${VERSION:-dev}"
LIBTORCH_VERSION="2.11.0"
LIBTORCH_CACHE_DIR="${LIBTORCH_CACHE_DIR:-$SCRIPT_DIR/libtorch-cache}"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"

# AppImage tooling and the upstream GPU libtorch archives are architecture
# specific. aarch64 currently has a CPU libtorch only (from the PyTorch wheel).
case "${HYPRSTREAM_APPIMAGE_ARCH:-$(uname -m)}" in
    x86_64|amd64) APPIMAGE_ARCH=x86_64 ;;
    aarch64|arm64) APPIMAGE_ARCH=aarch64 ;;
    *) echo "[ERROR] Unsupported AppImage architecture: $(uname -m)" >&2; exit 1 ;;
esac

# GPU AppImages are currently x86_64-only because upstream does not publish
# aarch64 CUDA/ROCm libtorch archives. The arm64 CI job supplies the CPU
# libtorch directory from the official PyTorch aarch64 wheel.
if [[ "$APPIMAGE_ARCH" == "aarch64" ]]; then
    ALL_VARIANTS=(cpu)
else
    ALL_VARIANTS=(cpu cuda128 cuda130 rocm71)
fi

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
    if [[ "$APPIMAGE_ARCH" == "aarch64" && "$variant" != "cpu" ]]; then
        log_error "AppImage variant $variant is unavailable on aarch64 (CPU only)"
        exit 1
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
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${APPIMAGE_ARCH}.AppImage"
    chmod +x "$APPIMAGETOOL"
}

# Download libtorch for a variant
download_libtorch() {
    local variant="$1"
    # The aarch64 PyTorch wheel is already installed by the arm64 builder image.
    # It supplies both headers and shared libraries at HYPRSTREAM_LIBTORCH_DIR.
    if [[ -n "${HYPRSTREAM_LIBTORCH_DIR:-}" ]]; then
        require_libtorch_version "$HYPRSTREAM_LIBTORCH_DIR" "$LIBTORCH_VERSION"
        return
    fi
    local url="${LIBTORCH_URLS[$variant]}"
    local cache_file="$LIBTORCH_CACHE_DIR/libtorch-${LIBTORCH_VERSION}-${variant}.zip"
    local extract_dir="$LIBTORCH_CACHE_DIR/$LIBTORCH_VERSION-$variant"
    mkdir -p "$LIBTORCH_CACHE_DIR"

    if [[ -d "$extract_dir" ]]; then
        [[ -f "$extract_dir/.complete" ]] || {
            log_error "Incomplete libtorch cache: $extract_dir"
            return 1
        }
        require_libtorch_version "$extract_dir/libtorch" "$LIBTORCH_VERSION"
        return
    fi
    # Publish only a fully extracted, version-checked tree. Failure leaves no
    # apparently reusable directory. Old variant-only caches remain untouched.
    (
        local scratch
        scratch=$(mktemp -d "$LIBTORCH_CACHE_DIR/.libtorch-$LIBTORCH_VERSION-$variant.XXXXXX")
        trap 'rm -rf "$scratch"' EXIT
        if [[ ! -f "$cache_file" ]]; then
            log_info "Downloading libtorch for $variant..."
            curl -fSL -o "$scratch/archive.zip" "$url" || exit 1
            mv "$scratch/archive.zip" "$cache_file"
        fi
        unzip -q "$cache_file" -d "$scratch/tree" || exit 1
        require_libtorch_version "$scratch/tree/libtorch" "$LIBTORCH_VERSION" || exit 1
        touch "$scratch/tree/.complete"
        # -T refuses to nest the tree if another invocation populated the cache.
        mv -T "$scratch/tree" "$extract_dir"
    )
}

# Build hyprstream binary for a variant
build_binary() {
    local variant="$1"
    local libtorch_dir
    libtorch_dir="$(libtorch_variant_dir "$variant")"
    require_libtorch_version "$libtorch_dir" "$LIBTORCH_VERSION" || return 1

    log_info "Building hyprstream for $variant..."

    mkdir -p "$BUILD_DIR/bin"

    export LIBTORCH="$libtorch_dir"
    export LD_LIBRARY_PATH="$libtorch_dir/lib:${LD_LIBRARY_PATH:-}"
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    export OPENSSL_NO_VENDOR=1

    (cd "$PROJECT_ROOT" && cargo build --release --features otel)

    # Copy from Cargo's RESOLVED target directory, never a hard-coded
    # target/release: BuildQ and CI cache mounts set CARGO_TARGET_DIR, and a
    # default-path copy would fail or package a stale binary. Fail closed on
    # a missing or foreign-architecture output before it reaches an AppImage.
    local target_dir binary
    target_dir="$(cargo_target_dir)" || {
        log_error "Could not resolve Cargo's target directory"
        exit 1
    }
    binary="$target_dir/release/hyprstream"
    require_elf_arch "$binary" "$APPIMAGE_ARCH" || {
        log_error "Built binary at $binary is missing or not an ${APPIMAGE_ARCH} ELF"
        exit 1
    }
    cp "$binary" "$BUILD_DIR/bin/hyprstream-$variant"
    log_success "Built hyprstream-$variant (from $binary)"
}

# Strip host symbol tables from bundled libtorch shared objects.
# libtorch ships its .so files unstripped; --strip-unneeded removes the symbol
# table while preserving the dynamic symbols needed for loading/relocation, so
# runtime behavior (including dlopen of backend libs) is unaffected. GPU device
# fatbinaries are not touched. Best-effort: any file that can't be stripped is
# left as-is. NOTE: validate a stripped GPU build actually runs inference before
# cutting a release.
strip_libtorch_libs() {
    local lib_dir="$1"
    command -v strip &>/dev/null || { log_info "strip not found; skipping"; return 0; }
    local before after
    before=$(du -sm "$lib_dir" 2>/dev/null | cut -f1)
    find "$lib_dir" -type f \( -name '*.so' -o -name '*.so.*' \) -print0 \
        | xargs -0 -r -n1 strip --strip-unneeded 2>/dev/null || true
    after=$(du -sm "$lib_dir" 2>/dev/null | cut -f1)
    log_info "Stripped libtorch libs in $lib_dir: ${before}MB -> ${after}MB"
}

# Create per-backend AppImage
create_appimage() {
    local variant="$1"
    local appdir="$BUILD_DIR/hyprstream-$variant.AppDir"
    local output="$OUTPUT_DIR/hyprstream-${VERSION}-${variant}-${APPIMAGE_ARCH}.AppImage"
    local libtorch_dir
    libtorch_dir="$(libtorch_variant_dir "$variant")"
    require_libtorch_version "$libtorch_dir" "$LIBTORCH_VERSION" || return 1

    log_info "Creating AppImage for $variant..."

    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/libtorch/lib"

    cp "$BUILD_DIR/bin/hyprstream-$variant" "$appdir/usr/bin/hyprstream"
    # Copy entire lib directory (includes subdirs with Tensile libraries for ROCm)
    cp -r "$libtorch_dir/lib/"* "$appdir/usr/lib/libtorch/lib/"
    strip_libtorch_libs "$appdir/usr/lib/libtorch/lib"

    sed "s/HYPRSTREAM_VARIANT:-cpu/HYPRSTREAM_VARIANT:-$variant/" \
        "$SCRIPT_DIR/AppRun-single" > "$appdir/AppRun"
    chmod +x "$appdir/AppRun"

    cp "$SCRIPT_DIR/hyprstream.desktop" "$appdir/"
    cp "$SCRIPT_DIR/hyprstream.svg" "$appdir/"

    mkdir -p "$OUTPUT_DIR"
    ARCH="$APPIMAGE_ARCH" "$APPIMAGETOOL" "$appdir" "$output"
    log_success "Created: $output"
}

# Create universal AppImage with all backends
create_universal_appimage() {
    local staged_only="${1:-0}"
    local appdir="$BUILD_DIR/hyprstream-universal.AppDir"
    local output="$OUTPUT_DIR/hyprstream-${VERSION}-${APPIMAGE_ARCH}.AppImage"
    local staging="$BUILD_DIR/universal-staging/$LIBTORCH_VERSION"

    log_info "Creating universal AppImage..."

    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib"

    for variant in "${ALL_VARIANTS[@]}"; do
        # Use staged files if available (from stage command), otherwise use build dirs
        if [[ -f "$staging/bin/hyprstream-$variant" ]]; then
            require_libtorch_version "$staging/lib/$variant/libtorch" "$LIBTORCH_VERSION" || return 1
            cp "$staging/bin/hyprstream-$variant" "$appdir/usr/bin/"
            mkdir -p "$appdir/usr/lib/$variant/libtorch/lib"
            cp -r "$staging/lib/$variant/libtorch/lib/"* "$appdir/usr/lib/$variant/libtorch/lib/"
        elif [[ "$staged_only" == "1" ]]; then
            log_error "Staged backend output is missing for $variant"
            return 1
        else
            cp "$BUILD_DIR/bin/hyprstream-$variant" "$appdir/usr/bin/"
            mkdir -p "$appdir/usr/lib/$variant/libtorch/lib"
            local libtorch_dir
            libtorch_dir="$(libtorch_variant_dir "$variant")"
            require_libtorch_version "$libtorch_dir" "$LIBTORCH_VERSION" || return 1
            cp -r "$libtorch_dir/lib/"* "$appdir/usr/lib/$variant/libtorch/lib/"
        fi
    done

    strip_libtorch_libs "$appdir/usr/lib"

    cp "$SCRIPT_DIR/AppRun" "$appdir/"
    chmod +x "$appdir/AppRun"
    cp "$SCRIPT_DIR/hyprstream.desktop" "$appdir/"
    cp "$SCRIPT_DIR/hyprstream.svg" "$appdir/"

    mkdir -p "$OUTPUT_DIR"
    ARCH="$APPIMAGE_ARCH" "$APPIMAGETOOL" "$appdir" "$output"
    log_success "Created: $output"
}

# Package a universal AppImage without compiling or downloading anything. The
# per-backend workflow steps call stage before clean, so this command consumes
# only outputs from this run and fails closed when one is absent.
cmd_package_universal() {
    validate_variant universal
    local phase_start
    phase_start="$(ci_phase_begin)"
    log_info "Packaging universal AppImage from staged backend outputs"
    ensure_appimagetool
    create_universal_appimage 1
    log_success "Universal package complete"
    ls -lh "$OUTPUT_DIR/hyprstream-${VERSION}-${APPIMAGE_ARCH}.AppImage"
    write_artifact_checksums "$OUTPUT_DIR"
    ci_phase_end "package-universal" "$phase_start"
}

# Command: build
cmd_build() {
    local variant="${1:-all}"
    validate_variant "$variant"

    local phase_start
    phase_start="$(ci_phase_begin)"

    log_info "Building hyprstream AppImage"
    log_info "Version: $VERSION"
    log_info "Variant: $variant"

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
    write_artifact_checksums "$OUTPUT_DIR"
    ci_phase_end "build $variant" "$phase_start"
}

# Command: stage - copy files needed for universal AppImage before cleaning
cmd_stage() {
    local variant="${1:-}"
    if [[ -z "$variant" ]] || [[ "$variant" == "all" ]] || [[ "$variant" == "universal" ]]; then
        log_error "stage requires a specific variant (cpu, cuda128, cuda130, rocm71)"
        exit 1
    fi
    validate_variant "$variant"

    local staging="$BUILD_DIR/universal-staging/$LIBTORCH_VERSION"
    log_info "Staging $variant for universal AppImage..."

    # Stage binary
    mkdir -p "$staging/bin"
    cp "$BUILD_DIR/bin/hyprstream-$variant" "$staging/bin/"

    # Stage entire lib directory (includes subdirs with Tensile libraries for ROCm)
    mkdir -p "$staging/lib/$variant/libtorch/lib"
    local libtorch_dir version_header
    libtorch_dir="$(libtorch_variant_dir "$variant")"
    require_libtorch_version "$libtorch_dir" "$LIBTORCH_VERSION" || return 1
    cp -r "$libtorch_dir/lib/"* "$staging/lib/$variant/libtorch/lib/"
    version_header=$(libtorch_version_header "$libtorch_dir")
    mkdir -p "$staging/lib/$variant/libtorch/$(dirname "$version_header")"
    cp "$libtorch_dir/$version_header" "$staging/lib/$variant/libtorch/$version_header"

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
        rm -rf "$LIBTORCH_CACHE_DIR/$LIBTORCH_VERSION-$variant"
        rm -f "$LIBTORCH_CACHE_DIR/libtorch-${LIBTORCH_VERSION}-${variant}.zip"
        rm -f "$BUILD_DIR/bin/hyprstream-$variant"
        rm -rf "$BUILD_DIR/hyprstream-$variant.AppDir"
        log_success "Cleaned $variant"
    fi
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
        package-universal)
            cmd_package_universal
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
