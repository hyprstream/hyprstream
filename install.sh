#!/bin/bash
# install.sh - One-liner installer for Hyprstream AppImage
#
# Usage:
#   curl -fsSL https://install.hyprstream.dev | bash
#   curl -fsSL https://install.hyprstream.dev | bash -s -- --auto
#   VERSION=0.3.0 curl -fsSL https://install.hyprstream.dev | bash
#
# Options:
#   --auto           Skip confirmation prompts (non-interactive)
#   --version X.Y.Z  Install a specific version
#   --help           Show this help message
#
# Environment:
#   HYPRSTREAM_AUTO_INSTALL  Set to 1 to skip confirmation prompts
#   VERSION                  Override version (skip GitHub API lookup)
#   GITHUB_TOKEN             GitHub API token (optional, avoids rate limits)

set -euo pipefail

# Colors (matches build-appimage.sh)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

GITHUB_REPO="hyprstream/hyprstream"
LICENSE_URL="https://github.com/hyprstream/hyprstream/blob/main/LICENSE-MIT"
AGPL_LICENSE_URL="https://github.com/hyprstream/hyprstream/blob/main/LICENSE-AGPLV3"
TMPDIR_PATH=""
AUTO_INSTALL="${HYPRSTREAM_AUTO_INSTALL:-0}"

show_help() {
    sed -n '2,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --auto)
                AUTO_INSTALL=1
                shift
                ;;
            --version)
                if [ -z "${2:-}" ]; then
                    log_error "--version requires a value"
                    exit 1
                fi
                VERSION="$2"
                shift 2
                ;;
            --version=*)
                VERSION="${1#--version=}"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Prompt user for y/N confirmation. Returns 0 on yes, 1 on no.
# Reads from /dev/tty so it works when piped via curl|bash.
confirm() {
    local prompt="$1"
    local default="${2:-N}"

    if [ "$AUTO_INSTALL" = "1" ]; then
        return 0
    fi

    local yn
    if [ "$default" = "Y" ]; then
        printf "%s [Y/n] " "$prompt"
    else
        printf "%s [y/N] " "$prompt"
    fi

    read -r yn </dev/tty || yn=""

    case "$yn" in
        [Yy]|[Yy][Ee][Ss]) return 0 ;;
        [Nn]|[Nn][Oo]) return 1 ;;
        "") [ "$default" = "Y" ] && return 0 || return 1 ;;
        *) return 1 ;;
    esac
}

check_arch() {
    local arch
    arch="$(uname -m)"
    if [ "$arch" != "x86_64" ]; then
        log_error "Unsupported architecture: $arch"
        log_error "Hyprstream AppImages are currently only available for x86_64."
        exit 1
    fi
    log_info "Architecture: $arch"
}

check_prerequisites() {
    local missing=()

    if ! command -v git &>/dev/null; then
        missing+=("git")
    fi

    if ! command -v git-lfs &>/dev/null; then
        missing+=("git-lfs")
    fi

    if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
        missing+=("curl or wget")
    fi

    # Check for FUSE (required by AppImage)
    if [ ! -e /dev/fuse ] && ! command -v fusermount &>/dev/null && ! command -v fusermount3 &>/dev/null; then
        missing+=("FUSE")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing[*]}"
        echo ""
        echo "Install them with your package manager:"
        echo ""
        echo "  Debian/Ubuntu:  sudo apt-get install git git-lfs fuse libfuse2 curl"
        echo "  Fedora:         sudo dnf install git git-lfs fuse fuse-libs curl"
        echo "  Arch Linux:     sudo pacman -S git git-lfs fuse2 curl"
        echo ""
        exit 1
    fi

    log_success "All prerequisites found"
}

setup_tmpdir() {
    TMPDIR_PATH="$(mktemp -d)"
    cleanup() {
        if [ -n "$TMPDIR_PATH" ] && [ -d "$TMPDIR_PATH" ]; then
            rm -rf "$TMPDIR_PATH"
        fi
    }
    trap cleanup EXIT INT TERM
    log_info "Working directory: $TMPDIR_PATH"
}

# Download a URL to a file, using curl or wget
download() {
    local url="$1"
    local output="$2"
    local auth_header=""

    if [ -n "${GITHUB_TOKEN:-}" ]; then
        auth_header="Authorization: token $GITHUB_TOKEN"
    fi

    if command -v curl &>/dev/null; then
        if [ -n "$auth_header" ]; then
            curl -fsSL -H "$auth_header" -o "$output" "$url"
        else
            curl -fsSL -o "$output" "$url"
        fi
    elif command -v wget &>/dev/null; then
        if [ -n "$auth_header" ]; then
            wget -q --header="$auth_header" -O "$output" "$url"
        else
            wget -q -O "$output" "$url"
        fi
    else
        log_error "Neither curl nor wget found"
        exit 1
    fi
}

# Fetch a URL to stdout
fetch() {
    local url="$1"
    local auth_header=""

    if [ -n "${GITHUB_TOKEN:-}" ]; then
        auth_header="Authorization: token $GITHUB_TOKEN"
    fi

    if command -v curl &>/dev/null; then
        if [ -n "$auth_header" ]; then
            curl -fsSL -H "$auth_header" "$url"
        else
            curl -fsSL "$url"
        fi
    elif command -v wget &>/dev/null; then
        if [ -n "$auth_header" ]; then
            wget -q --header="$auth_header" -O - "$url"
        else
            wget -q -O - "$url"
        fi
    fi
}

resolve_version() {
    if [ -n "${VERSION:-}" ]; then
        # Strip leading 'v' if present (e.g. v0.3.0 -> 0.3.0)
        VERSION="${VERSION#v}"
        log_info "Using VERSION override: $VERSION"
        return
    fi

    log_info "Fetching latest release from GitHub..."
    local api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"
    local response
    response="$(fetch "$api_url")" || {
        log_error "Failed to fetch latest release from GitHub API."
        log_error "Set VERSION=x.y.z to skip the API call, or set GITHUB_TOKEN for higher rate limits."
        exit 1
    }

    # Parse tag_name from JSON without jq
    VERSION="$(echo "$response" | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\?\([^"]*\)".*/\1/')"

    if [ -z "$VERSION" ]; then
        log_error "Could not parse version from GitHub API response."
        log_error "Set VERSION=x.y.z to install a specific version."
        exit 1
    fi

    log_success "Latest version: $VERSION"
}

download_appimage() {
    local base_url="https://github.com/${GITHUB_REPO}/releases/download/v${VERSION}"
    local appimage_name="hyprstream-${VERSION}-x86_64.AppImage"
    local appimage_url="${base_url}/${appimage_name}"
    local checksums_url="${base_url}/SHA256SUMS.txt"

    APPIMAGE_PATH="${TMPDIR_PATH}/${appimage_name}"
    CHECKSUMS_PATH="${TMPDIR_PATH}/SHA256SUMS.txt"

    log_info "Downloading ${appimage_name}..."
    download "$appimage_url" "$APPIMAGE_PATH" || {
        log_error "Failed to download AppImage from: $appimage_url"
        exit 1
    }
    log_success "Downloaded AppImage ($(du -h "$APPIMAGE_PATH" | cut -f1))"

    log_info "Downloading checksums..."
    download "$checksums_url" "$CHECKSUMS_PATH" || {
        log_warn "SHA256SUMS.txt not found — skipping checksum verification"
        CHECKSUMS_PATH=""
        return
    }
    log_success "Downloaded SHA256SUMS.txt"
}

verify_sha256() {
    if [ -z "${CHECKSUMS_PATH:-}" ]; then
        log_warn "Skipping checksum verification (no checksums file)"
        return
    fi

    if ! command -v sha256sum &>/dev/null; then
        log_warn "sha256sum not found — skipping checksum verification"
        return
    fi

    local appimage_name
    appimage_name="$(basename "$APPIMAGE_PATH")"

    local expected
    expected="$(grep "$appimage_name" "$CHECKSUMS_PATH" | awk '{print $1}')"

    if [ -z "$expected" ]; then
        log_warn "No checksum found for $appimage_name in SHA256SUMS.txt — skipping verification"
        return
    fi

    local actual
    actual="$(sha256sum "$APPIMAGE_PATH" | awk '{print $1}')"

    if [ "$expected" != "$actual" ]; then
        log_error "Checksum verification FAILED!"
        log_error "  Expected: $expected"
        log_error "  Actual:   $actual"
        log_error "The downloaded file may be corrupted or tampered with."
        exit 1
    fi

    log_success "Checksum verified"
}

run_service_install() {
    chmod +x "$APPIMAGE_PATH"

    log_info "Running service install..."
    "$APPIMAGE_PATH" service install || {
        log_error "'service install' failed."
        log_error "If you see a FUSE error, ensure FUSE is installed and /dev/fuse is accessible."
        exit 1
    }

    log_success "Hyprstream installed successfully"
}

print_post_install() {
    echo ""
    echo -e "${GREEN}=== Hyprstream v${VERSION} installed ===${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Add hyprstream to your PATH (if not already):"
    echo "       export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "     Or start a new terminal session."
    echo ""
    echo "  2. Apply the default access policy:"
    echo "       hyprstream quick policy apply-template local"
    echo ""
    echo "  3. Start services:"
    echo "       hyprstream service start"
    echo ""
    echo "  4. Check status:"
    echo "       hyprstream service status"
    echo ""
    echo "  5. (CUDA only) Enable GPU support:"
    echo "       systemctl --user set-environment LD_PRELOAD=libtorch_cuda.so"
    echo "       systemctl --user restart hyprstream-model"
    echo ""
    echo "  6. Clone your first model:"
    echo "       hyprstream quick clone https://huggingface.co/Qwen/Qwen3-0.6B"
    echo ""
    echo "Documentation: https://github.com/${GITHUB_REPO}"
    echo ""
}

main() {
    parse_args "$@"

    echo -e "${BLUE}Hyprstream Installer${NC}"
    echo ""

    check_arch
    check_prerequisites
    setup_tmpdir
    resolve_version

    echo ""
    log_info "Hyprstream is dual-licensed under MIT and AGPL-3.0."
    log_info "  MIT:      ${LICENSE_URL}"
    log_info "  AGPL-3.0: ${AGPL_LICENSE_URL}"
    echo ""
    if ! confirm "Do you accept the license terms?"; then
        log_error "License not accepted. Aborting."
        exit 1
    fi
    echo ""

    download_appimage
    verify_sha256

    echo ""
    log_info "Ready to install Hyprstream v${VERSION}."
    log_info "This will:"
    echo "    - Copy the AppImage to ~/.local/hyprstream/"
    echo "    - Create a symlink at ~/.local/bin/hyprstream"
    echo "    - Install systemd user units"
    echo ""
    if ! confirm "Proceed with installation?"; then
        log_error "Installation cancelled."
        exit 1
    fi
    echo ""

    run_service_install
    print_post_install
}

main "$@"
