#!/usr/bin/env bash
# Build the hyprstream RPM from a source checkout.
#
# Usage: packaging/rpm/build-rpm.sh [VERSION]
#   VERSION defaults to the hyprstream crate version (or the v-tag if set via $GITHUB_REF_NAME).
#
# Produces RPM(s) under ~/rpmbuild/RPMS/<arch>/ and prints the primary RPM path
# to stdout on the last line (RPM=/path/to.rpm).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPEC="$SCRIPT_DIR/hyprstream.spec"

# The Rust toolchain is provided on PATH (rustup), not as an rpm BuildRequires.
# Fail early with a clear message if it is missing.
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found on PATH. Install the Rust toolchain (e.g. rustup) first." >&2
    exit 1
fi

# --- Resolve version ---------------------------------------------------------
VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    if [[ "${GITHUB_REF_NAME:-}" == v* ]]; then
        VERSION="${GITHUB_REF_NAME#v}"
    else
        VERSION="$(grep -m1 '^version' "$REPO_ROOT/crates/hyprstream/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')"
    fi
fi
echo ">> Building hyprstream RPM version $VERSION"

NAME="hyprstream"
TOPDIR="${HOME}/rpmbuild"
mkdir -p "$TOPDIR"/{BUILD,BUILDROOT,RPMS,SRPMS,SOURCES,SPECS}

# --- Stage spec with the resolved version ------------------------------------
STAGED_SPEC="$TOPDIR/SPECS/${NAME}.spec"
sed "s/^Version:.*/Version:        ${VERSION}/" "$SPEC" > "$STAGED_SPEC"

# --- Build the source tarball (top dir must be name-version) -----------------
TARBALL="$TOPDIR/SOURCES/${NAME}-${VERSION}.tar.gz"
echo ">> Creating source tarball $TARBALL"
git -C "$REPO_ROOT" archive --format=tar.gz \
    --prefix="${NAME}-${VERSION}/" \
    -o "$TARBALL" HEAD

# --- Build -------------------------------------------------------------------
echo ">> rpmbuild -ba"
rpmbuild -ba "$STAGED_SPEC"

# --- Locate the primary binary RPM -------------------------------------------
ARCH="$(uname -m)"
RPM_PATH="$(ls -1 "$TOPDIR/RPMS/$ARCH/${NAME}-${VERSION}"-*.rpm 2>/dev/null | grep -v -- '-debug' | head -1 || true)"
if [[ -z "$RPM_PATH" ]]; then
    echo "ERROR: built RPM not found under $TOPDIR/RPMS/$ARCH/" >&2
    exit 1
fi
echo ">> Built: $RPM_PATH"
echo "RPM=$RPM_PATH"
