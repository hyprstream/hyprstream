#!/usr/bin/env bash
# lib.sh - shared helpers for hyprstream packaging builds (source, not execute).
#
# Consumed by build-appimage.sh and reusable by CI jobs so the target-path,
# architecture-verification, checksum, and phase-timing rules live in exactly
# one place (native CI plan N1.3). Keep every helper fail-closed: a packaging
# step that cannot prove what it packaged must fail, not warn.

# Resolve the Cargo target directory exactly as Cargo itself does:
#   1. $CARGO_TARGET_DIR when set (BuildQ locally, CI cache mounts),
#   2. else `cargo metadata`'s target_directory for the workspace.
# Never assume <workspace>/target: a hard-coded lookup copies a stale or
# missing binary whenever the real target directory lives elsewhere.
cargo_target_dir() {
    if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
        printf '%s\n' "$CARGO_TARGET_DIR"
        return 0
    fi
    command -v cargo >/dev/null 2>&1 || {
        echo "cargo_target_dir: cargo not found and CARGO_TARGET_DIR is unset" >&2
        return 1
    }
    (cd "$PROJECT_ROOT" && cargo metadata --format-version 1 --no-deps) \
        | grep -o '"target_directory":"[^"]*"' \
        | head -n1 \
        | cut -d'"' -f4
}

# ELF e_machine description Cargo's output must carry for the packaging arch.
elf_machine_for_arch() {
    case "$1" in
        x86_64)  printf 'Advanced Micro Devices X86-64' ;;
        aarch64) printf 'AArch64' ;;
        *) return 1 ;;
    esac
}

# Fail closed unless $1 is a non-empty ELF file whose e_machine matches the
# requested arch. Cross-compiled or foreign-architecture binaries are exactly
# what universal packaging must refuse to bundle (plan N3.4).
require_elf_arch() {
    local bin="$1" want_arch="$2" want machine
    [[ -f "$bin" && -s "$bin" ]] || {
        echo "require_elf_arch: missing or empty file: $bin" >&2
        return 1
    }
    want="$(elf_machine_for_arch "$want_arch")" || {
        echo "require_elf_arch: unsupported arch: $want_arch" >&2
        return 1
    }
    if command -v readelf >/dev/null 2>&1; then
        machine="$(
            readelf -h "$bin" \
                | awk -F: '/Machine:/ { gsub(/^[ \t]+/, "", $2); print $2; exit }'
        )"
    elif command -v file >/dev/null 2>&1; then
        machine="$(file -b "$bin")"
        case "$machine" in
            *x86-64*)  machine='Advanced Micro Devices X86-64' ;;
            *aarch64*) machine='AArch64' ;;
        esac
    else
        echo "require_elf_arch: need readelf or file to verify: $bin" >&2
        return 1
    fi
    [[ "$machine" == "$want" ]] || {
        echo "require_elf_arch: $bin reports ELF machine '$machine'," \
             "expected '$want' ($want_arch)" >&2
        return 1
    }
}

# Regenerate SHA256SUMS.txt over every AppImage in $1 and print it, so each
# build log carries per-artifact checksums for the manifest/artifact-identity
# contract. Fails when the directory holds no AppImage to attest.
write_artifact_checksums() {
    local out_dir="$1"
    [[ -d "$out_dir" ]] || {
        echo "write_artifact_checksums: no such directory: $out_dir" >&2
        return 1
    }
    local -a images=()
    while IFS= read -r -d '' f; do
        images+=( "$f" )
    done < <(find "$out_dir" -maxdepth 1 -type f -name '*.AppImage' -print0 | sort -z)
    (( ${#images[@]} > 0 )) || {
        echo "write_artifact_checksums: no AppImages under $out_dir" >&2
        return 1
    }
    ( cd "$out_dir" && sha256sum ./*.AppImage > SHA256SUMS.txt )
    echo "[INFO] Artifact checksums ($out_dir/SHA256SUMS.txt):"
    cat "$out_dir/SHA256SUMS.txt"
}

# Phase timing: ci_phase_begin <label> records the epoch; ci_phase_end
# <label> <start_epoch> prints the elapsed wall time. Cheap, uniform
# measurement so AppImage runs become comparable across runner classes
# (plan Wave 4).
ci_phase_begin() {
    date +%s
}

ci_phase_end() {
    local label="$1" start="$2"
    local elapsed=$(( $(date +%s) - start ))
    printf '[INFO] phase %s finished in %dm%02ds\n' \
        "$label" $(( elapsed / 60 )) $(( elapsed % 60 ))
}
