#!/usr/bin/env bash
# Prepare a minimal OCI context from a pinned canonical CI RPM artifact.
# The caller records the successful GitLab job/pipeline and verifies its source
# input artifact first; hashing an arbitrary RPM does not authenticate its origin.
# Arguments: RPM file, expected RPM SHA256, source SHA, RPM arch, output dir.
set -Eeuo pipefail
if [ "$#" -ne 5 ]; then
  echo 'usage: prepare-rpm-runtime.sh RPM_FILE RPM_SHA256 SOURCE_SHA RPM_ARCH OUTPUT_DIR' >&2
  exit 2
fi
rpm_file=$1 rpm_sha=$2 source_sha=$3 rpm_arch=$4 output_dir=$5
[[ "$rpm_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || exit 2
case "$rpm_arch" in x86_64|aarch64) ;; *) exit 2;; esac
[ -f "$rpm_file" ] || { echo 'RPM file not found' >&2; exit 2; }
printf '%s  %s\n' "$rpm_sha" "$rpm_file" | sha256sum --check -
test "$(rpm -qp --queryformat '%{ARCH}' "$rpm_file")" = "$rpm_arch"
rpm -qp --provides "$rpm_file" | grep -Fx "bundled(hyprstream-source) = $source_sha"
[ ! -e "$output_dir" ] || { echo 'output directory must not exist' >&2; exit 2; }
mkdir -m 0700 -p "$output_dir"
cp -- "$rpm_file" "$output_dir/hyprstream.rpm"
printf '%s  %s/hyprstream.rpm\n' "$rpm_sha" "$output_dir" | sha256sum --check -
test "$(rpm -qp --queryformat '%{ARCH}' "$output_dir/hyprstream.rpm")" = "$rpm_arch"
rpm -qp --provides "$output_dir/hyprstream.rpm" | grep -Fx "bundled(hyprstream-source) = $source_sha"
printf '%s\n' "$rpm_sha" >"$output_dir/rpm.sha256"
cp "$(dirname "${BASH_SOURCE[0]}")/../Containerfile.rpm" "$output_dir/Containerfile"
