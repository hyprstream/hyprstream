#!/usr/bin/env bash
# Tiny real RPM fixture tests preparation, not application/container execution.
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
arch="$(uname -m)"
case "$arch" in x86_64) wrong_arch=aarch64;; aarch64) wrong_arch=x86_64;; *) exit 2;; esac
source_sha=1111111111111111111111111111111111111111
mkdir -p "$work"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
cat >"$work/SPECS/fixture.spec" <<EOF
Name: hyprstream-runtime-fixture
Version: 1
Release: 1
Summary: Runtime preparation fixture only
License: MIT
BuildArch: $arch
Provides: bundled(hyprstream-source) = $source_sha
%description
No application payload; validates RPM metadata handling.
%install
mkdir -p %{buildroot}/usr/share/hyprstream-runtime-fixture
echo fixture > %{buildroot}/usr/share/hyprstream-runtime-fixture/value
%files
/usr/share/hyprstream-runtime-fixture/value
EOF
rpmbuild --define "_topdir $work" -bb "$work/SPECS/fixture.spec" >"$work/build.log" 2>&1
rpm_file="$work/RPMS/$arch/hyprstream-runtime-fixture-1-1.$arch.rpm"
rpm_sha="$(sha256sum "$rpm_file" | cut -d' ' -f1)"
prepare="$script_dir/prepare-rpm-runtime.sh"
bash "$prepare" "$rpm_file" "$rpm_sha" "$source_sha" "$arch" "$work/context" >/dev/null
cmp "$rpm_file" "$work/context/hyprstream.rpm"
cmp "$script_dir/../Containerfile.rpm" "$work/context/Containerfile"
test "$(<"$work/context/rpm.sha256")" = "$rpm_sha"
reject() {
  if bash "$prepare" "$@" >/dev/null 2>&1; then echo 'expected preparation rejection' >&2; exit 1; fi
}
reject "$rpm_file" "z${rpm_sha:1}" "$source_sha" "$arch" "$work/bad-hex"
reject "$rpm_file" 0000000000000000000000000000000000000000000000000000000000000000 "$source_sha" "$arch" "$work/wrong-hash"
reject "$rpm_file" "$rpm_sha" 2222222222222222222222222222222222222222 "$arch" "$work/wrong-source"
reject "$rpm_file" "$rpm_sha" "$source_sha" "$wrong_arch" "$work/wrong-arch"
reject "$rpm_file" "$rpm_sha" "$source_sha" "$arch" "$work/context"
test ! -e "$work/wrong-hash"
test ! -e "$work/bad-hex"
test ! -e "$work/wrong-source"
test ! -e "$work/wrong-arch"
echo 'prepare-rpm-runtime: PASS (real RPM metadata/hash/source/arch checks; existing context preserved)'
