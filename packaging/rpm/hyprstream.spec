# RPM spec file for hyprstream
#
# Build:   packaging/rpm/build-rpm.sh   (wrapper that stages version + tarball)
#          or: rpmbuild -ba packaging/rpm/hyprstream.spec
# CI:      see .github/workflows/rpm.yml (release publish + PR build/install check)
#
# Notes:
#   * Uses the existing Cargo build system (build.rs handles codegen).
#   * libtorch is fetched at build time via the `download-libtorch` feature
#     (PyTorch 2.10.0 CPU). The hyprstream binary links it as a hard DT_NEEDED
#     dependency (libtorch_cpu.so, libc10.so), so the .so files are bundled into
#     a private libdir and the binary's RUNPATH is set to find them. Without this
#     the binary cannot even exec.
#   * Forces openssl-sys to link the SYSTEM openssl-devel. rpmbuild exports
#     CROSS_COMPILE, which makes openssl-sys believe it is cross-compiling and
#     build OpenSSL from source (fails on missing perl-FindBin). We unset it.
#   * GPU stacks (CUDA / ROCm) are declared as weak deps (Recommends:) so a user
#     can override with a locally-installed version, or exclude them entirely with
#     --setopt=install_weak_deps=False. They require external vendor repos.

# Bundled libtorch is prebuilt and has no debug sources; skip debuginfo packaging.
%global debug_package %{nil}

# Private libdir for the bundled libtorch shared objects.
%global torchlibdir %{_libdir}/%{name}

Name:           hyprstream
Version:        0.5.0
Release:        1%{?dist}
Summary:        Agentic cloud infrastructure for continuously learning applications

License:        AGPL-3.0-only AND MIT
URL:            https://github.com/hyprstream/hyprstream
Source0:        %{url}/archive/v%{version}/hyprstream-%{version}.tar.gz

ExclusiveArch:  x86_64 aarch64

# Build dependencies
#
# NOTE: the Rust toolchain (cargo/rustc) is intentionally NOT a BuildRequires.
# It is provided on PATH via rustup (distro rust is often too old, and CI uses
# rustup). rpmbuild checks BuildRequires against the rpm database, which would
# not see a rustup toolchain. build-rpm.sh guards that cargo is on PATH.
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  cmake
BuildRequires:  make
BuildRequires:  pkgconfig
BuildRequires:  openssl-devel
BuildRequires:  systemd-devel
BuildRequires:  capnproto
BuildRequires:  patchelf
# NOTE: perl is deliberately NOT required. openssl-sys would only invoke perl
# for a from-source OpenSSL build, which OPENSSL_NO_VENDOR=1 (below) disables.

# Runtime dependencies
Requires:       git
Requires:       git-lfs
Requires:       ca-certificates

# --- Optional GPU acceleration (weak deps; require external vendor repos) ---
# Pinned to the versions the compiled code targets. Weak deps can be overridden
# with a local install or skipped with --setopt=install_weak_deps=False.
#
# NVIDIA CUDA  (package name encodes the toolkit version)
#   * cuda-toolkit-12-8 : requires NVIDIA driver >= 535
#   * cuda-toolkit-13-0 : requires NVIDIA driver >= 555
Recommends:     (cuda-toolkit-12-8 or cuda-toolkit-13-0)
# AMD ROCm (libtorch is built against ROCm 7.x; gfx1151 needs >= 7.2)
Recommends:     rocm >= 7.2

%description
HyprStream is an agentic cloud infrastructure for applications that learn,
build, and run. It provides continuous development, training, integration,
and deployment of software and AI/ML models.

Features:
- LLM inference and training engine (Qwen3 architecture)
- OpenAI-compatible API
- Git worktrees for model versioning
- Security: CURVE encryption, Ed25519 signatures, Casbin policy engine
- Model Context Protocol (MCP) integration
- Optional GPU acceleration (NVIDIA CUDA, AMD ROCm)

Backend support:
- CPU: bundled (PyTorch/libtorch CPU runtime)
- CUDA: requires the NVIDIA vendor repository
- ROCm: requires the AMD vendor repository

%prep
%autosetup -p1 -n %{name}-%{version}

# git2 dependency requires a git repository to be present
git init -q
git config user.email "rpm-build@localhost.localdomain"
git config user.name "RPM Build"
git add -A
git commit -q -m "RPM build"

%build
# rpmbuild injects RUSTFLAGS with -Cdebuginfo=2 for the whole dependency graph
# (arrow/datafusion/tonic/libtorch ...). We do not ship a -debuginfo subpackage
# (%%debug_package %%{nil}), so that debug info is pure bloat -- it inflated the
# build tree past 11 GB and exhausted disk. Build without debuginfo.
export RUSTFLAGS="-Cdebuginfo=0"

# Force openssl-sys to use the system library instead of a vendored source build.
# Unsetting CROSS_COMPILE is the critical step: rpmbuild sets it, which otherwise
# triggers openssl-sys's from-source path.
unset CROSS_COMPILE
export OPENSSL_NO_VENDOR=1
export OPENSSL_STATIC=0
export OPENSSL_DIR=%{_prefix}
export OPENSSL_LIB_DIR=%{_libdir}
export OPENSSL_INCLUDE_DIR=%{_includedir}

# CPU build. libtorch (PyTorch 2.10.0 CPU) is fetched at build time via the
# download-libtorch feature and extracted under target/.../torch-sys-*/out.
env -u CROSS_COMPILE cargo build --release \
    --features download-libtorch,otel,gittorrent,xet

%install
install -d %{buildroot}%{_bindir}
install -m 0755 target/release/hyprstream %{buildroot}%{_bindir}/hyprstream

# --- Bundle the libtorch runtime the binary links against -------------------
# download-libtorch extracts libtorch under the torch-sys build output dir.
torch_lib_dir="$(find target -type d -path '*/torch-sys-*/out/libtorch/libtorch/lib' | head -1)"
if [ -z "$torch_lib_dir" ]; then
    echo "ERROR: bundled libtorch lib dir not found under target/" >&2
    exit 1
fi
install -d %{buildroot}%{torchlibdir}
# Copy the shared objects (follow into any subdirs the runtime needs).
cp -a "$torch_lib_dir"/*.so* %{buildroot}%{torchlibdir}/

# Point the binary at the bundled libtorch. RUNPATH into a private %{_libdir}/%{name}
# subdir is accepted by check-rpaths (it is not a standard library search path).
patchelf --set-rpath %{torchlibdir} %{buildroot}%{_bindir}/hyprstream

# Documentation
install -d %{buildroot}%{_docdir}/%{name}
install -m 0644 README.md %{buildroot}%{_docdir}/%{name}/README.md

cat > %{buildroot}%{_docdir}/%{name}/GPU-SETUP.md << 'EOF'
# GPU Backend Setup for HyprStream

The CPU backend (libtorch) is bundled and works out of the box. GPU acceleration
is optional and requires an external vendor repository plus a matching driver.
The RPM declares these as weak dependencies so they can be overridden with a
locally-installed version, or skipped entirely:

    sudo dnf install --setopt=install_weak_deps=False hyprstream

## AMD ROCm (gfx1151 requires ROCm >= 7.2)

    sudo dnf config-manager --add-repo https://repo.radeon.com/rocm/rhel9/rocm.repo
    sudo dnf install rocm

## NVIDIA CUDA

    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
    sudo dnf install cuda-toolkit-12-8   # driver >= 535
    # or
    sudo dnf install cuda-toolkit-13-0   # driver >= 555
EOF

%check
# Smoke-check the built binary. The install-time RUNPATH points at the final
# (not-yet-installed) path, so resolve libtorch from the buildroot for this check.
LD_LIBRARY_PATH=%{buildroot}%{torchlibdir} \
    %{buildroot}%{_bindir}/hyprstream --version

%files
%license LICENSE-AGPLV3 LICENSE-MIT
%doc %{_docdir}/%{name}/README.md
%doc %{_docdir}/%{name}/GPU-SETUP.md
%{_bindir}/hyprstream
%dir %{torchlibdir}
%{torchlibdir}/*.so*

%changelog
* Fri Jun 26 2026 hyprstream maintainers <dev@hyprstream.com> - 0.5.0-1
- Initial RPM packaging wired into CI (release publish + Fedora/Rocky build-install check)
- Bundle libtorch CPU runtime and set RUNPATH (binary hard-links libtorch_cpu.so)
- Resolve openssl-sys source-build hang under rpmbuild (unset CROSS_COMPILE)
- GPU stacks declared as version-pinned weak deps (Recommends) for local override
