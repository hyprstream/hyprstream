# RPM spec file for hyprstream
#
# Build:   rpmbuild -ba packaging/rpm/hyprstream.spec
# CI:      see .github/workflows/rpm.yml (release publish + PR build/install check)
#
# Notes:
#   * Uses the existing Cargo build system (build.rs handles codegen).
#   * Forces openssl-sys to link the SYSTEM openssl-devel. rpmbuild exports
#     CROSS_COMPILE, which makes openssl-sys believe it is cross-compiling and
#     build OpenSSL from source (fails on missing perl-FindBin). We unset it.
#   * GPU stacks (CUDA / ROCm) are declared as weak deps (Recommends:) so a user
#     can override with a locally-installed version, or exclude them entirely with
#     --setopt=install_weak_deps=False. They require external vendor repos.

Name:           hyprstream
Version:        0.5.0
Release:        1%{?dist}
Summary:        Agentic cloud infrastructure for continuously learning applications

License:        AGPL-3.0-only AND MIT
URL:            https://github.com/hyprstream/hyprstream
Source0:        %{url}/archive/v%{version}/hyprstream-%{version}.tar.gz

ExclusiveArch:  x86_64 aarch64

# Build dependencies
BuildRequires:  cargo
BuildRequires:  rust
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  cmake
BuildRequires:  make
BuildRequires:  pkgconfig
BuildRequires:  openssl-devel
BuildRequires:  systemd-devel
BuildRequires:  capnproto
# openssl-sys may still probe for perl; provide it so a source fallback cannot hang the build
BuildRequires:  perl-interpreter
BuildRequires:  perl-FindBin

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
Recommends:     rocm >= 7.1

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
- CPU: all supported architectures
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
# Force openssl-sys to use the system library instead of a vendored source build.
# Unsetting CROSS_COMPILE is the critical step: rpmbuild sets it, which otherwise
# triggers openssl-sys's from-source path.
unset CROSS_COMPILE
export OPENSSL_NO_VENDOR=1
export OPENSSL_STATIC=0
export OPENSSL_DIR=%{_prefix}
export OPENSSL_LIB_DIR=%{_libdir}
export OPENSSL_INCLUDE_DIR=%{_includedir}

# CPU build. libtorch is fetched at build time via the download-libtorch feature.
env -u CROSS_COMPILE cargo build --release \
    --features download-libtorch,otel,gittorrent,xet

%install
install -d %{buildroot}%{_bindir}
install -m 0755 target/release/hyprstream %{buildroot}%{_bindir}/hyprstream

# Documentation
install -d %{buildroot}%{_docdir}/%{name}
install -m 0644 README.md %{buildroot}%{_docdir}/%{name}/README.md

cat > %{buildroot}%{_docdir}/%{name}/GPU-SETUP.md << 'EOF'
# GPU Backend Setup for HyprStream

GPU acceleration is optional and requires an external vendor repository plus a
matching driver. The RPM declares these as weak dependencies so they can be
overridden with a locally-installed version, or skipped entirely:

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
# Smoke-check the built binary
target/release/hyprstream --version

%files
%license LICENSE-AGPLV3 LICENSE-MIT
%doc %{_docdir}/%{name}/README.md
%doc %{_docdir}/%{name}/GPU-SETUP.md
%{_bindir}/hyprstream

%changelog
* Thu Jun 26 2026 hyprstream maintainers <dev@hyprstream.com> - 0.5.0-1
- Initial RPM packaging wired into CI (release publish + Fedora/Rocky build-install check)
- Resolve openssl-sys source-build hang under rpmbuild (unset CROSS_COMPILE)
- GPU stacks declared as version-pinned weak deps (Recommends) for local override
