# RPM packaging

Builds an RPM of `hyprstream` using the existing Cargo build system.

## Local build

```bash
# Build deps (Fedora): rpm-build rpmdevtools gcc gcc-c++ cmake make pkgconfig \
#   openssl-devel systemd-devel capnproto perl-interpreter perl-FindBin git git-lfs
packaging/rpm/build-rpm.sh            # version from crate (or v-tag)
packaging/rpm/build-rpm.sh 0.5.0      # explicit version
```

The script stages the spec with the resolved version, builds a `git archive`
source tarball, runs `rpmbuild -ba`, and prints `RPM=<path>` on the last line.

## CI

`.github/workflows/rpm.yml`:

- **PR check** — builds the RPM and `dnf install`s it inside **Fedora 43** and
  **Rocky 9** containers, verifies `hyprstream --version`, and asserts the GPU
  stacks are *not* hard requirements (installed with `install_weak_deps=False`).
- **Release** — on `v*` tags, attaches the built RPMs + checksums to the GitHub
  Release.

## GPU dependencies

CUDA and ROCm are **weak deps** (`Recommends:`), pinned to the versions the
compiled code targets, so a user can override them with a locally-installed
version or skip them entirely:

```bash
sudo dnf install --setopt=install_weak_deps=False hyprstream
```

- NVIDIA: `cuda-toolkit-12-8` (driver ≥ 535) or `cuda-toolkit-13-0` (driver ≥ 555)
- AMD: `rocm >= 7.2` (gfx1151 support)

Note: this RPM bundles the **CPU** libtorch runtime (PyTorch 2.10.0). The weak
deps prepare a host for GPU use and let an operator pin/override the vendor
stack; a GPU-accelerated libtorch build is a separate variant (see the ROCm
AppImage) and not produced by this spec yet.

## Notes

- `openssl-sys` would otherwise build OpenSSL from source under `rpmbuild`
  (rpmbuild exports `CROSS_COMPILE`); the spec unsets it so the system
  `openssl-devel` is linked.
- systemd units, shell completions, and a config example are intentionally not
  packaged yet — they are not present on `main`. Add them to `%install`/`%files`
  when they land.
