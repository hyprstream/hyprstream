# RPM packaging

Hyprstream's canonical RPM spec, source-provenance metadata, and build pipeline
live in the [FerruleOS RPM repository](https://gitlab.com/cyberdione/ferruleos/rpms/-/tree/main/rpms/hyprstream).
This repository publishes Hyprstream source commits and release tags; it does
not maintain a second RPM spec or RPM release workflow.

The shipped RPM is built and installed against the pinned
Hummingbird/FerruleOS runtime on both `x86_64` and `aarch64`. After the
FerruleOS default-branch pipeline publishes a build, the current package can
be installed with:

```sh
arch="$(uname -m)"
curl -fLO "https://gitlab.com/api/v4/projects/85081375/packages/generic/hyprstream/latest/hyprstream-${arch}.rpm"
sudo dnf install "./hyprstream-${arch}.rpm"
```

The floating `latest` path is intended for FerruleOS consumers. Immutable
version-release paths are also retained in the GitLab Generic Package
Registry; see the [package release and provenance contract](https://gitlab.com/cyberdione/ferruleos/rpms/-/blob/main/rpms/hyprstream/README.md)
before pinning one.

Fedora Rawhide builds compile and install the same source and spec on both
architectures as a forward-compatibility gate. Rawhide artifacts are not
published or supported as FerruleOS binaries because their newer glibc and
OpenSSL ABI requirements may not exist in Hummingbird. The former standalone
Rocky Linux 9 RPM compatibility promise is retired.
