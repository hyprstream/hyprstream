# RPM packaging

Hyprstream's canonical RPM spec, source-provenance metadata, and build pipeline
live in the [FerruleOS RPM repository](https://gitlab.com/cyberdione/ferruleos/rpms/-/tree/main/rpms/hyprstream).
This repository publishes Hyprstream source commits and release tags; it does
not maintain a second RPM spec or RPM release workflow.

The shipped RPM is built and installed against the pinned
Hummingbird/FerruleOS runtime on both `x86_64` and `aarch64`. An accepted
default-branch pipeline publishes the actual full RPM filenames in an
immutable, content-addressed generation. Its generation manifest binds the
pinned source commit and archive SHA-512 to each architecture's NEVRA,
immutable URL, and artifact SHA-256.

`dev`, compatibility alias `latest`, and manually promoted `staging` are
explicitly mutable discovery channels. Each channel manifest points to one
immutable generation manifest and records that manifest's SHA-256; a channel
URL is never a provenance pin. Promotion to `staging` verifies and reuses the
exact already-published generation bytes—it does not rebuild the RPMs.

The following example discovers `dev`, then pins and verifies both the
immutable generation manifest and the selected RPM before installation:

```sh
set -eu
project=85081375
channel=dev
arch="$(uname -m)"
registry="https://gitlab.com/api/v4/projects/${project}/packages/generic"

# Discovery only: this response is mutable and may point elsewhere later.
curl --fail --location --output channel.json \
  "${registry}/hyprstream-channels/${channel}/manifest.json"
jq -e --arg channel "$channel" \
  '.schema == "org.ferruleos.hyprstream.rpm-channel.v1" and
   .mutable == true and .channel == $channel' channel.json >/dev/null

manifest_url="$(jq -er '.target.immutable_manifest_url' channel.json)"
manifest_sha256="$(jq -er '.target.manifest_sha256' channel.json)"
curl --fail --location --output generation.json "$manifest_url"
printf '%s  %s\n' "$manifest_sha256" generation.json | sha256sum --check -
jq -e '.schema == "org.ferruleos.hyprstream.rpm-generation.v1"' \
  generation.json >/dev/null

rpm_url="$(jq -er --arg arch "$arch" \
  '.artifacts[] | select(.arch == $arch) | .immutable_url' generation.json)"
rpm_sha256="$(jq -er --arg arch "$arch" \
  '.artifacts[] | select(.arch == $arch) | .sha256' generation.json)"
rpm_file="$(jq -er --arg arch "$arch" \
  '.artifacts[] | select(.arch == $arch) | .filename' generation.json)"
test "$rpm_file" = "$(basename "$rpm_file")"
curl --fail --location --output "$rpm_file" "$rpm_url"
printf '%s  %s\n' "$rpm_sha256" "$rpm_file" | sha256sum --check -
sudo dnf install "./${rpm_file}"
```

Immutable artifact URLs have the shape
`hyprstream/<source-commit>/artifacts/<NEVRA>/<sha256>/<full-rpm-filename>`;
immutable generation manifests additionally use a content-derived generation
ID and manifest SHA-256 in their path. An identical publication retry is a
no-op, while different bytes at the same logical source/NEVRA identity fail
closed even if GitLab allows duplicate generic packages. See the
[package release and provenance contract](https://gitlab.com/cyberdione/ferruleos/rpms/-/blob/main/rpms/hyprstream/README.md)
for the manifest fields and update policy.

`stable` is reserved for future signed release governance. There is not yet a
signed DNF repository generation in this contract, so these direct downloads
must not be described as signed repository metadata or used to justify
weakening `gpgcheck` or `repo_gpgcheck`.

Fedora Rawhide builds compile and install the same source and spec on both
architectures as a forward-compatibility gate. Rawhide artifacts are not
published or supported as FerruleOS binaries because their newer glibc and
OpenSSL ABI requirements may not exist in Hummingbird. The former standalone
Rocky Linux 9 RPM compatibility promise is retired.
