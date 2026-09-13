# RPM-based staging application container

`Containerfile.rpm` installs the canonical FerruleOS CPU RPM into the same pinned
Hummingbird baseline used by the RPM pipeline's build/install gate. It runs
Hyprstream as UID/GID 65532 and keeps `/hyprstream` as an entrypoint alias for
existing Quadlets. The RPM database and private libtorch runtime remain intact.
It does not compile Cargo, migrate the Rocky host, boot systemd in the container,
or require SSH/SSM. The existing multi-backend `Dockerfile` is unchanged.

Request the canonical RPM pipeline on its protected main with
`HYPRSTREAM_SOURCE_REF=<full reviewed source SHA>`. Both architecture jobs consume
the same `hyprstream-source-input` artifact; `latest` is discovery only and must
be resolved before a deployment pin is selected.

Before preparing an image, record the successful canonical GitLab pipeline and
build job IDs, verify their source-input artifact equals the intended source,
and obtain the matching Hummingbird RPM over the authenticated job-artifact API.
Do not substitute a Rawhide compatibility artifact or a legacy mutable `latest`
package. Record its SHA-256. A hash calculated from an arbitrary download alone
does not prove its origin. The helper checks the bytes, architecture and embedded
source provide; the caller supplies the pipeline provenance.

Prepare a new, minimal build context (requires host `rpm`):

```sh
bash scripts/prepare-rpm-runtime.sh \
  /path/to/canonical.rpm "$RPM_SHA256" "$SOURCE_COMMIT" x86_64 /path/to/new-context
podman build --file /path/to/new-context/Containerfile \
  --build-arg SOURCE_COMMIT="$SOURCE_COMMIT" \
  --build-arg RPM_SHA256="$RPM_SHA256" --build-arg RPM_ARCH=x86_64 \
  --tag localhost/hyprstream-rpm-staging /path/to/new-context
podman run --rm --network=none localhost/hyprstream-rpm-staging --version
```

Build natively on the selected architecture; use `aarch64` for ARM. The image
build repeats artifact hash, architecture and embedded source checks, installs
through DNF with ordinary repository verification, and runs the actual installed
binary. It does not weaken GPG settings or label unsigned metadata as signed.

The application revision and RPM hash are OCI labels. After runtime validation,
publish through the authorized image path and pin the resulting OCI digest in
staging, recording source/RPM/pipeline/image provenance together. Never deploy a
mutable image tag or claim a tiny preparation fixture proves application startup.

`bash scripts/test-prepare-rpm-runtime.sh` checks preparation against a tiny real
RPM. The actual RPM container build, unprivileged smoke test and deployment remain
separate required gates.
