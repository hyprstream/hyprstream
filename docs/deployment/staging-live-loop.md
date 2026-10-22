# Build and update on the staging development host

Run `bash scripts/staging-live-loop.sh probe` as the staging runtime user
`rocky` from a checkout on the guest. It reports architecture, source state,
storage and service status without changing a service.

`bash scripts/staging-live-loop.sh build` builds the existing production
Dockerfile's CPU runtime image on that guest. Rust and native dependencies run
inside its builder stages; no Rust installation on the host is required.
The Dockerfile reads the checked-in `rust-toolchain.toml` in both builder roots.
The default is two Cargo jobs; `HYPRSTREAM_DEV_BUILD_JOBS=1..4` may select fewer
or more within that bound. The Cargo target is on persistent guest storage at
`/home/rocky/.cache/hyprstream-dev-build/target`, outside the container's writable
layer. Set `HYPRSTREAM_DEV_CACHE_DIR` to an absolute, user-owned directory on a
scratch disk when necessary. The cache and container storage filesystems must
support the guest's container access: the script privately relabels only this
guest-owned cache with Podman's `:Z` option for SELinux. This is separate from
the shared BuildQ pool on hypr1 and must never point at it. Both filesystems must
each have at least 30 GiB free before building, or 60 GiB total when they share
one filesystem. This is a minimum check; measure
the first build's peak space/RAM and cold/warm elapsed time before selecting a
default VM size. This script has not yet been qualified on a real guest.

Build prints a local image reference. Activate it for the service being
repaired, then run the relevant application smoke test:

```sh
bash scripts/staging-live-loop.sh activate policy localhost/hyprstream-staging-dev:REFERENCE
bash scripts/staging-live-loop.sh rollback policy
```

Activation stages a Quadlet change, disables registry auto-update for that
local image, restarts the service and checks systemd readiness. Repeated
activations preserve the original deployed configuration.
For `event`, the helper selects the existing `hyprstream-event.container`
and its matching `hyprstream-event.service`, or the legacy `hyprstream.container`
and `hyprstream.service`. It refuses to activate or roll back Event when both
filenames exist or neither supported filename exists.
After a normal deployment replaces the local override, a new activation
checkpoints that newer release; rollback refuses to overwrite an intervening
deployment with a stale checkpoint. Failed activation
restores that configuration; if recovery also fails, the saved rollback file
remains available. Activation/rollback share an exclusive lock. The script
does not alter trust, signing keys, database schemas or grants. An incompatible
change involving several services still needs its explicit coordinated update
and application test; one active service does not establish system acceptance.

After a useful repair, commit/review/test its source, publish the normal CI
image digest, deploy through the protected pipeline, and repeat acceptance on
a VM launched from that release. Record the source revision and dirty state,
image reference, affected service, test and timing during the diagnostic loop.
The script is for the remote guest; builds and Git hooks on hypr1 continue to
use the fleet's shared BuildQ instead of creating a private Cargo target here.
