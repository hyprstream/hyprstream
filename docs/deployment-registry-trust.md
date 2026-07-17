# Deployment registry trust provisioning

Production Discovery/PDS authority is rooted before commands, factories, plugins,
or generated clients run. The executable does not consult
`CREDENTIALS_DIRECTORY`, XDG/user configuration, `HYPRSTREAM__SECRETS__PATH`, a
public setter, or a caller-provided path for this authority.

The current deployment seam is deliberately small and fail-closed:

- `/etc/hyprstream/trust/deployment-ca.ed25519` is the independently provisioned
  32-byte Ed25519 deployment CA public key.
- `/run/hyprstream/credentials/registry-service.jwt` is the separately
  provisioned, short-lived `service:registry` credential. Its `cnf.jwk` names the
  registry key that certifies accepted `did:at9p:<cid512>` state.

Both files must be regular, root-owned, and not group/world writable. Missing,
malformed, symlinked, or incorrectly owned material makes production resolver
startup fail closed. The JWT is verified against the `/etc` pin before its key is
represented by an opaque verification-only capability. The raw key and one-shot
witness are not exposed through the public service or Discovery APIs.

The repository does not yet contain an operator enrollment protocol that can
derive this deployment-specific CA from a universally embedded cid512 capsule.
Deployments must therefore provision the `/etc` pin through their OS image,
configuration manager, measured-boot policy, or equivalent root-owned mechanism,
and project the registry JWT into the fixed `/run` location before starting
hyprstream. User-mode service-manager installations that cannot provide these
fixed OS-owned files are intentionally not production-authoritative and fail
closed; ambient credential-directory fallback is not supported.

The CA authenticates the registry credential. The registry key in that credential
then certifies the purpose-derived audit key used by accepted-state envelopes and
monotonic checkpoints. The accepted state remains self-addressed as
`did:at9p:<cid512>` and is still subject to the canonical/hash/signature GATE and
checkpoint/currentness validation described in
[`at9p-accepted-state.md`](at9p-accepted-state.md).
