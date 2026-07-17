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

## Registry credential profile

`registry-service.jwt` is a closed, one-hour-maximum deployment credential, not
a generic JWT or access token. Let `D` be the RFC 7638 Ed25519 JWK thumbprint of
the exact public key installed as `deployment-ca.ed25519`. Provisioning must use
the following profile exactly:

- The protected header contains only `alg`, `typ`, and `kid`, with values
  `EdDSA`, `wit+jwt`, and `D`, respectively.
- The claims object contains only `iss`, `sub`, `aud`, `exp`, `nbf`, `iat`,
  `deployment_domain`, `profile`, and `cnf`. `iss` is
  `urn:hyprstream:deployment:D`; `sub` is `service:registry`; `aud` is
  `urn:hyprstream:service:registry`; `deployment_domain` is `D`; and `profile`
  is `hyprstream.registry-deployment.v1`.
- `exp`, `nbf`, and `iat` are nonnegative integer NumericDate values. The
  credential must be currently valid, with strict `exp > now` and `nbf <= iat <
  exp`. The future-clock-skew endpoint is inclusive (`nbf` and `iat` may equal
  `now + 60`), as is the lifetime endpoint (`exp - iat` may equal 3600). All
  additions and subtractions are checked; overflow or underflow fails closed.
- `cnf` contains only one `jwk`; no `jkt` or alternate confirmation member is
  permitted. The JWK contains only `kty: "OKP"`, `crv: "Ed25519"`, and `x`.
  `x` is canonical unpadded base64url for exactly 32 bytes and is the registry
  public key installed as the process's verification-only PDS authority.

All JSON objects are parsed with duplicate-member rejection. Unknown members,
optional JOSE/JWK metadata (`crit`, `use`, `key_ops`, or a JWK-local `alg` or
`kid`), audience arrays, padded/noncanonical base64url, alternate algorithms or
token types, and a signature or key identifier that does not bind the pinned CA
fail closed before a registry witness can be minted. The credential file is the
compact JWT itself with no surrounding whitespace or trailing newline.
