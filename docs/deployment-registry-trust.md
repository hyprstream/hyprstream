# Deployment registry trust provisioning

Production Discovery/PDS authority is rooted before commands, factories, plugins,
or generated clients run. The executable does not consult
`CREDENTIALS_DIRECTORY`, XDG/user configuration, `HYPRSTREAM__SECRETS__PATH`, a
public setter, or a caller-provided path for this authority.

The OS-owned deployment seam is deliberately small and fail-closed:

- `/etc/hyprstream/trust/deployment-ca.hybrid` is the independently provisioned
  1984-byte deployment CA public key: exactly 32 bytes of Ed25519 followed by
  1952 bytes of ML-DSA-65. A legacy 32-byte Ed25519-only pin is rejected.
- `/run/hyprstream/credentials/registry-service.jwt` is the separately
  provisioned, short-lived `service:registry` credential. Its `cnf.jwk` names the
  registry key that certifies accepted `did:at9p:<cid512>` state.
- `/etc/hyprstream/trust/deployment-authority.log.json` is the public,
  root-anchored authority-set history required for every enrolled credential.
- `/etc/hyprstream/trust/deployment-authority.head.json` is the independently
  provisioned expected log head: schema, deployment domain, log DID, sequence,
  and head CID. The supplied log must match it exactly.

All installed files must be regular, root-owned, and not group/world writable. Missing,
malformed, symlinked, or incorrectly owned material makes production resolver
startup fail closed. Missing log or checkpoint never selects a less restrictive
verifier. Both JWT signature components are verified against the checkpointed
active authority before its key is represented by an opaque verification-only
capability. The raw keys and one-shot witness are not exposed through the
public service or Discovery APIs.

The repository does not yet contain an operator enrollment protocol. Deployments
using this OS-owned source must therefore provision the `/etc` pin through their
OS image, configuration manager, measured-boot policy, or equivalent root-owned
mechanism, and project the registry JWT into the fixed `/run` location before
starting hyprstream. User-mode service-manager installations that cannot provide
these fixed OS-owned files are intentionally not production-authoritative and
fail closed; ambient credential-directory fallback is not supported.

The CA authenticates the registry credential. The registry key in that credential
then certifies the purpose-derived audit key used by accepted-state envelopes and
monotonic checkpoints. The accepted state remains self-addressed as
`did:at9p:<cid512>` and is still subject to the canonical/hash/signature GATE and
checkpoint/currentness validation described in
[`at9p-accepted-state.md`](at9p-accepted-state.md).

## DID-anchored trust source

Deployments may explicitly select the first increment of the DID-anchored
source with two public root configuration values:

```toml
cluster_at9p_did = "did:at9p:<cid512>"
cluster_did_web = "did:web:discovery.hyprstream.com"
```

Both values must be set together. When both are absent, startup uses the
OS-owned source exactly as before. A partial pair is an error, and a failure in
the selected DID source never falls back to the OS-owned source.

The bootstrap fetches the `did:web` document and the at9p genesis capsule over
untrusted HTTPS. A root DID document's capsule is served at
`/.well-known/at9p/<cid512>.cbor`; for a path-form DID it is served in the
corresponding directory beside `did.json`. The capsule must pass the existing
canonical-encoding, BLAKE3-512 hash-to-configured-DID, and hybrid-signature
GATE. The DID document must have the configured `id` and name that exact
`did:at9p` in `alsoKnownAs`; the verified capsule must reciprocally name the
configured `did:web`. Only after both directions verify does startup accept the
at9p identity as authoritative.

The deployment CA and Discovery reach are taken from the GATE-verified capsule,
never from the `did:web` document. The CA is both halves of the capsule's
primary hybrid subject key (`body.subject_keys[0].ed25519Pub` and
`body.subject_keys[0].mldsa65Pub`), and reach is the capsule's `#ns`
`NinePExport` service, dialed by its independent iroh `nodeId` or signed QUIC
socket carrier. Missing or malformed key material in either half fails closed.
The document contributes only the reciprocal identifier vouch; any keys or
services it publishes are advisory and are never installed as trust material.

Capsule content only proves a content-bound reach claim, not that the endpoint
is currently live. Startup therefore dials the capsule-derived Discovery
transport and requires a successful signed `ping` before installing the
process resolver. Fetched bytes, DNS, TLS, relays, and transport endpoints do
not become trust decisions: identity remains pinned by the configured at9p
hash and mutual alias rule, while application responses remain pinned to the
separately authenticated Discovery service key.

The registry deployment credential and current root-anchored authority log are
fetched from beside the capsule at
`/.well-known/deployment/registry-service.jwt` and
`/.well-known/deployment/deployment-authority.log.json`. The log is
authenticated against the capsule-derived hybrid root and must match the
separately provisioned OS-owned head checkpoint exactly; the credential is then
validated against that active set with the same closed profile as the OS-owned
path. The HTTPS channel is untrusted by design and cannot redefine currentness.
The OS-owned JWT and log files are required only when the OS-owned source is
selected, but the OS-owned checkpoint is required for both enrolled sources.

### Serving the deployment well-known documents

The OAuth service terminates the deployment's did:web host when configured
with a static well-known directory (it is the only dual-stack HTTP+RPC
service, and it already owns `/.well-known/did.json`):

```toml
[oauth]
deployment_well_known_dir = "/var/lib/hyprstream/deployment-well-known"
```

The operator provisions four public, integrity-anchored documents, re-read
on every request (so the hourly credential refresh needs no restart):

```text
<dir>/did.json                              the deployment DID document
<dir>/at9p/<cid512>.cbor                    the cluster at9p genesis capsule
<dir>/deployment/deployment-authority.log.json
                                            the current root-anchored authority log
<dir>/deployment/registry-service.jwt       the current CA-signed credential
```

When the directory is configured, `/.well-known/did.json` serves the static
deployment document IN PLACE OF the dynamic node document; a missing file is
a 404, never a silent fall-through.

### Same-node vs remote-node discovery reach

A DID-anchored node that hosts Discovery on its local IPC fabric (the metal
stack's `service start … --ipc` containers; the default) uses a lazy local
discovery client — identical posture to the OS-owned path, which also never
pings. A node REMOTE from the cluster's Discovery sets
`cluster_remote_node = true`: startup then dials the document-advertised
transport and requires a successful signed liveness `ping` before installing
the resolver. The remote arm additionally requires the document's `#mesh-kem`
`keyAgreement` (request confidentiality — QUIC forbids cleartext envelopes)
and exactly one ML-DSA-65 `#mesh-pq` verification method (response
authentication); either absent fails the boot.

Private-PKI deployments whose did:web host terminates TLS with an internal CA
may add `cluster_anchor_root_cert = "/path/to/root.pem"` — an ADDITIVE trust
anchor (public roots remain enabled); an unreadable or malformed file fails
startup.

### Fresh nodes

A first boot has no checkpointed PDS store. The bootstrap initializes an
empty (genesis) store rather than deadlock — the registry service is the
sole writer and would otherwise be unstartable. If a node that previously
held at9p state boots with a missing store, that means the duplicity history
was lost; startup logs a loud warning and proceeds at genesis posture.

## Registry credential profile

`registry-service.jwt` is a closed, one-hour-maximum deployment credential, not
a generic JWT or access token. Let `D` be the composite `kid` of the exact
ML-DSA-65 + Ed25519 public-key pair selected as the deployment CA (from
`deployment-ca.hybrid` or the GATE-verified capsule). It is the RFC 7638
thumbprint of the AKP representation whose `alg` is `ML-DSA-65-Ed25519` and
whose public bytes are `ML-DSA-65 || Ed25519`. Provisioning must use the
following profile exactly:

- The protected header contains only `alg`, `typ`, and `kid`, with values
  `ML-DSA-65-Ed25519`, `wit+jwt`, and `D`, respectively.
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
- The compact JWT signature segment decodes to exactly 3373 bytes:
  `ML-DSA-65 signature (3309) || Ed25519 signature (64)`. Both components sign
  the exact `base64url(protected) + "." + base64url(claims)` input and both
  must verify against the pinned deployment CA. A missing, stripped, malformed,
  or invalid component is rejected; there is no classical fallback.

All JSON objects are parsed with duplicate-member rejection. Unknown members,
optional JOSE/JWK metadata (`crit`, `use`, `key_ops`, or a JWK-local `alg` or
`kid`), audience arrays, padded/noncanonical base64url, alternate algorithms or
token types, and a signature or key identifier that does not bind the pinned CA
fail closed before a registry witness can be minted. The credential file is the
compact JWT itself with no surrounding whitespace or trailing newline.

The common operational form adds exactly one `delegation` claim to that closed
claim set. It is canonical unpadded base64url JSON containing the scoped UCAN,
the delegated 1,984-byte hybrid public key, and the stable authority-log DID.
The separately installed or fetched root-anchored DidOp log must have that DID,
its verified sequence and head CID must equal the independent checkpoint, and
the UCAN issuer must still be in its active any-of-N authority set. The UCAN is
restricted to this deployment's registry-service mint ability, exact profile
and audience, and the one-hour ceiling. Direct rare-root credentials retain the
original claim set without `delegation`, but they are subject to the same
mandatory log/checkpoint test and fail after root retirement.

This profile is deployment/registry scoped. It has no PDS or host identifier.
The demo therefore projects one shared deployment CA and current credential to
both PDS hosts; a per-PDS binding would require a new profile or separate roots.
The exact mint, rotation, recovery, publisher, and Metal input contracts are in
[`deployment-trust-contract.md`](deployment-trust-contract.md).

Classical-only atproto peer keys remain supported only at the separately named
peer/federation record-resolution surface. That P-256 interoperability path
does not construct the hybrid deployment root, cannot authenticate a deployment
credential, and is never consulted by either deployment-root provider.
