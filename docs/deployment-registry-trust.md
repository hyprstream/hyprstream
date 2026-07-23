# Deployment registry trust provisioning

Production Discovery/PDS authority is rooted before commands, factories, plugins,
or generated clients run. The executable does not consult
`CREDENTIALS_DIRECTORY`, XDG/user configuration, `HYPRSTREAM__SECRETS__PATH`, a
public setter, or a caller-provided path for this authority.

The OS-owned deployment seam is deliberately small and fail-closed:

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
never from the `did:web` document. The CA is the capsule's primary hybrid
subject key's Ed25519 half (`body.subject_keys[0]`), and reach is the capsule's
`#ns` `NinePExport` service, dialed by its independent iroh `nodeId` or signed
QUIC socket carrier. The document contributes only the reciprocal identifier
vouch; any keys or services it publishes are advisory and are never installed
as trust material. Consequently:

- Control of the `did:web` origin cannot substitute the CA or reach because
  both remain content-bound to the hash-pinned capsule.
- Overlapping document keys during rotation do not break bootstrap; accepted
  keys rotate through the at9p state chain rather than document edits.
- The installed authority carries `PqHybrid` assurance from the capsule's
  hybrid subject key, and startup fails closed if it lands below that floor.

Capsule content only proves a content-bound reach claim, not that the endpoint
is currently live. Startup therefore dials the capsule-derived Discovery
transport and requires a successful signed `ping` before installing the
process resolver. Fetched bytes, DNS, TLS, relays, and transport endpoints do
not become trust decisions: identity remains pinned by the configured at9p
hash and mutual alias rule, while application responses remain pinned to the
separately authenticated Discovery service key.

This first increment changes server trust and reach only. The existing
OS-owned `registry-service.jwt` is still required to bind the deployment CA to
the registry verification key. Enrollment (including operator-approved device
flow and unattended machine attestation) and final `SecretsProfile` integration
remain separate work; no client-authentication authority is inferred from the
two public anchors.

## Registry credential profile

`registry-service.jwt` is a closed, one-hour-maximum deployment credential, not
a generic JWT or access token. Let `D` be the RFC 7638 Ed25519 JWK thumbprint of
the exact public key selected as the deployment CA (from
`deployment-ca.ed25519` or the GATE-verified capsule's primary subject key).
Provisioning
must use the following profile exactly:

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
