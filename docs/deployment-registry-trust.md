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

**The deployment CA and discovery reach are taken from the GATE-verified
capsule, never from the `did:web` document (option C, issue #1157).** The
deployment CA is the capsule's complete set of atomic hybrid subject-key pairs
(`body.subject_keys`); the credential's authenticated `kid` selects one live
pair. The reach is the capsule's `#ns` `NinePExport` service entry: iroh is
dialed by its independent `nodeId` carrier, and QUIC is dialed by its signed
`quic://IP:port` carrier. The `did:web` document contributes
**only** the reciprocal identifier vouch (its `alsoKnownAs` must name the
configured `did:at9p`); any `verificationMethod` keys or `service` reach it
publishes are advisory discovery/rotation hints and are never installed as trust
material. This is what ratified #905 §8 requires — *"everything authoritative is
created at the GATE (capsule content) ... addresses select; verification
asserts."* Consequences:

- An adversary who controls the `did:web` origin cannot substitute the CA or the
  reach: both are content-bound to the hash-pinned, self-certifying capsule that
  is byte-identical on every host. This is the property the multi-host bootstrap
  (#1135) depends on.
- The capsule may publish overlapping (old + new) CA pairs. Both are usable
  while present; once a refreshed trusted key set omits the old pair, a
  credential bearing its `kid` is rejected. Document edits never change this
  set.
- The installed authority is cryptographically Hybrid, not merely labelled
  `PqHybrid`: DID-anchored credentials require a nested COSE Ed25519 +
  ML-DSA-65 proof under the exact selected capsule pair.

Capsule content only proves a content-bound reach claim, not that the endpoint
is currently live. Startup therefore dials the capsule-derived Discovery
transport and requires a successful signed `ping` before installing the
process resolver. Fetched bytes, DNS, TLS, relays, and transport endpoints do
not become trust decisions: identity remains pinned by the configured at9p
hash and mutual alias rule, the CA and reach are pinned to the GATE-verified
capsule, and application responses remain pinned to the separately authenticated
Discovery service key. The capsule `nodeId` is an untrusted transport carrier
(#1031), never an identity or assurance source.

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
`deployment-ca.ed25519` or one GATE-verified capsule subject-key pair).
Provisioning must use the following profile exactly:

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

For the OS-owned-file trust source, the file is this compact JWT. For the
DID-anchored source, the file is `compact-jwt~base64url(cose-composite)`, where
the detached nested COSE composite signs the exact `protected.payload` bytes
under the selected pair's Ed25519 and ML-DSA-65 keys with the fixed
`hyprstream.registry-deployment.v1/hybrid` external AAD. Both layers are
required; an EdDSA-only JWT, a stripped outer layer, an unanchored PQ key, or a
`kid` outside the current capsule key set fails closed.

All JSON objects are parsed with duplicate-member rejection. Unknown members,
optional JOSE/JWK metadata (`crit`, `use`, `key_ops`, or a JWK-local `alg` or
`kid`), audience arrays, padded/noncanonical base64url, alternate algorithms or
token types, and a signature or key identifier that does not bind the pinned CA
fail closed before a registry witness can be minted. The credential file is the
compact JWT itself with no surrounding whitespace or trailing newline.
