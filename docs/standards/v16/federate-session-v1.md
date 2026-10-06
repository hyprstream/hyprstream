# Federate session v1 — draft v16 extension

Status: **DRAFT, pending exact-delta security review**. This document profiles
an opt-in, disabled-by-default interactive Federate session. It does not amend
the frozen generic v16 proof-v1 claim set or authorize runtime enablement.
Implementations MUST select this profile explicitly; unknown or partially
implemented profile claims deny, with no Bearer, DPoP, static-enrollment, or
generic token-exchange fallback.

## Outcome and boundaries

The website first completes an exact-client OIDC authorization-code + PKCE
login. A separate, narrow RFC 8693 host exchange verifies that ID-token
profile and a one-use challenge signed by the browser's dedicated Ed25519 and
ML-DSA-65 request-proof keys. The host maps the immutable upstream identity
to a local account, signed tenant, and current server-side grants; the ID
token is never itself an RPC credential. This profile defines the *resulting*
host access token and per-RPC proof, not a second browser login.

The host returns an OAuth access-token response with
`issued_token_type=urn:ietf:params:oauth:token-type:access_token` and the
registered `token_type=PoP`, not `Bearer`. The access token is a short-lived,
issuer-signed RFC 9068 `typ=at+jwt` JWT for one exact host audience. Bare
token presentation never authorizes Registry or Model dispatch. A receiver
requires a v16 authenticated `COSE_Sign` request proof from the same admitted
browser key suite and current Policy admission. RFC 9449 DPoP remains the
separate JOSE/HTTP profile; `cnf.jkt` is not accepted here and is not an
alternate COSE encoding. Use of the registered `PoP` token type does not by
itself claim ACE-OAuth conformance; this document supplies the particular
Hyprstream client-to-resource-server proof rules.

## JWT confirmation and hybrid-suite binding

The frozen generic v16 JWT profile identifies a primary signer suite with
`cnf.hs_signer_suite`. That member is a hash of **two** keys for the hybrid
suite, while RFC 7800 requires `cnf` to represent only one PoP key. This
Federate-only amendment instead requires:

```json
{
  "hs_profile": "federate-session-v1",
  "cnf": {"jwk": {"kty": "OKP", "crv": "Ed25519", "x": "<canonical Ed public key>"}},
  "hs_signer_suite_v1": "<canonical base64url SHA-256 suite thumbprint>"
}
```

`cnf.jwk` is the one RFC 7800 confirmation key, exactly the Ed25519 public
component. `hs_signer_suite_v1` is a signed, top-level, profile-specific
commitment to the **whole** ordered hybrid suite. Its 32 bytes are the
existing v16 signer-suite thumbprint:

```text
SHA-256(RFC 8949 deterministic-CBOR [
  "hs-cose-sign-ed25519-mldsa65-wns-v1",
  [raw Ed25519 public (32 bytes), raw ML-DSA-65 public (1952 bytes)]
])
```

The verifier obtains the exact second public component and its authority
state from the admitted session-primary record, recomputes the thumbprint,
matches the Ed component to `cnf.jwk`, and requires both component signatures
in the v16 primary group. The top-level claim is **not** a second OAuth
confirmation method and grants no authority without the record and proof.
No second `cnf` key, `cnf.hs_signer_suite`, `cnf.jkt`, or claim-placement
fallback is accepted in this profile. Generic v16 credentials retain their
existing wire format until a separately reviewed migration; this extension
cannot reinterpret them as Federate sessions.

All other applicable v16 JWT requirements remain: exact trusted `iss` and
`aud`, `client_id`, nonempty `sub`, `tenant`, `scope`, `clearance`, `iat`, `exp`,
unique `jti`, interactive `sid`, issuer signature, revocation and current
session status. The host additionally binds the admitted `(issuer, sub)`,
local account, tenant, client, resource, generation, and short expiry. Values
supplied by the browser are not authoritative for those mappings.

## Request proof and forwarded recipients

The request carries the exact credential bytes in the authorization slot and
their SHA-256 in signed proof claim `-70001`. The existing v16 signed claims
bind exact service audience, schema ID, encoded request body, request ID,
freshness and `response_binding` (`-70004`); the hybrid primary group uses
Ed25519 **and** ML-DSA-65. Replay admission is durable and occurs before
handler effects. Current account, tenant and grant authority is checked at
each protected request and each distinct tool call; an already-authorized
stream may finish without per-chunk Policy polling.

Federate requests additionally require signed CWT claim `-70009`, a closed
deterministic-CBOR map with exactly these integer keys:

| Key | Type | Meaning |
|---|---|---|
| 1 | `bstr .size 32` / `null` | SHA-256 of the exact canonical `RecipientPublic::encode()` for the forwarded response KEM recipient, or explicit absence. |
| 2 | `bstr .size 32` / `null` | The same commitment for the forwarded stream KEM recipient, or explicit absence. |
| 3 | `bstr .size 32` / `null` | Exact legacy forwarded `clientDhPublic`, or explicit absence. |

The verifier compares all three values to the recipients actually forwarded
in the authenticated envelope **before** dispatch. A missing key, duplicate,
wrong type/length, changed recipient, or omitted `-70009` denies. The generic
v16 parser continues to reject `-70009` as an unknown claim; response proofs
do not carry it. This extension supplements rather than replaces `-70004`,
the carrier's encryption policy, or response-proof comparison. It prevents a
blind rendezvous relay or envelope signer from substituting a response/stream
recipient while forwarding the original browser's request proof. It does not
prevent a relay from dropping or delaying traffic, nor protect a browser that
has lost both private keys.

The browser SDK opts in per Registry or Model client with
`RpcClient.withFederateProofSigner(...)`. It takes callbacks for dedicated
Ed25519 and ML-DSA-65 proof keys, checks their signatures before sending, and
derives the request schema ID from the selected service. Proof construction
occurs after the RPC client creates its response recipient. Claim `-70003`
contains the application Cap'n Proto bytes that dispatch recovers from the
browser carrier transcript; claim `-70009` commits the actual finalized
response, stream, and legacy DH fields. The current network response uses a
hybrid envelope KEM, so this sender encodes `-70004` as `null` rather than
inventing a different ML-KEM-only response-proof recipient. Registry and
Model admission must enforce the same forwarded recipient comparison before
this sender is enabled in the staging website.

## Required vectors and activation gate

The source-controlled native signed-proof checksum vector in
`proof::build::tests::federate_signed_proof_vector_is_stable_and_profile_isolated`
uses only fixed, non-secret test inputs: Ed25519 signing seed `09`×32,
ML-DSA-65 signing seed `0a`×32, kids `proof-fixture-ed` and
`proof-fixture-pq`. Both hybrid-KEM recipients use the pinned
`HyKemX25519MlKem768` suite: response seeds (`07`×32, `09`×64), stream seeds
(`08`×32, `0a`×64). Legacy client-DH public is `5a`×32,
request ID `3c`×16, credential bytes `fixture.sender.constrained.jwt`,
service `registry.svc.hyprstream.test`, schema ID `0xd4d0f2a1b3c58e67`,
body `federate-proof-vector-v1`, `iat=1799999995`, and `exp=1800000030`.
This vector encodes `response_binding` (`-70004`) as CBOR `null` and carries
no `Nonce` claim; it is a proof checksum fixture, not a complete forwarded
envelope.
The expected SHA-256 of the complete untagged `COSE_Sign` bytes is
`186b33dad6d4b41ed750b0d05e408dcaa78e708c33abac7bf4c2778bcf6826d6`.
The test also verifies both signatures and proves the frozen generic parser
rejects those same bytes. This is a **native checksum vector**, not yet the
required native/browser byte-identical vector or complete negative suite.

Before enabling this profile, publish byte-identical native/browser positive
vectors for the JWT claim layout and a signed v16 proof with `-70009` and an
actual forwarded recipient set. Negative vectors MUST cover: missing or
moved `hs_signer_suite_v1`; `cnf.hs_signer_suite`/`cnf.jkt` fallback; wrong Ed
anchor or PQ component; one missing signature; absent, duplicate, malformed,
or altered `-70009` field; wrong `-70004`, body, schema, audience, request ID,
or credential hash; replay and revoked session. Run the frozen generic v16
gate unchanged to prove the extension is confined to the Federate decoder.
The website response schema, issuer, browser validator, and Registry/Model
verifier MUST change atomically behind the disabled feature gate.

Standards anchors: [RFC 7800 §3.1](https://www.rfc-editor.org/rfc/rfc7800#section-3.1)
for the single-key `cnf` rule; [RFC 8693 §2.2.1](https://www.rfc-editor.org/rfc/rfc8693#section-2.2.1)
for exchange response fields; [RFC 9200 §5.8.4.2](https://www.rfc-editor.org/rfc/rfc9200#section-5.8.4.2)
for the registered `PoP` token type and its profile-specific proof method;
[RFC 9052](https://www.rfc-editor.org/rfc/rfc9052) for COSE signing.
