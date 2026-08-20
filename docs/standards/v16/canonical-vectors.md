# Canonical vectors — `hs-rpc-proof-v1`

Status: **FROZEN** by the accepted Gate-2 vote (v16 §19, 2026-08-19). These are
the canonical positive and negative test vectors for the v16 COSE RPC
request-proof profile. They are the executable half of the profile freeze: the
structure lives in [`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl), the
label allocation in [`private-label-registry.md`](private-label-registry.md),
and every rule either has a vector here or is not frozen.

Machine-readable files (the normative form — this page is the human index):

| File | Contents |
|---|---|
| [`vectors/proof-v1-keys.json`](vectors/proof-v1-keys.json) | Test keys, seeds, and fixture values |
| [`vectors/proof-v1-positive.json`](vectors/proof-v1-positive.json) | 9 vectors that MUST verify |
| [`vectors/proof-v1-negative.json`](vectors/proof-v1-negative.json) | 58 vectors that MUST deny |
| [`vectors/proof-v1-thumbprints.json`](vectors/proof-v1-thumbprints.json) | Cross-implementation replay-namespace thumbprint vectors (C1) |
| [`vectors/proof-v1-credentials.json`](vectors/proof-v1-credentials.json) | Frozen `verifier_now` clock (F1) and the issuer-signed at+jwt tokens the authenticated positives hash (F2) |

Each JSON vector carries `id`, `title`, `expect` (`accept` / `deny`),
`structure`, `size_bytes`, `sha256`, full `cbor_hex`, and — for negatives — a
`deny_class` and the exact `deny_rule` violated.

## Reproducing and checking

```sh
python3 docs/standards/v16/tools/gen_proof_vectors.py   # regenerates, byte-identical
python3 docs/standards/v16/tools/check_proof_vectors.py # verifies the checked-in files
python3 docs/standards/v16/tools/validate_profile.py    # the full CDDL+registry+fixtures gate
```

Every key is derived from a published constant seed, so regeneration is
byte-identical: Ed25519 signatures are deterministic by RFC 8032, and the
ML-DSA-65 keys and signatures are produced with seeded key generation
(`openssl genpkey -pkeyopt hexseed:`) and deterministic signing
(`-pkeyopt deterministic:1`). The keys in
[`vectors/proof-v1-keys.json`](vectors/proof-v1-keys.json) are **test keys**
and MUST NOT be enrolled or trusted.

`check_proof_vectors.py` independently re-derives each positive vector's
`Sig_structure`, verifies every component signature (Ed25519 in-process,
ML-DSA-65 via OpenSSL), decodes with a strict decoder that rejects indefinite
lengths, non-minimal integers, unsorted or duplicate map keys, tags, and
floating-point values, and asserts that every signature entry matches exactly
one component of the signed plan. It is not a profile verifier — it checks the
artifacts, and a verifier under test is expected to enforce considerably more.

The CDDL is now mechanically validated: `tools/validate_profile.py` compiles it
with a real CDDL validator (`pycddl`, the Rust `cddl` crate; pinned in
[`tools/requirements.txt`](tools/requirements.txt)) and validates every positive
fixture — its **protected bucket** and its **claims payload** directly against
the paired rules — alongside the exact private values, caps, closed response
map, orthogonal enum axes, recipient/encryption relation, collision review, and
a byte-identical regeneration. It also proves the negatives deny by their rule:

- the **typ × hs_domain** cross-product (the N-4 domain-confusion shape) is
  rejected by the paired protected rules for both `COSE_Sign1` and the
  `COSE_Sign` body — a request `typ` cannot pair with a response domain, or vice
  versa;
- a **response proof** carrying a `Nonce` or a non-null `credential_hash` is
  rejected by the distinct `hyprstream-response-proof-claims` rule;
- an **unattributed proof** without the server challenge (N-16) is rejected by
  the distinct `hyprstream-unattributed-proof-claims` rule, which makes `Nonce`
  a mandatory key rather than relying on a separate verifier check;
- a **cleartext unary `response_binding`** carried as a map (N-27) is rejected —
  the cleartext map alternative is `stream_setup` only, so cleartext unary has
  the single canonical null encoding;
- a **hybrid→classical downgrade** (N-28: the hybrid suite with only its Ed25519
  component) and an **in-range unknown suite_id** (N-12) are rejected at the
  `signature-plan` level — each suite is bound to its exact ordered algorithms
  and component count, and the checker additionally requires a `COSE_Sign1` plan
  to have exactly one component. N-12 is deliberately **in-range (≤64 bytes)** so
  its **sole** denial is registry closure (swapping in a known suite validates the
  same plan); the separate >64-byte `suite_id` size rule is proven by **N-52** (O1);
- an **`aud` violating the service-domain syntax** (N-29 uppercase, N-30 illegal
  first byte) is rejected — the gate ports `validate_service_domain` and applies
  it to every fixture, so the profile's audience namespace is no broader than the
  transport's;
- a **response proof whose `response_binding` mismatches its originating
  request** (N-32) is rejected — the gate compares P-7's (bound) and N-32's
  (mismatch) bindings against P-4's request binding field-for-field, not merely
  by local map shape;
- a **bound response proof whose `-70002` ≠ the realized binding's
  `root_type_id`** (N-31) is rejected — the gate and vector checker enforce that
  cross-field schema-ID equality (which CDDL cannot express) over every bound
  response fixture;
- a **`signature_plan` repeating one `(alg, kid)` across two groups** (N-33) is
  rejected — the gate and checker enforce whole-plan `(alg, kid)` uniqueness
  regardless of group ID, so a single signer cannot satisfy a two-group policy; and
  because labels are attacker-selected, a **resolved public-key identity**
  (`(alg, raw public key)`, never `kid`/`group_id`) may appear in **at most one**
  signer group after key resolution, so one key re-published under two kids (N-53)
  cannot satisfy a two-group policy either;
- an **unattributed proof is verified against its embedded key set** (not any
  out-of-band table): the gate and checker require exact ordered 1:1
  correspondence with the plan (keyed by `(alg, kid)`) and verify each signature
  against the embedded key, so a mismatched embedded key (N-35), surplus (N-18),
  duplicate (N-37), reordered (N-38), or over-the-1..2-ceiling (N-42) key set is
  rejected;
- an **authenticated request proof must carry the 32-byte `credential_hash`** —
  never null (null is the unattributed case): the CDDL pins the hash in the
  authenticated request claims rule and N-15 (authenticated, null hash, no key
  set) is CDDL-rejected;
- **signer-group IDs are unique and strictly ascending** across the plan: the
  gate and checker reject a duplicate `group_id` (N-40) or an out-of-order plan
  (N-41), so two logical groups cannot collapse to one;
- **every signature entry cites a group in the signed plan** (N-20) and a
  **response proof's `cti` echoes its originating request's `request_id`** (N-22,
  bound to P-4) — both asserted mechanically against the de-fanged mutations;
- the **replay-namespace thumbprints are frozen to bytes** (CDDL §7.1, two
  distinct domain separators): the gate and checker recompute the authenticated
  and unattributed thumbprints from the frozen derivation and assert they match
  the cross-implementation vectors in `proof-v1-thumbprints.json`;
- **documented vector counts must match the manifests** — the gate asserts
  README.md and canonical-vectors.md both read the actual positive/negative
  counts;
- the **size-cap** negatives N-52 (65-byte `suite_id`), N-13 (65-byte `kid`), and
  N-26 (129-byte `aud`) are each asserted over their numeric caps, and the CDDL
  cap text (`kid` = `1..64`, `aud` = `1..128` plus its `.regexp`) is pinned so a
  widening drifts the gate to red; and
- an **object over the 2 MiB total cap** is rejected by a validator-side numeric
  check (the complete-object CDDL cannot bound the signature byte strings), which
  the gate exercises by constructing a structurally-valid, fixed-size oversized
  object — the pinned `pycddl` version cannot enforce `.size` byte-length ranges
  or `.regexp`, so these numeric/ported checks are load-bearing.

See [`README.md`](README.md) for the one remaining tooling note.

## Verifier clock and authenticated credential context (F1/F2)

These fixtures are time-dependent, so the disposition of every `accept`/`deny`
vector is evaluated at one **frozen instant**, `verifier_now = 1786000015`
(`iat = 1786000000 <= verifier_now < exp = 1786000030`, strict at `exp`). A
conformance runner MUST inject this integer as its verifier clock instead of
reading wall-clock time; every artifact declares the same `verifier_now`, and the
gate rejects a missing, disagreeing, pre-`iat`, at-`exp`, or wall-clock instant.

The authenticated positives **P-2 / P-4 / P-5 / P-6** hash a real, deterministic,
profile-valid **at+jwt** access token — not a placeholder. Each token is a compact
JWS with the exact protected header `{"typ":"at+jwt","alg":"EdDSA","kid":"issuer-ed25519-1"}`,
signed by the seeded credential-issuer Ed25519 key over the required v16
authenticated-dispatch claims (`iss`, `sub`, service `aud`, `iat`/`exp`, unique
`jti`, the RFC 9068 `client_id` (`hyprstream-oauth-client-1`), Reusable-only
`tenant` and `clearance`, and `cnf`). Two per-suite tokens are
shipped in [`proof-v1-credentials.json`](vectors/proof-v1-credentials.json) with
the positive→credential map: a **classical** token (P-4/P-5/P-6, primary group
`[client Ed25519]`) and a **hybrid** token (P-2, primary group `[client Ed25519,
client ML-DSA-65]`). Each proof's `credential_hash` (−70001) is SHA-256 over the
exact token bytes.

The `cnf` uses a profile-defined confirmation method, `cnf.hs_signer_suite` =
base64url(SHA-256(RFC 8949 det-CBOR `[suite_id, [ordered raw component public
keys]]`)), which resolves to the credential-profile §5 signer-suite record of the
**primary** signer group — uniformly covering a classical (one-key) and a hybrid
(two-key) group. The gate and checker require it to resolve to **exactly one** plan
group; **approver** groups (e.g. P-5's group 2) bind their own enrollment and are
never placed in `cnf`. This thumbprint is distinct from the C1 replay thumbprint
(no domain separator, no enrollment epoch: it binds key material, replay binds the
enrollment). The gate proves the full chain load-bearing with re-signed
counter-proofs — a flipped/foreign issuer signature, wrong audience, missing
claim, wrong `typ`, an expired window, a `cnf` resolving to no group, and a
tampered proof hash each turn it red.

**Configured issuer binding (U2).** The verifier context pins a **configured
trusted issuer** (`issuer.iss` in `proof-v1-credentials.json`), and full
verification requires the signed JWT `iss` to equal it **exactly** — a non-empty
`iss` is not sufficient, and trust is never inferred from possession of the
signing key. The gate proves this load-bearing with a re-signed counter-case: a
token correctly signed by the issuer key but carrying `iss =
"https://evil-issuer.example"` is **denied** (neutralizing the configured issuer
turns the gate red), while every shipped credential carrying the exact configured
`iss` is admitted. This is the same issuer namespace that scopes `(iss, jti)`
credential revocation and `(iss, sid)` session resolution.

**Authoritative credential revocation (U1).** Individual credential revocation is
normative (credential-profile §6 / §3.1) and enforced: the authority holds an
off-wire revocation store keyed by the exact credential-ID tuple `(iss, jti)`,
consulted **after** issuer-signature and profile validation, failing **closed**
for an otherwise-valid, unexpired credential whose `(iss, jti)` is listed. The
shipped store lists `cred-revoked-1`; none of the live credentials are listed
(**positive unrevoked evidence**). The gate re-signs an otherwise-profile-valid
credential carrying the revoked `jti`, proves it passes signature/profile checks
yet is **denied** by the store (removing or bypassing the lookup turns the gate
red), and proves the match is the **exact tuple** so unrelated identities never
collapse: a different `jti`, or the same `jti` under a different `iss`, does not
match. This credential revocation is **distinct** from session-wide `(iss, sid)`
revocation and from enrollment revocation — it affects exactly one token, adds no
wire bit, and introduces no consume-once behavior.

**Encoding matrix (G1).** `cnf.hs_signer_suite` is a **JWT** confirmation method and
binds a signer-suite record of any component count. A **CWT** `cnf` is a single
RFC 8747 `COSE_Key` (claim 8) — it can pin exactly one key, so it binds a
**classical** (single-key) primary group only (N-1 is that shape). v16 defines no
multi-key CWT confirmation method (none is allocated or reserved), so a **hybrid**
proof credential MUST be an `at+jwt` (JWT) token; a CWT credential whose `cnf`
purports to confirm a hybrid suite is rejected — its single-key `cnf` resolves to
a classical record that matches no group of the hybrid proof's plan. The gate pins
this disposition (credential-profile §1.1), asserts the shipped hybrid credential
is a compact JWT and the only shipped CWT credential (N-1) is single-key, and
proves the causal reject: a single-`COSE_Key` `cnf` resolves the classical primary
(P-4) but denies the hybrid suite (P-2). See credential-profile.md §1.1.

**Credential clearance grammar (H2, Gate-2 value 11).** The credential `clearance`
claim (JWT text `clearance`; CWT integer key −70006) carries the **Level** and
**Compartments** axes only; **assurance is structurally absent** from the credential
wire (derived from verified key material at admission, never issuer-asserted). The
value is the same semantic two-element array in both encodings:
`[level, compartments]` where `level` is a uint `0..3`
(`0`=Public, `1`=Internal, `2`=Confidential, `3`=Secret, matching the Rust `Level`
discriminants) and `compartments` is an array of **bit indices** (uint `0..63`)
into the `CompartmentSet(u64)`, strictly ascending and unique (empty allowed) — the
credential wire projection of the versioned `InitialLabelMap`, **not** names and
**not** a bitmask integer. The fixtures' `[2,[5,7]]` conforms. The CDDL freezes the
shape and domains (`credential-clearance`); the gate enforces the strict-ascending/
unique order numerically (the pinned pycddl cannot), validates every shipped
credential clearance (the two at+jwt tokens and the CWT credential N-1), and proves
it load-bearing with re-signed counter-proofs: an unknown level, an out-of-range or
duplicate or descending compartment, compartment names, a bitmask integer, and an
extra (assurance) element each turn the gate red. See credential-profile.md §8.

## Common fixtures

| Field | Value |
|---|---|
| canonical service domain (`aud`) | `registry.svc.hyprstream.test` |
| `verifier_now` (frozen evaluation instant) | 1786000015 (integer Unix seconds) |
| `iat` / `exp` | 1786000000 / 1786000030 (integer Unix seconds) |
| `cti` (`request_id`) | `3f1c9a04b7d2416e8c05a9137b6e2d80` |
| `Nonce` (server challenge) | `8ad3011f4c76b25e90c4713f5a26ee08` |
| `capnp_schema_id` request / response | `0xd4d0f2a1b3c58e67` / `0x91b7c0e5a2f3d48a` |
| `credential_hash` | SHA-256 of the fixture credential bytes: `78fa43168352720cbbee52b6d593f0c9a7f4dd56b8c20cab20b924f4ba3ae081` |
| `external_aad` | zero-length in every `Sig_structure` |

Object encoding is untagged: typing is performed by the protected `typ` header
(RFC 9596 label 16), never by a CBOR tag.

## Positive vectors

| ID | Structure | Bytes | What it fixes | SHA-256 |
|---|---|---|---|---|
| P-1 | `COSE_Sign1` | 470 | Unattributed proof, classical suite, `Nonce` present, self-asserted `COSE_KeySet` matching the one plan component, `credential_hash` null | `cc967d18e3185e67d065e5c6650bb9afc76b1cdf49465e17ae8460df1ce6a46b` |
| P-2 | `COSE_Sign` | 3789 | Authenticated hybrid `hs-cose-sign-ed25519-mldsa65-wns-v1`: two component signatures in one logical signer group, counted as one signer | `c3001e8e108c3f9eef10891c3b32a5cd9177f10f6d55cf20818b1f1e398a44b8` |
| P-3 | `COSE_Sign1` | 374 | Response proof signed by the enrolled service key; `cti` echoes the request ID; no unattributed key set | `57e5fb1a5f26f28bb589d937e4df5c5b9136f3b58651225a7cbf74ec7bca96a8` |
| P-4 | `COSE_Sign1` | 1626 | Authenticated classical proof with an encrypted `response_binding` (unary, `protection_mode` encrypted, ML-KEM-768 recipient, alg −70200) | `68730f7ebf3b94c3b9379f01a6d9cc03fef76cc9f47dede44a6ccf8e99a2b637` |
| P-5 | `COSE_Sign` | 562 | `TokenBoundAndApproved`: two distinct logical signer groups, two principals, one approval each — signature entries, not countersignatures | `ca76143118078498920a862de7b084af8e4f264a49a8bf25c1d89c7b8213a723` |
| P-6 | `COSE_Sign1` | 411 | Authenticated classical proof with a cleartext stream-setup `response_binding` (`response_kind` stream_setup, `protection_mode` cleartext, null recipient) — exercises the orthogonal axes | `86f50e9862a45d0805786ae207fc310693e75f60dc50d44682b9ae1846030201` |
| P-7 | `COSE_Sign1` | 1605 | Bound response proof whose `response_binding` equals the originating request (P-4) field-for-field | `ab0b26bd2956ad31dcd047869a76e663171e2c57f0414d9d91d7795ba99aa5ab` |
| P-8 | `COSE_Sign` | 5862 | Hybrid unattributed proof; two embedded keys in plan order, each signature verified against its embedded key | `4f8fdbf677f7b28cf16b45af61067602f7f4082be2ced79b251ede6ecf82e6df` |
| P-9 | `COSE_Sign1` | 395 | Session-bound classical proof whose `exp` equals the authoritative session expiry (accepts within both bounds) | `bfa6026b0c940b15c050274b640344ef77117585881a6c0c25c1b39671bba77d` |

### P-1 — unattributed `COSE_Sign1` (complete CBOR)

```text
84590102a8013202843a000111d63a000111d53a000111d43a000111d30456756e617474726962
757465642d656432353531392d311078246170706c69636174696f6e2f766e642e687970727374
7265616d2e70726f6f662b6377743a000111d37768732d7270632d726571756573742d70726f6f
662d76313a000111d481a30101027768732d636f73652d7369676e2d656432353531392d763103
81a201320256756e617474726962757465642d656432353531392d313a000111d5013a000111d6
81a501010256756e617474726962757465642d656432353531392d3103322006215820cd14b37f
956e953194ff7fb73b3d81dcc561d61a7538094b7c3e1a643ee5f3aaa0588ba903781c72656769
737472792e7376632e6879707273747265616d2e74657374041a6a74329e061a6a74328007503f
1c9a04b7d2416e8c05a9137b6e2d800a508ad3011f4c76b25e90c4713f5a26ee083a00011170f6
3a000111711bd4d0f2a1b3c58e673a00011172581a000000000000000001000000170000000100
05000000000000003a00011173f65840566c171daaa2deefb674deb4a238d597086459975148df
ccb28a79567fbdbee6e639502ac31d30d535836b3435396951921cc43af507a54ed288e955f1a3
ba0b
```

Line breaks are page wrapping only; the authoritative bytes are `cbor_hex` in
[`vectors/proof-v1-positive.json`](vectors/proof-v1-positive.json), whose
SHA-256 is recorded in the table.

Protected bucket (decoded):

```text
1  (alg)  = -19                     Ed25519
2  (crit) = [-70103,-70102,-70101,-70100]
4  (kid)  = "unattributed-ed25519-1"
16 (typ)  = "application/vnd.hyprstream.proof+cwt"
-70100    = "hs-rpc-request-proof-v1"
-70101    = [ { 1:1, 2:"hs-cose-sign-ed25519-v1",
                3:[ { 1:-19, 2:"unattributed-ed25519-1" } ] } ]
-70102    = 1
-70103    = [ { 1:1, 2:"unattributed-ed25519-1", 3:-19, -1:6, -2:h'cd14…f3aa' } ]
```

Claims payload (decoded):

```text
3  (aud)   = "registry.svc.hyprstream.test"
4  (exp)   = 1786000030
6  (iat)   = 1786000000
7  (cti)   = h'3f1c9a04b7d2416e8c05a9137b6e2d80'
10 (Nonce) = h'8ad3011f4c76b25e90c4713f5a26ee08'
-70001     = null            ; no credential presented
-70002     = 15335023507516264039        ; 0xd4d0f2a1b3c58e67
-70003     = h'0000…0000'    ; exact Cap'n Proto request bytes
-70004     = null            ; unbound response
```

### P-3 — response proof (complete CBOR)

```text
8458b6a7013202833a000111d53a000111d43a000111d30451736572766963652d656432353531
392d3110782d6170706c69636174696f6e2f766e642e6879707273747265616d2e726573706f6e
73652d70726f6f662b6377743a000111d3781868732d7270632d726573706f6e73652d70726f6f
662d76313a000111d481a30101027768732d636f73652d7369676e2d656432353531392d763103
81a201320251736572766963652d656432353531392d313a000111d501a05878a803781c726567
69737472792e7376632e6879707273747265616d2e74657374041a6a74329e061a6a7432800750
3f1c9a04b7d2416e8c05a9137b6e2d803a00011170f63a000111711b91b7c0e5a2f3d48a3a0001
117258190000000000000000010000000f0000000200030000000000003a00011173f65840f2cb
3bfa5eba0d6699c084665f9ab4f3b39ede69a8f2b7c8a78eea17aa85f3edba7b09751c28399762
e77e1443512ed80f27693a98d646c00290658b4b33050f
```

`typ` is `application/vnd.hyprstream.response-proof+cwt`, `hs_domain` is
`hs-rpc-response-proof-v1`, `cti` is the request's `request_id`, `-70002` is the
response root type ID, `-70003` is the exact response bytes, and `-70001` is
null. N-19 and N-22 are its negatives.

### P-2, P-4, P-5, P-6, P-7, P-8

Their complete `cbor_hex`, protected buckets, and payloads are in
[`vectors/proof-v1-positive.json`](vectors/proof-v1-positive.json) (P-2 carries
a 3309-byte ML-DSA-65 signature and P-4 a 1184-byte ML-KEM-768 encapsulation
key, so inlining them is not useful). The body protected headers of P-2 carry
`crit = [-70101,-70100]` and each signature entry carries `crit = [-70102]`
with `hs_logical_signer_group = 1`; P-5 differs only in carrying two groups
(1 and 2) with distinct principals. P-4 and P-6 share one classical
`COSE_Sign1` shape and differ only in the `response_binding`: P-4 is
`protection_mode` encrypted (a present ML-KEM-768 recipient at alg −70200),
P-6 is `response_kind` stream_setup with `protection_mode` cleartext and a null
recipient — the two together fix all four values of the orthogonal axes. P-7 is
a **bound response proof**: a response-typed `COSE_Sign1` signed by the service
key whose `response_binding` equals P-4's request binding field-for-field, so the
suite tests §4's realized-binding equality rule, not only local map shape (its
mismatch counterpart is N-32).

## Negative vectors

Every negative vector is **cryptographically valid over its own mutated
bytes** — each was signed after mutation with the correct test key — so a
verifier that rejects it must do so for the stated profile rule, not because a
signature failed to check. The exceptions are stated in their notes: N-1 is a
genuine issuer-signed credential presented in the wrong slot, N-2 is the P-2
`COSE_Sign` proof verbatim presented in the wrong slot, and N-5 retains a genuine
Ed25519 signature over the stripped object.

| ID | Deny class | Vector | Bytes |
|---|---|---|---|
| N-1 | type-confusion | Profile-valid issuer-signed CWT credential (`cnf`/tenant/clearance) presented in the proof slot | 304 |
| N-2 | type-confusion | Proof CWT (`COSE_Sign`) presented in the credential/authorization slot | 3789 |
| N-3 | missing-typ | Protected `typ` (label 16) absent | 1587 |
| N-4 | domain-separation | Correct `typ` with the response-proof `hs_domain` | 1628 |
| N-5 | component-stripping | Hybrid proof with the ML-DSA-65 entry stripped | 438 |
| N-6 | parser-cap | Otherwise-valid nine signer-group `COSE_Sign`, over the 1\*8 group cap (cap is the sole denial) | 2803 |
| N-7 | parser-cap | Signer group with three components | 1684 |
| N-8 | closed-claim-set | Unknown claim key −70050 (unallocated) | 413 |
| N-9a | non-deterministic-encoding | Claims map keys not in deterministic order | 395 |
| N-9b | non-deterministic-encoding | Indefinite-length claims map | 396 |
| N-9c | non-deterministic-encoding | Floating-point `exp` | 399 |
| N-9d | non-deterministic-encoding | Duplicate `aud` claim key | 423 |
| N-10a | crit-set | `crit` omits `hs_domain` | 1621 |
| N-10b | crit-set | `crit` carries an unknown extension label | 1631 |
| N-10c | crit-set | `crit` names `hs_domain`, parameter absent from that bucket | 1597 |
| N-10d | crit-set | Duplicate `crit` entry | 1631 |
| N-10e | crit-set | `crit` in descending label order | 1626 |
| N-10f | disposition-confusion | Credential-bound proof carrying `hs_unattributed_key_set` | 1697 |
| N-11 | algorithm | Deprecated polymorphic `EdDSA` (−8) | 1626 |
| N-12 | suite-plan | In-range (≤64B) unknown `suite_id` outside the closed suite set (registry closure only) | 1633 |
| N-52 | parser-cap | `suite_id` of 65 bytes, over the 64-byte cap (the sole >64-byte suite proof) | 1669 |
| N-53 | cross-group-key-alias | One Ed25519 key under two kids in two signer groups (resolved-key alias; sole denial) | 633 |
| N-13 | parser-cap | `kid` of 65 bytes | 1727 |
| N-14 | closed-claim-set | Required `credential_hash` absent rather than null | 356 |
| N-15 | credential-binding | Credential presented with a null signed `credential_hash` | 362 |
| N-16 | freshness | Unattributed proof with no `Nonce` | 452 |
| N-17 | unprotected-authority | `alg`/`kid` only in the unprotected header | 1626 |
| N-18 | key-set-strictness | Unattributed key set with a surplus element (no matching plan component) | 2456 |
| N-19 | disposition-confusion | Response proof carrying `hs_unattributed_key_set` | 446 |
| N-20 | plan-mismatch | Signature entry citing a group absent from the plan | 562 |
| N-21 | non-deterministic-encoding | Tagged `COSE_Sign1` (CBOR tag 18) | 1627 |
| N-22 | response-cti-binding | Response proof whose `cti` does not echo the originating request's request_id | 374 |
| N-23 | response-binding | Encrypted binding (`protection_mode` 2) with a null KEM recipient | 411 |
| N-24 | response-binding | Cleartext binding (`protection_mode` 1) carrying a KEM recipient | 1626 |
| N-25 | response-binding | `response_kind` value 3 outside the closed enum {1,2} | 411 |
| N-26 | parser-cap | `aud` of 129 bytes, over the 128-byte `MAX_SERVICE_DOMAIN_BYTES` cap | 496 |
| N-27 | response-binding | Cleartext unary `response_binding` carried as a non-null map | 411 |
| N-28 | suite-plan | Hybrid suite plan with only one Ed25519 component (hybrid→classical downgrade) | 1639 |
| N-29 | aud-syntax | `aud` with an uppercase byte (`Registry.svc`) | 378 |
| N-30 | aud-syntax | `aud` with an illegal first byte (`-registry.svc`) | 379 |
| N-31 | response-schema-binding | Bound response proof whose `-70002` mismatches the realized `response_binding` root_type_id | 1601 |
| N-32 | response-binding-equality | Response proof whose `response_binding` mismatches the originating request (P-4) | 1605 |
| N-33 | plan-key-uniqueness | `COSE_Sign` plan repeating one `(alg, kid)` across two logical groups | 558 |
| N-35 | unattributed-keyset | Unattributed proof whose embedded key set does not match the signing key | 470 |
| N-37 | unattributed-keyset | Hybrid unattributed key set duplicating the Ed25519 element (no ML-DSA-65 key) | 3942 |
| N-38 | unattributed-keyset | Hybrid unattributed key set with elements reordered out of plan component order | 5862 |
| N-40 | group-id-order | `signature_plan` with two groups sharing one `group_id` | 562 |
| N-41 | group-id-order | `signature_plan` with group IDs out of ascending order | 562 |
| N-42 | unattributed-keyset | Unattributed key set with three elements, over the 1..2 ceiling | 944 |
| N-43 | response-aud-binding | Bound response proof whose `aud` differs from the originating request (P-4) | 1602 |
| N-44 | parser-cap | `capnp_body_bytes` of 1 MiB + 1, one byte over the 1048576 cap | 1048952 |
| N-45 | nonce-length | Unattributed proof whose Nonce is 15 bytes, under the 16-byte floor | 469 |
| N-46 | nonce-length | Unattributed proof whose Nonce is 65 bytes, over the 64-byte ceiling | 520 |
| N-47 | parser-cap | `kid` of 0 bytes, under the 1-byte floor | 1594 |
| N-48 | cbor-truncation | Truncated CBOR: response signature bstr header declares 65 bytes with 64 present | 374 |
| N-49 | integer-truncation | Truncated CBOR integer: claims `iat` argument `19 00` declares 2 bytes with 1 present | 244 |
| N-50 | proof-credential-expiry | Authenticated proof whose `exp` (1786000060) exceeds its mapped credential's `exp` (1786000030) | 244 |
| N-51 | proof-session-expiry | Session-bound proof whose `exp` (1786000025) is within the credential expiry but exceeds the authoritative session expiry (1786000020) | 395 |

### Notes on individual negatives

- **N-1 / N-2 (type confusion, both directions).** A proof and a credential are
  disjoint by construction: different protected `typ`, different signing keys
  (issuer key versus `cnf`-bound proof key), different domain separator. Both
  directions deny before any claim is interpreted. N-1 is a **profile-valid**
  issuer-signed CWT credential — `typ = application/cwt`, an issuer signature the
  gate re-verifies, an RFC 8747 `cnf` PoP binding, tenant (−70005), and clearance
  (−70006) — presented in the proof slot, so it exercises rejection of a
  *well-formed* credential, not a malformed token. It is a v16 **Reusable**
  credential and carries no use-profile field (v16 is Reusable-only; §4 of the
  credential profile). N-2 is the two-entry `COSE_Sign` P-2 verbatim in the
  credential slot, labelled `COSE_Sign` to match its bytes.
- **N-4 (domain confusion).** `typ` and `hs_domain` are **paired** in the
  normative CDDL, not independent choices: a request proof carries exactly
  (`proof-typ`, `request-proof-domain`) and a response proof exactly
  (`response-proof-typ`, `response-proof-domain`). The request-`typ` ×
  response-domain cross-product (and its reverse) fails structural CDDL
  validation, for both `COSE_Sign1` and the `COSE_Sign` body bucket — the gate
  submits the cross-product mutants and asserts the CDDL rejects them.
- **N-5 (component stripping).** The retained Ed25519 entry's `Sig_structure`
  still covers the hybrid `signature_plan`, so it cannot be reinterpreted under
  the standalone classical suite; the missing plan component denies
  independently. This is the vector that proves the weakly-non-separable
  property is enforced by both crypto and policy. The gate (causality inventory,
  §12) asserts N-5's exact violation shape: the set of `(group, alg, kid)` triples
  carried by its signature entries is a **proper subset** of its plan components —
  a plan component has no signature entry. Restoring full coverage (a de-fanged
  N-5 with the ML-DSA-65 entry re-added and a distinct payload) turns the gate red
  because the negative no longer exhibits its claimed stripping, mirroring the
  N-20 plan-mismatch treatment (§C4).
- **N-6 / N-7 / N-13 (parser caps).** The proof-v1 caps are exact:
  1..8 signer groups, exactly-per-suite components, and 1..64 bytes for `kid`.
  Raising a cap is an incompatible profile revision. The `kid` byte cap is pinned
  in the CDDL text and enforced numerically by the gate over every fixture, and
  N-13 is asserted to exceed the 64-byte cap — so a widening to 128 turns the
  gate red. **N-6 (Q2)** is a **fully otherwise-valid** nine signer-group
  `COSE_Sign`: every group uses a known (classical) suite with matching ordered key
  material, every component has a matching **valid** Ed25519 signature (nine
  independent enrolled signers), and all `(alg, kid)` pairs and `group_id`s are
  unique and ascending — so a cap-less verifier would accept it and the frozen
  `1*8` group cap is its **sole** denial. The gate verifies all nine signatures and
  proves the isolation by showing an **eight-group truncation validates** the plan
  (a de-fang to eight groups, or a verifier omitting the cap, turns the suite red).
- **Byte-range boundary coverage (E1) — N-44 / N-45 / N-46 / N-47.** The pinned
  pycddl 0.3.0 strips **every** `.size (LO..HI)` byte-length range (it mis-evaluates
  them as integer-value bounds), so each stripped range needs a causal boundary
  negative that the numeric gate rejects. Beyond `suite_id`/`kid`/`aud` uppers
  (N-12/N-13/N-26) these close the remaining ends: **N-44** carries a
  `capnp_body_bytes` of exactly 1 MiB + 1 (upper of `0..1048576`), staying under
  the 2 MiB object cap so it denies solely on the body-length rule; **N-45** and
  **N-46** carry a 15-byte and a 65-byte Nonce (the lower and upper of the
  `server-challenge` `16..64` range) on an otherwise-valid unattributed proof;
  **N-47** carries a 0-byte `kid` (the lower of `kid` `1..64`), independent of the
  `aud` `.regexp`. A gate meta-guard (§3h) discovers every stripped range straight
  from the CDDL and requires each violable end to map to a boundary negative
  sitting *exactly* on `LO-1`/`HI+1` (or an explicit mechanical justification: the
  `aud` lower end is subsumed by the service-domain `.regexp`, and the
  `capnp_body_bytes` lower end is 0, the minimum) — so a newly added `.size` range
  cannot ship without boundary coverage.
- **N-48 (strict CBOR truncation, E2).** A length-delimited (`bstr`/`tstr`) value
  whose header over-declares its length must be rejected as truncated, not
  silently accepted by lenient slicing. N-48 is **P-3 with its trailing Ed25519
  signature `bstr` header widened from `58 40` (64 bytes) to `58 41` (65 bytes)
  with no byte added** — the reviewer's exact exploit (`58 41` followed by only
  64 bytes). The shared strict decoder now raises when `len(rest) < declared`, so
  a proof whose surviving signature would otherwise verify is refused at the
  parser; the causality inventory (§12) asserts N-48 fails to decode with a
  truncation error. A `tstr` truncation probe (`63 61 61` — three declared bytes,
  two present) exercises the same guard on the text-string branch.
- **N-49 (strict CBOR integer-argument truncation, G2).** The same fail-closed
  rule applies to the **additional-information argument** of an integer (ai
  24/25/26/27 carry a 1/2/4/8-byte argument). N-49 is a proof whose claims payload
  encodes `iat` as `19 00` — an `ai=25` header declaring a 2-byte argument with
  only one byte present. The outer `COSE_Sign1` array decodes, but the payload's
  integer argument is truncated, so the shared strict decoder raises
  (`len(rest) < n`) instead of reading a short/zero value; the causality inventory
  (§12) decodes the payload and asserts the truncation. This closes the sibling of
  the length-delimited fix at the integer-argument level, and the same reader
  restores the correct minimal-argument floor (a value in `24..255` must use the
  1-byte `ai=24`, so `19 00 18` = 24-in-two-bytes is rejected as non-minimal). The
  exact-length counterpart `19 01 00` (= 256, the smallest value that legitimately
  needs `ai=25`) decodes.
- **N-50 (proof expiry cannot exceed the credential, H1).** A proof's `exp` MUST
  NOT exceed the `exp` of the credential (or session) it is bound to
  (credential-profile §5) — a proof cannot outlive the authority it presents. N-50
  is an otherwise-valid, correctly-signed authenticated proof whose own `exp` is
  `1786000060`, past its mapped classical credential's `exp` of `1786000030`; both
  are still valid at `verifier_now = 1786000015`, so a verifier that merely checks
  each artifact against the clock accepts it. The gate binds the two in the
  authenticated-context loop (`proof.exp <= mapped credential.exp` for every
  positive) and the causality inventory (§12) resolves N-50's mapped credential
  from its `credential_hash` and asserts `proof.exp > credential.exp`.
- **P-9 / N-51 (proof expiry cannot exceed the authoritative session, K1).** A
  credential carrying an OIDC `sid` (§3.2) is bound to an **authoritative session**
  whose state — including its expiry — the authority stores keyed by `(iss, sid)`
  (§3.4); it is **not** a credential wire field. A proof's `exp` MUST NOT exceed
  **both** the credential expiry **and** that session expiry. The fixtures ship a
  user-session credential (`credential_kind = user-session`, carrying `sid`) whose
  authoritative session (in `proof-v1-credentials.json` `sessions`, keyed by
  `(iss, sid)`, `status = active`) expires at `1786000020`, earlier than the
  credential's `1786000030`. **P-9** is the boundary-accept: a session-bound proof
  whose `exp` equals the session expiry (`1786000020`), within both bounds.
  **N-51** is the causal denial: an otherwise-valid, correctly-signed session-bound
  proof whose `exp` is `1786000025` — within the credential expiry but **past** the
  session expiry — so it denies **solely** on the session bound (the credential
  bound is satisfied and all three are unexpired at `verifier_now`). The gate and
  checker enforce `proof.exp <= session.exp` for a session-bound proof and validate
  the session as active, `(iss, sid)`-keyed, and `iss`/`sub`/`tenant`-coherent with
  the credential; an unknown, revoked, expired, or mismatched session denies. The
  authoritative session state also carries a **deterministic integer
  `clearance_epoch`** (§3.4, L3) — an off-wire authority field, never a credential
  claim; the gate and checker require it present and a non-negative integer, and a
  missing, non-integer, or negative `clearance_epoch` denies. A sid-keyed session
  is a user session, so its **`session_kind` MUST be the exact interactive kind**
  (`interactive`, M2); a missing, wrong-type, empty, or workload/service
  `session_kind` denies. The authoritative session also carries a
  **deterministic integer `created`** (§3.4, T2) — the issuance instant, an
  off-wire authority field, never a credential claim. Every shared
  session-validation path requires `created <= verifier_now < expires_at`,
  including coherent ordering against expiry; a missing, non-integer,
  future (`created > verifier_now`), or `created >= expires_at` session denies.
  The
  classical and hybrid credentials are typed `credential_kind = rfc8693` —
  **non-interactive** token-exchange / JWT-bearer tokens (a user subject with no
  interactive OIDC session) that carry no `sid`. The gate enforces this
  classification coherence (aligned with B's `IssueTokenProfile` enum) so
  sid-presence is never ambiguous: a `user-session` credential MUST carry `sid`, a
  `rfc8693`/`rfc7523` credential MUST NOT, and a `service` credential has a
  `service:`-prefixed subject and no `sid` (§3.2/§3.3).
- **N-12 / N-28 (suite ↔ component-plan binding).** Each suite_id is bound to its
  exact ordered algorithms and component count: `hs-cose-sign-ed25519-v1` is
  exactly one Ed25519 component, `hs-cose-sign-ed25519-mldsa65-wns-v1` is exactly
  Ed25519 then ML-DSA-65. The suite set is closed. N-28 is a causal
  **hybrid→classical downgrade** — the hybrid suite with only its Ed25519
  component — and denies because the hybrid group requires both. **N-12 (O1)** is
  a deterministic **in-range (≤64-byte)** unknown suite_id
  (`hs-cose-sign-unknown-suite-v1`) that satisfies every sibling structural, size,
  type, plan/key-set, and signature requirement, so its **sole** denial reason is
  that the suite is not in the frozen registry — the gate proves this isolation by
  swapping in a known suite (classical, for its single Ed25519 component) and
  showing the same plan then validates. The separate **>64-byte** suite rule is the
  sole province of **N-52** (a 65-byte `suite_id`), which stays armed as the size
  boundary. Both N-12 and N-28 are rejected by the normative CDDL at the
  `signature-plan` level, and the checker additionally requires a `COSE_Sign1` plan
  to have exactly one component. (The unknown value is **not** registered; a
  verifier that accepted in-range unknown suites, or a revert of N-12 to the
  65-byte value, turns the gate red.)
- **N-29 / N-30 (aud lexical syntax).** `aud` reuses the shared
  `validate_service_domain` syntax: lowercase ASCII only, first byte a lowercase
  letter or digit, alphabet `[a-z0-9._/-]`. N-29 (`Registry.svc`, uppercase) and
  N-30 (`-registry.svc`, illegal first byte) deny. The CDDL declares the
  `.regexp` normatively; because the pinned pycddl does not enforce `.regexp`,
  `validate_profile.py` ports the exact syntax and applies it to every fixture
  and to these causal negatives, so the profile's audience namespace matches the
  transport's. (**Tracked disposition, K2, non-blocking:** RFC 8610 §3.8.3 `.regexp`
  uses **XSD whole-string** matching, so the normative `.regexp` is already
  whole-string; the pinned pycddl's *substring* search is a non-conformant tool bug
  that is fully compensated by the mechanical `validate_service_domain` gate plus
  N-29/N-30 — not a v16 correctness defect. PCRE `^`/`$` anchors are **not** added:
  under XSD semantics they are literal characters, so anchoring would corrupt the
  grammar rather than tighten it.)
- **P-7 / N-32 (response-binding field-for-field equality).** A response proof's
  realized `response_binding` MUST equal the originating request's map
  field-for-field (§4). P-7 is a bound response proof whose binding equals P-4's
  request binding; N-32 is a response proof whose binding **differs from P-4's in
  the `response_kind`/`protection_mode` axes** (a valid `stream_setup` + cleartext
  binding) while **keeping `root_type_id` == `-70002`**, so it flips exactly
  `binding_eq` and leaves `schema_eq` (and `aud_eq`/`cti_eq`) true (L1 isolation) —
  it denies only under equality with the request it answers. Both carry the
  originating request id, and the gate compares the two maps directly — testing
  equality, not just local map shape.
- **N-31 (response schema-ID binding).** For a bound response proof, claim
  `-70002` (response root type ID) and `response_binding[1]` (`root_type_id`)
  denote the same schema commitment, so `-70002` MUST equal the realized
  binding's `root_type_id`. CDDL cannot express this cross-field equality of two
  free uints, so it is a normative rule the gate and vector checker enforce over
  every bound response fixture. N-31 changes **only** `-70002` (its binding still
  equals P-4's, and the signature is valid), so it denies by this rule, not by a
  signature or map-shape failure.
- **N-43 (response `aud` binding — D2).** A bound response proof's `aud` (claim 3)
  MUST be the same canonical service domain the originating request was bound to,
  not merely a lexically valid domain. N-43 is P-7 with **only** `aud` mutated to
  a different service domain (`other.svc.hyprstream.test`); its `response_binding`,
  `cti`, and `-70002` still equal P-4's and its signature is valid, so it denies
  only under the request↔response audience equality. The full request-derived
  response-context set is four independent axes — `aud_eq`, `cti_eq`, `binding_eq`,
  and `schema_eq` — compared in **one** place (`response_context_bindings`, gate
  §11). **One-false / three-true isolation (L1):** each response negative flips
  **exactly** its named axis and leaves the other three true — N-43 `aud_eq`, N-22
  `cti_eq`, N-32 `binding_eq`, N-31 `schema_eq` — which the gate asserts vector by
  vector (the false-axis set must equal `[named]`). Both positive controls bind
  **all four**: P-7 with a non-null `response_binding` against P-4, and **P-3 with
  a null binding against P-2 (L2)** — a response is never accepted through absent
  context, so every response-typ positive MUST carry an exact originating request.
  Allowing a second false axis, deleting an equality axis, or breaking a positive
  control turns the gate red. The causality inventory (§12) additionally asserts
  every one of the 56 negatives exhibits its advertised violation shape, with a
  meta-guard that the inventory covers exactly the full negative-vector ID set — so
  a future negative cannot be added without its own denial-shape check.
- **N-33 (plan `(alg, kid)` uniqueness).** Every `(alg, kid)` pair MUST be unique
  across the whole `signature_plan`, regardless of group ID — one key must not
  sign under two logical groups, or a single signer could satisfy a two-group
  (`k-of-n` / `all`) policy. CDDL cannot express this, so the gate and the vector
  checker enforce it (the checker keys duplicate detection on `(alg, kid)`, not
  `(group, alg, kid)`). N-33 is a fully signature-valid `COSE_Sign` repeating one
  `(alg, kid)` across groups 1 and 2, and denies by the uniqueness rule.
- **N-53 (cross-group resolved-key alias — S1).** `(alg, kid)` uniqueness (N-33) is
  necessary but not sufficient: the labels are attacker-selected, so the **same
  Ed25519 public key can be published under two *different* kids** in two groups and
  sign both entries — one key satisfying a two-logical-group policy. The frozen rule
  is content-based: a **resolved public-key identity — `(algorithm, raw public-key
  bytes)`, never `kid`/`group_id`/enrollment label/plan position — may participate
  in at most one logical signer group**. It is enforced *after* key resolution for
  both unattributed embedded-key proofs and credential/enrollment-bound proofs;
  different algorithms are distinct identities, so legitimate hybrid groups (Ed25519
  + ML-DSA-65 in one suite group) and the ordered components inside one group remain
  valid. N-53 is an otherwise-valid two-group `COSE_Sign` — **unique `(alg, kid)`
  labels, both signatures cryptographically valid, ascending unique `group_id`s,
  known suites** — whose sole defect is that one resolved Ed25519 identity appears in
  both groups. Its label uniqueness and signature coverage pass; **content-identity
  uniqueness alone denies**, and replacing the second public key with a genuinely
  distinct key makes the construction valid. The same rule denies a credential
  **primary/approver alias** — the same resolved key presented as both the `cnf`
  primary and an approver group under different labels — while distinct primary and
  approver keys (P-5) stay valid.
- **Total-object cap (2 MiB).** The complete-object CDDL cannot bound the
  signature byte strings, so the 2 MiB total-object cap is a validator-side
  numeric check. The gate constructs a **structurally valid** oversized object (a
  real `COSE_Sign1` whose signature `bstr` is enlarged to a fixed ~2.2 MiB — not
  trailing padding the decoder would reject) and runs it through the **same**
  object-cap path the fixtures use: it decodes cleanly (so size is a cause
  distinct from trailing-data) yet is rejected **by size**. The fixed size makes
  the comparison genuine — raising the cap above it would accept it and turn the
  check red.
- **N-19 (response overlay).** A response proof uses the distinct
  `hyprstream-response-proof-claims` rule, which forbids the `Nonce` claim key
  and requires `credential_hash` to be exactly `null`; a response claims set
  carrying either fails structural CDDL validation.
- **N-9a…N-9d (deterministic encoding).** The profile's encoding rules are
  themselves acceptance criteria, not hygiene: unsorted keys, indefinite
  lengths, floating-point timestamps, duplicate keys, and (N-21) tags all deny.
- **N-10a…N-10f (`crit`).** `crit` sets are exact, not minimums; entries are
  unique, ascending, and each must occur in the same protected bucket.
- **N-11 (EdDSA −8).** RFC 9864 fully-specified algorithms: Ed25519 components
  use −19. The current implementation signs with the polymorphic identifier, so
  this vector is also the migration's regression test.
- **N-15 (authenticated null credential_hash — B5).** An authenticated request
  proof is credential-bound, so `credential_hash` (−70001) is the REQUIRED
  32-byte hash — never null. Null means "no credential", which is the
  unattributed case (a key set is then required), so the authenticated request
  claims rule pins `-70001 => credential-hash` (no null) and N-15 (authenticated
  shape, null hash, no key set) fails structural CDDL validation. The reverse
  direction — no credential presented with a non-null signed hash — is the same
  rule read the other way (a presentation-context case); a verifier must reject
  both.
- **N-40 / N-41 (group-ID order — B6).** Signer-group IDs MUST be unique and
  strictly ascending across the plan, so two logical groups can never collapse to
  one `group_id` (which would let a single logical group satisfy a multi-party
  policy) and order is canonical. CDDL cannot express this, so the gate and the
  vector checker enforce it. N-40 repeats a `group_id` (both groups id 1) and
  N-41 is out of ascending order (ids 2 then 1); both carry valid signatures and
  deny solely on the group-ID rule.
- **N-20 (plan membership — C4).** Every signature entry MUST cite a group that
  is in the signed plan. N-20 declares a signature entry whose `group_id` is
  absent from the plan; the gate asserts that at least one entry's
  `(group, alg, kid)` is not a plan component, so a verifier that ignores
  signed-plan membership fails the suite (de-fanging N-20 to in-plan groups turns
  the gate red).
- **N-22 (response cti binding — C5).** A response proof's `cti` (claim 7) MUST
  echo the originating request's `request_id`. N-22 carries the originating
  request id (P-4), keeps its `response_binding`, `aud`, and `-70002` **equal** to
  P-4's, and mutates **only** `cti`, so it flips exactly `cti_eq` and leaves the
  other three response axes true (L1 isolation); the gate asserts the contextual
  mismatch — mirroring N-31/N-32, not merely that the bytes differ from a positive
  (de-fanging the `cti` to the request id turns the gate red).
- **Replay-namespace thumbprints (C1).** The Reusable replay key is
  `(authenticated primary signer-suite thumbprint, cti)` and the unattributed key
  is `(unattributed proof-key-set thumbprint, cti)`. Both thumbprints are frozen
  to bytes (CDDL §7.1): SHA-256 over the RFC 8949 core-deterministic encoding of a
  CBOR array whose first element is one of two distinct domain-separator literals
  (`hs-rpc-replay-primary-suite-v1` / `hs-rpc-replay-key-set-v1`). The
  authenticated preimage is `[sep, suite_id, [ordered primary-group public keys],
  enrollment_epoch]` (approver groups excluded). **Authoritative primary
  enrollment (T1):** the authenticated `enrollment_epoch` is not a vector literal —
  it is derived from the credential's off-wire **primary** enrollment record (in
  `proof-v1-credentials.json` `primary_enrollments`), resolved by the **same
  content-bound discipline** as Q1: the record is located by recomputing its
  signer-suite thumbprint over the record's own `suite_id` + ordered public keys
  and matching it to the `cnf`-bound signer suite (labels are never trusted). The
  gate and checker require an active, **unexpired**, `primary`-role record whose
  `tenant`/`principal` equal the credential's, then **derive** the published
  authenticated thumbprint from that record's `enrollment_epoch` and assert it
  reproduces the frozen bytes for P-2/P-4/P-5/P-6/P-9. An **unknown**, tampered,
  key/suite-mismatched, **inactive**, **expired**, **cross-tenant**,
  **wrong-principal**, or **wrong-role** primary record denies, and an
  `enrollment_epoch` change de-fangs the thumbprint (turning the gate red); primary
  and approver records stay distinct while the S1 primary/approver resolved-key
  alias denial is preserved. The unattributed preimage is
  **content-bound (M1)**: `[sep, [ [suite_id, [ordered public keys]] per signer
  group ]]` — it binds each group's suite and its public keys in component order
  and **normalizes the attacker-chosen `group_id`/`kid` labels out**, exactly as
  the authenticated derivation excludes them. A self-asserted unattributed signer
  therefore cannot mint a fresh replay namespace by permuting a `group_id` or a
  `kid` over identical ordered public keys: the gate and checker prove that a
  `group_id`-only relabel and a `kid`-only relabel each keep the **same**
  thumbprint (so the replay key `(thumbprint, cti)` is unchanged and the second use
  is rejected as replay), while a different suite, a public-key byte, or a key
  reordering yields a **different** thumbprint (component order **inside** each group
  is preserved, so distinct cryptographic identities are never over-collapsed).
  Reverting to the old verbatim-label derivation turns the gate red. The per-group
  content records are additionally **canonically sorted by their RFC 8949
  deterministic-CBOR encoding as unsigned byte strings before hashing (R1)** — never
  on `group_id`/`kid`/plan position — so the same signer set `{A, B}` in **any**
  plan order maps to **one** replay namespace. `proof-v1-thumbprints.json` ships two
  cryptographically valid two-group unattributed proofs (`A,B` and `B,A`, freshly
  re-signed); the gate and checker verify both signatures and prove they hash to the
  **identical** namespace, while replacing a group key, changing a group suite, or a
  different group multiset stays distinct, and an **unsorted** (plan-order)
  derivation is order-sensitive (the closed bypass) — so removing the sort turns the
  gate red. Cross-implementation expected-thumbprint vectors are in
  `vectors/proof-v1-thumbprints.json`, and the gate and checker recompute both from
  the frozen derivation and assert they match — so two verifier implementations
  sharing replay state derive the same namespace for one signer.
- **P-5 (TokenBoundAndApproved disposition — C3).** The credential-bound primary
  group is selected by content — the plan group whose suite ID and exact component
  keys equal the `cnf`-resolved signer-suite record — and the exact-`cnf`-key
  denial is scoped to that group; each additional approver group verifies against
  its own enrolled keys, not the client's `cnf`. So P-5's two groups each verify
  against their own enrolled test key (the checker does exactly this), and the
  approver group is not over-rejected. **Authoritative approver enrollment (Q1):**
  being *different from `cnf`* is not enough — P-5's approver group must be an
  **authorized** approver. The authority holds an off-wire `approver_enrollments`
  record (in `proof-v1-credentials.json`) keyed by **cryptographic content** — the
  group's signer-suite thumbprint over its `suite_id` + ordered public keys, the
  same content-bound discipline as `cnf`/M1, never the attacker-chosen `group_id`/
  `kid`. The gate and checker resolve the approver group by **recomputing** that
  thumbprint (the record's stored thumbprint field is never trusted — it is
  recomputed from the record's own suite/keys and both must agree), then require an
  active, **unexpired** (`expires_at` > `verifier_now`) `approver`-role record whose
  `tenant` matches the credential and whose `enrollment_epoch` is a non-negative
  integer. An **unknown**, tampered, key/suite-mismatched, **inactive**, **expired**,
  **cross-tenant**, or **wrong-role** enrollment denies; omitting the validation
  turns the gate red. It is not a credential/wire claim and allocates no wire space.
- **N-17 (unprotected authority).** Algorithm identifiers and key material in
  unprotected headers establish no authority; authenticated signer keys are
  resolved from the credential `cnf` and anchored trust stores.
- **P-8 / N-18 / N-35 / N-37 / N-38 / N-42 (unattributed key-set correspondence).**
  An unattributed proof's embedded `hs_unattributed_key_set` is the **only** key
  material it has, so verification MUST use the embedded keys — not any
  out-of-band table. The gate and the vector checker require exact ordered 1:1
  correspondence with the plan (kid, alg, closed COSE_Key field set, key type /
  curve or parameter set, exact public-key byte length, and the frozen 1..2
  element ceiling) and verify each unattributed signature against its embedded
  key — the embedded map is keyed by `(alg, kid)` (B7), so a plan validly reusing
  one kid across algorithms neither overwrites a key nor crashes. P-8 is a valid
  hybrid unattributed proof (Ed25519 + ML-DSA-65 embedded, in plan order). The
  causal failures: **N-35** keeps a well-formed key set with a *different* public key
  (correct kid/alg/crv, re-signed with the real key) and denies at embedded-key
  verification; **N-18** carries a surplus element; **N-37** duplicates the
  Ed25519 element (so element 1 is not the ML-DSA-65 key); **N-38** reorders the
  two elements out of plan order; **N-42** carries three key-set elements, over
  the frozen 1..2 ceiling. The pinned `pycddl` (0.3.0, embedding the Rust `cddl`
  crate 0.9.1) panics validating the AKP/ML-DSA-65 `COSE_Key`, so the key set is
  passed to it opaquely for the protected-bucket pass only; its full validation
  (including the 1..2 element cap) is the stronger gate/checker correspondence
  above, and the normative CDDL is unchanged.
- **N-23 / N-24 (recipient/encryption relation).** The frozen 4-key
  `response_binding` (Gate-2 §19 #3/#4) makes the recipient non-null iff
  `protection_mode` is encrypted. N-23 is the encrypted shape with a null
  recipient; N-24 is the cleartext shape carrying one. Both are rejected by the
  CDDL itself, which expresses the relation as the choice of two protection
  shapes — `validate_profile.py` asserts that rejection.
- **N-25 (closed enum axis).** `response_kind` is the closed enum {1 unary,
  2 stream_setup}; value 3 denies. The CDDL rejects it, proving the axis is a
  closed enum, not an open integer.
- **N-26 (aud cap).** `aud` is 1..128 bytes — the shared
  `MAX_SERVICE_DOMAIN_BYTES` constant (Gate-2 §19 #7). A 129-byte `aud` denies.
  The cap is a byte-length range that the pinned CDDL validator does not
  enforce; `validate_profile.py` enforces it numerically and ties the value to
  the Rust `MAX_SERVICE_DOMAIN_BYTES` constant.
- **N-16 (unattributed challenge).** An unattributed proof MUST carry the server
  challenge (§4.7): the distinct `hyprstream-unattributed-proof-claims` rule
  makes `Nonce` a mandatory key (and pins `credential_hash` to `null`), so the
  N-16 no-`Nonce` shape fails structural CDDL validation rather than depending on
  a separate verifier check.
- **N-27 (cleartext-unary canonical encoding).** `response_binding` is null
  exactly when the response is neither encrypted nor streamed. A cleartext unary
  response is neither, so it uses the null encoding; the cleartext map
  alternative is restricted to `stream_setup`, giving cleartext unary a single
  canonical encoding. A cleartext-unary map (N-27) denies.

## Coverage gaps recorded for gate 2

These profile rules are not byte-encodable in a standalone vector and are
therefore stated as verifier obligations rather than shipped here:

1. Replay admission (exact replay of a valid proof denies) — needs a stateful
   verifier harness. v16 credentials are Reusable, so the credential ID is never
   a replay key; the credential-ID consume path returns only with the deferred
   `OneShotTransaction` amendment.
2. Freshness bounds (`iat` within clock-skew tolerance,
   `exp − verifier_now ≤` the per-disposition maximum lifetime) — needs a
   controlled clock.
3. `challenge_accept_until` semantics and the single bounded retry.
4. Component-key non-reuse across suites, profiles, and logical groups — a CI
   enrollment check, not a wire vector.
5. Sign-then-encrypt wrapper vectors, pending the COSE HPKE profile
   (watch item; nothing in this profile depends on it).
6. A dedicated credential-plane positive vector set for the amendment-10 integer
   claim keys (`-70005`/`-70006`/`-70007`). The proof direction is already covered
   — a proof carrying any credential key denies via the same closed-claim-set rule
   as N-8 (the proof claims map is closed at `-70001..-70004`) — and the
   type-confusion negative N-1 exercises a profile-valid v16 credential (a valid
   `cnf` PoP binding, tenant `-70005`, clearance `-70006`, and **no** use-profile
   field). v16 credentials are **Reusable-only** (operator decision 2026-08-20):
   there is no `credential_use_profile` claim and none is allocated (`-70008` is
   not reserved). A full credential-plane positive/negative vector set is future
   work, coordinated with the credential (WS-B) seat.
