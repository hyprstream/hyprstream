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
| [`vectors/proof-v1-positive.json`](vectors/proof-v1-positive.json) | 6 vectors that MUST verify |
| [`vectors/proof-v1-negative.json`](vectors/proof-v1-negative.json) | 38 vectors that MUST deny |

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
  component) and an **unknown suite_id** (N-12) are rejected at the
  `signature-plan` level — each suite is bound to its exact ordered algorithms
  and component count, and the checker additionally requires a `COSE_Sign1` plan
  to have exactly one component;
- an **`aud` violating the service-domain syntax** (N-29 uppercase, N-30 illegal
  first byte) is rejected — the gate ports `validate_service_domain` and applies
  it to every fixture, so the profile's audience namespace is no broader than the
  transport's; and
- the **size-cap** negatives N-13 (65-byte `kid`) and N-26 (129-byte `aud`) are
  asserted over their numeric caps, and the CDDL cap text (`kid` = `1..64`,
  `aud` = `1..128` plus its `.regexp`) is pinned so a widening drifts the gate to
  red — the pinned `pycddl` version cannot enforce `.size` byte-length ranges or
  `.regexp`, so these numeric/ported checks are load-bearing.

See [`README.md`](README.md) for the one remaining tooling note.

## Common fixtures

| Field | Value |
|---|---|
| canonical service domain (`aud`) | `registry.svc.hyprstream.test` |
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

### P-2, P-4, P-5, P-6

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
recipient — the two together fix all four values of the orthogonal axes.

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
| N-1 | type-confusion | Valid issuer-signed CWT credential (`application/cwt`) presented in the proof slot | 213 |
| N-2 | type-confusion | Proof CWT (`COSE_Sign`) presented in the credential/authorization slot | 3789 |
| N-3 | missing-typ | Protected `typ` (label 16) absent | 1587 |
| N-4 | domain-separation | Correct `typ` with the response-proof `hs_domain` | 1628 |
| N-5 | component-stripping | Hybrid proof with the ML-DSA-65 entry stripped | 438 |
| N-6 | parser-cap | `signature_plan` with nine signer groups | 2044 |
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
| N-12 | suite-plan | Unknown `suite_id` (65 bytes) outside the closed suite set | 1669 |
| N-13 | parser-cap | `kid` of 65 bytes | 1727 |
| N-14 | closed-claim-set | Required `credential_hash` absent rather than null | 356 |
| N-15 | credential-binding | Credential presented with a null signed `credential_hash` | 362 |
| N-16 | freshness | Unattributed proof with no `Nonce` | 452 |
| N-17 | unprotected-authority | `alg`/`kid` only in the unprotected header | 1626 |
| N-18 | key-set-strictness | Key-set element with no matching plan component | 2456 |
| N-19 | disposition-confusion | Response proof carrying `hs_unattributed_key_set` | 446 |
| N-20 | plan-mismatch | Signature entry citing a group absent from the plan | 562 |
| N-21 | non-deterministic-encoding | Tagged `COSE_Sign1` (CBOR tag 18) | 1627 |
| N-22 | response-binding | Response proof whose `cti` is not the request ID | 374 |
| N-23 | response-binding | Encrypted binding (`protection_mode` 2) with a null KEM recipient | 411 |
| N-24 | response-binding | Cleartext binding (`protection_mode` 1) carrying a KEM recipient | 1626 |
| N-25 | response-binding | `response_kind` value 3 outside the closed enum {1,2} | 411 |
| N-26 | parser-cap | `aud` of 129 bytes, over the 128-byte `MAX_SERVICE_DOMAIN_BYTES` cap | 496 |
| N-27 | response-binding | Cleartext unary `response_binding` carried as a non-null map | 411 |
| N-28 | suite-plan | Hybrid suite plan with only one Ed25519 component (hybrid→classical downgrade) | 1639 |
| N-29 | aud-syntax | `aud` with an uppercase byte (`Registry.svc`) | 378 |
| N-30 | aud-syntax | `aud` with an illegal first byte (`-registry.svc`) | 379 |

### Notes on individual negatives

- **N-1 / N-2 (type confusion, both directions).** A proof and a credential are
  disjoint by construction: different protected `typ`, different signing keys
  (issuer key versus `cnf`-bound proof key), different domain separator. Both
  directions deny before any claim is interpreted. N-1 is a **genuine**
  issuer-signed CWT credential (`typ = application/cwt`, signed by the issuer
  key — the gate re-verifies that signature) presented in the proof slot, so it
  exercises rejection of a *well-formed* credential, not a malformed token. N-2
  is the two-entry `COSE_Sign` P-2 verbatim in the credential slot, labelled
  `COSE_Sign` to match its bytes.
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
  property is enforced by both crypto and policy.
- **N-6 / N-7 / N-13 (parser caps).** The proof-v1 caps are exact:
  1..8 signer groups, exactly-per-suite components, and 1..64 bytes for `kid`.
  Raising a cap is an incompatible profile revision. The `kid` byte cap is pinned
  in the CDDL text and enforced numerically by the gate over every fixture, and
  N-13 is asserted to exceed the 64-byte cap — so a widening to 128 turns the
  gate red.
- **N-12 / N-28 (suite ↔ component-plan binding).** Each suite_id is bound to its
  exact ordered algorithms and component count: `hs-cose-sign-ed25519-v1` is
  exactly one Ed25519 component, `hs-cose-sign-ed25519-mldsa65-wns-v1` is exactly
  Ed25519 then ML-DSA-65. The suite set is closed. N-28 is a causal
  **hybrid→classical downgrade** — the hybrid suite with only its Ed25519
  component — and denies because the hybrid group requires both. N-12 is an
  unknown suite_id (also over-long) and denies as an unrecognized suite. Both are
  rejected by the normative CDDL at the `signature-plan` level, and the checker
  additionally requires a `COSE_Sign1` plan to have exactly one component.
- **N-29 / N-30 (aud lexical syntax).** `aud` reuses the shared
  `validate_service_domain` syntax: lowercase ASCII only, first byte a lowercase
  letter or digit, alphabet `[a-z0-9._/-]`. N-29 (`Registry.svc`, uppercase) and
  N-30 (`-registry.svc`, illegal first byte) deny. The CDDL declares the
  `.regexp` normatively; because the pinned pycddl does not enforce `.regexp`,
  `validate_profile.py` ports the exact syntax and applies it to every fixture
  and to these causal negatives, so the profile's audience namespace matches the
  transport's.
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
- **N-15 (token stripping).** The reverse direction — no credential presented
  with a non-null signed hash — is the same rule read the other way and is a
  presentation-context case rather than a distinct byte string; a verifier under
  test must reject both.
- **N-17 (unprotected authority).** Algorithm identifiers and key material in
  unprotected headers establish no authority; authenticated signer keys are
  resolved from the credential `cnf` and anchored trust stores.
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

1. Replay admission (exact replay of a valid proof denies; the one-shot
   credential consume path) — needs a stateful verifier harness.
2. Freshness bounds (`iat` within clock-skew tolerance,
   `exp − verifier_now ≤` the per-disposition maximum lifetime) — needs a
   controlled clock.
3. `challenge_accept_until` semantics and the single bounded retry.
4. Component-key non-reuse across suites, profiles, and logical groups — a CI
   enrollment check, not a wire vector.
5. Sign-then-encrypt wrapper vectors, pending the COSE HPKE profile
   (watch item; nothing in this profile depends on it).
6. Credential-CWT vectors for the amendment-10 integer claim keys
   (`-70005`/`-70006`/`-70007`) and the **`credential_use_profile`** signed wire
   claim. The proof direction is already covered — a proof carrying any credential
   key denies via the same closed-claim-set rule as N-8 (the proof claims map is
   closed at `-70001..-70004`). Adding credential-CWT vectors is **blocked on an
   operator disposition**: Gate-2 froze exactly the three credential CWT keys
   `-70005..-70007`, and `credential_use_profile` (Reusable / OneShotTransaction)
   has no correct existing signed-claim encoding, so encoding it requires a new
   allocation the operator must approve. See the operator-disposition handoff
   `.fleet-coord/handoffs/mac-v16-a-credential-use-profile-disposition.md`. Until
   then the credential claims set (cnf/clearance encoding included) is not frozen
   here, so credential vectors are deferred rather than baking in unapproved
   choices.
