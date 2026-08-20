# Canonical vectors — `hs-rpc-proof-v1`

Status: gate-2 input draft. These are the canonical positive and negative test
vectors for the v16 COSE RPC request-proof profile. They are the executable
half of the profile freeze: the structure lives in
[`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl), the label allocation
in [`private-label-registry.md`](private-label-registry.md), and every rule
either has a vector here or is not frozen.

Machine-readable files (the normative form — this page is the human index):

| File | Contents |
|---|---|
| [`vectors/proof-v1-keys.json`](vectors/proof-v1-keys.json) | Test keys, seeds, and fixture values |
| [`vectors/proof-v1-positive.json`](vectors/proof-v1-positive.json) | 5 vectors that MUST verify |
| [`vectors/proof-v1-negative.json`](vectors/proof-v1-negative.json) | 30 vectors that MUST deny |

Each JSON vector carries `id`, `title`, `expect` (`accept` / `deny`),
`structure`, `size_bytes`, `sha256`, full `cbor_hex`, and — for negatives — a
`deny_class` and the exact `deny_rule` violated.

## Reproducing and checking

```sh
python3 docs/standards/v16/tools/gen_proof_vectors.py   # regenerates, byte-identical
python3 docs/standards/v16/tools/check_proof_vectors.py # verifies the checked-in files
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

The CDDL was reviewed by hand: no CDDL validator is available in this
environment, so mechanical validation of the schema file is an open gate-2
task recorded in [`README.md`](README.md).

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
| P-4 | `COSE_Sign1` | 1624 | Authenticated classical proof with a non-null `response_binding` carrying the ML-KEM-768 recipient | `4e046d6a206a46662499af8ba8b5bbc4c0bc039abb6c56f5d9d4c97944c15fcc` |
| P-5 | `COSE_Sign` | 562 | `TokenBoundAndApproved`: two distinct logical signer groups, two principals, one approval each — signature entries, not countersignatures | `ca76143118078498920a862de7b084af8e4f264a49a8bf25c1d89c7b8213a723` |

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

### P-2, P-4, P-5

Too large to inline usefully (P-2 carries a 3309-byte ML-DSA-65 signature and
P-4 a 1184-byte ML-KEM-768 encapsulation key). Their complete `cbor_hex`,
protected buckets, and payloads are in
[`vectors/proof-v1-positive.json`](vectors/proof-v1-positive.json). The body
protected headers of P-2 carry `crit = [-70101,-70100]` and each signature
entry carries `crit = [-70102]` with `hs_logical_signer_group = 1`; P-5 differs
only in carrying two groups (1 and 2) with distinct principals.

## Negative vectors

Every negative vector is **cryptographically valid over its own mutated
bytes** — each was signed after mutation with the correct test key — so a
verifier that rejects it must do so for the stated profile rule, not because a
signature failed to check. The two exceptions are stated in their notes: N-2 is
P-2 verbatim presented in the wrong slot, and N-5 retains a genuine Ed25519
signature over the stripped object.

| ID | Deny class | Vector | Bytes |
|---|---|---|---|
| N-1 | type-confusion | Credential (`at+jwt` typed) presented in the proof slot | 216 |
| N-2 | type-confusion | Proof CWT presented in the credential/authorization slot | 3789 |
| N-3 | missing-typ | Protected `typ` (label 16) absent | 1585 |
| N-4 | domain-separation | Correct `typ` with the response-proof `hs_domain` | 1626 |
| N-5 | component-stripping | Hybrid proof with the ML-DSA-65 entry stripped | 438 |
| N-6 | parser-cap | `signature_plan` with nine signer groups | 2042 |
| N-7 | parser-cap | Signer group with three components | 1682 |
| N-8 | closed-claim-set | Unknown claim key −70005 | 413 |
| N-9a | non-deterministic-encoding | Claims map keys not in deterministic order | 395 |
| N-9b | non-deterministic-encoding | Indefinite-length claims map | 396 |
| N-9c | non-deterministic-encoding | Floating-point `exp` | 399 |
| N-9d | non-deterministic-encoding | Duplicate `aud` claim key | 423 |
| N-10a | crit-set | `crit` omits `hs_domain` | 1619 |
| N-10b | crit-set | `crit` carries an unknown extension label | 1629 |
| N-10c | crit-set | `crit` names `hs_domain`, parameter absent from that bucket | 1595 |
| N-10d | crit-set | Duplicate `crit` entry | 1629 |
| N-10e | crit-set | `crit` in descending label order | 1624 |
| N-10f | disposition-confusion | Credential-bound proof carrying `hs_unattributed_key_set` | 1695 |
| N-11 | algorithm | Deprecated polymorphic `EdDSA` (−8) | 1624 |
| N-12 | parser-cap | `suite_id` of 65 encoded bytes | 1667 |
| N-13 | parser-cap | `kid` of 65 bytes | 1725 |
| N-14 | closed-claim-set | Required `credential_hash` absent rather than null | 356 |
| N-15 | credential-binding | Credential presented with a null signed `credential_hash` | 362 |
| N-16 | freshness | Unattributed proof with no `Nonce` | 452 |
| N-17 | unprotected-authority | `alg`/`kid` only in the unprotected header | 1624 |
| N-18 | key-set-strictness | Key-set element with no matching plan component | 2456 |
| N-19 | disposition-confusion | Response proof carrying `hs_unattributed_key_set` | 446 |
| N-20 | plan-mismatch | Signature entry citing a group absent from the plan | 562 |
| N-21 | non-deterministic-encoding | Tagged `COSE_Sign1` (CBOR tag 18) | 1625 |
| N-22 | response-binding | Response proof whose `cti` is not the request ID | 374 |

### Notes on individual negatives

- **N-1 / N-2 (type confusion, both directions).** A proof and a credential are
  disjoint by construction: different protected `typ`, different signing keys
  (issuer key versus `cnf`-bound proof key), different domain separator. Both
  directions deny before any claim is interpreted.
- **N-5 (component stripping).** The retained Ed25519 entry's `Sig_structure`
  still covers the hybrid `signature_plan`, so it cannot be reinterpreted under
  the standalone classical suite; the missing plan component denies
  independently. This is the vector that proves the weakly-non-separable
  property is enforced by both crypto and policy.
- **N-6 / N-7 / N-12 / N-13 (parser caps).** The proof-v1 caps are exact:
  1..8 signer groups, 1..2 components per group, 1..64 encoded bytes for
  `suite_id` and `kid`. Raising a cap is an incompatible profile revision.
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
