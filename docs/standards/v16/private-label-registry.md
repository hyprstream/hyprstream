# Hyprstream private-use label registry — `hs-rpc-proof-v1`

Status: **FROZEN** by the accepted Gate-2 vote (v16 §19, 2026-08-19). This
registry is the checked source of truth for every private-use CWT claim key,
COSE header parameter, and COSE algorithm identifier the v16 RPC request-proof
and credential profiles use. It is a **repository-local** registry: it allocates
nothing in an IANA registry and requests no codepoint. The two `vnd.hyprstream`
media types (§5) still require IANA vendor-tree registration before production
close, and −70200 stays project-private.

Companion artifacts: [`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl)
(normative structure), [`canonical-vectors.md`](canonical-vectors.md) (positive
and negative vectors), [`credential-profile.md`](credential-profile.md)
(the credential side of the type separation).

## 1. Allocation policy

All allocations sit in ranges IANA designates **Private Use**, so no
registration is required and no future registered assignment can collide with
them:

| Registry | Private-use range | Values used here |
|---|---|---|
| CBOR Web Token (CWT) Claims | integer values less than −65536 (RFC 8392 §9.1) | −70001 … −70004 (proof claims, §2); −70005 … −70007 (credential claims, §2.1) |
| COSE Header Parameters | integer values less than −65536 | −70100 … −70103 (§3) |
| COSE Algorithms | integer values less than −65536 | −70200 (`hs-kem-ml-kem-768-v1`, §3.1) |

The CWT-claims allocations occupy two disjoint blocks in the **one** CWT Claims
private-use namespace: the proof-CWT claim keys −70001…−70004 and the
credential-CWT claim keys −70005…−70007 (Gate-2 §19 amendment 10). They never
overlap; a proof and a credential are distinct claim sets, and a key allocated
in one is unknown to the other.

Additions to this registry are profile revisions: a new label is allocated only
with its CDDL, its presence/null rule, its `crit` placement, and at least one
positive and one negative vector. Labels are never reused with different
semantics; a retired label is tombstoned in this file. The value −70050 is
deliberately left **unallocated**; it is the unknown-claim-key negative vector
(N-8) and MUST NOT be assigned.

## 2. CWT claim keys (−70001 … −70004)

All four claims appear in **both** the request-proof and response-proof claims
sets. Every listed claim is always present; an inapplicable value is CBOR
`null`. An absent listed claim, or any unlisted claim key, denies — CWT has no
critical-claims mechanism, so the closed claim set is the profile's structural
field closure.

| Key | Name | Type | Presence / null rule | Semantics |
|---|---|---|---|---|
| −70001 | `credential_hash` | `bstr .size 32` / `null` | Always present. `null` exactly when no credential is presented. A presented credential with `null` denies; an absent credential with a non-null value denies. | SHA-256 over the exact presented credential bytes. Binds the proof to one credential instance. On a cache-hit path that omits token bytes, the verifier resolves one unambiguous verified credential record and compares its stored token hash. |
| −70002 | `capnp_schema_id` | `uint` | Always present, never `null`. | Cap'n Proto 64-bit type ID of the signed body's root type: the request root type in a request proof, the response root type in a response proof. Compatible append-only schema evolution preserves the ID; an incompatible change requires a new ID and a migration gate. |
| −70003 | `capnp_body_bytes` | `bstr` | Always present, never `null`. Zero length is legal (an empty body). | The exact Cap'n Proto body bytes this proof signs (Gate-2 §19 amendment 1: the claim is named `capnp_body_bytes` for both directions). In a request proof it is the request body; in a response proof the same key carries the exact response bytes — the `typ`/`hs_domain` pair already separates the profiles (Gate-2 value 2). This is the sole source of the dispatch method path — there is no second selector. |
| −70004 | `response_binding` | `response-binding` map / `null` | Always present. `null` exactly when the response is neither encrypted nor streamed. | The client's commitment to the response schema domain and the two orthogonal axes `response_kind` (unary / stream_setup) and `protection_mode` (cleartext / encrypted), plus — for an encrypted response — the ML-KEM-768 recipient material (Gate-2 §19 amendments 3 and 4). The closed 4-key map is `{1 root_type_id, 2 response_kind, 3 protection_mode, 4 kem_recipient-or-null}`; the recipient is non-null iff `protection_mode` is encrypted. In a response proof it is the realized binding and must equal the request's map where both are present. |

Registered CWT claims used unchanged, with their registered semantics: `aud`
(3), `exp` (4), `iat` (6), `cti` (7), and `Nonce` (10, RFC 9711). No private
key is allocated for anything a registered claim already means.

## 2.1 Credential CWT claim keys (−70005 … −70007)

Gate-2 §19 amendment 10 allocates integer private-use CWT claim keys for the
three Hyprstream-specific credential claims, so a **CWT** access token encodes
them as integers rather than text. The text names are retained **only in JWT**,
whose claim names are strings by construction. These keys live in the credential
claims set (see [`credential-profile.md`](credential-profile.md)), not in the
proof claims set, and are the next collision-free block after the proof claims.

| Key | Name (JWT text / CWT integer) | Type | Requirement | Semantics |
|---|---|---|---|---|
| −70005 | `tenant` | `tstr` | REQUIRED for authenticated dispatch | Verified tenant / Casbin domain. Missing, empty, or wildcard denies. |
| −70006 | `clearance` | array | REQUIRED for authenticated dispatch | Authority-issued MAC clearance carrying the `Level` and `Compartments` axes only; assurance is structurally absent (Gate-2 value 11) and derived from verified key material. |
| −70007 | `workload_session_id` | `bstr` | Only for an issuer-managed workload credential family | Distinct from OIDC `sid`, which has no CWT mapping and remains JWT-only. |

These are credential claim keys: a proof CWT that carries any of −70005…−70007
denies (they are not in the proof's closed claim set), and a credential CWT that
carries a proof claim key (−70001…−70004) is equally malformed.

## 3. COSE header parameters (−70100 … −70103)

| Label | Name | Type | Bucket | Presence rule | Semantics |
|---|---|---|---|---|---|
| −70100 | `hs_domain` | `tstr` | body protected (`COSE_Sign`); the single merged protected bucket (`COSE_Sign1`) | Always present; always in `crit` | Application cryptographic domain separator. Exactly `hs-rpc-request-proof-v1` for request proofs and `hs-rpc-response-proof-v1` for response proofs. Deliberately redundant with `typ`: `typ` types the complete COSE object for media-type dispatch, `hs_domain` separates domains inside every `Sig_structure`. Both are protected, both must match, neither may be removed as a simplification of the other. |
| −70101 | `hs_signature_plan` | `signature-plan` array | body protected; merged bucket for `COSE_Sign1` | Always present; always in `crit` | The signed canonical description of every logical signer group and every component expected from that group. Placed in the headers, not the claims map, so the COSE layer can validate signature shape, critical parameters, and key selection before interpreting the CWT payload. A plan moved into the claims map, or omitted from the body protected headers, denies. |
| −70102 | `hs_logical_signer_group` | `uint` | per-signature protected (`COSE_Sign`); merged bucket (`COSE_Sign1`) | Always present on every signature entry; always in that bucket's `crit` | The `group_id` of the plan group this signature entry belongs to. Component signatures of one hybrid suite share one group and count as one logical signer. A signature cannot be reassigned to another group even when its key is otherwise enrolled. |
| −70103 | `hs_unattributed_key_set` | `COSE_KeySet` (RFC 9052 §7) | body protected; merged bucket for `COSE_Sign1` | Present exactly when no credential is presented; forbidden in credential-bound proofs and in every response proof. In `crit` exactly when present | The self-asserted public component keys of an unattributed signer, in the plan's component order. Proves only internal proof consistency: it grants no identity, role, clearance, tenant, or assurance. Every element must parse, be understood, and match exactly one plan component; a malformed, unknown, duplicate, surplus, private, reordered, or mismatched key denies the complete proof rather than being ignored. |

Exact `crit` sets (ascending numeric-label order, entries unique, every listed
label occurring in the same protected bucket):

| Structure | Bucket | Disposition | `crit` |
|---|---|---|---|
| `COSE_Sign` | body protected | authenticated | `[-70101, -70100]` |
| `COSE_Sign` | body protected | unattributed | `[-70103, -70101, -70100]` |
| `COSE_Sign` | per-signature protected | either | `[-70102]` |
| `COSE_Sign1` | merged protected | authenticated | `[-70102, -70101, -70100]` |
| `COSE_Sign1` | merged protected | unattributed | `[-70103, -70102, -70101, -70100]` |

Registered COSE header parameters used unchanged: `alg` (1), `crit` (2), `kid`
(4), and `typ` (16, RFC 9596). The content type parameter (3) is deliberately
omitted: it describes only the payload, and no separate media type for the raw
claims-set payload is registered.

## 3.1 COSE algorithm identifier (−70200)

Gate-2 §19 amendment 5 allocates and versions one project-private COSE
algorithm identifier for the encrypted-response KEM recipient:

| Value | Suite ID | Meaning | Where used |
|---|---|---|---|
| −70200 | `hs-kem-ml-kem-768-v1` | ML-KEM-768 key encapsulation, v1 | The `alg` slot (key 1) of `kem-recipient` in an encrypted `response_binding` |

The value is **REQUIRED exactly**: an encrypted `response_binding` whose
`kem-recipient.alg` is anything other than −70200 denies. It is allocated in
the COSE Algorithms **Private Use** range (integer values less than −65536) and
stays project-private. When `draft-ietf-cose-hpke-pq-pqt` registers a COSE PQ/T
KEM identifier, adopting it is an explicit **incompatible profile revision**
(new suite version, new vectors), never a silent substitution of the private
value. Nothing in this profile depends on that draft today.

## 4. Collision review

Reviewed against the live IANA registries on 2026-08-10. Registry entries
checked:

**CBOR Web Token (CWT) Claims registry**
(<https://www.iana.org/assignments/cwt/cwt.xhtml>, established by RFC 8392).

- Registration procedures for integer keys: −256…255 Standards Action;
  −65536…−257 Specification Required; 256…65535 Specification Required;
  greater than 65535 Expert Review. Integer values **less than −65536 are
  reserved for Private Use**, which is the range this profile uses.
- Assigned negative keys at review time: −261 `globalplatform_component`,
  −260 `hcert`, −259 `EUPHNonce`, −258 `EATMAROEPrefix`, −257 `EAT-FDO`;
  −256…−1 reserved/unassigned. The most negative assigned key is **−261**.
- Positive keys checked for the registered claims this profile reuses: 3 `aud`,
  4 `exp`, 6 `iat`, 7 `cti`, 8 `cnf`, 9 `scope`, 10 `Nonce` (RFC 9711).
- **Result:** the two allocated blocks −70001…−70004 (proof) and −70005…−70007
  (credential) are contiguous, mutually disjoint, and more than 69 000 below the
  most negative assignment, inside the Private Use range. No collision within
  either block, none between them, and none against any registered claim; none
  is possible without IANA re-designating the range. −70050 is intentionally
  left unallocated (the N-8 unknown-claim-key vector).

**COSE Header Parameters registry**
(<https://www.iana.org/assignments/cose/cose.xhtml>).

- Registration procedures for integer labels: less than −65536 **Private Use**;
  −65536…−1 delegated to the *COSE Header Algorithm Parameters* registry
  (labels there are scoped to an algorithm, not global); 1…255 Standards Action
  with Expert Review; 256…65535 Specification Required; greater than 65535
  Expert Review.
- Assigned labels checked: 1 `alg`, 2 `crit`, 3 `content type`, 4 `kid`, 5 `IV`,
  6 `Partial IV`, 7 `counter signature`, 9 `CounterSignature0`, 10
  `kid context`, 11 `Countersignature version 2`, 12 `Countersignature0
  version 2`, 16 `typ` (RFC 9596).
- **Result:** −70100…−70103 are in the Private Use range and below the
  algorithm-scoped delegation window entirely. No collision. Choosing labels
  below −65536 also means the profile never depends on which algorithm is in
  use to interpret its own header parameters.

**COSE Algorithms registry** (same page). Checked: `Ed25519` = −19,
deprecated polymorphic `EdDSA` = −8, `ML-DSA-44` = −48, `ML-DSA-65` = −49,
`ML-DSA-87` = −50 (RFC 9964). Registration procedure: integer values less than
−65536 are **Private Use**. The profile uses signature algorithms −19 and −49
(both registered) and allocates exactly one private algorithm identifier,
`hs-kem-ml-kem-768-v1` = −70200 (§3.1), for the KEM recipient. **Result:**
−70200 is inside the Private Use range, more than 69 000 below the most negative
registered algorithm, and cannot collide without IANA re-designating the range.
A negative vector proves −8 denies.

**COSE Key Types / Key Type Parameters / Elliptic Curves** (same page).
Checked: `OKP` = 1 with `crv` = −1, `x` = −2, `d` = −4; `AKP` = 7 with
`pub` = −1, `priv` = −2 (RFC 9964); elliptic curve `Ed25519` = 6. The
unattributed key set uses exactly these registered key representations.

**Media Types registry**
(<https://www.iana.org/assignments/media-types/media-types.xhtml>).
`application/cose-key-set` exists and applies to a key set transported as a
standalone representation; it is deliberately **not** placed on the embedded
`hs_unattributed_key_set` header value.

## 5. Pending external registrations and watch items

| Item | Status | Profile dependency |
|---|---|---|
| `application/vnd.hyprstream.proof+cwt` | Not yet registered. RFC 6838 vendor tree, `+cwt` suffix (RFC 9782); registrable without IETF action. | REQUIRED before the production profile freeze. The literal string is already normative in the CDDL and vectors. |
| `application/vnd.hyprstream.response-proof+cwt` | Not yet registered; same tree and suffix. | REQUIRED before the production profile freeze. |
| `draft-ietf-jose-pq-composite-sigs` (`COMPSIG-MLDSA65-Ed25519-SHA512`) | WG-adopted draft. | Watch item only. Adoption is an incompatible profile revision, not a suite-table substitution. Nothing here depends on it. |
| `draft-ietf-cose-hpke`, `draft-ietf-cose-hpke-pq-pqt` | AD evaluation / adopted. | Watch item only. The `kem-recipient.alg` value is the **frozen** project-private identifier `hs-kem-ml-kem-768-v1` = −70200 (§3.1); a registered PQ/T KEM value would arrive only via an incompatible profile revision. |
| CWT `sid` mapping | No registered CWT claim key for `sid`. | The profile does not invent one; a session-bearing CWT is unavailable until a separate profile gate approves an encoding. See [`credential-profile.md`](credential-profile.md). |

Every draft revision above is re-verified at profile-freeze time; a watch item
is recorded, never depended upon.

## 6. Frozen Gate-2 dispositions in this registry

The values below were the twelve originally-proposed items; they are now frozen
by the accepted Gate-2 vote (v16 §19, 2026-08-19). Where the vote amended a
value, the amendment is recorded here. None of these is PROPOSED any longer.

1. **Amended (§19 #1).** Key −70003 is named `capnp_body_bytes` — the neutral
   name replacing the original request-specific one — and carries the exact
   Cap'n Proto body bytes for both request and response proofs, reusing the one
   key rather than allocating a second private block.
2. **Passed as written.** The response-proof claims set reuses `aud` (3) for the
   canonical service domain and `cti` (7) as the echoed `request_id`, with
   `credential_hash` always `null` and `Nonce` absent.
3. **Amended (§19 #3).** `response_binding` is the closed 4-key map
   `{1 root_type_id, 2 response_kind, 3 protection_mode, 4 kem_recipient-or-null}`.
4. **Amended (§19 #4).** Two orthogonal enum axes replace the old fused mode:
   `response_kind = {1 unary, 2 stream_setup}` and
   `protection_mode = {1 cleartext, 2 encrypted}`; the recipient is non-null iff
   `protection_mode` is encrypted.
5. **Amended (§19 #5).** `kem-recipient.alg` is the versioned project-private
   value `hs-kem-ml-kem-768-v1` = **−70200**, enforced exactly (§3.1).
6. **Passed as written.** `logical_signer_group` numeric ceiling of 255.
7. **Amended (§19 #7).** `aud` size ceiling is **128** bytes, the shared
   `MAX_SERVICE_DOMAIN_BYTES` canonicalization constant, not a second identity
   rule. The 1 MiB body-bytes and 2 MiB total-object caps are unchanged.
8. **Amended (§19 #10).** `tenant`, `clearance`, and `workload_session_id` are
   allocated integer CWT claim keys −70005…−70007 (§2.1); the text names are
   retained only in JWT. Value 11 (clearance = `Level` + `Compartments` only) is
   preserved as passed.

Production-close conditions still open: IANA vendor-tree registration of the two
`vnd.hyprstream` media types (§5); −70200 stays project-private until an
incompatible profile revision adopts a registered COSE PQ/T KEM value.
