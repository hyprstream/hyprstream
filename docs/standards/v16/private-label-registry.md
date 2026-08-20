# Hyprstream private-use label registry — `hs-rpc-proof-v1`

Status: gate-2 input draft. This registry is the checked source of truth for
every private-use CWT claim key and COSE header parameter the v16 RPC
request-proof profile uses. It is a **repository-local** registry: it allocates
nothing in an IANA registry and requests no codepoint. It exists so the profile
freeze can be reviewed against a written allocation with a recorded collision
review, rather than against values embedded in code.

Companion artifacts: [`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl)
(normative structure), [`canonical-vectors.md`](canonical-vectors.md) (positive
and negative vectors), [`credential-profile.md`](credential-profile.md)
(the credential side of the type separation).

## 1. Allocation policy

Both allocations sit in ranges IANA designates **Private Use**, so no
registration is required and no future registered assignment can collide with
them:

| Registry | Private-use range | Values used here |
|---|---|---|
| CBOR Web Token (CWT) Claims | integer values less than −65536 (RFC 8392 §9.1) | −70001 … −70004 |
| COSE Header Parameters | integer values less than −65536 | −70100 … −70103 |

Additions to this registry are profile revisions: a new label is allocated only
with its CDDL, its presence/null rule, its `crit` placement, and at least one
positive and one negative vector. Labels are never reused with different
semantics; a retired label is tombstoned in this file.

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
| −70003 | `capnp_request_bytes` | `bstr` | Always present, never `null`. Zero length is legal (an empty request body). | The exact Cap'n Proto body bytes this proof signs. In a response proof the same key carries the exact response bytes (PROPOSED: reuse rather than a second key block; the `typ`/`hs_domain` pair already separates the profiles). This is the sole source of the dispatch method path — there is no second selector. |
| −70004 | `response_binding` | `response-binding` map / `null` | Always present. `null` exactly when the response is neither encrypted nor streamed. | The client's commitment to the response schema domain, response mode, and — for an encrypted response — the ML-KEM-768 recipient material. In a response proof it is the realized binding and must equal the request's map where both are present. |

Registered CWT claims used unchanged, with their registered semantics: `aud`
(3), `exp` (4), `iat` (6), `cti` (7), and `Nonce` (10, RFC 9711). No private
key is allocated for anything a registered claim already means.

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
- **Result:** −70001…−70004 are more than 69 000 below the most negative
  assignment and inside the Private Use range. No collision, and none is
  possible without IANA re-designating the range.

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
`ML-DSA-87` = −50 (RFC 9964). The profile uses only −19 and −49 and allocates
no private algorithm identifier; a negative vector proves −8 denies.

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
| `draft-ietf-cose-hpke`, `draft-ietf-cose-hpke-pq-pqt` | AD evaluation / adopted. | Watch item only. The `kem-recipient.alg` value is the PROPOSED private-use identifier −70200 until the PQ/T KEM identifiers register. |
| CWT `sid` mapping | No registered CWT claim key for `sid`. | The profile does not invent one; a session-bearing CWT is unavailable until a separate profile gate approves an encoding. See [`credential-profile.md`](credential-profile.md). |

Every draft revision above is re-verified at profile-freeze time; a watch item
is recorded, never depended upon.

## 6. PROPOSED values in this registry

Values below are concrete proposals for gate-2 approval, not inherited from the
controlling design:

1. `capnp_request_bytes` carries the exact **response** bytes in a response
   proof, reusing key −70003 rather than allocating a second private block.
2. The response-proof claims set reuses `aud` (3) for the canonical service
   domain and `cti` (7) as the echoed `request_id`, with `credential_hash`
   always `null`.
3. `response_binding` map structure and the `response-mode` values
   (1 unary cleartext, 2 unary encrypted, 3 stream setup).
4. `kem-recipient.alg` private-use value **−70200** as an interim ML-KEM-768
   identifier.
5. `logical_signer_group` numeric ceiling of 255.
6. `aud` size ceiling of 253 bytes and the 1 MiB body-bytes / 2 MiB
   total-object caps recorded in the CDDL.
