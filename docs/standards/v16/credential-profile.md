# Hyprstream credential profile — JWT and CWT access tokens

Status: **FROZEN** by the accepted Gate-2 vote (v16 §19, 2026-08-19). This
document freezes the credential side of the v16 dispatch profile: the claims
table, the credential/session identifier rules, and the revocation semantics
that the request-proof profile cross-references. **v16 credentials are
Reusable-only** (operator decision 2026-08-20, `DECISION-defer-oneshot-credentials`):
there is no credential use-profile field, and `OneShotTransaction`/consume-once
semantics are deferred to a future amendment (§4). It is not production-closed:
the shared media-type registrations tracked in
[`private-label-registry.md`](private-label-registry.md) §5 remain open.

Companions: [`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl),
[`private-label-registry.md`](private-label-registry.md),
[`canonical-vectors.md`](canonical-vectors.md).

## 1. Two encodings, one credential model

The authorization server may issue a **JWT** access token or a **CWT** access
token. They are two encodings of one credential profile, not two authorization
models:

- JWT access tokens carry the RFC 9068 `at+jwt` header type.
- CWT access tokens carry the standard CWT content typing.
- Neither may carry the request-proof types
  (`application/vnd.hyprstream.proof+cwt`,
  `application/vnd.hyprstream.response-proof+cwt`). Type separation from the
  request proof is normative: a credential presented in the proof slot, or a
  proof presented as a credential, denies before any claim is interpreted
  (vectors N-1 and N-2).

A credential is signed by its issuer and never by a request-proof signer key. A
request proof is signed by keys the credential's `cnf` resolves to (or, with no
credential, by a self-asserted key set that grants nothing) and never by an
issuer key.

### 1.1 Confirmation-method encoding: hybrid credentials are JWT-only in v16

The `cnf` must resolve to the primary signer group's **suite ID + exact ordered
component keys** (§5). The two encodings express this differently:

- **JWT** carries `cnf.hs_signer_suite` = base64url(SHA-256(RFC 8949
  deterministic-CBOR `[suite_id, [ordered raw component public keys]]`)), a
  profile-defined RFC 8747 confirmation method that binds a signer-suite record of
  **any** component count — one key (classical) or two (hybrid Ed25519 + ML-DSA-65).
- **CWT** carries the standard RFC 8747 `cnf` (claim 8) as a **single `COSE_Key`**
  (member 1). One `COSE_Key` can pin exactly one key, so the CWT `cnf` binds a
  **classical (single-key)** primary group only.

**v16 defines no multi-key CWT confirmation method** — no label, no CBOR shape, is
allocated or reserved for it (the same deferral discipline this profile applies to
CWT `sid` in §3). Therefore, normatively:

- a **hybrid** proof credential MUST be an `at+jwt` (JWT) token using
  `cnf.hs_signer_suite`;
- an issuer MUST NOT issue a CWT credential whose `cnf` purports to bind a hybrid
  signer suite (a single `COSE_Key` cannot pin the hybrid group's two component
  keys), and a verifier MUST reject such a presentation: the single-key `cnf`
  resolves to a classical record that matches no group of the hybrid proof's plan,
  so it denies (no primary group resolves);
- a **classical** credential may be JWT **or** CWT; N-1's issuer-signed CWT with a
  single RFC 8747 `COSE_Key` `cnf` remains valid.

A future amendment may allocate and specify a multi-key CWT confirmation method
with its own canonical vectors; until then, CWT hybrid `cnf` is unavailable.

## 2. Claims table

Required semantic fields for authenticated dispatch:

| Field | JWT | CWT | Requirement | Semantics |
|---|---|---|---|---|
| Issuer | `iss` | `iss` (1) | REQUIRED, non-empty | The issuing node/authorization server. Scopes every identifier below. Empty issuer denies. |
| Subject | `sub` | `sub` (2) | REQUIRED, non-empty | The principal. Empty subject denies. |
| Audience | `aud` | `aud` (3) | REQUIRED | Exact service/resource audience (RFC 8707 resource indicator semantics). When it names a canonical service domain it uses the one shared `MAX_SERVICE_DOMAIN_BYTES` (128-byte) canonicalization rule (Gate-2 §19 #7), not a second identity rule. Wrong audience denies. |
| Expiry | `exp` | `exp` (4) | REQUIRED | Credential expiry. Bounds the proof: a proof `exp` MUST NOT exceed credential or session expiry — a proof cannot outlive the authority it presents, even when both are unexpired at the current clock (vector N-50; enforced by the gate's authenticated-context loop and the §12 causality inventory). |
| Issued at | `iat` | `iat` (6) | REQUIRED | Issuance time. |
| Credential ID | `jti` | `cti` (7) | REQUIRED | Unique credential instance identifier. JWT uses a text `jti`; CWT uses a byte-string `cti`. |
| OAuth client | `client_id` | — | REQUIRED for `at+jwt` (JWT only) | The OAuth client the access token was issued to (RFC 9068 §2.2.1); a non-empty string. This profile adopts the RFC 9068 `at+jwt` type, so its registered `client_id` is REQUIRED — no Hyprstream deviation and no private-use key. There is **no** CWT mapping: a CWT credential is not an RFC 9068 JWT and MUST NOT carry an invented `client_id` label. The fixtures use `hyprstream-oauth-client-1`. |
| PoP binding | `cnf` | `cnf` (8) | REQUIRED for dispatch | RFC 7800 / RFC 8747 proof-of-possession key binding. Resolves to exactly one signer-suite record identified by suite ID + the exact ordered component keys of the primary signer group (the verifier additionally consistency-checks principal and enrollment epoch from its own enrollment record; §5). **CWT `cnf` is a single RFC 8747 `COSE_Key` and binds a classical (single-key) primary group only.** A **hybrid** (multi-key) primary group has no v16 CWT confirmation method, so hybrid credentials are **`at+jwt` (JWT) only** — see §1.1. |
| Tenant | `tenant` | −70005 | REQUIRED | Verified tenant/Casbin domain: a **non-empty string** that is **not** the wildcard `*`. Missing, empty, wildcard, or non-string tenants deny (one predicate shared by the gate and checker, applied to both the JWT `tenant` and the CWT −70005 across every credential). CWT uses the integer private-use key −70005 (Gate-2 §19 #10); the text name is JWT-only. |
| Clearance | `clearance` | −70006 | REQUIRED for authenticated dispatch | Authority-issued MAC clearance (`Level`/`Compartments`; §8). Assurance is **not** carried here: it is derived from verified key material. A claim may restrict assurance but can never raise it. A clearance-less credential denies. CWT uses the integer private-use key −70006. |
| Scope / capability | `scope` | `scope` (9) | Where applicable | Least-authority TE/capability ceiling. Consumers deriving authority from scopes fail closed when the claim is absent. |
| User session | `sid` | — | REQUIRED for OIDC/user session credentials; otherwise absent | See §3. CWT has no `sid` mapping; user-session credentials are JWT-only. |
| Workload session | `workload_session_id` | −70007 | Only for an issuer-managed workload credential family | See §3. CWT uses the integer private-use key −70007; the text name is JWT-only. |
| Actor chain | `act` | `act` | Where applicable | RFC 8693 delegation chain; each hop composes into the clearance meet. |

Unknown critical claims, malformed identifiers, empty issuer, empty subject,
wildcard tenant, unacceptable algorithm, and wrong audience deny. No error
converts to an unauthenticated disposition: credential absence is the only
source of `Unauthenticated`, and it requires affirmative observation that the
credential slot is absent on a transport that permits absence.

Internal code normalizes both encodings to a typed
`CredentialId { issuer, value }` without stringifying a CWT byte string into an
ambiguous namespace.

## 3. Identifier rules

### 3.1 Credential ID (`jti` / `cti`)

The credential ID identifies one issued token instance. Its key is
**`(iss, jti)`** or **`(iss, cti)`** — never a bare identifier. Issuers satisfy
the underlying JWT/CWT uniqueness requirements; verifier stores namespace by
issuer as defense in depth.

Uses, exhaustively (v16, Reusable-only):

1. individual credential revocation;
2. audit correlation; and
3. cache invalidation.

The credential ID is never a request replay key in v16: request replay is always
keyed by the proof. (A single-use replay key tied to the credential ID returns
only with the deferred `OneShotTransaction` amendment; see §4.)

Seeing the same credential ID across many requests is expected and correct for
reusable credentials. The credential-ID store is a credential-revocation store,
not a general request replay store: request replay is keyed by the proof, not
the credential (see [`canonical-vectors.md`](canonical-vectors.md) and the
replay rules the proof profile cross-references).

### 3.2 User session ID (`sid`)

`sid` is the registered OIDC/JWT claim: an opaque identifier unique within an
issuer, grouping one user-agent authenticated session whose access-token
rotations have distinct credential IDs. Interactive/user session credentials
MUST carry it. Its key is `(iss, sid)`.

**CWT has no registered `sid` mapping.** This profile does not invent an ad hoc
CWT text or integer alias and allocates no private-use key for it: a
session-bearing CWT is unavailable until a separate profile gate approves an
encoding with canonical vectors. Non-session CWT service credentials remain
valid without it. This is a deliberate non-allocation, recorded in
[`private-label-registry.md`](private-label-registry.md) §5.

### 3.3 Workload credential families

A workload credential MUST NOT overload OIDC `sid`. When an issuer maintains a
real revocable workload credential family — one bootstrap or renewal
relationship producing several short-lived workload identity tokens — it may
carry a separately specified `workload_session_id`. A standalone service
credential with no such lifecycle omits it; no session is manufactured merely
to populate a claim.

The two session identifiers are disjoint types and disjoint wire namespaces.
Every session identifier is random, opaque, never reassigned, not derived from
a username, tenant, device identifier, network address, or credential ID, and
never exposed on a public error surface.

### 3.4 Session state

The authority stores at least: subject, tenant, session kind (interactive or
workload), creation and expiry times, active/revoked status, and a clearance
epoch, keyed by `(issuer, session identifier)`. The `created` time is a
deterministic integer, and every session-validation path requires it to be
temporally coherent with the verifier clock and the session expiry:
`created <= verifier_now < expires_at`. A session whose `created` is missing,
non-integer, in the future (`created > verifier_now`), or not strictly before its
own expiry (`created >= expires_at`) is not admissible. Like the rest of session
state, `created` is authoritative off-wire state, never a credential/wire claim.

### 3.5 Credential kind and `sid` coherence

`sid` presence is unambiguous by credential kind (a classification aligned with
the issuer's `IssueTokenProfile` enum, not a new wire claim):

- **`user-session`** — an interactive OIDC session token; **MUST** carry `sid`
  (§3.2), and a proof bound to it is additionally bounded by the authoritative
  session expiry (§3.4).
- **`rfc8693` / `rfc7523`** — a **non-interactive** token-exchange or JWT-bearer
  token with a user subject and no interactive session; **MUST NOT** carry `sid`.
- **`service`** — a standalone service identity with a `service:`-prefixed subject
  and **no** `sid`.

A user credential is therefore never left with an ambiguous session profile: it is
either a `user-session` credential with `sid`, or an explicitly non-interactive
`rfc8693`/`rfc7523` credential without it. The gate enforces this coherence over
every credential fixture.

## 4. Credential use: Reusable-only (v16)

**v16 credentials are Reusable.** There is no credential use-profile field and
no consume-once path: the credential ID is never consumed, and replay admission
is always keyed by the proof — `(credential-bound primary signer-suite
thumbprint, request_id)` — so many fresh proofs may use one credential.

| Use | Admission behavior |
|---|---|
| Reusable (the only v16 use) | The credential ID is **not** consumed; replay admission is keyed by the proof. |

**Forward-compatibility rule.** The absence of any use-profile field means
Reusable — no field is written or expected now. If a future amendment reintroduces
one-shot credentials, it will allocate a dedicated wire claim at that time, and a
credential carrying no such claim will continue to mean Reusable.

**Deferred: `OneShotTransaction`.** Single-use credentials and their atomic
consume-`(iss, cti)` replay path are **deferred to a future profile amendment**
(operator decision 2026-08-20). v16 allocates **no** claim key for a use-profile;
in particular it does not reserve or allocate one. When exactly-once mutation
semantics are needed, they return as a scoped amendment with its own claim
allocation and review.

Reusable credentials do not by themselves provide application idempotency.
Exactly-once business effects are a separate, explicitly declared method property
backed by an idempotency/result ledger whose lookup binds the retrying principal
— not the request-proof replay cache.

## 5. Credential binding to the request proof

- The proof's `credential_hash` claim (−70001) is SHA-256 over the **exact**
  presented credential bytes. The token bytes travel outside the signed
  payload, in the authorization slot; their hash is signed.
- On a trust-store or cache-hit path that omits token bytes, the verifier
  resolves one unambiguous verified credential record and compares its stored
  token hash. Hash absence is permitted only when no credential is presented.
- Both stripping directions deny: a presented credential with a null signed
  hash (vector N-15), and an absent credential with a non-null signed hash.
- **Primary-group selection (C3).** Plan `group_id`s are arbitrary ascending
  integers and do not name the primary group, so the primary is selected by
  content: the **primary logical signer group is the plan group whose suite ID
  and exact component keys equal the `cnf`-resolved signer-suite record.** Exactly
  one plan group matches; if none matches, or more than one does, the proof
  denies.
- **Scoped exact-key rule.** The exact-key denial applies to the **primary group
  only**: a component signature in the primary group verifying under any key not
  pinned in the `cnf`-resolved record denies — even a key validly enrolled to the
  same principal ("resolves to the same principal" is a consistency check, never
  the binding mechanism). Every **additional (approver) logical group is bound to
  its own enrolled keys** from that group's enrollment record, NOT the client's
  `cnf` record; approver signatures are not checked against the `cnf`-bound keys.
  So a `TokenBoundAndApproved` proof (e.g. P-5) has one interoperable disposition:
  the primary verifies against `cnf`, each approver against its own enrollment,
  and none over-rejects the other. That approver **enrollment record is
  authoritative, off-wire state** (never a credential/wire claim): it is keyed by
  the group's **cryptographic content** — a signer-suite thumbprint over its
  `suite_id` + ordered public keys, the same content-bound discipline as `cnf` and
  the replay namespace (§7.1), never the group's `group_id`/`kid` labels — and is
  resolved by **recomputing** that thumbprint from the record's own suite/keys, not
  by trusting any stored value. A conforming verifier admits an approver group only
  when its enrollment resolves to an **active, unexpired** record with the
  **`approver` role**, a `tenant` coherent with the credential, and an enrollment
  epoch; an unknown, tampered, key/suite-mismatched, inactive, expired,
  cross-tenant, or wrong-role enrollment denies. Being merely *different from `cnf`*
  is insufficient — an unenrolled group is not an authorized approver.
- **Primary enrollment authority (T1).** The `cnf`-bound primary group is itself
  backed by an **authoritative, off-wire primary enrollment record** (never a
  credential/wire claim), resolved by the same content-bound discipline as the
  approver record: the record is located by **recomputing** its signer-suite
  thumbprint over its own `suite_id` + ordered public keys and matching it to the
  `cnf`-resolved signer suite — labels are never trusted. A conforming verifier
  requires an **active, unexpired** record with the **`primary` role** whose
  `tenant` and `principal` equal the credential's `tenant` and subject, and a
  non-negative integer enrollment epoch. The **authenticated replay-namespace
  `enrollment_epoch` (§7.1) is taken from this resolved record, not from any wire
  or vector value**, so the published authenticated thumbprint is reproducible only
  from authoritative primary state; an unknown, tampered, key/suite-mismatched,
  inactive, expired, cross-tenant, wrong-principal, or wrong-role primary record
  denies. Primary and approver records are distinct, and the same resolved public
  key presented as both a `cnf` primary and an approver under different labels is
  denied (§7.1, content-identity uniqueness).
- Presenting a credential never leaves a proof in the unattributed branch: with
  a valid credential, the proof MUST be `cnf`-bound, and
  `hs_unattributed_key_set` is forbidden (vector N-10f).

## 6. Revocation semantics

| Operation | Effect |
|---|---|
| Revoke `CredentialId` = `(iss, jti/cti)` | Reject that credential; evict every handle derived from it. |
| Revoke `SessionKey` = `(iss, sid \| workload_session_id)` | Reject every credential and handle carrying that session ID; terminate or revalidate associated streams and continuations; prevent refresh within the session. |
| Disable subject or tenant | A separate authority operation that may revoke multiple sessions. |
| Expiry of token or session | The same rejection behavior, with no unauthenticated downgrade. |

Ordering is normative: revocation publication MUST precede derived-handle
eviction, so new verification fails before stale authority is flushed.

Long-lived streams and continuations MUST additionally revalidate their subject
handle at a reviewed maximum re-authorization interval even absent a revocation
event, so a stream whose setup proof is long past cannot outlive its authority
between revocation checkpoints.

## 7. Validation obligations

A conforming verifier is checked against at least:

1. `jti` and `cti` normalize without collisions and are issuer-scoped;
2. reusable tokens succeed across fresh proofs without consuming the
   credential ID (v16 credentials are Reusable, so the credential ID is never a
   replay key);
3. tokens from one user session have distinct credential IDs and one stable
   `sid`;
4. credential revocation affects one token; session revocation affects every
   associated token, handle, stream, continuation, and refresh;
5. a standalone workload token without a session is valid without a fabricated
   `sid`;
6. expired, revoked, wrong-audience, wrong-tenant, wrong-sender,
   clearance-less, or malformed tokens deny with no unauthenticated fallback;
7. credential bytes substituted after proof signing fail hash binding; and
8. a token whose primary signer suite does not match `cnf` denies.

## 8. Frozen Gate-2 dispositions

The claims table above restates the controlling design; the two items below are
frozen by the accepted Gate-2 vote (v16 §19, 2026-08-19):

1. **Amended (§19 #10).** `tenant`, `clearance`, and `workload_session_id` are
   carried under those exact **text** names in JWT, and under the allocated
   **integer** private-use CWT claim keys −70005, −70006, and −70007
   respectively in CWT (see [`private-label-registry.md`](private-label-registry.md)
   §2.1). The earlier proposal to keep CWT on text keys is superseded: the vote
   allocated the integer keys, and text names are retained only in JWT.
2. **Passed as written (value 11).** `clearance` encodes the `Level` and
   `Compartments` axes only; assurance is structurally absent from the
   credential rather than present-and-ignored, and is derived from verified key
   material at the boundary.

   **Frozen wire grammar.** The `clearance` value (JWT text `clearance`, CWT
   integer key −70006) is the same semantic two-element array in both encodings:

   ```
   clearance = [ level, compartments ]
   ```

   - `level` is a uint with the frozen v16 mapping `0 = Public`, `1 = Internal`,
     `2 = Confidential`, `3 = Secret` (the `Level` discriminants; §8 lattice).
   - `compartments` is an array of compartment **bit indices**, each a uint
     `0..63`, **strictly ascending and unique** (empty is allowed). It is the
     credential wire projection of the versioned `InitialLabelMap` /
     `CompartmentSet(u64)` — **not** a list of names, and **not** a bitmask
     integer.
   - The outer array has **exactly two** elements. Unknown levels, out-of-range
     compartments, duplicates, descending or otherwise non-canonical order, names,
     extra elements, and **any assurance field or value** deny.
   - **Assurance is never issuer-asserted** on the credential wire: it is derived
     only from verified key material during credential admission and clamps the
     resulting runtime security context; a credential can restrict assurance
     through its clearance level but can never raise it.

   The CDDL freezes the shape and the level/compartment domains
   (`credential-clearance`); `validate_profile.py` additionally enforces the
   strict-ascending order and uniqueness numerically (the pinned pycddl cannot),
   validates every shipped credential clearance, and proves each denial with a
   re-signed counter-proof. The fixture value `[2, [5, 7]]` (Confidential; bit
   indices 5 and 7) conforms.

   **WS-B seam.** A serializer that emits `Option<SecurityLabel>` directly cannot
   be the credential wire type: `SecurityLabel` carries the `assurance` axis, which
   is forbidden on the wire here. The credential producer/consumer MUST project to
   and from this two-axis `[level, compartments]` form and derive assurance from
   verified key material, never from the credential.
