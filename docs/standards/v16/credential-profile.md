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

## 2. Claims table

Required semantic fields for authenticated dispatch:

| Field | JWT | CWT | Requirement | Semantics |
|---|---|---|---|---|
| Issuer | `iss` | `iss` (1) | REQUIRED, non-empty | The issuing node/authorization server. Scopes every identifier below. Empty issuer denies. |
| Subject | `sub` | `sub` (2) | REQUIRED, non-empty | The principal. Empty subject denies. |
| Audience | `aud` | `aud` (3) | REQUIRED | Exact service/resource audience (RFC 8707 resource indicator semantics). When it names a canonical service domain it uses the one shared `MAX_SERVICE_DOMAIN_BYTES` (128-byte) canonicalization rule (Gate-2 §19 #7), not a second identity rule. Wrong audience denies. |
| Expiry | `exp` | `exp` (4) | REQUIRED | Credential expiry. Bounds the proof: a proof `exp` MUST NOT exceed credential or session expiry. |
| Issued at | `iat` | `iat` (6) | REQUIRED | Issuance time. |
| Credential ID | `jti` | `cti` (7) | REQUIRED | Unique credential instance identifier. JWT uses a text `jti`; CWT uses a byte-string `cti`. |
| PoP binding | `cnf` | `cnf` (8) | REQUIRED for dispatch | RFC 7800 / RFC 8747 proof-of-possession key binding. Resolves to exactly one signer-suite record pinning the exact component keys, principal, and enrollment epoch. |
| Tenant | `tenant` | −70005 | REQUIRED | Verified tenant/Casbin domain. Missing, empty, or wildcard tenants deny. CWT uses the integer private-use key −70005 (Gate-2 §19 #10); the text name is JWT-only. |
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
epoch, keyed by `(issuer, session identifier)`.

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
- The proof's primary logical signer group MUST equal the suite and the exact
  component keys of the `cnf`-resolved signer-suite record. A component
  signature verifying under any key not pinned in that record denies — even a
  key validly enrolled to the same principal. "Resolves to the same principal"
  is an additional consistency check, never the binding mechanism.
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
