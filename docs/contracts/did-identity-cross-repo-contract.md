# Cross-repo identity contract: `Did` / did:web / did:plc / did:key / did:at9p

**Status:** settled (#578 merged). Revision history:
- rev 1 (#1097): `did:at9p` resolution semantics (§4.1/§5/§6) + atproto-OAuth-flow terms (§10, added).
- rev 2 (#1097): corrected **header, §8 (ownership rows rewritten/added — AS delivery state, PLC resolver, DPoP gap, HTTPS fail-closed, binding semantics), and §10** (present-tense AS claims → tracked language + #1112 dep; server-side `did:plc` resolution specified; `getServiceAuth` token authority + replay properties + known DPoP-binding gap; §10.4 fail-closed for native `did:at9p` assertions). §§1–7,§9 unchanged in rev 2.
- rev 3 (#1097 — **this revision**): §§2/8/9 qualified to admit the exchange's server-side `did:plc` arm and point its ownership at **#1119**; `lxm` NSID `ai.hyprstream.identity.exchangeUcan` pinned, endpoint restated as **XRPC/Lexicon** (supersedes the consumer spec's Cap'n Proto shape — noted explicitly, consumer doc revised in parallel); §10.2 reworded as the **atproto-account→UCAN exchange** (`did:at9p` login/assertion language reserved for §10.3/§10.4); `aud` constrained to the host's atproto-compatible `did:web` service DID.
**Canonical owner:** `hyprstream` — `crates/hyprstream-rpc/src/identity.rs` (`Did`).
**Source issues/PRs:** #563 (design + sign-off) · #578 (def + field-level `$domainType`) · #579 (resolver + assurance, PQC pathway) · #1097 (`did:at9p` amendment — resolution + OAuth-flow + exchange terms, rev 3) · #1119 (atproto-account→UCAN exchange endpoint — server-side `did:plc` arm + `ai.hyprstream.identity.exchangeUcan` lexicon + jti one-time-use — **owns the exchange impl, open**) · #908 (`ai.hyprstream.*` lexicon namespace) · #1112 (authenticated `com.atproto.*` XRPC surface — **open** dependency of the flow) · #1113 (atproto OAuth AS conformance profile — **open**; `sub` scope excludes `did:at9p`) · #1114 (HTTPS `did:at9p` verification — **open**) · #948 (OAuth conformance spike — **open**).
**Consumers:** `www-cyberdione-ai` frontend (`src/api/atproto.ts`, MR !18 / #2 / #5) — mirrors this shape, does not re-derive. Security/MAC epic #547 (#548/#549) — consumes the #579 resolver. **Sibling spec:** `www-cyberdione-ai` `docs/contracts/atproto-identity-exchange.md` (DID→UCAN exchange + `did:at9p` binding record) — §10 mirrors its terms and MUST NOT contradict it. *Supersession note (rev 3):* that spec's earlier Cap'n Proto endpoint shape cannot carry a Lexicon `lxm` and is superseded by the XRPC `ai.hyprstream.identity.exchangeUcan` method (per #1119); the consumer doc is being revised in parallel to match, so the two remain consistent. That doc owns the *exchange spec*; this doc owns the `Did` shape, resolution posture, and `Assurance` enum it builds on.

This doc anchors the hyprstream ↔ www-cyberdione-ai identity boundary. **hyprstream owns the representation; everyone else mirrors it.**

---

## 1. `Did` is an opaque identifier string — never a key container
A `Did` is a **branded string** holding **any** DID method: `did:web:…`, `did:plc:…`, `did:key:…`, `did:at9p:…`. It carries **no key material inline**. Keys are established by the method-specific resolution and verification path; they are never fields of the `Did` value.

- Rust (canonical): `pub struct Did(String)` — `$domainType("hyprstream_rpc::identity::Did")`, **bare `Text` on the wire** (W3C DID strings; no wrapper/nesting).
- TS (mirror): `type Did = string & { readonly __brand: 'Did' }` (brand a string; do not build an object).

## 2. Method predicates
Rust (`identity.rs`) provides `is_did_key` · `is_did_web` (`startsWith("did:web:")`) · `is_did_at9p` (`startsWith("did:at9p:")`) — there is **no** `is_did_plc` on the Rust `Did` type (hyprstream does not resolve `did:plc` in general — see §4/§9 — **except** the atproto-account→UCAN exchange endpoint (§10.2.1), which adds a server-side `did:plc` arm for that one path, per #1119). The `did:at9p:` literal is canonically owned by `DID_AT9P_PREFIX`; mirrors MUST use that exact, case-sensitive prefix. The TS mirror may additionally implement an `is_did_plc` predicate for its own resolution path.

## 3. Validation posture — lenient construct, strict at the boundary
- **Construction is lenient:** `Did::new(s)` / the TS constructor accept **any** string; no validation, no rejection. Enables gradual adoption + opaque pass-through.
- **Validation/verification is strict at the trust boundary:** resolution + key verification + admission. (hyprstream: the admission gate; frontend: at resolve + before trusting.) **Never** trust a `Did` on construction alone.

## 4. Resolution contract — method dispatches resolution
The DID **method** decides how authority is resolved:
- `did:web:host[:path]` → HTTPS `GET https://host/.well-known/did.json` (or `/<path>/did.json`).
- `did:plc:…` → PLC directory.
- `did:key:…` → **no document**; the key *is* the identifier (self-certifying).
- `did:at9p:<cid512>` → resolve the genesis capsule from an untrusted source, verify it, then resolve current authority through accepted update-chain state. It is a capsule method, **not** a DID-document method.

For the methods that resolve to a DID document (`did:web` / `did:plc`), the resolved **document shape is method-agnostic**:
```jsonc
{
  "verificationMethod": [
    { "id": "…#atproto", "type": "Multikey", "publicKeyMultibase": "z…" }
  ],
  "service": [
    { "id": "#atproto_pds", "type": "AtprotoPersonalDataServer",
      "serviceEndpoint": "https://pds.host" }      // HTTPS URL (scheme+host+port)
  ]
}
```
- **`#atproto_pds.serviceEndpoint` MUST be an HTTPS URL for BOTH `did:web` AND `did:plc`** — it is a *document-content* rule, independent of the DID method. (did:web's HTTPS *resolution* is a separate, unrelated thing.)
- `verificationMethod` keys are `type: "Multikey"`, `publicKeyMultibase` (z-base58btc).

### 4.1 `did:at9p` durable identity and resolution

- **Canonical shape:** exactly `did:at9p:<cid512>`, where `<cid512>` is the canonical base32 CIDv1 using the `at9p-capsule` multicodec and a 64-byte BLAKE3-512 multihash of the canonical genesis capsule bytes. The full string remains a bare `Did` on the wire. Consumers MUST NOT substitute a handle, node ID, service URL, classical DID alias, or current record digest for it.
- **Genesis acceptance:** capsule bytes from any locator are untrusted until the complete GATE succeeds in order: deterministic DAG-CBOR/schema validation and byte-for-byte canonical round-trip; claimed CID512/hash equality; then the pinned Hybrid composite signature (Ed25519 **and** ML-DSA-65) with the record context. Failure at any gate is a hard resolution error.
- **Current authority:** the identifier remains anchored to the genesis CID across rotation. Current keys and other authoritative fields come only from accepted-current state chained from that genesis: same subject CID512, pre-committed next hybrid key, predecessor digest linkage, strictly increasing epoch, valid pinned-Hybrid signature, freshness, and duplicity/rollback checks. Empty `nextKeyCommitments` makes the accepted state terminal. Consumers MUST NOT fall back to genesis or a caller-supplied predecessor after an accepted rotation.
- **Resolution result:** identity-key resolution returns the accepted primary Ed25519 + ML-DSA-65 subject-key pair with `PqHybrid` assurance. A missing locator/resolver, missing accepted-current state where currentness is required, malformed or expired record, fork/rollback alarm, or verification failure MUST fail closed; it MUST NOT become a successful `Unverified` or `Classical` result.
- **Services are claims, not liveness:** a GATE-verified genesis service entry is content-bound, but does not by itself prove currentness, reachability, live possession, or admission authority. In particular, a relay claim without an independently authenticated carrier endpoint MUST NOT be converted into a dial target. Consumers select only the requested, correctly typed service from accepted-current state and fail closed on missing or wrong-shaped entries.
- **Current implementation boundary:** peer-presented capsule admission and accepted-state verification are live. Locator-backed identity-key resolution is not yet wired, and the discovery resolver cannot derive a live iroh target because the capsule has no independent carrier `EndpointId`; both paths intentionally return an error. Consumers MUST preserve that fail-closed behavior until the canonical resolver supplies the missing authority.

## 5. Key encodings
- **did:key / Multikey (Ed25519):** multicodec `ed25519-pub` prefix `0xed01`, multibase base58btc (`z…`). Ed25519 `did:key` ⇄ 32-byte key is lossless (hyprstream: `did_key.rs`).
- **PQC (forward):** an algorithm-agile multikey (ML-DSA-65 + others) + ML-DSA verification methods are the #579 generalization. `Did` is already algorithm-agnostic (it's a string); only the multikey codec + VM extraction generalize.
- **did:at9p:** the capsule carries a bound hybrid subject key pair: a 32-byte Ed25519 public key plus a 1952-byte ML-DSA-65 public key. These are canonical DAG-CBOR byte strings inside the verified capsule/current record, not multikey text embedded in the DID.

## 6. Assurance enum (consumers that compute trust level)
Locked 1:1 with epic #547 / S1 — **same names + discriminants, no translation**:
```
enum Assurance { Unverified = 0, Classical = 1, PqHybrid = 2 }
```
- no resolvable verifying key → **Unverified** (fail-closed; never silent-elevate)
- Ed25519 only → **Classical**
- Ed25519 + ML-DSA-65 → **PqHybrid**

For `did:at9p`, `PqHybrid` is valid only after the GATE/accepted-state path establishes the bound pair. A consumer that cannot perform or call that resolution MUST reject operations requiring verified keys; it may store or transport the opaque DID string, but MUST NOT downgrade it to `Classical` or infer assurance from the prefix.

hyprstream resolver (#579): `resolve_identity_keys(&Did) -> { ed25519: Option, ml_dsa_65: Option, assurance: Assurance }`.

## 7. Custom transport service entries — frontend IGNORES
hyprstream publishes extra `service` entries (`type: "IrohTransport"` / `"QuicTransport"`) with **map** `serviceEndpoint`s (W3C DID Core §5.4). These are non-atproto-canonical hyprstream extensions for dialing. **The www frontend uses only `#atproto_pds` + `verificationMethod`** and ignores transport entries.

## 8. Ownership boundary
| Concern | Owner |
|---|---|
| `Did` representation, predicates, validation posture | **hyprstream** (canonical) |
| Resolver (`resolve_identity_keys`) + `Assurance`, including `did:at9p` GATE/accepted-state semantics | **hyprstream** (#579/#1097) |
| did:web/plc/key resolution in the frontend | **www-cyberdione-ai** — mirrors §1–§6 |
| `did:at9p` handling in the frontend | **www-cyberdione-ai** — mirrors the opaque string/predicate; delegates canonical resolution or fails closed per §4.1 |
| MAC clearance / OAuth→UCAN / perimeter | **epic #547** — consumes #579 |
| Custom transports (iroh/quic) | **hyprstream** — frontend ignores |
| atproto OAuth (did:plc / did:web) login — AS conformance profile | **hyprstream** (`services/oauth`) — **tracked, not delivered**: profile target in #1113, validation in #948; authenticated XRPC surface #1112 is an open dependency |
| Server-side `did:plc` issuer resolution (for the atproto-account→UCAN exchange) | **hyprstream** — required new `MethodDispatchResolver` arm (PLC base configurable, default `https://plc.directory`, §9); **tracked under #1119**. Until it lands, `did:plc` service-auth issuers fail closed (§10.2.1) |
| atproto-account→UCAN exchange endpoint (mint + verify) | **hyprstream** — XRPC/Lexicon method `ai.hyprstream.identity.exchangeUcan` (§10.2, #1119; under #908 `ai.hyprstream.*`); **www-cyberdione-ai** `atproto-identity-exchange.md` owns the spec (its earlier Cap'n Proto shape is superseded by this XRPC method). No token→DPoP-key binding on the `getServiceAuth` surface — known gap (§10.2.2) |
| External HTTPS verification of `did:at9p` (login assertion) | **OPEN — #1114** (shape undecided). **Until it lands, native `did:at9p` assertions FAIL CLOSED; §10.2 is NOT a verification path (§10.4)** |
| `did:at9p` binding record (`ai.cyberdione.identity.at9pBinding`) | **www-cyberdione-ai** (record write) + **hyprstream** (dual-direction verify, §10.3 — the only thing that connects the two subjects) |

**Rule:** mirror string + predicates + applicable DID-doc/capsule dispatch + assurance enum. Do **not** re-derive the representation, parse `did:at9p` as a DID document, fork the enum, or treat an unverified service claim as authority. Changes to this contract go through the canonical owner (hyprstream) and are relayed to consumers.

---

## 9. Resolved edge cases (frontend ↔ canonical)
- **Missing `#atproto_pds`:** **HARD-ERROR (fail-closed).** Do not fall back to another typed `serviceEndpoint` for the PDS.
- **Service precedence:** `#atproto_pds` (`type: AtprotoPersonalDataServer`) is the **sole atproto-canonical PDS** service; hyprstream emits it **first** (`did_document.rs`: "atproto PDS first, then legacy HyprstreamService"). **`HyprstreamService` is legacy** — prefer `#atproto_pds`, do not prefer it. `serviceEndpoint` is **origin-only** (scheme+host+port).
- **Path-based `did:web`:** enforce **`doc.id === requested DID`** (anti-spoof); **percent-encode the port** (`did:web:example.com%3A6791:users:alice` → `https://example.com:6791/users/alice/did.json`); path form uses `/<path>/did.json` (`.well-known` only when there is no path).
- **`did:plc` directory base:** **configurable, default `https://plc.directory`.** hyprstream does not resolve `did:plc` for general/frontend use (the frontend owns it); pin the default, allow override for self-hosted/dev/federation. The one in-tree exception is the atproto-account→UCAN exchange endpoint (§10.2.1, #1119), whose server-side `did:plc` arm consumes the same configurable base.
- **`did:at9p` aliases:** a `did:web` / `did:key` / `did:plc` alias does not replace the durable `did:at9p` identity. Treat the at9p identity as authoritative only when both the GATE-verified capsule and the classical DID document mutually attest the alias; one-way `alsoKnownAs` is insufficient.
- **Unsupported `did:at9p` resolution:** opaque storage/pass-through is allowed, but any operation that needs keys, assurance, current services, or authority MUST hard-error until the canonical at9p resolver is available.

---

## 10. atproto login flow — method dispatch (amended by #1097)

The resolution contract in §§1–9 governs how a `Did` is *resolved and verified*. This section governs a different question: **how a DID enters an atproto login session**, and is the formal `did:at9p` supersession requested by #1097. It mirrors — and MUST NOT contradict — the `www-cyberdione-ai` sibling spec `docs/contracts/atproto-identity-exchange.md` (canonical owner of the exchange *spec*); this doc owns the `Did` shape, resolution posture, and `Assurance` enum that spec builds on.

### 10.1 `did:plc` / `did:web` — standard atproto OAuth

Login for `did:plc` / `did:web` is **standard atproto OAuth** end-to-end: PAR (RFC 9126) + DPoP (ES256) + PKCE (`code_challenge` with `code_challenge_method=S256` on authorization requests, per the atproto OAuth profile) + session refresh, against any PDS, driven by a stock atproto client (e.g. `@atproto/oauth-client-browser`). The target hyprstream conformance profile is **tracked, not delivered** — it is the open work in **#1113** (AS profile: advertise/accept scopes `atproto` / `transition:generic`, RFC 8414/9728 `authorization_servers` self-reference, token `sub` = the account DID for the atproto-OAuth-supported methods `did:plc` / `did:web` **only** — #1113's scope excludes `did:at9p`, which never enters this flow), with end-to-end validation deferred to the **#948** conformance spike. The token it mints is only useful against an authenticated `com.atproto.*` XRPC surface that does not yet exist — **#1112** (XRPC read slice) is an explicit **open dependency** of this flow (#1113's acceptance test pairs with it). As of this writing the in-tree AS advertises configured generic scopes and sets `sub` to `pending.username`; treat the profile above as the contract target, not the current state. `did:plc`/`did:web` verification keys are **classical** (secp256k1 / P-256, ES256 DPoP) — an atproto protocol constraint, not a hyprstream choice.

### 10.2 The atproto-account→UCAN exchange (`did:plc` / `did:web` → host UCAN)

A caller who holds a standard atproto OAuth session (`did:plc` / `did:web`) obtains a host-scoped hyprstream UCAN through the **atproto-account→UCAN exchange**: the caller presents an atproto service-auth token (`com.atproto.server.getServiceAuth`) whose `aud` is **the host's atproto-compatible `did:web` service DID** (see §10.2.2 — never a `did:at9p`; upstream PDSes reject non-atproto audiences) and whose `exp` is short (minutes); the host resolves the caller's atproto DID **itself** (per §4, never trusting a client-supplied document/key — and, for `did:plc`, via the §10.2.1 server-side arm), verifies the token signature against the resolved classical verification key, and on success **mints a hyprstream UCAN** for a subject derived from the atproto DID. The exchange is owned by **#1119** (`ai.hyprstream.identity.exchangeUcan` XRPC/Lexicon endpoint, under #908). `did:at9p` is **not** a registered atproto DID method and does not appear in this exchange — `did:at9p` login/assertion handling is reserved for §10.3 (verified binding) and §10.4 (#1114, fail closed). Note: `getServiceAuth` accepts `aud` / `exp` / optional `lxm` (and mints a JWT carrying `iss` / `aud` / `exp` / optional `lxm` / `jti`) — it has **no** generic `scope` field and **no** `cnf`/DPoP-key member; the exact authority + replay posture this yields is specified in §10.2.2, not assumed.

#### 10.2.1 Issuer resolution — required work (server-side `did:plc` arm)

The host MUST resolve the issuer's atproto DID document itself (the strict-at-the-boundary posture of §3/§4). Today the in-tree `MethodDispatchResolver` has arms for `did:web`, `did:key`, and `did:at9p` only; **every other method — including `did:plc` — fails closed**. `did:web` issuers are therefore resolvable for this exchange as-is; `did:plc` issuers are **rejected until a server-side `did:plc` resolver arm lands**. Because the login page's primary user population is `did:plc`, that arm is **required implementation work** for this exchange, not optional:

- A `did:plc` arm MUST be added to `MethodDispatchResolver` (querying the PLC directory; **base URL configurable, default `https://plc.directory`**, mirroring §9's `did:plc` directory-base rule; allow override for self-hosted/dev/federation), returning the resolved `verificationMethod` keys so `resolve_identity_keys` can derive `Assurance::Classical`.
- Until that arm exists, the exchange endpoint MUST reject `did:plc`-issued service-auth tokens at admission (fail closed) rather than accepting a client-supplied PLC document — it MUST NOT violate the "never trust a client-supplied document/key" rule to unblock the common case.
- This arm is tracked, together with the exchange endpoint itself, under **#1119** (which owns the full atproto-account→UCAN exchange: server-side `did:plc` resolution, the `ai.hyprstream.identity.exchangeUcan` lexicon, `jti` one-time-use, and the no-sender-binding posture).

This is the one in-tree exception to the general "hyprstream does not resolve `did:plc`" rule (§§2/8/9 now state that rule with this exchange as the named exception, and point its ownership at #1119); it is scoped to the exchange endpoint, not a general obligation.

#### 10.2.2 Service-auth token authority & replay properties — known gap

The exchange does **not** assert a sender-binding guarantee, because the atproto service-auth artifact provides none:

- **Fields actually available:** `iss` (**REQUIRED** — the caller's atproto DID, `did:plc`/`did:web`; it identifies whose DID document the host resolves for signature verification — reject tokens without it), `aud` (MUST equal the host's atproto-compatible `did:web` service DID — reject otherwise; never a `did:at9p`, which upstream PDSes reject as a non-atproto audience), `exp` (short, minutes), optional `lxm` (a lexicon method NSID the token authorizes), and `jti` (a unique token id). There is **no `cnf` member** and **no DPoP-key binding** in the JWT.
- **Concrete `lxm` pin (literal NSID):** the host MUST require `lxm` = **`ai.hyprstream.identity.exchangeUcan`** (so the token is scoped to this one call, not reusable against any other method the host exposes). The exchange endpoint is the **XRPC/Lexicon** method `ai.hyprstream.identity.exchangeUcan` (under the #908 `ai.hyprstream.*` namespace), implemented under #1119. *Explicit supersession:* the consumer-owned `atproto-identity-exchange.md` spec originally sketched the endpoint as a Cap'n Proto method; a Cap'n Proto method carries no Lexicon NSID and therefore cannot be placed in `lxm`, so the XRPC method **supersedes** that shape (the consumer doc is being revised in parallel to match — see the header's supersession note — so the two contracts do not silently contradict).
- **Host `did:web` discovery/binding:** how a caller learns the intended hyprstream host's atproto-compatible `did:web` service DID, and how the host binds that `did:web` to its identity, is specified by #1119 (out of scope for this contract beyond the requirement that `aud` MUST be that `did:web`).
- **Replay properties that actually hold:** short `exp` (minutes), audience-pin (`aud` = host's `did:web` ⇒ only the named host accepts it), `lxm` pin (limits the token to the exchange method), and **`jti` enforced as one-time-use server-side** (the host records each accepted `jti` for the duration of `exp` and rejects replays).
- **Known gap (do not paper over it):** a stolen service-auth JWT is replayable by a third party within its short lifetime — the token carries **no proof that the presenter holds the original atproto session key**. A real token→DPoP-key cryptographic linkage does not exist on the atproto `getServiceAuth` surface today; this contract MUST NOT claim one. Hardening (e.g. binding the exchange to the atproto session DPoP key via an out-of-band proof the host can verify against the resolved PDS) is a **deferred security task**, tracked under #1119, not a property of this revision.
- **Failure modes fail closed** (§6): unresolved DID document, no verifying key, signature mismatch, `aud` ≠ host `did:web`, expired, wrong/missing `lxm`, or a replayed `jti` never silently elevate.

#### 10.2.3 The assurance floor (invariant)

The bridge is **asymmetric and honest** (terms mirror `atproto-identity-exchange.md` §1–§2):

- **The artifact is strong:** the minted UCAN is hybrid-PQC composite-signed (EdDSA + ML-DSA-65 COSE), like every hyprstream UCAN.
- **The identity assurance is floored at `Assurance::Classical`, unconditionally** — because the root of trust the grant was derived from is a classically-rooted atproto DID. A host's configured `CryptoPolicy`/native-policy floor does **not** propagate to atproto-derived grants. This is the same pattern as hyprstream's **#698 Classical floor** and the **D2 clamp-to-importing-boundary** rule (the assurance of an imported identity is clamped to the assurance of the boundary it crossed, not inflated to the importing system's native floor). Concretely: `resolve_identity_keys` over the atproto DID yields a classical verifying key only → `Assurance::Classical` per §6.
- The grant is host-scoped, short-TTL, re-evaluated on every refresh, and **never broader than requested** — mirroring hyprstream's ZSP grant-exchange. The `Classical` floor is recomputed on each refresh, not just first mint. (Note: "sender-bound" is deliberately **not** claimed — see §10.2.2's known gap.)

### 10.3 The `did:at9p` binding record — the only sanctioned upgrade path

The **only** sanctioned touchpoint between the atproto and `did:at9p` namespaces is a lexicon record in the user's own PDS repo (proposed NSID `ai.cyberdione.identity.at9pBinding`), specified fully in `atproto-identity-exchange.md` §3. It is valid **only if both** directions verify:

- **(a) atproto-side proof (repo possession):** the record lives in the repo of the `atprotoDid`, confirmed by resolving `atprotoDid` to its PDS and reading the collection; repo owner must equal `atprotoDid`.
- **(b) at9p-side proof (hybrid signature):** the record's `at9pSignature` verifies against the `did:at9p` identity's **hybrid key material** (EdDSA + ML-DSA-65 composite) resolved per §4.1.

A host that verifies a binding **MAY** mint grants at `Assurance::PqHybrid` for the **`at9pDid` subject** — the §10.2 `Classical` floor is lifted because a hybrid-PQC root of trust now endorses the link. Without a verified binding, the `Classical` floor stands (no optimistic promotion). This is the same fail-closed posture as §6.

### 10.4 External HTTPS consumer verification of `did:at9p` — FAIL CLOSED until #1114

How a **plain HTTPS web app** (the `www-cyberdione-ai` login page, GitLab epic cyberdione/www-cyberdione-ai#23) verifies a `did:at9p` login assertion is **TBD, pending the shape decision in #1114** — it is explicitly not specified here, and consumers MUST NOT assume any of the candidate shapes. #1114 will decide between (at least): (1) a narrow read-only HTTPS verification endpoint exposing the resolver output; (2) riding the #909 `/9p/<did-url>` G4 HTTPS gateway projection; (3) requiring an #896-style `alsoKnownAs` did:web bridge for login, deferring native at9p resolution.

**Until #1114 resolves and is written back into this section, a native `did:at9p` login assertion MUST fail closed** at every external consumer: the native resolver (`at9p_resolver.rs` / `at9p_gate.rs`) is native-only, the G4 HTTPS gateway (#909) is low-priority, and an external HTTPS consumer cannot resolve/verify a `did:at9p` on its own.

Crucially, **§10.2 is NOT a way to verify a `did:at9p` login assertion** — do not treat it as one. §10.2 proves control of an **atproto account** (a `did:plc`/`did:web`) and mints a grant for a subject **derived from that atproto DID**; it does not verify possession of, or an assertion by, the `did:at9p` identity. Only a **fully-verified §10.3 binding** (both directions — atproto repo possession **and** the `did:at9p` hybrid signature) connects the two subjects, and what it proves is the **binding between them**, not a standalone `did:at9p` login assertion. A consumer that mistakes §10.2's atproto-account proof for proof of the named `did:at9p` identity bypasses the §10.3 floor and the #1114 decision; it MUST be treated as a fail-closed error. Whatever shape #1114 picks will be reflected back into this contract (per #1114's own exit criteria) so the GitLab consumer codes against the documented outcome, not an assumed one.
