# Cross-repo identity contract: `Did` / did:web / did:plc / did:key / did:at9p

**Status:** settled (#578 merged). Revision history: `did:at9p` resolution semantics amended (#1097, §4.1/§5/§6); `did:at9p` atproto-OAuth-flow terms amended (#1097, §10 — this revision). This revision supersedes the pre-#1097 text only by *addition* (new §10 + header/ownership updates); §§1–9 are unchanged in meaning.
**Canonical owner:** `hyprstream` — `crates/hyprstream-rpc/src/identity.rs` (`Did`).
**Source issues/PRs:** #563 (design + sign-off) · #578 (def + field-level `$domainType`) · #579 (resolver + assurance, PQC pathway) · #1097 (`did:at9p` amendment — resolution + OAuth-flow terms) · #1113 (atproto OAuth AS conformance profile) · #1114 (HTTPS `did:at9p` verification — **open**) · #948 (OAuth conformance spike).
**Consumers:** `www-cyberdione-ai` frontend (`src/api/atproto.ts`, MR !18 / #2 / #5) — mirrors this shape, does not re-derive. Security/MAC epic #547 (#548/#549) — consumes the #579 resolver. **Sibling spec:** `www-cyberdione-ai` `docs/contracts/atproto-identity-exchange.md` (DID→UCAN exchange + `did:at9p` binding record) — §10 mirrors its terms and MUST NOT contradict it; that doc owns the *exchange spec*, this doc owns the `Did` shape, resolution posture, and `Assurance` enum it builds on.

This doc anchors the hyprstream ↔ www-cyberdione-ai identity boundary. **hyprstream owns the representation; everyone else mirrors it.**

---

## 1. `Did` is an opaque identifier string — never a key container
A `Did` is a **branded string** holding **any** DID method: `did:web:…`, `did:plc:…`, `did:key:…`, `did:at9p:…`. It carries **no key material inline**. Keys are established by the method-specific resolution and verification path; they are never fields of the `Did` value.

- Rust (canonical): `pub struct Did(String)` — `$domainType("hyprstream_rpc::identity::Did")`, **bare `Text` on the wire** (W3C DID strings; no wrapper/nesting).
- TS (mirror): `type Did = string & { readonly __brand: 'Did' }` (brand a string; do not build an object).

## 2. Method predicates
Rust (`identity.rs`) provides `is_did_key` · `is_did_web` (`startsWith("did:web:")`) · `is_did_at9p` (`startsWith("did:at9p:")`) — there is **no** `is_did_plc` on the Rust `Did` type (hyprstream does not resolve `did:plc`; see §4/§9). The `did:at9p:` literal is canonically owned by `DID_AT9P_PREFIX`; mirrors MUST use that exact, case-sensitive prefix. The TS mirror may additionally implement an `is_did_plc` predicate for its own resolution path.

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
| atproto OAuth (did:plc / did:web) login — AS conformance profile | **hyprstream** (`services/oauth`, #1113/#948) — stock atproto client completes the flow unmodified |
| DID→UCAN exchange endpoint (mint + verify, did:at9p path) | **hyprstream** (`services/oauth`, #1097 §10) — endpoint; **www-cyberdione-ai** `atproto-identity-exchange.md` owns the spec |
| External HTTPS verification of `did:at9p` (login assertion) | **OPEN — #1114** (shape undecided: verification endpoint vs `/9p/<did-url>` gateway vs `alsoKnownAs` did:web bridge) |
| `did:at9p` binding record (`ai.cyberdione.identity.at9pBinding`) | **www-cyberdione-ai** (record write) + **hyprstream** (dual-direction verify, §10.3) |

**Rule:** mirror string + predicates + applicable DID-doc/capsule dispatch + assurance enum. Do **not** re-derive the representation, parse `did:at9p` as a DID document, fork the enum, or treat an unverified service claim as authority. Changes to this contract go through the canonical owner (hyprstream) and are relayed to consumers.

---

## 9. Resolved edge cases (frontend ↔ canonical)
- **Missing `#atproto_pds`:** **HARD-ERROR (fail-closed).** Do not fall back to another typed `serviceEndpoint` for the PDS.
- **Service precedence:** `#atproto_pds` (`type: AtprotoPersonalDataServer`) is the **sole atproto-canonical PDS** service; hyprstream emits it **first** (`did_document.rs`: "atproto PDS first, then legacy HyprstreamService"). **`HyprstreamService` is legacy** — prefer `#atproto_pds`, do not prefer it. `serviceEndpoint` is **origin-only** (scheme+host+port).
- **Path-based `did:web`:** enforce **`doc.id === requested DID`** (anti-spoof); **percent-encode the port** (`did:web:example.com%3A6791:users:alice` → `https://example.com:6791/users/alice/did.json`); path form uses `/<path>/did.json` (`.well-known` only when there is no path).
- **`did:plc` directory base:** **configurable, default `https://plc.directory`.** hyprstream does not resolve `did:plc` (the frontend owns it); pin the default, allow override for self-hosted/dev/federation.
- **`did:at9p` aliases:** a `did:web` / `did:key` / `did:plc` alias does not replace the durable `did:at9p` identity. Treat the at9p identity as authoritative only when both the GATE-verified capsule and the classical DID document mutually attest the alias; one-way `alsoKnownAs` is insufficient.
- **Unsupported `did:at9p` resolution:** opaque storage/pass-through is allowed, but any operation that needs keys, assurance, current services, or authority MUST hard-error until the canonical at9p resolver is available.

---

## 10. atproto login flow — method dispatch (amended by #1097)

The resolution contract in §§1–9 governs how a `Did` is *resolved and verified*. This section governs a different question: **how a DID enters an atproto login session**, and is the formal `did:at9p` supersession requested by #1097. It mirrors — and MUST NOT contradict — the `www-cyberdione-ai` sibling spec `docs/contracts/atproto-identity-exchange.md` (canonical owner of the exchange *spec*); this doc owns the `Did` shape, resolution posture, and `Assurance` enum that spec builds on.

### 10.1 `did:plc` / `did:web` — standard atproto OAuth

Login for `did:plc` / `did:web` is **standard atproto OAuth** end-to-end: PAR (RFC 9126) + DPoP (ES256) + session refresh, against any PDS, driven by a stock atproto client (e.g. `@atproto/oauth-client-browser`). hyprstream's OAuth AS implements this profile under **#1113** (conformance: scopes `atproto` / `transition:generic`, PDS=self AS per RFC 9414/9728 metadata, token `sub` = the account DID) and is validated end-to-end by the **#948** conformance spike. No hyprstream-specific deviation from the atproto OAuth profile is part of this contract; `did:plc`/`did:web` verification keys are **classical** (secp256k1 / P-256, ES256 DPoP) — an atproto protocol constraint, not a hyprstream choice.

### 10.2 `did:at9p` — does NOT enter the atproto OAuth flow

`did:at9p` is **not** a registered atproto DID method and **never** appears in the atproto OAuth login flow (no PAR, no DPoP, no `@atproto/oauth-client-browser` session). A `did:at9p` login is handled by a **DID→UCAN exchange on the hyprstream side**: the client presents an atproto service-auth token (`com.atproto.server.getServiceAuth`, `aud` = the hyprstream host's DID, short TTL, identity-proof scope); the host resolves the caller's atproto DID **itself** (per §4, never trusting a client-supplied document/key), verifies the token signature against the resolved classical verification key, and on success **mints a hyprstream UCAN** for a subject derived from the atproto DID.

The bridge is **asymmetric and honest** (terms mirror `atproto-identity-exchange.md` §1–§2):

- **The artifact is strong:** the minted UCAN is hybrid-PQC composite-signed (EdDSA + ML-DSA-65 COSE), like every hyprstream UCAN.
- **The identity assurance is floored at `Assurance::Classical`, unconditionally** — because the root of trust the grant was derived from is a classically-rooted atproto DID. A host's configured `CryptoPolicy`/native-policy floor does **not** propagate to atproto-derived grants. This is the same pattern as hyprstream's **#698 Classical floor** and the **D2 clamp-to-importing-boundary** rule (the assurance of an imported identity is clamped to the assurance of the boundary it crossed, not inflated to the importing system's native floor). Concretely: `resolve_identity_keys` over the atproto DID yields a classical verifying key only → `Assurance::Classical` per §6.
- The grant is host-scoped, sender-bound (DPoP from the atproto session), short-TTL, re-evaluated on every refresh, and **never broader than requested** — mirroring hyprstream's ZSP grant-exchange. The `Classical` floor is recomputed on each refresh, not just first mint.
- Failure modes fail closed (§6) — unresolved DID document, no verifying key, signature mismatch, `aud` ≠ host DID, expired, replayed, or over-broad scope never silently elevate.

### 10.3 The `did:at9p` binding record — the only sanctioned upgrade path

The **only** sanctioned touchpoint between the atproto and `did:at9p` namespaces is a lexicon record in the user's own PDS repo (proposed NSID `ai.cyberdione.identity.at9pBinding`), specified fully in `atproto-identity-exchange.md` §3. It is valid **only if both** directions verify:

- **(a) atproto-side proof (repo possession):** the record lives in the repo of the `atprotoDid`, confirmed by resolving `atprotoDid` to its PDS and reading the collection; repo owner must equal `atprotoDid`.
- **(b) at9p-side proof (hybrid signature):** the record's `at9pSignature` verifies against the `did:at9p` identity's **hybrid key material** (EdDSA + ML-DSA-65 composite) resolved per §4.1.

A host that verifies a binding **MAY** mint grants at `Assurance::PqHybrid` for the **`at9pDid` subject** — the §10.2 `Classical` floor is lifted because a hybrid-PQC root of trust now endorses the link. Without a verified binding, the `Classical` floor stands (no optimistic promotion). This is the same fail-closed posture as §6.

### 10.4 External HTTPS consumer verification of `did:at9p` — OPEN (#1114)

How a **plain HTTPS web app** (the `www-cyberdione-ai` login page, GitLab epic cyberdione/www-cyberdione-ai#23) verifies a `did:at9p` login assertion is **TBD, pending the shape decision in #1114** — it is explicitly not specified here, and consumers MUST NOT assume any of the candidate shapes. #1114 will decide between (at least): (1) a narrow read-only HTTPS verification endpoint exposing the resolver output; (2) riding the #909 `/9p/<did-url>` G4 HTTPS gateway projection; (3) requiring an #896-style `alsoKnownAs` did:web bridge for login, deferring native at9p resolution. **Until #1114 resolves and is written back into this section**, the native resolver (`at9p_resolver.rs` / `at9p_gate.rs`) is native-only and the G4 HTTPS gateway (#909) is low-priority; an external HTTPS consumer that needs to verify a `did:at9p` MUST either go through the host-side DID→UCAN exchange (§10.2) or fail closed — it cannot resolve/verify a `did:at9p` on its own. Whatever shape #1114 picks will be reflected back into this contract (per #1114's own exit criteria) so the GitLab consumer codes against the documented outcome, not an assumed one.
