# Cross-repo identity contract: `Did` / did:web / did:plc / did:key

**Status:** settled (pending #578 merge — shape is final, safe to mirror now).
**Canonical owner:** `hyprstream` — `crates/hyprstream-rpc/src/identity.rs` (`Did`).
**Source issues/PRs:** #563 (design + sign-off) · #578 (def + field-level `$domainType`) · #579 (resolver + assurance, PQC pathway).
**Consumers:** `www-cyberdione-ai` frontend (`src/api/atproto.ts`, MR !18 / #2 / #5) — mirrors this shape, does not re-derive. Security/MAC epic #547 (#548/#549) — consumes the #579 resolver.

This doc anchors the hyprstream ↔ www-cyberdione-ai identity boundary. **hyprstream owns the representation; everyone else mirrors it.**

---

## 1. `Did` is an opaque identifier string — never a key container
A `Did` is a **branded string** holding **any** DID method: `did:web:…`, `did:plc:…`, `did:key:…`. It carries **no key material inline**. Keys are *resolved from* it (DID document / trust store), never embedded.

- Rust (canonical): `pub struct Did(String)` — `$domainType("hyprstream_rpc::identity::Did")`, **bare `Text` on the wire** (W3C DID strings; no wrapper/nesting).
- TS (mirror): `type Did = string & { readonly __brand: 'Did' }` (brand a string; do not build an object).

## 2. Method predicates (both sides)
`is_did_key` · `is_did_web` (`startsWith("did:web:")`) · `is_did_plc`.

## 3. Validation posture — lenient construct, strict at the boundary
- **Construction is lenient:** `Did::new(s)` / the TS constructor accept **any** string; no validation, no rejection. Enables gradual adoption + opaque pass-through.
- **Validation/verification is strict at the trust boundary:** resolution + key verification + admission. (hyprstream: the admission gate; frontend: at resolve + before trusting.) **Never** trust a `Did` on construction alone.

## 4. Resolution contract — method governs *resolution*, not *document shape*
The DID **method** decides only *how* you fetch the document:
- `did:web:host[:path]` → HTTPS `GET https://host/.well-known/did.json` (or `/<path>/did.json`).
- `did:plc:…` → PLC directory.
- `did:key:…` → **no document**; the key *is* the identifier (self-certifying).

The resolved **document shape is method-agnostic**:
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

## 5. Key encodings
- **did:key / Multikey (Ed25519):** multicodec `ed25519-pub` prefix `0xed01`, multibase base58btc (`z…`). Ed25519 `did:key` ⇄ 32-byte key is lossless (hyprstream: `did_key.rs`).
- **PQC (forward):** an algorithm-agile multikey (ML-DSA-65 + others) + ML-DSA verification methods are the #579 generalization. `Did` is already algorithm-agnostic (it's a string); only the multikey codec + VM extraction generalize.

## 6. Assurance enum (consumers that compute trust level)
Locked 1:1 with epic #547 / S1 — **same names + discriminants, no translation**:
```
enum Assurance { Unverified = 0, Classical = 1, PqHybrid = 2 }
```
- no resolvable verifying key → **Unverified** (fail-closed; never silent-elevate)
- Ed25519 only → **Classical**
- Ed25519 + ML-DSA-65 → **PqHybrid**

hyprstream resolver (#579): `resolve_identity_keys(&Did) -> { ed25519: Option, ml_dsa_65: Option, assurance: Assurance }`.

## 7. Custom transport service entries — frontend IGNORES
hyprstream publishes extra `service` entries (`type: "IrohTransport"` / `"QuicTransport"`) with **map** `serviceEndpoint`s (W3C DID Core §5.4). These are non-atproto-canonical hyprstream extensions for dialing. **The www frontend uses only `#atproto_pds` + `verificationMethod`** and ignores transport entries.

## 8. Ownership boundary
| Concern | Owner |
|---|---|
| `Did` representation, predicates, validation posture | **hyprstream** (canonical) |
| Resolver (`resolve_identity_keys`) + `Assurance` | **hyprstream** (#579) |
| did:web/plc/key resolution in the frontend | **www-cyberdione-ai** — mirrors §1–§6 |
| MAC clearance / OAuth→UCAN / perimeter | **epic #547** — consumes #579 |
| Custom transports (iroh/quic) | **hyprstream** — frontend ignores |

**Rule:** mirror string + predicates + DID-doc parse + assurance enum. Do **not** re-derive the representation or fork the enum. Changes to this contract go through the canonical owner (hyprstream) and are relayed to consumers.

---
*Maintained by the hyprstream identity owner (pane 1:3.0). Frontend sync: pane 1:6.0.*

## 9. Resolved edge cases (frontend ↔ canonical)
- **Missing `#atproto_pds`:** **HARD-ERROR (fail-closed).** Do not fall back to another typed `serviceEndpoint` for the PDS.
- **Service precedence:** `#atproto_pds` (`type: AtprotoPersonalDataServer`) is the **sole atproto-canonical PDS** service; hyprstream emits it **first** (`did_document.rs`: "atproto PDS first, then legacy HyprstreamService"). **`HyprstreamService` is legacy** — prefer `#atproto_pds`, do not prefer it. `serviceEndpoint` is **origin-only** (scheme+host+port).
- **Path-based `did:web`:** enforce **`doc.id === requested DID`** (anti-spoof); **percent-encode the port** (`did:web:example.com%3A6791:users:alice` → `https://example.com:6791/users/alice/did.json`); path form uses `/<path>/did.json` (`.well-known` only when there is no path).
- **`did:plc` directory base:** **configurable, default `https://plc.directory`.** hyprstream does not resolve `did:plc` (the frontend owns it); pin the default, allow override for self-hosted/dev/federation.
