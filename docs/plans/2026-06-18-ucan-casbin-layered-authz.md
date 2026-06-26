# UCAN-native + Casbin-backed authz (layered) — design + decision

**Date:** 2026-06-18 · read-only spike · branch `ewindisch/310-multi-gpu`. Sources: ucan-wg/spec,
/delegation, /invocation, /revocation; ucan-wg/rs-ucan; crates.io/crates/ucan; tiles.run.

## LOAD-BEARING FACT — rs-ucan maturity
- **No published UCAN-1.0 crate.** crates.io `ucan` = **0.4.0 (2023), 0.x single-JWT, wire-incompatible
  with 1.0.** UCAN-1.0 code is **git-only** on `ucan-wg/rs-ucan` `main` = **1.0.0-rc.1, unpublished,
  Rust 1.90 MSRV, "not formally audited — use at your own risk."** Ed25519-first; DAG-CBOR + varsig
  (no JWT/JOSE); p256/rsa + Send posture UNVERIFIED.
- **Tiles rides that git rc.1 line, did:key-rooted.** Native UCAN interop ⇒ pin the SAME rs-ucan rev as
  Tiles. "Adopt UCAN" = vendor an unaudited git rev, not `cargo add`. Biggest risk in the plan.

## UCAN 1.0 model (brief)
Delegation (grant/attenuate: iss/aud/sub/cmd/pol/exp/nonce) · Invocation (exercise: iss/sub/cmd/args/
**prf**=delegation CIDs) · Revocation (an Invocation `ucan/revoke` over a delegation CID) · Promise.
Attenuation: `sub` constant, `cmd` equal-or-narrower, `pol` predicates intersect. Verify offline:
aud→iss linkage + sigs + no-widening + time bounds + root `iss==sub`. **Revocation is fail-open /
eventually-consistent — spec calls it "last line of defence."** Mitigate with short `exp` + least authority.

## 3-layer design (plugs into svc.rs verify_claims:367)
`identity (SVID/WIT/did:key, COSE verify envelope.rs:1471) → UCAN capability (chain verify, offline) →
Casbin overlay (PolicyService check, AUTHORITATIVE)`. Allow iff all three pass.
- **Casbin stays authoritative**, esp. deny: UCAN revoke is fail-open, so the instant-revoke + governance
  authority must be Casbin (reuse `jti_blocklist` svc.rs:331 for CID denylist; deny-by-default +
  fail-closed mirror federation_admission.rs:78). UCAN = native grant/delegation/interop layer.
- **"Casbin-backed UCAN" = (a) mint-authority** (Casbin gates who may delegate what: `check(minter,
  ucan:mint:mesh://…, delegate)`) **+ (b) standing overlay + instant-revoke backstop**.

## UCAN ⇄ Casbin 1:1 mesh vocab
Resource `mesh://<td>/<host|tenant>/…`, abilities `{infer.stage, delta.submit, query.status}`, caveats
`{tenant,layers,job,exp}`. Casbin model `(sub,obj,act)` has no caveat slot → non-`exp` caveats encode
into the resource path. Example: UCAN `cmd=/mesh/infer/stage, pol=[["==",".tenant","acme"],["==",
".job","j7"]]` ⇄ Casbin `p, service:inference:host-7, acme, mesh://acme/job/j7, infer.stage, allow`.
Extend `Operation` (auth/mod.rs:38) + `SERVICE_BASE_POLICIES` (policy_templates.rs:74), non-wildcard
host subjects; depends on the `*`/`service:*` wildcard fix (f3a09baa7, on-branch not main).

## Root authority + Tiles interop
PolicyService did:web (= FQDN trust domain, phase-4-spiffe:39/119) is the `mesh://` root owner DID
(+ did:key form via #280). Foreign (Tiles) UCAN roots admitted ONLY via the existing `federation:register`
gate (federation_admission.rs:63). Tiles interop = no translation gateway (both sides rs-ucan rc.1,
did:key, Ed25519 — drop our ML-DSA half for Tiles outbound; inbound accepts EdDSA-inner). Verify path:
COSE verify Ed25519 signer → UCAN chain verify → federation:register admits root did:key → Casbin overlay
on `Subject::federated` (envelope.rs:426).

## Reuse vs net-new
Reuse: did:key/Multikey, COSE sign/verify (envelope.rs:697/1471), KeyedPqTrustStore roster
(mesh_trust.rs:52), JwksKeySource, Casbin PolicyService + check, federation:register gate, jti_blocklist
(→ CID denylist), verify_claims seam, Subject::federated. Net-new: rs-ucan (git rc.1) dep; UCAN
store/verify module; CID revocation denylist surface; mesh:// Casbin vocab + Operation variants; envelope
field for invocation + prf. Send: trait is `#[async_trait(?Send)]` (svc.rs:240) → !Send UCAN verify OK,
but rs-ucan Send posture unverified.

## DECISIONS (recommended)
1. **rs-ucan:** pin `ucan-wg/rs-ucan` git rc.1 @ vendored rev **behind `feature="ucan"`** [rec]; reject
   `ucan 0.4.0` (0.x, Tiles-incompatible); fallback = minimal in-house 1.0 verify if audit/Rust-1.90/Send block.
2. **Authority:** **Casbin authoritative gate; UCAN = attenuating capability/interop layer on top** [rec]
   (UCAN revoke fail-open → not the deny authority).
3. **Revocation:** **Casbin-instant primary (deny/CID denylist) + UCAN-TTL secondary** [rec]; short delegation TTLs.
4. **Root DID:** **PolicyService did:web (+did:key) = mesh:// root** [rec]; foreign roots via federation:register.
5. **Sequencing:** **ship #328 (per-host identity, CRITICAL — one leaked key = god principal today) + #319
   (mesh Casbin vocab, UCAN-translatable) as the pure-Rust audited BASELINE first; layer UCAN behind the
   flag after** [rec]. UCAN must not block the #328 security fix.

## Risks
rs-ucan unaudited git/Rust-1.90/unpublished (supply-chain); spec at rc not final (churn; pin same rev as
Tiles); per-call chain verify cost on hot path (cache verified chains by invocation CID); Tiles classical-only.
Aligns with: tiles-alignment.md (UCAN bridge was DEFERRED — this is its un-deferral, layered on the #328/#319 baseline).
