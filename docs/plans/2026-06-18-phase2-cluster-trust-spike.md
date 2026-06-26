# Spike — Phase 2 (identity + trust + mesh policy): #328 / #318 / #319 / #321

**Date:** 2026-06-18 · **Status:** research spike (no code) · **Gate:** access/authorization/policy
+ one capnp decision (#321) are human-gated. Companion to the multi-gpu spike + the pipeline-rpc spike.

## Headline
Phase 2 is mostly **wiring existing primitives into the JWT/Casbin authz layer + a CLI/wizard
surface — not net-new cryptography.** Already present: per-node DID `#mesh` (Ed25519) + `#mesh-pq`
(ML-DSA-65) keys (`did_document.rs:162,176`), `derive_mesh_mldsa_key`, admin-anchored
`KeyedPqTrustStore`/`mesh_peers` (`mesh_trust.rs`), hybrid Ed25519+ML-DSA envelope verify
(`envelope.rs:1471`), and an RFC 8628 device-code server (`services/oauth/device.rs`).

**Wildcard-fix note:** GitHub #130 is the *merged integration-1 epic*, not the regression. The
wildcard regression fix is `f3a09baa7` on `ewindisch/policy-wildcard-fix-authz-tests` — **present
on `ewindisch/310-multi-gpu`** (stacked in) but NOT on `main`. Prereq for #319 satisfied on-branch;
has a publish/merge dependency.

## C-IDENT proof (#328) — fail-unsafe, confirmed in code
- Single shared CA key, `kid` discarded — `auth/key_source.rs:143` (`ClusterKeySource::get_key`
  returns the one `ca_verifying_key` for every trusted issuer).
- Empty issuer unconditionally trusted — `key_source.rs:151` (`is_trusted`: `issuer.is_empty() →
  true`); repeated in `FederatedKeySource:337`.
- Every key collapses to one `"system"` subject — `node_identity.rs:169` (`resolve` →
  `Subject::new("system")` for any known pubkey; all purpose keys HKDF-derived from one root).
- Bites authz — `service/svc.rs:132` hardcodes `key_derived_subject = "system"`; `verify_claims`
  falls back to `resolve_key_subject(cnf)` (`:374`); `SERVICE_BASE_POLICIES` grants
  `service:policy,*,*,*,allow` (`policy_templates.rs:84`). → one leaked key ⇒ whole fleet = god
  principal; no kid/roster ⇒ no revocation without rotating the shared root.

## #328 fix outline (pure-Rust; reuse, don't invent)
1. `ClusterKeySource` → roster `HashMap<kid,VerifyingKey>` + revoked-kid set; `get_key` selects by
   `kid`; `is_trusted` rejects empty `iss` on the mesh path (keep empty-iss only for `Inproc`).
2. `NodeIdentityProvider::resolve` → map each known pubkey to `service:inference:host-<id>`
   (extend `known_pubkeys` HashSet → `HashMap<[u8;32],Subject>`, populated at enrollment).
3. `svc.rs` `with_*`/`resolve_key_subject` → confine the "system" shortcut to genuine in-process
   callers; networked mesh peers resolve via the roster to a host subject (deny → anonymous).
4. Reuse the admin-anchored `mesh_peers`/`KeyedPqTrustStore` (`mesh_trust.rs`) as the roster.
5. Pin mesh to iroh until QUIC/WT raw-pubkey binding (#200/#185); re-verify envelope `cnf` per call.

## #319 mesh policy vocab (pure-Rust, Casbin strings + Operation enum)
New `mesh:rpc` / `inference:peer-call` granted only to specific `service:inference:host-X` (non-
wildcard cluster domain). Read (`query.status`) vs authority (`infer.stage`, `delta.submit`) split.
Deny-by-default + fail-closed (mirror `federation_admission.rs:79`). Test: `*`/`anonymous` never
match authority actions even under `federation-open`. Depends on #328 (host subjects) + the
wildcard fix landing.

## #318 PDS attach (mostly pure-Rust; pre-auth endpoint = net-new server)
Device-code server exists (`oauth/device.rs`, TTL 600s). Missing: `pds attach/join` CLI
(`bin/main.rs` clap `augment_subcommands` + new `cli/pds_handlers.rs`, mirror service_handlers) +
a wizard phase (`hyprstream-tui/src/wizard/phases.rs`). Flow: host did:key from iroh node_id →
device-code (attended) or `--token` pre-auth (unattended, needs new one-time mint/redeem endpoint)
→ PoP-bound credential (`cnf` plumbing exists). One home PDS/host (D4); mesh membership is a
separate policy grant (#319).

## #321 AEAD + provenance (capnp-gated long pole)
AES-256-GCM exists (`crypto/event_crypto.rs`) but only on the **events** path (group-key, opt-in,
`notification.capnp`). The **moq streaming plane is plaintext** (`moq_stream.rs` publish/write —
no encrypt). Mandating AEAD on mesh needs: per-job content key (not the events group key) bound to
iroh/host identity; an encrypt/decrypt hook in `moq_stream.rs`; provenance = per-host signature
(reuse `derive_mesh_mldsa_key` + `KeyedPqTrustStore`, depends on #328); and a capnp decision
(new streaming-plane `TaggedPayload`/`SignedStagePayload` vs overloading notification.capnp).

## HUMAN DECISION MENU
- **D-1 per-host identity (#328):** (a) shared root [status quo, fail-unsafe]; **(b) RECOMMENDED**
  per-host keypair + admin-anchored roster reusing existing mesh primitives; (c) full SPIRE SVID
  [defer].
- **D-2 mesh authz strictness (D3 vs C-IDENT):** (a) semi-open now; **(b) RECOMMENDED** semi-open
  reach/transport but strict deny-by-default for `mesh:rpc`/`infer.stage`/`delta.submit` from day
  one; gate internet/untrusted-tenant deploy on #328.
- **D-3 empty-issuer on mesh:** (a) keep trusting; **(b) RECOMMENDED** reject on networked/mesh,
  keep only for `Inproc`.
- **D-4 AEAD mandatory on mesh (#321):** (a) opt-in; **(b) RECOMMENDED** mandatory for all
  non-loopback mesh activation/delta payloads.
- **D-5 capnp for AEAD/provenance:** new dedicated streaming-plane struct (recommended) vs overload
  notification.capnp. (Mesh vocab itself = no capnp.)
- **D-6 pre-auth token (#318):** device-code only for Phase 2 (recommended) vs build pre-auth
  endpoint now.

## Sequencing
`#328 (pure-Rust, FIRST) → (#319 ∥ #318 in parallel) → #321 (capnp-gated)`. #328 touches
`auth/`+`node_identity.rs`+`svc.rs` — disjoint from the Phase-1 inference files, so it can run in
parallel with the in-flight pipeline work with zero branch conflict. Only hard human gates: #321
capnp struct (D-5) + AEAD-mandatory (D-4).
