# SVID/SPIFFE scope investigation → revised #328 (per-host mesh identity)

**Date:** 2026-06-18 · read-only spike · branch `ewindisch/310-multi-gpu` @ `e99e719d5`.
Triggered by: "we may already have extensive SVID code — investigate SPIFFE scope."

## Headline
No *literal* SVID/SPIFFE/TrustDomain type (only 2 comments in `key_source.rs:6,49` + a design-only
`docs/spiffe.md` whose referenced `docs/plans/phase-4-spiffe-fqdn-aligned.md` is **missing/dangling**).
**But the SVID concepts are implemented under WIMSE/WIT + did:web/did:key + COSE names** — the
multi-gpu spike already frames it "SPIFFE-forward, bridged" (spike.md:542). The prior Phase-2 spike
undercounted this; #328 is *more* "just wiring" than thought (~80% exists).

## Inventory (the SVID stack, renamed)
- **WIT = JWT-SVID:** `auth/jwt.rs:155 encode_composite_service_jwt` (`typ:"wit+jwt"`, ML-DSA-65+Ed25519
  hybrid, `kid`=JWK thumbprint); `oauth/wit_bootstrap.rs:38 POST /oauth/wit` (cnf-bound, 30d);
  `auth/claims.rs:74 Claims`+`Cnf` (RFC 8705/9449 PoP, "WIMSE WITs"); `claims.rs:325 subject()` →
  local bare / remote `Subject::federated(iss,sub)`.
- **Rosters (two, already built):** `key_source.rs:261 JwksKeySource` (kid cache + negative cache +
  kid_alg stripping defense, "Replaces ClusterKeySource"); `auth/mesh_trust.rs:46
  build_mesh_pq_trust_store` → admin-anchored kid-keyed `KeyedPqTrustStore` from `oauth.mesh_peers`
  (fail-fast, immutable).
- **Bundle/publication:** `oauth/did_document.rs:202 .well-known/did.json` (#mesh-pq Multikey);
  `oauth/federation_entity.rs .well-known/openid-federation`.
- **Federation/admission:** `auth/federation_admission.rs` (#137 gate, fail-closed); `admission.rs:81`
  (`FederationAdmissionGate`, did:key self-cert, `admit()→AdmittedIdentity`); `transport/iroh_admission.rs:55
  admit_peer(node_id)` (channel binding seam).
- **Identity core:** `identity.rs:37 IdentityProvider`; `node_identity.rs:169 resolve()→Subject::new("system")`
  (THE #328 defect); `envelope.rs:362 Subject` (`federated(iss,sub)→"iss:sub"`); `service/svc.rs:181`
  subject priority key>jwt>anon; `envelope.rs:697` per-call COSE_Sign hybrid (EdDSA+ML-DSA enforced).
- **Onboarding:** `oauth/device.rs` RFC 8628 (host attended, #318); `oauth/device_enrollment.rs` (device).

**Net-new (confirmed absent):** `TrustDomain`/`Svid`/`WorkloadIdentity` types; `spiffe://` scheme;
`.well-known/spiffe/bundle` route; `mesh:rpc`/`inference:peer-call` Casbin vocab;
`service:inference:host-<id>` subjects.

## Revised #328 (reuse-maximizing, pure-Rust)
1. `node_identity.rs:96-176`: `known_pubkeys: HashSet<[u8;32]>` → `HashMap<[u8;32], Subject>`;
   `resolve()` returns the mapped per-host subject (deny→anonymous); keep `"system"` only for the
   node's own root/derived keys.
2. Populate that map from the existing roster — extend `mesh_trust.rs:build_mesh_pq_trust_store` to
   also yield `(ed25519_pubkey → Subject "service:inference:host-<label>")`. `mesh_peers` config is the
   enrollment record; no new roster type.
3. Route mesh verification through `JwksKeySource`/`KeyedPqTrustStore`, not `ClusterKeySource`
   (`get_key(iss,kid)` already takes kid; JwksKeySource honors it). Delete/guard ClusterKeySource's
   single-key + empty-iss-trust path.
4. Reject empty `iss` off-`Inproc` (gate `is_local_iss`/`is_trusted` by transport).
5. `svc.rs:181,374`: confine the `"system"`/`resolve_key_subject` shortcut to in-process callers.
6. Bind host subject to iroh channel via `iroh_admission.rs:admit_peer`; per-call COSE signer re-verify
   already enforced.

## #319 mesh policy (net-new vocab, machinery exists)
Add `mesh:rpc`/`inference:peer-call` resource+action scoped to `service:inference:host-<id>`
(non-wildcard), read (`query.status`) vs authority (`infer.stage`,`delta.submit`) split, deny-by-default.
Extends existing `service:<name>` scheme (`policy_templates.rs:74`). Depends on #328 subjects + the
wildcard fix (the `*`/`service:*` at policy_templates.rs:78) landing. Per human (2026-06-18): strict +
deny-by-default at granular depth, mind group-policy mgmt + atproto authz interop, short-term via
existing policy templates.

## HUMAN DECISIONS (gate #328 approach)
- **D-A SVID model:** (a) **bridge — treat WIT as the SVID** [RECOMMENDED, matches spike.md:542]; add
  `spiffe://` naming + a `.well-known/spiffe/bundle` view over existing JWKS/did:web later. (b) build
  native SPIFFE (Workload API socket, X.509-SVID) now. The prior menu's "full SPIRE SVID [defer]" is
  re-scoped — most already exists as WIT.
- **D-B canonical mesh roster:** (a) **admin-anchored `KeyedPqTrustStore`/`mesh_peers`** [RECOMMENDED
  homelab — fail-fast, immutable, matches revocation goal]; (b) dynamic `JwksKeySource`.
- **D-C dangling `docs/spiffe.md`:** write the missing phase-4-spiffe plan vs fold its intent
  (bundle endpoint / Workload API / X.509-SVID) into #328/#319. (Code has drifted ahead via WIT+did:web.)
