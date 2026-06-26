# Inference-mesh authz — use-cases × journeys × threats × 3-layer model (DRAFT)

**Date:** 2026-06-18 · DRAFT scaffold for human refinement · grounds #310/#319/#328 authz.
Model: **L1 identity (SVID/WIT/did:key, COSE verify) → L2 UCAN (conveyed/delegated, offline) →
L3 Casbin/PolicyService (ambient authority, AUTHORITATIVE).** ALLOW iff all pass (AND). Compose seam =
`service/svc.rs:367 verify_claims`.

**Two asymmetries drive everything:** UCAN revocation is FAIL-OPEN (spec: "last line of defence") → UCAN
can never be the deny authority. Casbin denial fails CLOSED (`federation_admission.rs:38-49`) → Casbin is
the instant-revoke + org-invariant authority.

## Actors
ModelService/router (`service:model` SVID) · InferenceService host (`spiffe://td/inference/host-X`;
TODAY collapses to `"system"` — `node_identity.rs:169`, the #328 defect) · tenant workload (untrusted,
must NOT hold mesh:rpc) · admin (`service:policy`, mints delegations) · PolicyService (did:web=trust-domain,
owns `mesh://` UCAN root, authoritative Casbin, CID denylist) · Tiles peer (foreign did:key, rs-ucan rc.1) ·
browser (OAuth/OIDC, never mesh) · aggregator (#326, validates merges).

## Mapping (PRIMARY / supporting / N/A)
| Journey / Threat | L1 id | L2 UCAN | L3 Casbin | why |
|---|---|---|---|---|
| J1 host enroll (#318) | **P** | N/A | support | id issuance; mesh grant is later L3 |
| J2 mesh STAGE (internal) | support | **N/A or support❓** | **P** | internal — Casbin sandbox suffices; UCAN optional (OQ-1) |
| J3 delta submit (internal) | support(prov) | N/A❓ | **P** | role gate L3; poisoning is validation not authz |
| J4a Tiles inbound | support | **P** | support | owner-rooted cross-org delegation — UCAN's home turf |
| J4b Tiles outbound | support | **P** | support(mint-auth) | hyprstream conveys attenuated authority off-domain |
| J5 revocation | N/A | support(TTL) | **P** | instant fail-closed revoke must be L3 (CID denylist) |
| J6 model load / tenant routing | support | N/A | **P** | ambient tenant isolation = pure Casbin |
| T1 compromised host key | **P**(#328 scopes) | N/A | **P**(revoke) | TODAY NO — one key=god principal (`policy_templates.rs:80`) |
| T2 malicious co-tenant | support | N/A | **P** | deny-by-default ambient isolation |
| T3 rogue stage (C-PROV) | support | N/A | support | **all N/A for correctness** — validation layer |
| T4 poisoned delta (C-AGG) | support | N/A | support | scopes submitter; defense = reducer+eval |
| T5 stolen SVID | **P**(PoP/cnf) | N/A | support(revoke) | cnf-binding defeats bearer theft |
| T6 confused-deputy | support | support(attenuate) | **P** | L3 caps ceiling — why L3 stays authoritative over L2 |
| T7 replay | **P** | support(nonce) | support(denylist) | per-call verify + nonce + denylist |
| T8 Tiles over-reach | support | **P**(no-widen) | support(register) | L2 bounds grant, L3 bounds admitted root |
| T9 PolicyService outage | N/A | N/A | **P**(fail-closed) | only L3 has the closed-fail posture |
| T10 UCAN revoke gap | N/A | support | **P**(CID denylist) | L3 closes L2's fail-open gap |
| T11 empty-issuer trusted | **P** | N/A | support | reject empty iss off-Inproc (`key_source.rs:151` today=true) |

**What it shows:** UCAN necessary in ONE band = cross-org/Tiles (J4a/J4b/T8). Casbin necessary in the
largest band = ambient isolation (T2,J6), instant fail-closed revoke (J5,T1,T10), org invariants (T9),
confused-deputy ceiling (T6) — UCAN structurally cannot do these. Overlap (defense-in-depth): T1,T7,T8.
**Neither covers C-PROV/C-AGG correctness (T3/T4) — that's a validation layer, not authz.**

## Recommended division (DRAFT)
1. **Casbin = authoritative gate for ALL traffic** (deny-by-default, fail-closed, instant CID/jti revoke). The floor.
2. **#328 per-host identity FIRST** (critical: one leaked key=god principal today). UCAN must not block it.
3. **#319 mesh Casbin vocab** (`mesh://`, infer.stage/delta.submit/query.status, non-wildcard host subjects),
   designed UCAN-translatable, shipped Casbin-native first. Net-new `Operation` variants; depends on wildcard fix f3a09baa7.
4. **UCAN = additive `feature="ucan"` layer scoped to cross-org/Tiles** (where it earns its place). Pin Tiles' rs-ucan rc.1 rev (unaudited).
5. **C-PROV/C-AGG OUTSIDE the authz stack** — provenance sig (L1) + aggregator norm-bound/held-out eval.

## OPEN QUESTIONS (human sign-off)
- **OQ-1:** do INTERNAL mesh calls (J2/J3) need UCAN at all, or only cross-org/Tiles? (lean: Casbin-only internal.)
- **OQ-2:** per-job attenuation worth UCAN internally, or Casbin resource-path (`mesh://acme/job/j7`) + short-lived WIT `cnf` enough? (do internal hosts ever re-delegate?)
- **OQ-3:** is the (i)/(ii) line = the trust-domain boundary (federation:register)? Inside PDS=Casbin-core, cross-domain=UCAN-native. (recommended seam.)
- **OQ-4:** rs-ucan risk tolerance (vendor unaudited git rc.1 vs minimal in-house verify vs defer). [deferred per user]
- **OQ-5:** SVID model — WIT-as-SVID bridge [~80% exists] vs native SPIFFE Workload API now.
- **OQ-6:** canonical mesh roster — admin-anchored KeyedPqTrustStore vs dynamic JwksKeySource (affects revoke propagation speed). [user leaned JwksKeySource]
- **OQ-7:** make C-PROV/C-AGG validation an explicit **4th layer** (identity→UCAN→Casbin→provenance/validation) so the correctness gap is named, not an asterisk?
