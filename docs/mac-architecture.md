# Native MAC — Mandatory Access Control Architecture

**Status (July 2026):** the MAC library is complete and well-tested; **runtime enforcement is not yet wired into production paths** — the PDP has no production PEP caller yet, the grant path's HTTP dispatch fails closed pending resolver/object-label wiring, and the audit store requires explicit startup construction. Everything fails closed (nothing is bypassable), but do not assume MAC is actively protecting a resource until the enforcement wiring (issues #673–#676) lands. Epic: #547.

HyprStream's MAC layer is a from-scratch, label-based mandatory access control system for the 9P/VFS data plane. It sits **beneath** the RPC authorization stack (Casbin policy + JWT scopes + UCAN delegation): those are the *control plane* that decides what to grant; MAC is the *mandatory floor* that no control-plane grant can bypass.

## Where the code lives

| Module | Crate path | Role |
|---|---|---|
| `label`, `lattice`, `context`, `genesis`, `manifest` | `crates/hyprstream-rpc/src/auth/mac/` | Canonical label types, lattice vocabulary, subject context |
| `ucan` (`token`/`chain`/`capability`/`approval`) | `crates/hyprstream-rpc/src/auth/ucan/` | Capability delegation chains (grant *authoring*) |
| `te`, `avc`, `compiler`, `permission_map`, `compiled`, `exchange`, `audit`, `lattice` (re-export) | `crates/hyprstream/src/mac/` | The PDP, cache, compiler, grant exchange, audit |

## The control-plane / data-plane split

```text
CONTROL PLANE (authoring / compilation)             DATA PLANE (per-op enforcement)
───────────────────────────────────────             ───────────────────────────────────
UCAN                  grant/delegation source
Casbin Enforcer       policy STORE + AUTHORING       ── NEVER on the per-op hot path ──
S5 UCAN→TE compiler   lowers UCAN/Casbin → matrix ─► TeMatrix          (compiled policy)
PolicyService         compiles + SIGNS policy ─────► compiled::SignedPolicy
                                                      │ loader verifies ONCE (hybrid PQC)
                                                      ▼
                                                      te::LatticeTeEvaluator   (the PDP)
                                                      avc::CachingAvc          (the cache)
                                                      ▲ PEP calls this per op
```

Two invariants define the split:

1. **Casbin never runs per-op.** The `casbin::Enforcer` (in `auth::policy_manager::PolicyManager`) remains the policy store and authoring surface, but its string matcher is never invoked on the hot path. The S5 compiler lowers authored policy into a compact `TeMatrix` (interned-id allow-set, O(1) lookup) that the PDP evaluates.
2. **The lattice floor is independent of Casbin entirely.** Even a compiled matrix entry cannot authorize an access the label lattice denies.

## Labels and subjects

- **`SecurityLabel`** = Level × Assurance × CompartmentSet (`Copy + Ord + Hash`, deliberately no `Default`). Dominance and join are intrinsic methods on the label — the access check is `can_access` (Bell–LaPadula framing lives in the rustdoc, not the API names).
- **`Lattice`** is a versioned policy object: a *closed* compartment name↔bit vocabulary. Unknown compartments are errors, not extensions.
- **`SecurityContext`** (subject clearance) is derived from verified `Claims` × `VerifiedKeyMaterial` — **never from Claims alone**. The Assurance axis reflects how the subject authenticated (e.g., classical vs. PQ-hybrid key material), so a weakly-authenticated bearer of strong claims does not get strong clearance.
- **Delegation** (in progress, #680/#681): delegated calls derive a *two-principal* context — the effective clearance is the meet of delegator and delegatee, with signer-assurance carried through and both principals recorded in audit (`on_behalf_of`).

## Assurance is a labeling requirement, not an opt-in check (#698 Decision D)

The Assurance axis of `SecurityLabel` is not decorative — it is the enforced boundary between classical- and PQ-verified subjects, and it is **structural**: the lattice floor (`can_access`) requires the subject's assurance to meet the object's, independently of the TE matrix and with no per-rule opt-out. A TE rule can never grant a Classical subject access to a PqHybrid-labeled object.

This matters concretely for **delegated actors** (a UCAN grant whose audience is not the chain root — e.g. MCP acting on a user's behalf). A delegated actor's sender-binding today is DPoP proof-of-possession of a classical ephemeral key; nothing binds that key to a PQ-anchored DID identity (UCAN chain validation cryptographically proves *issuers*, never the audience). The `EnrollmentSubjectContextResolver` (`crate::mac::exchange_enrollment_resolver`, backed by the signed policy's enrollment table, #698 PR A) therefore **floors every delegated actor's assurance at `Classical`**, unconditionally — the truthful label for what the crypto actually proves. Assigning anything higher would let an unverified actor claim `PqHybrid` assurance, defeating the #548 invariant.

Consequence for policy/label authors: **an object labeled with a `PqHybrid` assurance requirement is structurally unreachable by any delegated actor.** This is a feature, not a limitation to route around — it makes the classical/PQ trust boundary auditable in the signed policy artifact itself (grep the label set for `PqHybrid` to see everything no delegated actor can ever reach), rather than scattered through enforcement code. Objects that legitimately need to be reachable by delegated actors **must** carry an assurance requirement of `Classical` or lower.

Raising a *specific* enrolled actor above the Classical floor is the sanctioned upgrade path: enrollment-time actor-key registration (#718), which extends the enrollment table with a registered, PQ-anchored key per DID and only relaxes the floor on a matched key — never as a blanket default. Self-issued root grants are unaffected by any of this: there the subject *is* a chain-verified issuer, so `PqHybrid` assurance remains directly provable via `GlobalPqUcanVerifier`.

## Per-op TCB (deliberately tiny)

A decision on the hot path is: one hash lookup in the **AVC** (`avc::CachingAvc`, sub-µs on a hit — no Casbin, no signature verification, no lattice walk), and on a miss, a set lookup in the `TeMatrix` plus one intrinsic `SecurityLabel::can_access` call in the **TE evaluator** (`te::LatticeTeEvaluator` — a TOTAL, default-deny `(subject_ctx, object_label, action) → Decision`). All heavy or bug-prone logic — UCAN chain validation, Casbin matching, compilation, signature verification — is concentrated off the hot path in PolicyService and the `compiled` loader.

## The compiler and the no-escalation check

`mac::compiler` + `mac::permission_map` lower UCAN/Casbin grants into the TE matrix:

- `PermissionMap` is the scope↔TE-rule vocabulary seam. The production impl (`ScopePermissionMap`) is **injective and exact by construction**: wildcards expand at compile time over a closed, sorted registry, so `granted_access` is trivially the most-permissive access — there is no join/LUB logic that could mask an escalation.
- `check_no_escalation` verifies the emitted matrix against the *independent* grant — never by re-running the forward map. (A deferred SMT proof is tracked as `TODO(#571)`; do not recreate the deleted trait-heavy `seam.rs`/`smt` scaffolding.)

## Signed policy distribution

`mac::compiled` — PolicyService hashes and signs the compiled matrix with the **hybrid PQC COSE composite** (EdDSA + ML-DSA-65 via `hyprstream_rpc::crypto::cose_sign`); the loader rejects unsigned, mismatched, or unapproved policy. Verification happens once at load, never per-op. Under `CryptoPolicy::Hybrid`, a missing PQ key fails closed at sign time — it never silently downgrades to classical.

## Grant exchange (ZSP)

`mac::exchange` implements the runtime grant path under **zero standing privilege**: a presented UCAN subset-grant is exchanged for a short-TTL, sender-bound (DPoP) OAuth access token, re-evaluated on every refresh, never minting more than the requested subset. A token is effectively a distributed AVC entry.

## Tamper-evident audit

`mac::audit` — `WalAuditStore` is an append-only, hash-chained (`prev_hash`), signed-checkpoint-anchored journal. `AuditedAvc` wraps any AVC so that **a decision that cannot be durably audited is downgraded to Deny** (fail-closed), never silently permitted. Delegated calls record both principals (`on_behalf_of`).

## Interaction with the VFS

Every `Mount` call in `hyprstream-vfs` carries a `Subject` (`caller: &Subject` on the trait), and composition is `Namespace::bind_mount(prefix, Arc<dyn Mount>, BindFlag)` — see ADR #651. This Subject propagation is MAC-load-bearing: a mount that loses the Subject (e.g., across a FUSE boundary) cannot participate in per-op MAC decisions and must be treated as a leaf with a single label. Federation-facing label metadata in Cap'n Proto messages is a **hint, not a guarantee** — trusted only for services we operate; imported objects clamp to the perimeter floor and bind labels clamp as `join(binder_floor, declared)`.

## Naming conventions (settled — don't relitigate)

`ceiling`→`grant`, `dominates`→`can_access`, `witness`→`granted_access`, `rules_for`→`permissions_for`. Plain MAC/RBAC vocabulary throughout; Bell–LaPadula/academic terms only in rustdoc explanations.

## Related reading

- `crates/hyprstream/src/mac/mod.rs` — the authoritative module-level rustdoc (control/data-plane split, TCB analysis)
- [`cryptography-architecture.md`](cryptography-architecture.md) — COSE hybrid signing, PQ trust anchoring
- [`vfs.md`](vfs.md) — the namespace/`Subject` model MAC enforces over
- CLAUDE.md "Native MAC" section — condensed developer guidance
