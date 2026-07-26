# MAC identity-aware activation runbook

Identity-aware MAC is an operator widening, not a boot default. Production
starts with every PEP installed and the subject context narrowed to
`anonymous_floor()`. The control changes only the context supplied to those
PEPs; it cannot uninstall a monitor or enter a permissive/audit-only mode.

## Evidence gate

Capture and retain one reviewed evidence bundle covering every epic gate:

1. G1: `hyprstream mac genesis` reports `COMPLETE`, with zero `unlabeled` and
   zero `ill_formed` nodes.
2. G2: the PEP/TOCTOU audit proves every live path is mediated and the labeled
   identity/object is bound to the subject/object actually served. Include the
   `VerifiedAttach` divergence-denial test.
3. G3: a staging exercise widens and then calls `narrow_to_floor`; the PEP
   remains installed and denials remain audited.
4. G4: denial-to-proposal handling passes end to end.
5. G5: deny audit export, metrics, and alerting are live.
6. G6: the release validation and Kimi security review are attached, and the
   named operator signs off this runbook.
7. G7: revocation, policy/resolver reload, and AVC flush are demonstrated.

`MacActivationControl::widen_identity_aware` accepts a
`MacActivationEvidence` containing G1-G7 attestations and refuses the widening
if any gate is false. No startup path constructs that evidence or calls the
widening method.

## Widen

The live daemon's operator control plane must:

1. Recompute `GenesisGate::production().report()` in the target process.
2. Verify the signed evidence bundle and operator authorization.
3. Construct `MacActivationEvidence` with that report and the approved G2-G7
   attestations.
4. Call
   `global_mac_activation_control().widen_identity_aware(&evidence)`.
5. Confirm the mode is `IdentityAware`, exercise representative RPC, 9P, CAS,
   VFS, exemptions, and MoQ/event traffic, and watch deny telemetry.

Do not treat a successful test fixture or a `mac genesis` report alone as
authorization to widen.

## Narrow-back kill switch

An authorized operator may always call
`global_mac_activation_control().narrow_to_floor()`. Confirm the mode is
`FloorOnly`, verify that each PEP remains installed, and retain the signed audit
records around the transition. Narrowing requires no coverage or health
precondition so it remains available during an incident.

## Current decision

Leave production floor-only. A release is activation-ready only after its
validation evidence is attached, but the human widening decision remains
blocked until all G1-G7 evidence above is green. In particular, an incomplete
genesis report or unfinished revocation/reload evidence makes widening return
an error.

The 2026-07-26 default-config evidence snapshot is **INCOMPLETE**: 93 nodes,
23 labeled, 70 unlabeled, and 0 ill-formed. That snapshot is evidence to keep
the gate closed, not approval to widen; recompute it in the target deployment.
