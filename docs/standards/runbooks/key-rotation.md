# Runbook: MAC/ledger key rotation and key-history verification

**Status:** pre-construction (2026-07-17). Operational target for #1068/#1072.

## Requirement

Key-release and key-rotation history verification must not depend on a single classical signature over the resource history (RA-REQ-024). All security-critical artifacts sign via the hybrid-PQC COSE composite (EdDSA + ML-DSA-65), per the project-wide policy.

## Rotation procedure

1. **Introduce.** A new MAC title/control key (or ledger attestation key) is introduced at a new policy generation; the previous key enters a drain window (mirrors `KeyDraining`).
2. **Pin policy epoch.** Each attestation records its `policy_generation` and `accepted_state_epoch`; a stale or rolled-back generation rejects (vectors `stale-accepted-state`, `policy-rollback`).
3. **Append-only history.** Rotations are recorded in the tamper-evident audit journal as hash-chained, checkpoint-anchored entries signed with the composite.
4. **Verify history.** Verification walks the checkpoint chain and the composite signatures; it never trusts a single classical signature over the whole history.

## Fail-closed

Under the hybrid policy, a missing PQ key fails closed at sign time — it never silently downgrades to classical. A key-history gap cannot promote a `Public` or unlabeled object to bearer access (content-bound labels, #699).
