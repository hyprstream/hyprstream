# Key-history verification analysis

**Status:** pre-construction (2026-07-17). Conformance closure view. Key-history verification and hybrid-PQC key rotation for resource history are owned by #1068/#1072.

## Requirement

Key-release and key-rotation history verification must not depend on a single classical signature over the resource history. This is traced as obligation RA-REQ-024 (specification-only).

## Rationale

A single classical signature over the resource history is a quantum-vulnerable single point of failure: an adversary who recovers a classical signing key can rewrite history undetectably. The resource-attestation profile therefore requires that history verification compose across hybrid-PQC signatures, consistent with the project-wide hybrid-PQC signing policy (EdDSA + ML-DSA-65 nested COSE), and that key rotation preserve a verifiable, append-only chain.

## Design inputs (owned by #1068/#1072)

- Hybrid-PQC COSE composite signatures for attestation and checkpoint records, mirroring `hyprstream_rpc::crypto::cose_sign::{sign_composite, verify_composite}`.
- Policy-epoch pinning and revocation for MAC title/control keys (#1068).
- Per-rotation key history anchored at tamper-evident checkpoints (the `WalAuditStore` hash-chained journal pattern).
- Content-bound labels (#699) so a key-history gap cannot promote a `Public` or unlabeled object to bearer access.

## Negative controls (structural, in the vector set)

The boundary checker does not model a full key-history graph, but the downgrade and replay controls prevent the obvious key-history weakening paths: a classical-claims-PqHybrid attestation rejects (`downgraded-assurance`), and a replayed operation cannot rebind to a rotated key (`replayed-operation-id`, `replay-across-resource`).

## Residual risk

Until #1068/#1072 implement the hybrid-PQC key-rotation chain and the checkpoint-anchored history, key-history verification is specification-only. No production resource history is verifiable under this profile today. See `resource-residual-risk.md`.
