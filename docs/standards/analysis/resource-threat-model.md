# Resource threat model

**Status:** pre-construction (2026-07-17). Conformance closure view; the design threat model is owned by #1067. This document records the adversaries, assets, and properties the boundary checker and vectors exercise. It makes no security claim beyond what the reviewed implementation will provide once the sibling issues land.

## Assets

- The canonical `ResourceIntent` and its digest.
- The MAC title/control attestation and the ledger economic attestation.
- The finalized manifest and the registrar state (predecessor/version, fencing generation).
- Stable resource identity, content CID, manifest CID, operation ID.
- Holder pseudonyms, commitments, nullifiers, and issuer liabilities.

## Adversaries

| Adversary | Capability | Profile defense |
|---|---|---|
| Unauthorized principal | forge or replay an attestation | both attestations over identical canonical digest; idempotent operation ID; fencing token |
| Confused deputy | confuse MAC authority with payment, or payment with title | distinct typed roles; registrar joins, does not invent |
| Colluding issuer + origin + ledger | link an anonymous holder across redemption | commitments/nullifiers in redemption-side accounting; no stable holder DID in public receipt or audit |
| Cross-resource/cross-operation replay | reuse an attestation for a different resource or operation | digest binds resource_id and operation_id; replay rejects |
| Downgrade adversary | claim PqHybrid from a classical outer admission | effective assurance is the minimum across all composed assurances |
| Crash/restart | duplicate a finalized transition, mint a second successor | compare-and-swap with fencing token; idempotent operation ID; durable spend-before-dispatch |
| Concurrency racer | finalize two successors for one predecessor | single finalized successor invariant; fencing |

## Properties the vectors exercise

The deterministic vector set in `vectors/resource-intent-canonicalization-v1.json` exercises: canonicalization rejection, replay (operation ID and cross-resource), crossing (digest, fencing token, profile, expiry), downgrade (assurance and state rollback), privacy leakage (anonymous-fabricates-DID and public-receipt-reveals-holder), crash recovery (crash-repeats-finalized, missing fencing token), and concurrency fencing (concurrent successors, stale predecessor, crossed/missing fencing token).

A structurally valid fixture still terminates with `construction-incomplete`. The vector set is a non-cryptographic structural boundary, not a cryptographic proof; it contains no signature, MAC, ledger attestation, or production credential.

## Out of scope

No unlinkability claim against issuer-origin-ledger collusion beyond data minimization. No traffic-analysis resistance. No protection against a compromised registrar host or compromised TCB. These are residual risks recorded in `resource-residual-risk.md` and gated on the sibling implementations and independent review.
