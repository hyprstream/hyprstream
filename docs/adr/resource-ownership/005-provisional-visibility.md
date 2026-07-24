# ADR RO-005: Visibility follows registrar state

**Status:** Accepted for review (#1067)

## Decision

Only `Finalized` resources appear through the ordinary CAS/9P namespace. Staged, sealed, reserved, posted, quarantined, and manual-review material is hidden. Recovery tooling may access quarantine through a separately authorized namespace that cannot be confused with normal publication.

The visibility gate is enforced by construction: `FinalizedResource` is constructible only by verifying a registrar-signed finalization statement against accepted-current registrar authority and matching its two attestation CIDs to verified MAC/ledger evidence. Only that private value mints the private-field public projection and ordered effect, so no namespace adapter, recovery tool, or test helper can fabricate a publish. Publication contains no attestation CIDs, payer, issuer, unit, amount, transfer ID, capability binding, or signer-key coordinates; every publish/withdraw carries operation ID, fencing token, resource version, optional entry identity, and (for withdraw) its exact target.

## Consequences

Byte existence, a PDS record, a single attestation, or an unauthenticated/invalid finalization cannot publish a resource. A valid signature produced during an undetected current-key compromise follows the dispute/revocation path; verification does not erase that residual risk. Finalize-before-publish uses a durable idempotent outbox, so publication failure creates temporary unavailability rather than premature visibility; a delayed outbox effect cannot overwrite a newer projection. Reads of finalized resources remain MAC-only and do not synchronously consult the ledger/PDS.
