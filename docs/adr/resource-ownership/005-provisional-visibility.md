# ADR RO-005: Visibility follows registrar state

**Status:** Accepted for review (#1067)

## Decision

Only `Finalized` resources appear through the ordinary CAS/9P namespace. Staged, sealed, reserved, posted, quarantined, and manual-review material is hidden. Recovery tooling may access quarantine through a separately authorized namespace that cannot be confused with normal publication.

The visibility gate is enforced by construction, not convention: `FinalizedResource` is constructible only by verifying a registrar-signed finalization statement against accepted-current registrar authority, so no namespace adapter, recovery tool, or test helper can fabricate one. Publication consumes only the privacy-minimized public projection — no attestation CIDs, payer, issuer, unit, amount, transfer ID, capability binding, or signer-key coordinates — and every publish/withdraw effect is an ordered desired-state event carrying operation ID, fencing token, and resource version, applied by compare-and-apply that rejects stale generations (architecture §6).

## Consequences

Byte existence, a PDS record, or a single attestation cannot publish a resource, and a forged finalization value cannot either. Finalize-before-publish uses a durable idempotent outbox, so publication failure creates temporary unavailability rather than premature visibility; a delayed outbox effect can never overwrite a newer projection. Reads of finalized resources remain MAC-only and do not synchronously consult the ledger/PDS.
