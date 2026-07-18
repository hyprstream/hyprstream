# ADR RO-003: Stable resource and operation identities

**Status:** Accepted for review (#1067)

## Decision

A resource gets a random opaque 128-bit `resource_id` at create. It is independent of owner, path, content CID, and manifest CID and survives mutation, rename, hard links, and title transfer. A deterministic 128-bit `operation_id` identifies one logical transition and its retries. Reuse with different canonical intent bytes is an invariant violation.

## Consequences

Paths remain namespace entries, content remains immutable data, and manifests remain version facts. Ledger transfer IDs and staging IDs are domain-separated derivatives of operation ID. A resource-ID collision rejects rather than being interpreted as retry; operation ID, not resource ID, carries idempotency.
