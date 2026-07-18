# ADR RO-001: One canonical claim, split authority

**Status:** Accepted for review (#1067)

## Decision

A resource transition has one canonical `ResourceIntent`. Native MAC alone owns title/control authorization; the cellular ledger alone owns entitlement accounting; the resource registrar alone owns lifecycle state and successor selection. MAC and ledger sign the identical intent digest. Storage, namespace, and PDS expose material/projections/evidence but never infer title.

## Consequences

Payment cannot transfer title, authorization cannot manufacture credit, and no attestation can finalize independently. Operational co-location does not collapse typed roles. The registrar verifies both attestations but cannot mint either. See the exactly-one-owner table in `docs/resource-ownership-architecture.md`.
