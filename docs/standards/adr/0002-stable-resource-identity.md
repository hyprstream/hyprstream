# ADR 0002 — Stable resource identity is independent of content, manifest, and operation

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §identity.

## Context

If a resource identifier is defined as a hash or signature over its content, manifest, or operation identifier, then changing content, recomputing a manifest, or reissuing an operation would mutate the resource identity, breaking stable ownership, transfer, and history verification. It would also create a cyclic hash/signature construction with no clean verification order.

## Decision

Stable resource identity (`resource_id`), content CID (`content_cid`), manifest CID (`manifest_cid`), and operation ID (`operation_id`) are **separate, independent values**. A construction must not define one as a hash or signature over another. A finalized manifest cites all four independently.

## Consequences

- `resource_id` is stable across content, manifest, and operation changes; ownership and transfer bind to it.
- `content_cid` binds to sealed bytes; `manifest_cid` is self-addressed; `operation_id` is deterministic for idempotency.
- Vector `cyclic-identifier` rejects an identifier defined as a hash of another (RA-REQ-001).
- Every request and external effect is idempotent by deterministic operation ID (RA-REQ-003/004).

## Alternatives considered

- **Content-addressed resource identity:** rejected — content mutation would change ownership.
- **Self-signed manifest as identity:** rejected — creates a cyclic construction and prevents clean history verification.
