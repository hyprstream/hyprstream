# CAS/blob and 9P interoperability reports

**Status:** pre-construction (2026-07-17). These are scoped evidence plans, not completion claims. They demonstrate only the resource-specific layers exercised. The stock-relay and stock-carrier evidence shared with the #1058/#1059 draft family remains owned by those drafts.

## Boundary

The first vertical of resource ownership is immutable content-addressed storage write-then-seal (#1066). Mutable 9P title, directory-entry, rename, hard-link, bind/mount, and transfer semantics follow only after the blob lifecycle and the crash/reconciliation suites pass (#1071). A CAS or blob integration must not expose provisional bytes as a finalized resource; a 9P operation must not confer title that the registrar has not finalized.

## CAS/blob reference interoperability plan (owned with #1066)

**Goal:** demonstrate that a finalized manifest's `content_cid` matches the bytes sealed by the content-addressed store, and that provisional bytes are not exposed as finalized.

1. **Topology.** A content-addressed store, a controlled registrar, and two clients (authorized and unauthorized). The store has no title, MAC authority, ledger authority, or registrar state.
2. **Write-then-seal.** Seal a blob; record its content CID. The manifest's `content_cid` must equal the sealed CID (vector control: `content-cid-mismatch` rejects).
3. **Provisional isolation.** Bytes from a `Materialized`-but-not-`Finalized` transition must be unavailable through the normal namespace (vector control: `cas-exposes-provisional` rejects).
4. **Negative substitutions.** Mutate sealed bytes and assert the content CID no longer matches; mutate the manifest `content_cid` and assert rejection before finalization.
5. **Evidence.** Save sanitized content CIDs, byte lengths, store version/config hash, registrar decision trace with opaque IDs, and negative-control outcomes. Do not retain raw holder identity, plaintext keys, or production credentials.

## 9P ownership-semantics reference interoperability plan (owned with #1071)

**Goal:** demonstrate that mutable 9P title operations follow only after the blob lifecycle and the crash/reconciliation suites pass, and that no 9P operation confers unfinalized title.

1. **Topology.** A 9P/VFS namespace, a registrar, and a content store. The 9P server has no independent title authority.
2. **Title-after-finalize.** A rename, hard-link, bind/mount, or transfer succeeds only when the registrar state is `Finalized` for the cited resource and version (vector control: `ninep-confers-unfinalized-title` rejects).
3. **MAC-only reads.** Ordinary reads must remain MAC-only; no ledger or proof-plane I/O on the read path.
4. **Negative substitutions.** Present a 9P title op for a provisional/unfinalized resource and assert denial; present a title op with a crossed profile and assert denial.
5. **Evidence.** Save sanitized 9P trace, registrar decision trace, MAC decision trace, and negative-control outcomes. Record the 9P server binary/config hash.

## Status

Both plans are pre-construction; the implementations (#1066 CAS, #1071 9P) are not complete. The boundary checker enforces the structurally checkable rules today (`cas-exposes-provisional`, `cas:content-cid-mismatch`, `ninep:title-without-finalized`) and refuses a structurally valid fixture with `construction-incomplete`. No carrier-transport interoperability is claimed here; that evidence belongs to #1058/#1059.
