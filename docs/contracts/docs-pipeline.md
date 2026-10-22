# Documentation pipeline contract

`docs/system-ontology.md` remains the authoritative vocabulary and machine-readable ontology.

## Docs-catalog attestation gate: removed

The docs-catalog attestation gate was removed by owner decision (2026-09-19). It is not what the owner wanted, and it will be replaced by a docs publishing pipeline (follow-up PRs). Removed in this change:

- `.github/workflows/docs-catalog.yml` — the advisory Docs catalog workflow.
- The `Validate docs catalog for merge candidate` step in `.github/workflows/rust.yml`.
- `tools/check-docs-catalog.py` and `tools/test-docs-catalog.py` — the verifier, mutation suite, and Git fixtures.
- `docs/schema-catalog.json` and `docs/corpus-sources.json` — the attested manifests.

The gate's provenance layer required a manual re-attestation commit for any audited-input change (license policy, manifests, docs), which was toil without corresponding value.

## Replacement: docs publishing pipeline (follow-up PRs)

The replacement publishes documentation instead of gating merges. Everything below is **not yet landed**; it arrives in follow-up PRs:

1. `.github/workflows/docs-publish.yml` builds full-workspace rustdoc on pushes to `main`.
2. The workflow packs `target/doc` into a CAR with a pinned `ipfs-car` binary.
3. The CAR plus a manifest are delivered to the GitLab generic package registry.
4. Delivery triggers the metal/ingest promotion pipeline, which re-derives the CID from the CAR, pins it to the kubo gateway over the mesh, and updates the DNSLink TXT record in PowerDNS.

This pattern is proven by cyberdione-corp's `ipfs-publish` and ingest's `dnslink-promoter`.
