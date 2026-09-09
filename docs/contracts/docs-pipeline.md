# Documentation pipeline contract

`docs/system-ontology.md` is the authoritative vocabulary and machine-readable ontology. This contract and every generated projection consume it; none may redefine a canonical noun, action, identity, or path.

## Version 1 inputs and outputs

`docs/schema-catalog.json` is the complete, fail-closed inventory of every tracked Cap'n Proto source. It records the file identity, owning package and source license, whether it is a service/type/annotation/protocol/test fixture, CGR producer, and deliberately different consumer surfaces. A surface state is `active` only when a repository source registers it. `declared` means a future/package contract exists; `absent` means no active source is present. A schema that is compiled or has CGR is not thereby a CLI, MCP, VFS, or TypeScript API.

The API manifest is JSON version 1. Entries use stable IDs `api:{service}:{method}`, stable paths `api/v1/{service}/{method}.json`, canonical UTF-8 JSON, SHA-256 integrity, and a 1 MiB entry limit. The complete corpus manifest uses `doc:{repository-relative-path}`, paths `corpus/v1/{document_id}.md`, SHA-256 of source bytes, and a 2 MiB document limit. Both manifests must include source revision, license, provenance, size, and digest; a publisher rejects a missing or mismatched value.

`docs/corpus-sources.json` is the public-prose allowlist. It intentionally excludes coordination state, secrets, unpublished plans, editor state, generated/release outputs, and separately licensed RFC renderings. A new source path is not publishable until it is classified with provenance and license review.

The future `@hyprstream/docs` package is `declared-not-yet-published`: it will export the validated manifest, versioned API entries, and corpus entries only. It must preserve provenance and source license notices and must not copy generated content into a different license boundary. The permissive `hyprstream-docs`, VFS, RPC, and codegen paths remain separated from AGPL application integration as required by `.github/license-boundary.toml`.

## Verification

Run `python3 tools/check-docs-catalog.py`. The checker asks Git for tracked `.capnp` files, so a new schema and a stale removed schema both fail. It compares the catalog to the actual CLI builder, MCP registrations, factory attributes, VFS dispatch registrations, and tracked TS/JS sources. It rejects duplicate source IDs or service identities, omitted exclusion reasons, unknown surface states, unauthorized public-prose patterns, and a catalog that stops naming the ontology as authority.

The consumer sets are intentionally not equal: CLI omits hidden and streaming methods at runtime; MCP has its own registration and policy boundary; VFS has dispatch registrations rather than a promise that every service is mounted; and TypeScript has no tracked package at the pinned base. Update source and catalog together, then run the checker and its mutation probes with `python3 tools/check-docs-catalog.py --self-test`.
