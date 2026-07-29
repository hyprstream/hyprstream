# Public Rust crate release policy

This document defines the supported crates.io surface for this repository. The
machine-readable source of truth is [`.github/crate-release.toml`](../.github/crate-release.toml),
and `scripts/check_crate_release_policy.py` fails closed if any package is absent,
duplicated, or publishable outside the allowlist.

## Registry, namespace, and support boundary

- Registry: the default Cargo registry, `crates-io` (`https://crates.io`). No
  alternate registry or source replacement is part of this release channel.
- Namespace: crates.io has one flat, global, first-come-first-served namespace;
  it has no organization scope analogous to npm. A `hyprstream-*` prefix is a
  naming convention, not an access-control boundary.
- Initial supported crates, in dependency/publish order:
  1. `hyprstream-rpc-build` `0.1.0`
  2. `hyprstream-rpc-derive` `0.1.0` (exactly depends on
     `hyprstream-rpc-build =0.1.0`)
- Everything else sets `publish = false`. Looking reusable is not an API or
  maintenance commitment. Promotion to `public_now` requires an owner, API and
  SemVer review, package documentation, security review where applicable,
  crates.io name ownership, versioned local dependencies, and a successful
  packaged-source build.

The existing crates.io package `hyprstream` is owned by the project maintainer
account but is **not** the local application release channel; the current
application crate remains internal/non-publishable. The name `chat-core` is
owned by an unrelated crates.io publisher and cannot be used by this project.

## Complete inventory

| Crate | Class | Reason / promotion gate |
|---|---|---|
| `hyprstream-rpc-build` | **public now** | Documented build/codegen API; packages independently; first in release order. |
| `hyprstream-rpc-derive` | **public now** | Documented proc-macro API; exact versioned dependency on `hyprstream-rpc-build`. |
| `bitsandbytes-sys` | public later | External CUDA/ROCm FFI/toolchain contract needs ownership and platform support policy. |
| `cas-serve` | public later | XET git-source dependency and missing package docs. |
| `git-xet-filter` | public later | XET git-source/default-feature dependency chain; API review required. |
| `git2db` | public later | XET git-source/default-feature chain and broad storage API commitment. |
| `hyprstream-9p` | public later | Depends on unpublished RPC/VFS crates; wire/API stability review required. |
| `hyprstream-compositor` | public later | Missing package docs and explicit external API owner. |
| `hyprstream-containedfs` | public later | Security-sensitive path-containment contract needs dedicated review and support owner. |
| `hyprstream-crypto` | public later | Cryptographic API requires independent security/API review before external commitment. |
| `hyprstream-discovery` | public later | Deep unpublished dependency graph and security-sensitive discovery/admission surface. |
| `hyprstream-flight` | public later | Depends on unpublished metrics/git2db stack; missing package docs. |
| `hyprstream-k8s` | public later | CRD/version compatibility is a public API; ownership and compatibility policy required. |
| `hyprstream-ledger` | public later | Accounting/security semantics need independent review and an API owner. |
| `hyprstream-metrics` | public later | Depends on unpublished git2db and native data stack; missing package docs. |
| `hyprstream-p2p` | public later | Depends on unpublished RPC; network protocol/support commitment not yet declared. |
| `hyprstream-pds` | public later | Security/signing and atproto storage APIs need review; unpublished dependencies. |
| `hyprstream-pds-service` | public later | Tenant service depends on unpublished PDS/RPC/VFS stack. |
| `hyprstream-resource` | public later | Early authority/lifecycle interface scaffold; API not committed. |
| `hyprstream-rpc` | public later | Large security/wire API plus unreleased patched `moq-net`; not ready for SemVer support. |
| `hyprstream-rpc-std` | public later | Generated schema/client compatibility depends on unpublished RPC/VFS/worker crates. |
| `hyprstream-service` | public later | Depends on unpublished RPC/PDS and evolving orchestration API. |
| `hyprstream-util` | public later | Packages cleanly, but generic internal helpers lack an external API owner/docs commitment. |
| `hyprstream-vfs` | public later | Core namespace/security API depends on unpublished RPC; stability review required. |
| `hyprstream-vfs-server` | public later | Linux/FUSE/vhost-user platform contract and unpublished dependency chain. |
| `hyprstream-workers` | public later | Git-source Kata/Nydus dependencies and a broad sandbox security/platform contract. |
| `hyprstream-workers-python` | public later | Depends on unpublished VFS/wasmtime stack; guest/runtime compatibility policy required. |
| `hyprstream-workers-tcl` | public later | Custom git-source `molt` fork and unpublished VFS dependency. |
| `hyprstream-workers-wasmtime` | public later | Sandbox security API and unpublished RPC/VFS dependencies need review. |
| `waxterm` | public later | Packages cleanly, but lacks package docs and an explicit standalone compatibility owner. |
| `chat-core` | **internal/not publishable** | Current crates.io name belongs to an unrelated publisher; rename and API review required. |
| `hyprstream` | **internal/not publishable** | Product application, not an SDK; git dependencies and the full internal graph are unsuitable for crates.io. |
| `hyprstream-appview` | **internal/not publishable** | Product-internal derived identity service with no standalone support contract. |
| `hyprstream-tui` | **internal/not publishable** | Product UI/application crate, not a supported library API. |
| `hyprstream-workers-python-guest` | **internal/not publishable** | Compiled WASM guest artifact, not a Rust consumer library. |
| `hyprstream-workers-wasmtime-fsguest` | **internal/not publishable** | Test/validation WASI guest binary, not a consumer package. |

## Versioning and release tags

The initial public crates use a synchronized release train because the derive
crate consumes the build crate's generated structures. A release tag is
`crates-v<SemVer>`, for example `crates-v0.1.0` or
`crates-v0.2.0-rc.1`. The workflow requires:

1. a protected **annotated** tag;
2. a tag target reachable from protected `main`;
3. an exact match between tag SemVer and every `public_now` manifest version;
4. dependency order and exact sibling version pins from the policy checker.

Stable tags are immutable production releases. crates.io versions are
permanent: they cannot be overwritten or deleted (yanking only changes future
resolution). Never move or reuse a release tag/version after a partial release;
fix forward with a new version.

crates.io does **not** provide a meaningful main-branch staging channel. Main
runs validation only. When registry-level consumer testing is needed before a
stable release, publish an intentionally immutable SemVer prerelease such as
`0.2.0-alpha.1` or `0.2.0-rc.1` from the matching protected tag. It is not an
ephemeral snapshot and must not be called “staging.”

## Trusted publishing and one-time bootstrap

The release workflow is `.github/workflows/publish-crates.yml` and the protected
GitHub environment is `crates-io-publish`. It requests only `contents: read`,
`id-token: write`, and `attestations: write`; checkout credentials are not
persisted. `rust-lang/crates-io-auth-action` exchanges the GitHub OIDC JWT for a
short-lived, job-scoped crates.io token and revokes it in its post step.

Trusted-publisher tokens cannot create new crate names. Therefore a verified
crates.io owner must perform the **first** publish once, manually, in dependency
order from the reviewed release commit (commands shown for the operator; CI and
this policy change do not execute them):

```bash
cargo package --locked -p hyprstream-rpc-build
cargo publish --locked --registry crates-io -p hyprstream-rpc-build
# Wait until cargo search/info resolves 0.1.0 from crates.io.
cargo package --locked -p hyprstream-rpc-derive
cargo publish --locked --registry crates-io -p hyprstream-rpc-derive
```

The local authenticated owner is currently `ewindisch`; the token is stored by
Cargo for the default registry and must never be copied into GitHub secrets or
logs. After each bootstrap package exists, configure its crates.io **Settings →
Trusted Publishing** entry exactly as follows:

| Field | Value |
|---|---|
| GitHub owner | `hyprstream` |
| repository | `hyprstream` |
| workflow filename | `publish-crates.yml` |
| environment | `crates-io-publish` |

Do this separately for `hyprstream-rpc-build` and
`hyprstream-rpc-derive`. Add a second accountable maintainer/team owner where
organizational policy permits, verify all owner emails, test one prerelease,
then enable crates.io's **require trusted publishing** control for each crate so
long-lived API tokens cannot publish updates. Retain a documented break-glass
owner procedure; never store a crates.io token in repository or environment
secrets for routine releases.

GitHub repository setup (not performed by this change):

1. Create environment `crates-io-publish`; allow deployments only from protected
   `crates-v*` tags and require release-maintainer approval.
2. Create an active tag ruleset matching `refs/tags/crates-v*`; restrict tag
   creation to release maintainers and block updates, force-pushes, and deletion.
3. Protect `main` with the existing merge queue/checks. Review changes to the
   publish workflow, policy, scripts, or public manifests as release-security
   changes.
4. Ensure unrelated cloud OIDC trust policies do not accept the publish
   environment subject. The crates.io OIDC binding is repository owner/name,
   exact workflow filename, and exact environment.

## Verification, provenance, and smoke test

For every crate in order, CI performs Cargo's full `cargo package --locked`
packaged-source build (no `--no-verify`), rejects archives over 10 MiB, records
`cargo package --list`, and computes SHA-256. On retry after a partial run, an
existing crates.io version is accepted only when its registry checksum exactly
matches the local `.crate`; a mismatch fails closed. New uploads wait for index
propagation before the next dependent crate.

GitHub artifact attestations bind each `.crate` digest to the protected workflow
run and release commit. The `.crate` files, content inventories, and
`SHA256SUMS` are retained as workflow evidence. Consumers verify downloaded
evidence with:

```bash
gh attestation verify hyprstream-rpc-build-0.1.0.crate \
  --repo hyprstream/hyprstream
sha256sum -c SHA256SUMS
```

The post-publish smoke test creates a new temporary Cargo project with exact
`=<version>` dependencies, generates a fresh lockfile, runs `cargo check
--locked`, and asserts metadata sources are registry URLs rather than workspace
paths or Git sources.

## Browser WASM/npm coordination

The browser npm publisher is intentionally unchanged. The two channels must
share these policy invariants without sharing credentials or pretending the
registries have identical staging semantics:

- production npm and Rust artifacts must identify the same protected source
  commit and compatible RPC/schema revision;
- each registry uses its own least-privilege trusted publisher/provenance and
  immutable version;
- checksums/content manifests and clean external-consumer smoke tests are
  mandatory on both sides;
- promotion requires explicit version changes and protected release intent; no
  production pin may silently drift to a main-branch package;
- npm may have its own staging dist-tag/version strategy, while crates.io uses
  only immutable SemVer prereleases for pre-stable registry testing.
