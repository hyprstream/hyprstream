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
  3. `hyprstream-crypto` `0.1.0` (leaf cryptographic primitives)
  4. `hyprstream-rpc` `0.1.0` (exactly depends on the three preceding crates)
- Everything else sets `publish = false`, except that the stale `hyprstream`
  crates.io name is reserved by the separately packageable Apache-2.0
  `release/hyprstream-deprecation-shim` artifact at exactly `0.5.0`, rather
  than republishing the real service. Looking reusable is not an API or
  maintenance commitment. Promotion to `public_now` requires an owner, API and
  SemVer review, package documentation, security review where applicable,
  crates.io name ownership, versioned local dependencies, and a successful
  packaged-source build.

The existing crates.io package `hyprstream` is owned by the project maintainer
account but is **not** the local application release channel: its seven
Apache-2.0 versions (up to `0.1.0-alpha-6`) describe a different, stale product.
The real service remains internal/non-publishable because of its broad local
graph and `tch`/libtorch runtime. The excluded shim manifest must match the
real service's exact `0.5.0` version, while containing no service dependencies
or product implementation. The release order is fail-closed: **(1)** publish
the Apache-2.0 shim, **(2)** verify `hyprstream@0.5.0` resolves from crates.io,
then and only then **(3)** yank the seven stale versions. Do not delete the
package name or republish the service to occupy it. The name `chat-core` is
owned by an unrelated crates.io publisher and cannot be used by this project.

## Complete inventory

| Crate | Class | Reason / promotion gate |
|---|---|---|
| `hyprstream-rpc-build` | **public now** | Documented build/codegen API; packages independently; first in release order. |
| `hyprstream-rpc-derive` | **public now** | Documented proc-macro API; exact versioned dependency on `hyprstream-rpc-build`. |
| `hyprstream-crypto` | **public now** | Transport-free cryptographic primitives; a dependency-free local leaf in the Rust SDK closure. |
| `hyprstream-rpc` | **public now** | Rust SDK protocol surface; exact versioned dependencies on build, derive, and crypto complete the publishable closure. |
| `bitsandbytes-sys` | public later | External CUDA/ROCm FFI/toolchain contract needs ownership and platform support policy. |
| `cas-serve` | public later | XET git-source dependency and missing package docs. |
| `git-xet-filter` | public later | XET git-source/default-feature dependency chain; API review required. |
| `git2db` | public later | XET git-source/default-feature chain and broad storage API commitment. |
| `hyprstream-9p` | public later | Depends on unpublished RPC/VFS crates; wire/API stability review required. |
| `hyprstream-compositor` | public later | Missing package docs and explicit external API owner. |
| `hyprstream-containedfs` | public later | Security-sensitive path-containment contract needs dedicated review and support owner. |
| `hyprstream-discovery` | public later | Deep unpublished dependency graph and security-sensitive discovery/admission surface. |
| `hyprstream-flight` | public later | Depends on unpublished metrics/git2db stack; missing package docs. |
| `hyprstream-k8s` | public later | CRD/version compatibility is a public API; ownership and compatibility policy required. |
| `hyprstream-ledger` | public later | Accounting/security semantics need independent review and an API owner. |
| `hyprstream-metrics` | public later | Depends on unpublished git2db and native data stack; missing package docs. |
| `hyprstream-p2p` | public later | Depends on unpublished RPC; network protocol/support commitment not yet declared. |
| `hyprstream-pds` | public later | Security/signing and atproto storage APIs need review; unpublished dependencies. |
| `hyprstream-pds-service` | public later | Tenant service depends on unpublished PDS/RPC/VFS stack. |
| `hyprstream-resource` | public later | Early authority/lifecycle interface scaffold; API not committed. |
| `hyprstream-rpc-std` | public later | Requires path dependencies on RPC/VFS/workers-tcl/9p and the unreleased patched `moq-net` wasm fix; `[patch.crates-io]` does not reach downstream consumers, so publishing it would resolve an unpatched crates.io `moq-net`. |
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
| `hyprstream` | **internal/not publishable** | Real product/service crate (broad local graph and `tch`/libtorch); keep it non-publishable. The excluded standalone Apache-2.0 `hyprstream` 0.5.0 deprecation shim, not this crate, replaces the stale public name. |
| `hyprstream-appview` | **internal/not publishable** | Product-internal derived identity service with no standalone support contract. |
| `hyprstream-tui` | public later | Supported MIT client TUI for trust/crypto, bootstrap, and enrolment; its publish promotion follows the client dependency and accelerator-runtime download/verification work. |
| `hyprstream-workers-python-guest` | **internal/not publishable** | Compiled WASM guest artifact, not a Rust consumer library. |
| `hyprstream-workers-wasmtime-fsguest` | **internal/not publishable** | Test/validation WASI guest binary, not a consumer package. |

## Licensing boundary and client/server split

The repository currently has a workspace default of `license = "MIT"` in
`[workspace.package]`, and carries both `LICENSE-MIT` and `LICENSE-AGPLV3`.
Today every crate is effectively MIT except `crates/hyprstream`, whose manifest
is `MIT OR AGPL-3.0-only`. That is a disjunction: recipients may select MIT and
disregard AGPL. It therefore supplies no copyleft protection for the server.

The approved target is a mixed, per-crate boundary, not a disjunction:

| Boundary | Approved target |
|---|---|
| Services | `AGPL-3.0-only`: the service/orchestration portions of `hyprstream`, plus `hyprstream-pds`, `hyprstream-pds-service`, `hyprstream-appview`, `hyprstream-ledger`, `hyprstream-discovery`, `hyprstream-service`, and `hyprstream-k8s`. |
| Libraries, client, and runtimes | `Apache-2.0`: `hyprstream-rpc`, `hyprstream-rpc-std`, `hyprstream-rpc-build`, `hyprstream-rpc-derive`, `hyprstream-crypto`, `hyprstream-tui`, `hyprstream-util`, `hyprstream-vfs`, the `hyprstream-workers*` execution runtimes, the inference runtime, and the future client binary. |

`hyprstream-rpc` is the named exception that protects the client architecture:
it remains Apache-2.0 regardless of whether a client or a service consumes it.
Making the protocol AGPL would prevent independent clients and collapse the
client story. The TUI is a supported client artifact, not the server/product
application; it provides the TUI, offline trust/crypto primitives, and
bootstrap/enrolment while downloading and verifying the heavy accelerator
runtime instead of bundling it. `hyprstream` remains internal/not publishable
as the server/product application.

### Runtime/service sub-boundary and required extraction

The split is not “monolith equals AGPL.” Execution engines are permissive; the
services that schedule, route, admit tenants, and integrate ledger/billing are
AGPL-only. The inference runtime—including model execution, kernels,
device/accelerator handling, and the `tch` binding—is Apache-2.0. The
`hyprstream-workers*` sandbox execution runtimes are also Apache-2.0 targets.
This is the strongest
case for Apache-2.0's express patent grant and defensive termination: inference
is the most patent-dense code in the tree, the Apache ICLA already supplies the
inbound rights, and the established ML-runtime ecosystem is permissive.

That target cannot be expressed in current Cargo packages. `hyprstream` is the
only crate that links `tch`, but it contains both the intended Apache-2.0
runtime (`src/runtime`: 25,719 lines in 28 files; `src/inference`: 1,068 lines
in 5 files) and the intended AGPL orchestration service—about 26.8k runtime
lines in total. Relicensing the whole crate as AGPL-only now would silently
relicense that runtime incorrectly. The runtime must first be extracted into a
Apache-2.0 crate; this is a prerequisite, not optional cleanup. The current
`hyprstream-workers` facade likewise still has AGPL service-wiring dependencies,
so it must be split before it can carry the runtime's permissive license.

This is the same seam identified by client/server layering, licensing, and now
runtime/service separation. It elevates the extraction in
`PLAN-crypto-wizard-terraform.md` section 0b from architectural hygiene to a
prerequisite for the license split. A permissive downloadable runtime also keeps
the thin client's verified install path free of a separate-process or
mere-aggregation argument.

The existing graph assertion enforces all currently separable Apache-2.0 roots,
including the worker runtime crates that already have clean dependency graphs.
It deliberately records, rather than pretends to enforce, the two
topology-blocked targets (`hyprstream`'s inference runtime and the mixed
`hyprstream-workers` facade). Once extracted, each must move into the enforced
Apache-2.0-root set before its license field changes.

This prospective split aligns three independent boundaries: client/server,
licensing, and runtime/service. It is not yet implemented in manifests. Every
crate must receive an explicit per-crate
`license =` value when the separate relicensing change lands: inheriting the
workspace MIT default is unsafe because a new service crate could silently take
the wrong license. No license field is changed by this release-policy change.

Apache-2.0 is the selected, single permissive license. The Apache ICLA's
sections 2 and 3 supply inbound copyright sublicensing and patent grants, while
Apache-2.0 section 3 passes an express patent grant to downstream users with
defensive termination. `MIT OR Apache-2.0` is not this policy's choice because
a downstream recipient could elect the MIT disjunct and lose that express
patent protection. The former `MIT OR AGPL-3.0-only` combination was different
again: its MIT branch nullified the intended copyleft.

The legal basis for the prospective outbound licenses is the Apache ICLA:
section 2 grants a perpetual, irrevocable copyright license including
sublicensing, and section 3 grants patents. The change is prospective only.
The repository is public, and the pre-split MIT releases `v0.2.0` (2026-02-04),
`v0.3.0-rc1`, and `v0.4.0` (2026-06-18) already granted MIT rights to the
service code; an AGPL-only release cannot revoke those grants. Every additional
release before the split lands broadens that MIT-granted surface.

The current disjunction was not an approved split implementation: before
`fcf3f31a5` (2026-02-22, “Locking/threading improvements”), `hyprstream` used
the workspace MIT license. That commit changed it to `MIT OR AGPL-3.0`.
`806c167c3` later changed the AGPL spelling to `AGPL-3.0-only` for cargo-deny,
but retained the `MIT OR` disjunction. License changes must therefore be their
own reviewed commits, never incidental refactor changes.

`scripts/check_license_boundary.py` enforces the currently expressible
architectural half of this policy in CI: no listed Apache-2.0 crate may
transitively depend on a listed AGPL service crate, and the topology-blocked
runtime targets must remain explicitly recorded. The separate extraction and
relicensing changes must retain and expand that assertion with explicit
per-crate license declarations; a documented boundary without the graph check
is not a boundary.

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
# The derive crate and crypto are independent after build is available.
cargo package --locked -p hyprstream-crypto
cargo publish --locked --registry crates-io -p hyprstream-crypto
# Wait until both derive and crypto resolve from crates.io.
cargo package --locked -p hyprstream-rpc
cargo publish --locked --registry crates-io -p hyprstream-rpc
```

### Stale `hyprstream` deprecation shim

The `hyprstream` 0.5.0 deprecation shim is a distinct, manually authorized
operation. Its excluded manifest is deliberately not part of the SDK release
train or the service workspace, so use the explicit manifest path:

```bash
cargo package --locked --manifest-path release/hyprstream-deprecation-shim/Cargo.toml
cargo publish --locked --registry crates-io --manifest-path release/hyprstream-deprecation-shim/Cargo.toml
cargo info hyprstream@0.5.0
```

The `cargo info` verification must confirm the public 0.5.0 shim before an
authorized owner may yank any stale version. A publish failure, ambiguous
registry result, or failed verification stops the procedure: do not yank first,
do not republish `crates/hyprstream`, and do not substitute `0.2.0`.

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

Do this separately for `hyprstream-rpc-build`, `hyprstream-rpc-derive`,
`hyprstream-crypto`, and `hyprstream-rpc`. Add a second accountable
maintainer/team owner where organizational policy permits, verify all owner
emails, test one prerelease, then enable crates.io's **require trusted
publishing** control for each crate so long-lived API tokens cannot publish
updates. Retain a documented break-glass owner procedure; never store a
crates.io token in repository or environment secrets for routine releases.

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

### npm/crates.io source-artifact asymmetry

`hyprstream-rpc-std` remains public later for crates.io because that registry
ships Rust source and therefore requires its full local dependency graph to be
published first. The sibling WASM SDK lane may publish the same interface to npm
now because npm receives a compiled WASM artifact, which does not expose that
Rust source dependency graph. This is intentional rather than a contradiction:
the release order is WASM SDK first and the source-distributed Rust SDK closure
second. `hyprstream-rpc` is in that four-crate closure; only `hyprstream-rpc-std`
remains blocked because its browser path requires the unreleased patched
`moq-net` behavior and additional unpublished path dependencies.
