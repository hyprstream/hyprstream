# hyprstream Development Guide

**Claude AI Assistant Guide** - Updated Jul 2, 2026

## Core Philosophy

- **Models are Git repositories** — version-controlled via git2db
- **Adapters are files** — stored in `model/adapters/` as `.safetensors` (NOT branch-based)
- **git2db handles all Git operations** — no custom git wrappers, no raw `Repository::open()`
- **Storage drivers optimize disk** — overlay2 on Linux (~80% savings)

## Status

**Production**: PyTorch inference (CPU/CUDA/ROCm), git2db model management, file-based LoRA adapters, OpenAI-compatible REST API, UTF-8 streaming
**Experimental**: XET large file storage (enabled by default), Test-Time Training (TTT) with per-tenant LoRA deltas

## Build

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release                    # Standard (includes xet, gittorrent, systemd, otel)
cargo build --features cuda --release    # CUDA marker (backend from tch-rs/libtorch)
cargo build --features bnb --release     # bitsandbytes quantization
cargo build --features overlayfs --release
cargo build --no-default-features --features "gittorrent,xet,otel" --release  # No systemd
cargo test --workspace --release
```

**Feature flags**: `default = [gittorrent, xet, systemd, otel]`, `cuda` (empty marker), `bnb`, `overlayfs`, `download-libtorch`, `experimental`
**Backend**: CPU/CUDA/ROCm controlled by tch-rs dependency (fork: github.com/hyprstream/tch-rs branch: hip), NOT cargo features

## Working Tree & Worktrees (multi-agent safety)

**Never run `git checkout`/`switch`/`reset`/`clean` in the shared main checkout. Use a dedicated worktree per branch — always `git worktree add`, never `git checkout -b` in the shared tree.**

Multiple agents may share the one checkout at the repo root. That checkout has a single HEAD + index: a branch switch there drags every other agent's uncommitted edits onto the wrong branch, and a `reset --hard`/`checkout .` silently wipes them. One branch → one worktree isolates this completely.

- **Create:** `git worktree add ../<name>` (or under `.worktrees/<name>` / your session scratchpad) for the branch you're working on. Stack a dependent branch off its base worktree (e.g. `git worktree add .worktrees/mytask <base-branch>`).
- **Shared checkout stays neutral:** leave the root checkout on `main`; use it only for read-only ops (`fetch`, `log`, `show`, `ls-tree`) and `git worktree add`/`gh` — never for active branch work.
- **Clean up:** `git worktree remove <path>` when done; `git worktree prune` for stale entries.

### Commit authorship (GitHub ruleset, enforced on `main`)

**Never add a `Co-authored-by: Claude <noreply@anthropic.com>` trailer, or any AI-attribution marker, to a commit message.** A repo ruleset on the default branch rejects commits whose message matches that pattern. Commit as whatever git identity is already configured for the session (do not fabricate or override author identity to route around this or any other rule) and write a plain, human-style commit message — the ruleset is about the trailer text, not about disguising who/what made the change.

## Crate Map

| Crate | Purpose |
|-------|---------|
| `hyprstream` | Main app: runtime, storage, git, training, API, services, CLI, native MAC (MIT OR AGPL-3.0) |
| `git2db` | Git repository management library (has own CLAUDE.md) |
| `git-xet-filter` | XET/LFS large file storage (libgit2 filter) |
| `gittorrent` | P2P model distribution |
| `cas-serve` | Content-addressable storage server |
| `hyprstream-flight` | Arrow Flight SQL server |
| `hyprstream-metrics` | Metrics (DuckDB/DataFusion) |
| `hyprstream-rpc` | Cap'n Proto RPC: ZMQ transport, crypto (hybrid PQC, UCAN), signed envelopes, JWT auth |
| `hyprstream-rpc-derive` | Proc macros: `ToCapnp`, `FromCapnp`, `#[authorize]`, `#[service_factory]`, `generate_rpc_service!` |
| `hyprstream-rpc-build` | CGR → JSON metadata extraction for macros |
| `hyprstream-rpc-std` | Standard hyprstream service schemas + generated clients |
| `hyprstream-util` | Shared generic utility primitives (TTL cache, etc.) |
| `hyprstream-service` | Pluggable service orchestration (spawner, factory, manager) |
| `hyprstream-discovery` | Service discovery and endpoint resolution |
| `hyprstream-workers` | Kata-based worker isolation |
| `hyprstream-workers-wasmtime` | Embedded wasmtime host — sandboxes untrusted wasm guests behind capability profiles |
| `hyprstream-workers-tcl` | Tcl (molt) shell for the hyprstream VFS namespace |
| `hyprstream-workers-python` | RustPython engine + `/lang/python` 9P shell over the wasmtime sandbox |
| `hyprstream-containedfs` | Contained filesystem ops |
| `hyprstream-vfs` | Plan 9-inspired VFS namespace multiplexer |
| `hyprstream-vfs-server` | vhost-user-fs server exposing a VFS Namespace to a Cloud Hypervisor guest (#362) |
| `hyprstream-9p` | 9P2000.L codec + server-side translator bridging 9P to hyprstream RPC (Cap'n Proto) |
| `hyprstream-pds` | atproto PDS record store — DAG-CBOR records, MST, signed commits, CAR proofs (#392) |
| `hyprstream-tui` | Terminal UI — wizard, dashboards, status |
| `hyprstream-compositor` | Pure Rust TUI compositor — WASM-compatible chrome state machine |
| `waxterm` | Run ratatui apps natively and in the browser via WASI |
| `chat-core` | Chat orchestration state machine — token parsing, tool call detection, agentic loop |
| `bitsandbytes-sys` | bitsandbytes FFI bindings |

`hyprstream-tcl` above was renamed **`hyprstream-workers-tcl`**; worker-engine crates are always `hyprstream-workers-{engine}`, never `hyprstream-<lang>` (decided, don't relitigate).

## Key Source Layout (`crates/hyprstream/src/`)

- `runtime/` — PyTorch engine (`torch_engine.rs`), architectures, model_factory, kv_cache, sampling, templates, LoRA
- `storage/` — Model storage (`model_ref.rs`, `paths.rs`), `adapter_manager.rs` (file-based adapters)
- `git/` — Thin wrapper over git2db + `helpers.rs` (tag creation etc.)
- `training/` — TTT (`ttt.rs`), per-tenant deltas (`tenant_delta.rs`, `delta_pool.rs`), merge strategies, quality filter, checkpoints
- `api/` — OpenAI compat types, tool calling
- `server/` — Axum routes (`/v1/models` lists worktrees as `model:branch`), state management
- `services/` — RPC service implementations + `factories.rs` (inventory-based auto-discovery)
- `schema/` — Cap'n Proto schemas: `registry.capnp`, `inference.capnp`, `model.capnp`, `policy.capnp`, `worker.capnp`, `mcp.capnp`

## Services (13 total, registered via `#[service_factory]` in `services/factories.rs`)

**ZMQ RPC** (Cap'n Proto over ZMQ, REQ/REP): registry, model, policy, worker, mcp, discovery, metrics, tui
**HTTP** (Axum): oai (OpenAI-compat), oauth (OAuth 2.1), flight (Arrow Flight SQL)
**Proxies** (ZMQ): streams (PULL→XPUB), event (XSUB→XPUB)

**Security**: CURVE transport encryption → Ed25519 signed envelopes → Casbin + JWT authorization
**Spawner modes**: Tokio (async), Thread (!Send types like tch-rs), Subprocess (systemd/standalone)
**Endpoints**: Inproc (daemon, zero-copy) or IPC (systemd socket activation)

## Key Patterns

**Git operations** — always use git2db: `registry.repo(&id)?.branch().create(...)`, `handle.staging().add_all()?`
**Adapters** — `AdapterManager::new(path)`, files in `adapters/` as `00_name.safetensors`
**Tags** — `crate::git::helpers::create_tag(path, "v1.0")?`
**Generation** — `engine.generate(req)?` returns `TextStream` (futures::Stream), handles UTF-8 internally via `DecodeStream`
**TTT** — `TestTimeTrainer::adapt_tenant()` for inference-time adaptation, `DeltaPool` for per-tenant LRU management
**XET/LFS** — Filter initialized in main.rs via `git2db::xet_filter::initialize()`, auto-smudges on git ops; fallback: `git2db::LfsStorage`
**Hybrid PQC signing** — security-critical artifacts (UCAN grants/approvals, compiled MAC policy, audit records, minted tokens) sign via `hyprstream_rpc::crypto::cose_sign::{sign_composite, verify_composite}` (EdDSA + ML-DSA-65 nested COSE). Never Ed25519-only for these paths. `CryptoPolicy::{Hybrid, Classical}` picks the enforced suite; under `Hybrid`, a missing PQ key **fails closed** at sign time — it never silently downgrades to classical.
**UCAN capability delegation** — `hyprstream_rpc::auth::ucan` (`token.rs`/`chain.rs`/`capability.rs`/`approval.rs`) is the delegation-chain primitive underlying the MAC grant path: a `Ucan` carries `Capability` triples (resource/ability/caveats), `chain::validate` walks/attenuates a delegation chain, `Capability::authorizes`/`covers` do directional wildcard matching. UCAN *authors* capability; it never enforces on its own — see Native MAC below.
**9P/VFS surface — three distinct crates, don't conflate**: `hyprstream-vfs` (the Plan 9-inspired in-process namespace multiplexer — mounts, binds, the `Mount` trait), `hyprstream-9p` (the 9P2000.L wire-protocol codec + translator bridging 9P ⇄ hyprstream RPC), `hyprstream-vfs-server` (vhost-user-fs server exposing a VFS `Namespace` to a Cloud Hypervisor guest, #362).

## RPC & Security

### Code Generation Pipeline

```
.capnp schema → capnpc → CGR binary → hyprstream-rpc-build → JSON metadata
                                                                    ↓
                                          generate_rpc_service! macro → typed clients + handlers + dispatch
```

### Security Model (3 layers)

1. **Transport**: CURVE encryption (Curve25519) on TCP sockets via `CurveConfig`
2. **Application**: Ed25519 signed `SignedEnvelope` — survives message forwarding
3. **Authorization**: Casbin policy + JWT scopes (`action:resource:identifier`)

**Envelope flow**: Client builds `RequestEnvelope` → signs with Ed25519 → `SignedEnvelope` → server verifies → extracts `EnvelopeContext` with verified identity → handler checks authorization

### Derive Macros (`hyprstream-rpc-derive`)

| Macro | Purpose |
|-------|---------|
| `ToCapnp` / `FromCapnp` | Cap'n Proto serialization (derive) |
| `#[authorize]` | Declarative JWT + Casbin authorization on handlers |
| `#[register_scopes]` | Compile-time scope registration |
| `#[service_factory]` | Inventory-based service registration |
| `generate_rpc_service!` | Full client/handler/dispatch from CGR metadata |

### Schema Locations

- `crates/hyprstream-rpc/schema/` — `common.capnp` (envelopes, identity, claims), `streaming.capnp`, `events.capnp`, `annotations.capnp`
- `crates/hyprstream/schema/` — `registry.capnp`, `inference.capnp`, `model.capnp`, `policy.capnp`, `worker.capnp`, `mcp.capnp`

### Key RPC Types (`crates/hyprstream-rpc/src/`)

- `service/zmq.rs` — `ZmqService` trait, `RequestLoop`, `EnvelopeContext`
- `envelope.rs` — `SignedEnvelope`, `RequestIdentity` (Local/ApiToken/Peer/Anonymous)
- `crypto/mod.rs` — Ed25519 signing, Ristretto255 ECDH, `ChainedStreamHmac`
- `streaming.rs` — `StreamPublisher`, `ResponseStream` (PUSH/PULL→XPUB, prevents message loss)
- `auth/` — JWT `Claims`, `Scope`, `ScopeRegistry`

### Spawner Details

- **Tokio mode**: `tokio::task::spawn_blocking()` — async-safe services
- **Thread mode**: `std::thread` + single-threaded tokio runtime — required for `!Send` types (tch-rs tensors)
- **Subprocess mode**: `ProcessSpawner` — auto-detects systemd (`systemd-run`) or standalone (PID file tracking)
- **`Spawnable` trait**: blanket impl for all `ZmqService + Send + Sync`; `!Send` types (e.g., inference) implement directly

## Native MAC — Mandatory Access Control (epic #547)

`crates/hyprstream/src/mac/` (`label`/`lattice`/`context` live in `hyprstream-rpc/src/auth/mac/`) is a from-scratch **Bell–LaPadula-style MAC layer** for the 9P/VFS data plane — distinct from, and layered underneath, the RPC authorization model above (Casbin/JWT scopes are the **control plane**; MAC is the **mandatory floor** no control-plane grant can bypass).

**Control-plane / data-plane split** (be precise about this when touching MAC code):
```
UCAN (grant/delegation source) → Casbin (policy STORE + authoring, string-matcher NEVER on the per-op hot path)
  → S5 compiler lowers UCAN/Casbin → compiled::SignedPolicy (hybrid-PQC-signed TeMatrix)
    → loader verifies ONCE at load → te::LatticeTeEvaluator (the PDP) + avc::CachingAvc (the per-op cache, PEP calls this)
```

**Key modules:**
- `hyprstream_rpc::auth::mac` — `SecurityLabel` (Level × Assurance × CompartmentSet, `Copy+Ord+Hash`), `Lattice` (versioned compartment vocabulary), `SecurityContext` (clearance derived from verified `Claims` × `VerifiedKeyMaterial` — **never Claims alone**). Dominance is the `can_access` method (Bell–LaPadula framing kept in its rustdoc), not a free trait method.
- `mac::te` — the type-enforcement PDP: total, default-deny `(subject_ctx, object_label, action) → Decision` over the compiled matrix AND the independent lattice floor.
- `mac::avc` — the Access Vector Cache the PEP calls per op (sub-µs hits, no Casbin/signature/lattice walk on a hit).
- `mac::compiler` + `mac::permission_map` — the UCAN→TE compiler. `PermissionMap` is the S3-scope↔TE-rule vocabulary seam; `ScopePermissionMap` (`permission_map.rs`) is the production impl, made **injective + exact by construction** (wildcards expand at compile time over a closed/sorted registry) so `granted_access` is trivially the most-permissive access — no join/LUB logic that could mask an escalation. `check_no_escalation` verifies the emitted matrix against the independent grant, never by re-running the forward map.
- `mac::exchange` — S6 runtime grant path (ZSP: zero standing privilege): a presented UCAN subset-grant → short-ttl, sender-bound (DPoP) OAuth token, re-evaluated on every refresh, never minting more than the requested subset.
- `mac::audit` — S7 tamper-evident audit: `WalAuditStore` is an append-only, hash-chained (`prev_hash`), signed-checkpoint-anchored journal; `AuditedAvc` wraps any AVC so a decision that cannot be durably audited is downgraded to Deny (fail-closed), never silently permitted.
- `mac::compiled` — signed policy distribution (hybrid-PQC COSE), verified once at load.

**Naming (don't relitigate):** `ceiling`→`grant`, `dominates`→`can_access`, `witness`→`granted_access`, `rules_for`→`permissions_for` — plain MAC/RBAC vocabulary, not academic/poetic terms.

**Current status (check before assuming otherwise):** the MAC library is real and well-tested, but **enforcement is not active in production as of this writing** — the PDP has no wired-in PEP caller, the S6 grant path's HTTP dispatch still fails closed pending resolver/object-label wiring, and the audit store needs explicit startup construction. Everything fails closed (nothing is exploitable), but do not assume MAC is "live" — check the current wiring state (`services::oauth`, service factories) before relying on it protecting anything at runtime.

## Adding an RPC Method

1. Define in `.capnp` schema (request/response structs + method variant)
2. `cargo build -p hyprstream` (regenerates CGR metadata)
3. Implement handler with `#[authorize(action = "...", resource = "...")]`
4. Client code auto-generated by `generate_rpc_service!`

## Adding a New Service

1. Implement `ZmqService` (or `Spawnable` for !Send types)
2. Add `#[service_factory("name", schema = "...")]` fn in `factories.rs`
3. Auto-discoverable via inventory — `hyprstream service start name --foreground` works
4. `hyprstream service install` generates systemd units

## Testing

```bash
cargo test --workspace
cargo test -p hyprstream
cargo test -p git2db
RUST_LOG=debug cargo test
cargo test -p hyprstream storage::adapter_manager
```

## Common Issues

- **"libtorch not found"**: Set `LIBTORCH` and `LD_LIBRARY_PATH`
- **"overlayfs mount failed"**: Try `GIT2DB_OVERLAY_BACKEND=userns` or `GIT2DB_WORKTREE_DRIVER=vfs`
- **"BranchManager not found"**: Removed Oct 2025 — use git2db or `git::helpers`
- **"AdapterStorage not found"**: Removed Oct 2025 — use `AdapterManager` (file-based)

## Removed Code (don't recreate)

- `api/adapter_storage.rs` — branch-based adapters (Oct 2025)
- `git/branch_manager.rs` — custom git wrapper (Oct 2025)
- `storage/sharing.rs` — ModelSharing (P2P belongs at transport layer)
- `storage/xet_native.rs` — consolidated into git2db/git-xet-filter
- `training/lora_trainer.rs` — replaced by TTT system
- `mac/compiler.rs`'s `seam.rs` (`BundleEmitter`/`FaithfulnessCheck`/`ActionVocabulary` traits) and its `smt` no-op module — deleted in the S5 simplification (June 2026): over-engineered/academic naming collapsed to plain free functions (`compile`/`check_no_escalation`/`authorize`). Don't recreate trait-heavy ceremony here; the deferred SMT proof is a `// TODO(#571)` marker, not a stub module.

## Quick Commands

```bash
cargo build --release
cargo test --workspace
cargo run --bin hyprstream -- --help
cargo run --bin hyprstream serve --port 8080
cargo fmt --all
cargo clippy --all-targets --all-features
./appimage/build-appimage.sh build rocm71
```

## Docs

See: `README.md`, `DEVELOP.md`, `CONTRIBUTING.md`, `crates/git2db/CLAUDE.md`, `docs/` (quickstart, tool calling, KV cache, RPC, workers, streaming, events, cryptography)

## Licensing

- **hyprstream**: MIT OR AGPL-3.0 (user's choice)
- **All other crates**: MIT
