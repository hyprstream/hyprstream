# hyprstream Development Guide

**Claude AI Assistant Guide** - Updated Feb 22, 2026

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

## Crate Map

| Crate | Purpose |
|-------|---------|
| `hyprstream` | Main app: runtime, storage, git, training, API, services, CLI (MIT OR AGPL-3.0) |
| `git2db` | Git repository management library (has own CLAUDE.md) |
| `git-xet-filter` | XET/LFS large file storage (libgit2 filter) |
| `gittorrent` | P2P model distribution |
| `cas-serve` | Content-addressable storage server |
| `hyprstream-flight` | Arrow Flight SQL server |
| `hyprstream-metrics` | Metrics (DuckDB/DataFusion) |
| `hyprstream-rpc` | Cap'n Proto RPC: ZMQ transport, crypto, signed envelopes, JWT auth |
| `hyprstream-rpc-derive` | Proc macros: `ToCapnp`, `FromCapnp`, `#[authorize]`, `#[service_factory]`, `generate_rpc_service!` |
| `hyprstream-rpc-build` | CGR → JSON metadata extraction for macros |
| `hyprstream-workers` | Kata-based worker isolation |
| `hyprstream-containedfs` | Contained filesystem ops |
| `hyprstream-vfs` | Plan 9-inspired VFS namespace multiplexer |
| `hyprstream-tcl` | Tcl (molt) shell for VFS namespace |
| `bitsandbytes-sys` | bitsandbytes FFI bindings |

## Key Source Layout (`crates/hyprstream/src/`)

- `runtime/` — PyTorch engine (`torch_engine.rs`), architectures, model_factory, kv_cache, sampling, templates, LoRA
- `storage/` — Model storage (`model_ref.rs`, `paths.rs`), `adapter_manager.rs` (file-based adapters)
- `git/` — Thin wrapper over git2db + `helpers.rs` (tag creation etc.)
- `training/` — TTT (`ttt.rs`), per-tenant deltas (`tenant_delta.rs`, `delta_pool.rs`), merge strategies, quality filter, checkpoints
- `api/` — OpenAI compat types, tool calling
- `server/` — Axum routes (`/v1/models` lists worktrees as `model:branch`), state management
- `services/` — RPC service implementations + `factories.rs` (inventory-based auto-discovery)
- `schema/` — Cap'n Proto schemas: `registry.capnp`, `inference.capnp`, `model.capnp`, `policy.capnp`, `worker.capnp`, `mcp.capnp`

## Services (10 total, registered via `#[service_factory]` in `services/factories.rs`)

**ZMQ RPC** (Cap'n Proto over ZMQ, REQ/REP): registry, model, policy, worker, mcp
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
