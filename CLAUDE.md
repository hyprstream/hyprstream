# hyprstream Development Guide

**Claude AI Assistant Guide** - Updated Jul 2, 2026

## Big Picture

HyprStream is the runtime for AI that gets smarter the more you use it: a Plan 9-inspired "operating system for agents" where models, inference streams, MCP tools, repositories, and sandboxed apps are all files in one composable, per-process namespace — spanning host, microVM guests, and the browser (Wanix over 9P). Nodes federate by cryptographic identity (atproto `did:web`/`did:key`), not host location, over QUIC/WebTransport or iroh P2P. Self-improvement follows the STEP loop: **Stage → Train → Evaluate → Promote**, every gain a reviewable Git branch of the weights. See `README.md` and https://talks.cyberdione.ai/hyprstream-wanix for the full narrative.

## Core Philosophy

**Normative ontology:** [`docs/system-ontology.md`](docs/system-ontology.md) is the source of truth for canonical nouns/actions, status, laws, and module-aware layer assignments, and takes precedence over descriptive summaries in this guide for those facts. It incorporates this guide's **Naming (don't relitigate)** block, ADR #651, and #1090 revision 7 by reference; those narrower ratified decisions remain authoritative in their scopes and are not silently superseded.

- **Everything is a file** — models, streams, tools, and apps compose in one VFS namespace; capability-scoped by construction
- **Models are Git repositories** — version-controlled via git2db; promotion = merge, rollback = checkout
- **Adapters are files** — stored in `model/adapters/` as `.safetensors` (NOT branch-based)
- **git2db handles all Git operations** — no custom git wrappers, no raw `Repository::open()`
- **Identity over location** — federation addresses atproto DIDs; transport (QUIC/iroh/UDS/inproc) is pluggable and replaceable
- **Storage drivers optimize disk** — overlay2 on Linux (~80% savings)

## Status

**Production**: PyTorch inference (CPU/CUDA/ROCm), git2db model management, file-based LoRA adapters, OpenAI-compatible REST API, UTF-8 streaming, MoQ streaming/event planes, collaborative TUI
**Experimental**: XET large file storage (enabled by default), Test-Time Training (TTT) with per-tenant LoRA deltas, federation (off by default; opt in via wizard), Kata microVM workers, atproto PDS

## Build

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release                    # Standard (includes xet, gittorrent, systemd, otel, kata-vm)
cargo build --features cuda --release    # CUDA marker (backend from tch-rs/libtorch)
cargo build --features bnb --release     # bitsandbytes quantization
cargo build --features overlayfs --release
cargo build --no-default-features --features "gittorrent,xet,otel" --release  # No systemd/kata
cargo test --workspace --release
```

**Feature flags**: `default = [gittorrent, xet, systemd, otel, kata-vm]`, `cuda` (empty marker), `bnb`, `valkey` (Valkey/Redis user+token stores), `overlayfs`, `oci-image` (RAFS ImageFs), `kata-vm` (implies `oci-image`), `download-libtorch`, `experimental`, `pq-hybrid` (NO-OP alias — PQ primitives always compiled, Classical/Hybrid selected at runtime)
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
| `hyprstream-rpc` | RPC core: Cap'n Proto over pluggable transports (inproc/UDS/QUIC/iroh, ZMTP framing), hybrid-PQC COSE envelopes, JWT/UCAN auth, MoQ streaming |
| `hyprstream-rpc-derive` | Proc macros: `ToCapnp`, `FromCapnp`, `#[authorize]`, `#[service_factory]`, `generate_rpc_service!` |
| `hyprstream-rpc-build` | CGR → JSON metadata extraction for macros |
| `hyprstream-rpc-std` | Standard hyprstream service schemas + generated clients |
| `hyprstream-util` | Shared generic utility primitives (TTL cache, etc.) |
| `hyprstream-service` | Pluggable service orchestration (spawner, factory, manager, trust store) |
| `hyprstream-discovery` | Service discovery and endpoint resolution |
| `hyprstream-workers` | Worker isolation: pluggable sandbox backends (Kata microVM, rootless OCI/podman, nspawn, WASM) |
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

Worker-engine crates are always `hyprstream-workers-{engine}`, never `hyprstream-<lang>` (decided, don't relitigate; `hyprstream-tcl` was renamed `hyprstream-workers-tcl` under this rule).

## Key Source Layout (`crates/hyprstream/src/`)

- `runtime/` — PyTorch engine (`torch_engine.rs`), architectures, model_factory, kv_cache, sampling, templates, LoRA
- `storage/` — Model storage (`model_ref.rs`, `paths.rs`), `adapter_manager.rs` (file-based adapters)
- `git/` — Thin wrapper over git2db + `helpers.rs` (tag creation etc.)
- `training/` — TTT (`ttt.rs`), per-tenant deltas (`tenant_delta.rs`, `delta_pool.rs`), merge strategies, quality filter, checkpoints
- `api/` — OpenAI compat types, tool calling
- `server/` — Axum routes (`/v1/models` lists worktrees as `model:branch`), state management
- `services/` — RPC service implementations + `factories.rs` (inventory-based auto-discovery)
- `mac/` — Native MAC data-plane: TE evaluator, AVC, UCAN→TE compiler, grant exchange, audit (see Native MAC below)
- `events/` — group-keyed EventService integration (epic #600)
- `auth/`, `inference/`, `tui/`, `cli/`, `config/`, `archetypes/` — supporting modules
- `schema/` — app-local Cap'n Proto schemas: `tui.capnp`, `compositor_ipc.capnp` (service schemas live in `hyprstream-rpc-std/schema/`)

## Services (13 total, registered via `#[service_factory]` in `services/factories.rs`)

**RPC** (Cap'n Proto request/reply over the pluggable transport layer): registry, model, policy, worker, mcp, discovery, metrics, tui, oauth
**HTTP surfaces** (Axum, spawned by their factories): oai (OpenAI-compat), oauth (OAuth 2.1 endpoints alongside its RPC schema), mcp (HTTP/SSE for external MCP clients), flight (Arrow Flight SQL)
**MoQ planes** (process-global origins + a UDS moq server for cross-process; NO proxy threads): streams (`MoqStreamOrigin`, #138), event (`MoqEventOrigin`, #167 — replaced the old ZMQ XPUB/XSUB proxies)

**Security**: transport channel auth (TLS 1.3 on QUIC, UDS peer credentials) → hybrid-PQC COSE signed envelopes → Casbin + JWT/UCAN authorization
**Spawner modes** (`hyprstream-service`): Tokio (async), Thread (!Send types like tch-rs), Subprocess (systemd/standalone)
**Endpoints**: Inproc (daemon, zero-copy), IPC/UDS (incl. systemd socket activation via SystemdFd), QUIC (cross-host), Iroh (P2P/federation)

## Key Patterns

**Git operations** — always use git2db: `registry.repo(&id)?.branch().create(...)`, `handle.staging().add_all()?`
**Adapters** — `AdapterManager::new(path)`, files in `adapters/` as `00_name.safetensors`
**Tags** — `crate::git::helpers::create_tag(path, "v1.0")?`
**Generation** — `engine.generate(req)?` returns `TextStream` (futures::Stream), handles UTF-8 internally via `DecodeStream`
**TTT** — `TestTimeTrainer::adapt_tenant()` for inference-time adaptation, `DeltaPool` for per-tenant LRU management
**XET/LFS** — Filter initialized in main.rs via `git2db::xet_filter::initialize()`, auto-smudges on git ops; fallback: `git2db::LfsStorage`
**Streaming & events** — MoQ (moq-lite) everywhere: token streams via `moq_stream.rs` (`StreamPublisher`, QoS via `StreamOpt`), events via `moq_event.rs` (group-keyed EventService, epic #600). There is no ZMQ pub/sub anywhere.
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

### Transport Layer (`hyprstream-rpc/src/transport/`)

ZMQ/ZeroMQ is **gone**. RPC uses ZMTP 3.1 *framing* (the wire serialization only — no libzmq) over pluggable transport backends behind `TransportConfig`:

- **Inproc** — in-process channels (fastest, zero-copy)
- **IPC/UDS** — Unix domain sockets (same-host, peer-credential auth), incl. **SystemdFd** socket activation
- **QUIC** — quinn, TLS 1.3 (`QuicServerAuth`: `web_pki()` / `pinned(hashes)` / both)
- **Iroh** — P2P with Ed25519 endpoint binding (NAT traversal, federation)

`dial.rs` lazy-loads the backend chain (inproc → quinn → iroh); `resolver.rs` + `hyprstream-discovery` resolve federated peers via atproto DIDs. Transport auth is channel-level only — identity and trust live in the signed envelope, which survives forwarding across any transport.

### Security Model (3 layers)

1. **Transport**: TLS 1.3 (QUIC) or UDS peer credentials — confidentiality + anti-MITM, NOT the trust root
2. **Application**: `SignedEnvelope` with COSE composite signature (EdDSA + ML-DSA-65, detached); optional ML-KEM-768 hybrid encryption; verified against the peer's key material
3. **Authorization**: Casbin policy + JWT scopes (`action:resource:identifier`) + UCAN delegation chains; native MAC as the mandatory floor (below)

**Envelope flow**: Client builds `RequestEnvelope` → signs (COSE composite per `CryptoPolicy`) → `SignedEnvelope` → server verifies → extracts `EnvelopeContext` with verified identity → handler checks authorization

### Derive Macros (`hyprstream-rpc-derive`)

| Macro | Purpose |
|-------|---------|
| `ToCapnp` / `FromCapnp` | Cap'n Proto serialization (derive) |
| `#[authorize]` | Declarative JWT + Casbin authorization on handlers |
| `#[register_scopes]` | Compile-time scope registration |
| `#[service_factory]` | Inventory-based service registration |
| `generate_rpc_service!` | Full client/handler/dispatch from CGR metadata |

### Schema Locations

- `crates/hyprstream-rpc/schema/` — `common.capnp` (envelopes, identity, claims), `streaming.capnp`, `events.capnp`, `annotations.capnp`, `nine.capnp` (9P bridge), `optional.capnp`
- `crates/hyprstream-rpc-std/schema/` — `registry.capnp`, `model.capnp`, `policy.capnp`, `mcp.capnp`, `oauth.capnp`, `inference.capnp`, `metrics.capnp`, `chat_core.capnp`, `service_events.capnp`
- `crates/hyprstream/schema/` — `tui.capnp`, `compositor_ipc.capnp`
- `crates/hyprstream-workers/schema/` — `worker.capnp`, `workflow.capnp`

### Key RPC Types (`crates/hyprstream-rpc/src/`)

- `service/svc.rs` — `RequestService` trait (the service abstraction; `ZmqService` is long gone), `EnvelopeContext`
- `service/spawnable.rs` — `Spawnable` (blanket impl for `RequestService + Send + Sync`; `!Send` types implement directly)
- `envelope.rs` — `SignedEnvelope` (COSE composite + optional ML-KEM ciphertext), `RequestIdentity` (Local/ApiToken/Peer/Anonymous)
- `crypto/` — `cose_sign.rs` (composite sign/verify), `pq.rs` (ML-DSA-65, ML-KEM-768), `signing.rs`, `key_exchange.rs`, `group_key.rs`/`event_crypto.rs` (group-keyed event confidentiality)
- `moq_stream.rs` / `moq_event.rs` / `streaming.rs` — MoQ streaming + event planes, `StreamPublisher`, QoS/relay options
- `auth/` — JWT `Claims`, `Scope`, `ScopeRegistry`, `ucan/` (delegation), `mac/` (labels/lattice/context), `federation.rs`, `atproto_perimeter.rs`

### Spawner Details (`hyprstream-service`)

- **Tokio mode**: `tokio::spawn` — async-safe services
- **Thread mode**: `std::thread` + single-threaded tokio runtime — required for `!Send` types (tch-rs tensors)
- **Subprocess mode**: `ProcessSpawner` — Standalone or Systemd backends (auto-detected; `systemd-run` or PID-file tracking)
- Service lifecycle, dependency ordering (`depends_on`), and the per-service signing-key **trust store** live in `hyprstream-service::service::{spawner, manager, ordering, trust_store}`

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

**Interface policy — MAC on contracts that don't carry it (ratified, epic #547):** MAC context is **never a plaintext field of an API contract**. Contracts carry either verified credentials (checked once at session/attach time) or opaque handles to TCB-resolved context. Select the move by what already crosses the interface:
1. **Derive, don't extend** — a verified identity already crosses (RPC `EnvelopeContext`): compute `SecurityContext` at the boundary from `Claims × VerifiedKeyMaterial` (S1 invariant), cache, enforce via the AVC. No schema change.
2. **Extend at attach time, never per-op** — a session exists but no identity crosses (9P `Tattach`, #568): present a verified credential **once** at session establishment; cache the derived `SubjectCtx` fid/connection-scoped inside the TCB (the `mac::avc` amortization model); revoke via the AVC generation counter. Per-op params carry at most an opaque, unforgeable handle.
3. **Fill with deny/clamp** — nothing crosses or the source is outside the TCB: subjects **deny** (`DenyUnlabeledResolver` pattern), object labels **clamp to the importing boundary's floor** (D2 — join, never "unrestricted"), unverifiable key material **floors at Classical** (Decision D, #698).
4. **Forbidden** — caller-supplied labels/clearance as authoritative PDP inputs (a plain struct param is unauthenticated data: whoever constructs the call constructs the clearance). "Add a `SecurityContext` parameter to every method" is the *wrong* extension even though it looks most explicit. Labels in wire schemas are **hints** (D1 — trusted only from services we operate).

**Current status (check before assuming otherwise):** the MAC library is real and well-tested, but **enforcement is not active in production as of this writing** — the PDP has no wired-in PEP caller, the S6 grant path's HTTP dispatch still fails closed pending resolver/object-label wiring, and the audit store needs explicit startup construction. Everything fails closed (nothing is exploitable), but do not assume MAC is "live" — check the current wiring state (`services::oauth`, service factories) before relying on it protecting anything at runtime.

## Adding an RPC Method

1. Define in the `.capnp` schema (request/response structs + method variant) — service schemas live in `crates/hyprstream-rpc-std/schema/`
2. `cargo build -p hyprstream` (regenerates CGR metadata)
3. Implement handler with `#[authorize(action = "...", resource = "...")]`
4. Client code auto-generated by `generate_rpc_service!`

## Adding a New Service

1. Implement `RequestService` (or `Spawnable` directly for !Send types)
2. Add `#[service_factory("name", schema = "...", depends_on = [...])]` fn in `factories.rs`
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

- **The entire ZMQ/ZeroMQ stack** — libzmq sockets, CURVE transport encryption, `CurveConfig`, `ZmqService`, the XPUB/XSUB stream/event proxies, the ZMQ `StreamService` (#138 N4, #167). Replaced by ZMTP *framing* over pluggable transports (inproc/UDS/QUIC/iroh) + MoQ streaming/event planes. Do not reintroduce a zmq dependency or describe the RPC layer as "ZMQ".
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
cargo run --bin hyprstream service start
cargo fmt --all
cargo clippy --all-targets --all-features
./appimage/build-appimage.sh build rocm71
```

## Docs

See: `README.md`, `DEVELOP.md`, `CONTRIBUTING.md`, `crates/git2db/CLAUDE.md`, `docs/` (quickstart, vfs, mac-architecture, workers-architecture, rpc-architecture, cryptography-architecture, eventservice-architecture, service-runtime-architecture, streaming-service-architecture, tui-architecture, tool calling, KV cache), and the architecture talk: https://talks.cyberdione.ai/hyprstream-wanix

## Licensing

- **hyprstream**: MIT OR AGPL-3.0 (user's choice)
- **All other crates**: MIT
