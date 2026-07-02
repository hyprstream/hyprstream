# Worker GA — Epic plan

**Date:** 2026-06-19 · **Branch:** `feature/kata-ga-epic` (off `main` @ 3176362a4) · all epic PRs
target this branch; merging it closes the epic. **Status:** planning (investigation done; no impl).

## Goal
Take `hyprstream-workers` to GA: **first-class Kata 4.0 (runtime-rs GA/default)** support in cooperation
with the Kata team, **plus** a casual-user path via **podman/docker + nested-userspace containers**, with
feature-gated builds so casual installs don't pay the Kata toolchain cost.

## Current state (investigation — `crates/hyprstream-workers`, `docs/workers-architecture.md`)
A mature **CRI-aligned** framework, NOT greenfield:
- **`WorkerService`** implements Kubernetes CRI (`RuntimeService` + `ImageService`): PodSandbox = sandbox
  VM, Container = OCI container. Schema `workers.capnp`.
- **`SandboxBackend` trait** (`runtime/backend.rs`) — clean pluggable abstraction (start/stop/destroy/
  reset/exec_sync/get_pids/update_resources/`is_available`). `BackendType` config selects it
  (**default Kata**, `services/factories.rs:538`, `spawner/mod.rs:131`). Two impls:
  - **`KataBackend`** (`kata_backend.rs`): embeds Kata **runtime-rs** `hypervisor` crate **directly as a
    library** (git tag **3.25.0**, cloud-hypervisor; `dragonball` feature-gated). Rootless config, VM
    lifecycle, virtiofs + **Nydus RAFS** image streaming, cloud-init ISO. Good unit tests.
  - **`NspawnBackend`** (`nspawn.rs`): **not an OCI runtime** — boots the *host* rootfs (`--directory=/`)
    under `systemd-nspawn` running `hyprstream service start --all`; bind-mounts libtorch+GPU; veth; exec
    via `machinectl shell`. A "daemon-in-a-namespace" model, ephemeral.
- **Nydus RAFS image store** (`image/`): chunk-level CAS, lazy-load, OCI/Docker manifest fetch.
- **Event bus** (ZMQ XPUB/XSUB), **WorkflowService** (GH-Actions YAML), **D-Bus bridge**, drafted
  **policy vocab** (`worker:`/`sandbox:`/`container:`/`image:`/`tool:` + Execute/Subscribe/Publish).

## GA-blocking gaps
1. **No guest agent → Kata can't run workloads.** `supports_exec()=false`; `exec_sync` errors "requires
   agent"; `runtime/service.rs:928–962` are placeholders. Kata today = VM lifecycle only; the kata-agent
   vsock channel + end-to-end CRI create/start/exec of a container *inside* the VM is unbuilt. **Biggest
   functional gap.**
2. **Kata 4.0 / runtime-rs integration-surface decision (coordination-critical).** Already on runtime-rs
   but pinned **3.25.0** and **embed the `hypervisor` crate directly** rather than via
   `containerd-shim-kata-v2` / the runtime-rs binary. For 4.0 GA: absorb breaking `hypervisor`/
   `kata-types`/`ch-config` API changes AND **negotiate with the Kata team whether library-embedding is a
   blessed/stable surface** — if not, first-class support may mean adopting the shim/runtime-rs entrypoint.
   **Top architectural decision.** (Kata-team brief deferred per user.)
3. **No podman/docker backend** — casual-user requirement unmet. Need a net-new `SandboxBackend` running
   arbitrary OCI images as rootless nested-userspace containers (podman/docker shell-out, or embed an OCI
   runtime — youki/crun). CRI trait + OCI manifest plumbing exist to build on; nspawn's host-rootfs model
   does NOT serve this.
4. **Build/packaging punishes casual users.** `kata-hypervisor`/`kata-types`/`ch-config`/`nydus-*`/
   `fuse-backend-rs` are **unconditional deps**; a podman-only user still compiles Kata+Nydus+CH. GA needs
   **feature-gating** (`kata`/`nspawn`/`podman`) + a lightweight default.
5. **No availability-based selection/fallback.** Backend is pure config; `is_available()` exists but isn't
   used to auto-pick or fall back (kata→podman→nspawn). Needs auto-detect + clear UX.
6. **WorkflowService partial** — TODOs in `scan_repo`, subscribe/unsubscribe, YAML round-trip, repo/
   training/metrics event routing (only `worker.*` wired); stubbed stats (`service.rs:708/748`), warm-pool
   replenish (`pool.rs:272`), image-size (`store.rs:430/576`).
7. **Still on ZMQ, not moq.** Crate uses `tmq`/`zmq`/`ZmqService` while the rest of hyprstream completed
   the ZMQ→moq/capnp-RPC migration. GA reconciliation (transport + secure event plane).
8. **Nydus git-dep tech debt** (published crates API mismatch → pinned git deps).

## Epic shape — 3 tracks + cross-cutting
**Track 1 — Kata 4.0 first-class**
- T1-A Bump runtime-rs 3.25.0 → 4.0 GA; absorb `hypervisor`/`kata-types`/`ch-config` API breaks.
- T1-B **Integration-surface decision** (embed hypervisor crate vs containerd-shim-kata-v2 / runtime-rs
  binary) — gated on Kata-team coordination. *(decision ticket)*
- T1-C **Guest agent (kata-agent vsock)** — make CRI create/start/exec_sync of a container-in-VM actually
  work; replace the `service.rs:928–962` placeholders. *(largest build)*
- T1-D Fill VM stats/console/get_pids/update_resources for real.

**Track 2 — Casual path**
- T2-A New rootless **podman/docker (or youki/crun) OCI `SandboxBackend`** for nested-userspace containers.
- T2-B **Feature-gate** the backends (`kata`/`nspawn`/`podman`); lightweight default for casual installs.
- T2-C **Availability-based auto-selection + fallback** (kata→podman→nspawn) + wizard/CLI UX.

**Track 3 — Productionize**
- T3-A Finish WorkflowService event routing (repo/training/metrics) + scan_repo + YAML round-trip.
- T3-B **ZMQ→moq/capnp-RPC migration** for the workers crate (transport + secure event plane).
- T3-C Warm-pool replenish, image-size calc, stats.
- T3-D Nydus dependency cleanup (drop pinned git deps).

**Cross-cutting — authz reconciliation (with epic #310).** Worker sandboxes ARE the "untrusted tenant
workloads / Kata-Wanix sandboxes" in the #310/#319/#328 threat model. Reconcile the workers policy vocab
(`worker:`/`sandbox:`/`container:`) with the per-host identity (#328) + Casbin mesh policy (#319) so
sandbox access is governed uniformly (deny-by-default, per-tenant isolation). No file overlap with #310;
this is the one real intersection.

## Key decisions
- **D1 (Kata integration surface):** embed `hypervisor` crate (status quo) vs adopt shim/runtime-rs binary.
  Needs Kata-team input (brief deferred). Blocks T1-A scope.
- **D2 (casual backend):** podman/docker shell-out vs embedded OCI runtime (youki/crun). Affects T2-A +
  dependency/feature surface.
- **D3 (default backend for casual install):** podman-by-availability vs explicit. Affects T2-C UX.
- **D4 (capnp/RPC, human-gated):** the guest-agent vsock channel and any podman-backend additions to
  `workers.capnp` / the RPC surface — engage human before schema changes.

## Risks
- Kata 4.0 API churn (rc/breaking) + the embed-vs-shim decision can reshape T1 substantially.
- Guest-agent vsock + in-VM exec is the deepest build; security (vsock auth, CURVE/AEAD deferred today).
- Feature-gating touches the whole crate's dep graph; must not break the Kata default for power users.
- ZMQ→moq migration in workers must reconcile with the already-migrated rest of the tree.

## Sequencing
T2-B (feature-gate) + T2-A (podman) deliver casual-user value fastest and are independent of the Kata-team
decision → reasonable to start in parallel with T1-A (the 4.0 bump). T1-C (guest agent) is the long pole.
T3-B (moq migration) should land before/with the cross-cutting authz reconciliation. Hardware/Kata-team
items gate T1-B/T1-C.
