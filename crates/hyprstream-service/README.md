# hyprstream-service

Pluggable service orchestration: lifecycle management, trust store, and factory helpers.

## What it does

Three subsystems:

**Spawner** — run a `Spawnable` service as a process or in-process task:
- `ServiceSpawner` — dispatch table: look up a `ServiceFactory` by name, spawn it.
- `Spawnable` trait — implemented by every service type (already blanket-impl'd for `RequestService + Send + Sync`).
- `ProcessBackend`, `StandaloneBackend`, `SystemdBackend` — control how the process is started (fork, systemd unit, etc).
- `InprocManager` — run services on an in-process tokio LocalSet (the post-ZMQ default).

**Factory** — inventory-based service registration:
- `ServiceFactory` / `ServiceFactoryFn` — a constructor that returns `Box<dyn Spawnable>`.
- `get_factory(name)` / `list_factories()` — global inventory keyed by service name.
- `ServiceContext` — per-service context: signing key, ZMQ context, transport configs, verifying key derivation.
- `QuicSharedConfig` — QUIC cert + endpoint shared across all QUIC-serving services.

**Trust store** — peer key distribution at startup:
- `TrustStore` + `global_trust_store()` — maps service name → `VerifyingKey`; populated by each service factory before dependents start.
- `Attestation` — optional signed proof of key ownership.

## Architecture position

```
hyprstream-rpc           (RequestService, Spawnable trait, transport)
hyprstream-discovery     (DiscoveryService for startup ordering)
    ↑
hyprstream-service       ← you are here
    ↑
hyprstream               (calls get_factory / ServiceSpawner in main.rs)
```

Built as `rlib` only — `OnceLock` global state must not be duplicated across cdylib boundaries.

## Key re-exports

`ServiceSpawner`, `Spawnable`, `ServiceFactory`, `ServiceContext`, `TrustStore`, `global_trust_store`, `ServiceManager`, `startup_stages`.
