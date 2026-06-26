# hyprstream-rpc

Core RPC infrastructure for the hyprstream service mesh.

## What it does

This crate is the foundation of the RPC plane. It provides:

- **Transport abstraction**: `TransportConfig` + `EndpointType` (QUIC/WebTransport, iroh, UDS inproc, in-memory). All transports implement the same bidi-stream interface.
- **Lazy transports**: `LazyQuinnTransport`, `LazyIrohTransport`, `LazyUdsTransport` — reconnect on demand with exponential backoff.
- **Dial**: `dial()` — resolve a `TransportConfig` to a live `RpcClient`.
- **Signed envelopes**: COSE_Sign1 request/response envelopes (Ed25519 + ML-DSA-65 hybrid). Every RPC call is authenticated at the application layer independent of transport.
- **Cap'n Proto codec**: `ToCapnp` / `FromCapnp` traits + generated schema modules (`common_capnp`, `events_capnp`, `streaming_capnp`, `nine_capnp`, `annotations_capnp`).
- **Service dispatch**: `RequestService` + `IrohRequestProcessor` + `LocalServiceBridge` — the post-ZMQ server-side dispatch path.
- **In-memory registry**: `register_inproc` / `lookup_inproc` — zero-copy inproc dial for same-process services.
- **DID-doc service entry codec**: `service_entry` — encode/decode `IrohTransport` and `QuicTransport` entries in DID documents.

## Architecture position

```
hyprstream-rpc           ← you are here
    ↑
hyprstream-{rpc-std,discovery,service,workers,vfs,9p}
    ↑
hyprstream               (application, factories, OAuth, CLI)
```

ZMQ/ZMTP was removed in #131/#138; all transport code lives here.

## Key types

| Type | Purpose |
|------|---------|
| `TransportConfig` | Dialable endpoint (QUIC, iroh, UDS, inproc) |
| `QuicServerAuth` | TLS auth policy for QUIC dial (WebPKI / pinned cert hash) |
| `LazyState<T>` | Cached session + `ReconnectBackoff` under one `Mutex` |
| `LocalServiceBridge` | Bridge a `RequestService` to an `IrohRequestProcessor` |
| `SignedEnvelope` | COSE_Sign1 wrapper for authenticated RPC payloads |
| `RpcClient` | Async bidi-stream RPC client |
| `RpcConfig` | Server-side concurrency + timeout configuration |
| `MoqStreamHandle` | Reconnectable moq-lite stream consumer handle |
| `RpcServerConfig` | QUIC/iroh server bind configuration |

## Feature flags

- Default: Ristretto255 DH, BLAKE3 MAC.
- `fips`: ECDH P-256 + HMAC-SHA256 for FIPS 140-2.
- `wasm32`: transport subset (no ZMQ, no UDS); WASM bindings in `hyprstream-rpc-std`.
