# hyprstream-discovery

Service discovery and endpoint resolution for the hyprstream cluster.

## What it does

`DiscoveryService` is the authoritative source for service endpoint lookup. It wraps the `EndpointRegistry` from `hyprstream-rpc` and exposes it as:

1. A local `Resolver` implementation — in-process lookup for services in the same daemon.
2. A Cap'n Proto RPC service — remote endpoint queries from clients or peer daemons.

### Operations

- `list_services` — enumerate registered services + their endpoint sets (QUIC, iroh, UDS, inproc).
- `get_endpoints(name)` — resolve all endpoints for a named service.
- `ping` — liveness check with latency measurement.
- `announce` — register a service endpoint (used at startup by each service).
- `auth_metadata` — return OAuth metadata for a service (issuer URL, scopes).

## Key types

| Type | Description |
|------|-------------|
| `DiscoveryService` | `RequestService` impl; wraps `EndpointRegistry`, authorises via `AuthorizationProvider` |
| `DiscoveryClient` | Generated typed client for the discovery RPC service |
| `AuthorizationProvider` | Trait: `authorize(subject, operation) → bool` |
| `ServiceEndpoints` | Name + list of `EndpointInfo` (kind, addr/nodeId, accept protocols) |

## Architecture position

```
hyprstream-rpc           (EndpointRegistry, Resolver trait, SocketKind)
    ↑
hyprstream-discovery     ← you are here
    ↑
hyprstream               (DiscoveryService factory, DiscoveryClient in OAuthService)
hyprstream-rpc-std       (re-exports DiscoveryClient for browser use)
```

Built as `rlib` only — `OnceLock` singletons must be shared across the workspace, not duplicated per cdylib.
