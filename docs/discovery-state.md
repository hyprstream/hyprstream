# Discovery volatile state

Discovery keeps endpoint announcements, node liveness, and cached federation
artifacts behind one typed state contract. These records are volatile routing
inputs, not identity or policy authority: handlers validate writes and
resolution re-checks accepted-current state and policy before use.

The default is bounded in-process memory and is appropriate for one Discovery
process, tests, embedded use, and WASM. An active-active deployment must build
Hyprstream with the `discovery-valkey` feature and explicitly select either
`valkey` or `tiered`. Startup rejects `active_active = true` with the memory
backend; there is no connectivity-driven fallback.

```toml
[services.discovery.state]
backend = "tiered"
active_active = true

[services.discovery.state.memory]
announcement_capacity = 16384
liveness_capacity = 16384
artifact_capacity = 4096

[services.discovery.state.valkey]
url = "rediss://discovery-state.example:6379"
key_prefix = "production"
pool_size = 8
announcement_capacity = 65536
liveness_capacity = 65536
artifact_capacity = 16384
command_timeout_ms = 2000

[services.discovery.state.tiered]
l1_max_ttl_ms = 1000
```

`tiered` is a bounded memory L1 with write-through to Valkey. Every L1 use
first verifies its L2 revision, and cached lifetime is clamped to both the L1
freshness window and the signed/effective record expiry. An L2 error is
returned to the caller rather than serving an isolated or expired L1 value.

All keys for one configured prefix share a Valkey cluster hash tag so atomic
Lua updates remain in one slot. Use a deployment-specific prefix when multiple
independent Hyprstream environments share a Valkey cluster.
