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
[discovery.state]
backend = "tiered"
active_active = true

[discovery.state.memory]
announcement_capacity = 16384
liveness_capacity = 16384
artifact_capacity = 4096

[discovery.state.valkey]
url = "rediss://discovery-state.example:6379"
key_prefix = "production"
pool_size = 8
announcement_capacity = 65536
liveness_capacity = 65536
artifact_capacity = 16384
command_timeout_ms = 2000

[discovery.state.tiered]
l1_max_ttl_ms = 1000
```

`discovery` is a root configuration table; `services` contains only the startup
list. Putting these settings under `services.discovery` does not configure the
backend.

`tiered` uses bounded memory for point lookups over authoritative Valkey state.
Writes invalidate local entries before updating Valkey; they never cache a
caller value under a subsequently observed revision. Local cache operations
are serialized, and fills bind their snapshot to a revision read before the
values. Every L1 use verifies its L2 revision, and cached lifetime is clamped
to both the L1 freshness window and the signed/effective record expiry.
Listings (announcements, live nodes, issuers) read Valkey directly. Oversized
point results skip L1 admission; a full L1 never rejects a healthy L2 result
or turns a successful shared write into a cache-capacity error. Cache values
and revision bookkeeping are bounded; unknown-key misses are not cached. An L2
error is returned to the caller rather than serving an isolated or expired L1
value.
The Valkey URL has no implicit loopback default and must be configured
explicitly whenever `valkey` or `tiered` is selected.

All keys for one configured prefix share a Valkey cluster hash tag so atomic
Lua updates remain in one slot. Use a deployment-specific prefix when multiple
independent Hyprstream environments share a Valkey cluster.

Shared backends own a dedicated connection-driver runtime until the last state
handle is dropped. They remain usable after synchronous factory bootstrap exits;
startup and commands fail closed on connection errors, with bounded timeouts.

Candidate queries enumerate the shared, capacity-bounded liveness index. A
replica then ingests missing placement records through the existing verified
repository resolver and bounded retry gate. Nodes without verified placement
facts, query authorization, or a still-live heartbeat are excluded. Shared
liveness never grants identity, placement labels, or policy authority.

Identity-bound announcements remain limited by signed and accepted-state
expiry. Legacy non-identity-bound announcements use the heartbeat TTL when the
wire identity-expiry field is absent; they do not acquire identity evidence.

Run the repaired state encoding on every active-active replica. Liveness
records written by the earlier experimental implementation lack the node DID
needed for enumeration and must be refreshed by a heartbeat (or expire). A
mixed deployment with old writers fails closed on those records; it does not
silently omit nodes or substitute local candidates.
