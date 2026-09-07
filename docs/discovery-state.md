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
caller value under a subsequently observed revision. Operations on the same
cache scope are serialized through a fixed array of 64 mutexes; unrelated
scopes can use the Valkey pool concurrently. Hash collisions share a mutex,
but identity churn cannot grow a lock map. Fills bind their snapshot to a revision read before the
values. Every L1 use verifies its L2 revision, and cached lifetime is clamped
to both the L1 freshness window and the signed/effective record expiry.
Listings (announcements, live nodes, issuers) read Valkey directly. Oversized
point results skip L1 admission; a full L1 never rejects a healthy L2 result
or turns a successful shared write into a cache-capacity error. Cache values
and revision bookkeeping are bounded; unknown-key misses are not cached. An L2
error is returned to the caller rather than serving an isolated or expired L1
value.
Announcement, liveness, and issuer-cache revisions use one persistent generation counter per
family, not one key per service/node. A write can therefore invalidate unrelated
L1 entries in the same family; counters must never expire or reset while any
replica retains L1 state. Announcement writes and listings atomically reap
expired values, service indexes, and name metadata before admitting new state
or enumerating services. Backend metadata and listing work are bounded by the
configured live capacity even when identities continually change.

Entity statements are inert cached artifacts, not a registered-issuer authority.
At capacity, a new issuer atomically evicts the existing entry with the oldest
server `fetched_at` timestamp, including its name and ordering metadata. Updating
an issuer refreshes its eviction order. Memory and Valkey use the same policy;
JWT claims do not control cache eviction or replace verification at use. Issuer
listing reads the bounded names hash in one command; registration obtains only
the count with `SCARD`. Envelope keysets retain their separate bounded admission
policy and do not share the issuer eviction index.

When upgrading from the experimental per-scope revision implementation, stop
all old replicas and retire its configured **volatile Discovery key prefix**,
then restart all replicas with an empty prefix and let services reannounce.
Old writers do not maintain the new issuer eviction index or liveness generation, and previously orphaned
metadata has already lost its expiry-index membership; it cannot be reclaimed
by normal expiry traversal. This is not a rolling-compatible state migration.
Do not clear shared generations with running replicas, or touch the separate
checkpointed PDS identity store as part of this volatile-state reset.

The Valkey URL has no implicit loopback default and must be configured
explicitly whenever `valkey` or `tiered` is selected. Native Valkey builds enable
Fred's Rustls backend: `rediss://` uses TLS and rejects a plaintext server.
The connection initializes the existing external-interoperability TLS provider;
Valkey and its TLS dependencies remain excluded from the WASM target.

All keys for one configured prefix share a Valkey cluster hash tag so atomic
Lua updates remain in one slot. Use a deployment-specific prefix when multiple
independent Hyprstream environments share a Valkey cluster.

Shared backends own a dedicated connection-driver runtime until the last state
handle is dropped. They remain usable after synchronous factory bootstrap exits;
startup and commands fail closed on connection errors, with bounded timeouts.

Candidate queries enumerate the shared, capacity-bounded liveness index. A
replica then ingests missing placement records through the existing verified
repository resolver and bounded retry gate. Resource and already-warm label filtering precede RPC work;
Policy authorizes bounded vectors of at most 256 resources, with at most 16
batches in flight. The batch handler reuses the exact single-check verified
subject/tenant/audit boundary and requires an ordered decision for every entry.
There is no authorization cache or duplicate second authorization pass. Denied
nodes never trigger hydration. Two bulk liveness snapshots replace per-node
Valkey reads; the second excludes expiry during authorization/hydration. Ranking
and exact authorized `totalMatching` still inspect all eligible survivors before
applying maxCandidates. The exact-count contract inherently requires O(fleet)
local work; a small result limit cannot justify an incomplete count. A warm
65,536-node tiered query is exercised at the advertised capacity.

Cold verified-repository hydration remains bounded at 16 operations. A completely
cold large fleet can require retries or prior heartbeat-driven ingestion; bounded
batch authorization does not pretend that arbitrary remote repository reads can
finish within one deadline. Native queries retain the five-second deadline
covering enumeration through final filtering. A deadline or still-pending
ingest returns an error requiring retry, never a successful partial candidate
set or understated `totalMatching`. Completed verified records remain available
for the retry; cancelled ingests are retryable and are not cached as absence.
The WASM memory path uses the same concurrency bound and checks elapsed time
between operations, without pulling in a native timer or socket backend.
Nodes without verified placement facts, query authorization, or a still-live
heartbeat are excluded. Shared
liveness never grants identity, placement labels, or policy authority.

Cross-service announcement listings read names and values in one Lua operation
over the capacity-bounded expiry index, together with atomic metadata cleanup.
They do not perform network round trips per service or populate point-cache L1.

Heartbeat `last_seen` is the admitted server receipt time; the node's `ts` cannot
poison ordering after future skew or clock rollback. Shared writes take Valkey
TIME atomically with storage and PXAT expiry (at most the 45-second heartbeat
TTL); replica absolute timestamps never select ordering or expiry. Concurrent
shared writes follow Valkey receipt order, not a comparison of caller clocks.
Shared listings reap with Valkey TIME and point reads use Valkey key expiry;
tiered liveness does not use host-clock L1 caching. Memory-only deployments keep
their local receipt ordering. The bounded listing retires legacy volatile rows
without the shared-receipt marker, since their clock-derived lifetime cannot be
trusted; a new admitted heartbeat replaces them. No PDS state reset is involved.
Deploy the batch-capable Policy service before Discovery uses batching; an old
Policy endpoint fails closed, without a fleet-sized per-node RPC fallback.
This changes no identity authority:
signed/accepted announcement expiry
is independently enforced, and refreshing liveness does not renew it.

Identity-bound announcements remain limited by signed and accepted-state
expiry. Legacy non-identity-bound announcements use the heartbeat TTL when the
wire identity-expiry field is absent; they do not acquire identity evidence.

Run the repaired state encoding on every active-active replica. Liveness
records written by the earlier experimental implementation lack the node DID
needed for enumeration and must be refreshed by a heartbeat (or expire). A
mixed deployment with old writers fails closed on those records; it does not
silently omit nodes or substitute local candidates.
