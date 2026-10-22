# Native Streams deployment contract

This change stacks on Event PR1581. Set `quic.enabled=true`, `quic.iroh=true`,
and `quic.native_network_profile="network-iroh-required"`. Required native
processes neither serve nor dial local MoQ sockets, and native reach/relay
selection rejects browser QUIC alternatives. The compatibility/browser paths
remain available when this profile is not selected.

Registry, Model, TUI, and Metrics keep service-owned publisher origins attached
to the same native handler and signed stream reach their responses advertise.
The separate `streams` service owns an independent authenticated Iroh rendezvous
origin. Its signed announcement claims only `hyprstream-moq/1`, not RPC or the
tracks hosted by other producer services. Readiness follows successful carrier
bind and initial signed announcement; cancellation shuts down both publication
and the carrier. A listener being ready is not proof of authorized payload access.

Every native server uses the production live accepted-state authority, and
admission tenants are derived from it (#1652): a capsule service entry that
matches a registered deployment service derives the `local` tenant, everything
else — including foreign at9p records admitted through public ingest — is
unresolved and denied. There is no `quic.moql_subject_tenants` config map to go
stale at service-identity churn; each server's own DID must resolve to its own
service name or it refuses to spawn. Each server receives an independent
confirmation identity/replay cache; carrier addresses never select application
identity. Existing accepted-state expiry/rotation/removal checks remain active
throughout a session, and the derived tenant is re-resolved on every recheck.

`quic.stream_publishers` is a distinct explicit DID set, deliberately empty by
design: no production writer emits it and admitted native peers are read-only
on Streams. An empty set cannot go stale; it stays DID-keyed until a first real
use exists. It grants remote stream ingress only for the same DID's derived
tenant. Tenant membership alone leaves a peer read-only, and `event_publishers`
(a name-keyed Event grant) never authorizes Streams ingress. Populate any
non-empty ingress set from verified provisioning artifacts and an approved
roster, never invented DIDs or labels.

This transport does not grant MAC clearance. Existing producing RPC authorization,
per-recipient stream key provisioning, payload authentication/encryption and epoch
controls remain separate. No Event label table or permissive MAC policy is added.
IdentityAware production activation, approved policy rows, and fresh CLI identity/
proof/PEP provisioning remain deployment prerequisites. There is no fabricated
Streams health/probe CLI. Event's existing command is spelled:

```
hyprstream quick notify probe --timeout 10
```

It requires the separate legitimate CLI proof and Event authorization bootstrap.

## Inference ownership boundary

`ModelService` constructs `InferenceServiceConfig` with its own reach/origin
handles (`model.rs`, `with_stream_plane`), so those inference publishers use the
Model endpoint updated here. The inference runner's distinct Iroh endpoint serves
RPC and explicitly refuses MoQ; it is not advertised as the Model stream origin.

A separately deployed standalone inference process (`owns_stream_plane` plus its
custom `serve_inference_bridged` QUIC configuration) still has only a QUIC streaming
bridge. That separate path is unchanged and is not native Streams acceptance.
The normal eight-service staging roster uses Model-owned inference creation;
a standalone inference deployment requires its own bounded transport work.

## Verification boundary

The causal test uses the production config-to-admission/ingress/handler path,
independent real Iroh producer/subscriber peers, and the required-profile reach
dialer to carry an actual frame. It checks that Event ingress does not authorize
stream injection, another tenant cannot inject/read that frame, and removing an
accepted publisher closes its active session and denies reconnect. Other tests
cover no-reach/browser-only rejection, socket-free initialization, producer-origin
isolation, and announcement/readiness/shutdown ordering. Existing transport,
stream epoch, admission, relay and compatibility suites remain intact.

These are transport implementation results. Crosshost deployment acceptance still
requires actual provisioned identities, approved policy, fixed cold-start authority
ordering, a reviewed image, and real authenticated runtime probes.
