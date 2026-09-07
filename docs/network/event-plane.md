# Native Event plane

With `quic.native_network_profile = "network-iroh-required"`, the Event
service owns an Iroh substrate and publishes its initial signed Discovery
announcement before reporting service readiness. Binding or announcement failure
aborts startup. Shutdown cancels publication and closes the substrate. Event
refuses RPC and advertises only `hyprstream-moq/1`; resolve it with
`ServiceQuery::network_moq("event")`.

Other services install their checkpoint-selected process proof before factories
start. Their Event link resolves reach and server witness together, runs mutual
hybrid MoQL admission, and reconnects through a fresh resolution after failure.
The native Event initializer never opens a UDS. Compatibility mode retains the
existing UDS behavior.

## Independent authorization boundaries

The deployment must explicitly map each admitted service/client DID to `local`
in `quic.moql_subject_tenants`. The fixed Event namespace is `local/events`;
other tenants cannot see or publish it. Carrier EndpointIds do not select tenants.

A separate `quic.event_publishers` set grants the listed DIDs the ability to seek
Event ingress. Membership in the tenant map does not imply that grant. On top of
admission and ingress, the installed Event MAC reference monitor decides the
allowed sources and directions. The carrier exposes only those source scopes.
An empty policy table, absent verified clearance, or undeclared source denies.
Current accepted state, ingress grants, and Event MAC scopes are rechecked while
the session lives; loss/change of authority closes the connection.

The temporary source inventory is `system`, `worker`, and `model`. It declares
source names, **not security labels or clearances**. Per-OID publication paths
are not exposed by this initial network port. Generated inventory integration
is tracked by #1530 / #1505.

Local plaintext Event publishers and subscribers receive the checkpoint-selected
process DID before construction. This identity is an input to MAC; it does not
create a verified subject context or authorize any operation. Explicit caller
and publisher-identity overrides retain their existing behavior.

## Probe and remaining activation requirements

`hyprstream notify probe --timeout 10` requires the native profile, a provisioned
checkpointed client proof, fresh Event resolution, mutual admission, and actual
delivery on `local/events/system`'s `events` track. It uses a fresh subscriber
origin, so a local retained broadcast cannot satisfy the probe. No Event RPC or
UDS fallback is involved. A handshake alone is insufficient. The probe needs a
live/retained system event; there is no periodic Event heartbeat in this port.

This transport port does **not** activate production Event access:

- Both production MAC policy tables remain empty until reviewed declared labels
  are available. IdentityAware activation and signed policy/clearance provisioning
  remain required; this change does not install system-low rows or FloorOnly
  staging exceptions.
- CLI `notify subscribe`, `notify probe`, and `load --wait` select the native path,
  but CLI bootstrap still needs a legitimate checkpointed client proof. When that
  proof is absent, initialization fails explicitly instead of using a service
  identity or falling back to UDS.
- The deployment must provision actual DID-to-tenant bindings and independent
  ingress grants, then verify allowed and denied operations across hosts.
- No live staging deployment, image publication, or cloud configuration is part
  of this change. Transport tests do not establish staging readiness.
