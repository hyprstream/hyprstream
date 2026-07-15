# NodeId role audit (#1031)

Snapshot: 2026-07-15, current-main standalone slice. This audit covers the
trust-relevant service-entry, native/wasm iroh, admission, discovery/PDS, and
OAuth federation/JWKS surfaces requested by #1031. A NodeId/`EndpointId` is an
iroh carrier address and diagnostic only. It is never a DID, admission subject,
assurance source, response-verification key, KEM selector, or authorization
identity.

## Retained transport-address occurrences

- `hyprstream-rpc/src/transport/mod.rs`: `EndpointType::Iroh.node_id` is the
  typed dial address. Formatting it creates only an `iroh://` endpoint string.
- `hyprstream-rpc/src/transport/lazy_iroh.rs`: converts that address to iroh's
  `EndpointId` and supplies relay/direct candidates to `Endpoint::connect`.
  Wrong-address handshake failure is transport path integrity, not identity
  admission or assurance.
- `hyprstream-rpc/src/transport/iroh_substrate.rs`: owns the local endpoint key,
  exposes its address for advertisement/diagnostics, and publishes/consumes
  pkarr relay/direct reach hints. The pkarr newtype has no identity conversion.
- `hyprstream-rpc/src/service_entry.rs`: encodes and decodes `nodeId` only inside
  `TransportConfig`. `DecodedEntry` has no identity/key field.
- `hyprstream-rpc/src/iroh_peer.rs` and
  `hyprstream-rpc-std/src/iroh_exports.rs`: wasm exposes endpoint bytes/z-base32
  and pkarr reach lookup. NodeId-to-`did:key` conversion APIs were removed.
- `hyprstream-service/src/service/spawner/service.rs` and
  `hyprstream/src/services/oauth/{state,mod,did_document}.rs`: derive/bind and
  advertise the local iroh address. The DID document carries it only in an
  `IrohTransport` service entry, never a verification method or JWKS entry.

## Retained liveness and routing hints

- Relay URLs, direct socket addresses, successful dials, and pkarr records are
  passed only to typed transport constructors. None is passed to the identity
  resolver, federation gate, MAC label/assurance logic, response-key selection,
  or KEM selection.
- A signed at9p genesis service relay is a durable reach claim, not live reach.
  `hyprstream-discovery/src/at9p_resolver.rs` now fails closed because the
  current schema lacks an independent carrier `EndpointId`; it does not derive
  one from the genesis subject Ed25519 key.

## Retained carrier comparison/diagnostic occurrences

- Native iroh internally verifies that the connection reached the addressed
  `EndpointId`. This may reject a wrong route, but it cannot grant or alter
  identity assurance.
- `Connection::remote_id()` is not consumed by RPC admission, federation, or
  MoQ tenant authorization. MoQ supplies `PeerIdentity::anonymous()` until
  #1027 provides fresh inside-carrier proof against accepted current state.

## Removed authority surfaces

- `DecodedEntry::identity_key` and its NodeId assignment.
- `IrohPeerAdmission`, optional accept-open admission, and the adapter that
  synthesized `did:key(remote_id)` and used it as a policy subject.
- wasm `did:key` ↔ NodeId converters and `IrohPeer.didKey()`.
- OAuth root-DID `#iroh` verification-method publication.
- Genesis subject-key → iroh NodeId/reach synthesis.

Generic `did:key` codecs and method-dispatched identity resolution remain: they
operate on an explicitly supplied DID and clamp assurance to Classical. They do
not discover reach or consume a NodeId.
