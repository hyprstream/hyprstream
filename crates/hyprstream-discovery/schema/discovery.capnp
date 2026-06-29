@0xe7a3b5c9d1f2e4a6;

# Cap'n Proto schema for discovery service
#
# The discovery service exposes the EndpointRegistry over ZMQ RPC,
# allowing remote clients to discover registered services, their endpoints,
# socket kinds, and schemas.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".scope;
using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".optional;

# Unified discovery request with union discriminator
struct DiscoveryRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # List all registered services with descriptions
    listServices @1 :Void $scope(query) $mcpDescription("List all registered services with descriptions");

    # Get all endpoints for a named service
    getEndpoints @2 :Text $scope(query) $mcpDescription("Get all endpoints for a named service");

    # Get schema bytes for a service
    getSchema @3 :Text $scope(query) $mcpDescription("Get schema bytes for a service");

    # Health check
    ping @4 :Void $scope(query) $mcpDescription("Health check");

    # Get OAuth protected resource metadata (RFC 9728) for services
    # Text parameter filters by service name (empty = all services)
    getAuthMetadata @5 :Text $scope(query) $mcpDescription("Get OAuth protected resource metadata for services");

    # Removed: prepareStream (was insecure — bypassed DH key exchange).
    # Use StreamChannel::prepare_stream for authenticated streaming.
    # Reserved tombstone wire slots — scoped $manage so the dead handler stays
    # non-public under mandatory-scope (S3, #547).
    prepareStream @6 :Void $scope(manage);

    # Removed: getStream
    getStream @7 :Void $scope(manage);

    # Removed: listStreams
    listStreams @8 :Void $scope(manage);

    # Announce a service endpoint (used by services after QUIC binding)
    announce @9 :ServiceAnnouncement $scope(write) $mcpDescription("Announce a service endpoint for discovery");

    # Phase 0.5 Stage D — federation directory
    # Push a signed OpenID Federation 1.0 entity statement for an issuer (called by IdP/OAuth service)
    registerEntityStatement @10 :RegisterEntityStatementRequest $scope(write) $mcpDescription("Register a signed OIDF entity statement for an issuer");

    # Fetch a cached signed entity statement for an issuer (used by FederationKeyResolver before HTTPS fallback)
    getEntityStatement @11 :Text $scope(query) $mcpDescription("Fetch cached signed entity statement for issuer URL");

    # Push a COSE_KeySet (CBOR) for a service's envelope-signing keys (called by each service at startup + rotation)
    registerEnvelopeKeyset @12 :RegisterEnvelopeKeysetRequest $scope(write) $mcpDescription("Register envelope COSE_KeySet for a service");

    # Fetch a cached COSE_KeySet for a service's envelope keys
    getEnvelopeKeyset @13 :Text $scope(query) $mcpDescription("Fetch cached envelope COSE_KeySet for service DID");

    # List all known issuer URLs whose entity statements are cached (authenticated)
    listKnownIssuers @14 :Void $scope(query) $mcpDescription("List issuers with cached entity statements");

    # #431 — federated record lookup. Fetch an atproto record (ai.hyprstream.model)
    # as a verifiable CARv1 proof so the caller can validate it offline. The
    # auto-generated discovery:query gate runs first; the handler additionally
    # access-control-checks the *target* DID/collection (an at:// CID is public/
    # predictable, so a valid address alone must NOT grant a read).
    getRecord @15 :GetRecordRequest $scope(query) $mcpDescription("Fetch an atproto record (ai.hyprstream.model) as a verifiable CAR proof");

    # #431 — fetch a full atproto repo CAR by DID (commit + MST + all records).
    getRepo @16 :Text $scope(query) $mcpDescription("Fetch a full atproto repo CAR by DID");
  }
}

# Unified discovery response
struct DiscoveryResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload
  # Convention: response variant = request name + "Result"
  union {
    # Error occurred
    error @1 :ErrorInfo;

    # List of registered services (for listServices)
    listServicesResult @2 :ServiceList;

    # Endpoints for a service (for getEndpoints)
    getEndpointsResult @3 :ServiceEndpoints;

    # Schema bytes for a service (for getSchema)
    getSchemaResult @4 :Data;

    # Ping result (for ping)
    pingResult @5 :PingInfo;

    # Auth metadata result (for getAuthMetadata)
    getAuthMetadataResult @6 :AuthMetadataList;

    # Removed: prepareStreamResult
    prepareStreamResult @7 :Void;

    # Removed: getStreamResult
    getStreamResult @8 :Void;

    # Removed: listStreamsResult
    listStreamsResult @9 :Void;

    # Acknowledge endpoint announcement
    announceResult @10 :Void;

    # Phase 0.5 Stage D — federation directory responses
    registerEntityStatementResult @11 :Void;
    getEntityStatementResult @12 :EntityStatement;
    registerEnvelopeKeysetResult @13 :Void;
    getEnvelopeKeysetResult @14 :EnvelopeKeyset;
    listKnownIssuersResult @15 :IssuerList;

    # #431 — record/repo lookup results (paired with the requests above).
    getRecordResult @16 :RecordCar;
    getRepoResult @17 :RecordCar;
  }
}

# Summary of a registered service
struct ServiceSummary {
  # Service name
  name @0 :Text;
  # Optional human-readable description
  description @1 :Text;
  # Socket kinds (lowercase text, e.g. "rep", "router", "xpub")
  socketKinds @2 :List(Text);
  # Whether the service has a schema registered
  hasSchema @3 :Bool;
}

# List of service summaries
struct ServiceList {
  services @0 :List(ServiceSummary);
}

# Endpoint information
# FLAG-DAY: field numbers renumbered in integration-1 (self-proof fields removed).
# All services and clients must be upgraded together.
struct EndpointInfo {
  # Socket kind as lowercase text (e.g. "rep", "router")
  socketKind @0 :Text;
  # Endpoint string (e.g. "inproc://hyprstream/policy")
  endpoint @1 :Text;
  # Service identity JWT (carries sub, pub, exp — replaces self-proof fields)
  serviceJwt @2 :Text;
  # TLS endorsement: Sign(tls_key, ed25519_pubkey || domain) — optional
  tlsEndorsement @3 :Data;
  # Domain the TLS cert covers — optional, present when tlsEndorsement is set
  tlsDomain @4 :Text;
}

# Endpoints for a specific service
struct ServiceEndpoints {
  endpoints @0 :List(EndpointInfo);
}

# Ping/health information
struct PingInfo {
  # Status string (e.g. "ok")
  status @0 :Text;
  # Number of registered services
  serviceCount @1 :UInt32;
  # Uptime in seconds
  uptime @2 :UInt64;
}

# OAuth protected resource metadata for a service (RFC 9728)
struct AuthMetadata {
  # Service name (e.g. "registry", "model", "oai")
  serviceName @0 :Text;
  # Resource URL (e.g. "https://localhost:4433/registry")
  resource @1 :Text;
  # Authorization server URLs
  authorizationServers @2 :List(Text);
  # Supported OAuth scopes
  scopesSupported @3 :List(Text);
  # Human-readable resource name
  resourceName @4 :Text;
}

# List of auth metadata entries
struct AuthMetadataList {
  services @0 :List(AuthMetadata);
}

# Phase 0.5 Stage D — federation directory types

# Request: push a signed entity statement for an issuer
struct RegisterEntityStatementRequest {
  # OAuth issuer URL (e.g. "https://hyprstream.example.com")
  issuer @0 :Text;
  # Signed OpenID Federation 1.0 entity statement (compact JWS)
  jwt @1 :Text;
}

# Cached signed entity statement
struct EntityStatement {
  # OAuth issuer URL this statement is about
  issuer @0 :Text;
  # Signed OpenID Federation 1.0 entity statement (compact JWS)
  jwt @1 :Text;
  # Unix seconds when this copy was registered in DiscoveryService
  fetchedAt @2 :Int64;
}

# Request: push a service's COSE_KeySet (CBOR) for envelope-signing keys
struct RegisterEnvelopeKeysetRequest {
  # did:web of the service this keyset belongs to
  serviceDid @0 :Text;
  # CBOR-encoded COSE_KeySet (RFC 9052 §7) containing the service's envelope verification keys
  coseKeysetCbor @1 :Data;
}

# Cached envelope COSE_KeySet
struct EnvelopeKeyset {
  serviceDid @0 :Text;
  coseKeysetCbor @1 :Data;
  fetchedAt @2 :Int64;
}

# List of known issuer URLs
struct IssuerList {
  issuers @0 :List(Text);
}

# Service endpoint announcement (sent by services after QUIC binding)
struct ServiceAnnouncement {
  # Service name (e.g. "registry", "policy", "model")
  serviceName @0 :Text;
  # Socket kind (e.g. "quic", "rep")
  socketKind @1 :Text;
  # Endpoint string (e.g. "quic://localhost:0.0.0.0:4433")
  endpoint @2 :Text;
  # Service JWT attesting to the service's pubkey and identity
  serviceJwt @3 :Text $optional;
}

# #431 — federated record lookup as a verifiable CAR proof.

# A record returned as a CARv1 proof: signed commit + MST path + record block.
# The caller validates it offline via hyprstream-pds verify_record_proof. The
# DiscoveryService is treated as an untrusted relay — integrity comes from the
# signed proof, not from trusting the responder.
struct RecordCar {
  uri @0 :Text;   # at:// URI this CAR answers (caller binds the proof to its request)
  car @1 :Data;   # CARv1 bytes: roots=[commit CID]; blocks = commit + MST path + record
}

# Resolve a single record by at:// or by (did, collection, rkey).
struct GetRecordRequest {
  uri @0 :Text;          # full at://<did>/<collection>/<rkey>; if set, fields below ignored
  did @1 :Text;
  collection @2 :Text;   # e.g. "ai.hyprstream.model"
  rkey @3 :Text;         # the TID record key
}
