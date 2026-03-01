@0xe7a3b5c9d1f2e4a6;

# Cap'n Proto schema for discovery service
#
# The discovery service exposes the EndpointRegistry over ZMQ RPC,
# allowing remote clients to discover registered services, their endpoints,
# socket kinds, and schemas.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;

# Unified discovery request with union discriminator
struct DiscoveryRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # List all registered services with descriptions
    listServices @1 :Void $mcpScope(query) $mcpDescription("List all registered services with descriptions");

    # Get all endpoints for a named service
    getEndpoints @2 :Text $mcpScope(query) $mcpDescription("Get all endpoints for a named service");

    # Get schema bytes for a service
    getSchema @3 :Text $mcpScope(query) $mcpDescription("Get schema bytes for a service");

    # Health check
    ping @4 :Void $mcpScope(query) $mcpDescription("Health check");

    # Get OAuth protected resource metadata (RFC 9728) for services
    # Text parameter filters by service name (empty = all services)
    getAuthMetadata @5 :Text $mcpScope(query) $mcpDescription("Get OAuth protected resource metadata for services");
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
  }
}

# Error Information
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
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
struct EndpointInfo {
  # Socket kind as lowercase text (e.g. "rep", "router")
  socketKind @0 :Text;
  # Endpoint string (e.g. "inproc://hyprstream/policy")
  endpoint @1 :Text;
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
