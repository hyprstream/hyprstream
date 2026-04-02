@0xc8d9e0f1a2b3c4d5;

# Event bus schema for hyprstream EventService
#
# This schema defines ONLY the generic EventEnvelope used for pub/sub messaging
# over the XPUB/XSUB proxy. Service-specific event types are defined in
# their respective crates:
#   - hyprstream-workers: WorkerEvent (sandbox/container lifecycle)
#   - hyprstream: RegistryEvent, ModelEvent, InferenceEvent
#
# Topic Format: {source}.{entity}.{event}
#   Examples:
#     worker.sandbox123.started    - WorkerService: sandbox started
#     registry.repo789.push        - RegistryService: repository push event
#     model.qwen3.loaded           - ModelService: model loaded
#     inference.session123.completed - InferenceService: generation completed
#
# Delivery Semantics:
#   - At-most-once (fire-and-forget, no persistence)
#   - In-order per publisher (no cross-publisher guarantees)
#   - Late-join subscribers only see events after subscribing

# Generic event envelope for pub/sub messaging
#
# The payload contains service-specific event data serialized as bytes.
# Consumers deserialize based on topic prefix:
#   - topic "worker.*" -> deserialize as WorkerEvent (from hyprstream-workers)
#   - topic "registry.*" -> deserialize as RegistryEvent (from hyprstream)
#   - etc.
struct EventEnvelope {
  # Unique event ID (UUID v4, 16 bytes)
  id @0 :Data;

  # Unix timestamp in milliseconds
  timestamp @1 :Int64;

  # Source service name (e.g., "worker", "registry", "model", "inference")
  source @2 :Text;

  # Full topic string for ZMQ prefix filtering
  # Format: {source}.{entity}.{event}
  topic @3 :Text;

  # Service-specific event data (serialized Cap'n Proto or other format)
  payload @4 :Data;

  # Optional correlation ID for distributed tracing (UUID v4, 16 bytes)
  # Links related events across services (e.g., workflow run ID)
  correlationId @5 :Data;
}

# Event registration for secure pub/sub prefix management
struct EventRegistration {
  topicPrefix @0 :Text;
  schema @1 :Text;
  publishScope @2 :Text;
  subscribeScope @3 :Text;
}

# V2 envelope with publisher authentication (E2E signed)
struct EventEnvelopeV2 {
  topic @0 :Text;
  payload @1 :Data;
  publisherSignature @2 :Data;
  timestamp @3 :Int64;
  publisherPubkey @4 :Data;
}

# Announcement that group key is rotating
struct RekeyAnnouncement {
  prefix @0 :Text;
  effectiveAt @1 :Int64;
  wrappedKeys @2 :List(WrappedKey);
  versionId @3 :UInt64;
  publisherEphemeralPubkey @4 :Data;
}

# A wrapped symmetric key for a single subscriber
struct WrappedKey {
  routingTag @0 :Data;
  wrapped @1 :Data;
}

# Control-plane announcement union
struct Announcement {
  union {
    prefixAvailable @0 :PrefixAvailableAnnouncement;
    rekey @1 :RekeyAnnouncement;
    prefixOffline @2 :PrefixOfflineAnnouncement;
  }
}

# A prefix is available for subscription
struct PrefixAvailableAnnouncement {
  prefix @0 :Text;
  publisherEphemeralPubkey @1 :Data;
  schema @2 :Text;
}

# A prefix is going offline
struct PrefixOfflineAnnouncement {
  prefix @0 :Text;
  reason @1 :Text;
}
