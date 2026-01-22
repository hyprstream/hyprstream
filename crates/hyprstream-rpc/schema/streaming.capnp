@0xe7f8a9b0c1d2e3f4;

# Streaming Cap'n Proto types for hyprstream RPC
#
# This schema defines all streaming primitives for PUB/SUB communication.
# Consolidates streaming types that were previously split across common.capnp
# and application schemas (inference.capnp, model.capnp).
#
# Architecture:
#   Publisher (InferenceService) --PUSH--> StreamService --XPUB--> Subscriber (Client)
#
# Security Model (E2E Authentication):
#   - DH key exchange: Client and Publisher derive shared secret via Ristretto255
#   - Topic derivation: HKDF(shared_secret) -> (topic, mac_key)
#   - HMAC chain: Each chunk authenticated, chain enforces ordering
#   - StreamService: Blind forwarder (no HMAC verification)
#   - Client: Verifies HMAC chain end-to-end

# =============================================================================
# Stream Setup
# =============================================================================

# Stream metadata returned when starting a stream
#
# Contains everything the client needs to subscribe and derive keys.
# Returned by generateStream/inferStream RPC calls.
struct StreamInfo {
  streamId @0 :Text;      # Unique stream identifier (e.g., "stream-{uuid}")
  endpoint @1 :Text;      # XPUB endpoint to subscribe to
  serverPubkey @2 :Data;  # Server's ephemeral Ristretto255 public key (32 bytes) for DH
}

# Stream registration - wrapped in SignedEnvelope for authorization
#
# Publisher sends this to StreamService before streaming.
# StreamService verifies signature and checks claims.
# Topic is DH-derived (64 hex chars), unpredictable to StreamService.
struct StreamRegister {
  topic @0 :Text;    # DH-derived topic (e.g., hex(HKDF(shared)[..32]))
  exp @1 :Int64;     # Expiration timestamp (Unix millis)
}

# Request to start/authorize a stream subscription
#
# Client calls this after receiving StreamInfo to authorize subscription.
# Future: may include client ephemeral pubkey for late DH exchange.
struct StreamStartRequest {
  streamId @0 :Text;        # Stream ID from StreamInfo
  clientPubkey @1 :Data;    # Client's ephemeral Ristretto255 public key (32 bytes)
}

# Response confirming stream authorization
struct StreamAuthResponse {
  streamId @0 :Text;        # Confirmed stream ID
  serverPubkey @1 :Data;    # Server's ephemeral Ristretto255 public key (if not in StreamInfo)
}

# =============================================================================
# Wire Format (PUSH/PULL -> XPUB/SUB)
# =============================================================================

# Stream chunk - wire format for individual chunks
#
# Uses chained HMAC-SHA256 for authentication AND cryptographic ordering.
# No sequence field needed - ordering is enforced by the HMAC chain:
#   mac_0 = HMAC(key, topic_bytes || data_0)       # First chunk
#   mac_n = HMAC(key, mac_{n-1} || data_n)         # Subsequent chunks
#
# Topic is inside the capnp message for generic extraction.
struct StreamChunk {
  topic @0 :Text;     # DH-derived routing topic (for XPUB prefix matching)
  data @1 :Data;      # Serialized StreamPayload bytes
  hmac @2 :Data;      # Chained HMAC-SHA256 (32 bytes)
  prevHmac @3 :Data;  # Previous chunk's HMAC (empty for first chunk)
}

# Stream block - batched chunks with E2E authentication
#
# Wire format (ZMQ multipart):
#   Frame 0:      topic (64 hex chars, DH-derived)
#   Frame 1..N-1: capnp segments (this struct)
#   Frame N:      mac (16 bytes HMAC-SHA256 truncated)
#
# MAC chain:
#   Block 0: mac = HMAC(mac_key, topic_bytes || segments)[..16]
#   Block N: mac = HMAC(mac_key, prev_mac || segments)[..16]
#
# Client verifies MAC using DH-derived mac_key. StreamService forwards blindly.
struct StreamBlock {
  prevMac @0 :Data;              # topic[..16] for block 0, mac_{n-1} for block N
  payloads @1 :List(StreamPayload);
}

# =============================================================================
# Payload Types (inside StreamChunk.data or StreamBlock.payloads)
# =============================================================================

# Stream payload - the actual content being streamed
#
# This is what gets serialized into StreamChunk.data.
# Separates wire format (StreamChunk) from content (StreamPayload).
struct StreamPayload {
  streamId @0 :Text;    # Stream identifier for correlation

  union {
    token @1 :Text;               # Generated token text
    complete @2 :StreamStats;     # Stream completion with stats
    error @3 :StreamError;        # Error during generation
    heartbeat @4 :Void;           # Keep-alive (no data)
  }
}

# Stream completion statistics
struct StreamStats {
  tokensGenerated @0 :UInt32;
  finishReason @1 :Text;          # "stop", "length", "eos", "error"
  generationTimeMs @2 :UInt64;
  tokensPerSecond @3 :Float32;
  # Optional quality metrics
  perplexity @4 :Float32;
  avgEntropy @5 :Float32;
}

# Stream error information
struct StreamError {
  message @0 :Text;
  code @1 :Text;                  # "timeout", "cancelled", "internal", etc.
  details @2 :Text;               # Optional additional context
}

# =============================================================================
# Reconnection / Resume
# =============================================================================

# Stream resume request - for retransmission after disconnect
#
# Client sends the last HMAC it successfully verified. StreamService finds
# this HMAC in its buffer and resends all subsequent chunks. This provides
# zero-knowledge of sequence numbers - the HMAC chain is the ordering proof.
struct StreamResume {
  topic @0 :Text;           # Topic to resume
  resumeFromHmac @1 :Data;  # Last verified HMAC (server resends chunks after this)
}

