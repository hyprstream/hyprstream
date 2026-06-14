@0xe7f8a9b0c1d2e3f4;

using import "annotations.capnp".fixedSize;

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
# Stream QoS policy (#213) — schema-declared delivery/integrity contract.
#
# Applied to a streaming method's StreamInfo response variant; codegen (#216/#217/
# #218) realizes a *matched* producer + consumer from it, so the two ends can never
# silently disagree on QoS or integrity mode. This defines the *vocabulary* only;
# resolver + codegen land in #216+. Policy *values* are declared per-application —
# inline at the call site, or an app-local `const :StreamContract` — never standardized
# here until real call sites prove a grouping is generic.
#
# Each axis is a union-of-(void|group) so a strategy carries its own parameters.
#
# **`@0`-is-safe discipline:** capnp zero-fills an absent field, so the `@0` variant of
# each *security-bearing* axis is the strict/fail-closed choice (ordering=ordered → gap
# fatal; completion=terminal → require terminal). QoS axes (delivery/retention/
# backpressure) take the most-conservative-surface `@0` (atMostOnce / liveOnly / block).
# Codegen additionally *requires* every axis to be set for a compile-time policy, so this
# ordering is load-bearing only for externally-sourced / advertised policy.
# =============================================================================

# Ordering + (media) replay-window. ordered = strict (gap fatal, chained MAC).
struct Ordering {
  union {
    ordered @0 :Void;
    unordered :group {
      replayWindow @1 :UInt32;   # media: reject group seq <= (last-seen - window)
    }
  }
}

# Delivery guarantee. atLeastOnce carries dedup + resume params.
struct Delivery {
  union {
    atMostOnce @0 :Void;
    atLeastOnce :group {
      dedupWindow @1 :UInt32;    # # of recent group seqs remembered for client dedup
      resumable   @2 :Bool;      # offset-resume from last-acked seq across reconnect
    }
  }
}

# Truncation policy — its OWN axis (terminal frames are opt-in; inference uses none).
# terminal = require a Complete/Error payload before EOF (EOF-without-terminal = reject).
struct Completion {
  union {
    terminal @0 :Void;
    none     @1 :Void;
  }
}

# Relay-side retention window (late-join / resume). liveOnly = smallest surface (#174).
struct Retention {
  union {
    liveOnly @0 :Void;
    groups   @1 :UInt32;
    seconds  @2 :UInt32;
  }
}

# Backpressure when the publish path saturates. block = lossless (EAGAIN contract).
struct Backpressure {
  union {
    block @0 :Void;
    dropOldest :group {
      highWater @1 :UInt32;
    }
  }
}

# The full per-stream delivery contract. Composed of the axes above; one per stream.
struct StreamContract {
  ordering     @0 :Ordering;
  delivery     @1 :Delivery;
  completion   @2 :Completion;
  retention    @3 :Retention;
  backpressure @4 :Backpressure;
}

# Apply to a streaming method's StreamInfo response variant:
#   streamInfo @0 :StreamInfo $streamPolicy(.tokenStream);   # app-local const, or inline a literal
# Naming: annotation `streamPolicy` (noun-form, like `mcpScope`/`fixedSize`), value type
# `StreamContract` — a distinct identifier per the `mcpScope :ScopeAction` precedent (which
# capnpc-rust requires: a struct and annotation that snake_case to the same module collide).
annotation streamPolicy(field) :StreamContract;

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
  dhPublic @2 :Data $fixedSize(32);  # Server's ephemeral Ristretto255 public key for DH
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

# =============================================================================
# Wire Format (PUSH/PULL -> XPUB/SUB)
# =============================================================================

# Stream block - batched payloads with E2E authentication
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
# Payload Types (inside StreamBlock.payloads)
# =============================================================================

# Stream payload - the actual content being streamed
#
# This is what gets serialized into StreamBlock.payloads.
# Separates wire format (StreamBlock) from content (StreamPayload).
#
# Generic design: Applications interpret the binary data as needed:
#   - Inference: UTF-8 text tokens
#   - Worker I/O: Arbitrary binary (stdout/stderr/stdin)
#
# Note: Stream identity comes from the DH-derived topic, not from payload fields.
# The topic cryptographically binds the stream to the DH key exchange.
struct StreamPayload {
  union {
    data @0 :Data;                # Generic binary payload (tokens, I/O, etc.)
    complete @1 :Data;            # App-specific completion metadata (serialized)
    error @2 :StreamError;        # Error during streaming
    heartbeat @3 :Void;           # Keep-alive (no data)
    tagged @4 :TaggedPayload;     # Encrypted tagged payload with key commitment
  }
}

# Tagged payload with authenticated encryption metadata
struct TaggedPayload {
  tag @0 :Data;
  payload @1 :Data;
  nonce @2 :Data;
  keyCommitment @3 :Data;
}

# =============================================================================
# Completion Metadata
# =============================================================================
#
# StreamPayload.complete contains app-specific completion metadata as raw bytes.
# Applications serialize their own types into this field:
#
#   - Inference: Serialize InferenceComplete (see inference.capnp)
#   - Worker I/O: Empty or app-specific status
#
# Example for inference (in inference.capnp):
#   struct InferenceComplete {
#     stats @0 :StreamStats;
#   }
#
# The generic Data field allows each domain to define completion semantics.

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

# =============================================================================
# Control Channel (Consumer -> Producer)
# =============================================================================

# Control message sent on the DH-derived control channel.
#
# Wire format (ZMQ multipart):
#   Frame 0: ctrl_topic (64 hex chars, DH-derived)
#   Frame 1: capnp segments (this struct)
#   Frame 2: mac (16 bytes HMAC-SHA256 truncated, using ctrl_mac_key)
struct StreamControl {
  union {
    cancel @0 :Void;        # Request stream cancellation
    ping @1 :Void;          # Keep-alive probe (reserved)
  }
}

