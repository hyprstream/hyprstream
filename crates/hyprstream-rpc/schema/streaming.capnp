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
# StreamOpt (#213) — service-declared delivery/integrity contract.
#
# Returned in the signed StreamInfo handshake response so the contract is:
#   1. Authenticated: covered by the Hybrid (EdDSA + ML-DSA-65) COSE composite
#      signature on the ResponseEnvelope wrapping StreamInfo (#275). PQ-strength
#      and fail-closed: a Hybrid verifier rejects a classical-only or stripped
#      signature, so the "authenticated, fail-closed" guarantee holds at PQ
#      strength.
#   2. Wire-visible: encoded in the Cap'n Proto StreamInfo message body
#   3. Cross-language: capnp codegens native readers for Rust/TS/Python (#217/#218)
#   4. IETF-friendly: a struct in the handshake response, not a schema annotation
#
# The service asserts the contract it will enforce; clients MUST enforce the same
# options on ingress. A client that cannot honour the received options (e.g.,
# unordered delivery not yet implemented) MUST disconnect rather than silently
# downgrade. See `StreamVerifier::with_policy`.
#
# Named Rust presets (Job, Log, Pipe) live in hyprstream_rpc::stream_info
# rather than here — the schema vocabulary covers all axes; the preset types
# validate received values match the expected combination at compile time.
#
# **`@0`-is-safe discipline:** capnp zero-fills an absent/unrecognized union
# discriminant. The `@0` variant of every *security-bearing* axis is the
# strictest/fail-closed choice:
#   Ordering:      ordered   (gap = fatal)
#   Completion:    endOfStream (terminal frame required)
# QoS axes take the most-conservative-surface default:
#   Delivery:      atMostOnce
#   Retention:     live
#   OverflowPolicy: block
# Codegen MUST require every axis to be set for compile-time presets; this
# discipline is load-bearing only for externally-sourced (advertised) opts.
#
# Out of scope for StreamOpt:
#   - exactlyOnce delivery (MQTT QoS 2): incompatible with low-latency streaming
#   - causal / total ordering: use ordered + external coordination
#   - transport encryption: provided by QUIC (RFC 9001) + application HPKE
#   - message compression: Cap'n Proto packing or negotiated separately
#   - subscriber fan-out scope: defined by MoQ namespace/track model
# =============================================================================

# Ordering + anti-replay window.
# ordered = strict in-order delivery; gap detection is fatal (DTLS anti-replay,
# RFC 9147 §4.2.3). Fail-closed default (@0).
struct Ordering {
  union {
    ordered @0 :Void;
    unordered :group {
      # Anti-replay window for out-of-order / media delivery.
      # Reject block with sequenceNumber <= (highest-seen - antiReplayWindow).
      # Analogous to SRTP anti-replay window (RFC 3711 §3.3.2).
      antiReplayWindow @1 :UInt32;
    }
  }
}

# Delivery guarantee. Aligned with MQTT QoS 0/1 (OASIS MQTT 5.0 §4.3).
# exactlyOnce (QoS 2) is intentionally out of scope; use atLeastOnce with
# idempotent consumers.
struct Delivery {
  union {
    atMostOnce @0 :Void;
    atLeastOnce :group {
      # Number of recent sequenceNumbers remembered for client-side dedup.
      dedupWindow @1 :UInt32;
      # Resume delivery from last-acked sequenceNumber after reconnect.
      resumable   @2 :Bool;
    }
  }
}

# Stream termination contract. endOfStream = a StreamPayload.complete or
# StreamPayload.error frame MUST be received before EOF; EOF without one is
# a truncation attack and MUST be rejected. Fail-closed default (@0).
# Analogous to gRPC END_STREAM / HTTP/2 DATA+END_STREAM / WebTransport FIN.
struct Completion {
  union {
    endOfStream @0 :Void;  # terminal frame required before close
    none        @1 :Void;  # stream may close without explicit terminator
  }
}

# Relay-side retention / late-join buffer policy.
# live     = no buffering beyond the live delivery window (lossless relay surface).
# blocks   = retain the last N delivery blocks (~N MoQ Transport Groups).
# seconds  = retain for N wall-clock seconds.
# NOTE: live is declared, not enforced, by the relay (which is blind to MAC keys).
# Clients MUST enforce liveness by checking epoch: reject blocks from a prior DH
# session epoch on reconnect. See also: OverflowPolicy.
struct Retention {
  union {
    live    @0 :Void;      # no buffering beyond live window (fail-closed default)
    blocks  @1 :UInt32;   # retain N delivery blocks (~N MoQ Transport Groups)
    seconds @2 :UInt32;   # retain for N seconds
  }
}

# Application-layer overflow policy when the publish buffer saturates.
# block = lossless EAGAIN-style backpressure on the publisher side.
# dropOldest = ring-buffer semantics; highWaterMark is the buffer element cap.
# NOTE: this is an application-layer policy; QUIC flow control (RFC 9000 §4)
# operates independently at the transport layer.
struct OverflowPolicy {
  union {
    block @0 :Void;
    dropOldest :group {
      highWaterMark @1 :UInt32;
    }
  }
}

# The full per-stream QoS options. One instance per stream, carried in
# the signed StreamInfo handshake response.
struct StreamOpt {
  ordering       @0 :Ordering;
  delivery       @1 :Delivery;
  completion     @2 :Completion;
  retention      @3 :Retention;
  overflowPolicy @4 :OverflowPolicy;
}

# =============================================================================
# Stream Setup
# =============================================================================

# A single way to reach the moq streaming plane for this stream (#274).
#
# `role` distinguishes a direct producer endpoint from a relay; `transport`
# carries the network-routable dial parameters (no JSON-in-Text — native capnp
# so the generated codec covers it). The same-host UDS fast path is NOT encoded
# here: co-located clients resolve `ipc`/`inproc` from LOCAL config, never from
# the wire. A client picks a reach it can dial and subscribes to `broadcastPath`.
struct Destination {
  role @0 :Role;
  transport @1 :TransportConfig;
}

# Whether this reach is the producer itself (direct) or a relay in front of it.
enum Role {
  direct @0;
  relay  @1;
}

# Network-routable transport for a reach. A union so new transports (e.g. Iroh)
# can be added later without a wire break. Each arm is a struct so its fields —
# including `List(Data)` cert pins — are covered by the generated codec.
struct TransportConfig {
  union {
    # Quic / WebTransport (web-transport-quinn). Dialed over `web_transport_quinn`.
    quic @0 :QuicReach;
    # Reserved for NAT-traversing iroh dial (#274 follow-up). Capnp requires a
    # union to have >= 2 members; this placeholder leaves room without a wire
    # break when the iroh reach lands.
    iroh @1 :Void;
  }
}

# Dial parameters for the Quic / WebTransport arm.
struct QuicReach {
  addr       @0 :Text;        # "host:port" socket address to dial.
  serverName @1 :Text;        # TLS SNI / WebPKI validation name.
  certHashes @2 :List(Data);  # Acceptable leaf-cert SHA-256 pins (self-signed mesh).
}

# Stream metadata returned when starting a stream
#
# Contains everything the client needs to subscribe and derive keys.
# Returned by generateStream/inferStream RPC calls.
# Wrapped in a ResponseEnvelope whose Hybrid (EdDSA + ML-DSA-65) COSE composite
# signature authenticates dhPublic and the QoS contract at PQ strength (#275).
struct StreamInfo {
  streamId @0 :Text;      # Unique stream identifier (e.g., "stream-{uuid}")
  dhPublic @1 :Data $fixedSize(32);  # Server's ephemeral Ristretto255 public key for DH
  # Service-declared QoS options. Authenticated by the Hybrid COSE composite
  # signature over the enclosing ResponseEnvelope (#275). Clients MUST enforce
  # these options and MUST disconnect (not silently downgrade) if a required
  # mode is unsupported.
  qos   @2 :StreamOpt;
  # Broadcast path within the moq origin (e.g. "local/streams/{topic_hex}").
  broadcastPath @3 :Text;
  # Ways to reach the moq plane for this stream (#274). The client dials the
  # first reach it supports (see `dial_stream`) and subscribes to `broadcastPath`.
  # Co-located clients ignore this and use the same-host UDS fast path.
  announcedAt @4 :List(Destination);
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
  # Producer-assigned monotonic counter per epoch (#219). On the moq transport
  # this equals the moq Group id; it is the resume/dedup offset and the
  # anti-replay/ordering anchor. Authenticated implicitly — the MAC covers the
  # whole serialized StreamBlock. Consumer ordering/replay enforcement (gap-fatal
  # vs media per-Group) is qos-selected via StreamOpt (#163).
  # Named `sequenceNumber` per IETF convention (DTLS RFC 9147, RTP RFC 3550).
  sequenceNumber @2 :UInt64;
  # Key-epoch (#223): analogous to DTLS 1.3 epoch (RFC 9147 §4.2.3). Bumps on
  # re-key / producer restart so (epoch, sequenceNumber) is globally unique.
  # 0 until the epoch lifecycle lands; ordinal reserved so that change is wire-safe.
  epoch @3 :UInt64;
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
