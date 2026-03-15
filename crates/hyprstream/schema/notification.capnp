@0xa1b2c3d4e5f6a7b8;

# Cap'n Proto schema for notification service
#
# Blind relay with broadcast encryption. NotificationService mediates
# pubkey exchange and routes encrypted capsules — never sees plaintext.
# Uses REQ/REP pattern with Continuation for async delivery fanout.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;

struct NotificationRequest {
  id @0 :UInt64;

  union {
    subscribe @1 :SubscribeRequest
      $mcpScope(subscribe) $mcpDescription("Subscribe to notifications matching a claim scope");

    publishIntent @2 :PublishIntentRequest
      $mcpScope(publish) $mcpDescription("Get subscriber pubkeys for scope-matched broadcast encryption");

    deliver @3 :DeliverRequest
      $mcpScope(publish) $mcpDescription("Deliver encrypted broadcast to scope-matched subscribers");

    unsubscribe @4 :UnsubscribeRequest
      $mcpScope(subscribe) $mcpDescription("Tear down a notification stream");

    listSubscriptions @5 :Void
      $mcpScope(query) $mcpDescription("List active subscriptions for current identity");

    ping @6 :Void
      $mcpScope(query) $mcpDescription("Health check");
  }
}

struct SubscribeRequest {
  scopePattern @0 :Text;       # Claim scope pattern: "serve:model:*"
  ephemeralPubkey @1 :Data;    # Client Ristretto255 pubkey (32 bytes)
  ttlSeconds @2 :UInt32;       # Subscription TTL (default 600, max 3600)
}

struct PublishIntentRequest {
  scope @0 :Text;              # Claim scope: "serve:model:qwen3"
  publisherPubkey @1 :Data;    # Publisher's ephemeral Ristretto255 pubkey (32 bytes)
}

struct DeliverRequest {
  intentId @0 :Text;
  capsules @1 :List(RecipientCapsule);
  encryptedPayload @2 :Data;   # AES-256-GCM ciphertext (shared across recipients)
  nonce @3 :Data;              # AES-GCM nonce (12 bytes)
}

struct RecipientCapsule {
  pubkeyFingerprint @0 :Data;  # Blake3(BLINDED_pubkey)[..16] for routing (128-bit)
  wrappedKey @1 :Data;         # AES-GCM(enc_key, data_key, aad=blinded_fingerprint)
  keyNonce @2 :Data;           # AES-GCM nonce for key wrapping (12 bytes, random OsRng)
  mac @3 :Data;                # One-shot MAC(mac_key, ciphertext) — 32 bytes
}

struct UnsubscribeRequest {
  subscriptionId @0 :Text;
}

struct NotificationResponse {
  requestId @0 :UInt64;

  union {
    error @1 :ErrorInfo;
    subscribeResult @2 :SubscribeResponse;
    publishIntentResult @3 :PublishIntentResponse;
    deliverResult @4 :DeliverResponse;
    unsubscribeResult @5 :Void;
    listSubscriptionsResult @6 :SubscriptionList;
    pingResult @7 :PingInfo;
  }
}

struct SubscribeResponse {
  subscriptionId @0 :Text;
  assignedTopic @1 :Text;      # XPUB topic (pre-registered with StreamService)
  streamEndpoint @2 :Text;     # StreamService XPUB endpoint to connect to
}

struct PublishIntentResponse {
  intentId @0 :Text;           # UUID, valid for 30s
  recipientPubkeys @1 :List(Data);  # BLINDED pubkeys: sub_pub + r_i * G
}

struct DeliverResponse {
  deliveredCount @0 :UInt32;
}

# Wire format for messages forwarded through StreamService.
# NS constructs this from DeliverRequest fields — subscriber parses it.
# Sent as StreamPayload::data inside a StreamBlock (via StreamPublisher API).
struct NotificationBlock {
  publisherPubkey @0 :Data;     # Publisher's ephemeral Ristretto pubkey (32 bytes)
  blindingScalar @1 :Data;      # r_i (32 bytes) — subscriber needs for blinding-aware DH
  wrappedKey @2 :Data;          # AES-GCM(enc_key, data_key, aad=fingerprint)
  keyNonce @3 :Data;            # AES-GCM nonce for key wrapping (12 bytes)
  encryptedPayload @4 :Data;    # AES-GCM ciphertext (shared across recipients)
  nonce @5 :Data;               # AES-GCM nonce for payload (12 bytes)
  intentId @6 :Text;            # For length-prefixed AAD reconstruction
  scope @7 :Text;               # For length-prefixed AAD reconstruction
  publisherMac @8 :Data;        # One-shot MAC(mac_key, ciphertext) — 32 bytes
}

struct SubscriptionInfo {
  subscriptionId @0 :Text;
  scopePattern @1 :Text;
  createdAt @2 :Int64;
  expiresAt @3 :Int64;
}

struct SubscriptionList {
  subscriptions @0 :List(SubscriptionInfo);
}

struct PingInfo {
  status @0 :Text;
  activeSubscriptions @1 :UInt32;
  totalDelivered @2 :UInt64;
}
