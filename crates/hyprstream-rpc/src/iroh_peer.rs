//! Browser (wasm32) iroh peer identity and pkarr helpers.
//!
//! Phase 2 adds iroh as a first-class wasm32 dependency, enabling:
//!
//! 1. **Browser iroh identity** — `BrowserIrohPeer` generates a fresh Ed25519
//!    SecretKey and exposes the derived NodeId / EndpointId, which equals the
//!    32-byte pubkey. The NodeId IS the iroh peer identity.
//!
//! 2. **did:key ↔ NodeId helpers** — bidirectional conversion between iroh's
//!    `EndpointId` (32-byte ed25519 pubkey) and the W3C `did:key` DID that
//!    atproto and hyprstream use to identify peers.
//!
//! 3. **Pkarr lookup** — `resolve_pkarr` wraps `PkarrRelayClient` to fetch a
//!    peer's relay URL from the N0 pkarr relay without a full iroh endpoint bind.
//!    This replaces the manual HTTP fetch + DNS-wire parse in `atproto.ts`.
//!
//! # What's NOT in Phase 2
//!
//! **Full iroh Endpoint bind**: creating a listening `iroh::Endpoint` (own relay
//! address, pkarr publisher, relay keep-alive) is expensive and deferred. It's
//! needed for dial_iroh_reach() (Phase 3) but not for identity + resolution.
//!
//! **moq-net wasm32**: moq-net 0.1.8 has `Session + Sync` bounds incompatible
//! with browser `!Sync` types. Upstream fix needed before moq subscribe/publish
//! can be wired into the browser via the same wasm seam.
//!
//! **dial_iroh_reach()**: connecting to a hyprstream server by iroh NodeId
//! requires a new RPC transport over `iroh::Connection` bidi streams. Deferred;
//! current browser RPC still uses ZMTP/WebTransport (`dial_wasm::dial`).

#![cfg(target_arch = "wasm32")]

use anyhow::{anyhow, Result};
use iroh::{EndpointId, SecretKey};

// ============================================================================
// BrowserIrohPeer — ephemeral iroh identity without a full Endpoint bind
// ============================================================================

/// A browser iroh peer identity.
///
/// Generates a fresh Ed25519 `SecretKey` and derives the corresponding
/// `EndpointId` (NodeId). This provides the browser with a stable iroh
/// identity for the session without the cost of binding a full
/// `iroh::Endpoint` (relay connection, pkarr publisher, background tasks).
///
/// The SecretKey is held in memory only; it is not persisted. Each call to
/// `new()` produces a fresh identity.
pub struct BrowserIrohPeer {
    secret_key: SecretKey,
}

impl BrowserIrohPeer {
    /// Generate a fresh ephemeral iroh peer identity.
    pub fn new() -> Self {
        Self {
            secret_key: SecretKey::generate(),
        }
    }

    /// The iroh `EndpointId` (NodeId) for this peer.
    ///
    /// The EndpointId is the Ed25519 public key corresponding to `secret_key`.
    /// It is stable for the lifetime of this struct.
    pub fn endpoint_id(&self) -> EndpointId {
        self.secret_key.public()
    }

    /// The NodeId as a 32-byte array (raw Ed25519 pubkey).
    pub fn node_id_bytes(&self) -> [u8; 32] {
        *self.secret_key.public().as_bytes()
    }

    /// Encode the NodeId as z-base32 (Zooko's base32).
    ///
    /// Used in pkarr gateway URLs: `https://dns.iroh.link/pkarr/<z32>`.
    pub fn node_id_z32(&self) -> String {
        self.endpoint_id().to_z32()
    }

    /// Express this peer's NodeId as a W3C `did:key`.
    ///
    /// The did:key format encodes Ed25519 keys as base58btc multibase with the
    /// `0xed 0x01` multicodec prefix.
    pub fn did_key(&self) -> String {
        did_key_from_node_id(&self.node_id_bytes())
    }
}

impl Default for BrowserIrohPeer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// did:key ↔ NodeId conversions
// ============================================================================

/// Convert a 32-byte iroh NodeId to a `did:key:z6Mk...` DID.
///
/// Encodes the pubkey with multicodec prefix `0xed 0x01` (ed25519-pub) in
/// base58btc, then prepends `did:key:z`.
pub fn did_key_from_node_id(node_id_bytes: &[u8; 32]) -> String {
    let mut payload = Vec::with_capacity(34);
    payload.push(0xed); // multicodec ed25519-pub high byte (varint)
    payload.push(0x01); // multicodec ed25519-pub low byte
    payload.extend_from_slice(node_id_bytes);
    let encoded = bs58::encode(&payload).into_string();
    format!("did:key:z{encoded}")
}

/// Extract the raw 32-byte NodeId from a `did:key:z6Mk...` DID.
///
/// The `did:key` spec encodes ed25519 keys as multibase base58btc (`z` prefix)
/// with the `0xed 0x01` multicodec prefix. The 32 bytes following the
/// multicodec prefix are the iroh `EndpointId`.
///
/// # Errors
///
/// Returns an error if the DID is not base58btc, not an ed25519 key, or
/// malformed.
pub fn node_id_from_did_key(did: &str) -> Result<[u8; 32]> {
    if !did.starts_with("did:key:z") {
        return Err(anyhow!(
            "did:key must use base58btc multibase (z-prefix), got: {did}"
        ));
    }
    let encoded = &did["did:key:z".len()..];
    let decoded = bs58::decode(encoded)
        .into_vec()
        .map_err(|e| anyhow!("base58btc decode failed: {e}"))?;

    if decoded.len() < 34 || decoded[0] != 0xed || decoded[1] != 0x01 {
        return Err(anyhow!(
            "did:key is not an ed25519 key (expected multicodec 0xed01): {did}"
        ));
    }
    decoded[2..34]
        .try_into()
        .map_err(|_| anyhow!("truncated did:key pubkey"))
}

// ============================================================================
// Pkarr relay lookup (N0 relay — same HTTP endpoint as atproto.ts fallback)
// ============================================================================

/// Resolve a peer's relay URL from the N0 pkarr relay via iroh's PkarrRelayClient.
///
/// Fetches `https://dns.iroh.link/pkarr/<z32-node-id>` using the iroh
/// `PkarrRelayClient`. The client verifies the Ed25519 signature on the
/// response and parses the `EndpointInfo` (relay URLs + IP addresses).
///
/// This is equivalent to the `atproto.ts` HTTP-fetch workaround but uses
/// iroh's verified parser instead of manual DNS-wire parsing.
///
/// # Returns
///
/// `Ok(Some(relay_url))` if the peer has published a relay URL.
/// `Ok(None)` if the peer has no record or no relay URL in the record.
/// `Err(_)` if the pkarr relay is unreachable or the response is invalid.
pub async fn resolve_pkarr_relay_url(node_id_bytes: &[u8; 32]) -> Result<Option<String>> {
    let node_id = EndpointId::from_bytes(node_id_bytes)
        .map_err(|e| anyhow!("invalid node_id bytes: {e:?}"))?;
    let pkarr_relay_url: url::Url = "https://dns.iroh.link/pkarr"
        .parse()
        .expect("N0 pkarr relay URL is valid");

    let client = iroh::address_lookup::PkarrRelayClient::new(pkarr_relay_url);

    let signed_packet = client
        .resolve(node_id)
        .await
        .map_err(|e| anyhow!("pkarr resolve failed: {e:?}"))?;

    // Parse the DNS wire packet into EndpointInfo and extract the first relay URL.
    let endpoint_info = iroh::endpoint_info::EndpointInfo::from_pkarr_signed_packet(&signed_packet)
        .map_err(|e| anyhow!("EndpointInfo parse failed: {e:?}"))?;

    let relay_url = endpoint_info
        .relay_urls()
        .next()
        .map(|url| url.to_string());

    Ok(relay_url)
}
