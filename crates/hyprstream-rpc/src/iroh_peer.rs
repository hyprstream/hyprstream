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
use iroh::address_lookup::PkarrError;
use iroh::{EndpointId, SecretKey};
use n0_error::StackError;

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
//
// The actual `did:key` (Ed25519) codec lives in the cross-target
// [`crate::did_key`] module — the SAME implementation the native `did_web`
// resolver uses (#475). An iroh NodeId IS a 32-byte Ed25519 public key, so the
// NodeId ⇄ did:key mapping is exactly the Ed25519 ⇄ did:key codec. Delegating
// here keeps the multicodec constant and the DID-URL fragment/query stripping in
// one place so the native and wasm32 paths can never drift.

/// Convert a 32-byte iroh NodeId to a `did:key:z6Mk...` DID.
///
/// Thin wasm32 alias for [`crate::did_key::ed25519_to_did_key`] (the NodeId is
/// the Ed25519 public key).
pub fn did_key_from_node_id(node_id_bytes: &[u8; 32]) -> String {
    crate::did_key::ed25519_to_did_key(node_id_bytes)
}

/// Extract the raw 32-byte NodeId from a `did:key:z6Mk...` DID.
///
/// Thin wasm32 alias for [`crate::did_key::did_key_to_ed25519`]. The 32 decoded
/// bytes are the iroh `EndpointId`. Unlike the previous local copy, this strips a
/// DID URL fragment / query (`did:key:z6Mk…#z6Mk…`) before decoding, matching the
/// native `did_web` behavior.
///
/// # Errors
///
/// Returns an error if the DID is not base58btc, not an ed25519 key, or
/// malformed.
pub fn node_id_from_did_key(did: &str) -> Result<[u8; 32]> {
    crate::did_key::did_key_to_ed25519(did)
}

// ============================================================================
// Pkarr relay lookup (N0 relay — same HTTP endpoint as atproto.ts fallback)
// ============================================================================

/// A relay URL sourced from a pkarr record — an **unverified reach hint**.
///
/// This is the typed output of [`resolve_pkarr_relay_url`]. It exists to make
/// the D3 liveness-only contract (#895) structural: a pkarr record is signed by
/// the peer's Ed25519 NodeId, so it is an integrity-protected *reach claim*
/// ("I am reachable at this relay"), but it carries **zero identity/trust
/// authority** — it says nothing about which capsule / `did:web` / `did:key`
/// owns that NodeId or what its PQ key material is.
///
/// **Forbidden use:** this value MUST NOT be treated as an identity authority
/// input. Authority for a `did:at9p` peer comes only from a GATE-verified
/// capsule (D1 / #893); for a raw `did:key` peer, from the channel-bound
/// `remote_id()`. The hint is a dial candidate only — feed it to the iroh dial
/// path as a relay address, never to admission / trust decisions.
///
/// The newtype has no conversion to any identity type by design; reach its URL
/// via [`PkarrReachHint::url`] when constructing a dial target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PkarrReachHint(String);

impl PkarrReachHint {
    /// Wrap a relay URL parsed from a pkarr record as an unverified reach hint.
    pub fn new(relay_url: String) -> Self {
        Self(relay_url)
    }

    /// The relay URL, for use as a dial-candidate address only.
    pub fn url(&self) -> &str {
        &self.0
    }
}

/// Resolve a peer's relay URL from the N0 pkarr relay via iroh's PkarrRelayClient.
///
/// Fetches `https://dns.iroh.link/pkarr/<z32-node-id>` using the iroh
/// `PkarrRelayClient`. The client verifies the Ed25519 signature on the
/// response and parses the `EndpointInfo` (relay URLs + IP addresses).
///
/// This is equivalent to the `atproto.ts` HTTP-fetch workaround but uses
/// iroh's verified parser instead of manual DNS-wire parsing.
///
/// **D3 (#895): the returned [`PkarrReachHint`] is an unverified reach hint,
/// not an identity authority source** — see its doc. pkarr rides the same
/// mainline DHT as the at9p locator, but only at9p (GATE-verified capsule) is
/// zero-trust.
///
/// # Returns
///
/// `Ok(Some(hint))` if the peer has published a relay URL.
/// `Ok(None)` if the peer has no record or no relay URL in the record.
/// `Err(_)` if the pkarr relay is unreachable or the response is invalid.
pub async fn resolve_pkarr_relay_url(node_id_bytes: &[u8; 32]) -> Result<Option<PkarrReachHint>> {
    let node_id = EndpointId::from_bytes(node_id_bytes)
        .map_err(|e| anyhow!("invalid node_id bytes: {e:?}"))?;
    let pkarr_relay_url: url::Url = "https://dns.iroh.link/pkarr"
        .parse()
        .expect("N0 pkarr relay URL is valid");

    let client = iroh::address_lookup::PkarrRelayClient::new(pkarr_relay_url);

    let signed_packet = match client.resolve(node_id).await {
        Ok(packet) => packet,
        // The pkarr relay returns HTTP 404 when the peer has never published a
        // record. That is a *negative answer*, not a failure — surface it as
        // `Ok(None)` so callers can distinguish "peer not found" from a genuine
        // network/transport error (which stays `Err`). Without this the 404
        // `PkarrError::HttpRequest` would bubble up as a thrown JsError and the
        // documented `Ok(None)` branch would be dead code.
        Err(e) if is_pkarr_not_found(&e) => return Ok(None),
        Err(e) => return Err(anyhow!("pkarr resolve failed: {e:?}")),
    };

    // Parse the DNS wire packet into EndpointInfo and extract the first relay URL.
    let endpoint_info = iroh::endpoint_info::EndpointInfo::from_pkarr_signed_packet(&signed_packet)
        .map_err(|e| anyhow!("EndpointInfo parse failed: {e:?}"))?;

    let relay_url = endpoint_info
        .relay_urls()
        .next()
        .map(|url| url.to_string());

    // Wrap as an unverified reach hint so the D3 contract (pkarr derives zero
    // authority) is enforced at the type — callers get a PkarrReachHint, never a
    // bare String that could be mistaken for authoritative reach.
    Ok(relay_url.map(PkarrReachHint::new))
}

/// Whether a pkarr resolve error is the "no record for this peer" case (HTTP
/// 404), as opposed to a genuine network/transport/verification failure.
///
/// `PkarrRelayClient::resolve` returns the type-erased
/// [`iroh::address_lookup::Error`], which wraps the underlying [`PkarrError`]
/// through n0-error's `AnyError`. We walk the [`StackError`] source chain and
/// downcast each link to [`PkarrError`], matching the `HttpRequest { status }`
/// variant carrying a `404 Not Found`. Any other status (or a `HttpSend` /
/// `Verify` / etc. variant) is a real error the caller should see.
fn is_pkarr_not_found(err: &iroh::address_lookup::Error) -> bool {
    err.stack().any(|source| {
        matches!(
            source.downcast_ref::<PkarrError>(),
            Some(PkarrError::HttpRequest { status, .. }) if status.as_u16() == 404
        )
    })
}
