//! DID-document transport `service`-entry codec.
//!
//! The canonical encoder/decoder for the typed transport entries a node
//! publishes in its DID document, shared by the producer (the did:web document
//! builder) and the consumer (the DID resolver that feeds [`crate::dial::dial`]).
//!
//! # Layering (settled design)
//!
//! Peer **identity** is established at the application layer — the response
//! `SignedEnvelope` is verified against the node's published keys (the `#mesh`
//! verification method / JWKS). The transport entries here carry **reach info +
//! channel auth**, not the identity root. For iroh the `nodeId` *is* the
//! Ed25519 identity key (so [`DecodedEntry::identity_key`] is populated and the
//! resolver can bind it to `#mesh`); for QUIC, identity comes from `#mesh`
//! separately and the cert pin is channel-only (#185).
//!
//! # Wire shape
//!
//! `serviceEndpoint` is a DIDComm-style map (W3C DID Core §5.4 permits a map).
//! Binary values use multibase (`z` = base58btc) over a multiformats prefix,
//! matching the doc's `Multikey` `publicKeyMultibase` encoding (#280; ed25519
//! VMs were formerly `Ed25519VerificationKey2020`):
//!
//! ```jsonc
//! // type: "IrohTransport"
//! { "nodeId": "z<multicodec ed25519-pub || key>", "relays": ["https://…"], "accept": ["hyprstream-rpc/1","moql"] }
//! // type: "QuicTransport"
//! { "uri": "https://host:port", "webpki": false,
//!   "certHashes": ["z<multihash sha2-256 || digest>"], "accept": [...] }
//! ```
//!
//! `certHashes` is a multibase-encoded **multihash** (self-describing algorithm,
//! `sha2-256` = 0x12), a **set** so cert rotation can overlap. Whole-cert pins
//! are the browser-compatible projection of the identity key — see #185 / #200.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::transport::{QuicServerAuth, TransportConfig};

/// multicodec `ed25519-pub` varint prefix (did:key spec).
const MULTICODEC_ED25519_PUB: [u8; 2] = [0xed, 0x01];
/// multihash prefix for `sha2-256` with a 32-byte digest (`0x12` code, `0x20` len).
const MULTIHASH_SHA2_256: [u8; 2] = [0x12, 0x20];

/// A decoded transport entry: a dialable config plus, for identity-bearing
/// transports, the peer's identity public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedEntry {
    /// Dial target for [`crate::dial::dial`].
    pub config: TransportConfig,
    /// The peer's Ed25519 identity key, when the transport binds to it
    /// (iroh: the `nodeId`). `None` for QUIC, where identity comes from the
    /// `#mesh` verification method separately.
    pub identity_key: Option<[u8; 32]>,
}

// ── multibase helpers ────────────────────────────────────────────────────────

fn multibase_encode(prefix: &[u8], payload: &[u8]) -> String {
    let mut buf = Vec::with_capacity(prefix.len() + payload.len());
    buf.extend_from_slice(prefix);
    buf.extend_from_slice(payload);
    format!("z{}", bs58::encode(buf).into_string())
}

/// Decode a multibase base58btc string carrying `prefix || <32-byte payload>`.
fn multibase_decode_32(s: &str, prefix: [u8; 2], what: &str) -> Result<[u8; 32]> {
    let b58 = s
        .strip_prefix('z')
        .ok_or_else(|| anyhow!("{what}: expected multibase base58btc ('z' prefix)"))?;
    let bytes = bs58::decode(b58)
        .into_vec()
        .map_err(|e| anyhow!("{what}: invalid base58: {e}"))?;
    if bytes.len() != prefix.len() + 32 {
        bail!("{what}: expected {} bytes, got {}", prefix.len() + 32, bytes.len());
    }
    if bytes[..2] != prefix[..] {
        bail!("{what}: wrong multiformats prefix");
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes[2..]);
    Ok(out)
}

// ── encode (producer side) ───────────────────────────────────────────────────

/// Encode an `IrohTransport` `serviceEndpoint` map for a peer identified by
/// `node_id` (its Ed25519 `EndpointId`), reachable via `relays`.
pub fn encode_iroh(node_id: &[u8; 32], relays: &[String], accept: &[&str]) -> Value {
    json!({
        "nodeId": multibase_encode(&MULTICODEC_ED25519_PUB, node_id),
        "relays": relays,
        "accept": accept,
    })
}

/// Encode a `QuicTransport` `serviceEndpoint` map.
pub fn encode_quic(uri: &str, auth: &QuicServerAuth, accept: &[&str]) -> Value {
    let cert_hashes: Vec<String> = auth
        .accept_cert_hashes()
        .iter()
        .map(|h| multibase_encode(&MULTIHASH_SHA2_256, h))
        .collect();
    json!({
        "uri": uri,
        "webpki": auth.require_web_pki(),
        "certHashes": cert_hashes,
        "accept": accept,
    })
}

// ── decode (consumer / resolver side) ────────────────────────────────────────

/// Decode one DID-doc transport `service` entry (`{type, serviceEndpoint}`) into
/// a dialable [`DecodedEntry`]. Unknown `type`s error so the caller can skip
/// them.
pub fn decode_service_entry(entry: &Value) -> Result<DecodedEntry> {
    let ty = entry
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("service entry missing string `type`"))?;
    let svc = entry
        .get("serviceEndpoint")
        .ok_or_else(|| anyhow!("service entry missing `serviceEndpoint`"))?;
    match ty {
        "IrohTransport" => decode_iroh(svc),
        "QuicTransport" => decode_quic(svc),
        other => bail!("unsupported transport service type: {other}"),
    }
}

fn decode_iroh(svc: &Value) -> Result<DecodedEntry> {
    let node_id = multibase_decode_32(
        svc.get("nodeId").and_then(Value::as_str).ok_or_else(|| anyhow!("iroh: missing `nodeId`"))?,
        MULTICODEC_ED25519_PUB,
        "iroh nodeId",
    )?;
    let relays: Vec<String> = svc
        .get("relays")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    // Direct addresses are not published (privacy + iroh discovery); the resolver
    // supplies a relay and lets iroh discover direct paths.
    let relay_url = relays.into_iter().next();
    Ok(DecodedEntry {
        config: TransportConfig::iroh(node_id, Vec::new(), relay_url),
        identity_key: Some(node_id),
    })
}

fn decode_quic(svc: &Value) -> Result<DecodedEntry> {
    let uri = svc.get("uri").and_then(Value::as_str).ok_or_else(|| anyhow!("quic: missing `uri`"))?;
    let (host, port) = parse_https_authority(uri)?;
    // Default `webpki` to true (safe-by-default: a CA-fronted public peer).
    let require_web_pki = svc.get("webpki").and_then(Value::as_bool).unwrap_or(true);
    let cert_hashes: Vec<[u8; 32]> = svc
        .get("certHashes")
        .and_then(Value::as_array)
        .map(|a| -> Result<Vec<[u8; 32]>> {
            a.iter()
                .map(|v| {
                    let s = v.as_str().ok_or_else(|| anyhow!("quic certHashes: non-string entry"))?;
                    multibase_decode_32(s, MULTIHASH_SHA2_256, "quic certHash")
                })
                .collect()
        })
        .transpose()?
        .unwrap_or_default();

    let auth = match (require_web_pki, cert_hashes.is_empty()) {
        (true, true) => QuicServerAuth::web_pki(),
        (false, false) => QuicServerAuth::pinned(cert_hashes)?,
        (true, false) => QuicServerAuth::web_pki_pinned(cert_hashes)?,
        (false, true) => bail!("quic entry has no auth (webpki=false and no certHashes)"),
    };

    // Pinned dials by IP; WebPKI dials by `server_name` (DNS), ignoring the IP —
    // so a hostname `uri` gets an unspecified placeholder address with the real
    // port, and the host becomes the validation name.
    let addr: SocketAddr = match host.parse::<IpAddr>() {
        Ok(ip) => SocketAddr::new(ip, port),
        Err(_) => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port),
    };
    Ok(DecodedEntry {
        config: TransportConfig::quic_with_auth(addr, host, auth),
        identity_key: None,
    })
}

/// Parse `https://host:port` (or `quic://…`) into `(host, port)`. Defaults to
/// 443 when no port is given.
fn parse_https_authority(uri: &str) -> Result<(String, u16)> {
    let url = url::Url::parse(uri).with_context(|| format!("quic uri parse: {uri}"))?;
    let host = url.host_str().ok_or_else(|| anyhow!("quic uri has no host: {uri}"))?.to_owned();
    let port = url.port().unwrap_or(443);
    Ok((host, port))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::EndpointType;

    #[test]
    fn iroh_round_trip() {
        let node_id = [7u8; 32];
        let relays = vec!["https://relay.example".to_owned()];
        let entry = json!({
            "id": "did:web:ex#iroh",
            "type": "IrohTransport",
            "serviceEndpoint": encode_iroh(&node_id, &relays, &["hyprstream-rpc/1", "moql"]),
        });
        let decoded = decode_service_entry(&entry).unwrap();
        assert_eq!(decoded.identity_key, Some(node_id));
        match &decoded.config.endpoint {
            EndpointType::Iroh { node_id: n, relay_url, direct_addrs } => {
                assert_eq!(*n, node_id);
                assert_eq!(relay_url.as_deref(), Some("https://relay.example"));
                assert!(direct_addrs.is_empty());
            }
            other => panic!("expected Iroh, got {other:?}"),
        }
    }

    #[test]
    fn quic_pinned_round_trip() {
        let auth = QuicServerAuth::pinned(vec![[1u8; 32], [2u8; 32]]).unwrap();
        let entry = json!({
            "type": "QuicTransport",
            "serviceEndpoint": encode_quic("https://10.0.0.1:4433", &auth, &["hyprstream-rpc/1"]),
        });
        let decoded = decode_service_entry(&entry).unwrap();
        assert_eq!(decoded.identity_key, None);
        match &decoded.config.endpoint {
            EndpointType::Quic { addr, server_name, auth: a } => {
                assert_eq!(addr.to_string(), "10.0.0.1:4433");
                assert_eq!(server_name, "10.0.0.1");
                assert!(!a.require_web_pki());
                assert_eq!(a.accept_cert_hashes(), &[[1u8; 32], [2u8; 32]]);
            }
            other => panic!("expected Quic, got {other:?}"),
        }
    }

    #[test]
    fn quic_webpki_default_and_hostname_placeholder() {
        // No `webpki` field ⇒ default true; no certHashes ⇒ WebPKI only.
        let entry = json!({
            "type": "QuicTransport",
            "serviceEndpoint": { "uri": "https://host.example:8443", "accept": ["hyprstream-rpc/1"] },
        });
        let decoded = decode_service_entry(&entry).unwrap();
        match &decoded.config.endpoint {
            EndpointType::Quic { addr, server_name, auth } => {
                assert!(auth.require_web_pki());
                assert!(auth.accept_cert_hashes().is_empty());
                assert_eq!(server_name, "host.example");
                assert_eq!(addr.port(), 8443);
                assert!(addr.ip().is_unspecified(), "hostname ⇒ placeholder IP");
            }
            other => panic!("expected Quic, got {other:?}"),
        }
    }

    #[test]
    fn quic_web_pki_pinned() {
        let entry = json!({
            "type": "QuicTransport",
            "serviceEndpoint": {
                "uri": "https://host.example:4433",
                "webpki": true,
                "certHashes": [multibase_encode(&MULTIHASH_SHA2_256, &[9u8; 32])],
            },
        });
        let decoded = decode_service_entry(&entry).unwrap();
        if let EndpointType::Quic { auth, .. } = &decoded.config.endpoint {
            assert!(auth.require_web_pki());
            assert_eq!(auth.accept_cert_hashes(), &[[9u8; 32]]);
        } else {
            panic!("expected Quic");
        }
    }

    #[test]
    fn rejects_no_auth_and_unknown_type() {
        // webpki:false + no certHashes ⇒ no auth ⇒ reject.
        let bad = json!({
            "type": "QuicTransport",
            "serviceEndpoint": { "uri": "https://10.0.0.1:4433", "webpki": false },
        });
        assert!(decode_service_entry(&bad).is_err());

        let unknown = json!({ "type": "OnionTransport", "serviceEndpoint": {} });
        assert!(decode_service_entry(&unknown).is_err());
    }

    #[test]
    fn rejects_malformed_multibase() {
        // Wrong multicodec prefix (a sha2-256 multihash where an ed25519 key is expected).
        let entry = json!({
            "type": "IrohTransport",
            "serviceEndpoint": { "nodeId": multibase_encode(&MULTIHASH_SHA2_256, &[0u8; 32]), "relays": [] },
        });
        assert!(decode_service_entry(&entry).is_err());
    }
}
