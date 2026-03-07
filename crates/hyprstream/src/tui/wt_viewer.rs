//! WebTransport utilities for TUI viewers.
//!
//! Provides TLS certificate generation for WebTransport connections.
//! RPC is handled by `handle_wt_stream()` in zmtp_quic.rs.
//! Frame subscription is handled by the WASM ZMQ bridge (SUB over XPUB).
//!
//! Native CLI viewers connect via ZMQ directly (no WebTransport needed).

/// Generate a short-lived TLS certificate for TUI WebTransport.
///
/// Uses ECDSA P-256 with ≤14 day validity per WebTransport spec.
/// Returns (DER certificate, DER private key, SHA-256 hash).
pub fn generate_wt_cert(validity_days: u32) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    use rcgen::{CertificateParams, KeyPair, PKCS_ECDSA_P256_SHA256};
    use sha2::{Sha256, Digest};

    let validity_days = validity_days.min(14); // WebTransport max

    let key_pair = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)?;
    let mut params = CertificateParams::new(vec!["localhost".to_owned()])?;

    let now = time::OffsetDateTime::now_utc();
    params.not_before = now;
    params.not_after = now + time::Duration::days(validity_days as i64);

    let cert = params.self_signed(&key_pair)?;

    let cert_der = cert.der().to_vec();
    let key_der = key_pair.serialize_der();

    // SHA-256 hash for serverCertificateHashes
    let hash = Sha256::digest(&cert_der).to_vec();

    Ok((cert_der, key_der, hash))
}
