//! Post-quantum hybrid TLS crypto provider (#557 / S6 of epic #550).
//!
//! Pins the TLS 1.3 key exchange to **X25519MLKEM768** (the hybrid PQ group of
//! draft-ietf-tls-ecdhe-mlkem / RFC 9370 framing) with classical **X25519** as a
//! fallback for interop with non-PQ peers. ML-KEM is only available via the
//! **aws-lc-rs** rustls provider — the `ring` provider this replaces has no
//! ML-KEM, which is why every `ring::default_provider().install_default()` site
//! is swapped for [`install_pq_crypto_provider`].
//!
//! This is transport-layer **defense-in-depth**. The real, transport-independent
//! confidentiality guarantee is the application-layer hybrid KEM (`HyKEM`,
//! [`crate::crypto::hybrid_kem`], S0) which also covers the wasm/browser path
//! where we cannot touch the WebTransport TLS handshake. Here we harden the QUIC
//! channel on the paths we own (zmtp, iroh, quinn/WebTransport).
//!
//! Native-only: rustls/aws-lc-rs are not part of the wasm32 build.

use std::sync::Arc;

use rustls::crypto::{aws_lc_rs, CryptoProvider};

/// Build the pinned PQ-hybrid crypto provider: aws-lc-rs with
/// `kx_groups = [X25519MLKEM768, X25519]` (hybrid first, classical fallback). All
/// other parameters are aws-lc-rs defaults (cipher suites, signature schemes).
///
/// Pinning `kx_groups` explicitly (rather than relying on the `prefer-post-quantum`
/// default order) makes the policy deterministic and auditable: a PQ-capable peer
/// negotiates X25519MLKEM768; a classical-only peer falls back to X25519 (the
/// channel is then classical, but the app-layer HyKEM stays hybrid).
fn build_provider() -> CryptoProvider {
    CryptoProvider {
        kx_groups: vec![
            aws_lc_rs::kx_group::X25519MLKEM768,
            aws_lc_rs::kx_group::X25519,
        ],
        ..aws_lc_rs::default_provider()
    }
}

pub fn pq_crypto_provider() -> Arc<CryptoProvider> {
    Arc::new(build_provider())
}

/// Install [`pq_crypto_provider`] as the process-wide rustls default.
///
/// Idempotent — the first install in the process wins and later calls are a
/// no-op (`install_default` returns `Err` once set; we ignore it). Must run
/// before any rustls `ServerConfig`/`ClientConfig::builder()`, quinn endpoint, or
/// web-transport endpoint that resolves the process default
/// (`CryptoProvider::get_default()`). This is the single replacement for the
/// former `rustls::crypto::ring::default_provider().install_default()` calls; it
/// also covers the WebTransport (`web-transport-quinn`) path, whose builder has
/// no provider setter and resolves the process default.
pub fn install_pq_crypto_provider() {
    // `install_default` consumes an owned `CryptoProvider` (not an `Arc`); the
    // first install in the process wins and returns `Ok`, later calls return
    // `Err(already_set)` which we ignore.
    let _ = build_provider().install_default();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PQ-hybrid group MUST be offered first, with classical X25519 as the
    /// only fallback — a regression here would silently drop transport PQ.
    #[test]
    fn pins_x25519mlkem768_first_then_x25519() {
        let provider = build_provider();
        let names: Vec<rustls::NamedGroup> = provider.kx_groups.iter().map(|g| g.name()).collect();
        assert_eq!(
            names.first().copied(),
            Some(rustls::NamedGroup::X25519MLKEM768),
            "X25519MLKEM768 must be the first (preferred) kx group"
        );
        let pq = names
            .iter()
            .position(|n| *n == rustls::NamedGroup::X25519MLKEM768);
        let classical = names.iter().position(|n| *n == rustls::NamedGroup::X25519);
        assert!(
            matches!((pq, classical), (Some(p), Some(c)) if p < c),
            "PQ-hybrid must be offered ahead of classical X25519, got {names:?}"
        );
    }
}
