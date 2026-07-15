//! Post-quantum hybrid TLS crypto-provider policies (#557 / S6 of epic #550).
//!
//! The owned internal mesh (zmtp and both iroh ALPNs) is hybrid-only: it has no
//! classical key-exchange fallback. The process-wide provider used by the
//! external WebTransport/HTTP perimeter deliberately retains X25519 fallback
//! for browser and third-party interoperability. Application-layer HyKEM is the
//! primary, transport-independent confidentiality guarantee on every path.

use std::sync::Arc;

use rustls::crypto::{aws_lc_rs, CryptoProvider};

#[cfg(test)]
const INTERNAL_MESH_GROUPS: &[rustls::NamedGroup] = &[rustls::NamedGroup::X25519MLKEM768];
const EXTERNAL_INTEROP_GROUPS: &[rustls::NamedGroup] = &[
    rustls::NamedGroup::X25519MLKEM768,
    rustls::NamedGroup::X25519,
];

/// The effective process-wide rustls provider does not match hyprstream's
/// declared external-interoperability policy.
#[derive(Debug)]
pub struct ProviderPolicyError {
    actual: Vec<rustls::NamedGroup>,
}

impl std::fmt::Display for ProviderPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "rustls process-default crypto policy mismatch: expected \
             [X25519MLKEM768, X25519], got {:?}; another provider was installed first",
            self.actual
        )
    }
}

impl std::error::Error for ProviderPolicyError {}

fn provider_with_groups(
    groups: impl IntoIterator<Item = &'static dyn rustls::crypto::SupportedKxGroup>,
) -> CryptoProvider {
    CryptoProvider {
        kx_groups: groups.into_iter().collect(),
        ..aws_lc_rs::default_provider()
    }
}

/// Provider for an owned internal-mesh connection. Classical fallback is
/// intentionally absent, so an X25519-only peer fails the TLS handshake.
pub fn internal_mesh_crypto_provider() -> Arc<CryptoProvider> {
    Arc::new(provider_with_groups([aws_lc_rs::kx_group::X25519MLKEM768]))
}

/// Provider for an explicitly declared external-interoperability boundary.
/// PQ hybrid is preferred, with classical X25519 retained for non-PQ peers.
pub fn pq_crypto_provider() -> Arc<CryptoProvider> {
    Arc::new(provider_with_groups([
        aws_lc_rs::kx_group::X25519MLKEM768,
        aws_lc_rs::kx_group::X25519,
    ]))
}

fn group_names(provider: &CryptoProvider) -> Vec<rustls::NamedGroup> {
    provider
        .kx_groups
        .iter()
        .map(|group| group.name())
        .collect()
}

/// Install and validate the external-interoperability provider as rustls's
/// process-wide default.
///
/// rustls is first-install-wins. Consequently, silently ignoring
/// `install_default` failure can leave a process on ring/X25519. This function
/// always inspects the effective provider after installation and returns an
/// error if an
/// earlier initializer installed anything other than the exact declared
/// external policy. Call it before constructing any WebTransport/rustls
/// builder; a policy violation fails startup instead of downgrading transport.
pub fn install_pq_crypto_provider() -> Result<(), ProviderPolicyError> {
    let _ = provider_with_groups([
        aws_lc_rs::kx_group::X25519MLKEM768,
        aws_lc_rs::kx_group::X25519,
    ])
    .install_default();

    let actual =
        CryptoProvider::get_default().map_or_else(Vec::new, |provider| group_names(provider));
    if actual != EXTERNAL_INTEROP_GROUPS {
        return Err(ProviderPolicyError { actual });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn external_interop_is_pq_first_with_declared_x25519_fallback() {
        assert_eq!(group_names(&pq_crypto_provider()), EXTERNAL_INTEROP_GROUPS);
    }

    #[test]
    fn internal_mesh_is_hybrid_only() {
        assert_eq!(
            group_names(&internal_mesh_crypto_provider()),
            INTERNAL_MESH_GROUPS
        );
    }

    #[test]
    fn external_interop_live_handshake_allows_classical_peer() {
        let certified = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])
            .expect("generate test certificate");
        let cert = certified.cert.der().clone();
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(certified.key_pair.serialize_der());
        let server_config = rustls::ServerConfig::builder_with_provider(pq_crypto_provider())
            .with_safe_default_protocol_versions()
            .expect("external provider supports TLS 1.3")
            .with_no_client_auth()
            .with_single_cert(vec![cert.clone()], key.into())
            .expect("valid test certificate");

        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert).expect("add test root");
        let client_config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("ring supports TLS 1.3")
        .with_root_certificates(roots)
        .with_no_client_auth();

        let mut server = rustls::ServerConnection::new(Arc::new(server_config))
            .expect("build server connection");
        let mut client = rustls::ClientConnection::new(
            Arc::new(client_config),
            "localhost".try_into().expect("valid server name"),
        )
        .expect("build client connection");

        for _ in 0..8 {
            let mut client_records = Vec::new();
            client
                .write_tls(&mut client_records)
                .expect("write client TLS records");
            if !client_records.is_empty() {
                server
                    .read_tls(&mut std::io::Cursor::new(client_records))
                    .expect("read client TLS records");
                server
                    .process_new_packets()
                    .expect("process client TLS records");
            }

            let mut server_records = Vec::new();
            server
                .write_tls(&mut server_records)
                .expect("write server TLS records");
            if !server_records.is_empty() {
                client
                    .read_tls(&mut std::io::Cursor::new(server_records))
                    .expect("read server TLS records");
                client
                    .process_new_packets()
                    .expect("process server TLS records");
            }
            if !client.is_handshaking() && !server.is_handshaking() {
                break;
            }
        }
        assert!(!client.is_handshaking() && !server.is_handshaking());
        assert_eq!(
            server
                .negotiated_key_exchange_group()
                .expect("completed handshake has a group")
                .name(),
            rustls::NamedGroup::X25519,
            "classical fallback is allowed only by the external interop policy"
        );
    }

    // This mutation must run in a fresh process because rustls's default is a
    // OnceLock. The ignored child test is invoked explicitly by its parent.
    #[test]
    #[ignore = "subprocess mutation fixture"]
    fn non_pq_provider_installed_first_child() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("fresh child process must not already have a provider");
        install_pq_crypto_provider().expect("non-PQ provider must fail closed");
    }

    #[test]
    fn non_pq_provider_installed_first_fails_closed() {
        let exe = std::env::current_exe().expect("test executable path");
        let output = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "transport::pq_provider::tests::non_pq_provider_installed_first_child",
            ])
            .output()
            .expect("run provider-ordering mutation child");
        assert!(
            !output.status.success(),
            "a ring provider installed first must make startup fail"
        );
    }
}
