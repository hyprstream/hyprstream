//! TLS-validated JWKS retrieval for the service-authentication path.

use hyprstream_rpc::auth::JwksFetcher;
use std::sync::Arc;

/// Fetch JWKS documents using reqwest's default TLS verifier.
///
/// The returned client trusts the platform CA store. Deployments with a private
/// PKI must add that CA to the platform trust store or use an explicitly
/// configured trust mechanism; this fetcher never accepts an invalid server
/// certificate.
pub fn default_jwks_fetcher() -> JwksFetcher {
    jwks_fetcher_with_client(reqwest::Client::new())
}

fn jwks_fetcher_with_client(client: reqwest::Client) -> JwksFetcher {
    Arc::new(move |url: String| {
        let client = client.clone();
        Box::pin(async move {
            let response = client.get(&url).send().await?.error_for_status()?;
            Ok(response.json().await?)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::get};

    struct TestTls {
        root_pem: Vec<u8>,
        chain_der: Vec<Vec<u8>>,
        key_der: Vec<u8>,
    }

    fn test_tls() -> anyhow::Result<TestTls> {
        let ca_key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
        let mut ca_params = rcgen::CertificateParams::default();
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        ca_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "hyprstream-jwks-test-root");
        let ca = ca_params.self_signed(&ca_key)?;

        let leaf_key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
        let mut leaf_params = rcgen::CertificateParams::new(vec!["localhost".to_owned()])?;
        leaf_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "localhost");
        let leaf = leaf_params.signed_by(&leaf_key, &ca, &ca_key)?;

        Ok(TestTls {
            root_pem: ca.pem().into_bytes(),
            chain_der: vec![leaf.der().to_vec(), ca.der().to_vec()],
            key_der: leaf_key.serialize_der(),
        })
    }

    #[tokio::test]
    async fn default_fetcher_rejects_untrusted_tls_and_accepts_declared_ca() -> anyhow::Result<()> {
        hyprstream_rpc::transport::install_pq_crypto_provider()?;
        let tls = test_tls()?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let rustls = axum_server::tls_rustls::RustlsConfig::from_der(
            tls.chain_der.clone(),
            tls.key_der.clone(),
        )
        .await?;
        let app = Router::new().route(
            "/oauth/jwks",
            get(|| async { Json(serde_json::json!({ "keys": [] })) }),
        );
        let server = tokio::spawn(async move {
            let _ = axum_server::from_tcp_rustls(listener, rustls)
                .serve(app.into_make_service())
                .await;
        });
        let url = format!("https://localhost:{}/oauth/jwks", address.port());

        let error = match default_jwks_fetcher()(url.clone()).await {
            Ok(jwks) => anyhow::bail!("untrusted TLS certificate returned JWKS: {jwks}"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("certificate") || error.to_string().contains("TLS"),
            "expected TLS validation error, got: {error:#}"
        );

        let declared_ca = reqwest::Certificate::from_pem(&tls.root_pem)?;
        let trusted_client = reqwest::Client::builder()
            .no_proxy()
            .add_root_certificate(declared_ca)
            .build()?;
        let jwks = jwks_fetcher_with_client(trusted_client)(url).await?;
        assert_eq!(jwks, serde_json::json!({ "keys": [] }));

        server.abort();
        Ok(())
    }
}
