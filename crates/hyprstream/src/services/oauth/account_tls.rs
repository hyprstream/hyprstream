//! Account-zone certificate rotation stays inside the owned account worker.

use crate::account::{AccountHttpConfig, AccountZone};
use crate::server::tls::{serve_bound, BoundHttpListener};
use hyprstream_rpc::error::RpcError;
use std::{sync::Arc, time::Duration};
use tokio::sync::Notify;

pub(super) async fn serve_with_reload(
    bound: BoundHttpListener,
    app: axum::Router,
    shutdown: Arc<Notify>,
    config: AccountHttpConfig,
    zone: AccountZone,
    reload_interval: Duration,
) -> Result<(), RpcError> {
    let live_tls = match &bound {
        BoundHttpListener::Https(_, tls) => tls.clone(),
        BoundHttpListener::Http(_) => {
            return Err(RpcError::SpawnFailed(
                "account listener requires TLS".to_owned(),
            ));
        }
    };
    let reload = async {
        let mut tick = tokio::time::interval(reload_interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Startup already validated the first pair. Subsequent reads also
        // catch atomic file/symlink replacement without a filesystem watcher.
        tick.tick().await;
        loop {
            tick.tick().await;
            match super::resolve_account_http_tls(&config, &zone).await {
                Ok(candidate) => live_tls.reload_from_config(candidate.get_inner()),
                Err(error) => tracing::warn!(
                    "account TLS rotation rejected; retaining previous certificate and retrying: {error:#}"
                ),
            }
        }
    };
    // No spawned watcher can outlive this serve future. A slow reload does not
    // stop polling the listener/shutdown; the enclosing ContainedWorker owns
    // the runtime and preserves the existing bounded process-exit guarantee.
    tokio::select! {
        result = serve_bound(bound, app, shutdown, "AccountHttpService") => result,
        () = reload => unreachable!("account certificate reload loop is permanent"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;

    #[tokio::test]
    async fn account_tls_rotation_updates_live_https_and_preserves_shutdown() -> anyhow::Result<()>
    {
        hyprstream_rpc::transport::install_pq_crypto_provider()?;
        let directory = tempfile::tempdir()?;
        let zone = AccountZone::new("acct.example.com")?;
        let first = rcgen::generate_simple_self_signed(vec![zone.wildcard_domain().to_owned()])?;
        let second = rcgen::generate_simple_self_signed(vec![zone.wildcard_domain().to_owned()])?;
        let wrong_zone =
            rcgen::generate_simple_self_signed(vec!["*.other.example.com".to_owned()])?;
        let config = AccountHttpConfig {
            host: "127.0.0.1".to_owned(),
            port: 8443,
            tls_cert: directory.path().join("cert.pem"),
            tls_key: directory.path().join("key.pem"),
        };
        std::fs::write(&config.tls_cert, first.cert.pem())?;
        std::fs::write(&config.tls_key, first.key_pair.serialize_pem())?;
        let tls = super::super::resolve_account_http_tls(&config, &zone).await?;
        let bound = crate::server::tls::bind_listener(
            "127.0.0.1:0".parse()?,
            Some(tls),
            "AccountRotationTest",
        )?;
        let address = match &bound {
            BoundHttpListener::Https(listener, _) => listener.local_addr()?,
            _ => unreachable!(),
        };
        let shutdown = Arc::new(Notify::new());
        let server = tokio::spawn(serve_with_reload(
            bound,
            axum::Router::new().route("/", axum::routing::get(|| async { "account-ok" })),
            shutdown.clone(),
            config.clone(),
            zone,
            Duration::from_millis(20),
        ));
        let client = reqwest::Client::builder()
            .use_rustls_tls()
            .no_proxy()
            .tls_info(true)
            .pool_max_idle_per_host(0)
            .add_root_certificate(reqwest::Certificate::from_der(first.cert.der())?)
            .add_root_certificate(reqwest::Certificate::from_der(second.cert.der())?)
            .resolve("alice.acct.example.com", address)
            .timeout(Duration::from_secs(2))
            .build()?;
        let url = format!("https://alice.acct.example.com:{}/", address.port());
        let peer = || async {
            let response = client.get(&url).send().await?.error_for_status()?;
            let certificate = response
                .extensions()
                .get::<reqwest::tls::TlsInfo>()
                .and_then(|info| info.peer_certificate())
                .expect("live TLS handshake must report the peer certificate")
                .to_vec();
            assert_eq!(response.text().await?, "account-ok");
            Ok::<_, anyhow::Error>(certificate)
        };
        assert_eq!(peer().await?, first.cert.der().as_ref());

        // Reject a valid keypair for the wrong account zone, then an incomplete
        // cert/key rotation. Both must leave the trusted live pair untouched.
        std::fs::write(&config.tls_cert, wrong_zone.cert.pem())?;
        std::fs::write(&config.tls_key, wrong_zone.key_pair.serialize_pem())?;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(peer().await?, first.cert.der().as_ref());
        std::fs::write(&config.tls_cert, second.cert.pem())?;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(peer().await?, first.cert.der().as_ref());

        std::fs::write(&config.tls_key, second.key_pair.serialize_pem())?;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if peer().await? == second.cert.der().as_ref() {
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await??;
        shutdown.notify_one();
        tokio::time::timeout(Duration::from_secs(2), server).await???;
        drop(std::net::TcpListener::bind(address)?);
        Ok(())
    }
}
