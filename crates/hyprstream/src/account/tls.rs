//! DNS-01 wildcard certificate issuance for the account zone
//! (epic #1158, ticket #1162 / A3).
//!
//! `rustls-acme`, which the crate already uses for HTTP-01/TLS-ALPN-01,
//! **only fulfills TLS-ALPN-01 challenges** — its `AcmeState` cannot perform
//! DNS-01, and DNS-01 is the *only* validation method public CAs accept for a
//! wildcard identifier (`*.<apex>`). Wildcards cannot use HTTP-01.
//!
//! hyprstream therefore defines the issuance path itself, **behind a trait**:
//! [`WildcardCertIssuer`] owns the ACME DNS-01 dance (order → authorization →
//! publish TXT via the [`DnsProvider`](super::dns::DnsProvider) seam → poll →
//! finalize), and returns a DER-encoded cert+key. The concrete ACME-directory
//! client and the DNS backend are both product-layer concerns above this crate —
//! hyprstream must not depend on a specific DNS server (the self-hoster rule).
//!
//! What hyprstream *does* own here is:
//! - the shape of the request (zone, zone-scoped credential, ACME directory),
//! - the **sane-degrade** when no zone is configured (clear error, no panic,
//!   no synthesized default),
//! - the renewal + expiry-alarm plumbing (see [`super::renewal`]).
//!
//! Converting an [`IssuedCert`] into a live `rustls::ServerConfig` for the
//! account HTTP face is the job of B3 (the credential-free origin, #1165); A3
//! stops at the issuance contract and its monitoring.

use async_trait::async_trait;
use thiserror::Error;
use time::OffsetDateTime;
use tokio::sync::watch;
use zeroize::Zeroizing;

use super::config::AccountZoneConfig;
use super::dns::{DnsError, DnsProvider};
use super::zone::{AccountZone, AccountZoneError};

/// A DER-encoded wildcard certificate + its private key, freshly issued.
///
/// The key is wrapped in `Zeroizing` so it is cleared on drop.
#[derive(Debug, Clone)]
pub struct IssuedCert {
    /// End-entity certificate DER (leaf).
    pub cert_der: Vec<u8>,
    /// Optional intermediate issuer DERs.
    pub chain_der: Vec<Vec<u8>>,
    /// The private key DER (PKCS#8 or SEC1).
    pub key_der: Zeroizing<Vec<u8>>,
    /// The certificate's NotAfter, for renewal scheduling + the expiry alarm.
    pub not_after: OffsetDateTime,
    /// The name this cert was issued for (`*.<apex>`).
    pub san: String,
}

/// An error from the DNS-01 wildcard issuance path.
#[derive(Debug, Error)]
pub enum IssuanceError {
    /// The deployment configured no account zone / credential. **Sane-degrade**.
    #[error("account zone DNS-01 issuance not provisioned: {0}")]
    Unprovisioned(String),
    /// The configured zone name was invalid.
    #[error(transparent)]
    InvalidZone(#[from] AccountZoneError),
    /// The DNS backend failed to publish/clean the challenge TXT record.
    #[error(transparent)]
    Dns(#[from] DnsError),
    /// The issuer backend returned an ACME/protocol error.
    #[error("ACME DNS-01 issuance failed for {san}: {reason}")]
    Backend { san: String, reason: String },
}

/// The interface a deployment's DNS-01 wildcard issuer implements.
///
/// Implementations live **outside** this crate (metal repo / product layer):
/// they drive an ACME directory against `*.<apex>`, publish the `_acme-challenge`
/// TXT record via the supplied [`DnsProvider`] (so the same zone-scoped
/// credential plumbing is reused), poll the authorization to completion,
/// finalize with a CSR, and return the resulting [`IssuedCert`].
///
/// [`NullWildcardCertIssuer`] is the in-crate fail-closed default.
#[async_trait]
pub trait WildcardCertIssuer: Send + Sync {
    /// Issue (or renew) the wildcard certificate for `zone`. `dns` is the
    /// deployment's DNS provider; `cfg` carries the ACME directory, contact,
    /// and zone-scoped credential.
    async fn issue(
        &self,
        zone: &AccountZone,
        cfg: &AccountZoneConfig,
        dns: &dyn DnsProvider,
    ) -> Result<IssuedCert, IssuanceError>;
}

/// Fail-closed issuer for deployments that wired no ACME-DNS-01 backend.
///
/// Every call returns [`IssuanceError::Unprovisioned`]. The process keeps
/// running on whatever TLS materials it already has (self-signed or operator
/// `files`), and account minting surfaces the missing zone; nothing panics and
/// no certificate is synthesized.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullWildcardCertIssuer;

#[async_trait]
impl WildcardCertIssuer for NullWildcardCertIssuer {
    async fn issue(
        &self,
        zone: &AccountZone,
        _cfg: &AccountZoneConfig,
        _dns: &dyn DnsProvider,
    ) -> Result<IssuedCert, IssuanceError> {
        Err(IssuanceError::Unprovisioned(format!(
            "no ACME DNS-01 wildcard issuer wired up by the deployment; cannot obtain a certificate for {}",
            zone.wildcard_domain()
        )))
    }
}

/// Validate that the deployment has provisioned DNS-01 wildcard issuance.
///
/// Returns the resolved [`AccountZone`] on success, or an
/// [`IssuanceError::Unprovisioned`] / [`IssuanceError::InvalidZone`] otherwise.
/// This is the sane-degrade check the serving/renewal layers call before
/// attempting issuance — it never panics and never synthesizes a zone.
pub fn require_provisioned(cfg: &AccountZoneConfig) -> Result<AccountZone, IssuanceError> {
    if !cfg.is_configured() {
        return Err(IssuanceError::Unprovisioned(
            "no [account] zone configured — did:web minting and wildcard DNS-01 TLS issuance are disabled. Set [account] zone to enable.".to_owned(),
        ));
    }
    if !cfg.dns01_ready() {
        return Err(IssuanceError::Unprovisioned(
            "account zone is set but [account] dns01_credential is missing — wildcard DNS-01 TLS issuance is disabled. Set dns01_credential (zone-scoped) to enable.".to_owned(),
        ));
    }
    Ok(cfg.resolve_zone()?)
}

/// A live, hot-swappable handle to the currently-served account-zone cert.
///
/// The renewal task pushes freshly issued [`IssuedCert`]s into the underlying
/// `watch::Sender`; consumers read [`CertHandle::current`] for the latest. This
/// is the total-outage surface the expiry alarm watches — one cert failing
/// breaks all account resolution for the deployment.
#[derive(Clone)]
pub struct CertHandle {
    rx: watch::Receiver<Option<IssuedCert>>,
}

impl CertHandle {
    /// Create a handle plus the sender the renewal task writes to.
    pub fn channel() -> (watch::Sender<Option<IssuedCert>>, Self) {
        let (tx, rx) = watch::channel(None);
        (tx, Self { rx })
    }

    /// The currently-served cert, if one has been issued.
    pub fn current(&self) -> Option<IssuedCert> {
        self.rx.borrow().clone()
    }

    /// The cert NotAfter, if a cert has been issued.
    pub fn not_after(&self) -> Option<OffsetDateTime> {
        self.rx.borrow().as_ref().map(|c| c.not_after)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::account::dns::UnconfiguredDnsProvider;

    fn cfg(zone: Option<&str>, cred: Option<&str>) -> AccountZoneConfig {
        AccountZoneConfig {
            zone: zone.map(str::to_owned),
            dns01_credential: cred.map(str::to_owned),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn null_issuer_fails_closed() {
        let zone = AccountZone::new("acct.example.com").unwrap();
        let issuer = NullWildcardCertIssuer;
        let cfg = cfg(Some("acct.example.com"), Some("ref:tok"));
        let dns = UnconfiguredDnsProvider::new();
        let err = issuer.issue(&zone, &cfg, &dns).await.unwrap_err();
        assert!(matches!(err, IssuanceError::Unprovisioned(_)));
    }

    #[test]
    fn require_provisioned_degrades_sanely() {
        // No zone.
        assert!(matches!(
            require_provisioned(&cfg(None, None)),
            Err(IssuanceError::Unprovisioned(_))
        ));
        // Zone but no credential.
        assert!(matches!(
            require_provisioned(&cfg(Some("acct.example.com"), None)),
            Err(IssuanceError::Unprovisioned(_))
        ));
        // Both → ok.
        let z = require_provisioned(&cfg(Some("acct.example.com"), Some("ref:tok"))).unwrap();
        assert_eq!(z.wildcard_domain(), "*.acct.example.com");
    }
}
