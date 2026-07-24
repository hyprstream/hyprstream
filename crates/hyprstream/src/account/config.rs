//! Configuration surface for the account zone (epic #1158, ticket #1162 / A3).
//!
//! This is the deployment-facing knob. The `[account]` section names the
//! delegated zone apex, the zone-scoped DNS-01 credential, the ACME directory
//! to obtain the wildcard certificate from, and the wildcard address targets.
//!
//! ## Sane-degrade, not synthesized default
//!
//! A deployment that configures **no** account zone must degrade sanely:
//! account minting is disabled with a clear error. [`AccountZoneConfig`]
//! therefore has **no** `Default` zone value — `Default` leaves `zone = None`,
//! and [`AccountZoneConfig::resolve_zone`] returns
//! [`AccountZoneError::Unset`] in that state. We must never synthesize a zone
//! name: a default would silently mint permanent DIDs under a domain the
//! operator never chose (epic one-way door #1).
//!
//! ## TOML
//!
//! ```toml
//! [account]
//! zone = "acct.example.com"
//! # Zone-scoped credential: the deployment's DNS provider accepts this and
//! # only this when writing records under the apex. Format is opaque to
//! # hyprstream — the product-layer DnsProvider interprets it.
//! dns01_credential = "ref:secretstore/account-zone-token"
//! acme_directory = "https://acme-v02.api.letsencrypt.org/directory"
//! acme_contact = "mailto:ops@example.com"
//! wildcard_ipv4 = "203.0.113.10"
//! wildcard_ipv6 = "2001:db8::20"
//! ```

use std::net::{Ipv4Addr, Ipv6Addr};

use serde::{Deserialize, Serialize};

use super::zone::{AccountZone, AccountZoneError};

/// The `[account]` configuration section.
///
/// `Default` is deliberately the **unconfigured** state (`zone = None`); see the
/// module docs on sane-degrade.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccountZoneConfig {
    /// The delegated account-zone apex, e.g. `acct.example.com`. `None` ⇒
    /// account minting disabled ([`Self::resolve_zone`] errors).
    #[serde(default)]
    pub zone: Option<String>,

    /// An opaque reference to the zone-scoped DNS-01 credential (e.g.
    /// `ref:secretstore/...` or an env var name). hyprstream treats this as an
    /// opaque string; the product-layer [`DnsProvider`](super::dns::DnsProvider)
    /// implementation interprets it. `None` ⇒ DNS-01 issuance disabled even if
    /// `zone` is set.
    #[serde(default)]
    pub dns01_credential: Option<String>,

    /// ACME directory URL for wildcard issuance. Defaults to Let's Encrypt
    /// production when `mode = "acme-dns01"` is selected and this is unset.
    #[serde(default)]
    pub acme_directory: Option<String>,

    /// ACME contact URI, e.g. `mailto:ops@example.com`.
    #[serde(default)]
    pub acme_contact: Option<String>,

    /// IPv4 target for the wildcard A record (`*.<apex>`). `None` ⇒ no A
    /// record published by hyprstream; the operator may publish one out-of-band.
    #[serde(default)]
    pub wildcard_ipv4: Option<Ipv4Addr>,

    /// IPv6 target for the wildcard AAAA record (`*.<apex>`).
    #[serde(default)]
    pub wildcard_ipv6: Option<Ipv6Addr>,
}

impl AccountZoneConfig {
    /// Whether the deployment has configured an account zone at all.
    pub fn is_configured(&self) -> bool {
        self.zone.as_deref().filter(|s| !s.trim().is_empty()).is_some()
    }

    /// Resolve the configured zone, or return [`AccountZoneError::Unset`].
    ///
    /// This is the single entry point the rest of the codebase uses; it never
    /// invents a zone. Callers should surface the `Unset` error to the operator
    /// rather than unwrapping.
    pub fn resolve_zone(&self) -> Result<AccountZone, AccountZoneError> {
        let Some(raw) = self.zone.as_deref() else {
            return Err(AccountZoneError::Unset);
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(AccountZoneError::Unset);
        }
        AccountZone::new(trimmed)
    }

    /// Whether the DNS-01 issuance path is fully provisioned: a zone **and** a
    /// zone-scoped credential are present. When false, TLS wildcard issuance is
    /// skipped with a clear log message (the process keeps running on its
    /// existing/self-signed cert; account minting surfaces the missing zone).
    pub fn dns01_ready(&self) -> bool {
        self.is_configured()
            && self
                .dns01_credential
                .as_deref()
                .map(|c| !c.trim().is_empty())
                .unwrap_or(false)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_is_unset_and_degrades() {
        let cfg = AccountZoneConfig::default();
        assert!(!cfg.is_configured());
        assert!(matches!(cfg.resolve_zone(), Err(AccountZoneError::Unset)));
        assert!(!cfg.dns01_ready());
    }

    #[test]
    fn empty_zone_string_is_unset() {
        let cfg = AccountZoneConfig {
            zone: Some("   ".to_owned()),
            ..Default::default()
        };
        assert!(matches!(cfg.resolve_zone(), Err(AccountZoneError::Unset)));
    }

    #[test]
    fn resolves_valid_zone() {
        let cfg = AccountZoneConfig {
            zone: Some("acct.example.com".to_owned()),
            dns01_credential: Some("ref:tok".to_owned()),
            ..Default::default()
        };
        let z = cfg.resolve_zone().unwrap();
        assert_eq!(z.wildcard_domain(), "*.acct.example.com");
        assert!(cfg.dns01_ready());
    }

    #[test]
    fn dns01_ready_requires_credential() {
        let cfg = AccountZoneConfig {
            zone: Some("acct.example.com".to_owned()),
            ..Default::default()
        };
        assert!(!cfg.dns01_ready()); // zone set but no credential
    }

    #[test]
    fn dns01_ready_rejects_blank_credential() {
        // A blank/whitespace credential is not usable; treat as unconfigured so
        // `require_provisioned` fails closed rather than proceeding (#1182 review).
        for blank in ["", "   ", "\t"] {
            let cfg = AccountZoneConfig {
                zone: Some("acct.example.com".to_owned()),
                dns01_credential: Some(blank.to_owned()),
                ..Default::default()
            };
            assert!(!cfg.dns01_ready(), "blank credential {blank:?} must not be ready");
        }
    }

    #[test]
    fn toml_roundtrip() {
        let toml = r#"
zone = "acct.example.com"
dns01_credential = "ref:tok"
acme_contact = "mailto:ops@example.com"
wildcard_ipv4 = "203.0.113.10"
"#;
        let cfg: AccountZoneConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.zone.as_deref(), Some("acct.example.com"));
        assert_eq!(cfg.wildcard_ipv4, Some("203.0.113.10".parse().unwrap()));
    }
}
