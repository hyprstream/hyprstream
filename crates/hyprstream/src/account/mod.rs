//! Account identity — host-form `did:web` account zone, DNS interface, and
//! DNS-01 wildcard TLS (epic #1158, Phase A3 / ticket #1162).
//!
//! This module owns the deployment-facing configuration surface for the account
//! zone and the **interfaces** (not implementations) for the two product-layer
//! concerns above it: managed DNS and ACME DNS-01 wildcard issuance.
//!
//! # What hyprstream owns vs. what the product layer owns
//!
//! | Concern | Owner |
//! |---|---|
//! | Account-zone apex name (no default — sane-degrade) | hyprstream (`config`) |
//! | Wildcard shape `*.<apex>` (exactly one label — the A2 seam) | hyprstream (`zone`) |
//! | Renewal schedule + expiry alarm (total-outage surface) | hyprstream (`renewal`) |
//! | Sane-degrade when unconfigured | hyprstream (`config`, `tls`) |
//! | Publishing DNS-01 TXT + wildcard A/AAAA records | **product layer** (`dns::DnsProvider`) |
//! | Driving the ACME directory for DNS-01 issuance | **product layer** (`tls::WildcardCertIssuer`) |
//!
//! The account **label grammar** (LDH, NFKC, confusable rejection, reserved
//! list, never-reuse) is owned by A2 (`#1160`); B1 enforces it at mint time.
//! This crate only guarantees the wildcard covers exactly one label.
//!
//! # No PowerDNS, anywhere
//!
//! hyprstream must not require a specific DNS server. If it did, self-hosters
//! could not run it. Managed DNS is a product layer above a protocol that does
//! not depend on it — the same posture the epic takes for managed DNS.

pub mod config;
pub mod dns;
pub mod renewal;
pub mod tls;
pub mod zone;

pub use config::AccountZoneConfig;
pub use dns::{DnsError, DnsProvider, UnconfiguredDnsProvider};
pub use renewal::{CertHealth, CertHealthMonitor, CertRenewer, AlarmThresholds};
pub use tls::{CertHandle, IssuedCert, IssuanceError, NullWildcardCertIssuer, WildcardCertIssuer};
pub use zone::{AccountZone, AccountZoneError};

use std::net::{Ipv4Addr, Ipv6Addr};

/// Publish the wildcard A/AAAA records for the delegated account zone, if the
/// deployment configured targets and wired a DNS backend.
///
/// The wildcard covers **exactly one** label (see [`zone::AccountZone`]). Both
/// `v4`/`v6` are optional — an IPv6-only deployment publishes AAAA only.
///
/// This is a convenience entry point over [`DnsProvider`]; it logs each
/// publication and a clear message when a target is configured but no backend
/// is wired (sane-degrade, not a panic).
pub async fn publish_wildcard_records(
    zone: &AccountZone,
    dns: &dyn DnsProvider,
    v4: Option<Ipv4Addr>,
    v6: Option<Ipv6Addr>,
    ttl: u32,
) -> anyhow::Result<()> {
    if let Some(v4) = v4 {
        if let Err(e) = dns.publish_wildcard_a(zone, v4, ttl).await {
            tracing::warn!(
                zone = %zone.apex(),
                backend = dns.backend(),
                error = %e,
                "failed to publish wildcard A for account zone"
            );
            anyhow::bail!(
                "wildcard A publication for {zone} via {} failed: {e}",
                dns.backend()
            );
        }
        tracing::info!(zone = %zone.apex(), target = %v4, "published wildcard A for account zone");
    }
    if let Some(v6) = v6 {
        if let Err(e) = dns.publish_wildcard_aaaa(zone, v6, ttl).await {
            tracing::warn!(
                zone = %zone.apex(),
                backend = dns.backend(),
                error = %e,
                "failed to publish wildcard AAAA for account zone"
            );
            anyhow::bail!(
                "wildcard AAAA publication for {zone} via {} failed: {e}",
                dns.backend()
            );
        }
        tracing::info!(zone = %zone.apex(), target = %v6, "published wildcard AAAA for account zone");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publish_records_fails_loudly_on_unconfigured() {
        let z = AccountZone::new("acct.example.com").unwrap();
        let dns = UnconfiguredDnsProvider::new();
        let err = publish_wildcard_records(
            &z,
            &dns,
            Some("203.0.113.10".parse().unwrap()),
            None,
            300,
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("wildcard A"), "{msg}");
    }

    #[tokio::test]
    async fn publish_records_noop_when_no_targets() {
        let z = AccountZone::new("acct.example.com").unwrap();
        let dns = UnconfiguredDnsProvider::new();
        // No targets configured → no publication attempted → no error even on
        // an unconfigured backend (operator may publish records out-of-band).
        publish_wildcard_records(&z, &dns, None, None, 300)
            .await
            .unwrap();
    }
}
