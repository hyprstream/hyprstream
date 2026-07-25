//! The DNS management interface (epic #1158, ticket #1162 / Phase A3).
//!
//! hyprstream **defines** this interface; it does **not** depend on any specific
//! DNS implementation. Managed DNS (PowerDNS, Route53, Cloudflare, a hand-rolled
//! RFC 2136 client, an out-of-process HTTP webhook, …) is a *product layer
//! above* hyprstream — if this crate ever required a particular DNS server,
//! self-hosters could not run it.
//!
//! Two jobs flow through this trait:
//!
//! 1. **ACME DNS-01 challenge responses** — publish/delete a TXT record at the
//!    `_acme-challenge.<apex>` owner name so a public CA can validate the
//!    `*.<apex>` wildcard certificate.
//! 2. **Wildcard address records** — publish wildcard A/AAAA for the delegated
//!    zone so `alice.<apex>` resolves to the hyprstream node. The wildcard
//!    covers **exactly one** label (the A2 depth invariant; see
//!    [`crate::account::zone`]).
//!
//! The credential a deployment hands to its `DnsProvider` implementation must be
//! **zone-scoped** — able to write records only inside the delegated account
//! zone and nowhere else. The plan is explicit: do not hand a zone-wide DNS
//! credential to the ACME client.

use std::fmt;
use std::net::{Ipv4Addr, Ipv6Addr};

use async_trait::async_trait;
use thiserror::Error;

use super::zone::AccountZone;

/// A DNS operation failed. Implementations wrap their backend's native error.
#[derive(Debug, Error)]
pub enum DnsError {
    /// The configured credential is missing or refused.
    #[error("DNS provider is unconfigured or unauthorized for {0}: {1}")]
    Unconfigured(String, String),
    /// The backend returned an error.
    #[error("DNS backend error: {0}")]
    Backend(String),
    /// The caller asked for an operation outside the delegated zone.
    #[error("owner name {0} is outside the delegated account zone")]
    OutOfZone(String),
}

/// The interface a deployment's DNS management layer implements.
///
/// Implementations live **outside** this crate (metal repo / product layer).
/// [`UnconfiguredDnsProvider`] is the in-crate fail-closed default used when a
/// deployment has not wired one up: every operation returns
/// [`DnsError::Unconfigured`], so the ACME DNS-01 issuance path fails loudly
/// rather than silently no-op'ing or inventing records.
#[async_trait]
pub trait DnsProvider: Send + Sync {
    /// Human-readable backend identifier (e.g. `"unconfigured"`, `"powerdns"`,
    /// `"route53"`). Used in logs and the expiry alarm.
    fn backend(&self) -> &'static str;

    /// Publish a TXT record at `owner` (a fully-qualified name, no trailing
    /// dot) with `value` for the ACME DNS-01 challenge. `ttl` is a hint in
    /// seconds; backends may clamp. The record must be retrievable by public
    /// resolvers within ~minutes.
    ///
    /// Implementations MUST reject `owner` values outside `zone` (the delegated
    /// account zone) with [`DnsError::OutOfZone`].
    async fn publish_txt(
        &self,
        zone: &AccountZone,
        owner: &str,
        value: &str,
        ttl: u32,
    ) -> Result<(), DnsError>;

    /// Delete the TXT record previously published by [`Self::publish_txt`].
    /// Must be idempotent (deleting a record that is already gone is success).
    async fn delete_txt(&self, zone: &AccountZone, owner: &str, value: &str)
        -> Result<(), DnsError>;

    /// Publish the wildcard A record (`*.<apex>`) for the delegated zone. The
    /// record covers exactly one label — `alice.<apex>` resolves, but
    /// `a.b.<apex>` does not.
    async fn publish_wildcard_a(
        &self,
        zone: &AccountZone,
        target: Ipv4Addr,
        ttl: u32,
    ) -> Result<(), DnsError>;

    /// Publish the wildcard AAAA record (`*.<apex>`) for the delegated zone.
    async fn publish_wildcard_aaaa(
        &self,
        zone: &AccountZone,
        target: Ipv6Addr,
        ttl: u32,
    ) -> Result<(), DnsError>;
}

/// Fail-closed no-op provider used when a deployment wires no DNS backend.
///
/// Every operation returns [`DnsError::Unconfigured`]. This is the sane-degrade
/// behavior at the DNS layer: DNS-01 wildcard issuance cannot proceed without a
/// real backend, and silent success would leave the deployment serving a
/// self-signed cert (or none) under a zone it never proved control of.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnconfiguredDnsProvider;

impl UnconfiguredDnsProvider {
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DnsProvider for UnconfiguredDnsProvider {
    fn backend(&self) -> &'static str {
        "unconfigured"
    }

    async fn publish_txt(
        &self,
        _zone: &AccountZone,
        owner: &str,
        _value: &str,
        _ttl: u32,
    ) -> Result<(), DnsError> {
        Err(DnsError::Unconfigured(
            owner.to_owned(),
            "no DNS provider wired up by the deployment".to_owned(),
        ))
    }

    async fn delete_txt(
        &self,
        _zone: &AccountZone,
        owner: &str,
        _value: &str,
    ) -> Result<(), DnsError> {
        Err(DnsError::Unconfigured(
            owner.to_owned(),
            "no DNS provider wired up by the deployment".to_owned(),
        ))
    }

    async fn publish_wildcard_a(
        &self,
        zone: &AccountZone,
        _target: Ipv4Addr,
        _ttl: u32,
    ) -> Result<(), DnsError> {
        Err(DnsError::Unconfigured(
            zone.wildcard_domain().to_owned(),
            "no DNS provider wired up by the deployment".to_owned(),
        ))
    }

    async fn publish_wildcard_aaaa(
        &self,
        zone: &AccountZone,
        _target: Ipv6Addr,
        _ttl: u32,
    ) -> Result<(), DnsError> {
        Err(DnsError::Unconfigured(
            zone.wildcard_domain().to_owned(),
            "no DNS provider wired up by the deployment".to_owned(),
        ))
    }
}

impl fmt::Display for UnconfiguredDnsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unconfigured")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unconfigured_fails_closed() {
        let p = UnconfiguredDnsProvider::new();
        let z = AccountZone::new("acct.example.com").unwrap();
        let owner = z.dns01_validation_name();
        assert!(matches!(
            p.publish_txt(&z, &owner, "token", 60).await,
            Err(DnsError::Unconfigured(_, _))
        ));
        assert!(matches!(
            p.delete_txt(&z, &owner, "token").await,
            Err(DnsError::Unconfigured(_, _))
        ));
        assert!(matches!(
            p.publish_wildcard_a(&z, "127.0.0.1".parse().unwrap(), 60).await,
            Err(DnsError::Unconfigured(_, _))
        ));
        assert!(matches!(
            p.publish_wildcard_aaaa(&z, "::1".parse().unwrap(), 60).await,
            Err(DnsError::Unconfigured(_, _))
        ));
        assert_eq!(p.backend(), "unconfigured");
    }
}
