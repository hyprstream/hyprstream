//! The deployment-configured account zone (epic #1158, ticket #1162 / Phase A3).
//!
//! The account zone is the DNS apex under which every host-form `did:web` this
//! deployment mints lives — e.g. `did:web:alice.acct.example.com`. It is
//! **deployment-relative configuration**: the operator picks it, in the metal
//! repo, when deploying hyprstream. hyprstream never bakes a zone name in.
//!
//! # No default — ever
//!
//! A deployment that configures no account zone must **degrade sanely**: account
//! minting is disabled with a clear error. It must not panic, and above all it
//! must not synthesize a default zone name — a default would silently mint
//! permanent DIDs under a domain the operator never chose (one-way door #1 in
//! the epic). `AccountZone` therefore has no `Default` impl and no constructor
//! that invents a name; the only way to obtain one is
//! [`AccountZoneConfig::resolve_zone`](super::config::AccountZoneConfig::resolve_zone),
//! which returns `Err` when the zone is unset.
//!
//! # Label depth — the A2 seam (#1160)
//!
//! A wildcard certificate matches **exactly one** DNS label: `*.acct.example.com`
//! covers `alice.acct.example.com` and *never* `a.b.acct.example.com`. This
//! module fixes the *wildcard* shape (one label below the apex). The grammar a
//! single account label must satisfy (LDH, ≤63 chars, NFKC, no mixed-script,
//! reserved list, never-reuse) is owned by A2 (`#1160`); B1 enforces it at
//! mint time. A3 only guarantees the wildcard covers exactly one label, which
//! [`AccountZone::wildcard_domain`] does structurally.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An error raised while resolving or parsing an account zone.
#[derive(Debug, Error)]
pub enum AccountZoneError {
    /// The deployment configured no account zone. **This is the sane-degrade
    /// path**, not a panic and not a synthesized default. Account minting is
    /// disabled; callers surface this to the operator.
    #[error("no account zone configured — account minting is disabled. Set [account] zone (e.g. \"acct.example.com\") to enable did:web minting under a deployment-chosen apex.")]
    Unset,
    /// The configured zone name was not a syntactically valid DNS name.
    #[error("invalid account zone name {name:?}: {reason}")]
    Invalid { name: String, reason: String },
}

/// A deployment-configured, validated account-zone apex.
///
/// Constructed only via [`AccountZone::new`] / [`FromStr`], both of which
/// reject anything that is not a normalized, lowercase, fully-qualified DNS
/// name with at least one delegation label below a public suffix (so an
/// operator cannot accidentally configure a bare TLD). There is intentionally
/// **no `Default`**: see the module docs on sane-degrade.
///
/// The apex is the zone NS-delegated to hyprstream — e.g. `acct.example.com`,
/// NOT `*.acct.example.com` and NOT a per-account host. Wildcard and
/// per-account hostnames are derived from it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AccountZone {
    // Normalized lowercase ASCII DNS name without a trailing dot.
    apex: String,
    // Precomputed `*.apex` — the wildcard that covers exactly one label.
    wildcard: String,
}

impl AccountZone {
    /// Validate and wrap a zone apex like `acct.example.com`.
    ///
    /// Rejects empty input, trailing dots, uppercase (use the normalized form),
    /// underscores, leading digits-in-TLD-style oddities, labels over 63 octets,
    /// total length over 253, consecutive dots, and bare public suffixes
    /// (`com`, `co.uk`) which an operator almost certainly did not intend to
    /// delegate whole to hyprstream.
    pub fn new(apex: impl Into<String>) -> Result<Self, AccountZoneError> {
        let apex = apex.into();
        validate_zone_apex(&apex)?;
        // `validate_zone_apex` guarantees lowercase + no trailing dot, so this
        // wildcard is the exact one-label form a CA will issue for DNS-01.
        let wildcard = format!("*.{apex}");
        Ok(Self { apex, wildcard })
    }

    /// The bare apex, e.g. `acct.example.com`. Lowercase, no trailing dot.
    pub fn apex(&self) -> &str {
        &self.apex
    }

    /// The wildcard that covers **exactly one** label below the apex, e.g.
    /// `*.acct.example.com`. This is the name the DNS-01 wildcard TLS
    /// certificate is issued for, and the name wildcard A/AAAA records are
    /// published under.
    pub fn wildcard_domain(&self) -> &str {
        &self.wildcard
    }

    /// The DNS-01 `_acme-challenge` validation owner name for the wildcard
    /// certificate. Per RFC 8555, a `*.` identifier is validated by publishing
    /// the TXT record at `_acme-challenge.<apex>` (the apex, not the wildcard).
    pub fn dns01_validation_name(&self) -> String {
        format!("_acme-challenge.{}", self.apex)
    }

    /// Build a per-account host label under this zone, asserting the label is a
    /// **single** DNS label (the depth the wildcard covers).
    ///
    /// This is the A3 side of the A2 seam: A2 (`#1160`) owns the *grammar* of
    /// the label; A3 owns the *depth*. We reject any label that contains a `.`
    /// or is empty, because a multi-label label would escape the wildcard's
    /// single-label coverage and produce a hostname that does not resolve under
    /// the issued certificate.
    pub fn host_for_label(&self, label: &str) -> Result<String, AccountZoneError> {
        if label.is_empty() {
            return Err(AccountZoneError::Invalid {
                name: label.to_owned(),
                reason: "account label is empty".to_owned(),
            });
        }
        if label.contains('.') {
            return Err(AccountZoneError::Invalid {
                name: label.to_owned(),
                reason: "account label must be a single DNS label (the wildcard covers exactly one label below the zone); multi-label labels are not representable".to_owned(),
            });
        }
        Ok(format!("{label}.{}", self.apex))
    }
}

impl FromStr for AccountZone {
    type Err = AccountZoneError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl fmt::Display for AccountZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.apex)
    }
}

impl Serialize for AccountZone {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.apex)
    }
}

impl<'de> Deserialize<'de> for AccountZone {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

/// Validate a zone apex. See [`AccountZone::new`] for the rules.
fn validate_zone_apex(apex: &str) -> Result<(), AccountZoneError> {
    let bad = |reason: &str| AccountZoneError::Invalid {
        name: apex.to_owned(),
        reason: reason.to_owned(),
    };

    if apex.is_empty() {
        return Err(bad("zone name is empty"));
    }
    if apex != apex.trim() {
        return Err(bad("zone name has surrounding whitespace"));
    }
    if apex.ends_with('.') {
        return Err(bad("zone name must not have a trailing dot"));
    }
    if apex.to_ascii_lowercase() != apex {
        return Err(bad("zone name must be lowercase (use the NFKC/ASCII-normalized form)"));
    }
    if apex.len() > 253 {
        return Err(bad("zone name exceeds 253 octets"));
    }
    if apex.starts_with('.') || apex.ends_with('.') {
        return Err(bad("zone name has an empty label"));
    }

    let labels: Vec<&str> = apex.split('.').collect();
    if labels.len() < 2 {
        return Err(bad(
            "zone name must be delegated below a public suffix (a bare TLD is not an account zone)",
        ));
    }

    for label in &labels {
        if label.is_empty() {
            return Err(bad("zone name has a consecutive dot / empty label"));
        }
        if label.len() > 63 {
            return Err(bad("a label exceeds 63 octets"));
        }
        if !label.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') {
            return Err(bad("labels must be LDH (lowercase letters, digits, hyphens) only"));
        }
        if label.starts_with('-') || label.ends_with('-') {
            return Err(bad("labels must not start or end with a hyphen"));
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn valid_zone_normalizes_derived_names() {
        let z = AccountZone::new("acct.example.com").unwrap();
        assert_eq!(z.apex(), "acct.example.com");
        assert_eq!(z.wildcard_domain(), "*.acct.example.com");
        assert_eq!(z.dns01_validation_name(), "_acme-challenge.acct.example.com");
    }

    #[test]
    fn host_label_is_single_depth() {
        let z = AccountZone::new("acct.example.com").unwrap();
        assert_eq!(z.host_for_label("alice").unwrap(), "alice.acct.example.com");
        // Multi-label labels escape the wildcard — rejected.
        let err = z.host_for_label("a.b").unwrap_err();
        assert!(matches!(err, AccountZoneError::Invalid { .. }));
        assert!(z.host_for_label("").is_err());
    }

    #[test]
    fn rejects_bare_tld_and_bad_forms() {
        assert!(matches!(
            AccountZone::new("com").unwrap_err(),
            AccountZoneError::Invalid { .. }
        ));
        for bad in [
            "",
            "Acct.Example.com",
            "acct..example.com",
            "acct.example.com.",
            "-acct.example.com",
            &format!("{}.example.com", "a".repeat(64)),
        ]
        .into_iter()
        {
            let err = AccountZone::new(bad);
            assert!(err.is_err(), "expected {bad:?} to be rejected, got {err:?}");
        }
    }

    #[test]
    fn serde_roundtrips() {
        let z = AccountZone::new("acct.example.com").unwrap();
        let json = serde_json::to_string(&z).unwrap();
        assert_eq!(json, "\"acct.example.com\"");
        let back: AccountZone = serde_json::from_str(&json).unwrap();
        assert_eq!(z, back);
        // Deserializing a bad zone fails (no silent default).
        assert!(serde_json::from_str::<AccountZone>("\"\"").is_err());
    }

    #[test]
    fn from_str_matches_new() {
        let z: AccountZone = "acct.example.com".parse().unwrap();
        assert_eq!(z.apex(), "acct.example.com");
    }
}
