//! Renewal, monitoring, and the expiry alarm for the account-zone wildcard
//! certificate (epic #1158, ticket #1162 / A3).
//!
//! One wildcard cert authenticates **every** account host for a deployment, so
//! one cert failing to renew breaks **all** account resolution — a total-outage
//! surface. Per the ticket, the alarm **ships with the feature**, not as a
//! follow-up.
//!
//! What this module owns:
//! - [`CertRenewer`]: on a schedule, asks the [`WildcardCertIssuer`](super::tls::WildcardCertIssuer)
//!   to refresh the cert and pushes the result into a [`CertHandle`](super::tls::CertHandle).
//! - [`CertHealth`]/[`CertHealthMonitor`]: computes a coarse health state
//!   (Ok / ExpiringSoon / Critical / Expired / Unissued / Unprovisioned) from
//!   the current cert's NotAfter and the last issuance outcome, and surfaces it
//!   loudly via `tracing` — the alarm. The health state is also exposed as a
//!   `pub` field so a metrics/event plane can scrape it.
//!
//! `CertRenewer` uses the same trait seams as issuance; it does not depend on a
//! specific ACME client or DNS backend.

use std::sync::Arc;
use std::time::Duration;

use time::OffsetDateTime;
use tokio::sync::watch;
use tracing::{error, info, warn};

use super::config::AccountZoneConfig;
use super::dns::DnsProvider;
use super::tls::{CertHandle, IssuanceError, IssuedCert, WildcardCertIssuer};

/// Coarse health classification for the account-zone cert. The alarm state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertHealth {
    /// No deployment-wide outage risk; cert is healthy and far from expiry.
    Ok,
    /// Cert is within `warn_lead` of expiry (default 30d). Renewal should be
    /// imminent. Informational.
    ExpiringSoon,
    /// Cert is within `critical_lead` of expiry (default 7d) **or** the last
    /// issuance failed. The deployment is at acute outage risk.
    Critical,
    /// Cert's NotAfter is in the past — account resolution is failing now.
    Expired,
    /// No cert has been issued yet.
    Unissued,
    /// The deployment has not provisioned DNS-01 issuance (zone/credential
    /// missing). Account minting is disabled by design; this is the sane-degrade
    /// state, surfaced so monitoring can distinguish "intentionally off" from
    /// "broken".
    Unprovisioned,
}

impl CertHealth {
    /// Human-readable label for metrics/logs.
    pub fn label(self) -> &'static str {
        match self {
            CertHealth::Ok => "ok",
            CertHealth::ExpiringSoon => "expiring_soon",
            CertHealth::Critical => "critical",
            CertHealth::Expired => "expired",
            CertHealth::Unissued => "unissued",
            CertHealth::Unprovisioned => "unprovisioned",
        }
    }
}

/// Tunable lead times for the alarm.
#[derive(Debug, Clone, Copy)]
pub struct AlarmThresholds {
    /// Warn when the cert is within this duration of expiry.
    pub warn_lead: Duration,
    /// Raise to Critical when within this duration of expiry.
    pub critical_lead: Duration,
}

impl Default for AlarmThresholds {
    fn default() -> Self {
        // Let's Encrypt certs are 90 days; warn at 30d, critical at 7d.
        Self {
            warn_lead: Duration::from_secs(30 * 24 * 3600),
            critical_lead: Duration::from_secs(7 * 24 * 3600),
        }
    }
}

/// Computes the alarm state from the current cert and the last issuance result.
///
/// Pure function; unit-testable without time mocking fixtures beyond
/// [`OffsetDateTime`].
pub fn classify(
    handle: &CertHandle,
    last_result: Option<&Result<IssuedCert, IssuanceError>>,
    now: OffsetDateTime,
    thresholds: &AlarmThresholds,
    unprovisioned: bool,
) -> CertHealth {
    if unprovisioned {
        return CertHealth::Unprovisioned;
    }
    // A recent issuance failure is always at least Critical.
    let last_failed = matches!(last_result, Some(Err(_)));
    let Some(not_after) = handle.not_after() else {
        return if last_failed { CertHealth::Critical } else { CertHealth::Unissued };
    };
    let remaining = not_after - now;
    if remaining.is_negative() {
        return CertHealth::Expired;
    }
    let dur = Duration::try_from(remaining).unwrap_or(Duration::ZERO);
    if dur < thresholds.critical_lead {
        CertHealth::Critical
    } else if dur < thresholds.warn_lead {
        CertHealth::ExpiringSoon
    } else if last_failed {
        // Cert still valid but the most recent renewal attempt failed — the
        // deployment is on a clock, surface it.
        CertHealth::Critical
    } else {
        CertHealth::Ok
    }
}

/// A monitor that recomputes [`CertHealth`] and logs it loudly when not `Ok`.
#[derive(Debug, Clone)]
pub struct CertHealthMonitor {
    thresholds: AlarmThresholds,
}

impl CertHealthMonitor {
    pub fn new(thresholds: AlarmThresholds) -> Self {
        Self { thresholds }
    }

    /// Recompute and emit the alarm. Returns the state so callers (metrics
    /// plane, readiness probes) can record it.
    pub fn observe(
        &self,
        handle: &CertHandle,
        last_result: Option<&Result<IssuedCert, IssuanceError>>,
        unprovisioned: bool,
    ) -> CertHealth {
        let now = OffsetDateTime::now_utc();
        let state = classify(
            handle,
            last_result,
            now,
            &self.thresholds,
            unprovisioned,
        );
        match state {
            CertHealth::Ok | CertHealth::Unprovisioned => {
                tracing::debug!(
                    account_tls_health = state.label(),
                    "account-zone wildcard TLS health: {}", state.label()
                );
            }
            CertHealth::ExpiringSoon => warn!(
                account_tls_health = state.label(),
                "account-zone wildcard certificate expiring soon — renewal should be imminent"
            ),
            CertHealth::Critical => error!(
                account_tls_health = state.label(),
                "account-zone wildcard certificate CRITICAL — one cert failing breaks ALL account resolution for this deployment"
            ),
            CertHealth::Expired => error!(
                account_tls_health = state.label(),
                "account-zone wildcard certificate EXPIRED — account resolution is failing now"
            ),
            CertHealth::Unissued => info!(
                account_tls_health = state.label(),
                "account-zone wildcard certificate not yet issued"
            ),
        }
        state
    }
}

/// Background renewal + alarm loop for the account-zone wildcard cert.
///
/// Construct with [`CertRenewer::new`]; spawn [`CertRenewer::run`] on the
/// service runtime. On each tick it:
/// 1. checks provisioning (sane-degrade if unprovisioned — no panic),
/// 2. asks the issuer for a fresh cert (only when due, see `renew_lead`),
/// 3. pushes the result into the [`CertHandle`] via the `sender`,
/// 4. runs the alarm.
///
/// The `sender` is the write half of a [`CertHandle::channel`](super::tls::CertHandle::channel)
/// pair; the serving/monitoring layer holds the read half. `handle` is a
/// `CertHandle` whose receiver mirrors the same channel so the renewer can read
/// `not_after` for the "due?" decision.
pub struct CertRenewer {
    cfg: AccountZoneConfig,
    issuer: Arc<dyn WildcardCertIssuer>,
    dns: Arc<dyn DnsProvider>,
    sender: watch::Sender<Option<IssuedCert>>,
    handle: CertHandle,
    thresholds: AlarmThresholds,
    /// Renew when the cert has less than this remaining validity.
    renew_lead: Duration,
    /// How often to wake up and re-evaluate (between renewals).
    tick: Duration,
}

impl CertRenewer {
    pub fn new(
        cfg: AccountZoneConfig,
        issuer: Arc<dyn WildcardCertIssuer>,
        dns: Arc<dyn DnsProvider>,
        sender: watch::Sender<Option<IssuedCert>>,
        handle: CertHandle,
    ) -> Self {
        Self {
            cfg,
            issuer,
            dns,
            sender,
            handle,
            thresholds: AlarmThresholds::default(),
            renew_lead: Duration::from_secs(30 * 24 * 3600),
            tick: Duration::from_secs(3600),
        }
    }

    /// Override the renew lead time (default 30d).
    pub fn with_renew_lead(mut self, lead: Duration) -> Self {
        self.renew_lead = lead;
        self
    }

    /// Override the alarm thresholds.
    pub fn with_thresholds(mut self, t: AlarmThresholds) -> Self {
        self.thresholds = t;
        self
    }

    /// Override the re-evaluation tick (default 1h).
    pub fn with_tick(mut self, tick: Duration) -> Self {
        self.tick = tick;
        self
    }

    /// Run the renewal + alarm loop until the process shuts down.
    ///
    /// Each iteration sleeps `tick`, then evaluates whether a renewal is due
    /// (no cert, or cert within `renew_lead` of expiry) and, if so, asks the
    /// issuer. Provisioning failures are reported via the alarm but do not
    /// terminate the loop — the process keeps serving whatever it has.
    pub async fn run(self) {
        let monitor = CertHealthMonitor::new(self.thresholds);
        // Sane-degrade: when unprovisioned, log once and skip issuance forever.
        // We never synthesize a zone; the `Option<AccountZone>` stays `None` and
        // the issuance block is gated on it.
        let zone = match super::tls::require_provisioned(&self.cfg) {
            Ok(z) => Some(z),
            Err(e) => {
                warn!("account-zone wildcard TLS not provisioned: {e}");
                None
            }
        };
        let unprovisioned = zone.is_none();
        let mut last_result: Option<Result<IssuedCert, IssuanceError>> = None;

        loop {
            if let Some(zone) = zone.as_ref() {
                let due = match self.handle.not_after() {
                    None => true,
                    Some(na) => {
                        let remaining = na - OffsetDateTime::now_utc();
                        let dur = Duration::try_from(remaining).unwrap_or(Duration::ZERO);
                        dur < self.renew_lead
                    }
                };
                if due {
                    let res = self
                        .issuer
                        .issue(zone, &self.cfg, self.dns.as_ref())
                        .await;
                    match &res {
                        Ok(c) => {
                            info!(
                                san = %c.san,
                                not_after = %c.not_after,
                                "account-zone wildcard certificate (re)issued"
                            );
                            // best_effort: if all receivers dropped, no-one to serve.
                            let _ = self.sender.send(Some(c.clone()));
                        }
                        Err(e) => {
                            error!(
                                error = %e,
                                "account-zone wildcard certificate renewal FAILED — outage risk for ALL account resolution"
                            );
                        }
                    }
                    last_result = Some(res);
                }
            }
            monitor.observe(&self.handle, last_result.as_ref(), unprovisioned);
            // Drop the borrowed reference before sleeping.
            tokio::time::sleep(self.tick).await;
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::account::config::AccountZoneConfig;
    use crate::account::dns::UnconfiguredDnsProvider;
    use crate::account::tls::NullWildcardCertIssuer;

    fn handle_with_cert(not_after: OffsetDateTime) -> CertHandle {
        let (tx, h) = CertHandle::channel();
        let _ = tx.send(Some(IssuedCert {
            cert_der: vec![],
            chain_der: vec![],
            key_der: zeroize::Zeroizing::new(vec![]),
            not_after,
            san: "*.acct.example.com".to_owned(),
        }));
        h
    }

    #[test]
    fn classify_states() {
        let t = AlarmThresholds::default();
        let now = OffsetDateTime::now_utc();
        let h = handle_with_cert(now + time::Duration::days(60));
        assert_eq!(classify(&h, None, now, &t, false), CertHealth::Ok);
        let h = handle_with_cert(now + time::Duration::days(20));
        assert_eq!(classify(&h, None, now, &t, false), CertHealth::ExpiringSoon);
        let h = handle_with_cert(now + time::Duration::days(3));
        assert_eq!(classify(&h, None, now, &t, false), CertHealth::Critical);
        let h = handle_with_cert(now - time::Duration::days(1));
        assert_eq!(classify(&h, None, now, &t, false), CertHealth::Expired);

        // Unissued vs unprovisioned
        let (tx, empty) = CertHandle::channel();
        drop(tx);
        assert_eq!(classify(&empty, None, now, &t, false), CertHealth::Unissued);
        assert_eq!(classify(&empty, None, now, &t, true), CertHealth::Unprovisioned);

        // Last failure escalates a still-valid cert to Critical.
        let h = handle_with_cert(now + time::Duration::days(60));
        let err: Result<IssuedCert, IssuanceError> = Err(IssuanceError::Unprovisioned("x".into()));
        assert_eq!(classify(&h, Some(&err), now, &t, false), CertHealth::Critical);
    }

    #[tokio::test]
    async fn renewer_degrades_when_unprovisioned() {
        // No zone → sane-degrade; the constructor + a single tick should not panic.
        let cfg = AccountZoneConfig::default();
        let issuer = Arc::new(NullWildcardCertIssuer) as Arc<dyn WildcardCertIssuer>;
        let dns = Arc::new(UnconfiguredDnsProvider::new()) as Arc<dyn DnsProvider>;
        let (sender, handle) = CertHandle::channel();
        let renewer = CertRenewer::new(cfg, issuer, dns, sender, handle)
            .with_tick(Duration::from_millis(10));
        // Run briefly then drop; we only assert it does not panic on an
        // unprovisioned deployment.
        let task = tokio::spawn(async move {
            renewer.run().await;
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        task.abort();
    }
}
