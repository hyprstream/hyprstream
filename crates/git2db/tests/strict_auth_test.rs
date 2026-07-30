//! Causal tests for the strict certificate + explicit-only auth boundary.
//!
//! These tests prove the security invariants required by issue #1429 using
//! only the public API:
//!   * The default certificate mode is `Strict`, and under `Strict` the
//!     certificate decision is `CertificatePassthrough` (defer to libgit2's
//!     built-in verification) — never `CertificateOk`. A MITM/wrong cert
//!     therefore fails the fetch rather than being silently accepted.
//!   * The default auth mode is `ExplicitOnly`.
//!   * `AcceptAll` survives only as an explicit opt-in.
//!   * The legacy sync `AuthManager` path defaults to strict/explicit-only and
//!     carries no ambient `Default` strategy.
//!
//! The ambient-credential *refusal* under `ExplicitOnly` is a pure decision of
//! the private `handle_auth` helper and is covered by in-crate unit tests in
//! `callback_config.rs` (which can reach private items); the end-to-end proof
//! is that `AuthStrategy::Default` is never emitted by the credentials callback
//! under the default mode.

#![cfg(test)]

use git2::CertificateCheckStatus;
use git2db::auth::{AuthManager, AuthStrategy};
use git2db::callback_config::{
    AuthMode, CallbackConfig, CertificateConfig, CertificatePinning, RedirectPolicy,
};

/// The default `CertificateConfig` must be `Strict`, not `AcceptAll`.
#[test]
fn default_certificate_mode_is_strict() {
    let cfg = CertificateConfig::default();
    assert!(
        matches!(cfg, CertificateConfig::Strict),
        "default CertificateConfig must be Strict (got {cfg:?}); the forge/untrusted default \
         must never accept all certificates",
    );
}

/// The default `AuthMode` must be `ExplicitOnly`.
#[test]
fn default_auth_mode_is_explicit_only() {
    let mode = AuthMode::default();
    assert!(
        matches!(mode, AuthMode::ExplicitOnly),
        "default AuthMode must be ExplicitOnly (got {mode:?})",
    );
}

/// The default `CallbackConfig` carries strict certs + explicit-only auth.
#[test]
fn default_callback_config_is_strict_and_explicit() {
    let cfg = CallbackConfig::default();
    assert!(matches!(cfg.certificates, CertificateConfig::Strict));
    assert!(matches!(cfg.auth_mode, AuthMode::ExplicitOnly));
}

// ---------------------------------------------------------------------------
// Off-site redirect policy — issue #1429 Sol P1 (revision round 2).
// ---------------------------------------------------------------------------

/// The default `RedirectPolicy` must be `None` (no off-site redirects), not
/// libgit2's own default (`Initial`, which allows one). Under the pinned
/// libgit2 1.9.x, the credential callback receives the pre-redirect URL even
/// after libgit2 follows an off-site redirect, so a host-scoped credential
/// could otherwise be offered to the redirect target.
#[test]
fn default_redirect_policy_is_none() {
    assert_eq!(RedirectPolicy::default(), RedirectPolicy::None);
}

/// The default `CallbackConfig` (the send-safe clone path) carries the secure
/// redirect default.
#[test]
fn default_callback_config_redirect_policy_is_none() {
    let cfg = CallbackConfig::default();
    assert_eq!(cfg.redirect_policy, RedirectPolicy::None);
}

/// `RedirectPolicy::None` maps to `git2::RemoteRedirect::None` (block every
/// off-site redirect); `Initial` maps to libgit2's own default
/// (`RemoteRedirect::Initial`, follow on the first request only). This is the
/// literal knob applied to `FetchOptions::follow_redirects`.
#[test]
fn redirect_policy_maps_to_git2_remote_redirect() {
    assert!(matches!(
        RedirectPolicy::None.to_git2(),
        git2::RemoteRedirect::None
    ));
    assert!(matches!(
        RedirectPolicy::Initial.to_git2(),
        git2::RemoteRedirect::Initial
    ));
}

/// The legacy sync `AuthManager` path — which has no `FetchOptions` of its
/// own to enforce the policy against — must still default to `None` so any
/// caller that plumbs `AuthManager::redirect_policy()` into its own
/// `FetchOptions::follow_redirects` inherits the secure default rather than
/// libgit2's `Initial`.
#[test]
fn auth_manager_default_redirect_policy_is_none() {
    assert_eq!(AuthManager::new().redirect_policy(), RedirectPolicy::None);
    assert_eq!(
        AuthManager::with_strategies(vec![]).redirect_policy(),
        RedirectPolicy::None
    );
}

/// `AuthManager::with_redirect_policy` is the explicit, opt-in-only escape
/// hatch — the default is never silently `Initial`.
#[test]
fn auth_manager_redirect_policy_is_explicit_opt_in() {
    let mgr = AuthManager::new().with_redirect_policy(RedirectPolicy::Initial);
    assert_eq!(mgr.redirect_policy(), RedirectPolicy::Initial);
}

// ---------------------------------------------------------------------------
// Certificate decision — the security boundary on cert handling.
// ---------------------------------------------------------------------------

/// Under `Strict`, the decision is `CertificatePassthrough`: libgit2 performs
/// its built-in verification and rejects on failure. It must never be
/// `CertificateOk`, which would accept an untrusted/MITM certificate.
#[test]
fn strict_certificate_returns_passthrough_not_ok() {
    let mode = CertificateConfig::Strict;
    let decision = CallbackConfig::certificate_decision(&mode, None);
    assert!(
        matches!(decision, Ok(CertificateCheckStatus::CertificatePassthrough)),
        "Strict mode must defer to libgit2 (Passthrough), not accept the cert (Ok)",
    );
}

/// Strict ignores the presented hostkey entirely — even with hostkey material
/// present, it still defers (never Ok).
#[test]
fn strict_certificate_ignores_presented_hostkey() {
    let mode = CertificateConfig::Strict;
    let hostkey: (&str, &[u8]) = ("evil-mitm.example", b"not-a-real-key");
    let decision = CallbackConfig::certificate_decision(&mode, Some(hostkey));
    assert!(matches!(
        decision,
        Ok(CertificateCheckStatus::CertificatePassthrough)
    ));
}

/// `AcceptAll` is the explicit, insecure opt-in and returns `CertificateOk`.
/// It must only ever be reached by an explicit caller choice, never by default.
#[test]
fn accept_all_is_explicit_opt_in_returning_ok() {
    let mode = CertificateConfig::AcceptAll;
    let decision = CallbackConfig::certificate_decision(&mode, None);
    assert!(matches!(
        decision,
        Ok(CertificateCheckStatus::CertificateOk)
    ));
}

/// `Pinned` with no matching fingerprint rejects — it never falls back to
/// accepting the certificate.
#[test]
fn pinned_certificate_rejects_non_matching() {
    let mode = CertificateConfig::Pinned(vec![CertificatePinning {
        host: "github.com".to_owned(),
        fingerprint: b"real-fingerprint".to_vec(),
    }]);
    // A presented hostkey that does NOT match the pin.
    let hostkey: (&str, &[u8]) = ("github.com", b"attacker-fingerprint");
    let res = CallbackConfig::certificate_decision(&mode, Some(hostkey));
    assert!(
        res.is_err(),
        "Pinned mode must reject a non-matching fingerprint, not accept it",
    );
}

/// `Pinned` accepts only an exact host + fingerprint match.
#[test]
fn pinned_certificate_accepts_exact_match() {
    let mode = CertificateConfig::Pinned(vec![CertificatePinning {
        host: "github.com".to_owned(),
        fingerprint: b"deadbeef".to_vec(),
    }]);
    let hostkey: (&str, &[u8]) = ("github.com", b"deadbeef");
    let decision = CallbackConfig::certificate_decision(&mode, Some(hostkey));
    assert!(matches!(
        decision,
        Ok(CertificateCheckStatus::CertificateOk)
    ));
}

// ---------------------------------------------------------------------------
// AuthManager (sync path) defaults — the legacy `auth.rs` surface.
// ---------------------------------------------------------------------------

/// `AuthManager::new()` must default to strict certs + explicit-only auth, and
/// must NOT carry an ambient `Default` strategy the way the legacy constructor
/// did (the old `new()` seeded `vec![AuthStrategy::Default]`).
#[test]
fn auth_manager_defaults_are_strict_and_explicit() {
    let mgr = AuthManager::new();
    assert_eq!(
        mgr.strategy_count(),
        0,
        "default AuthManager carries no ambient strategy"
    );
    assert!(matches!(mgr.certificate_mode(), CertificateConfig::Strict));
    assert!(matches!(mgr.auth_mode(), AuthMode::ExplicitOnly));
}

/// `AuthManager::with_strategies` also defaults to strict/explicit-only even
/// when the caller passes an explicit strategy list.
#[test]
fn auth_manager_with_strategies_defaults_strict() {
    let mgr = AuthManager::with_strategies(vec![AuthStrategy::Token {
        token: "x".to_owned(),
        host: None,
    }]);
    assert!(matches!(mgr.certificate_mode(), CertificateConfig::Strict));
    assert!(matches!(mgr.auth_mode(), AuthMode::ExplicitOnly));
}

/// The presets that bundle `AuthStrategy::Default` must opt into `AllowAmbient`
/// so their ambient intent is explicit rather than relying on the old default.
#[test]
fn ambient_presets_opt_into_allow_ambient() {
    use git2db::auth::presets;
    assert_eq!(presets::ssh_standard().auth_mode(), AuthMode::AllowAmbient);
    assert_eq!(
        presets::github_token("t").auth_mode(),
        AuthMode::AllowAmbient
    );
    assert_eq!(presets::public_only().auth_mode(), AuthMode::AllowAmbient);
}

// ---------------------------------------------------------------------------
// NetworkConfig default — ambient must be off by default (issue #1429).
// ---------------------------------------------------------------------------

/// `use_credential_helper` must default to `false` so ordinary `GitManager`
/// clones do not opt into ambient credential discovery (SSH agent / credential
/// helper / ~/.gitconfig) by default.
#[test]
fn use_credential_helper_defaults_to_false() {
    use git2db::config::NetworkConfig;
    let cfg = NetworkConfig::default();
    assert!(
        !cfg.use_credential_helper,
        "use_credential_helper must default to false (got true); ordinary clones must not \
         consult ambient credentials unless explicitly enabled",
    );
}

/// `AuthStrategy::is_ambient()` classifies `Default` and `SshAgent` as ambient
/// (the two environment-backed strategies), and explicit strategies as not.
#[test]
fn is_ambient_classifies_environment_sources() {
    assert!(AuthStrategy::Default.is_ambient());
    assert!(AuthStrategy::SshAgent { username: None }.is_ambient());
    assert!(
        !AuthStrategy::Token {
            token: "t".into(),
            host: None
        }
        .is_ambient()
    );
}
