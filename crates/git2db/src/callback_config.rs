//! Send-safe callback configuration for async git operations
//!
//! This module provides a bridge between git2's non-Send callbacks and async/Send requirements.
//! Instead of holding closures directly, we capture the configuration needed to create them.

use crate::auth::AuthStrategy;
use git2::cert::Cert;
use std::sync::Arc;

/// Progress reporting configuration
#[derive(Clone, Default)]
pub enum ProgressConfig {
    /// No progress reporting
    #[default]
    None,
    /// Simple progress to stdout
    Stdout,
    /// Progress via channel (Send-safe)
    Channel(Arc<dyn ProgressReporter>),
}

impl std::fmt::Debug for ProgressConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Stdout => write!(f, "Stdout"),
            Self::Channel(_) => write!(f, "Channel(<custom>)"),
        }
    }
}

/// Trait for Send-safe progress reporting
pub trait ProgressReporter: Send + Sync {
    fn report(&self, stage: &str, current: usize, total: usize);
}

/// Certificate validation configuration.
///
/// Controls how the TLS/SSH host-key certificate check is handled during a
/// network git operation. The default is [`CertificateConfig::Strict`], which
/// defers to libgit2's built-in validation against the system trust store and
/// **rejects on certificate failure**. This is the only acceptable mode for
/// untrusted or forge-facing fetches (the merge gate's PR-head fetch, P2P
/// distribution, etc.).
///
/// `AcceptAll` is an explicit, insecure opt-in reserved for tests/operator
/// bootstrapping where the transport is otherwise authenticated. It must never
/// be the default for a code path that fetches from an untrusted origin.
#[derive(Debug, Clone, Default)]
pub enum CertificateConfig {
    /// Strict validation: defer to libgit2's built-in certificate verification
    /// (system CA store for HTTPS, `known_hosts` for SSH) and reject on
    /// failure. This is the secure default.
    ///
    /// Implemented by returning [`git2::CertificateCheckStatus::CertificatePassthrough`]
    /// from the certificate callback, which signals "no application decision —
    /// use libgit2's result". libgit2 then applies its strict default: an
    /// untrusted CA, expired leaf, or hostname mismatch fails the fetch.
    #[default]
    Strict,
    /// Accept every certificate without validation. **Insecure.**
    ///
    /// Explicit opt-in only. Appropriate for unit tests against a local
    /// self-signed server, or operator bootstrapping over an already-trusted
    /// transport. The untrusted/forge default must never select this mode.
    AcceptAll,
    /// Custom validation against a set of pinned host fingerprints.
    Pinned(Vec<CertificatePinning>),
}

/// Authentication provenance mode — whether ambient credential sources may be
/// consulted.
///
/// `ExplicitOnly` (the default) refuses ambient strategies —
/// [`AuthStrategy::Default`] (git credential helper, `~/.gitconfig`) **and**
/// [`AuthStrategy::SshAgent`] (the running SSH agent's loaded keys). This
/// prevents an untrusted fetch from silently picking up the operator's personal
/// credentials via `Cred::ssh_key_from_agent` or `Cred::default`. Explicit
/// strategies (`SshKey` file, `UserPass`, `Token`) are always honored. Callers
/// that intentionally want ambient discovery (e.g. a trusted first-party mirror
/// clone driven by the operator's own identity) must explicitly opt in via
/// [`AuthMode::AllowAmbient`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuthMode {
    /// Only explicit authentication strategies are honored.
    /// Ambient strategies ([`AuthStrategy::Default`], [`AuthStrategy::SshAgent`])
    /// are rejected, so the credential helper, `~/.gitconfig`, and the SSH
    /// agent are never consulted. Secure default for untrusted/forge-facing
    /// fetches.
    #[default]
    ExplicitOnly,
    /// Permit ambient credential discovery via [`AuthStrategy::Default`] and
    /// [`AuthStrategy::SshAgent`]. Opt-in for trusted first-party mirrors
    /// where operator identity is expected.
    AllowAmbient,
}

/// Certificate pinning configuration
#[derive(Debug, Clone)]
pub struct CertificatePinning {
    pub host: String,
    pub fingerprint: Vec<u8>,
}

/// Send-safe callback configuration
///
/// This type captures the *configuration* for callbacks rather than the callbacks themselves.
/// It can be safely sent across threads and used to create the actual callbacks when needed.
#[derive(Debug, Clone, Default)]
pub struct CallbackConfig {
    /// Authentication strategies to try
    pub auth: Vec<AuthStrategy>,

    /// Whether ambient credential sources (`AuthStrategy::Default`) may be
    /// consulted. Defaults to [`AuthMode::ExplicitOnly`].
    pub auth_mode: AuthMode,

    /// Progress reporting configuration
    pub progress: ProgressConfig,

    /// Certificate validation configuration
    pub certificates: CertificateConfig,

    /// Pack progress reporting
    pub pack_progress: bool,

    /// Push update reference callback enabled
    pub push_update_reference: bool,
}

impl CallbackConfig {
    /// Create a new callback configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style method to add authentication
    pub fn with_auth(mut self, strategy: AuthStrategy) -> Self {
        self.auth.push(strategy);
        self
    }

    /// Builder-style method to set the authentication provenance mode.
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        self.auth_mode = mode;
        self
    }

    /// Builder-style method to set progress reporting
    pub fn with_progress(mut self, progress: ProgressConfig) -> Self {
        self.progress = progress;
        self
    }

    /// Builder-style method to set certificate validation
    pub fn with_certificates(mut self, certs: CertificateConfig) -> Self {
        self.certificates = certs;
        self
    }

    /// Extract the progress reporter (if using Channel config)
    pub fn progress_reporter(&self) -> Option<Arc<dyn ProgressReporter>> {
        match &self.progress {
            ProgressConfig::Channel(reporter) => Some(Arc::clone(reporter)),
            _ => None,
        }
    }

    /// Create actual git2::RemoteCallbacks from this configuration
    ///
    /// This is called within spawn_blocking where non-Send is acceptable
    pub fn create_callbacks(&self) -> git2::RemoteCallbacks<'_> {
        let mut callbacks = git2::RemoteCallbacks::new();

        // Set up authentication. Always install the credentials callback so
        // that AuthMode is enforced even when no explicit strategy is provided:
        // ExplicitOnly must refuse ambient discovery rather than silently fall
        // through to libgit2's default credential helper.
        let auth_strategies = self.auth.clone();
        let auth_mode = self.auth_mode;
        callbacks.credentials(move |url, username_from_url, allowed_types| {
            Self::handle_auth(
                &auth_strategies,
                auth_mode,
                url,
                username_from_url,
                allowed_types,
            )
        });

        // Set up certificate checking. Strict (the default) defers to libgit2's
        // built-in verification via CertificatePassthrough; it does NOT return
        // CertificateOk, so a cert that fails system trust is rejected.
        let cert_config = self.certificates.clone();
        callbacks
            .certificate_check(move |cert, _host| Self::handle_certificate(&cert_config, cert));

        // Set up progress if configured
        if !matches!(self.progress, ProgressConfig::None) {
            let progress_config = self.progress.clone();
            callbacks.transfer_progress(move |stats| {
                Self::handle_progress(&progress_config, stats);
                true
            });
        }

        callbacks
    }

    fn handle_auth(
        strategies: &[AuthStrategy],
        auth_mode: AuthMode,
        url: &str,
        username_from_url: Option<&str>,
        allowed_types: git2::CredentialType,
    ) -> Result<git2::Cred, git2::Error> {
        // Try each strategy in order. Under ExplicitOnly, ambient strategies
        // (AuthStrategy::Default → git credential helper / ~/.gitconfig;
        // AuthStrategy::SshAgent → ssh-agent loaded keys) are refused so an
        // untrusted fetch cannot silently pick up the operator's credentials.
        // Explicit strategies (SshKey file, UserPass, Token) are always honored.
        for strategy in strategies {
            if matches!(auth_mode, AuthMode::ExplicitOnly) && strategy.is_ambient() {
                tracing::debug!(
                    "Refusing ambient strategy {:?} for {url} under ExplicitOnly auth mode",
                    strategy
                );
                continue;
            }
            match Self::try_auth_strategy(strategy, url, username_from_url, allowed_types) {
                Ok(cred) => return Ok(cred),
                Err(_) => continue,
            }
        }

        if matches!(auth_mode, AuthMode::ExplicitOnly) {
            Err(git2::Error::from_str(
                "No suitable explicit authentication method (ambient discovery — credential \
                 helper, ~/.gitconfig, SSH agent — refused under ExplicitOnly auth mode)",
            ))
        } else {
            Err(git2::Error::from_str("No suitable authentication method"))
        }
    }

    fn try_auth_strategy(
        strategy: &AuthStrategy,
        url: &str,
        username_from_url: Option<&str>,
        allowed_types: git2::CredentialType,
    ) -> Result<git2::Cred, git2::Error> {
        use git2::{Cred, CredentialType};

        match strategy {
            AuthStrategy::SshAgent { username } => {
                if allowed_types.contains(CredentialType::SSH_KEY) {
                    let user = username.as_deref().or(username_from_url).unwrap_or("git");
                    Cred::ssh_key_from_agent(user)
                } else {
                    Err(git2::Error::from_str("SSH key not allowed"))
                }
            }
            AuthStrategy::SshKey {
                username,
                public_key,
                private_key,
                passphrase,
            } => {
                if allowed_types.contains(CredentialType::SSH_KEY) {
                    Cred::ssh_key(
                        username,
                        public_key.as_deref(),
                        private_key,
                        passphrase.as_deref(),
                    )
                } else {
                    Err(git2::Error::from_str("SSH key not allowed"))
                }
            }
            AuthStrategy::UserPass { username, password } => {
                if allowed_types.contains(CredentialType::USER_PASS_PLAINTEXT) {
                    Cred::userpass_plaintext(username, password)
                } else {
                    Err(git2::Error::from_str("Username/password not allowed"))
                }
            }
            AuthStrategy::Token { token, host } => {
                if !allowed_types.contains(CredentialType::USER_PASS_PLAINTEXT) {
                    return Err(git2::Error::from_str("Token authentication not allowed"));
                }
                // Enforce host scope (issue #1429 Sol P1): a token bound to an
                // exact origin host is only offered when the request URL's host
                // matches, preventing exfiltration to a caller-selected remote.
                if let Some(bound) = host {
                    match crate::auth::extract_git_host(url) {
                        Some(req_host) if req_host == bound.as_str() => {}
                        Some(req_host) => {
                            tracing::debug!(
                                "Refusing host-scoped token for {url}: bound to {bound:?}, \
                                 request host is {req_host:?}"
                            );
                            return Err(git2::Error::from_str(
                                "Token is scoped to a different host",
                            ));
                        }
                        None => {
                            tracing::debug!(
                                "Refusing host-scoped token for {url}: could not parse host"
                            );
                            return Err(git2::Error::from_str(
                                "Token is host-scoped but request URL host is unparseable",
                            ));
                        }
                    }
                }
                Cred::userpass_plaintext("", token)
            }
            AuthStrategy::Default => {
                if allowed_types.contains(CredentialType::DEFAULT) {
                    Cred::default()
                } else {
                    Err(git2::Error::from_str("Default credentials not allowed"))
                }
            }
        }
    }

    fn handle_certificate(
        config: &CertificateConfig,
        cert: &Cert<'_>,
    ) -> Result<git2::CertificateCheckStatus, git2::Error> {
        // Extract the hostkey material once; the decision itself is a pure
        // function of (mode, hostkey) and is tested directly.
        let hostkey = cert.as_hostkey().and_then(|hk| {
            hk.hostkey().map(|bytes| {
                let host = std::str::from_utf8(bytes).unwrap_or("");
                (host, bytes)
            })
        });
        Self::certificate_decision(config, hostkey)
    }

    /// Pure certificate decision, separated from FFI cert inspection so the
    /// security boundary is unit-testable without a live TLS connection.
    ///
    /// - [`CertificateConfig::Strict`] returns `CertificatePassthrough`, which
    ///   tells libgit2 to apply its built-in verification and **reject on
    ///   failure**. It never returns `CertificateOk`.
    /// - [`CertificateConfig::AcceptAll`] returns `CertificateOk` — the
    ///   explicit, insecure opt-in.
    /// - [`CertificateConfig::Pinned`] returns `Ok` only when the resolved
    ///   hostkey matches a pin, otherwise rejects. Never falls back to
    ///   `AcceptAll`.
    pub fn certificate_decision(
        mode: &CertificateConfig,
        hostkey: Option<(&str, &[u8])>,
    ) -> Result<git2::CertificateCheckStatus, git2::Error> {
        use git2::CertificateCheckStatus;
        match mode {
            CertificateConfig::Strict => Ok(CertificateCheckStatus::CertificatePassthrough),
            CertificateConfig::AcceptAll => Ok(CertificateCheckStatus::CertificateOk),
            CertificateConfig::Pinned(pins) => {
                if let Some((host, key_bytes)) = hostkey {
                    for pin in pins {
                        if pin.host == host && pin.fingerprint.as_slice() == key_bytes {
                            return Ok(CertificateCheckStatus::CertificateOk);
                        }
                    }
                }
                Err(git2::Error::from_str(
                    "Certificate does not match any pinned fingerprint",
                ))
            }
        }
    }

    fn handle_progress(config: &ProgressConfig, stats: git2::Progress<'_>) {
        // Determine stage and progress based on git2::Progress stats
        // Clone phases: fetch objects → index deltas → (checkout handled separately)
        let (stage, current, total) = if stats.indexed_deltas() > 0 && stats.total_deltas() > 0 {
            // Indexing deltas phase
            ("indexing", stats.indexed_deltas(), stats.total_deltas())
        } else if stats.indexed_objects() > 0
            && stats.indexed_objects() > stats.received_objects() / 2
        {
            // Indexing objects phase (when most objects received)
            ("indexing", stats.indexed_objects(), stats.total_objects())
        } else {
            // Fetching objects phase
            ("fetch", stats.received_objects(), stats.total_objects())
        };

        match config {
            ProgressConfig::None => {}
            ProgressConfig::Stdout => {
                tracing::info!("Progress [{stage}]: {current}/{total}");
            }
            ProgressConfig::Channel(reporter) => {
                reporter.report(stage, current, total);
            }
        }
    }
}

/// Builder for callback configuration
pub struct CallbackConfigBuilder {
    config: CallbackConfig,
}

impl Default for CallbackConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CallbackConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: CallbackConfig::new(),
        }
    }

    /// Add an authentication strategy
    pub fn auth(mut self, strategy: AuthStrategy) -> Self {
        self.config.auth.push(strategy);
        self
    }

    /// Add multiple authentication strategies
    pub fn auth_strategies(mut self, strategies: Vec<AuthStrategy>) -> Self {
        self.config.auth.extend(strategies);
        self
    }

    /// Set the authentication provenance mode (default: [`AuthMode::ExplicitOnly`]).
    pub fn auth_mode(mut self, mode: AuthMode) -> Self {
        self.config.auth_mode = mode;
        self
    }

    /// Set progress configuration
    pub fn progress(mut self, progress: ProgressConfig) -> Self {
        self.config.progress = progress;
        self
    }

    /// Set certificate configuration
    pub fn certificates(mut self, certs: CertificateConfig) -> Self {
        self.config.certificates = certs;
        self
    }

    /// Enable pack progress
    pub fn pack_progress(mut self, enabled: bool) -> Self {
        self.config.pack_progress = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CallbackConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_callback_config_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CallbackConfig>();
    }

    #[test]
    fn test_builder_pattern() {
        let config = CallbackConfigBuilder::new()
            .auth(AuthStrategy::SshAgent {
                username: Some("git".to_owned()),
            })
            .progress(ProgressConfig::Stdout)
            .certificates(CertificateConfig::AcceptAll)
            .build();

        assert_eq!(config.auth.len(), 1);
        assert!(matches!(config.progress, ProgressConfig::Stdout));
        assert!(matches!(config.certificates, CertificateConfig::AcceptAll));
    }

    // ---- AuthMode enforcement (private handle_auth seam) ----

    fn user_pass_allowed() -> git2::CredentialType {
        git2::CredentialType::USER_PASS_PLAINTEXT
    }

    fn default_allowed() -> git2::CredentialType {
        git2::CredentialType::DEFAULT
    }

    fn ssh_allowed() -> git2::CredentialType {
        git2::CredentialType::SSH_KEY
    }

    /// Under ExplicitOnly, a sole `AuthStrategy::Default` is refused — ambient
    /// credential sources (credential helper, ~/.gitconfig, SSH agent fallback)
    /// are not consulted.
    #[test]
    fn explicit_only_refuses_ambient_default() {
        let auth = vec![AuthStrategy::Default];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://github.com/example/repo.git",
            None,
            default_allowed(),
        );
        assert!(
            res.is_err(),
            "ExplicitOnly must refuse AuthStrategy::Default so ambient credentials are not \
             consulted",
        );
    }

    /// Under ExplicitOnly, `AuthStrategy::SshAgent` is ALSO refused — it
    /// consults the running SSH agent (`Cred::ssh_key_from_agent`), an ambient
    /// source. An untrusted fetch must not silently use the operator's loaded
    /// SSH keys.
    #[test]
    fn explicit_only_refuses_ambient_ssh_agent() {
        let auth = vec![AuthStrategy::SshAgent {
            username: Some("git".to_owned()),
        }];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://github.com/example/repo.git",
            None,
            ssh_allowed(),
        );
        assert!(
            res.is_err(),
            "ExplicitOnly must refuse AuthStrategy::SshAgent (ssh-agent is ambient)",
        );
    }

    /// Under AllowAmbient, `AuthStrategy::SshAgent` is honored — the explicit
    /// opt-in path for trusted first-party mirrors.
    #[test]
    fn allow_ambient_permits_ssh_agent() {
        let auth = vec![AuthStrategy::SshAgent {
            username: Some("git".to_owned()),
        }];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::AllowAmbient,
            "https://github.com/example/repo.git",
            None,
            ssh_allowed(),
        );
        assert!(
            res.is_ok(),
            "AllowAmbient must permit AuthStrategy::SshAgent"
        );
    }

    /// `is_ambient()` classifies exactly the environment-backed strategies.
    #[test]
    fn is_ambient_classification() {
        assert!(AuthStrategy::Default.is_ambient());
        assert!(AuthStrategy::SshAgent { username: None }.is_ambient());
        // Explicit strategies are never ambient.
        assert!(
            !AuthStrategy::Token {
                token: "t".into(),
                host: None
            }
            .is_ambient()
        );
        assert!(
            !AuthStrategy::UserPass {
                username: "u".into(),
                password: "p".into(),
            }
            .is_ambient()
        );
        assert!(
            !AuthStrategy::SshKey {
                username: "u".into(),
                public_key: None,
                private_key: PathBuf::from("/k"),
                passphrase: None,
            }
            .is_ambient()
        );
    }

    /// Under AllowAmbient, `AuthStrategy::Default` is honored when the server
    /// permits DEFAULT credentials — the explicit opt-in path.
    #[test]
    fn allow_ambient_permits_default() {
        let auth = vec![AuthStrategy::Default];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::AllowAmbient,
            "https://github.com/example/repo.git",
            None,
            default_allowed(),
        );
        assert!(
            res.is_ok(),
            "AllowAmbient must permit AuthStrategy::Default"
        );
    }

    /// Under ExplicitOnly, an explicit Token strategy succeeds and a trailing
    /// `Default` fallback is skipped rather than poisoning resolution or
    /// silently consulting ambient sources.
    #[test]
    fn explicit_only_skips_default_uses_explicit_token() {
        let auth = vec![
            AuthStrategy::Token {
                token: "ghp_explicit".to_owned(),
                host: None,
            },
            AuthStrategy::Default,
        ];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://github.com/example/repo.git",
            None,
            user_pass_allowed(),
        );
        assert!(
            res.is_ok(),
            "explicit Token must succeed under ExplicitOnly"
        );
    }

    /// Under ExplicitOnly with only ambient strategies, resolution fails even
    /// when DEFAULT is allowed by the server.
    #[test]
    fn explicit_only_no_strategies_fails_closed() {
        let auth: Vec<AuthStrategy> = vec![AuthStrategy::Default];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://github.com/example/repo.git",
            None,
            default_allowed(),
        );
        assert!(
            res.is_err(),
            "ExplicitOnly with only ambient strategies must fail closed",
        );
    }

    // ---- Host-scoped Token: a token for host A is never offered to host B ----

    /// A token bound to `github.com` must NOT be offered when the request URL
    /// points at a different host (`evil.example`). This is the host-A/host-B
    /// negative test required by the #1429 Sol P1 fix.
    #[test]
    fn host_scoped_token_refused_for_different_host() {
        let auth = vec![AuthStrategy::Token {
            token: "trusted-forge-token".to_owned(),
            host: Some("github.com".to_owned()),
        }];
        // Host B: the caller-selected remote is NOT github.com.
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://evil.example/untrusted/repo.git",
            None,
            user_pass_allowed(),
        );
        assert!(
            res.is_err(),
            "A host-scoped token bound to github.com must not be offered to evil.example",
        );
    }

    /// The same host-scoped token IS offered when the request URL's host
    /// matches the bound origin — the positive control for the negative test.
    #[test]
    fn host_scoped_token_offered_for_matching_host() {
        let auth = vec![AuthStrategy::Token {
            token: "trusted-forge-token".to_owned(),
            host: Some("github.com".to_owned()),
        }];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://github.com/user/repo.git",
            None,
            user_pass_allowed(),
        );
        assert!(
            res.is_ok(),
            "A host-scoped token must be offered to its bound host",
        );
    }

    /// An unscoped token (host: None) is offered to any host — this is the
    /// behavior the caller explicitly constructs; it is not attached to the
    /// default clone path (see manager.rs).
    #[test]
    fn unscoped_token_offered_to_any_host() {
        let auth = vec![AuthStrategy::Token {
            token: "explicit-unscoped".to_owned(),
            host: None,
        }];
        let res = CallbackConfig::handle_auth(
            &auth,
            AuthMode::ExplicitOnly,
            "https://anyhost.example/repo.git",
            None,
            user_pass_allowed(),
        );
        assert!(res.is_ok(), "An unscoped explicit Token is always honored");
    }
}
