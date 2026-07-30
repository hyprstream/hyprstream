//! Authentication and credential management
//!
//! Consolidated authentication patterns from the original codebase.

use crate::callback_config::{AuthMode, CertificateConfig};
use git2::{Cred, CredentialType, RemoteCallbacks};
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Authentication strategy for git operations
#[derive(Debug, Clone)]
pub enum AuthStrategy {
    /// Use SSH key from agent
    SshAgent { username: Option<String> },
    /// Use specific SSH key file
    SshKey {
        username: String,
        public_key: Option<PathBuf>,
        private_key: PathBuf,
        passphrase: Option<String>,
    },
    /// Use username/password
    UserPass { username: String, password: String },
    /// Use a personal access token.
    ///
    /// `host` optionally binds the token to an exact origin host (e.g.
    /// `"github.com"`). When set, the credential callback only offers the
    /// token when the request URL's host matches — a token configured for
    /// host A is never sent to host B. `None` means unscoped; unscoped tokens
    /// must not be attached to default/untrusted clone paths (see
    /// [`crate::manager::GitManager::default_clone_options`]).
    Token { token: String, host: Option<String> },
    /// Use default credentials (for public repos)
    Default,
}

impl AuthStrategy {
    /// Whether this strategy consults **ambient** credential sources — the
    /// running SSH agent (`ssh_key_from_agent`), the git credential helper,
    /// or `~/.gitconfig` — rather than explicit material the caller provided.
    ///
    /// `Default` and `SshAgent` are ambient. `SshKey` (explicit file path),
    /// `UserPass`, and `Token` are explicit and always honored, including
    /// under [`crate::callback_config::AuthMode::ExplicitOnly`].
    pub fn is_ambient(&self) -> bool {
        matches!(self, AuthStrategy::Default | AuthStrategy::SshAgent { .. })
    }
}

/// Extract the host component from a git remote URL.
///
/// Handles the common forms:
/// - `https://github.com/user/repo.git` → `github.com`
/// - `https://user@github.com/user/repo.git` → `github.com`
/// - `ssh://git@github.com:22/user/repo.git` → `github.com`
/// - `git@github.com:user/repo.git` (scp-like) → `github.com`
///
/// Returns `None` for unparseable URLs or local paths (`file://`, bare paths).
pub fn extract_git_host(url: &str) -> Option<&str> {
    let url = url.trim();
    // scp-like: user@host:path
    if let Some(at) = url.find('@') {
        let after_at = &url[at + 1..];
        if let Some(colon) = after_at.find(':') {
            if !url.contains("://") {
                let host = &after_at[..colon];
                if !host.is_empty() {
                    return Some(host);
                }
            }
        }
    }
    // scheme://[user@]host[:port]/path
    let after_scheme = url.split_once("://").map(|(_, rest)| rest)?;
    // strip userinfo
    let host_port = after_scheme.rsplit('@').next().unwrap_or(after_scheme);
    // strip path
    let host_port = host_scheme_host(host_port);
    // strip port
    let host = host_port.split(':').next().unwrap_or(host_port);
    if host.is_empty() || host == "file" {
        return None;
    }
    Some(host)
}

/// Take the leading host segment before the first `/`.
fn host_scheme_host(s: &str) -> &str {
    s.split('/').next().unwrap_or(s)
}

/// Credential manager for handling authentication.
///
/// Defaults to a **strict** security posture:
/// - [`CertificateConfig::Strict`] — certificates are validated by libgit2
///   against the system trust store; a cert that fails verification rejects
///   the fetch. The previous always-`CertificateOk` behavior is gone.
/// - [`AuthMode::ExplicitOnly`] — ambient credential discovery
///   (`AuthStrategy::Default`, i.e. the git credential helper, `~/.gitconfig`,
///   SSH agent fallback) is refused unless the caller explicitly opts in via
///   [`AuthManager::with_auth_mode`]`(AllowAmbient)`.
pub struct AuthManager {
    strategies: Vec<AuthStrategy>,
    certificate_mode: CertificateConfig,
    auth_mode: AuthMode,
}

impl AuthManager {
    /// Create a new authentication manager with no strategies, strict
    /// certificate validation, and `ExplicitOnly` auth mode.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            certificate_mode: CertificateConfig::Strict,
            auth_mode: AuthMode::ExplicitOnly,
        }
    }

    /// Create authentication manager with strategies. Certificate validation
    /// defaults to strict and auth mode to `ExplicitOnly`.
    pub fn with_strategies(strategies: Vec<AuthStrategy>) -> Self {
        Self {
            strategies,
            certificate_mode: CertificateConfig::Strict,
            auth_mode: AuthMode::ExplicitOnly,
        }
    }

    /// Set the certificate validation mode.
    pub fn with_certificate_mode(mut self, mode: CertificateConfig) -> Self {
        self.certificate_mode = mode;
        self
    }

    /// Set the authentication provenance mode.
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        self.auth_mode = mode;
        self
    }

    /// Add an authentication strategy
    pub fn add_strategy(&mut self, strategy: AuthStrategy) {
        self.strategies.push(strategy);
    }

    /// Create remote callbacks with authentication and certificate handling.
    ///
    /// Security posture:
    /// - Under `Strict` (default) the certificate callback returns
    ///   `CertificatePassthrough`, deferring to libgit2's built-in verification.
    ///   It never returns `CertificateOk` unconditionally, so an invalid or
    ///   untrusted certificate fails the fetch.
    /// - Under `ExplicitOnly` (default) `AuthStrategy::Default` is skipped, so
    ///   ambient credential sources are not consulted.
    pub fn create_callbacks(&self) -> RemoteCallbacks<'_> {
        let mut callbacks = RemoteCallbacks::new();

        let auth_mode = self.auth_mode;
        callbacks.credentials(move |url, username_from_url, allowed_types| {
            self.handle_credentials(auth_mode, url, username_from_url, allowed_types)
        });

        let cert_mode = self.certificate_mode.clone();
        callbacks.certificate_check(move |cert, _host| Self::handle_certificate(&cert_mode, cert));

        callbacks
    }

    /// Handle credential requests.
    fn handle_credentials(
        &self,
        auth_mode: AuthMode,
        url: &str,
        username_from_url: Option<&str>,
        allowed_types: CredentialType,
    ) -> Result<Cred, git2::Error> {
        info!("Attempting authentication for URL: {}", url);
        info!("Username from URL: {:?}", username_from_url);
        info!("Allowed credential types: {:?}", allowed_types);

        // Try each strategy in order. Under ExplicitOnly, ambient strategies
        // (AuthStrategy::Default → git credential helper / ~/.gitconfig;
        // AuthStrategy::SshAgent → ssh-agent loaded keys) are refused so an
        // untrusted fetch cannot silently pick up the operator's credentials.
        for strategy in &self.strategies {
            if matches!(auth_mode, AuthMode::ExplicitOnly) && strategy.is_ambient() {
                debug!(
                    "Refusing ambient strategy {:?} for {url} under ExplicitOnly auth mode",
                    strategy
                );
                continue;
            }
            match self.try_strategy(strategy, url, username_from_url, allowed_types) {
                Ok(cred) => {
                    info!("Authentication successful with strategy: {:?}", strategy);
                    return Ok(cred);
                }
                Err(e) => {
                    debug!("Authentication failed with strategy {:?}: {}", strategy, e);
                    continue;
                }
            }
        }

        warn!("All authentication strategies failed for {}", url);
        if matches!(auth_mode, AuthMode::ExplicitOnly) {
            Err(git2::Error::from_str(
                "No suitable explicit authentication method (ambient discovery — credential \
                 helper, ~/.gitconfig, SSH agent — refused under ExplicitOnly auth mode)",
            ))
        } else {
            Err(git2::Error::from_str("No suitable authentication method"))
        }
    }

    /// Certificate check shared between the sync and send-safe callback paths.
    /// Delegates to [`crate::callback_config::CallbackConfig::certificate_decision`]
    /// so the two paths cannot drift on the security boundary.
    fn handle_certificate(
        mode: &CertificateConfig,
        cert: &git2::cert::Cert<'_>,
    ) -> Result<git2::CertificateCheckStatus, git2::Error> {
        let hostkey = cert.as_hostkey().and_then(|hk| {
            hk.hostkey().map(|bytes| {
                let host = std::str::from_utf8(bytes).unwrap_or("");
                (host, bytes)
            })
        });
        let status = crate::callback_config::CallbackConfig::certificate_decision(mode, hostkey)?;
        if matches!(mode, CertificateConfig::AcceptAll) {
            debug!(
                "Accepting certificate unconditionally (AcceptAll mode): hostkey={:?}",
                cert.as_hostkey().map(|h| h.hostkey())
            );
        }
        Ok(status)
    }

    /// Try a specific authentication strategy
    fn try_strategy(
        &self,
        strategy: &AuthStrategy,
        url: &str,
        username_from_url: Option<&str>,
        allowed_types: CredentialType,
    ) -> Result<Cred, git2::Error> {
        match strategy {
            AuthStrategy::SshAgent { username } => {
                if allowed_types.contains(CredentialType::SSH_KEY) {
                    let user = username.as_deref().or(username_from_url).unwrap_or("git");

                    info!("Trying SSH key from agent for user: {}", user);
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
                    info!("Trying SSH key file: {:?}", private_key);
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
                    info!("Trying username/password authentication");
                    Cred::userpass_plaintext(username, password)
                } else {
                    Err(git2::Error::from_str("Username/password not allowed"))
                }
            }

            AuthStrategy::Token { token, host } => {
                if !allowed_types.contains(CredentialType::USER_PASS_PLAINTEXT) {
                    return Err(git2::Error::from_str("Token authentication not allowed"));
                }
                // Enforce host scope: a token bound to an exact origin host is
                // only offered when the request URL's host matches. This closes
                // credential exfiltration via caller-selected remotes (issue
                // #1429 Sol P1): a process-global token configured for a
                // trusted forge is never sent to an unrelated host.
                if let Some(ref bound) = host {
                    if let Some(req_host) = extract_git_host(url) {
                        if req_host != bound.as_str() {
                            debug!(
                                "Refusing host-scoped token for {url}: bound to {bound:?}, \
                                 request host is {req_host:?}"
                            );
                            return Err(git2::Error::from_str(
                                "Token is scoped to a different host",
                            ));
                        }
                    } else {
                        debug!("Refusing host-scoped token for {url}: could not parse host");
                        return Err(git2::Error::from_str(
                            "Token is host-scoped but request URL host is unparseable",
                        ));
                    }
                }
                info!("Trying token authentication");
                // For GitHub and similar services, use token as password with empty username
                Cred::userpass_plaintext("", token)
            }

            AuthStrategy::Default => {
                if allowed_types.contains(CredentialType::DEFAULT) {
                    info!("Trying default credentials");
                    Cred::default()
                } else {
                    Err(git2::Error::from_str("Default credentials not allowed"))
                }
            }
        }
    }

    /// Get the number of configured strategies (for testing)
    pub fn strategy_count(&self) -> usize {
        self.strategies.len()
    }

    /// Get the configured certificate validation mode.
    pub fn certificate_mode(&self) -> CertificateConfig {
        self.certificate_mode.clone()
    }

    /// Get the configured authentication provenance mode.
    pub fn auth_mode(&self) -> AuthMode {
        self.auth_mode
    }

    /// Get the strategies (for testing)
    #[cfg(test)]
    pub fn strategies(&self) -> &[AuthStrategy] {
        &self.strategies
    }
}

impl Default for AuthManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating authentication strategies
pub struct AuthBuilder {
    strategies: Vec<AuthStrategy>,
}

impl AuthBuilder {
    /// Create a new auth builder
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add SSH agent authentication
    pub fn ssh_agent(mut self, username: Option<String>) -> Self {
        self.strategies.push(AuthStrategy::SshAgent { username });
        self
    }

    /// Add SSH key file authentication
    pub fn ssh_key<U, P>(
        mut self,
        username: U,
        public_key: Option<PathBuf>,
        private_key: P,
        passphrase: Option<String>,
    ) -> Self
    where
        U: Into<String>,
        P: Into<PathBuf>,
    {
        self.strategies.push(AuthStrategy::SshKey {
            username: username.into(),
            public_key,
            private_key: private_key.into(),
            passphrase,
        });
        self
    }

    /// Add username/password authentication
    pub fn userpass<U, P>(mut self, username: U, password: P) -> Self
    where
        U: Into<String>,
        P: Into<String>,
    {
        self.strategies.push(AuthStrategy::UserPass {
            username: username.into(),
            password: password.into(),
        });
        self
    }

    /// Add token authentication (host-unscoped — use `token_scoped` to bind
    /// the token to an exact origin host).
    pub fn token<T>(mut self, token: T) -> Self
    where
        T: Into<String>,
    {
        self.strategies.push(AuthStrategy::Token {
            token: token.into(),
            host: None,
        });
        self
    }

    /// Add token authentication bound to an exact origin host. The token is
    /// only offered when the request URL's host matches `host`.
    pub fn token_scoped<T, H>(mut self, token: T, host: H) -> Self
    where
        T: Into<String>,
        H: Into<String>,
    {
        self.strategies.push(AuthStrategy::Token {
            token: token.into(),
            host: Some(host.into()),
        });
        self
    }

    /// Add default credentials (fallback). Maps to libgit2's ambient credential
    /// discovery (git credential helper, `~/.gitconfig`, SSH agent fallback).
    /// Only honored when the `AuthManager` is built with
    /// [`AuthMode::AllowAmbient`]; under the default `ExplicitOnly` mode this
    /// strategy is refused at resolution time.
    pub fn default_fallback(mut self) -> Self {
        self.strategies.push(AuthStrategy::Default);
        self
    }

    /// Build the authentication manager (strict certs, `ExplicitOnly` auth).
    pub fn build(self) -> AuthManager {
        AuthManager::with_strategies(self.strategies)
    }
}

impl Default for AuthBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create common authentication configurations.
///
/// Presets that include ambient discovery (`AuthStrategy::Default`) opt into
/// [`AuthMode::AllowAmbient`] explicitly, since their purpose is operator-style
/// access to first-party/trusted remotes. The strict certificate default is
/// preserved in every preset.
pub mod presets {
    use super::*;

    /// Standard SSH configuration (agent + ambient fallback). Opts into
    /// `AllowAmbient` because the ambient fallback is the explicit intent.
    pub fn ssh_standard() -> AuthManager {
        AuthBuilder::new()
            .ssh_agent(Some("git".to_owned()))
            .default_fallback()
            .build()
            .with_auth_mode(AuthMode::AllowAmbient)
    }

    /// GitHub personal access token with SSH-agent and ambient fallback.
    pub fn github_token<T: Into<String>>(token: T) -> AuthManager {
        AuthBuilder::new()
            .token(token)
            .ssh_agent(Some("git".to_owned()))
            .default_fallback()
            .build()
            .with_auth_mode(AuthMode::AllowAmbient)
    }

    /// SSH key file authentication with agent and ambient fallback.
    pub fn ssh_key_file<U, P>(
        username: U,
        private_key: P,
        passphrase: Option<String>,
    ) -> AuthManager
    where
        U: Into<String>,
        P: Into<PathBuf>,
    {
        AuthBuilder::new()
            .ssh_key(username, None, private_key, passphrase)
            .ssh_agent(Some("git".to_owned()))
            .default_fallback()
            .build()
            .with_auth_mode(AuthMode::AllowAmbient)
    }

    /// Public repository access via ambient discovery only. Opts into
    /// `AllowAmbient`.
    pub fn public_only() -> AuthManager {
        AuthBuilder::new()
            .default_fallback()
            .build()
            .with_auth_mode(AuthMode::AllowAmbient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_builder() {
        let auth = AuthBuilder::new()
            .ssh_agent(Some("git".to_owned()))
            .token("ghp_test_token")
            .default_fallback()
            .build();

        assert_eq!(auth.strategies.len(), 3);
    }

    #[test]
    fn test_presets() {
        let ssh_auth = presets::ssh_standard();
        assert_eq!(ssh_auth.strategies.len(), 2);

        let github_auth = presets::github_token("test_token");
        assert_eq!(github_auth.strategies.len(), 3);

        let public_auth = presets::public_only();
        assert_eq!(public_auth.strategies.len(), 1);
    }

    #[test]
    fn auth_manager_builder_chains_modes() {
        let mgr = AuthManager::with_strategies(vec![AuthStrategy::Token {
            token: "t".into(),
            host: None,
        }])
        .with_certificate_mode(CertificateConfig::AcceptAll)
        .with_auth_mode(AuthMode::AllowAmbient);
        assert!(matches!(
            mgr.certificate_mode(),
            CertificateConfig::AcceptAll
        ));
        assert_eq!(mgr.auth_mode(), AuthMode::AllowAmbient);
    }

    /// The credentials callback under ExplicitOnly must refuse the Default
    /// strategy even though it is present in the strategy list (ambient
    /// discovery never consulted). Mirrors the callback_config unit test but
    /// exercises the AuthManager (sync) path.
    #[test]
    fn auth_manager_explicit_only_refuses_default() {
        let mgr = AuthManager::with_strategies(vec![AuthStrategy::Default]);
        // allowed_types includes DEFAULT, so the only reason to fail is the
        // ExplicitOnly refusal of the ambient Default strategy.
        let res = mgr.handle_credentials(
            AuthMode::ExplicitOnly,
            "https://example.com/repo.git",
            None,
            CredentialType::DEFAULT,
        );
        assert!(res.is_err());
    }

    #[test]
    fn extract_git_host_parses_common_url_forms() {
        assert_eq!(
            extract_git_host("https://github.com/user/repo.git"),
            Some("github.com")
        );
        assert_eq!(
            extract_git_host("https://user@github.com/user/repo.git"),
            Some("github.com")
        );
        assert_eq!(
            extract_git_host("ssh://git@github.com:22/user/repo.git"),
            Some("github.com")
        );
        assert_eq!(
            extract_git_host("git@github.com:user/repo.git"),
            Some("github.com")
        );
        assert_eq!(
            extract_git_host("https://huggingface.co/Qwen/Qwen3"),
            Some("huggingface.co")
        );
    }

    #[test]
    fn extract_git_host_rejects_local_and_unparseable() {
        assert_eq!(extract_git_host("file:///path/to/repo"), None);
        assert_eq!(extract_git_host("/local/path"), None);
        assert_eq!(extract_git_host("not-a-url"), None);
    }
}
