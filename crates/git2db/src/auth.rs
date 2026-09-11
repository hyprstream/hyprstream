//! Authentication and credential management
//!
//! Consolidated authentication patterns from the original codebase.

use crate::callback_config::{AuthMode, CertificateConfig, RedirectPolicy};
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

/// Git remote schemes accepted for authority-based credential scoping.
///
/// Anything else (`file`, `data`, an unrecognized custom scheme, ...) is
/// rejected — [`extract_git_authority`] returns `None` and the caller fails
/// closed rather than guessing an authority for a scheme it does not
/// understand.
const ALLOWED_SCHEMES: &[&str] = &["https", "http", "ssh", "git"];

/// Extract the **canonical authority** from a git remote URL for host-scoping.
///
/// The authority is the `host[:port]` portion of the URL, canonicalized per
/// the WHATWG URL Standard's host-parsing algorithm (via the `url` crate):
/// - Domain names are IDNA-processed and ASCII case-folded.
/// - IPv4 addresses (including non-canonical forms like octal/hex octets)
///   are normalized to canonical dotted-decimal.
/// - IPv6 literals are normalized and returned in bracketed form
///   (`[2001:db8::cafe]`).
/// - Default scheme ports are stripped (`:443`/`:80` for HTTPS/HTTP, `:22`
///   for SSH, `:9418` for the git protocol); any other port is preserved.
///
/// **Security-critical / standards-based:** the URL is parsed structurally
/// by [`url::Url::parse`] — the authority (scheme, userinfo, host, port) is
/// separated from the path, query, and fragment by the same state machine
/// browsers use, so an `@`, `/`, `?`, or `#` appearing in the path or query
/// can never be misread as part of the authority. This closes the Sol P1
/// path/query `@` confusion finding (#1429), which affected the previous
/// hand-written parser.
///
/// `url::Url::parse` only applies IDNA/case canonicalization to the host
/// automatically for its built-in "special" schemes (`http`/`https`); `ssh`
/// and `git` are not "special" to the `url` crate, so their host is
/// re-canonicalized explicitly via [`url::Host::parse`] — the same
/// standards-based host algorithm, applied uniformly regardless of scheme.
///
/// **Backslash is rejected outright (fail closed), not parsed.** The WHATWG
/// URL Standard treats `\` as equivalent to `/` within a "special" scheme
/// (`http`/`https`), ending the authority early — but the pinned libgit2
/// 1.9.x C parser treats `\` as an ordinary authority character and keeps
/// splitting userinfo at the last `@`. The same input string is therefore
/// assigned a *different* authority by this function than by the transport
/// that actually opens the connection: `https://github.com\@evil.example/x`
/// resolves to `github.com` here but libgit2 connects to `evil.example`.
/// Any input containing `\` is a parser-differential attack surface, not a
/// case this function can resolve by parsing more cleverly — it returns
/// `None` unconditionally rather than risk agreeing with the wrong parser.
/// See #1429 Kimi-K3 r3 P1.
///
/// Returns `None` for local paths (`file://`, bare paths), malformed URLs,
/// URLs using a scheme outside [`ALLOWED_SCHEMES`], URLs containing a
/// backslash, or URLs where the authority cannot be reliably determined.
/// Callers must treat `None` as "authority unknown" and refuse the
/// credential, not fall back to an unscoped release.
///
/// # Examples
/// ```text
/// https://github.com/user/repo.git        → "github.com"
/// https://USER@github.com/user/repo.git   → "github.com"
/// https://github.com:443/user/repo.git    → "github.com"   (default port stripped)
/// https://github.com:8443/user/repo.git   → "github.com:8443"
/// https://evil.example/repo@github.com    → "evil.example" (path @ ignored)
/// ssh://git@github.com:22/user/repo.git   → "github.com"
/// git@github.com:user/repo.git            → "github.com"   (scp-like)
/// https://[2001:db8::cafe]/repo.git       → "[2001:db8::cafe]"
/// https://github.com\@evil.example/x.git  → None            (backslash rejected)
/// ```
pub fn extract_git_authority(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Fail closed on any backslash before doing any other parsing — see the
    // doc comment above for the exact WHATWG-vs-libgit2 differential this
    // closes. This must run before the scp-like/URL branch split and before
    // `url::Url::parse`, since the differential exists regardless of which
    // branch would otherwise handle the input.
    if trimmed.contains('\\') {
        return None;
    }

    if !trimmed.contains("://") {
        return extract_scp_like_authority(trimmed);
    }

    let parsed = url::Url::parse(trimmed).ok()?;
    let scheme = parsed.scheme();
    if !ALLOWED_SCHEMES.contains(&scheme) {
        return None;
    }
    // Reject cannot-be-a-base URLs (e.g. `mailto:`-shaped, no authority at
    // all) up front — a scheme in ALLOWED_SCHEMES combined with `://` in the
    // input always parses to a base URL, but guard explicitly rather than
    // relying on that invariant.
    if parsed.cannot_be_a_base() {
        return None;
    }
    let raw_host = parsed.host_str()?;

    // Re-run the raw host through the standards host-parsing algorithm
    // unconditionally (IDNA + case-fold for domains, canonical IPv4, bracketed
    // IPv6). `Url::parse` already did this for "special" schemes (http/https);
    // for "non-special" git schemes (ssh/git) it left the host as an opaque,
    // uncanonicalized string, so this step is required for those too.
    let canonical_host = url::Host::parse(raw_host).ok()?;
    let host_display = canonical_host.to_string();

    match parsed.port() {
        Some(port) if !is_default_port(scheme, port) => Some(format!("{host_display}:{port}")),
        _ => Some(host_display),
    }
}

/// Extract the canonical authority from git's scp-like remote syntax:
/// `[user@]host:path` (no `scheme://`, e.g. `git@github.com:owner/repo.git`).
/// This form has no port; a colon always introduces the path, never a port
/// (`git@host:22/path` means path `22/path`, matching real git behavior).
///
/// The host is bounded by the **last** `@` before the first `:` — mirroring
/// the WHATWG URL Standard's userinfo/host boundary (userinfo is everything
/// up to the last `@`) — so an embedded `@` in a spoofed userinfo segment
/// (`user@name@host:path`) cannot smuggle `name@host` in as the host.
fn extract_scp_like_authority(input: &str) -> Option<String> {
    let colon = input.find(':')?;
    let before_colon = &input[..colon];
    // A `/` before the first `:` means this isn't scp-like syntax at all
    // (e.g. a bare local path containing a colon) — reject rather than guess.
    if before_colon.contains('/') {
        return None;
    }
    let at = before_colon.rfind('@')?;
    let host = &before_colon[at + 1..];
    if host.is_empty() {
        return None;
    }
    let canonical_host = url::Host::parse(host).ok()?;
    Some(canonical_host.to_string())
}

/// Whether a port is the default for the given scheme.
fn is_default_port(scheme: &str, port: u16) -> bool {
    match scheme {
        "https" => port == 443,
        "http" => port == 80,
        "ssh" => port == 22,
        "git" => port == 9418,
        _ => false,
    }
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
/// - [`RedirectPolicy::None`] — callers that build their own `git2::FetchOptions`
///   from this manager's [`AuthManager::create_callbacks`] must also apply
///   [`AuthManager::redirect_policy`] via [`RedirectPolicy::to_git2`] /
///   `FetchOptions::follow_redirects`. libgit2's own default
///   (`RemoteRedirect::Initial`) allows an off-site redirect on the initial
///   request, and — per the pinned libgit2 1.9.x behavior — the credential
///   callback still receives the *original* request URL after that redirect,
///   not the effective peer. Combined with a host-scoped [`AuthStrategy::Token`],
///   that would let a credential bound to host A be offered to a redirect
///   target host B. See issue #1429 Sol P1.
pub struct AuthManager {
    strategies: Vec<AuthStrategy>,
    certificate_mode: CertificateConfig,
    auth_mode: AuthMode,
    redirect_policy: RedirectPolicy,
}

impl AuthManager {
    /// Create a new authentication manager with no strategies, strict
    /// certificate validation, `ExplicitOnly` auth mode, and no off-site
    /// redirects.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            certificate_mode: CertificateConfig::Strict,
            auth_mode: AuthMode::ExplicitOnly,
            redirect_policy: RedirectPolicy::None,
        }
    }

    /// Create authentication manager with strategies. Certificate validation
    /// defaults to strict, auth mode to `ExplicitOnly`, and redirect policy to
    /// no off-site redirects.
    pub fn with_strategies(strategies: Vec<AuthStrategy>) -> Self {
        Self {
            strategies,
            certificate_mode: CertificateConfig::Strict,
            auth_mode: AuthMode::ExplicitOnly,
            redirect_policy: RedirectPolicy::None,
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

    /// Set the off-site redirect policy. Defaults to [`RedirectPolicy::None`].
    /// Callers driving a real `git2::FetchOptions` from this manager must
    /// apply the result of [`AuthManager::redirect_policy`] via
    /// `follow_redirects` — this struct only carries the *decision*; it does
    /// not own a `FetchOptions` to enforce it against (see
    /// [`crate::clone_options`] for the send-safe path that does).
    pub fn with_redirect_policy(mut self, policy: RedirectPolicy) -> Self {
        self.redirect_policy = policy;
        self
    }

    /// Get the configured off-site redirect policy.
    pub fn redirect_policy(&self) -> RedirectPolicy {
        self.redirect_policy
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
                // Enforce authority scope: a token bound to an exact origin
                // authority is only offered when the request URL's canonical
                // authority matches. The authority is extracted path-safely
                // (the path is separated before userinfo handling, so path/query
                // `@` characters cannot spoof the authority). See #1429 Sol P1.
                if let Some(bound) = host {
                    match extract_git_authority(url) {
                        Some(req_auth) if &req_auth == bound => {}
                        Some(req_auth) => {
                            debug!(
                                "Refusing authority-scoped token for {url}: bound to \
                                 {bound:?}, request authority is {req_auth:?}"
                            );
                            return Err(git2::Error::from_str(
                                "Token is scoped to a different authority",
                            ));
                        }
                        None => {
                            debug!(
                                "Refusing authority-scoped token for {url}: could not parse \
                                 authority"
                            );
                            return Err(git2::Error::from_str(
                                "Token is authority-scoped but request URL is unparseable",
                            ));
                        }
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

    /// Add token authentication **without** an authority binding.
    ///
    /// **Warning:** an unscoped token is offered to ANY remote that challenges
    /// for `USER_PASS_PLAINTEXT`. This is an explicit trusted-caller opt-in —
    /// callers must understand the credential-exfiltration implications.
    /// Prefer [`AuthBuilder::token_scoped`] which binds the token to an exact
    /// origin authority.
    pub fn token_unscoped<T>(mut self, token: T) -> Self
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

    /// GitHub personal access token scoped to the canonical `github.com`
    /// authority, with SSH-agent and ambient fallback.
    pub fn github_token<T: Into<String>>(token: T) -> AuthManager {
        AuthBuilder::new()
            .token_scoped(token, "github.com")
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
            .token_unscoped("ghp_test_token")
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
    fn extract_git_authority_parses_common_url_forms() {
        assert_eq!(
            extract_git_authority("https://github.com/user/repo.git"),
            Some("github.com".to_owned())
        );
        assert_eq!(
            extract_git_authority("https://user@github.com/user/repo.git"),
            Some("github.com".to_owned())
        );
        assert_eq!(
            extract_git_authority("ssh://git@github.com:22/user/repo.git"),
            Some("github.com".to_owned()) // default SSH port stripped
        );
        assert_eq!(
            extract_git_authority("git@github.com:user/repo.git"),
            Some("github.com".to_owned())
        );
        assert_eq!(
            extract_git_authority("https://huggingface.co/Qwen/Qwen3"),
            Some("huggingface.co".to_owned())
        );
    }

    #[test]
    fn extract_git_authority_rejects_local_and_unparseable() {
        assert_eq!(extract_git_authority("file:///path/to/repo"), None);
        assert_eq!(extract_git_authority("/local/path"), None);
        assert_eq!(extract_git_authority("not-a-url"), None);
    }

    #[test]
    fn extract_git_authority_path_at_does_not_spoof() {
        // The critical P1 fix: @ in the path must NOT be treated as userinfo.
        assert_eq!(
            extract_git_authority("https://evil.example/repo@github.com/owned.git"),
            Some("evil.example".to_owned())
        );
        assert_eq!(
            extract_git_authority("https://evil.example/repo?x=@github.com"),
            Some("evil.example".to_owned())
        );
    }

    #[test]
    fn extract_git_authority_normalizes_case_and_ports() {
        assert_eq!(
            extract_git_authority("https://GitHub.COM/user/repo.git"),
            Some("github.com".to_owned()) // case normalized
        );
        assert_eq!(
            extract_git_authority("https://github.com:443/user/repo.git"),
            Some("github.com".to_owned()) // default HTTPS port stripped
        );
        assert_eq!(
            extract_git_authority("https://github.com:8443/user/repo.git"),
            Some("github.com:8443".to_owned()) // non-default port preserved
        );
    }

    #[test]
    fn extract_git_authority_handles_ipv6() {
        assert_eq!(
            extract_git_authority("https://[2001:db8::1]/repo.git"),
            Some("[2001:db8::1]".to_owned())
        );
        assert_eq!(
            extract_git_authority("https://[2001:db8::1]:8443/repo.git"),
            Some("[2001:db8::1]:8443".to_owned())
        );
    }

    /// IPv6 authority is also canonicalized for the non-"special" `ssh`
    /// scheme, where `url::Url::parse` does not canonicalize automatically —
    /// proving the explicit `Host::parse` re-canonicalization step is load-bearing.
    #[test]
    fn extract_git_authority_handles_ipv6_over_ssh() {
        assert_eq!(
            extract_git_authority("ssh://git@[2001:DB8::1]:22/repo.git"),
            Some("[2001:db8::1]".to_owned()) // default SSH port stripped, hex lowercased
        );
    }

    /// IPv4 literals are canonicalized to dotted-decimal, including
    /// non-canonical numeric forms (octal/hex octets) that could otherwise be
    /// used to make two textually different authorities resolve to the same
    /// address and evade a naive string comparison.
    #[test]
    fn extract_git_authority_canonicalizes_ipv4() {
        assert_eq!(
            extract_git_authority("https://192.168.1.1/repo.git"),
            Some("192.168.1.1".to_owned())
        );
        // Octal-looking octet (leading zero) canonicalizes to the same
        // decimal address per the WHATWG IPv4 parser.
        assert_eq!(
            extract_git_authority("https://192.168.1.001/repo.git"),
            Some("192.168.1.1".to_owned())
        );
        assert_eq!(
            extract_git_authority("git@192.168.1.1:owner/repo.git"),
            Some("192.168.1.1".to_owned())
        );
    }

    /// A port that is the ssh/git default but NOT the https/http default (or
    /// vice versa) must be scoped per the URL's own scheme, never treated as
    /// "default" merely because it matches a *different* scheme's default.
    /// This specifically covers GitHub's real-world "SSH over port 443"
    /// endpoint, which must not collide with the https:443 default.
    #[test]
    fn extract_git_authority_scopes_default_port_per_scheme() {
        assert_eq!(
            extract_git_authority("ssh://git@ssh.github.com:443/owner/repo.git"),
            Some("ssh.github.com:443".to_owned()), // 443 is NOT the ssh default
        );
        assert_eq!(
            extract_git_authority("ssh://git@github.com:22/owner/repo.git"),
            Some("github.com".to_owned()), // 22 IS the ssh default
        );
        assert_eq!(
            extract_git_authority("git://github.com:9418/owner/repo.git"),
            Some("github.com".to_owned()), // 9418 is the git:// default
        );
        assert_eq!(
            extract_git_authority("git://github.com:9419/owner/repo.git"),
            Some("github.com:9419".to_owned()),
        );
    }

    /// IDNA: two different Unicode case/normalization variants of the same
    /// domain must canonicalize to the identical authority string (and to
    /// ASCII punycode, not raw Unicode), so a lookalike-case Unicode domain
    /// cannot bypass a string-equality host-scope check.
    #[test]
    fn extract_git_authority_normalizes_idna() {
        let lower = extract_git_authority("https://münchen.example/repo.git");
        let upper = extract_git_authority("https://MÜNCHEN.example/repo.git");
        assert_eq!(lower, upper, "IDNA case folding must be applied uniformly");
        assert!(
            matches!(&lower, Some(host) if host.starts_with("xn--")),
            "domain must be canonicalized to ASCII punycode, got {lower:?}",
        );

        // Same requirement over the non-"special" ssh scheme, where
        // `url::Url::parse` alone would NOT apply IDNA — proving the explicit
        // re-canonicalization step covers this scheme too.
        let ssh_lower = extract_git_authority("ssh://git@münchen.example/repo.git");
        let ssh_upper = extract_git_authority("ssh://git@MÜNCHEN.example/repo.git");
        assert!(ssh_lower.is_some());
        assert_eq!(ssh_lower, ssh_upper);
        assert_eq!(lower, ssh_lower, "https and ssh must canonicalize identically");
    }

    /// Malformed authorities are rejected outright (`None`) rather than
    /// silently truncated or misparsed into something that could
    /// accidentally match a configured bound host.
    #[test]
    fn extract_git_authority_rejects_malformed() {
        // Non-numeric port.
        assert_eq!(
            extract_git_authority("https://github.com:notaport/repo.git"),
            None
        );
        // Unterminated IPv6 bracket.
        assert_eq!(extract_git_authority("https://[2001:db8::1/repo.git"), None);
        // Empty host: for a "non-special" git scheme (ssh/git) three slashes
        // after the scheme yield a genuinely empty authority (unlike https,
        // where WHATWG's "special authority ignore slashes" leniency would
        // otherwise make the following path segment look like a host).
        assert_eq!(extract_git_authority("ssh:///repo.git"), None);
        assert_eq!(extract_git_authority("git:///repo.git"), None);
        // Empty string / whitespace only.
        assert_eq!(extract_git_authority(""), None);
        assert_eq!(extract_git_authority("   "), None);
    }

    /// A scheme outside the git-relevant allowlist is rejected even though
    /// `url::Url::parse` would happily parse it — an unrecognized scheme must
    /// fail closed rather than be guessed to share an authority scope.
    #[test]
    fn extract_git_authority_rejects_disallowed_scheme() {
        assert_eq!(extract_git_authority("ftp://github.com/repo.git"), None);
        assert_eq!(
            extract_git_authority("javascript://github.com/repo.git"),
            None
        );
        assert_eq!(
            extract_git_authority("mailto:git@github.com"),
            None,
            "cannot-be-a-base URLs must never resolve to an authority"
        );
    }

    /// Fragment `#` must terminate the authority exactly like `/` and `?` —
    /// an `@` after `#` must not be read as authority userinfo.
    #[test]
    fn extract_git_authority_fragment_at_does_not_spoof() {
        assert_eq!(
            extract_git_authority("https://evil.example/repo.git#@github.com"),
            Some("evil.example".to_owned())
        );
    }

    /// The classic human-misreading authority-confusion form: a full hostname
    /// used as *userinfo* immediately before the real (attacker) host. Unlike
    /// the path/query cases above, this is not a parser bug — the real
    /// network authority genuinely is `evil.example`, and the parser must
    /// report that faithfully so the caller's scope check refuses it.
    #[test]
    fn extract_git_authority_userinfo_lookalike_resolves_to_real_host() {
        assert_eq!(
            extract_git_authority("https://github.com@evil.example/repo.git"),
            Some("evil.example".to_owned())
        );
    }

    /// scp-like syntax: an embedded `@` within what looks like userinfo must
    /// not smuggle a trailing `@host` segment in as the host — the host is
    /// bounded by the LAST `@` before the first `:`, mirroring URL userinfo
    /// semantics.
    #[test]
    fn extract_scp_like_authority_rejects_embedded_at_ambiguity() {
        assert_eq!(
            extract_git_authority("user@name@github.com:owner/repo.git"),
            Some("github.com".to_owned())
        );
    }

    /// scp-like syntax requires an explicit `user@`; a bare `host:path` with
    /// no `@` and no `://` is not accepted (fails closed rather than
    /// guessing this is scp-like syntax vs. an unrelated colon-bearing path).
    #[test]
    fn extract_scp_like_authority_requires_at_sign() {
        assert_eq!(extract_git_authority("github.com:owner/repo.git"), None);
    }

    // ---- Backslash: WHATWG-vs-libgit2 authority differential (#1429 r3 P1) ----

    /// The exact differential the r3 review causally proved against the real
    /// pinned libgit2 stack: for a "special" scheme, WHATWG (`url::Url::parse`)
    /// treats `\` as ending the authority early, resolving to `github.com` —
    /// but libgit2's own C parser treats `\` as an ordinary authority
    /// character and splits userinfo at the last `@`, resolving to
    /// `evil.example`. Because the two parsers disagree on what the
    /// authority even is, the function must refuse to answer at all.
    #[test]
    fn extract_git_authority_rejects_backslash_differential() {
        assert_eq!(
            extract_git_authority("https://github.com\\@evil.example/repo.git"),
            None,
            "a backslash-containing authority is a WHATWG/libgit2 parser \
             differential and must fail closed, not resolve to either parser's answer",
        );
    }

    /// Backslash is rejected regardless of scheme or position — this is not
    /// specific to the `https` special-scheme case above.
    #[test]
    fn extract_git_authority_rejects_backslash_anywhere() {
        assert_eq!(extract_git_authority("ssh://git@github.com\\evil/x"), None);
        assert_eq!(extract_git_authority("https://evil.example/\\@github.com"), None);
        assert_eq!(extract_git_authority("http://github.com\\.evil.example/x"), None);
    }

    /// scp-like syntax is covered by the same top-level guard: a backslash
    /// anywhere in scp-like input is rejected before the `@`/`:` boundary
    /// logic ever runs.
    #[test]
    fn extract_git_authority_rejects_backslash_in_scp_like() {
        assert_eq!(
            extract_git_authority("git@github.com\\evil.example:owner/repo.git"),
            None
        );
    }
}
