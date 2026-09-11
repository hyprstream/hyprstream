//! Clone operation options
//!
//! This module provides a Send-safe way to configure git clone operations.
//! Instead of holding non-Send callbacks directly, we use a configuration approach.
//!
//! # Example
//!
//! ```rust,ignore
//! use git2db::clone_options::CloneOptions;
//! use git2db::callback_config::{CallbackConfigBuilder, ProgressConfig};
//! use git2db::auth::AuthStrategy;
//!
//! let options = CloneOptions::builder()
//!     .callback_config(
//!         CallbackConfigBuilder::new()
//!             .auth(AuthStrategy::SshAgent { username: Some("git".to_owned()) })
//!             .progress(ProgressConfig::Stdout)
//!             .build()
//!     )
//!     .shallow(true)
//!     .depth(1)
//!     .branch("main")
//!     .build();
//! ```

use crate::callback_config::{AuthMode, CallbackConfig};
use git2::{FetchOptions, RemoteCallbacks, build::CheckoutBuilder};

/// Trust posture for a clone/fetch operation.
///
/// `Untrusted` is the mode required for any repository whose remote or
/// contents are not first-party-controlled — concretely, a PR head fetched by
/// a merge gate. Selecting it clamps three independently-dangerous properties
/// together (issue #1430), rather than requiring the caller to remember all
/// three every time:
///
/// - [`SubmoduleMode`] is forced to [`SubmoduleMode::Disabled`] regardless of
///   what the caller configured — `.gitmodules` is attacker-controlled
///   content, and initializing a declared submodule means fetching from an
///   arbitrary attacker-chosen remote URL.
/// - [`FilterMode`] is forced to [`FilterMode::Passthrough`] regardless of
///   what the caller configured — a `.gitattributes`-declared content filter
///   (XET/LFS smudge) would resolve attacker-controlled pointer content
///   against a real endpoint during checkout.
/// - Auth is required to be [`AuthMode::ExplicitOnly`] (already the
///   `CallbackConfig`/`AuthManager` default per B2/#1429) with no unscoped
///   (`host: None`) [`crate::auth::AuthStrategy::Token`] present — see
///   [`CloneOptions::validate_trust`].
///
/// This is deliberately the *only* notion of "untrusted" in `git2db` — it
/// composes the ambient-vs-explicit auth split B2 already established rather
/// than introducing a second, parallel trust concept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CloneTrust {
    /// First-party, operator-controlled repository. The caller-selected
    /// [`SubmoduleMode`], [`FilterMode`], and auth configuration are honored
    /// as configured — this is the default, matching today's behavior.
    #[default]
    Trusted,
    /// Arbitrary or attacker-influenced repository (PR heads, forge-facing
    /// fetches, anything the merge gate builds). See [`CloneTrust`] docs for
    /// exactly what this clamps.
    Untrusted,
}

/// Controls whether `.gitmodules`-declared submodules are initialized and
/// updated after a clone.
///
/// Replaces the previous plain `update_submodules: bool` field — folding
/// submodule handling into a named mode makes the untrusted clamp in
/// [`CloneTrust::Untrusted`] a single, unambiguous source of truth instead of
/// a bool the caller could independently flip back on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubmoduleMode {
    /// Never initialize or update submodules, even if the repository
    /// declares them via `.gitmodules`. Secure default — matches the
    /// pre-#1430 behavior, where submodule initialization was never wired
    /// into [`crate::manager::GitManager::clone_repository`] at all.
    #[default]
    Disabled,
    /// Initialize and update submodules after clone (equivalent to
    /// `git clone --recurse-submodules`). Only appropriate for trusted,
    /// first-party repositories, since a submodule's remote URL is taken
    /// verbatim from the repository's own `.gitmodules` content.
    Enabled,
}

/// Controls whether libgit2 content filters (XET/LFS smudge, and libgit2's
/// own built-in `ident`/`crlf` filters) are applied during checkout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Apply registered content filters normally during checkout — the
    /// default, matching today's behavior.
    #[default]
    Enabled,
    /// Disable all content filters during checkout
    /// (`git2::build::CheckoutBuilder::disable_filters(true)`), so no
    /// registered filter — XET/LFS smudge included — resolves checked-out
    /// file content against an external endpoint. Required for untrusted
    /// checkouts, where the `.gitattributes` filter declaration and the
    /// pointer content it would smudge are both attacker-controlled.
    Passthrough,
}

/// Send-safe options for cloning a repository
#[derive(Default, Clone)]
pub struct CloneOptions {
    /// Send-safe callback configuration
    pub callback_config: Option<CallbackConfig>,

    /// Whether to perform a shallow clone
    pub shallow: bool,

    /// Depth for shallow clones
    pub depth: Option<i32>,

    /// Branch to checkout
    pub branch: Option<String>,

    /// Custom refspecs
    pub refspecs: Vec<String>,

    /// Trust posture for this clone. See [`CloneTrust`].
    pub trust: CloneTrust,

    /// Submodule handling, subject to the [`CloneTrust::Untrusted`] clamp —
    /// see [`CloneOptions::effective_submodule_mode`].
    pub submodule_mode: SubmoduleMode,

    /// Content filter handling, subject to the [`CloneTrust::Untrusted`]
    /// clamp — see [`CloneOptions::effective_filter_mode`].
    pub filter_mode: FilterMode,

    /// Network proxy URL
    pub proxy_url: Option<String>,

    /// Timeout in seconds for network operations
    pub timeout_seconds: Option<u32>,
}

// Ensure CloneOptions is Send
#[allow(dead_code)]
const _: () = {
    fn assert_send<T: Send>() {}
    fn assert_clone_options_is_send() {
        assert_send::<CloneOptions>();
    }
};

impl CloneOptions {
    /// Create a new CloneOptions builder
    pub fn builder() -> CloneOptionsBuilder {
        CloneOptionsBuilder::new()
    }

    /// Create options with callback configuration
    pub fn with_callback_config(config: CallbackConfig) -> Self {
        Self {
            callback_config: Some(config),
            ..Default::default()
        }
    }

    /// The submodule mode actually in effect, after applying the
    /// [`CloneTrust::Untrusted`] clamp. Under `Untrusted` this is always
    /// [`SubmoduleMode::Disabled`], regardless of `self.submodule_mode` —
    /// this is the single source of truth callers and tests should read,
    /// rather than the raw field.
    pub fn effective_submodule_mode(&self) -> SubmoduleMode {
        match self.trust {
            CloneTrust::Untrusted => SubmoduleMode::Disabled,
            CloneTrust::Trusted => self.submodule_mode,
        }
    }

    /// The filter mode actually in effect, after applying the
    /// [`CloneTrust::Untrusted`] clamp. Under `Untrusted` this is always
    /// [`FilterMode::Passthrough`], regardless of `self.filter_mode`.
    pub fn effective_filter_mode(&self) -> FilterMode {
        match self.trust {
            CloneTrust::Untrusted => FilterMode::Passthrough,
            CloneTrust::Trusted => self.filter_mode,
        }
    }

    /// Validate that an [`CloneTrust::Untrusted`] clone cannot silently widen
    /// its own credential surface — checked independently of the submodule/
    /// filter clamps above, which cannot be bypassed by construction, but
    /// auth is a `Vec<AuthStrategy>` the caller assembles separately.
    ///
    /// Refuses:
    /// - [`AuthMode::AllowAmbient`] — untrusted fetches must stay
    ///   [`AuthMode::ExplicitOnly`] (credential helper / `~/.gitconfig` / SSH
    ///   agent must never be consulted for an attacker-influenced remote).
    /// - An unscoped [`crate::auth::AuthStrategy::Token`] (`host: None`) —
    ///   such a token is offered to *any* remote that challenges for it
    ///   (see `auth.rs`), which for an untrusted, caller-selected remote is a
    ///   credential-exfiltration vector. Untrusted clones must use either no
    ///   token or one bound to an exact origin authority.
    ///
    /// `Trusted` options always validate successfully — this check exists
    /// only to keep `Untrusted` honest.
    pub fn validate_trust(&self) -> Result<(), crate::errors::Git2DBError> {
        if self.trust != CloneTrust::Untrusted {
            return Ok(());
        }

        let Some(config) = self.callback_config.as_ref() else {
            return Ok(());
        };

        if config.auth_mode == AuthMode::AllowAmbient {
            return Err(crate::errors::Git2DBError::configuration(
                "CloneTrust::Untrusted requires AuthMode::ExplicitOnly; \
                 AllowAmbient would permit ambient credential discovery \
                 (credential helper, ~/.gitconfig, SSH agent) against an \
                 untrusted remote",
            ));
        }

        for strategy in &config.auth {
            if let crate::auth::AuthStrategy::Token { host: None, .. } = strategy {
                return Err(crate::errors::Git2DBError::configuration(
                    "CloneTrust::Untrusted refuses an unscoped Token (host: \
                     None): it would be offered to any remote, including an \
                     untrusted caller-selected one — bind it to an exact \
                     origin authority or omit it",
                ));
            }
        }

        Ok(())
    }

    /// Convert to legacy git2 options for use within spawn_blocking
    /// This is called inside spawn_blocking where lifetime constraints are satisfied
    pub(crate) fn to_git2_options(&self) -> LegacyCloneOptions<'_> {
        let mut options = LegacyCloneOptions {
            _shallow: self.shallow,
            depth: self.depth,
            branch: self.branch.clone(),
            _refspecs: self.refspecs.clone(),
            submodule_mode: self.effective_submodule_mode(),
            filter_mode: self.effective_filter_mode(),
            proxy_url: self.proxy_url.clone(),
            _timeout_seconds: self.timeout_seconds,
            ..Default::default()
        };

        // Create callbacks from config if present, and extract the redirect
        // policy so it can be applied to FetchOptions.
        if let Some(ref config) = self.callback_config {
            options.callbacks = Some(config.create_callbacks());
            options.redirect_policy = config.redirect_policy;
        }

        options
    }
}

/// Internal legacy options structure for git2 interop
/// Only used within spawn_blocking where non-Send is acceptable
#[derive(Default)]
pub(crate) struct LegacyCloneOptions<'cb> {
    pub callbacks: Option<RemoteCallbacks<'cb>>,
    pub _shallow: bool,
    pub depth: Option<i32>,
    pub branch: Option<String>,
    pub _refspecs: Vec<String>,
    /// Already resolved via [`CloneOptions::effective_submodule_mode`] — the
    /// [`CloneTrust`] clamp has already been applied by the time this is set.
    pub submodule_mode: SubmoduleMode,
    /// Already resolved via [`CloneOptions::effective_filter_mode`].
    pub filter_mode: FilterMode,
    pub proxy_url: Option<String>,
    pub _timeout_seconds: Option<u32>,
    /// Off-site redirect policy (default: None — no off-site redirects).
    pub redirect_policy: crate::callback_config::RedirectPolicy,
}

impl<'cb> LegacyCloneOptions<'cb> {
    /// Create FetchOptions from these clone options
    pub fn create_fetch_options(
        &mut self,
    ) -> Result<FetchOptions<'cb>, crate::errors::Git2DBError> {
        let mut fetch_opts = FetchOptions::new();

        // Apply callbacks if provided (move out of self)
        if let Some(callbacks) = self.callbacks.take() {
            fetch_opts.remote_callbacks(callbacks);
        }

        // Apply redirect policy: default is None (no off-site redirects) to
        // prevent credential exfiltration via redirect when host-scoped
        // credentials are present (Sol P1 #1429).
        fetch_opts.follow_redirects(self.redirect_policy.to_git2());

        // Apply proxy settings
        if let Some(proxy_url) = &self.proxy_url {
            let mut proxy_opts = git2::ProxyOptions::new();
            proxy_opts.url(proxy_url);
            fetch_opts.proxy_options(proxy_opts);
        }

        // Apply depth for shallow clones
        if let Some(depth) = self.depth {
            fetch_opts.depth(depth);
        }

        Ok(fetch_opts)
    }

    /// Create CheckoutBuilder for this clone.
    ///
    /// Under [`FilterMode::Passthrough`] this disables libgit2 content
    /// filters entirely (`GIT_CHECKOUT_DISABLE_FILTERS`) so no
    /// `.gitattributes`-registered filter — XET/LFS smudge included — runs
    /// during checkout. See issue #1430.
    pub fn create_checkout_builder(&self) -> CheckoutBuilder<'static> {
        let mut builder = CheckoutBuilder::new();
        if matches!(self.filter_mode, FilterMode::Passthrough) {
            builder.disable_filters(true);
        }
        builder
    }
}

/// Builder for CloneOptions with fluent interface
pub struct CloneOptionsBuilder {
    options: CloneOptions,
}

impl CloneOptionsBuilder {
    /// Create a new builder with default options
    pub fn new() -> Self {
        Self {
            options: CloneOptions::default(),
        }
    }

    /// Set callback configuration
    pub fn callback_config(mut self, config: CallbackConfig) -> Self {
        self.options.callback_config = Some(config);
        self
    }

    /// Enable shallow clone
    pub fn shallow(mut self, shallow: bool) -> Self {
        self.options.shallow = shallow;
        self
    }

    /// Set clone depth
    pub fn depth(mut self, depth: i32) -> Self {
        self.options.depth = Some(depth);
        self
    }

    /// Set branch to checkout
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.options.branch = Some(branch.into());
        self
    }

    /// Add a refspec
    pub fn refspec(mut self, refspec: impl Into<String>) -> Self {
        self.options.refspecs.push(refspec.into());
        self
    }

    /// Set the trust posture for this clone. See [`CloneTrust`] — selecting
    /// [`CloneTrust::Untrusted`] clamps submodule and filter handling
    /// regardless of what is configured below.
    pub fn trust(mut self, trust: CloneTrust) -> Self {
        self.options.trust = trust;
        self
    }

    /// Set the submodule handling mode. Subject to the [`CloneTrust`] clamp
    /// — see [`CloneOptions::effective_submodule_mode`].
    pub fn submodule_mode(mut self, mode: SubmoduleMode) -> Self {
        self.options.submodule_mode = mode;
        self
    }

    /// Set the content filter handling mode. Subject to the [`CloneTrust`]
    /// clamp — see [`CloneOptions::effective_filter_mode`].
    pub fn filter_mode(mut self, mode: FilterMode) -> Self {
        self.options.filter_mode = mode;
        self
    }

    /// Set proxy URL
    pub fn proxy_url(mut self, proxy_url: impl Into<String>) -> Self {
        self.options.proxy_url = Some(proxy_url.into());
        self
    }

    /// Set timeout in seconds
    pub fn timeout(mut self, seconds: u32) -> Self {
        self.options.timeout_seconds = Some(seconds);
        self
    }

    /// Build the CloneOptions
    pub fn build(self) -> CloneOptions {
        self.options
    }
}

impl Default for CloneOptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let options = CloneOptions::builder()
            .shallow(true)
            .depth(5)
            .branch("main")
            .timeout(60)
            .build();

        assert!(options.shallow);
        assert_eq!(options.depth, Some(5));
        assert_eq!(options.branch, Some("main".to_owned()));
        assert_eq!(options.timeout_seconds, Some(60));
    }

    #[test]
    fn test_with_callback_config() {
        use crate::callback_config::CallbackConfig;

        let config = CallbackConfig::new();
        let options = CloneOptions::with_callback_config(config);
        assert!(options.callback_config.is_some());
    }

    #[test]
    fn test_clone_options_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CloneOptions>();
    }

    // ---- CloneTrust clamp ----

    /// Defaults are `Trusted` + `SubmoduleMode::Disabled` +
    /// `FilterMode::Enabled` — matches pre-#1430 behavior (submodules were
    /// never wired up at all, filters ran normally).
    #[test]
    fn defaults_are_trusted_and_match_legacy_behavior() {
        let options = CloneOptions::default();
        assert_eq!(options.trust, CloneTrust::Trusted);
        assert_eq!(options.effective_submodule_mode(), SubmoduleMode::Disabled);
        assert_eq!(options.effective_filter_mode(), FilterMode::Enabled);
    }

    /// A `Trusted` clone honors whatever `SubmoduleMode`/`FilterMode` the
    /// caller explicitly configured.
    #[test]
    fn trusted_honors_configured_modes() {
        let options = CloneOptions::builder()
            .trust(CloneTrust::Trusted)
            .submodule_mode(SubmoduleMode::Enabled)
            .filter_mode(FilterMode::Passthrough)
            .build();
        assert_eq!(options.effective_submodule_mode(), SubmoduleMode::Enabled);
        assert_eq!(options.effective_filter_mode(), FilterMode::Passthrough);
    }

    /// `Untrusted` clamps submodule and filter mode to the safe values EVEN
    /// WHEN the caller explicitly (mis)configured the opposite — this is the
    /// property that makes `Untrusted` a single source of truth rather than
    /// a bool that can be independently flipped back on.
    #[test]
    fn untrusted_clamps_submodule_and_filter_mode_despite_caller_override() {
        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .submodule_mode(SubmoduleMode::Enabled) // attempted override
            .filter_mode(FilterMode::Enabled) // attempted override
            .build();

        assert_eq!(
            options.effective_submodule_mode(),
            SubmoduleMode::Disabled,
            "Untrusted must clamp submodule_mode to Disabled regardless of the raw field"
        );
        assert_eq!(
            options.effective_filter_mode(),
            FilterMode::Passthrough,
            "Untrusted must clamp filter_mode to Passthrough regardless of the raw field"
        );
    }

    // ---- validate_trust() ----

    #[test]
    fn trusted_options_always_validate() {
        use crate::auth::AuthStrategy;
        use crate::callback_config::{AuthMode, CallbackConfigBuilder};

        let options = CloneOptions::builder()
            .trust(CloneTrust::Trusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth(AuthStrategy::Token {
                        token: "t".into(),
                        host: None,
                    })
                    .auth_mode(AuthMode::AllowAmbient)
                    .build(),
            )
            .build();
        assert!(options.validate_trust().is_ok());
    }

    #[test]
    fn untrusted_rejects_allow_ambient_auth_mode() {
        use crate::callback_config::{AuthMode, CallbackConfigBuilder};

        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth_mode(AuthMode::AllowAmbient)
                    .build(),
            )
            .build();
        assert!(
            options.validate_trust().is_err(),
            "Untrusted + AllowAmbient must be refused at validation time"
        );
    }

    #[test]
    fn untrusted_rejects_unscoped_token() {
        use crate::auth::AuthStrategy;
        use crate::callback_config::CallbackConfigBuilder;

        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth(AuthStrategy::Token {
                        token: "unscoped".into(),
                        host: None,
                    })
                    .build(),
            )
            .build();
        assert!(
            options.validate_trust().is_err(),
            "Untrusted + an unscoped Token (host: None) must be refused"
        );
    }

    #[test]
    fn untrusted_accepts_host_scoped_token() {
        use crate::auth::AuthStrategy;
        use crate::callback_config::CallbackConfigBuilder;

        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth(AuthStrategy::Token {
                        token: "scoped".into(),
                        host: Some("github.com".into()),
                    })
                    .build(),
            )
            .build();
        assert!(
            options.validate_trust().is_ok(),
            "Untrusted + a host-scoped Token must validate — this is the intended scoped-token path"
        );
    }

    #[test]
    fn untrusted_with_no_callback_config_validates() {
        let options = CloneOptions::builder().trust(CloneTrust::Untrusted).build();
        assert!(options.validate_trust().is_ok());
    }

    // FilterMode's effect on the actual checkout (disable_filters wiring) is
    // proved end-to-end in manager.rs's ident-filter adversarial test — a
    // git2::build::CheckoutBuilder has no getter to unit-test the flag in
    // isolation, and a fake assertion here would prove nothing about the
    // real libgit2 behavior.
}
