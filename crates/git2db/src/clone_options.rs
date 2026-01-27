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

use crate::callback_config::CallbackConfig;
use git2::{build::CheckoutBuilder, FetchOptions, RemoteCallbacks};

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

    /// Whether to update submodules
    pub update_submodules: bool,

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

    /// Convert to legacy git2 options for use within spawn_blocking
    /// This is called inside spawn_blocking where lifetime constraints are satisfied
    pub(crate) fn to_git2_options(&self) -> LegacyCloneOptions<'_> {
        let mut options = LegacyCloneOptions {
            _shallow: self.shallow,
            depth: self.depth,
            branch: self.branch.clone(),
            _refspecs: self.refspecs.clone(),
            _update_submodules: self.update_submodules,
            proxy_url: self.proxy_url.clone(),
            _timeout_seconds: self.timeout_seconds,
            ..Default::default()
        };

        // Create callbacks from config if present
        if let Some(ref config) = self.callback_config {
            options.callbacks = Some(config.create_callbacks());
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
    pub _update_submodules: bool,
    pub proxy_url: Option<String>,
    pub _timeout_seconds: Option<u32>,
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

    /// Create CheckoutBuilder for this clone
    pub fn create_checkout_builder(&self) -> CheckoutBuilder<'static> {
        CheckoutBuilder::new()
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

    /// Set whether to update submodules
    pub fn update_submodules(mut self, update: bool) -> Self {
        self.options.update_submodules = update;
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
}
