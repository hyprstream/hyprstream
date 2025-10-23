//! Send-safe callback configuration for async git operations
//!
//! This module provides a bridge between git2's non-Send callbacks and async/Send requirements.
//! Instead of holding closures directly, we capture the configuration needed to create them.

use crate::auth::AuthStrategy;
use git2::cert::Cert;
use std::sync::Arc;

/// Progress reporting configuration
#[derive(Clone)]
pub enum ProgressConfig {
    /// No progress reporting
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

/// Certificate validation configuration
#[derive(Debug, Clone)]
pub enum CertificateConfig {
    /// Accept all certificates (insecure, for testing)
    AcceptAll,
    /// Default system validation
    SystemDefault,
    /// Custom validation with pinned certificates
    Pinned(Vec<CertificatePinning>),
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

    /// Create actual git2::RemoteCallbacks from this configuration
    ///
    /// This is called within spawn_blocking where non-Send is acceptable
    pub fn create_callbacks(&self) -> git2::RemoteCallbacks<'_> {
        let mut callbacks = git2::RemoteCallbacks::new();

        // Set up authentication
        if !self.auth.is_empty() {
            let auth_strategies = self.auth.clone();
            callbacks.credentials(move |url, username_from_url, allowed_types| {
                Self::handle_auth(&auth_strategies, url, username_from_url, allowed_types)
            });
        }

        // Set up certificate checking
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
        url: &str,
        username_from_url: Option<&str>,
        allowed_types: git2::CredentialType,
    ) -> Result<git2::Cred, git2::Error> {
        // Try each strategy in order
        for strategy in strategies {
            match Self::try_auth_strategy(strategy, url, username_from_url, allowed_types) {
                Ok(cred) => return Ok(cred),
                Err(_) => continue,
            }
        }

        Err(git2::Error::from_str("No suitable authentication method"))
    }

    fn try_auth_strategy(
        strategy: &AuthStrategy,
        _url: &str,
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
            AuthStrategy::Token { token } => {
                if allowed_types.contains(CredentialType::USER_PASS_PLAINTEXT) {
                    Cred::userpass_plaintext("", token)
                } else {
                    Err(git2::Error::from_str("Token authentication not allowed"))
                }
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
        cert: &Cert,
    ) -> Result<git2::CertificateCheckStatus, git2::Error> {
        use git2::CertificateCheckStatus;

        match config {
            CertificateConfig::AcceptAll => Ok(CertificateCheckStatus::CertificateOk),
            CertificateConfig::SystemDefault => {
                // Let git2 handle default validation
                Ok(CertificateCheckStatus::CertificateOk)
            }
            CertificateConfig::Pinned(pins) => {
                // Check if certificate matches any pinned certificates
                if let Some(hostkey) = cert.as_hostkey() {
                    if let Some(hostkey_bytes) = hostkey.hostkey() {
                        let host = std::str::from_utf8(hostkey_bytes).unwrap_or("");

                        for pin in pins {
                            if pin.host == host && pin.fingerprint.as_slice() == hostkey_bytes {
                                return Ok(CertificateCheckStatus::CertificateOk);
                            }
                        }
                    }
                }

                Err(git2::Error::from_str("Certificate not pinned"))
            }
        }
    }

    fn handle_progress(config: &ProgressConfig, stats: git2::Progress) {
        match config {
            ProgressConfig::None => {}
            ProgressConfig::Stdout => {
                let current = stats.received_objects();
                let total = stats.total_objects();
                println!("Progress: {}/{} objects", current, total);
            }
            ProgressConfig::Channel(reporter) => {
                let current = stats.received_objects();
                let total = stats.total_objects();
                reporter.report("fetch", current, total);
            }
        }
    }
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self::None
    }
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self::SystemDefault
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

    #[test]
    fn test_callback_config_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CallbackConfig>();
    }

    #[test]
    fn test_builder_pattern() {
        let config = CallbackConfigBuilder::new()
            .auth(AuthStrategy::SshAgent {
                username: Some("git".to_string()),
            })
            .progress(ProgressConfig::Stdout)
            .certificates(CertificateConfig::AcceptAll)
            .build();

        assert_eq!(config.auth.len(), 1);
        assert!(matches!(config.progress, ProgressConfig::Stdout));
        assert!(matches!(config.certificates, CertificateConfig::AcceptAll));
    }
}
