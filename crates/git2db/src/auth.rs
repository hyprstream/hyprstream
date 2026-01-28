//! Authentication and credential management
//!
//! Consolidated authentication patterns from the original codebase

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
    /// Use personal access token
    Token { token: String },
    /// Use default credentials (for public repos)
    Default,
}

/// Credential manager for handling authentication
pub struct AuthManager {
    strategies: Vec<AuthStrategy>,
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new() -> Self {
        Self {
            strategies: vec![AuthStrategy::Default],
        }
    }

    /// Create authentication manager with strategies
    pub fn with_strategies(strategies: Vec<AuthStrategy>) -> Self {
        Self { strategies }
    }

    /// Add an authentication strategy
    pub fn add_strategy(&mut self, strategy: AuthStrategy) {
        self.strategies.push(strategy);
    }

    /// Create remote callbacks with authentication
    pub fn create_callbacks(&self) -> RemoteCallbacks<'_> {
        let mut callbacks = RemoteCallbacks::new();

        callbacks.credentials(move |url, username_from_url, allowed_types| {
            self.handle_credentials(url, username_from_url, allowed_types)
        });

        callbacks.certificate_check(|cert, valid| {
            // For now, accept all certificates
            // TODO: Implement proper certificate validation based on config
            debug!(
                "Certificate check: valid={}, host={:?}",
                valid,
                cert.as_hostkey().map(|h| h.hostkey())
            );
            Ok(git2::CertificateCheckStatus::CertificateOk)
        });

        callbacks
    }

    /// Handle credential requests
    fn handle_credentials(
        &self,
        url: &str,
        username_from_url: Option<&str>,
        allowed_types: CredentialType,
    ) -> Result<Cred, git2::Error> {
        info!("Attempting authentication for URL: {}", url);
        info!("Username from URL: {:?}", username_from_url);
        info!("Allowed credential types: {:?}", allowed_types);

        // Try each strategy in order
        for strategy in &self.strategies {
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
        Err(git2::Error::from_str("No suitable authentication method"))
    }

    /// Try a specific authentication strategy
    fn try_strategy(
        &self,
        strategy: &AuthStrategy,
        _url: &str,
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

            AuthStrategy::Token { token } => {
                if allowed_types.contains(CredentialType::USER_PASS_PLAINTEXT) {
                    info!("Trying token authentication");
                    // For GitHub and similar services, use token as password with empty username
                    Cred::userpass_plaintext("", token)
                } else {
                    Err(git2::Error::from_str("Token authentication not allowed"))
                }
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

    /// Add token authentication
    pub fn token<T>(mut self, token: T) -> Self
    where
        T: Into<String>,
    {
        self.strategies.push(AuthStrategy::Token {
            token: token.into(),
        });
        self
    }

    /// Add default credentials (fallback)
    pub fn default_fallback(mut self) -> Self {
        self.strategies.push(AuthStrategy::Default);
        self
    }

    /// Build the authentication manager
    pub fn build(self) -> AuthManager {
        AuthManager::with_strategies(self.strategies)
    }
}

impl Default for AuthBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create common authentication configurations
pub mod presets {
    use super::*;

    /// Standard SSH configuration (agent + fallback)
    pub fn ssh_standard() -> AuthManager {
        AuthBuilder::new()
            .ssh_agent(Some("git".to_owned()))
            .default_fallback()
            .build()
    }

    /// GitHub personal access token
    pub fn github_token<T: Into<String>>(token: T) -> AuthManager {
        AuthBuilder::new()
            .token(token)
            .ssh_agent(Some("git".to_owned()))
            .default_fallback()
            .build()
    }

    /// SSH key file authentication
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
    }

    /// Public repository access only
    pub fn public_only() -> AuthManager {
        AuthBuilder::new().default_fallback().build()
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
}
