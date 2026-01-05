//! SSH-based CAS client for self-hosted XET servers.
//!
//! This module provides an [`SshStorage`] implementation that communicates with
//! a remote `cas-serve` binary over SSH, enabling XET operations on self-hosted
//! infrastructure.
//!
//! # Architecture
//!
//! ```text
//! Client                              Server
//! ──────                              ──────
//! SshStorage                          cas-serve
//!    │                                   │
//!    │  ssh user@host cas-serve          │
//!    │──────────────────────────────────>│
//!    │                                   │
//!    │  JSON request (NDJSON)            │
//!    │──────────────────────────────────>│
//!    │                                   │
//!    │  JSON response (NDJSON)           │
//!    │<──────────────────────────────────│
//! ```
//!
//! # Example
//!
//! ```ignore
//! use git_xet_filter::ssh_client::SshStorage;
//!
//! // Connect via SSH URL
//! let storage = SshStorage::connect("ssh://user@host/path/to/storage").await?;
//!
//! // Use like any other StorageBackend
//! let data = storage.smudge_from_hash(&hash).await?;
//! ```

#[cfg(feature = "ssh-transport")]
use async_trait::async_trait;
#[cfg(feature = "ssh-transport")]
use std::path::Path;
#[cfg(feature = "ssh-transport")]
use std::sync::Arc;
#[cfg(feature = "ssh-transport")]
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
#[cfg(feature = "ssh-transport")]
use tokio::sync::Mutex;
#[cfg(feature = "ssh-transport")]
use tracing::{debug, error, info, warn};

#[cfg(feature = "ssh-transport")]
use cas_serve::{ErrorCode, Request, Response};

#[cfg(feature = "ssh-transport")]
use crate::error::{Result, XetError, XetErrorKind};
#[cfg(feature = "ssh-transport")]
use crate::storage::StorageBackend;

/// SSH connection configuration parsed from URL
#[cfg(feature = "ssh-transport")]
#[derive(Debug, Clone)]
pub struct SshConfig {
    /// Username for SSH connection
    pub user: String,
    /// Hostname or IP address
    pub host: String,
    /// SSH port (default: 22)
    pub port: u16,
    /// Remote storage path (passed to cas-serve via CAS_STORAGE env var)
    pub storage_path: Option<String>,
}

#[cfg(feature = "ssh-transport")]
impl SshConfig {
    /// Parse an SSH URL into configuration
    ///
    /// Supported formats:
    /// - `ssh://user@host/path/to/storage`
    /// - `ssh://user@host:port/path/to/storage`
    /// - `ssh://host/path` (uses current user)
    pub fn from_url(url: &str) -> Result<Self> {
        let url = url
            .strip_prefix("ssh://")
            .ok_or_else(|| XetError::new(XetErrorKind::InvalidConfig, "URL must start with ssh://"))?;

        // Parse user@host:port/path
        let (user_host, path) = match url.find('/') {
            Some(idx) => (&url[..idx], Some(&url[idx + 1..])),
            None => (url, None),
        };

        let (user, host_port) = match user_host.find('@') {
            Some(idx) => (&user_host[..idx], &user_host[idx + 1..]),
            None => {
                let current_user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
                (current_user.as_str(), user_host)
            }
        };

        let (host, port) = match host_port.find(':') {
            Some(idx) => {
                let port_str = &host_port[idx + 1..];
                let port = port_str.parse().map_err(|_| {
                    XetError::new(XetErrorKind::InvalidConfig, format!("Invalid port: {}", port_str))
                })?;
                (&host_port[..idx], port)
            }
            None => (host_port, 22),
        };

        Ok(Self {
            user: user.to_string(),
            host: host.to_string(),
            port,
            storage_path: path.map(|s| s.to_string()),
        })
    }
}

/// SSH-based storage backend for remote CAS operations.
///
/// Connects to a remote server running `cas-serve` and executes XET operations
/// over the SSH channel.
#[cfg(feature = "ssh-transport")]
pub struct SshStorage {
    config: SshConfig,
    /// SSH session (held for the lifetime of the storage)
    session: Arc<Mutex<russh::client::Handle<SshClientHandler>>>,
    /// Channel for communication with cas-serve
    channel: Arc<Mutex<russh::Channel<russh::client::Msg>>>,
}

/// SSH client event handler
#[cfg(feature = "ssh-transport")]
struct SshClientHandler;

#[cfg(feature = "ssh-transport")]
impl russh::client::Handler for SshClientHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &russh_keys::key::PublicKey,
    ) -> std::result::Result<bool, Self::Error> {
        // TODO: Implement proper host key verification
        // For now, accept all keys (INSECURE - suitable for development only)
        warn!("SSH host key verification disabled - not suitable for production");
        Ok(true)
    }
}

#[cfg(feature = "ssh-transport")]
impl SshStorage {
    /// Connect to a remote CAS server via SSH.
    ///
    /// # Arguments
    ///
    /// * `url` - SSH URL in the format `ssh://user@host:port/storage/path`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let storage = SshStorage::connect("ssh://user@192.168.1.100/var/lib/xet").await?;
    /// ```
    pub async fn connect(url: &str) -> Result<Self> {
        let config = SshConfig::from_url(url)?;
        Self::connect_with_config(config).await
    }

    /// Connect with explicit configuration.
    pub async fn connect_with_config(config: SshConfig) -> Result<Self> {
        info!(
            "Connecting to SSH CAS server: {}@{}:{}",
            config.user, config.host, config.port
        );

        // Create SSH client configuration
        let ssh_config = russh::client::Config::default();

        // Connect to server
        let addr = format!("{}:{}", config.host, config.port);
        let mut session = russh::client::connect(Arc::new(ssh_config), &addr, SshClientHandler)
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::RuntimeError,
                    format!("SSH connection failed: {}", e),
                )
            })?;

        // Authenticate
        let authenticated = Self::authenticate(&mut session, &config).await?;
        if !authenticated {
            return Err(XetError::new(
                XetErrorKind::RuntimeError,
                "SSH authentication failed",
            ));
        }

        // Open channel and execute cas-serve
        let channel = session.channel_open_session().await.map_err(|e| {
            XetError::new(
                XetErrorKind::RuntimeError,
                format!("Failed to open SSH channel: {}", e),
            )
        })?;

        // Set CAS_STORAGE environment variable if specified
        if let Some(ref storage_path) = config.storage_path {
            channel
                .set_env(true, "CAS_STORAGE", storage_path)
                .await
                .map_err(|e| {
                    XetError::new(
                        XetErrorKind::RuntimeError,
                        format!("Failed to set CAS_STORAGE: {}", e),
                    )
                })?;
        }

        // Execute cas-serve
        channel.exec(true, "cas-serve").await.map_err(|e| {
            XetError::new(
                XetErrorKind::RuntimeError,
                format!("Failed to execute cas-serve: {}", e),
            )
        })?;

        info!("SSH CAS connection established");

        Ok(Self {
            config,
            session: Arc::new(Mutex::new(session)),
            channel: Arc::new(Mutex::new(channel)),
        })
    }

    /// Authenticate with the SSH server.
    async fn authenticate(
        session: &mut russh::client::Handle<SshClientHandler>,
        config: &SshConfig,
    ) -> Result<bool> {
        // Try SSH agent first
        if let Ok(mut agent) = russh_keys::agent::client::AgentClient::connect_env().await {
            let identities = agent.request_identities().await.unwrap_or_default();
            for identity in identities {
                if session
                    .authenticate_publickey_with(&config.user, Arc::new(identity))
                    .await
                    .is_ok()
                {
                    debug!("Authenticated via SSH agent");
                    return Ok(true);
                }
            }
        }

        // Try default SSH key locations
        let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        let key_paths = [
            home.join(".ssh/id_ed25519"),
            home.join(".ssh/id_rsa"),
            home.join(".ssh/id_ecdsa"),
        ];

        for key_path in &key_paths {
            if key_path.exists() {
                match russh_keys::load_secret_key(key_path, None) {
                    Ok(key) => {
                        if session
                            .authenticate_publickey(&config.user, Arc::new(key))
                            .await
                            .is_ok()
                        {
                            debug!("Authenticated with key: {:?}", key_path);
                            return Ok(true);
                        }
                    }
                    Err(e) => {
                        debug!("Failed to load key {:?}: {}", key_path, e);
                    }
                }
            }
        }

        Err(XetError::new(
            XetErrorKind::RuntimeError,
            "No valid SSH authentication method found",
        ))
    }

    /// Send a request to cas-serve and receive a response.
    async fn request(&self, request: Request) -> Result<Response> {
        let mut channel = self.channel.lock().await;

        // Serialize request
        let json = serde_json::to_string(&request).map_err(|e| {
            XetError::new(
                XetErrorKind::RuntimeError,
                format!("Failed to serialize request: {}", e),
            )
        })?;

        debug!("Sending request: {}", json);

        // Send request
        channel
            .data(format!("{}\n", json).as_bytes())
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::RuntimeError,
                    format!("Failed to send request: {}", e),
                )
            })?;

        // Read response (wait for data from channel)
        // Note: This is a simplified implementation. A production version would
        // need proper buffering and handling of SSH channel messages.
        let mut response_data = Vec::new();
        loop {
            match channel.wait().await {
                Some(russh::ChannelMsg::Data { data }) => {
                    response_data.extend_from_slice(&data);
                    // Check if we have a complete line
                    if response_data.contains(&b'\n') {
                        break;
                    }
                }
                Some(russh::ChannelMsg::Eof) => {
                    return Err(XetError::new(
                        XetErrorKind::RuntimeError,
                        "cas-serve closed connection",
                    ));
                }
                Some(russh::ChannelMsg::ExitStatus { exit_status }) => {
                    if exit_status != 0 {
                        return Err(XetError::new(
                            XetErrorKind::RuntimeError,
                            format!("cas-serve exited with status: {}", exit_status),
                        ));
                    }
                }
                None => {
                    return Err(XetError::new(
                        XetErrorKind::RuntimeError,
                        "SSH channel closed unexpectedly",
                    ));
                }
                _ => continue,
            }
        }

        // Parse response
        let response_str = String::from_utf8_lossy(&response_data);
        let response: Response = serde_json::from_str(response_str.trim()).map_err(|e| {
            XetError::new(
                XetErrorKind::RuntimeError,
                format!("Failed to parse response: {} - raw: {}", e, response_str),
            )
        })?;

        debug!("Received response: {:?}", response);

        // Convert error responses to XetError
        if let Response::Error { code, message } = &response {
            return Err(XetError::new(
                match code {
                    ErrorCode::NotFound => XetErrorKind::DownloadFailed,
                    ErrorCode::InvalidHash => XetErrorKind::InvalidPointer,
                    ErrorCode::UploadFailed => XetErrorKind::UploadFailed,
                    _ => XetErrorKind::RuntimeError,
                },
                message.clone(),
            ));
        }

        Ok(response)
    }

    /// Gracefully close the SSH connection.
    pub async fn close(&self) -> Result<()> {
        let _ = self.request(Request::Shutdown).await;
        Ok(())
    }
}

#[cfg(feature = "ssh-transport")]
#[async_trait]
impl StorageBackend for SshStorage {
    async fn clean_file(&self, path: &Path) -> Result<String> {
        // Read file and upload
        let data = tokio::fs::read(path).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read file: {}", e),
            )
        })?;
        self.clean_bytes(&data).await
    }

    fn is_pointer(&self, content: &str) -> bool {
        // Try to parse as JSON with expected fields
        serde_json::from_str::<serde_json::Value>(content)
            .map(|v| v.get("merkle_hash").is_some() || v.get("filesize").is_some())
            .unwrap_or(false)
    }

    async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);

        let response = self.request(Request::UploadXorb { data: encoded }).await?;

        match response {
            Response::UploadSuccess { hash } => {
                // Create a minimal pointer JSON
                let pointer = serde_json::json!({
                    "merkle_hash": hash,
                    "filesize": data.len()
                });
                Ok(pointer.to_string())
            }
            _ => Err(XetError::new(
                XetErrorKind::UploadFailed,
                "Unexpected response from upload",
            )),
        }
    }

    async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()> {
        let data = self.smudge_bytes(pointer).await?;
        tokio::fs::write(output_path, data).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to write file: {}", e),
            )
        })
    }

    async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>> {
        // Parse pointer to extract hash
        let pointer_value: serde_json::Value = serde_json::from_str(pointer).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid pointer JSON: {}", e),
            )
        })?;

        let hash = pointer_value
            .get("merkle_hash")
            .and_then(|h| h.as_str())
            .ok_or_else(|| {
                XetError::new(XetErrorKind::InvalidPointer, "Missing merkle_hash in pointer")
            })?;

        let response = self
            .request(Request::GetFile {
                hash: hash.to_string(),
            })
            .await?;

        match response {
            Response::File { data } => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(&data)
                    .map_err(|e| {
                        XetError::new(
                            XetErrorKind::DownloadFailed,
                            format!("Failed to decode base64: {}", e),
                        )
                    })
            }
            _ => Err(XetError::new(
                XetErrorKind::DownloadFailed,
                "Unexpected response from download",
            )),
        }
    }

    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
        let response = self
            .request(Request::GetFile {
                hash: hash.hex(),
            })
            .await?;

        match response {
            Response::File { data } => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(&data)
                    .map_err(|e| {
                        XetError::new(
                            XetErrorKind::DownloadFailed,
                            format!("Failed to decode base64: {}", e),
                        )
                    })
            }
            _ => Err(XetError::new(
                XetErrorKind::DownloadFailed,
                "Unexpected response from download",
            )),
        }
    }

    async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()> {
        let data = self.smudge_from_hash(hash).await?;
        tokio::fs::write(output_path, data).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to write file: {}", e),
            )
        })
    }
}

#[cfg(all(test, feature = "ssh-transport"))]
mod tests {
    use super::*;

    #[test]
    fn test_ssh_config_parsing() {
        // Full URL
        let config = SshConfig::from_url("ssh://user@host:2222/path/to/storage").unwrap();
        assert_eq!(config.user, "user");
        assert_eq!(config.host, "host");
        assert_eq!(config.port, 2222);
        assert_eq!(config.storage_path, Some("path/to/storage".to_string()));

        // Without port
        let config = SshConfig::from_url("ssh://user@host/path").unwrap();
        assert_eq!(config.port, 22);

        // Without path
        let config = SshConfig::from_url("ssh://user@host").unwrap();
        assert_eq!(config.storage_path, None);
    }
}
