//! OAuth 2.1 server state management.
//!
//! Manages registered clients, pending authorization codes, refresh tokens,
//! and delegates token issuance to PolicyService via ZMQ.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::config::OAuthConfig;
use crate::services::PolicyClient;

/// A dynamically registered OAuth client (RFC 7591) or Client ID Metadata Document client.
#[derive(Debug, Clone)]
pub struct RegisteredClient {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
    /// True if this client was registered via Client ID Metadata Document (HTTPS URL client_id)
    pub is_cimd: bool,
    pub registered_at: Instant,
}

/// A pending authorization code awaiting token exchange.
#[derive(Debug, Clone)]
pub struct PendingAuthCode {
    pub code: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub scopes: Vec<String>,
    /// RFC 8707 resource indicator (the audience for the token)
    pub resource: Option<String>,
    pub created_at: Instant,
    pub expires_at: Instant,
}

impl PendingAuthCode {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Status of a pending device authorization code (RFC 8628).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceCodeStatus {
    /// User has not yet approved or denied.
    Pending,
    /// User approved the authorization request.
    Approved,
    /// User denied the authorization request.
    Denied,
}

/// A pending device authorization code (RFC 8628).
#[derive(Debug, Clone)]
pub struct PendingDeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    /// RFC 8707 resource indicator (the audience for the token)
    pub resource: Option<String>,
    pub status: DeviceCodeStatus,
    pub created_at: Instant,
    pub expires_at: Instant,
    /// Minimum polling interval in seconds
    pub interval: u64,
    /// Last time the client polled for this code
    pub last_polled: Option<Instant>,
}

impl PendingDeviceCode {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// A stored refresh token entry (OAuth 2.1 rotation).
#[derive(Debug, Clone)]
pub struct RefreshTokenEntry {
    pub client_id: String,
    pub scopes: Vec<String>,
    pub resource: Option<String>,
    pub expires_at: Instant,
}

impl RefreshTokenEntry {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Shared OAuth server state.
pub struct OAuthState {
    /// Registered clients (dynamic + CIMD)
    pub clients: RwLock<HashMap<String, RegisteredClient>>,
    /// Pending authorization codes (single-use, 60s TTL)
    pub pending_codes: RwLock<HashMap<String, PendingAuthCode>>,
    /// Pending device authorization codes (RFC 8628), keyed by device_code
    pub pending_device_codes: RwLock<HashMap<String, PendingDeviceCode>>,
    /// Reverse lookup: user_code -> device_code
    pub device_code_by_user_code: RwLock<HashMap<String, String>>,
    /// Refresh tokens (keyed by opaque token string, rotated on use)
    pub refresh_tokens: RwLock<HashMap<String, RefreshTokenEntry>>,
    /// PolicyClient for JWT token issuance via ZMQ
    pub policy_client: PolicyClient,
    /// Issuer URL (e.g., "http://localhost:6791")
    pub issuer_url: String,
    /// Default scopes for new clients
    pub default_scopes: Vec<String>,
    /// Access token TTL in seconds
    pub token_ttl: u32,
    /// Refresh token TTL in seconds
    pub refresh_token_ttl: u32,
    /// HTTP client for fetching Client ID Metadata Documents
    pub http_client: reqwest::Client,
}

impl OAuthState {
    pub fn new(config: &OAuthConfig, policy_client: PolicyClient) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            pending_codes: RwLock::new(HashMap::new()),
            pending_device_codes: RwLock::new(HashMap::new()),
            device_code_by_user_code: RwLock::new(HashMap::new()),
            refresh_tokens: RwLock::new(HashMap::new()),
            policy_client,
            issuer_url: config.issuer_url(),
            default_scopes: config.default_scopes.clone(),
            token_ttl: config.token_ttl_seconds,
            refresh_token_ttl: config.refresh_token_ttl_seconds,
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Spawn a background task that sweeps expired codes every 30 seconds.
    pub fn spawn_code_sweeper(self: &Arc<Self>) {
        let state = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;

                // Sweep expired auth codes
                {
                    let mut codes = state.pending_codes.write().await;
                    codes.retain(|_, code| !code.is_expired());
                }

                // Sweep expired device codes
                {
                    let mut device_codes = state.pending_device_codes.write().await;
                    let mut user_code_map = state.device_code_by_user_code.write().await;
                    device_codes.retain(|_, dc| {
                        if dc.is_expired() {
                            user_code_map.remove(&dc.user_code);
                            false
                        } else {
                            true
                        }
                    });
                }

                // Sweep expired refresh tokens
                {
                    let mut tokens = state.refresh_tokens.write().await;
                    tokens.retain(|_, entry| !entry.is_expired());
                }
            }
        });
    }
}
