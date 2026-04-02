//! Server-side session management for the OAuth browser login flow.
//!
//! Sessions are stored server-side and identified by a cryptographic cookie.
//! Cookie attributes: HttpOnly, Secure, SameSite=Lax.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::RngCore;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use tokio::sync::RwLock;

/// A server-side session for an authenticated user.
#[derive(Debug, Clone)]
pub struct Session {
    /// Authenticated username (local subject or mapped external subject).
    pub username: String,
    /// How the user authenticated ("local" or provider slug).
    pub auth_method: String,
    /// When the user authenticated.
    pub authenticated_at: Instant,
    /// When the session expires.
    pub expires_at: Instant,
}

/// Session store with sweeping support.
pub struct SessionStore {
    sessions: RwLock<HashMap<String, Session>>,
    ttl: Duration,
}

pub const SESSION_COOKIE_NAME: &str = "hyprstream_session";

impl SessionStore {
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// Create a new session and return the session ID (for the cookie).
    pub async fn create(&self, username: String, auth_method: String) -> String {
        let mut id_bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut id_bytes);
        let session_id = URL_SAFE_NO_PAD.encode(id_bytes);

        let session = Session {
            username,
            auth_method,
            authenticated_at: Instant::now(),
            expires_at: Instant::now() + self.ttl,
        };

        self.sessions.write().await.insert(session_id.clone(), session);
        session_id
    }

    /// Look up a session by ID. Returns None if not found or expired.
    pub async fn get(&self, session_id: &str) -> Option<Session> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).and_then(|s| {
            if s.expires_at > Instant::now() {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    /// Remove a session (logout).
    pub async fn remove(&self, session_id: &str) {
        self.sessions.write().await.remove(session_id);
    }

    /// Sweep expired sessions.
    pub async fn sweep(&self) {
        let now = Instant::now();
        self.sessions.write().await.retain(|_, s| s.expires_at > now);
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new(8 * 3600) // 8 hours
    }
}

/// Extract session ID from request cookies.
pub fn extract_session_id(headers: &axum::http::HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies
                .split(';')
                .map(str::trim)
                .find(|c| c.starts_with(SESSION_COOKIE_NAME))
                .and_then(|c| c.strip_prefix(&format!("{SESSION_COOKIE_NAME}=")))
                .map(String::from)
        })
}

/// Build a Set-Cookie header value for the session.
pub fn session_cookie(session_id: &str, secure: bool) -> String {
    let secure_flag = if secure { "; Secure" } else { "" };
    format!(
        "{SESSION_COOKIE_NAME}={session_id}; HttpOnly; SameSite=Lax; Path=/{secure_flag}; Max-Age=28800"
    )
}

/// Build a Set-Cookie header that clears the session.
pub fn clear_session_cookie() -> String {
    format!(
        "{SESSION_COOKIE_NAME}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"
    )
}
