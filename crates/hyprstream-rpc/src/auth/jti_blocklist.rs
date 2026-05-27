//! JWT ID (jti) blocklist for access token revocation.
//!
//! Revoked token IDs are stored until their `exp` time passes, at which
//! point the natural JWT expiry check rejects them anyway.

use std::collections::HashMap;

/// Blocklist of revoked JWT IDs.
pub trait JtiBlocklist: Send + Sync {
    /// Returns `true` if the given jti has been revoked.
    fn is_revoked(&self, jti: &str) -> bool;

    /// Revoke a jti. `expires_at` is the token's `exp` — the entry can be
    /// garbage-collected after this time.
    fn revoke(&self, jti: String, expires_at: i64);
}

/// In-memory jti blocklist with periodic cleanup.
pub struct InMemoryJtiBlocklist {
    #[cfg(not(target_arch = "wasm32"))]
    revoked: parking_lot::RwLock<HashMap<String, i64>>,
    #[cfg(target_arch = "wasm32")]
    revoked: std::sync::RwLock<HashMap<String, i64>>,
}

impl Default for InMemoryJtiBlocklist {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryJtiBlocklist {
    pub fn new() -> Self {
        Self {
            revoked: Default::default(),
        }
    }

    fn cleanup(&self, now: i64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut map = self.revoked.write();
            map.retain(|_, exp| *exp > now);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut map = self.revoked.write().expect("jti blocklist lock poisoned");
            map.retain(|_, exp| *exp > now);
        }
    }
}

impl JtiBlocklist for InMemoryJtiBlocklist {
    fn is_revoked(&self, jti: &str) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.revoked.read().contains_key(jti)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.revoked.read().expect("jti blocklist lock poisoned").contains_key(jti)
        }
    }

    fn revoke(&self, jti: String, expires_at: i64) {
        let now = chrono::Utc::now().timestamp();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut map = self.revoked.write();
            map.insert(jti, expires_at);
            if map.len() > 10_000 {
                drop(map);
                self.cleanup(now);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut map = self.revoked.write().expect("jti blocklist lock poisoned");
            map.insert(jti, expires_at);
            if map.len() > 10_000 {
                drop(map);
                self.cleanup(now);
            }
        }
    }
}
