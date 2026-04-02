//! JWT lifecycle manager for event scope authorization.
//!
//! Manages token issuance and automatic refresh for event publishers and subscribers
//! that need to authenticate with PolicyService for group key operations.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Manages JWT lifecycle for event scope authorization.
///
/// Automatically refreshes tokens before expiry to maintain continuous access
/// to PolicyService event prefix operations.
pub struct EventTokenManager {
    /// Current valid token.
    current_token: Arc<RwLock<String>>,
    /// Scope string for token issuance (e.g., "publish:events:worker.*").
    scope: String,
    /// Audience for token issuance (e.g., "events").
    audience: String,
    /// Token TTL in seconds.
    ttl_secs: u64,
    /// Background refresh task handle.
    refresh_handle: Option<JoinHandle<()>>,
    /// Cancellation token for the refresh loop.
    cancel: CancellationToken,
}

impl EventTokenManager {
    /// Create a new EventTokenManager with an initial token.
    pub fn new(
        initial_token: String,
        scope: String,
        audience: String,
        ttl_secs: u64,
    ) -> Self {
        Self {
            current_token: Arc::new(RwLock::new(initial_token)),
            scope,
            audience,
            ttl_secs,
            refresh_handle: None,
            cancel: CancellationToken::new(),
        }
    }

    /// Get the current valid token.
    pub async fn token(&self) -> String {
        self.current_token.read().await.clone()
    }

    /// Get the scope this manager is configured for.
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Get the audience.
    pub fn audience(&self) -> &str {
        &self.audience
    }

    /// Start the background refresh loop.
    ///
    /// The refresh callback is called at `max(ttl * 3/4, ttl - 60s)` intervals
    /// to ensure overlap before expiry.
    ///
    /// # Arguments
    ///
    /// * `refresh_fn` - Async function that issues a new token given (scope, audience, ttl_secs)
    pub fn start_refresh_loop<F, Fut>(&mut self, refresh_fn: F)
    where
        F: Fn(String, String, u64) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String, String>> + Send + 'static,
    {
        let token = Arc::clone(&self.current_token);
        let scope = self.scope.clone();
        let audience = self.audience.clone();
        let ttl_secs = self.ttl_secs;
        let cancel = self.cancel.clone();

        // Refresh at max(ttl * 3/4, ttl - 60), with a minimum of 1s to prevent busy-loop.
        let refresh_interval = Duration::from_secs(
            std::cmp::max(1, std::cmp::max(ttl_secs * 3 / 4, ttl_secs.saturating_sub(60)))
        );

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(refresh_interval) => {
                        match refresh_fn(scope.clone(), audience.clone(), ttl_secs).await {
                            Ok(new_token) => {
                                *token.write().await = new_token;
                                tracing::debug!(scope = %scope, "Refreshed event token");
                            }
                            Err(e) => {
                                tracing::warn!(
                                    scope = %scope,
                                    error = %e,
                                    "Failed to refresh event token, will retry"
                                );
                            }
                        }
                    }
                }
            }
        });

        self.refresh_handle = Some(handle);
    }

    /// Stop the refresh loop.
    pub fn stop(&self) {
        self.cancel.cancel();
    }

    /// Update the token manually (e.g., after initial issuance).
    pub async fn set_token(&self, token: String) {
        *self.current_token.write().await = token;
    }
}

impl Drop for EventTokenManager {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_initial_token() {
        let mgr = EventTokenManager::new(
            "initial-jwt".to_owned(),
            "publish:events:worker.*".to_owned(),
            "events".to_owned(),
            600,
        );
        assert_eq!(mgr.token().await, "initial-jwt");
    }

    #[tokio::test]
    async fn test_set_token() {
        let mgr = EventTokenManager::new(
            "old".to_owned(),
            "scope".to_owned(),
            "aud".to_owned(),
            600,
        );
        mgr.set_token("new".to_owned()).await;
        assert_eq!(mgr.token().await, "new");
    }

    #[tokio::test]
    async fn test_refresh_loop_calls_callback() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut mgr = EventTokenManager::new(
            "initial".to_owned(),
            "scope".to_owned(),
            "aud".to_owned(),
            2, // 2 second TTL → refresh at max(1.5, 0) = 1.5s, but we use min 1s
        );

        mgr.start_refresh_loop(move |_scope, _aud, _ttl| {
            let c = counter_clone.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(format!("token-{}", c.load(Ordering::SeqCst)))
            }
        });

        // Wait enough for at least one refresh
        tokio::time::sleep(Duration::from_secs(3)).await;
        mgr.stop();

        assert!(counter.load(Ordering::SeqCst) >= 1);
        assert!(mgr.token().await.starts_with("token-"));
    }

    #[tokio::test]
    async fn test_stop_cancels_refresh() {
        let mut mgr = EventTokenManager::new(
            "initial".to_owned(),
            "scope".to_owned(),
            "aud".to_owned(),
            600,
        );

        mgr.start_refresh_loop(|_s, _a, _t| async { Ok("refreshed".to_owned()) });
        mgr.stop();

        // Token should still be initial since refresh didn't fire yet
        assert_eq!(mgr.token().await, "initial");
    }
}
