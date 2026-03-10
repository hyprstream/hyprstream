//! Abstraction layer for multi-issuer JWT key resolution on the ZMQ transport.
//!
//! Defines `FederationKeySource`, an async trait implemented by
//! `hyprstream::auth::FederationKeyResolver`. The indirection avoids a circular
//! crate dependency: `hyprstream-rpc` cannot import from `hyprstream` directly.

use anyhow::Result;
use ed25519_dalek::VerifyingKey;

/// Resolves external JWT issuer URLs to Ed25519 verifying keys.
///
/// Implemented by `hyprstream::auth::FederationKeyResolver`. Services that
/// serve external traffic return `Some(Arc<dyn FederationKeySource>)` from
/// `ZmqService::federation_key_source()` so the default `verify_claims()`
/// can perform real key resolution instead of rejecting federated JWTs.
///
/// # Note on `Send` bounds
///
/// `get_key` uses the default (`Send`) flavour of `#[async_trait]` because
/// `FederationKeyResolver` performs real async I/O (HTTPS JWKS fetch) whose
/// future must be `Send`. Call sites inside `#[async_trait(?Send)]` contexts
/// (e.g. the single-threaded ZMQ reactor in `ZmqService::verify_claims`)
/// **must not** `.await` this future directly inside the `?Send` async fn.
/// Instead, either:
/// - `tokio::task::spawn` the lookup and `.await` the `JoinHandle`, or
/// - restructure `verify_claims` so the federation lookup happens outside
///   the `?Send` reactor (recommended approach for Task 3).
#[async_trait::async_trait]
pub trait FederationKeySource: Send + Sync + 'static {
    /// Return `true` if `issuer` is in the configured trusted-issuer list.
    ///
    /// Callers should check this before calling `get_key` to distinguish an
    /// untrusted issuer (policy reject → 401) from a transient fetch error
    /// (retry eligible → 503).
    fn is_trusted(&self, issuer: &str) -> bool;

    /// Fetch (or return from cache) the Ed25519 verifying key for `issuer`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - the issuer is not in the trusted list (`is_trusted` returns `false`), or
    /// - the JWKS endpoint is unreachable or returns an invalid key.
    ///
    /// Callers that need to distinguish these cases should call `is_trusted`
    /// first; a subsequent `Err` from `get_key` then indicates a fetch/parse
    /// failure rather than a policy rejection.
    async fn get_key(&self, issuer: &str) -> Result<VerifyingKey>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Minimal stub that rejects all issuers. Verifies the trait compiles as
    /// `Arc<dyn FederationKeySource>` and that `Send + Sync + 'static` bounds hold.
    struct AlwaysReject;

    #[async_trait::async_trait]
    impl FederationKeySource for AlwaysReject {
        fn is_trusted(&self, _issuer: &str) -> bool {
            false
        }

        async fn get_key(&self, issuer: &str) -> Result<VerifyingKey> {
            anyhow::bail!("Issuer not trusted: {}", issuer)
        }
    }

    #[tokio::test]
    async fn trait_object_compiles_and_rejects() {
        let src: Arc<dyn FederationKeySource> = Arc::new(AlwaysReject);
        assert!(!src.is_trusted("https://evil.example.com"));
        assert!(src.get_key("https://evil.example.com").await.is_err());
    }
}
