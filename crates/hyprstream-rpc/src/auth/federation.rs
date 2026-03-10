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
#[async_trait::async_trait]
pub trait FederationKeySource: Send + Sync + 'static {
    /// Return true if `issuer` is in the configured trusted-issuer list.
    fn is_trusted(&self, issuer: &str) -> bool;

    /// Fetch (or return from cache) the Ed25519 verifying key for `issuer`.
    ///
    /// Returns `Err` if the issuer is untrusted or key fetch fails.
    async fn get_key(&self, issuer: &str) -> Result<VerifyingKey>;
}
