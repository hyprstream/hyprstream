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
/// `RequestService::federation_key_source()` so the default `verify_claims()`
/// can perform real key resolution instead of rejecting federated JWTs.
///
/// # Key-set semantics (rotation-aware, #1185)
///
/// A published JWKS is a **named set**, never an ordered singleton. `get_keys`
/// returns every usable Ed25519 entry from the issuer's current JWKS so the
/// caller can try each candidate — this is what makes overlap rotation (old +
/// new keys published simultaneously) and future PQ-hybrid publication
/// possible. When the JWT carries a `kid`, the matching candidate is ordered
/// first; the rest follow so a verifier that prefers the kid still benefits
/// from overlap fallback if the named key has been retired mid-window.
///
/// The resolver MUST NOT collapse the set to a positional singleton: returning
/// the "first" Ed25519 key forecloses rotation (#1183). An empty result is an
/// `Err`.
///
/// # Note on `Send` bounds
///
/// `get_keys` uses the default (`Send`) flavour of `#[async_trait]` because
/// `FederationKeyResolver` performs real async I/O (HTTPS JWKS fetch) whose
/// future must be `Send`. Call sites inside `#[async_trait(?Send)]` contexts
/// (e.g. `RequestService::verify_claims`) may `.await` this future directly:
/// it is valid in Rust to await a `Send` future from within a `!Send`
/// async function — the `?Send` bound constrains the outer function's
/// future, not the futures it polls. `RequestService::verify_claims` does
/// exactly this and compiles correctly.
#[async_trait::async_trait]
pub trait FederationKeySource: Send + Sync + 'static {
    /// Return `true` if `issuer` is in the configured trusted-issuer list.
    ///
    /// Callers should check this before calling `get_keys` to distinguish an
    /// untrusted issuer (policy reject → 401) from a transient fetch error
    /// (retry eligible → 503).
    fn is_trusted(&self, issuer: &str) -> bool;

    /// Fetch (or return from cache) the Ed25519 candidate verifying keys for
    /// `issuer`.
    ///
    /// Returns every usable Ed25519 key from the issuer's JWKS, ordered so
    /// that when `kid` is `Some` the matching candidate comes first and the
    /// remaining overlap candidates follow. The caller SHOULD try each
    /// candidate against the JWT and accept the first that verifies — this is
    /// the rotation/pQ-hybrid publication model (#1183).
    ///
    /// On a cache miss for the requested `kid`, the resolver refetches the
    /// JWKS once and re-checks. A `kid` that is still absent after a fresh
    /// fetch fails closed (`Err`); the resolver MUST NOT silently substitute
    /// another key for a named `kid`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - the issuer is not in the trusted list (`is_trusted` returns `false`), or
    /// - the JWKS endpoint is unreachable or returns no usable Ed25519 key, or
    /// - `kid` is `Some` and no candidate with that `kid` exists after refetch.
    async fn get_keys(&self, issuer: &str, kid: Option<&str>) -> Result<Vec<VerifyingKey>>;
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

        async fn get_keys(
            &self,
            issuer: &str,
            _kid: Option<&str>,
        ) -> Result<Vec<VerifyingKey>> {
            let _ = issuer;
            anyhow::bail!("Issuer not trusted: {}", issuer)
        }
    }

    #[tokio::test]
    async fn trait_object_compiles_and_rejects() {
        let src: Arc<dyn FederationKeySource> = Arc::new(AlwaysReject);
        assert!(!src.is_trusted("https://evil.example.com"));
        assert!(src.get_keys("https://evil.example.com", None).await.is_err());
    }
}
