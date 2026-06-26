//! Identity provider abstraction for unified key management.
//!
//! Three implementations cover the trust boundaries:
//! - `NodeIdentityProvider` — native server/CLI (file + keyring)
//! - `AegisIdentityProvider` — browser (aegis-vault via JS callback)
//! - `FederatedIdentityProvider` — cross-node (entity statements)
//!
//! HKDF derivation: `HKDF(root_seed, purpose_bytes)` — no prefix.
//! The purpose string IS the domain separator.

use anyhow::Result;
use async_trait::async_trait;

use crate::Subject;

/// A purpose-keyed signing identity derived from a root seed.
///
/// Holds a derived Ed25519 keypair for a specific purpose (e.g.,
/// `"hyprstream-rpc-envelope-v1"`). The private key never leaves
/// the identity provider — callers get the pubkey and a sign function.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait SigningIdentity: Send + Sync {
    /// 32-byte Ed25519 public key for this purpose.
    fn pubkey(&self) -> [u8; 32];

    /// Sign canonical bytes. Returns 64-byte Ed25519 signature.
    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]>;
}

/// Unified identity provider — key management + trust resolution.
///
/// Implementations manage key storage, derivation, and peer trust.
/// The `Signer` trait (used by `RpcClientImpl`) adapts over this.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait IdentityProvider: Send + Sync {
    /// Open a purpose-keyed signing identity.
    ///
    /// Derives a unique Ed25519 keypair via `HKDF(root_seed, purpose)`.
    /// Same purpose always produces the same keypair from the same root.
    async fn identity_open(&self, purpose: &str) -> Result<Box<dyn SigningIdentity>>;

    /// Resolve a peer's verified pubkey to a subject (authorization identity).
    ///
    /// Called after Ed25519 signature verification succeeds. Maps the
    /// cryptographically verified signer pubkey to a permission subject.
    fn resolve(&self, pubkey: &[u8; 32]) -> Subject;
}
