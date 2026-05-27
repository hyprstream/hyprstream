//! Signer implementations for Ed25519 envelope signing.
//!
//! - `LocalSigner`: Native — owns the `SigningKey`, signs synchronously.
//! - `IdentitySigner`: Wraps a `SigningIdentity` from an `IdentityProvider`.
//! - `JsSigner`: WASM — delegates to a JS callback (aegis-vault), awaits the result.

use anyhow::Result;
use async_trait::async_trait;

use crate::crypto::SigningKey;
use crate::identity::SigningIdentity;
use crate::transport_traits::Signer;

/// Native signer that owns an Ed25519 signing key.
///
/// Signs synchronously — the async wrapper resolves immediately.
/// Used for server-to-server RPC where the signing key is local.
pub struct LocalSigner {
    signing_key: SigningKey,
    #[cfg(feature = "pq-hybrid")]
    pq_signing_key: Option<crate::crypto::pq::MlDsaSigningKey>,
}

impl LocalSigner {
    pub fn new(signing_key: SigningKey) -> Self {
        Self {
            signing_key,
            #[cfg(feature = "pq-hybrid")]
            pq_signing_key: {
                let (sk, _) = crate::crypto::pq::ml_dsa_generate_keypair();
                Some(sk)
            },
        }
    }

    #[cfg(feature = "pq-hybrid")]
    pub fn with_pq_key(mut self, key: crate::crypto::pq::MlDsaSigningKey) -> Self {
        self.pq_signing_key = Some(key);
        self
    }

    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Signer for LocalSigner {
    fn pubkey(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]> {
        use ed25519_dalek::Signer as _;
        Ok(self.signing_key.sign(canonical_bytes).to_bytes())
    }

    fn pq_pubkey(&self) -> Option<Vec<u8>> {
        #[cfg(feature = "pq-hybrid")]
        {
            self.pq_signing_key.as_ref().map(|sk| {
                let vk = ml_dsa::Keypair::verifying_key(sk);
                crate::crypto::pq::ml_dsa_vk_bytes(&vk)
            })
        }
        #[cfg(not(feature = "pq-hybrid"))]
        None
    }

    async fn pq_sign(&self, canonical_bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        #[cfg(feature = "pq-hybrid")]
        {
            return match &self.pq_signing_key {
                Some(sk) => Ok(Some(crate::crypto::pq::ml_dsa_sign(sk, canonical_bytes))),
                None => Ok(None),
            };
        }
        #[cfg(not(feature = "pq-hybrid"))]
        {
            let _ = canonical_bytes;
            Ok(None)
        }
    }
}

/// Signer backed by an `IdentityProvider`-derived `SigningIdentity`.
///
/// Bridges the `IdentityProvider` abstraction to the `Signer` trait
/// used by `RpcClientImpl`. Created via `IdentityProvider::identity_open()`.
pub struct IdentitySigner {
    identity_handle: Box<dyn SigningIdentity>,
}

impl IdentitySigner {
    pub fn new(identity_handle: Box<dyn SigningIdentity>) -> Self {
        Self { identity_handle }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Signer for IdentitySigner {
    fn pubkey(&self) -> [u8; 32] {
        self.identity_handle.pubkey()
    }

    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]> {
        self.identity_handle.sign(canonical_bytes).await
    }
}

// ============================================================================
// WASM signer — delegates to JS callback (aegis-vault)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm {
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use js_sys::Function;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    use crate::transport_traits::Signer;

    /// WASM signer that delegates to a JavaScript async callback.
    ///
    /// The callback signature is `(canonicalBytes: Uint8Array) => Promise<Uint8Array>`.
    /// The signing key never enters this WASM module — it lives in aegis-vault,
    /// a cross-origin iframe, or a hardware token.
    pub struct JsSigner {
        pubkey: [u8; 32],
        sign_fn: Function,
    }

    // SAFETY: wasm32 is single-threaded. js_sys::Function is !Send only because
    // the JS spec doesn't define cross-thread semantics, but there are no threads.
    unsafe impl Send for JsSigner {}
    unsafe impl Sync for JsSigner {}

    impl JsSigner {
        pub fn new(pubkey: &[u8], sign_fn: Function) -> Result<Self> {
            if pubkey.len() != 32 {
                return Err(anyhow!("signer_pubkey must be 32 bytes, got {}", pubkey.len()));
            }
            let mut pk = [0u8; 32];
            pk.copy_from_slice(pubkey);
            Ok(Self { pubkey: pk, sign_fn })
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl Signer for JsSigner {
        fn pubkey(&self) -> [u8; 32] {
            self.pubkey
        }

        async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]> {
            let canonical_js = js_sys::Uint8Array::from(canonical_bytes);
            let promise_jsvalue = self
                .sign_fn
                .call1(&JsValue::NULL, &canonical_js)
                .map_err(|e| anyhow!("sign callback threw: {:?}", e))?;
            let promise: js_sys::Promise = promise_jsvalue
                .dyn_into()
                .map_err(|_| anyhow!("sign callback must return a Promise<Uint8Array>"))?;
            let signature_jsvalue = JsFuture::from(promise)
                .await
                .map_err(|e| anyhow!("sign callback rejected: {:?}", e))?;
            let signature_js: js_sys::Uint8Array = signature_jsvalue
                .dyn_into()
                .map_err(|_| anyhow!("sign callback must resolve to a Uint8Array"))?;
            let signature_vec = signature_js.to_vec();
            if signature_vec.len() != 64 {
                return Err(anyhow!(
                    "signature must be 64 bytes, got {}",
                    signature_vec.len()
                ));
            }
            let mut signature = [0u8; 64];
            signature.copy_from_slice(&signature_vec);
            Ok(signature)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::JsSigner;
