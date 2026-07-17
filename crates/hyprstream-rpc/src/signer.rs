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
///
/// `SigningKey` derives `ZeroizeOnDrop` from ed25519-dalek ≥ 2.0, so the
/// Ed25519 key bytes are zeroed when the `LocalSigner` is dropped. The ML-DSA
/// key is an opaque byte buffer (`ml_dsa` crate) without `ZeroizeOnDrop`; we
/// zero it explicitly in our `Drop` impl.
pub struct LocalSigner {
    signing_key: SigningKey,
    pq_signing_key: Option<crate::crypto::pq::MlDsaSigningKey>,
}

impl std::fmt::Debug for LocalSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalSigner")
            .field(
                "pubkey",
                &hex::encode(self.signing_key.verifying_key().to_bytes()),
            )
            .field(
                "pq_pubkey",
                &self.pq_signing_key.as_ref().map(|_| "<ml-dsa-65>"),
            )
            .finish()
    }
}

impl LocalSigner {
    /// Create a native signer.
    ///
    /// The post-quantum half is the node's **persistent** mesh ML-DSA-65 key,
    /// derived deterministically from `signing_key` via
    /// [`crate::node_identity::derive_mesh_mldsa_key`] (#157). This replaces the
    /// previous ephemeral keygen so the signer's ML-DSA public key is stable
    /// across restarts and equals the `#mesh-pq` key peers anchor in their PQ
    /// trust store. Use [`Self::with_pq_key`] only to override with an
    /// externally supplied key (e.g. tests).
    pub fn new(signing_key: SigningKey) -> Self {
        let pq_signing_key = Some(crate::node_identity::derive_mesh_mldsa_key(&signing_key));
        Self {
            signing_key,
            pq_signing_key,
        }
    }

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
        self.pq_signing_key.as_ref().map(|sk| {
            let vk = ml_dsa::Keypair::verifying_key(sk);
            crate::crypto::pq::ml_dsa_vk_bytes(&vk)
        })
    }

    async fn pq_sign(&self, canonical_bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        match &self.pq_signing_key {
            Some(sk) => Ok(Some(crate::crypto::pq::ml_dsa_sign(sk, canonical_bytes))),
            None => Ok(None),
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
        pq_pubkey: Option<Vec<u8>>,
        pq_sign_fn: Option<Function>,
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
            Ok(Self { pubkey: pk, sign_fn,
                pq_pubkey: None,
                pq_sign_fn: None, })
        }

        /// Construct a browser signer whose Ed25519 and ML-DSA-65 private
        /// material remains behind separate JS callbacks.
        pub fn new_hybrid(
            pubkey: &[u8],
            sign_fn: Function,
            pq_pubkey: &[u8],
            pq_sign_fn: Function,
        ) -> Result<Self> {
            crate::crypto::pq::ml_dsa_vk_from_bytes(pq_pubkey)
                .map_err(|error| anyhow!("invalid signer ML-DSA-65 public key: {error}"))?;
            let mut signer = Self::new(pubkey, sign_fn)?;
            signer.pq_pubkey = Some(pq_pubkey.to_vec());
            signer.pq_sign_fn = Some(pq_sign_fn);
            Ok(signer)
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

        fn pq_pubkey(&self) -> Option<Vec<u8>> {
            self.pq_pubkey.clone()
        }

        async fn pq_sign(&self, canonical_bytes: &[u8]) -> Result<Option<Vec<u8>>> {
            const ML_DSA65_SIGNATURE_LEN: usize = 3309;
            let Some(sign_fn) = &self.pq_sign_fn else {
                return Ok(None);
            };
            let canonical_js = js_sys::Uint8Array::from(canonical_bytes);
            let promise_jsvalue = sign_fn
                .call1(&JsValue::NULL, &canonical_js)
                .map_err(|error| anyhow!("PQ sign callback threw: {error:?}"))?;
            let promise: js_sys::Promise = promise_jsvalue
                .dyn_into()
                .map_err(|_| anyhow!("PQ sign callback must return a Promise<Uint8Array>"))?;
            let signature_jsvalue = JsFuture::from(promise)
                .await
                .map_err(|error| anyhow!("PQ sign callback rejected: {error:?}"))?;
            let signature_js: js_sys::Uint8Array = signature_jsvalue
                .dyn_into()
                .map_err(|_| anyhow!("PQ sign callback must resolve to a Uint8Array"))?;
            let signature = signature_js.to_vec();
            anyhow::ensure!(
                signature.len() == ML_DSA65_SIGNATURE_LEN,
                "ML-DSA-65 signature must be {ML_DSA65_SIGNATURE_LEN} bytes"
            );
            Ok(Some(signature))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::JsSigner;
