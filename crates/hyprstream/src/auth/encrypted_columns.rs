//! Envelope encryption for relational `UserStore` value columns (#1377).
//!
//! A random root DEK is created per user and sealed to the deployment's age
//! recipient ring. Age plugin recipients and identity files are the same seam
//! used by the deployment trust mint, including KMS and YubiKey plugins. Value
//! columns use per-field keys derived from the root DEK. Deleting the user's
//! wrapped DEK therefore makes every remaining ciphertext cryptographically
//! unreadable without retaining a plaintext-key cache.
//!
//! [`DekSealer`] keeps the field cipher testable, while its production adapter
//! calls the same [`AgeRecipients`](super::age_seal::AgeRecipients) and
//! [`AgeIdentities`](super::age_seal::AgeIdentities) component used by the
//! deployment trust mint. The crypto design — per-user root DEK, HKDF-derived
//! per-field keys, AES-256-GCM-SIV with `(username, column, row-binding)` AAD —
//! remains independent of the deployment key-wrapping implementation.

use super::age_seal::{AgeIdentities, AgeRecipients};
use aes_gcm_siv::{
    aead::{Aead, KeyInit, Payload},
    Aes256GcmSiv, Nonce,
};
use anyhow::{anyhow, ensure, Context, Result};
use hkdf::Hkdf;
use rand::RngCore as _;
use sha2::Sha256;
use std::{ffi::OsString, path::PathBuf, sync::Arc};
use zeroize::{ZeroizeOnDrop, Zeroizing};

pub const AGE_RECIPIENTS_ENV: &str = "HYPRSTREAM_USERSTORE_AGE_RECIPIENTS";
pub const AGE_IDENTITIES_ENV: &str = "HYPRSTREAM_USERSTORE_AGE_IDENTITIES";

pub(crate) const ROOT_DEK_BYTES: usize = 32;
const NONCE_BYTES: usize = 12;
const TAG_BYTES: usize = 16;
const CIPHERTEXT_MAGIC: &[u8; 4] = b"HSC1";
const CIPHERTEXT_HEADER_BYTES: usize = CIPHERTEXT_MAGIC.len() + NONCE_BYTES;
const MAX_WRAPPED_DEK_BYTES: usize = 64 * 1024;
const MAX_COLUMN_PLAINTEXT_BYTES: usize = 1024 * 1024;
const FIELD_KEY_SALT: &[u8] = b"hyprstream.userstore.field-key.v1";
const FIELD_CONTEXT_DOMAIN: &[u8] = b"hyprstream.userstore.column.v1";

/// Deployment key-sealing configuration for relational credential storage.
///
/// Recipients may include age plugin recipients backed by a KMS or YubiKey.
/// Identities are age identity files, including plugin identity files.
///
/// Constructed at deployment-startup time from environment variables (see
/// [`Self::from_env`]) and passed to [`ColumnCipher::from_age_config`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserStoreEncryptionConfig {
    recipients: AgeRecipients,
    identities: AgeIdentities,
}

impl UserStoreEncryptionConfig {
    pub fn new(recipients: Vec<String>, identities: Vec<PathBuf>) -> Result<Self> {
        Ok(Self {
            recipients: AgeRecipients::new(recipients)
                .context("invalid UserStore age recipients")?,
            identities: AgeIdentities::new(identities)
                .context("invalid UserStore age identities")?,
        })
    }

    /// Load fail-closed deployment configuration.
    ///
    /// Recipients are comma- or newline-separated. Identities use the platform
    /// path-list separator (`:` on Unix).
    pub fn from_env() -> Result<Self> {
        let recipients = std::env::var(AGE_RECIPIENTS_ENV)
            .with_context(|| format!("{AGE_RECIPIENTS_ENV} is required"))?
            .split([',', '\n'])
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect();
        let identity_list: OsString = std::env::var_os(AGE_IDENTITIES_ENV)
            .ok_or_else(|| anyhow!("{AGE_IDENTITIES_ENV} is required"))?;
        let identities = std::env::split_paths(&identity_list).collect();
        Self::new(recipients, identities)
    }
}

/// Authenticated field domains. Lookup keys remain plaintext, but are bound as
/// AAD so ciphertext cannot be transplanted between users, columns, or keys.
///
/// Variants map 1:1 to the BYTEA value columns in the relational `UserStore`
/// schema (#1376 final). The profile variants carry individual PII columns
/// (name, email, external_id); the pubkey variants carry the per-row fingerprint
/// as AAD so two keys owned by the same user cannot be swapped.
#[derive(Clone, Copy, Debug)]
pub(crate) enum EncryptedColumn<'a> {
    /// `users.name BYTEA` — display name.
    ProfileName,
    /// `users.email BYTEA` — email address.
    ProfileEmail,
    /// `users.external_id BYTEA` — external identity ID.
    ProfileExternalId,
    /// `pubkeys.pubkey BYTEA` — Ed25519 verifying key bytes.
    PublicKey { fingerprint: &'a str },
    /// `pubkeys.label BYTEA` — optional user-supplied key label.
    PublicKeyLabel { fingerprint: &'a str },
    /// `pubkeys.pq_pubkey BYTEA` — bound ML-DSA-65 verifying key bytes.
    PqPublicKey { fingerprint: &'a str },
}

impl<'a> EncryptedColumn<'a> {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::ProfileName => b"users.name",
            Self::ProfileEmail => b"users.email",
            Self::ProfileExternalId => b"users.external_id",
            Self::PublicKey { .. } => b"pubkeys.pubkey",
            Self::PublicKeyLabel { .. } => b"pubkeys.label",
            Self::PqPublicKey { .. } => b"pubkeys.pq_pubkey",
        }
    }

    fn row_binding(self) -> Option<&'static str> {
        // This method is never used directly. Keeping the exhaustive match here
        // makes adding a new multi-row encrypted column require an explicit AAD
        // decision.
        match self {
            Self::ProfileName | Self::ProfileEmail | Self::ProfileExternalId => None,
            Self::PublicKey { .. } | Self::PublicKeyLabel { .. } | Self::PqPublicKey { .. } => {
                Some("fingerprint")
            }
        }
    }

    fn row_value(self) -> Option<&'a str> {
        match self {
            Self::ProfileName | Self::ProfileEmail | Self::ProfileExternalId => None,
            Self::PublicKey { fingerprint }
            | Self::PublicKeyLabel { fingerprint }
            | Self::PqPublicKey { fingerprint } => Some(fingerprint),
        }
    }
}

trait DekSealer: Send + Sync {
    fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>>;
    fn open(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>>;
}

/// Production adapter over the shared deployment trust-mint age seam.
struct TrustMintAgeDekSealer {
    config: UserStoreEncryptionConfig,
}

impl TrustMintAgeDekSealer {
    fn new(config: UserStoreEncryptionConfig) -> Self {
        Self { config }
    }
}

impl DekSealer for TrustMintAgeDekSealer {
    fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        self.config
            .recipients
            .seal(plaintext, MAX_WRAPPED_DEK_BYTES)
            .context("seal UserStore DEK through deployment trust")
    }

    fn open(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
        ensure!(
            !ciphertext.is_empty() && ciphertext.len() <= MAX_WRAPPED_DEK_BYTES,
            "wrapped UserStore DEK has invalid size"
        );
        let plaintext = self
            .config
            .identities
            .open(ciphertext, ROOT_DEK_BYTES)
            .context("open UserStore DEK through deployment trust")?;
        ensure!(
            plaintext.len() == ROOT_DEK_BYTES,
            "unwrapped UserStore DEK has invalid length"
        );
        Ok(plaintext)
    }
}

pub(crate) struct NewUserKey {
    pub root: Zeroizing<[u8; ROOT_DEK_BYTES]>,
    pub wrapped: Vec<u8>,
}

/// Column encryption facade. It deliberately has no plaintext DEK cache:
/// deleting a wrapped key immediately makes subsequent reads fail closed.
#[derive(Clone)]
pub(crate) struct ColumnCipher {
    sealer: Arc<dyn DekSealer>,
}

impl ColumnCipher {
    fn from_age_config(config: UserStoreEncryptionConfig) -> Self {
        Self::new(Arc::new(TrustMintAgeDekSealer::new(config)))
    }

    pub(crate) fn from_deployment_env() -> Result<Self> {
        Ok(Self::from_age_config(UserStoreEncryptionConfig::from_env()?))
    }

    fn new(sealer: Arc<dyn DekSealer>) -> Self {
        Self { sealer }
    }

    pub(crate) async fn create_user_key(&self) -> Result<NewUserKey> {
        let mut root = Zeroizing::new([0u8; ROOT_DEK_BYTES]);
        rand::rngs::OsRng.fill_bytes(&mut *root);
        let sealer = Arc::clone(&self.sealer);
        let seal_input = Zeroizing::new(root.to_vec());
        let wrapped = tokio::task::spawn_blocking(move || sealer.seal(&seal_input))
            .await
            .context("join age DEK sealing task")??;
        ensure!(
            !wrapped
                .windows(ROOT_DEK_BYTES)
                .any(|window| window == root.as_slice()),
            "age returned a wrapped DEK containing its plaintext"
        );
        Ok(NewUserKey { root, wrapped })
    }

    pub(crate) async fn open_user_key(
        &self,
        wrapped: &[u8],
    ) -> Result<Zeroizing<[u8; ROOT_DEK_BYTES]>> {
        let sealer = Arc::clone(&self.sealer);
        let wrapped = wrapped.to_vec();
        let plaintext = tokio::task::spawn_blocking(move || sealer.open(&wrapped))
            .await
            .context("join age DEK unsealing task")??;
        plaintext
            .as_slice()
            .try_into()
            .map(Zeroizing::new)
            .map_err(|_| anyhow!("unwrapped UserStore DEK has invalid length"))
    }

    pub(crate) fn encrypt(
        &self,
        root: &[u8; ROOT_DEK_BYTES],
        username: &str,
        column: EncryptedColumn<'_>,
        plaintext: &[u8],
    ) -> Result<Vec<u8>> {
        ensure!(
            plaintext.len() <= MAX_COLUMN_PLAINTEXT_BYTES,
            "UserStore column plaintext exceeds size limit"
        );
        let context = field_context(username, column)?;
        let key = derive_field_key(root, &context)?;
        let cipher = Aes256GcmSiv::new_from_slice(&*key)
            .map_err(|_| anyhow!("initialize UserStore column cipher"))?;
        let mut nonce = [0u8; NONCE_BYTES];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let encrypted = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad: &context,
                },
            )
            .map_err(|_| anyhow!("encrypt UserStore value column"))?;
        let mut output = Vec::with_capacity(CIPHERTEXT_HEADER_BYTES + encrypted.len());
        output.extend_from_slice(CIPHERTEXT_MAGIC);
        output.extend_from_slice(&nonce);
        output.extend_from_slice(&encrypted);
        Ok(output)
    }

    pub(crate) fn decrypt(
        &self,
        root: &[u8; ROOT_DEK_BYTES],
        username: &str,
        column: EncryptedColumn<'_>,
        ciphertext: &[u8],
    ) -> Result<Zeroizing<Vec<u8>>> {
        ensure!(
            ciphertext.len() >= CIPHERTEXT_HEADER_BYTES + TAG_BYTES
                && ciphertext.len()
                    <= CIPHERTEXT_HEADER_BYTES + MAX_COLUMN_PLAINTEXT_BYTES + TAG_BYTES,
            "UserStore column ciphertext has invalid size"
        );
        ensure!(
            ciphertext.starts_with(CIPHERTEXT_MAGIC),
            "UserStore column ciphertext has invalid version"
        );
        let context = field_context(username, column)?;
        let key = derive_field_key(root, &context)?;
        let cipher = Aes256GcmSiv::new_from_slice(&*key)
            .map_err(|_| anyhow!("initialize UserStore column cipher"))?;
        let nonce = Nonce::from_slice(&ciphertext[CIPHERTEXT_MAGIC.len()..CIPHERTEXT_HEADER_BYTES]);
        let plaintext = cipher
            .decrypt(
                nonce,
                Payload {
                    msg: &ciphertext[CIPHERTEXT_HEADER_BYTES..],
                    aad: &context,
                },
            )
            .map_err(|_| anyhow!("authenticate/decrypt UserStore value column"))?;
        Ok(Zeroizing::new(plaintext))
    }

    // ── Convenience: text/bytes seal/open for store backends ─────────────
    // These are backend-neutral. Any relational store (pglite, PostgresUserStore)
    // calls them at the BYTEA column boundary.

    /// Seal an optional text field for storage. `None` in → `None` out.
    pub(crate) fn seal_text(
        &self,
        root: &Zeroizing<[u8; ROOT_DEK_BYTES]>,
        username: &str,
        column: EncryptedColumn<'_>,
        value: Option<String>,
    ) -> Result<Option<Vec<u8>>> {
        match value {
            None => Ok(None),
            Some(text) => Ok(Some(self.encrypt(
                root,
                username,
                column,
                text.as_bytes(),
            )?)),
        }
    }

    /// Open an optional text field from stored ciphertext.
    pub(crate) fn open_text(
        &self,
        root: &Zeroizing<[u8; ROOT_DEK_BYTES]>,
        username: &str,
        column: EncryptedColumn<'_>,
        raw: Option<Vec<u8>>,
    ) -> Result<Option<String>> {
        match raw {
            None => Ok(None),
            Some(bytes) => self
                .decrypt(root, username, column, &bytes)
                .and_then(zeroizing_bytes_into_string)
                .map(Some),
        }
    }

    /// Seal raw bytes (e.g. pubkey material) for storage.
    pub(crate) fn seal_raw(
        &self,
        root: &Zeroizing<[u8; ROOT_DEK_BYTES]>,
        username: &str,
        column: EncryptedColumn<'_>,
        value: &[u8],
    ) -> Result<Vec<u8>> {
        self.encrypt(root, username, column, value)
    }

    /// Open raw bytes from storage.
    pub(crate) fn open_raw(
        &self,
        root: &Zeroizing<[u8; ROOT_DEK_BYTES]>,
        username: &str,
        column: EncryptedColumn<'_>,
        raw: &[u8],
    ) -> Result<Zeroizing<Vec<u8>>> {
        self.decrypt(root, username, column, raw)
    }

    /// Construct a cipher backed by an in-process test sealer.
    #[cfg(test)]
    pub(crate) fn test_cipher() -> Self {
        Self::new(Arc::new(TestDekSealer))
    }
}

/// In-process DekSealer for tests. Uses a fixed AES key.
#[cfg(test)]
struct TestDekSealer;

#[cfg(test)]
impl DekSealer for TestDekSealer {
    fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256GcmSiv::new_from_slice(&TEST_KEK)
            .map_err(|_| anyhow!("initialize test DEK sealer"))?;
        let mut nonce = [0u8; NONCE_BYTES];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let ct = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad: TEST_WRAP_AAD,
                },
            )
            .map_err(|_| anyhow!("seal test DEK"))?;
        let mut out = nonce.to_vec();
        out.extend_from_slice(&ct);
        Ok(out)
    }

    fn open(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
        ensure!(
            ciphertext.len() >= NONCE_BYTES + TAG_BYTES,
            "test wrapped key is truncated"
        );
        let cipher = Aes256GcmSiv::new_from_slice(&TEST_KEK)
            .map_err(|_| anyhow!("initialize test DEK sealer"))?;
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&ciphertext[..NONCE_BYTES]),
                Payload {
                    msg: &ciphertext[NONCE_BYTES..],
                    aad: TEST_WRAP_AAD,
                },
            )
            .map_err(|_| anyhow!("test wrapped key authentication failed"))?;
        Ok(Zeroizing::new(plaintext))
    }
}

#[cfg(test)]
const TEST_KEK: [u8; ROOT_DEK_BYTES] = [0xA7; ROOT_DEK_BYTES];
#[cfg(test)]
const TEST_WRAP_AAD: &[u8] = b"hyprstream.userstore.test-wrap.v1";

fn zeroizing_bytes_into_string(mut plaintext: Zeroizing<Vec<u8>>) -> Result<String> {
    match String::from_utf8(std::mem::take(&mut *plaintext)) {
        Ok(value) => Ok(value),
        Err(error) => {
            let _invalid_plaintext = Zeroizing::new(error.into_bytes());
            Err(anyhow!("decrypted UserStore text is not UTF-8"))
        }
    }
}

// aes-gcm-siv stores its key-generating schedule as `aes::Aes256`. This
// assertion fails compilation if feature resolution ever drops AES's
// `zeroize` implementation.
const _: fn() = || {
    fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}
    assert_zeroize_on_drop::<aes::Aes256>();
};

fn field_context(username: &str, column: EncryptedColumn<'_>) -> Result<Vec<u8>> {
    ensure!(
        !username.is_empty(),
        "encrypted UserStore username is empty"
    );
    let mut context = Vec::with_capacity(128);
    append_context_part(&mut context, FIELD_CONTEXT_DOMAIN)?;
    append_context_part(&mut context, username.as_bytes())?;
    append_context_part(&mut context, column.tag())?;
    if let Some(row_binding) = column.row_binding() {
        append_context_part(&mut context, row_binding.as_bytes())?;
        append_context_part(
            &mut context,
            column
                .row_value()
                .ok_or_else(|| anyhow!("encrypted UserStore row binding is absent"))?
                .as_bytes(),
        )?;
    }
    Ok(context)
}

fn append_context_part(output: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    let length = u32::try_from(value.len()).context("UserStore encryption context is too large")?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

fn derive_field_key(
    root: &[u8; ROOT_DEK_BYTES],
    context: &[u8],
) -> Result<Zeroizing<[u8; ROOT_DEK_BYTES]>> {
    let hkdf = Hkdf::<Sha256>::new(Some(FIELD_KEY_SALT), root);
    let mut key = Zeroizing::new([0u8; ROOT_DEK_BYTES]);
    hkdf.expand(context, &mut *key)
        .map_err(|_| anyhow!("derive UserStore per-field key"))?;
    Ok(key)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::op_ref)]
mod tests {
    use super::*;

    /// A sealer that can be made to fail on open, for testing fail-closed behavior.
    struct FailableTestSealer {
        fail_open: bool,
    }

    impl DekSealer for FailableTestSealer {
        fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
            let cipher = Aes256GcmSiv::new_from_slice(&TEST_KEK).unwrap();
            let mut nonce = [0u8; NONCE_BYTES];
            rand::rngs::OsRng.fill_bytes(&mut nonce);
            let encrypted = cipher
                .encrypt(
                    Nonce::from_slice(&nonce),
                    Payload {
                        msg: plaintext,
                        aad: TEST_WRAP_AAD,
                    },
                )
                .unwrap();
            let mut output = nonce.to_vec();
            output.extend_from_slice(&encrypted);
            Ok(output)
        }

        fn open(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
            if self.fail_open {
                anyhow::bail!("test key material is unavailable");
            }
            ensure!(
                ciphertext.len() >= NONCE_BYTES + TAG_BYTES,
                "test wrapped key is truncated"
            );
            let cipher = Aes256GcmSiv::new_from_slice(&TEST_KEK).unwrap();
            let plaintext = cipher
                .decrypt(
                    Nonce::from_slice(&ciphertext[..NONCE_BYTES]),
                    Payload {
                        msg: &ciphertext[NONCE_BYTES..],
                        aad: TEST_WRAP_AAD,
                    },
                )
                .map_err(|_| anyhow!("test wrapped key authentication failed"))?;
            Ok(Zeroizing::new(plaintext))
        }
    }

    fn cipher() -> ColumnCipher {
        ColumnCipher::test_cipher()
    }

    #[tokio::test]
    async fn envelope_and_field_round_trip_contains_no_plaintext() {
        let cipher = cipher();
        let key = cipher.create_user_key().await.unwrap();
        let email = Zeroizing::new(b"private.person@example.test".to_vec());
        let encrypted = cipher
            .encrypt(
                &key.root,
                "private-person",
                EncryptedColumn::ProfileEmail,
                &email,
            )
            .unwrap();

        assert!(!key.wrapped.windows(key.root.len()).any(|w| w == &*key.root));
        assert!(!encrypted.windows(email.len()).any(|w| w == &**email));
        let opened = cipher.open_user_key(&key.wrapped).await.unwrap();
        let decrypted = cipher
            .decrypt(
                &opened,
                "private-person",
                EncryptedColumn::ProfileEmail,
                &encrypted,
            )
            .unwrap();
        assert_eq!(&*decrypted, &**email);
    }

    #[tokio::test]
    async fn ciphertext_is_bound_to_user_column_and_lookup_key() {
        let cipher = cipher();
        let key = cipher.create_user_key().await.unwrap();
        let encrypted = cipher
            .encrypt(
                &key.root,
                "alice",
                EncryptedColumn::PublicKey {
                    fingerprint: "key-a",
                },
                &[0x42; 32],
            )
            .unwrap();

        assert!(cipher
            .decrypt(
                &key.root,
                "bob",
                EncryptedColumn::PublicKey {
                    fingerprint: "key-a"
                },
                &encrypted
            )
            .is_err());
        assert!(cipher
            .decrypt(
                &key.root,
                "alice",
                EncryptedColumn::PublicKeyLabel {
                    fingerprint: "key-a"
                },
                &encrypted
            )
            .is_err());
        assert!(cipher
            .decrypt(
                &key.root,
                "alice",
                EncryptedColumn::PublicKey {
                    fingerprint: "key-b"
                },
                &encrypted
            )
            .is_err());
    }

    #[tokio::test]
    async fn tampering_and_unavailable_key_material_fail_closed() {
        let cipher = cipher();
        let key = cipher.create_user_key().await.unwrap();
        let mut encrypted = cipher
            .encrypt(
                &key.root,
                "alice",
                EncryptedColumn::ProfileEmail,
                b"sensitive-id",
            )
            .unwrap();
        let last = encrypted.last_mut().unwrap();
        *last ^= 1;
        assert!(cipher
            .decrypt(
                &key.root,
                "alice",
                EncryptedColumn::ProfileEmail,
                &encrypted
            )
            .is_err());

        let unavailable = ColumnCipher::new(Arc::new(FailableTestSealer { fail_open: true }));
        assert!(unavailable.open_user_key(&key.wrapped).await.is_err());
    }

    #[test]
    fn configuration_rejects_absent_or_invalid_key_material() {
        assert!(UserStoreEncryptionConfig::new(Vec::new(), Vec::new()).is_err());
        assert!(UserStoreEncryptionConfig::new(
            vec!["age1valid-looking-recipient".to_owned()],
            vec![PathBuf::from("/definitely/not/an/identity")],
        )
        .is_err());
    }
}
