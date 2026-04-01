//! Secure event publisher for encrypted event transport.
//!
//! Uses group key encryption via AES-256-GCM, with key commitment for fast rejection.
//! Events are signed with Ed25519 before encryption for publisher authentication.
//!
//! # Key Agreement
//!
//! Publisher and subscriber both use Ristretto255 for ephemeral keypairs and DH.
//! The publisher performs `ristretto_dh_raw(publisher_secret, subscriber_pubkey)` to
//! derive a shared secret, which is then fed into `derive_wrap_key` to produce the
//! AES-256-GCM key for wrapping the group key.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use tokio::sync::RwLock;
use zeroize::Zeroizing;

use hyprstream_rpc::crypto::event_crypto::{
    EventPrivacy, encrypt_event, build_event_sig_message,
    derive_wrap_key, wrap_group_key,
};
use hyprstream_rpc::crypto::backend::keyed_mac;

/// Maximum key lifetime (24 hours). Keys MUST be rotated before this.
pub const MAX_KEY_LIFETIME: Duration = Duration::from_secs(86400);

/// Default rotation interval (1 hour).
pub const DEFAULT_ROTATION_INTERVAL: Duration = Duration::from_secs(3600);

/// Grace period for old key acceptance after rotation (120 seconds).
pub const GRACE_PERIOD: Duration = Duration::from_secs(120);

/// Rekey policy configuration.
#[derive(Clone, Debug)]
pub enum RekeyPolicy {
    /// Rotate on fixed schedule. Revocations deferred to next rotation.
    Scheduled { interval: Duration },
    /// Rotate immediately on revocation.
    Immediate,
    /// Scheduled with jitter for timing attack resistance.
    Jittered { interval: Duration, jitter: Duration },
}

impl RekeyPolicy {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Scheduled { interval } | Self::Jittered { interval, .. } => {
                if *interval > MAX_KEY_LIFETIME {
                    return Err(format!(
                        "interval {:?} exceeds MAX_KEY_LIFETIME ({:?})",
                        interval, MAX_KEY_LIFETIME
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// State for a single topic prefix's group key.
struct GroupKeyState {
    current: Zeroizing<[u8; 32]>,
    pending: Option<PendingRekey>,
    created_at: Instant,
}

struct PendingRekey {
    new_key: Zeroizing<[u8; 32]>,
    effective_at: Instant,
    /// New ephemeral keypair to promote alongside the group key.
    new_ephemeral_secret: Zeroizing<[u8; 32]>,
    new_ephemeral_pubkey: [u8; 32],
}

/// Per-prefix publisher state.
struct PrefixState {
    key_state: GroupKeyState,
    /// Publisher's ephemeral Ristretto255 secret scalar bytes.
    ephemeral_secret: Zeroizing<[u8; 32]>,
    /// Publisher's ephemeral Ristretto255 public key bytes.
    ephemeral_pubkey: [u8; 32],
    /// Known subscriber pubkeys (hash -> pubkey).
    subscribers: HashMap<[u8; 32], [u8; 32]>,
}

/// Secure event publisher that encrypts events with group keys.
///
/// Each topic prefix has its own group key and Ristretto255 ephemeral keypair.
/// Events are Ed25519-signed before encryption for publisher authentication.
pub struct SecureEventPublisher {
    /// Per-prefix state.
    prefixes: Arc<RwLock<HashMap<String, PrefixState>>>,
    /// Ed25519 signing key for event signatures.
    signing_key: SigningKey,
    /// Privacy mode (deployment-wide).
    privacy_mode: EventPrivacy,
    /// Rekey policy.
    rekey_policy: RekeyPolicy,
}

impl SecureEventPublisher {
    /// Create a new SecureEventPublisher.
    pub fn new(
        signing_key: SigningKey,
        privacy_mode: EventPrivacy,
        rekey_policy: RekeyPolicy,
    ) -> Result<Self, String> {
        rekey_policy.validate()?;
        Ok(Self {
            prefixes: Arc::new(RwLock::new(HashMap::new())),
            signing_key,
            privacy_mode,
            rekey_policy,
        })
    }

    /// Register a new topic prefix with a fresh group key.
    ///
    /// Returns the publisher's Ristretto255 ephemeral pubkey for this prefix.
    pub async fn register_prefix(&self, prefix: &str) -> Result<[u8; 32], String> {
        let mut group_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *group_key);

        // Generate Ristretto255 ephemeral keypair (same group as subscriber)
        let (eph_secret, eph_public) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let ephemeral_secret = Zeroizing::new(eph_secret.scalar().to_bytes());
        let ephemeral_pubkey = eph_public.to_bytes();

        let state = PrefixState {
            key_state: GroupKeyState {
                current: group_key,
                pending: None,
                created_at: Instant::now(),
            },
            ephemeral_secret,
            ephemeral_pubkey,
            subscribers: HashMap::new(),
        };

        self.prefixes.write().await.insert(prefix.to_owned(), state);
        Ok(ephemeral_pubkey)
    }

    /// Wrap the current group key for a new subscriber.
    ///
    /// Performs Ristretto255 DH to derive a shared secret, then derives a wrap key.
    /// Returns the wrapped key blob (opaque, suitable for deposit to PolicyService).
    pub async fn wrap_for_subscriber(
        &self,
        prefix: &str,
        subscriber_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>, String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{}' not registered", prefix))?;

        // Promote pending key if past effective_at (consistent with publish/wrap_for_new_subscribers)
        maybe_promote_pending(state);

        let sub_hash = hash_pubkey(subscriber_pubkey);

        // Use the current (promoted) group key
        let group_key = &state.key_state.current;

        // DH(publisher_secret, subscriber_pubkey) → shared secret
        let shared_secret = Zeroizing::new(hyprstream_rpc::crypto::ristretto_dh_raw(
            &state.ephemeral_secret,
            subscriber_pubkey,
        ).map_err(|e| format!("DH failed: {e}"))?);

        let wrap_key = derive_wrap_key(&shared_secret, subscriber_pubkey, &state.ephemeral_pubkey);
        wrap_group_key(&wrap_key, group_key, &sub_hash, prefix)
    }

    /// Wrap group key for multiple new subscribers.
    ///
    /// Returns Vec of (sub_pubkey_hash, wrapped_blob) pairs ready for PolicyService deposit.
    pub async fn wrap_for_new_subscribers(
        &self,
        prefix: &str,
        subscriber_pubkeys: &[[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{}' not registered", prefix))?;

        // Promote pending key if effective (write lock held, safe to mutate)
        maybe_promote_pending(state);

        let mut results = Vec::with_capacity(subscriber_pubkeys.len());
        for sub_pubkey in subscriber_pubkeys {
            let sub_hash = hash_pubkey(sub_pubkey);

            let shared_secret = Zeroizing::new(hyprstream_rpc::crypto::ristretto_dh_raw(
                &state.ephemeral_secret,
                sub_pubkey,
            ).map_err(|e| format!("DH failed: {e}"))?);

            let wrap_key = derive_wrap_key(&shared_secret, sub_pubkey, &state.ephemeral_pubkey);
            let wrapped =
                wrap_group_key(&wrap_key, &state.key_state.current, &sub_hash, prefix)?;
            state.subscribers.insert(sub_hash, *sub_pubkey);
            results.push((sub_hash, wrapped));
        }
        Ok(results)
    }

    /// Publish an encrypted event.
    ///
    /// Returns the encrypted payload components and Ed25519 signature.
    /// The caller is responsible for sending these via StreamService.
    pub async fn publish(
        &self,
        topic: &str,
        payload: &[u8],
    ) -> Result<EncryptedEvent, String> {
        let prefix = topic.split('.').next().unwrap_or(topic);

        // Take write lock to potentially promote pending key
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{}' not registered", prefix))?;

        // Promote pending key to current if past effective_at
        maybe_promote_pending(state);

        let group_key: &[u8; 32] = &state.key_state.current;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        // Sign the event (before encryption)
        let sig_message = build_event_sig_message(topic, payload, timestamp);
        let signature = self.signing_key.sign(&sig_message);

        // Encrypt
        let (tag, ciphertext, nonce, commitment) =
            encrypt_event(group_key, prefix, payload, self.privacy_mode)?;

        // Compute LK-mode tag (keyed HMAC of prefix under group key)
        let lk_tag = match self.privacy_mode {
            EventPrivacy::LimitedKnowledge => {
                let mac = keyed_mac(group_key, prefix.as_bytes());
                mac[..16].to_vec()
            }
            EventPrivacy::ZeroKnowledge => vec![],
        };

        Ok(EncryptedEvent {
            topic: topic.to_owned(),
            tag,
            ciphertext,
            nonce,
            key_commitment: commitment,
            lk_tag,
            signature: signature.to_bytes().to_vec(),
            publisher_pubkey: self.signing_key.verifying_key().to_bytes(),
            timestamp,
        })
    }

    /// Check if a prefix needs key rotation based on the rekey policy.
    pub async fn needs_rotation(&self, prefix: &str) -> bool {
        let prefixes = self.prefixes.read().await;
        if let Some(state) = prefixes.get(prefix) {
            let age = state.key_state.created_at.elapsed();
            match &self.rekey_policy {
                RekeyPolicy::Scheduled { interval } => age >= *interval,
                RekeyPolicy::Immediate => false,
                RekeyPolicy::Jittered { interval, jitter } => {
                    let jitter_offset = {
                        let h = hash_prefix_bytes(prefix.as_bytes());
                        let frac =
                            (h[0] as u64 * 256 + h[1] as u64) as f64 / 65536.0;
                        Duration::from_secs_f64(jitter.as_secs_f64() * frac)
                    };
                    age >= *interval + jitter_offset
                }
            }
        } else {
            false
        }
    }

    /// Rotate the group key for a prefix.
    ///
    /// Generates new group key + ephemeral keypair, wraps new key for all known subscribers.
    /// The caller should broadcast a RekeyAnnouncement and deposit wrapped keys.
    ///
    /// Note: The new ephemeral keypair is stored alongside the pending key. The old
    /// ephemeral keypair is kept until the pending key is promoted (via `maybe_promote_pending`
    /// in `publish()` or `wrap_for_new_subscribers()`).
    pub async fn rotate_key(
        &self,
        prefix: &str,
        effective_delay: Duration,
    ) -> Result<RotationResult, String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{}' not registered", prefix))?;

        // Generate new group key
        let mut new_group_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *new_group_key);

        // Generate new Ristretto255 ephemeral keypair
        let (new_eph_secret, new_eph_public) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let new_ephemeral_secret = Zeroizing::new(new_eph_secret.scalar().to_bytes());
        let new_ephemeral_pubkey = new_eph_public.to_bytes();

        // Wrap new key for all known subscribers using the NEW ephemeral keypair
        let mut wrapped_keys = Vec::with_capacity(state.subscribers.len());
        for (sub_hash, sub_pubkey) in &state.subscribers {
            let shared_secret = Zeroizing::new(hyprstream_rpc::crypto::ristretto_dh_raw(
                &new_ephemeral_secret,
                sub_pubkey,
            ).map_err(|e| format!("DH failed: {e}"))?);

            let wrap_key = derive_wrap_key(&shared_secret, sub_pubkey, &new_ephemeral_pubkey);
            let wrapped =
                wrap_group_key(&wrap_key, &new_group_key, sub_hash, prefix)?;
            let mut routing_tag = [0u8; 16];
            rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut routing_tag);
            wrapped_keys.push(WrappedKeyEntry {
                routing_tag,
                wrapped_blob: wrapped,
            });
        }

        let effective_at = Instant::now() + effective_delay;

        // Store pending key with its new ephemeral keypair.
        // Both are promoted atomically in maybe_promote_pending().
        // During the pending window, wrap_for_subscriber uses the CURRENT (old)
        // ephemeral keypair and current group key — new subscribers get a working
        // key that will be rekeyed at effective_at via the RekeyAnnouncement.
        state.key_state.pending = Some(PendingRekey {
            new_key: new_group_key,
            effective_at,
            new_ephemeral_secret: new_ephemeral_secret,
            new_ephemeral_pubkey,
        });

        let effective_at_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
            + effective_delay.as_millis() as i64;

        Ok(RotationResult {
            new_ephemeral_pubkey,
            wrapped_keys,
            effective_at_millis,
        })
    }

    /// Get the privacy mode.
    pub fn privacy_mode(&self) -> EventPrivacy {
        self.privacy_mode
    }

    /// Get the signing verifying key.
    pub fn verifying_key(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }
}

/// Select the active group key (pending if past effective_at, else current).
fn active_group_key(key_state: &GroupKeyState) -> &[u8; 32] {
    match &key_state.pending {
        Some(p) if Instant::now() >= p.effective_at => &p.new_key,
        _ => &key_state.current,
    }
}

/// Promote pending key + ephemeral keypair to current if past effective_at.
/// Must be called under write lock.
fn maybe_promote_pending(state: &mut PrefixState) {
    if let Some(pending) = state.key_state.pending.take() {
        if Instant::now() >= pending.effective_at {
            state.key_state.current = pending.new_key;
            state.key_state.created_at = Instant::now();
            state.ephemeral_secret = pending.new_ephemeral_secret;
            state.ephemeral_pubkey = pending.new_ephemeral_pubkey;
            // pending is now None (taken above)
        } else {
            // Not yet effective, put it back
            state.key_state.pending = Some(pending);
        }
    }
}

/// Result of encrypting an event.
#[derive(Debug, Clone)]
pub struct EncryptedEvent {
    pub topic: String,
    pub tag: Vec<u8>,
    pub ciphertext: Vec<u8>,
    pub nonce: [u8; 12],
    pub key_commitment: [u8; 16],
    /// LK mode routing tag (keyed HMAC of prefix). Empty in ZK mode.
    pub lk_tag: Vec<u8>,
    /// Ed25519 signature over the event.
    pub signature: Vec<u8>,
    /// Publisher's Ed25519 verifying key.
    pub publisher_pubkey: [u8; 32],
    /// Event timestamp (unix millis).
    pub timestamp: i64,
}

/// Result of a key rotation.
#[derive(Debug)]
pub struct RotationResult {
    pub new_ephemeral_pubkey: [u8; 32],
    pub wrapped_keys: Vec<WrappedKeyEntry>,
    pub effective_at_millis: i64,
}

/// A wrapped key entry for a single subscriber.
#[derive(Debug)]
pub struct WrappedKeyEntry {
    /// Random 16-byte routing tag (unlinkable across rekeys).
    pub routing_tag: [u8; 16],
    /// Opaque wrapped group key blob.
    pub wrapped_blob: Vec<u8>,
}

/// Hash a pubkey for use as subscriber identity.
fn hash_pubkey(pubkey: &[u8; 32]) -> [u8; 32] {
    *blake3::hash(pubkey).as_bytes()
}

/// Hash arbitrary prefix bytes for deterministic jitter derivation.
fn hash_prefix_bytes(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::event_crypto::{decrypt_event_full, unwrap_group_key};

    fn test_signing_key() -> SigningKey {
        let mut secret = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut secret);
        SigningKey::from_bytes(&secret)
    }

    #[tokio::test]
    async fn test_register_prefix() {
        let publisher = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();

        let pubkey = publisher.register_prefix("worker").await.unwrap();
        assert_ne!(pubkey, [0u8; 32]);
    }

    #[tokio::test]
    async fn test_publish_encrypt_event() {
        let publisher = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();

        publisher.register_prefix("worker").await.unwrap();

        let encrypted = publisher
            .publish("worker.sandbox1.started", b"test payload")
            .await
            .unwrap();
        assert!(!encrypted.ciphertext.is_empty() || !encrypted.tag.is_empty());
        assert_eq!(encrypted.nonce.len(), 12);
        assert_eq!(encrypted.key_commitment.len(), 16);
        assert!(encrypted.lk_tag.is_empty()); // ZK mode
    }

    #[tokio::test]
    async fn test_publish_lk_mode_has_tag() {
        let publisher = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::LimitedKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();

        publisher.register_prefix("worker").await.unwrap();

        let encrypted = publisher
            .publish("worker.sandbox1.started", b"payload")
            .await
            .unwrap();
        assert!(!encrypted.lk_tag.is_empty());
    }

    #[tokio::test]
    async fn test_wrap_for_subscriber_roundtrip() {
        // Verify that publisher wrap + subscriber unwrap produce the same group key
        let publisher = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();

        let pub_pubkey = publisher.register_prefix("worker").await.unwrap();

        // Generate subscriber Ristretto255 keypair
        let (sub_secret, sub_public) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let sub_secret_bytes = sub_secret.scalar().to_bytes();
        let sub_pubkey_bytes = sub_public.to_bytes();

        // Publisher wraps
        let wrapped = publisher
            .wrap_for_subscriber("worker", &sub_pubkey_bytes)
            .await
            .unwrap();
        assert!(wrapped.len() > 12);

        // Subscriber unwraps: DH(sub_secret, publisher_pubkey)
        let shared_secret = Zeroizing::new(hyprstream_rpc::crypto::ristretto_dh_raw(
            &sub_secret_bytes,
            &pub_pubkey,
        ).unwrap());
        let wrap_key = derive_wrap_key(&shared_secret, &pub_pubkey, &sub_pubkey_bytes);
        let sub_hash = hash_pubkey(&sub_pubkey_bytes);
        let group_key = unwrap_group_key(&wrap_key, &wrapped, &sub_hash, "worker").unwrap();

        // Verify: encrypt with the group key, then decrypt
        let plaintext = b"end-to-end test";
        let encrypted = publisher
            .publish("worker.test", plaintext)
            .await
            .unwrap();
        let decrypted = decrypt_event_full(
            &group_key,
            &encrypted.nonce,
            &encrypted.tag,
            &encrypted.ciphertext,
            "worker",
        ).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_rotate_key() {
        let publisher = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();

        publisher.register_prefix("worker").await.unwrap();

        // Add a subscriber
        let (_sub_secret, sub_public) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let sub_pubkey_bytes = sub_public.to_bytes();
        publisher
            .wrap_for_new_subscribers("worker", &[sub_pubkey_bytes])
            .await
            .unwrap();

        let result = publisher
            .rotate_key("worker", Duration::from_secs(60))
            .await
            .unwrap();
        assert_ne!(result.new_ephemeral_pubkey, [0u8; 32]);
        assert_eq!(result.wrapped_keys.len(), 1);
    }

    #[tokio::test]
    async fn test_pending_key_promotion() {
        let publisher = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();

        publisher.register_prefix("worker").await.unwrap();

        // Rotate with zero delay (immediately effective)
        publisher
            .rotate_key("worker", Duration::ZERO)
            .await
            .unwrap();

        // Publish should promote the pending key to current
        let encrypted = publisher
            .publish("worker.test", b"after rotation")
            .await
            .unwrap();
        assert!(!encrypted.tag.is_empty() || !encrypted.ciphertext.is_empty());

        // Verify pending is cleared after publish
        let prefixes = publisher.prefixes.read().await;
        let state = prefixes.get("worker").unwrap();
        assert!(state.key_state.pending.is_none(), "pending should be promoted to current");
    }

    #[tokio::test]
    async fn test_rekey_policy_validation() {
        let result = SecureEventPublisher::new(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(100_000),
            },
        );
        assert!(result.is_err());
    }
}
