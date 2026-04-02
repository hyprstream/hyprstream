//! Secure event subscriber with per-prefix group key management.
//!
//! Handles receiving encrypted events, managing group keys per-prefix,
//! and performing rekey operations as part of Phase 7 (Secure Event Transport).
//!
//! # Design
//!
//! Each prefix the subscriber joins has its own `PrefixState`:
//! - Ephemeral Ristretto255 keypair for DH-based key agreement
//! - `KeyRing` holding the current group key and an optional previous key
//!   during grace periods after rekey
//!
//! Key commitment (truncated HMAC) enables fast rejection before AEAD,
//! avoiding expensive decryption attempts with wrong keys.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tokio::time::Instant;
use tracing::debug;
use zeroize::Zeroizing;

use hyprstream_rpc::crypto::event_crypto::{
    check_key_commitment, decrypt_event_full, derive_wrap_key, unwrap_group_key,
};
use hyprstream_rpc::crypto::generate_ephemeral_keypair;

/// Duration after rekey during which the previous key is still accepted.
const DEFAULT_GRACE_PERIOD: std::time::Duration = std::time::Duration::from_secs(30);

/// Holds the current and previous group key for a prefix.
struct KeyRing {
    current: Zeroizing<[u8; 32]>,
    previous: Option<Zeroizing<[u8; 32]>>,
    grace_until: Option<Instant>,
}

impl KeyRing {
    fn new(key: Zeroizing<[u8; 32]>) -> Self {
        Self {
            current: key,
            previous: None,
            grace_until: None,
        }
    }

    /// Return true if the previous key's grace period has expired.
    fn is_grace_expired(&self) -> bool {
        match self.grace_until {
            Some(deadline) => Instant::now() >= deadline,
            None => true,
        }
    }

    /// Drop the previous key if its grace period has expired.
    fn gc_previous(&mut self) {
        if self.is_grace_expired() {
            self.previous = None;
            self.grace_until = None;
        }
    }
}

/// Per-prefix subscriber state: ephemeral DH keypair + key ring.
struct PrefixState {
    /// Our ephemeral secret scalar bytes (zeroized on drop).
    ephemeral_secret: Zeroizing<[u8; 32]>,
    /// Our ephemeral public key bytes.
    ephemeral_pubkey: [u8; 32],
    /// Publisher's public key bytes (needed for DH wrap key derivation).
    publisher_pubkey: [u8; 32],
    /// Current and previous group keys.
    ring: KeyRing,
}

/// A pending rekey message sent over the rekey channel.
#[derive(Debug)]
pub struct RekeyEvent {
    /// The prefix this rekey applies to.
    pub prefix: String,
    /// The new group key (zeroized on drop).
    pub new_key: Zeroizing<[u8; 32]>,
    /// When the new key becomes effective.
    pub effective_at: Instant,
}

/// Secure event subscriber managing per-prefix encrypted event reception.
///
/// Thread-safe: prefix state is behind an `RwLock` so multiple async tasks
/// can decrypt concurrently while rekey operations acquire exclusive access.
pub struct SecureEventSubscriber {
    prefixes: Arc<RwLock<HashMap<String, PrefixState>>>,
    rekey_tx: mpsc::Sender<RekeyEvent>,
}

impl SecureEventSubscriber {
    /// Create a new subscriber with a channel for rekey notifications.
    ///
    /// Returns `(subscriber, rekey_receiver)`.
    pub fn new(channel_size: usize) -> (Self, mpsc::Receiver<RekeyEvent>) {
        let (tx, rx) = mpsc::channel(channel_size);
        let subscriber = Self {
            prefixes: Arc::new(RwLock::new(HashMap::new())),
            rekey_tx: tx,
        };
        (subscriber, rx)
    }

    /// Join a prefix: generate an ephemeral keypair, unwrap the initial group key.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The topic prefix to subscribe to
    /// * `publisher_pubkey` - Publisher's Ristretto255 public key (32 bytes)
    /// * `wrapped_key_blob` - Wrapped group key blob from the publisher
    ///
    /// The wrap key is derived via DH between our ephemeral secret and the
    /// publisher's public key. The subscriber hash used for AAD is
    /// `blake3::hash(ephemeral_pubkey)`.
    pub async fn join_prefix(
        &self,
        prefix: &str,
        publisher_pubkey: &[u8; 32],
        wrapped_key_blob: &[u8],
    ) -> Result<[u8; 32], String> {
        // Generate ephemeral Ristretto255 keypair
        let (eph_secret, eph_public) = generate_ephemeral_keypair();
        let eph_pubkey_bytes = eph_public.to_bytes();
        let eph_secret_bytes: [u8; 32] = eph_secret.scalar().to_bytes();

        // Compute DH shared secret via raw scalar * point
        let shared_secret = Zeroizing::new(hyprstream_rpc::crypto::ristretto_dh_raw(
            &eph_secret_bytes,
            publisher_pubkey,
        )
        .map_err(|e| format!("DH failed: {e}"))?);

        // Derive wrap key from DH shared secret
        let wrap_key = derive_wrap_key(&shared_secret, publisher_pubkey, &eph_pubkey_bytes);

        // Subscriber hash = blake3(our ephemeral pubkey)
        let sub_hash: [u8; 32] = *blake3::hash(&eph_pubkey_bytes).as_bytes();

        // Unwrap the group key
        let group_key = unwrap_group_key(&wrap_key, wrapped_key_blob, &sub_hash, prefix)?;

        let state = PrefixState {
            ephemeral_secret: Zeroizing::new(eph_secret_bytes),
            ephemeral_pubkey: eph_pubkey_bytes,
            publisher_pubkey: *publisher_pubkey,
            ring: KeyRing::new(group_key),
        };

        let pubkey_out = eph_pubkey_bytes;
        self.prefixes
            .write()
            .await
            .insert(prefix.to_owned(), state);
        Ok(pubkey_out)
    }

    /// Try to decrypt an encrypted event.
    ///
    /// Attempts decryption with the current key first, then the previous key
    /// (if within grace period). Uses key commitment for fast rejection.
    ///
    /// Keys are cloned from the `RwLock` before trial-decrypt to avoid holding
    /// the lock across crypto operations.
    pub async fn try_decrypt(
        &self,
        prefix: &str,
        tag: &[u8],
        ciphertext: &[u8],
        nonce: &[u8; 12],
        key_commitment: &[u8; 16],
    ) -> Result<Vec<u8>, String> {
        // Clone keys under read lock, then release
        let (current_key, previous_key) = {
            let prefixes = self.prefixes.read().await;
            let state = prefixes
                .get(prefix)
                .ok_or_else(|| format!("not joined to prefix: {prefix}"))?;

            let current = state.ring.current.clone();
            let previous = if !state.ring.is_grace_expired() {
                state.ring.previous.clone()
            } else {
                None
            };
            (current, previous)
        };

        // Try current key: fast rejection via commitment check
        if check_key_commitment(&current_key, nonce, key_commitment) {
            match decrypt_event_full(&current_key, nonce, tag, ciphertext, prefix) {
                Ok(plaintext) => return Ok(plaintext),
                Err(e) => {
                    debug!(prefix, "current key commitment matched but decrypt failed: {e}");
                }
            }
        }

        // Try previous key if within grace period
        if let Some(ref prev_key) = previous_key {
            if check_key_commitment(prev_key, nonce, key_commitment) {
                match decrypt_event_full(prev_key, nonce, tag, ciphertext, prefix) {
                    Ok(plaintext) => {
                        debug!(prefix, "decrypted with previous key (grace period)");
                        return Ok(plaintext);
                    }
                    Err(e) => {
                        debug!(prefix, "previous key commitment matched but decrypt failed: {e}");
                    }
                }
            }
        }

        Err("decryption failed: no matching key".to_owned())
    }

    /// Handle a rekey: trial-decrypt the wrapped key entries and send the
    /// new key to the rekey channel.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix being rekeyed
    /// * `wrapped_blobs` - List of wrapped key blobs (with opaque routing tags)
    /// * `new_publisher_pubkey` - The publisher's new ephemeral pubkey for this rekey
    /// * `effective_at` - When the new key should become active
    pub async fn handle_rekey(
        &self,
        prefix: &str,
        wrapped_blobs: &[Vec<u8>],
        new_publisher_pubkey: &[u8; 32],
        effective_at: Instant,
    ) -> Result<(), String> {
        // Read our state (clone what we need)
        let (eph_secret, eph_pubkey) = {
            let prefixes = self.prefixes.read().await;
            let state = prefixes
                .get(prefix)
                .ok_or_else(|| format!("not joined to prefix: {prefix}"))?;
            (
                state.ephemeral_secret.clone(),
                state.ephemeral_pubkey,
            )
        };

        let sub_hash: [u8; 32] = *blake3::hash(&eph_pubkey).as_bytes();

        // Derive wrap key using the NEW publisher ephemeral pubkey
        let shared_secret = Zeroizing::new(hyprstream_rpc::crypto::ristretto_dh_raw(
            &eph_secret,
            new_publisher_pubkey,
        )
        .map_err(|e| format!("DH failed during rekey: {e}"))?);

        let wrap_key = derive_wrap_key(&shared_secret, new_publisher_pubkey, &eph_pubkey);

        // Trial-decrypt each wrapped blob to find ours (O(N) per rekey, infrequent)
        for wrapped_blob in wrapped_blobs {
            if let Ok(new_key) = unwrap_group_key(&wrap_key, wrapped_blob, &sub_hash, prefix) {
                // Update stored publisher pubkey for future rekeys
                {
                    let mut prefixes = self.prefixes.write().await;
                    if let Some(state) = prefixes.get_mut(prefix) {
                        state.publisher_pubkey = *new_publisher_pubkey;
                    }
                }

                // Send rekey event through channel
                self.rekey_tx
                    .send(RekeyEvent {
                        prefix: prefix.to_owned(),
                        new_key,
                        effective_at,
                    })
                    .await
                    .map_err(|_| "rekey channel closed".to_owned())?;

                return Ok(());
            }
        }

        Err("no wrapped entry found for our key".to_owned())
    }

    /// Promote a pending key to current, demoting current to previous
    /// with a grace period.
    ///
    /// Should be called at (or after) the `effective_at` time from a `RekeyEvent`.
    pub async fn promote_key(
        &self,
        prefix: &str,
        new_key: Zeroizing<[u8; 32]>,
        grace_period: Option<std::time::Duration>,
    ) -> Result<(), String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("not joined to prefix: {prefix}"))?;

        let grace = grace_period.unwrap_or(DEFAULT_GRACE_PERIOD);
        let old_current = std::mem::replace(&mut state.ring.current, new_key);
        state.ring.previous = Some(old_current);
        state.ring.grace_until = Some(Instant::now() + grace);

        debug!(prefix, ?grace, "promoted new key, old key in grace period");
        Ok(())
    }

    /// Garbage-collect expired previous keys across all prefixes.
    pub async fn gc_expired_keys(&self) {
        let mut prefixes = self.prefixes.write().await;
        for (prefix, state) in prefixes.iter_mut() {
            if state.ring.previous.is_some() && state.ring.is_grace_expired() {
                state.ring.gc_previous();
                debug!(prefix, "expired previous key removed");
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::event_crypto::{encrypt_event, wrap_group_key, EventPrivacy};

    /// Helper: set up a publisher keypair, group key, and wrap it for a subscriber.
    ///
    /// Returns `(publisher_secret_bytes, publisher_pubkey_bytes, group_key, wrapped_blob, sub_pubkey_bytes)`.
    fn setup_publisher_and_wrap(
        prefix: &str,
        sub_pubkey: &[u8; 32],
        group_key: &[u8; 32],
        pub_secret_bytes: &[u8; 32],
        pub_pubkey_bytes: &[u8; 32],
    ) -> Vec<u8> {
        // DH: publisher side
        let shared_secret =
            hyprstream_rpc::crypto::ristretto_dh_raw(pub_secret_bytes, sub_pubkey).unwrap();
        let wrap_key = derive_wrap_key(&shared_secret, sub_pubkey, pub_pubkey_bytes);
        let sub_hash: [u8; 32] = *blake3::hash(sub_pubkey).as_bytes();
        wrap_group_key(&wrap_key, group_key, &sub_hash, prefix).unwrap()
    }

    #[tokio::test]
    async fn test_join_prefix_and_decrypt() {
        let prefix = "serve:model:test";
        let group_key = [0xAAu8; 32];

        // Generate publisher keypair
        let (pub_secret, pub_public) = generate_ephemeral_keypair();
        let pub_secret_bytes = pub_secret.scalar().to_bytes();
        let pub_pubkey_bytes = pub_public.to_bytes();

        // We need the subscriber's ephemeral pubkey to wrap the key, but
        // join_prefix generates it internally. So we do the wrapping inside
        // a two-step dance: first join to get the pubkey, then manually
        // set up. Instead, we'll test with a known keypair by directly
        // constructing state.

        // Generate subscriber keypair manually
        let (sub_secret, sub_public) = generate_ephemeral_keypair();
        let sub_secret_bytes = sub_secret.scalar().to_bytes();
        let sub_pubkey_bytes = sub_public.to_bytes();

        // Wrap group key from publisher side
        let wrapped = setup_publisher_and_wrap(
            prefix,
            &sub_pubkey_bytes,
            &group_key,
            &pub_secret_bytes,
            &pub_pubkey_bytes,
        );

        // Manually construct subscriber with known keypair
        let (subscriber, _rx) = SecureEventSubscriber::new(16);

        // Derive the wrap key subscriber-side and unwrap
        let shared_secret =
            hyprstream_rpc::crypto::ristretto_dh_raw(&sub_secret_bytes, &pub_pubkey_bytes)
                .unwrap();
        let wrap_key = derive_wrap_key(&shared_secret, &pub_pubkey_bytes, &sub_pubkey_bytes);
        let sub_hash: [u8; 32] = *blake3::hash(&sub_pubkey_bytes).as_bytes();
        let unwrapped = unwrap_group_key(&wrap_key, &wrapped, &sub_hash, prefix).unwrap();

        // Insert state directly
        {
            let mut prefixes = subscriber.prefixes.write().await;
            prefixes.insert(
                prefix.to_owned(),
                PrefixState {
                    ephemeral_secret: Zeroizing::new(sub_secret_bytes),
                    ephemeral_pubkey: sub_pubkey_bytes,
                    publisher_pubkey: pub_pubkey_bytes,
                    ring: KeyRing::new(unwrapped),
                },
            );
        }

        // Encrypt an event with the group key
        let plaintext = b"hello secure event";
        let (tag, ct, nonce, commitment) =
            encrypt_event(&group_key, prefix, plaintext, EventPrivacy::ZeroKnowledge).unwrap();

        // Decrypt via subscriber
        let decrypted = subscriber
            .try_decrypt(prefix, &tag, &ct, &nonce, &commitment)
            .await
            .unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_key_commitment_fast_rejection() {
        let prefix = "serve:model:reject";
        let good_key = Zeroizing::new([0xBBu8; 32]);
        let wrong_key = [0xCCu8; 32];

        let (subscriber, _rx) = SecureEventSubscriber::new(16);

        // Generate a dummy keypair for the state
        let (sub_secret, sub_public) = generate_ephemeral_keypair();
        {
            let mut prefixes = subscriber.prefixes.write().await;
            prefixes.insert(
                prefix.to_owned(),
                PrefixState {
                    ephemeral_secret: Zeroizing::new(sub_secret.scalar().to_bytes()),
                    ephemeral_pubkey: sub_public.to_bytes(),
                    publisher_pubkey: [0u8; 32],
                    ring: KeyRing::new(good_key),
                },
            );
        }

        // Encrypt with the wrong key
        let plaintext = b"wrong key event";
        let (tag, ct, nonce, commitment) =
            encrypt_event(&wrong_key, prefix, plaintext, EventPrivacy::ZeroKnowledge).unwrap();

        // Attempt decryption - should fail via commitment check (fast rejection)
        let result = subscriber
            .try_decrypt(prefix, &tag, &ct, &nonce, &commitment)
            .await;
        assert!(result.is_err(), "wrong key should be rejected");
        assert!(
            result.unwrap_err().contains("no matching key"),
            "should report no matching key"
        );
    }

    #[tokio::test]
    async fn test_rekey_and_grace_period() {
        let prefix = "serve:model:rekey";
        let old_key = [0xAAu8; 32];
        let new_key = [0xBBu8; 32];

        let (subscriber, _rx) = SecureEventSubscriber::new(16);

        // Set up with old key
        let (sub_secret, sub_public) = generate_ephemeral_keypair();
        {
            let mut prefixes = subscriber.prefixes.write().await;
            prefixes.insert(
                prefix.to_owned(),
                PrefixState {
                    ephemeral_secret: Zeroizing::new(sub_secret.scalar().to_bytes()),
                    ephemeral_pubkey: sub_public.to_bytes(),
                    publisher_pubkey: [0u8; 32],
                    ring: KeyRing::new(Zeroizing::new(old_key)),
                },
            );
        }

        // Promote new key with a generous grace period
        subscriber
            .promote_key(prefix, Zeroizing::new(new_key), Some(std::time::Duration::from_secs(60)))
            .await
            .unwrap();

        // Encrypt with OLD key - should still work during grace period
        let plaintext_old = b"old key event";
        let (tag_old, ct_old, nonce_old, commit_old) =
            encrypt_event(&old_key, prefix, plaintext_old, EventPrivacy::ZeroKnowledge).unwrap();

        let decrypted_old = subscriber
            .try_decrypt(prefix, &tag_old, &ct_old, &nonce_old, &commit_old)
            .await
            .unwrap();
        assert_eq!(decrypted_old, plaintext_old);

        // Encrypt with NEW key - should also work
        let plaintext_new = b"new key event";
        let (tag_new, ct_new, nonce_new, commit_new) =
            encrypt_event(&new_key, prefix, plaintext_new, EventPrivacy::ZeroKnowledge).unwrap();

        let decrypted_new = subscriber
            .try_decrypt(prefix, &tag_new, &ct_new, &nonce_new, &commit_new)
            .await
            .unwrap();
        assert_eq!(decrypted_new, plaintext_new);
    }

    #[tokio::test]
    async fn test_expired_grace_rejects_old_key() {
        let prefix = "serve:model:expired";
        let old_key = [0xAAu8; 32];
        let new_key = [0xBBu8; 32];

        let (subscriber, _rx) = SecureEventSubscriber::new(16);

        // Set up with old key
        let (sub_secret, sub_public) = generate_ephemeral_keypair();
        {
            let mut prefixes = subscriber.prefixes.write().await;
            prefixes.insert(
                prefix.to_owned(),
                PrefixState {
                    ephemeral_secret: Zeroizing::new(sub_secret.scalar().to_bytes()),
                    ephemeral_pubkey: sub_public.to_bytes(),
                    publisher_pubkey: [0u8; 32],
                    ring: KeyRing::new(Zeroizing::new(old_key)),
                },
            );
        }

        // Promote with zero grace period (immediately expired)
        subscriber
            .promote_key(prefix, Zeroizing::new(new_key), Some(std::time::Duration::ZERO))
            .await
            .unwrap();

        // Force GC to drop the previous key
        // Sleep a tiny bit to ensure the instant has passed
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        subscriber.gc_expired_keys().await;

        // Encrypt with old key - should fail (grace expired, previous key dropped)
        let plaintext = b"old key after grace";
        let (tag, ct, nonce, commitment) =
            encrypt_event(&old_key, prefix, plaintext, EventPrivacy::ZeroKnowledge).unwrap();

        let result = subscriber
            .try_decrypt(prefix, &tag, &ct, &nonce, &commitment)
            .await;
        assert!(result.is_err(), "old key should be rejected after grace expiry");

        // New key should still work
        let plaintext_new = b"new key works";
        let (tag_new, ct_new, nonce_new, commit_new) =
            encrypt_event(&new_key, prefix, plaintext_new, EventPrivacy::ZeroKnowledge).unwrap();

        let decrypted = subscriber
            .try_decrypt(prefix, &tag_new, &ct_new, &nonce_new, &commit_new)
            .await
            .unwrap();
        assert_eq!(decrypted, plaintext_new);
    }
}
