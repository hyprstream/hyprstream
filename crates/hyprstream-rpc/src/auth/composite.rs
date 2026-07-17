//! Atomic exact-pair authority for composite JWT signing and verification.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::crypto::pq::{MlDsaSigningKey, MlDsaVerifyingKey};

/// The local issuer path that is authorized to mint with an exact pair.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompositePairRole {
    OAuth,
    Policy,
}

/// Lifecycle state of an exact pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositePairState {
    Active,
    Drain,
}

/// One exact ML-DSA-65 + Ed25519 pair in the authoritative ledger.
#[derive(Clone)]
pub struct CompositeKeyPair {
    kid: String,
    ml_dsa: MlDsaVerifyingKey,
    ed25519: VerifyingKey,
    role: CompositePairRole,
    state: CompositePairState,
    not_before: i64,
    expires_at: i64,
    ml_dsa_signing: Option<Arc<MlDsaSigningKey>>,
    ed25519_signing: Option<Arc<SigningKey>>,
}

impl CompositeKeyPair {
    #[allow(clippy::too_many_arguments)]
    pub fn signing(
        kid: String,
        ml_dsa: Arc<MlDsaSigningKey>,
        ed25519: Arc<SigningKey>,
        role: CompositePairRole,
        state: CompositePairState,
        not_before: i64,
        expires_at: i64,
    ) -> Self {
        let ml_dsa_vk = ml_dsa::Keypair::verifying_key(&*ml_dsa).clone();
        let ed25519_vk = ed25519.verifying_key();
        Self {
            kid,
            ml_dsa: ml_dsa_vk,
            ed25519: ed25519_vk,
            role,
            state,
            not_before,
            expires_at,
            ml_dsa_signing: Some(ml_dsa),
            ed25519_signing: Some(ed25519),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verifying(
        kid: String,
        ml_dsa: MlDsaVerifyingKey,
        ed25519: VerifyingKey,
        role: CompositePairRole,
        state: CompositePairState,
        not_before: i64,
        expires_at: i64,
    ) -> Self {
        Self {
            kid,
            ml_dsa,
            ed25519,
            role,
            state,
            not_before,
            expires_at,
            ml_dsa_signing: None,
            ed25519_signing: None,
        }
    }

    pub fn kid(&self) -> &str {
        &self.kid
    }
    pub fn ml_dsa(&self) -> &MlDsaVerifyingKey {
        &self.ml_dsa
    }
    pub fn ed25519(&self) -> &VerifyingKey {
        &self.ed25519
    }
    pub fn role(&self) -> CompositePairRole {
        self.role
    }
    pub fn state(&self) -> CompositePairState {
        self.state
    }
    pub fn not_before(&self) -> i64 {
        self.not_before
    }
    pub fn expires_at(&self) -> i64 {
        self.expires_at
    }

    pub fn signing_keys(&self) -> Option<(Arc<MlDsaSigningKey>, Arc<SigningKey>)> {
        Some((
            self.ml_dsa_signing.as_ref()?.clone(),
            self.ed25519_signing.as_ref()?.clone(),
        ))
    }
}

/// One immutable view of the exact-pair ledger.
#[derive(Clone)]
pub struct CompositeKeySetSnapshot {
    version: u64,
    component_digest: Arc<str>,
    pairs: Arc<[CompositeKeyPair]>,
}

impl CompositeKeySetSnapshot {
    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn component_digest(&self) -> &str {
        &self.component_digest
    }
    pub fn pairs(&self) -> &[CompositeKeyPair] {
        &self.pairs
    }

    pub fn pair(&self, kid: &str) -> Option<&CompositeKeyPair> {
        self.pairs.iter().find(|pair| pair.kid == kid)
    }

    pub fn active_signing_pair(&self, role: CompositePairRole) -> Option<&CompositeKeyPair> {
        self.pairs.iter().find(|pair| {
            pair.role == role
                && pair.state == CompositePairState::Active
                && pair.ml_dsa_signing.is_some()
                && pair.ed25519_signing.is_some()
        })
    }
}

/// Process-shared atomic publication point for exact composite pairs.
pub struct CompositeKeySet {
    snapshot: RwLock<Arc<CompositeKeySetSnapshot>>,
    authority: RwLock<Option<CompositeAuthorityPaths>>,
}

#[derive(Clone)]
struct CompositeAuthorityPaths {
    ledger: PathBuf,
    committed: PathBuf,
    committed_ledger_prefix: PathBuf,
    ledger_lock: PathBuf,
}

impl Default for CompositeKeySet {
    fn default() -> Self {
        Self {
            snapshot: RwLock::new(Arc::new(CompositeKeySetSnapshot {
                version: 0,
                component_digest: Arc::from(""),
                pairs: Arc::from([]),
            })),
            authority: RwLock::new(None),
        }
    }
}

impl CompositeKeySet {
    pub fn snapshot(&self) -> Arc<CompositeKeySetSnapshot> {
        self.snapshot.read().clone()
    }

    pub fn configure_authority(
        &self,
        ledger: PathBuf,
        committed: PathBuf,
        committed_ledger_prefix: PathBuf,
        ledger_lock: PathBuf,
    ) {
        *self.authority.write() = Some(CompositeAuthorityPaths {
            ledger,
            committed,
            committed_ledger_prefix,
            ledger_lock,
        });
    }

    /// Return a signing snapshot only when it matches the committed disk generation.
    pub fn mint_snapshot(&self) -> anyhow::Result<Arc<CompositeKeySetSnapshot>> {
        use nix::fcntl::{flock, FlockArg};
        use std::os::fd::AsRawFd;
        let authority = self
            .authority
            .read()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("composite authority is not configured"))?;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&authority.ledger_lock)?;
        flock(lock.as_raw_fd(), FlockArg::LockShared)?;
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&authority.committed)?)?;
        let committed_version = committed
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("committed composite version missing"))?;
        let committed_digest = committed
            .get("component_digest")
            .and_then(serde_json::Value::as_str)
            .filter(|digest| !digest.is_empty())
            .ok_or_else(|| anyhow::anyhow!("committed component digest missing"))?;
        let prefix_name = authority
            .committed_ledger_prefix
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("committed ledger prefix has no file name"))?;
        let committed_ledger = authority.committed_ledger_prefix.with_file_name(format!(
            "{prefix_name}-{committed_version}-{committed_digest}.json"
        ));
        let ledger_bytes = match std::fs::read(&committed_ledger) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // Legacy migration is safe only while the mutable ledger still
                // names the exact generation selected by the commit marker.
                std::fs::read(&authority.ledger)?
            }
            Err(error) => return Err(error.into()),
        };
        let ledger: serde_json::Value = serde_json::from_slice(&ledger_bytes)?;
        let disk_version = ledger
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("composite ledger version missing"))?;
        let disk_digest = ledger
            .get("component_digest")
            .and_then(serde_json::Value::as_str)
            .filter(|digest| !digest.is_empty())
            .ok_or_else(|| anyhow::anyhow!("composite ledger component digest missing"))?;
        let snapshot = self.snapshot();
        anyhow::ensure!(
            snapshot.version == disk_version
                && snapshot.version == committed_version
                && snapshot.component_digest() == disk_digest
                && snapshot.component_digest() == committed_digest,
            "composite signing authority is stale or cutover is pending"
        );
        Ok(snapshot)
    }

    /// Atomically replace the complete ledger. Versions must increase.
    pub fn publish(
        &self,
        version: u64,
        component_digest: String,
        pairs: Vec<CompositeKeyPair>,
    ) -> anyhow::Result<()> {
        let mut guard = self.snapshot.write();
        anyhow::ensure!(
            version > guard.version,
            "composite key-set version must increase"
        );
        anyhow::ensure!(
            !component_digest.is_empty(),
            "composite component digest must not be empty"
        );
        let mut kids = std::collections::HashSet::new();
        let mut active_roles = std::collections::HashSet::new();
        for pair in &pairs {
            anyhow::ensure!(kids.insert(pair.kid.clone()), "duplicate composite kid");
            if pair.state == CompositePairState::Active {
                anyhow::ensure!(
                    active_roles.insert(pair.role),
                    "multiple active composite pairs for one issuer role"
                );
            }
            anyhow::ensure!(
                pair.kid == super::jwt::composite_kid(&pair.ml_dsa, &pair.ed25519),
                "composite kid does not bind its exact key pair"
            );
        }
        *guard = Arc::new(CompositeKeySetSnapshot {
            version,
            component_digest: Arc::from(component_digest),
            pairs: pairs.into(),
        });
        Ok(())
    }
}

static GLOBAL_COMPOSITE_KEY_SET: OnceLock<Arc<CompositeKeySet>> = OnceLock::new();

pub fn global_composite_key_set() -> Arc<CompositeKeySet> {
    GLOBAL_COMPOSITE_KEY_SET
        .get_or_init(|| Arc::new(CompositeKeySet::default()))
        .clone()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use ed25519_dalek::Signer as _;
    use std::sync::Barrier;

    fn pair_from_keys(
        pq: Arc<MlDsaSigningKey>,
        ed: Arc<SigningKey>,
        state: CompositePairState,
    ) -> CompositeKeyPair {
        let kid =
            crate::auth::composite_kid(&ml_dsa::Keypair::verifying_key(&*pq), &ed.verifying_key());
        CompositeKeyPair::signing(kid, pq, ed, CompositePairRole::OAuth, state, 0, i64::MAX)
    }

    fn pair(seed: u8) -> CompositeKeyPair {
        let ed = Arc::new(SigningKey::from_bytes(&[seed; 32]));
        let (pq, _) = crate::crypto::pq::ml_dsa_generate_keypair();
        pair_from_keys(Arc::new(pq), ed, CompositePairState::Active)
    }

    fn mint(pair: &CompositeKeyPair) -> String {
        let (pq, ed) = pair.signing_keys().unwrap();
        let header = format!(
            r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{}"}}"#,
            pair.kid()
        );
        let now = chrono::Utc::now().timestamp();
        let claims = crate::auth::Claims::new("subject".to_owned(), now, now + 3600)
            .with_audience(Some("resource".to_owned()));
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(header),
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
        );
        let pq_signature = crate::crypto::pq::ml_dsa_sign(&pq, input.as_bytes());
        let ed_signature = ed.sign(input.as_bytes());
        let mut signature = pq_signature;
        signature.extend_from_slice(&ed_signature.to_bytes());
        format!("{}.{}", input, URL_SAFE_NO_PAD.encode(signature))
    }

    fn verify(token: &str, snapshot: &CompositeKeySetSnapshot) -> bool {
        let Ok(dispatch) = crate::auth::parse_composite_dispatch(token, &["at+jwt"]) else {
            return false;
        };
        let Some(pair) = snapshot.pair(dispatch.kid()) else {
            return false;
        };
        crate::auth::jwt::decode_composite(
            token,
            pair.ml_dsa(),
            pair.ed25519(),
            Some("resource"),
            &dispatch,
        )
        .is_ok()
    }

    #[test]
    fn publication_is_atomic_across_mint_verify_and_jwks_barriers() {
        let keys = Arc::new(CompositeKeySet::default());
        let old = pair(1);
        let old_kid = old.kid().to_owned();
        keys.publish(1, "generation-1".into(), vec![old.clone()])
            .unwrap();
        let ed_promotion = Arc::new(Barrier::new(2));
        let pq_promotion = Arc::new(Barrier::new(2));
        let mint_barrier = Arc::new(Barrier::new(2));
        let publication_barrier = Arc::new(Barrier::new(2));
        let verify_barrier = Arc::new(Barrier::new(2));
        let jwks_barrier = Arc::new(Barrier::new(2));

        let reader_keys = Arc::clone(&keys);
        let reader_ed = Arc::clone(&ed_promotion);
        let reader_pq = Arc::clone(&pq_promotion);
        let reader_mint = Arc::clone(&mint_barrier);
        let reader_publication = Arc::clone(&publication_barrier);
        let reader_verify = Arc::clone(&verify_barrier);
        let reader_jwks = Arc::clone(&jwks_barrier);
        let reader_old_kid = old_kid.clone();
        let reader = std::thread::spawn(move || {
            reader_ed.wait();
            reader_pq.wait();
            let mint_snapshot = reader_keys.snapshot();
            let token = mint(
                mint_snapshot
                    .active_signing_pair(CompositePairRole::OAuth)
                    .unwrap(),
            );
            reader_mint.wait();
            reader_publication.wait();
            let verify_snapshot = reader_keys.snapshot();
            let verify_version = verify_snapshot.version();
            let token_verified = verify(&token, &verify_snapshot);
            reader_verify.wait();
            let jwks_snapshot = reader_keys.snapshot();
            let jwks_has_old = jwks_snapshot.pair(&reader_old_kid).is_some();
            let jwks_len = jwks_snapshot.pairs().len();
            reader_jwks.wait();
            (
                token,
                verify_version,
                token_verified,
                jwks_has_old,
                jwks_len,
            )
        });

        let promoted_ed = Arc::new(SigningKey::from_bytes(&[2; 32]));
        ed_promotion.wait();
        let (promoted_pq, _) = crate::crypto::pq::ml_dsa_generate_keypair();
        pq_promotion.wait();
        let promoted = pair_from_keys(
            Arc::new(promoted_pq),
            promoted_ed,
            CompositePairState::Active,
        );
        let promoted_kid = promoted.kid().to_owned();
        mint_barrier.wait();
        let mut old_drain = old;
        old_drain.state = CompositePairState::Drain;
        keys.publish(2, "generation-2".into(), vec![old_drain, promoted.clone()])
            .unwrap();
        publication_barrier.wait();
        verify_barrier.wait();
        jwks_barrier.wait();
        let (old_token, verify_version, token_verified, jwks_has_old, jwks_len) =
            reader.join().unwrap();
        assert_eq!(verify_version, 2);
        assert!(token_verified, "a returned token became unverifiable");
        assert!(jwks_has_old);
        assert_eq!(jwks_len, 2);

        // Drain expiry/revocation is a later atomic publication: the old pair
        // disappears completely, while the new exact pair remains usable.
        keys.publish(3, "generation-3".into(), vec![promoted])
            .unwrap();
        let after = keys.snapshot();
        assert_eq!(after.version(), 3);
        assert!(after.pair(&promoted_kid).is_some());
        assert!(
            after.pair(&old_kid).is_none(),
            "evicted pair remained accepted"
        );
        assert!(!verify(&old_token, &after));
        let new_token = mint(after.active_signing_pair(CompositePairRole::OAuth).unwrap());
        assert!(verify(&new_token, &after));
    }

    #[test]
    fn mint_snapshot_fails_closed_for_missing_pending_and_stale_authority() {
        let dir =
            std::env::temp_dir().join(format!("hyprstream-composite-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let ledger = dir.join("ledger.json");
        let committed = dir.join("committed");
        let committed_ledger_prefix = dir.join("committed-ledger");
        let keys = CompositeKeySet::default();
        keys.configure_authority(
            ledger.clone(),
            committed.clone(),
            committed_ledger_prefix.clone(),
            dir.join("lock"),
        );
        keys.publish(1, "generation-1".into(), vec![pair(9)])
            .unwrap();
        assert!(keys.mint_snapshot().is_err());
        std::fs::write(
            &ledger,
            br#"{"version":2,"component_digest":"generation-2"}"#,
        )
        .unwrap();
        std::fs::write(
            &committed,
            br#"{"version":1,"component_digest":"generation-1"}"#,
        )
        .unwrap();
        assert!(
            keys.mint_snapshot().is_err(),
            "legacy fallback accepted a pending mutable ledger"
        );
        std::fs::write(
            dir.join("committed-ledger-1-generation-1.json"),
            br#"{"version":1,"component_digest":"generation-1"}"#,
        )
        .unwrap();
        assert_eq!(
            keys.mint_snapshot().unwrap().version(),
            1,
            "pending mutable authority displaced the last committed snapshot"
        );
        std::fs::write(
            &committed,
            br#"{"version":2,"component_digest":"generation-2"}"#,
        )
        .unwrap();
        assert!(keys.mint_snapshot().is_err(), "stale cache minted");
        std::fs::write(
            dir.join("committed-ledger-2-generation-2.json"),
            br#"{"version":2,"component_digest":"generation-2"}"#,
        )
        .unwrap();
        keys.publish(2, "generation-2".into(), vec![pair(10)])
            .unwrap();
        assert_eq!(keys.mint_snapshot().unwrap().version(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
