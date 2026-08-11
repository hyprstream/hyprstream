//! Proof replay admission and challenge manager wiring.
//!
//! Follows the #1516 `CredentialRevocationStore` pattern: a trait with
//! process-global `OnceLock` registration, an in-memory default, and
//! `set_global_*()` / `global_*()` accessors. Installed at startup;
//! dispatch calls the globals.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use super::{ProofDisposition, RequestId};

// ---------------------------------------------------------------------------
// Replay admission store (canonical #1516 pattern)
// ---------------------------------------------------------------------------

/// The replay admission key: (signer thumbprint, request_id).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProofReplayKey {
    pub signer_thumbprint: [u8; 32],
    pub request_id: RequestId,
}

/// The outcome of a replay admission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofAdmissionResult {
    Admitted,
    Replayed,
    Failed,
}

/// Proof replay admission store trait. Follows the same pattern as
/// `CredentialRevocationStore` — process-global, trait-based, with
/// per-partition fail-closed capacity and per-entry expiry.
pub trait ProofReplayStore: Send + Sync {
    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ProofReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult;
}

/// In-memory proof replay store with per-partition capacity and
/// fail-closed behavior. Entries expire after their `expires_at` timestamp.
pub struct InMemoryProofReplayStore {
    authenticated: parking_lot::Mutex<HashMap<ProofReplayKey, u64>>,
    unattributed: parking_lot::Mutex<HashMap<ProofReplayKey, u64>>,
    max_per_partition: usize,
}

impl Default for InMemoryProofReplayStore {
    fn default() -> Self {
        Self::new(100_000)
    }
}

impl InMemoryProofReplayStore {
    pub fn new(max_per_partition: usize) -> Self {
        Self {
            authenticated: Default::default(),
            unattributed: Default::default(),
            max_per_partition,
        }
    }

    fn lock(
        &self,
        partition: ProofDisposition,
    ) -> parking_lot::MutexGuard<'_, HashMap<ProofReplayKey, u64>> {
        match partition {
            ProofDisposition::Authenticated => self.authenticated.lock(),
            ProofDisposition::Unattributed => self.unattributed.lock(),
        }
    }
}

impl ProofReplayStore for InMemoryProofReplayStore {
    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ProofReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        let mut map = self.lock(partition);
        if map.contains_key(key) {
            return ProofAdmissionResult::Replayed;
        }
        // Opportunistic GC
        let now = current_unix_seconds();
        map.retain(|_, exp| *exp > now);
        if map.len() >= self.max_per_partition {
            return ProofAdmissionResult::Failed;
        }
        map.insert(key.clone(), expires_at);
        ProofAdmissionResult::Admitted
    }
}

// Process-global registration (same pattern as CredentialRevocationStore)
static PROOF_REPLAY_STORE: OnceLock<Box<dyn ProofReplayStore>> = OnceLock::new();

pub fn set_global_proof_replay_store(
    store: Box<dyn ProofReplayStore>,
) -> std::result::Result<(), Box<dyn ProofReplayStore>> {
    PROOF_REPLAY_STORE.set(store)
}

pub fn global_proof_replay_store() -> Option<&'static dyn ProofReplayStore> {
    PROOF_REPLAY_STORE.get().map(|s| &**s)
}

// Removed: auto-install was a fail-open. The store must be explicitly
// installed via set_global_proof_replay_store(). Dispatch denies when absent.

// ---------------------------------------------------------------------------
// Process-global challenge manager
// ---------------------------------------------------------------------------

static CHALLENGE_MANAGER: OnceLock<super::challenge::ChallengeManager> = OnceLock::new();

/// Install a challenge manager for the process.
pub fn set_global_challenge_manager(
    mgr: super::challenge::ChallengeManager,
) -> std::result::Result<(), super::challenge::ChallengeManager> {
    CHALLENGE_MANAGER.set(mgr)
}

pub fn global_challenge_manager() -> Option<&'static super::challenge::ChallengeManager> {
    CHALLENGE_MANAGER.get()
}

// ---------------------------------------------------------------------------
// Helper: compute unattributed replay thumbprint from proof
// ---------------------------------------------------------------------------

/// Compute the SHA-256 thumbprint of the canonical (plan, key_set) tuple
/// for unattributed proof replay keying.
pub fn unattributed_thumbprint(protected_bytes: &[u8]) -> Result<[u8; 32]> {
    let protected: ciborium::Value =
        ciborium::de::from_reader(&mut std::io::Cursor::new(protected_bytes))
            .map_err(|e| anyhow::anyhow!("protected header decode: {e}"))?;

    let map = match &protected {
        ciborium::Value::Map(m) => m,
        _ => bail!("protected header not a map"),
    };

    let plan_val = map
        .iter()
        .find(|(k, _)| {
            matches!(k,
                ciborium::Value::Integer(i)
                if i128::from(*i) == super::HEADER_HS_SIGNATURE_PLAN as i128
            )
        })
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow::anyhow!("no hs_signature_plan"))?;

    let key_set_val = map
        .iter()
        .find(|(k, _)| {
            matches!(k,
                ciborium::Value::Integer(i)
                if i128::from(*i) == super::HEADER_HS_UNATTRIBUTED_KEY_SET as i128
            )
        })
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow::anyhow!("no hs_unattributed_key_set"))?;

    let tuple = ciborium::Value::Array(vec![plan_val.clone(), key_set_val.clone()]);
    let mut tuple_bytes = Vec::new();
    ciborium::ser::into_writer(&tuple, &mut tuple_bytes)
        .map_err(|e| anyhow::anyhow!("tuple encode: {e}"))?;

    let mut hasher = Sha256::new();
    hasher.update(b"hs-proof-key-set-replay-v1");
    hasher.update(&tuple_bytes);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn current_unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn current_unix_seconds() -> u64 {
    0
}
