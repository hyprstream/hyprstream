//! Journal, checkpoint, and outbox types + the checkpoint signing seam (plan
//! §2a, §2d, §2e).
//!
//! Scope note: the hash-chained journal *data model* and the checkpoint
//! *commitments* live here so the [`crate::LedgerBackend`] trait is complete and
//! MemLedger is a true oracle. The RocksDB-persisted journal, group commit, and
//! the standalone `hyprstream ledger verify` offline tool are later work items
//! (1.2/1.3/1.10); this module builds only the in-memory reference.

use serde::{Deserialize, Serialize};

use crate::engine::Op;
use crate::errors::LedgerError;
use crate::types::{Account, Did, PendingReservation, TransferId, TransferResult};

/// Domain-separation tag for checkpoint signatures (plan §2d).
pub const CHECKPOINT_DOMAIN: &[u8] = b"hs-ledger-checkpoint-v1";

/// One committed operation in the hash chain. `prev_hash` links it to the prior
/// entry; the *outcome* is in the chain too (plan §2a step 3), so tampering with
/// a result breaks the chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Monotonic sequence, starting at 1.
    pub seq: u64,
    /// Hash of entry `seq - 1` (`[0u8; 32]` for the genesis entry).
    pub prev_hash: [u8; 32],
    /// Logical commit timestamp.
    pub ts: u64,
    /// The op that was applied.
    pub op: Op,
    /// The recorded result (`Ok` summary or the deterministic error).
    pub result: Result<TransferResult, LedgerError>,
}

impl JournalEntry {
    /// The blake3-256 hash of this entry's canonical encoding. Used as the next
    /// entry's `prev_hash` and as the checkpoint `head_hash`.
    pub fn hash(&self) -> Result<[u8; 32], LedgerError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hs-ledger-journal-entry-v1");
        cbor_into(&mut hasher, self)?;
        Ok(*hasher.finalize().as_bytes())
    }
}

/// The head of the hash chain — the constant-size commitment the backend keeps
/// in memory and (for RocksLedger) at `CF(meta)/"head"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ChainHead {
    /// Sequence of the last committed entry (`0` when empty).
    pub seq: u64,
    /// Hash of the last committed entry (`[0u8; 32]` when empty).
    pub head_hash: [u8; 32],
}

/// Ordered position of an outbox item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OutboxSeq(pub u64);

/// What a proof-plane item evidences (plan §2e). The ledger crate only *stages*
/// these; the service layer drains and emits them to the PDS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutboxKind {
    /// A dual-signed usage receipt for a posted spend/issue.
    Receipt,
    /// A signed checkpoint to publish.
    Checkpoint,
}

/// A committed-but-unemitted proof-plane item, staged transactionally with the
/// ledger commit (the outbox pattern) and drained by
/// [`crate::LedgerBackend::outbox_peek`] / `outbox_ack`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutboxItem {
    /// Ordered position.
    pub seq: OutboxSeq,
    /// The kind of proof this item carries.
    pub kind: OutboxKind,
    /// The transfer this evidences (for receipts).
    pub transfer_id: Option<TransferId>,
    /// The journal sequence of the commit that produced it.
    pub journal_seq: u64,
}

/// The injection seam for checkpoint signing (mirrors `mac::audit::AuditSigner`).
///
/// PLAN DECISION 8: the seam exists for **key/policy injection**, not
/// optionality. The production impl (service layer) wraps
/// `hyprstream_crypto::cose_sign::sign_composite` and is policy-aware
/// (Hybrid/Classical); tests supply a deterministic signer. `hyprstream-crypto`
/// is an unconditional dependency — a build that cannot sign checkpoints must not
/// exist.
pub trait CheckpointSigner {
    /// Sign the domain-separated signing input (see [`checkpoint_signing_input`]).
    fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, LedgerError>;
    /// The signing identity (the cell ledger's service DID).
    fn ledger_id(&self) -> &Did;
}

/// A periodic, PQC-hybrid-signed commitment to ledger state (plan §2d). It is the
/// anchorable object and the settlement reference. Per-commit chaining is cheap
/// and unsigned; only checkpoints carry a signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedCheckpoint {
    /// The cell ledger's service identity.
    pub ledger_id: Did,
    /// Journal head at checkpoint time.
    pub seq: u64,
    /// Hash of journal entry `seq`.
    pub head_hash: [u8; 32],
    /// Merkle-style root over all accounts sorted by id (plan §2d).
    pub balances_root: [u8; 32],
    /// Root over open reservations sorted by id.
    pub pending_root: [u8; 32],
    /// Logical commit time of the checkpoint.
    pub ts: u64,
    /// Hash of the previous checkpoint (checkpoints chain too).
    pub prev_checkpoint_hash: [u8; 32],
    /// COSE composite signature over [`checkpoint_signing_input`].
    pub sig: Vec<u8>,
}

impl SignedCheckpoint {
    /// A digest over the whole (signed) checkpoint — the value the next
    /// checkpoint chains to via `prev_checkpoint_hash`, and the object an OTS
    /// anchor attests (plan §2h).
    pub fn digest(&self) -> Result<[u8; 32], LedgerError> {
        let mut h = blake3::Hasher::new();
        h.update(b"hs-ledger-checkpoint-digest-v1");
        cbor_into(&mut h, self)?;
        Ok(*h.finalize().as_bytes())
    }

    /// The bytes the signature covers (everything except `sig`).
    pub fn signing_input(&self) -> Result<Vec<u8>, LedgerError> {
        let content = CheckpointContent {
            ledger_id: &self.ledger_id,
            seq: self.seq,
            head_hash: self.head_hash,
            balances_root: self.balances_root,
            pending_root: self.pending_root,
            ts: self.ts,
            prev_checkpoint_hash: self.prev_checkpoint_hash,
        };
        content.signing_input()
    }
}

/// The signable content of a checkpoint (the `sig`-free projection).
#[derive(Serialize)]
pub struct CheckpointContent<'a> {
    /// See [`SignedCheckpoint::ledger_id`].
    pub ledger_id: &'a Did,
    /// See [`SignedCheckpoint::seq`].
    pub seq: u64,
    /// See [`SignedCheckpoint::head_hash`].
    pub head_hash: [u8; 32],
    /// See [`SignedCheckpoint::balances_root`].
    pub balances_root: [u8; 32],
    /// See [`SignedCheckpoint::pending_root`].
    pub pending_root: [u8; 32],
    /// See [`SignedCheckpoint::ts`].
    pub ts: u64,
    /// See [`SignedCheckpoint::prev_checkpoint_hash`].
    pub prev_checkpoint_hash: [u8; 32],
}

impl CheckpointContent<'_> {
    /// Domain-separated signing input: a content hash prefixed by the domain tag
    /// (the `mac::audit::audit_signing_input` shape — sign the hash, not the raw
    /// bytes, so the signature is independent of the canonical-bytes choice).
    pub fn signing_input(&self) -> Result<Vec<u8>, LedgerError> {
        let mut hasher = blake3::Hasher::new();
        cbor_into(&mut hasher, self)?;
        Ok(checkpoint_signing_input(hasher.finalize().as_bytes()))
    }
}

/// Assemble the domain-separated signing input from a 32-byte content hash.
pub fn checkpoint_signing_input(content_hash: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(CHECKPOINT_DOMAIN.len() + 32);
    v.extend_from_slice(CHECKPOINT_DOMAIN);
    v.extend_from_slice(content_hash);
    v
}

/// Verify a checkpoint's composite signature against the ledger DID's key
/// material, delegating to `hyprstream_crypto::cose_sign::verify_composite`.
///
/// This is the load-bearing use of the unconditional `hyprstream-crypto`
/// dependency: checkpoint signatures gate anchoring, settlement, and dispute
/// evidence, so the verification primitive belongs in the ledger crate. Full
/// journal replay + multi-checkpoint verification (`verify.rs` / the CLI) is a
/// later work item; this verifies a single checkpoint's signature.
pub fn verify_checkpoint_signature(
    cp: &SignedCheckpoint,
    ed_vk: &ed25519_dalek::VerifyingKey,
    pq_vk: Option<&hyprstream_crypto::pq::MlDsaVerifyingKey>,
    require_pq: bool,
) -> Result<(), LedgerError> {
    let input = cp.signing_input()?;
    hyprstream_crypto::cose_sign::verify_composite(&cp.sig, ed_vk, pq_vk, &input, &[], require_pq)
        .map(|_| ())
        .map_err(|e| LedgerError::Internal(format!("checkpoint signature invalid: {e}")))
}

/// Root over accounts sorted by id: `blake3(concat(blake3(id || cbor(account))))`.
/// A deterministic commitment (not a sparse-Merkle proof system — sufficient for
/// the drift/equivalence properties; a full accumulator can replace it later
/// without changing the trait).
pub fn balances_root<'a>(
    accounts: impl Iterator<Item = &'a Account>,
) -> Result<[u8; 32], LedgerError> {
    let mut sorted: Vec<&Account> = accounts.collect();
    sorted.sort_by_key(|a| a.id.0);
    let mut root = blake3::Hasher::new();
    root.update(b"hs-ledger-balances-root-v1");
    for a in sorted {
        let mut leaf = blake3::Hasher::new();
        leaf.update(&a.id.0.to_be_bytes());
        cbor_into(&mut leaf, a)?;
        root.update(leaf.finalize().as_bytes());
    }
    Ok(*root.finalize().as_bytes())
}

/// Root over open (`Pending`) reservations sorted by id.
pub fn pending_root<'a>(
    reservations: impl Iterator<Item = &'a PendingReservation>,
) -> Result<[u8; 32], LedgerError> {
    let mut sorted: Vec<&PendingReservation> = reservations
        .filter(|r| r.state == crate::types::PendingState::Pending)
        .collect();
    sorted.sort_by_key(|r| r.transfer.id.0);
    let mut root = blake3::Hasher::new();
    root.update(b"hs-ledger-pending-root-v1");
    for r in sorted {
        let mut leaf = blake3::Hasher::new();
        leaf.update(&r.transfer.id.0.to_be_bytes());
        cbor_into(&mut leaf, r)?;
        root.update(leaf.finalize().as_bytes());
    }
    Ok(*root.finalize().as_bytes())
}

/// What one `tick` did (plan §2b.4 sweep + scheduled checkpoint).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TickReport {
    /// How many reservations the sweep expired.
    pub expired: usize,
    /// The sequence at which a checkpoint was cut, if any.
    pub checkpointed: Option<u64>,
}

/// Serialize `value` as CBOR directly into a `Write` (a blake3 hasher), mapping
/// the (practically unreachable) serializer error to a fail-closed
/// [`LedgerError::Internal`] rather than unwrapping.
fn cbor_into<W: std::io::Write, T: serde::Serialize>(
    w: &mut W,
    value: &T,
) -> Result<(), LedgerError> {
    ciborium::into_writer(value, w)
        .map_err(|e| LedgerError::Internal(format!("cbor encoding failed: {e}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_crypto::cose_sign::sign_composite;
    use hyprstream_crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes};

    /// End-to-end proof that the unconditional `hyprstream-crypto` dependency is
    /// load-bearing: a checkpoint signed with the hybrid composite (EdDSA +
    /// ML-DSA-65) verifies, and tampering with a committed field breaks it.
    #[test]
    fn checkpoint_hybrid_signature_roundtrips_and_detects_tamper() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        let ed_vk = ed_sk.verifying_key();
        let pq_sk = ml_dsa_sk_from_seed(&[9u8; 32]);
        let pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&pq_sk)).unwrap();

        let content = CheckpointContent {
            ledger_id: &Did("did:web:cell.test".to_owned()),
            seq: 42,
            head_hash: [1u8; 32],
            balances_root: [2u8; 32],
            pending_root: [3u8; 32],
            ts: 1000,
            prev_checkpoint_hash: [0u8; 32],
        };
        let input = content.signing_input().unwrap();
        let sig = sign_composite(&ed_sk, Some(&pq_sk), &input, &[]).unwrap();

        let mut cp = SignedCheckpoint {
            ledger_id: Did("did:web:cell.test".to_owned()),
            seq: 42,
            head_hash: [1u8; 32],
            balances_root: [2u8; 32],
            pending_root: [3u8; 32],
            ts: 1000,
            prev_checkpoint_hash: [0u8; 32],
            sig,
        };

        // Valid signature verifies (fail-closed require_pq = true).
        verify_checkpoint_signature(&cp, &ed_vk, Some(&pq_vk), true).unwrap();

        // Tampering with a committed field changes the signing input ⇒ rejected.
        cp.balances_root = [0xFFu8; 32];
        assert!(verify_checkpoint_signature(&cp, &ed_vk, Some(&pq_vk), true).is_err());
    }
}
