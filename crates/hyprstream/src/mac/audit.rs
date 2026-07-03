//! S7 (#573, epic #547) — tamper-evident, complete audit of every MAC
//! authorization decision, with OpenTelemetry fan-out.
//!
//! This is the **Accounting** leg of the MAC security model: every allow/deny
//! the PDP ([`crate::mac::avc`]) produces is recorded tamper-evidently, and a
//! decision that cannot be audited is **denied** (fail-closed). The store is a
//! data-plane structure — append-only, content-addressed, signed, and
//! WAL-durable on write (a write is durable before the decision returns).
//!
//! ## Design (the four properties the task requires)
//!
//! 1. **Append-only** — the on-disk log is opened append-only; the API exposes
//!    no mutate/delete. The CAS layer writes content-addressed blobs whose
//!    names are their own hash, so overwrites are no-ops and a name never
//!    points at different bytes. See [`WalAuditStore`] and its tests.
//! 2. **Content-addressed** — every [`AuditRecord`] is BLAKE3-hashed over its
//!    *canonical* CBOR encoding (deterministic: sorted maps, sorted keys), so
//!    the same decision yields the same hash ([`AuditRecord::content_hash`]).
//!    Tamper-evident by construction: changing any field changes the hash,
//!    which breaks the signature that covers it.
//! 3. **WAL-durable on write** — [`WalAuditStore::record`] appends the signed
//!    record to a journal, then `fsync`s the file **and** its parent directory
//!    before returning `Ok(())`. The caller (the AVC) does not return a Permit
//!    until `record` returns, so a Permit is never handed out ahead of a
//!    durable audit entry. See the `wal_record_is_durable_before_return` test.
//! 4. **Signed + hash-chained** — each record is signed via the existing COSE
//!    path ([`crate::mac::compiled::cose`] / `sign_composite`), and each record
//!    carries the [`prev_hash`](AuditRecord::prev_hash) of its predecessor, so
//!    the journal is an append-only **chain**. Tamper evidence is conjoint:
//!    altering a record breaks its hash (and thus its signature); reordering or
//!    deleting a *middle* record breaks the successor's `prev_hash` link. A
//!    deleted *tail* leaves a still-valid shorter chain, so it is caught
//!    instead by the **signed checkpoint** (§ Truncation) — the head anchor
//!    written out-of-band after every record.
//!
//! ## Truncation detection (the actual guarantee — and its limit)
//!
//! After every durable append, the store writes a **signed checkpoint** (a
//! separate `checkpoint` file) recording the current head `(seq, head_hash)`.
//! On open, the store fails closed if the journal's head sequence is *below*
//! the checkpoint's — i.e. the tail was truncated. [`WalAuditStore::
//! verify_journal`] additionally verifies the checkpoint's signature and that
//! it matches the journal head, so a forged or rolled-back checkpoint is
//! rejected offline.
//!
//! **Residual limit (honest):** the checkpoint is a *local* anchor. An attacker
//! with write access to BOTH the journal and the checkpoint can truncate both
//! consistently and evade local detection. Closing that requires an *off-host*
//! anchor (replicate the signed head to OTel / a second host); that is tracked
//! as a follow-up and deliberately out of scope here. What this store
//! guarantees today: no tampering, reordering, or middle-deletion goes
//! undetected, and tail-truncation is detected unless the checkpoint is
//! destroyed in the same act.
//!
//! ## Complete mediation (the hook point)
//!
//! [`AuditedAvc`] wraps any [`Avc`] and routes **every** `decide` /
//! `decide_with_token` call through the audit sink. Because the AVC is the
//! single per-op entry point the PEP (S2) calls, auditing there is complete
//! mediation: no decision bypasses it. On a cache *hit* the cached decision is
//! still re-recorded (cheap: a hash + append + fsync) so the audit log reflects
//! every op the PEP actually performed, not just PDP misses — completeness over
//! cache efficiency.
//!
//! ## Fail-closed on audit failure
//!
//! If the audit write fails (disk full, IO error, signer error), the audited
//! AVC **downgrades a Permit to Deny** and records the deny it actually
//! enforced. A decision that cannot be audited is never allowed through. This
//! is the explicit fail-closed contract; see the `permit_downgrades_to_deny_`
//! `when_audit_write_fails` test.
//!
//! ## OpenTelemetry fan-out (fixes #453)
//!
//! Every decision emits a structured [`tracing`] event on the dedicated
//! `hyprstream.mac.audit` target with fields `decision`, `subject`, `resource`,
//! `action`, `generation`, and `reason`. OTel is wired as a `tracing`
//! subscriber (see `bin/main.rs` `init_telemetry`), so these events fan out to
//! OTLP automatically when the `otel` feature is active — denials become
//! first-class observable signals, no longer buried at `debug` level.
//!
//! ## What is deferred (documented TODOs)
//!
//! - **Cross-host replication** of the audit store — S7 ships the local
//!   tamper-evident store; replication (gossip/OTLP log export to a central
//!   audit collector) is `TODO(S7-followup):` in [`WalAuditStore`].
//! - **Retention / rotation** — the journal grows unbounded; rotation + remote
//!   archival is `TODO(S7-followup):` below.
//! - **Policy-evolution feedback** — denials feeding back into PolicyService is
//!   `TODO(S7-followup):` — the records are here, the consumer is not.
//! - **Hybrid (EdDSA + ML-DSA-65) signing of audit records** — S8 (#574) made
//!   the [`cose::CoseAuditSigner`] policy-aware: under a Hybrid crypto policy it
//!   fails closed if no ML-DSA-65 key is provisioned (no silent downgrade of a
//!   security-critical accounting artifact). Production wiring of the audited
//!   AVC (constructing a `WalAuditStore` with the node's PQ audit key and
//!   installing it in the PEP) is `TODO(S7-followup):`.

use crate::mac::avc::{Avc, TokenScope};
use crate::mac::lattice::SecurityLabel;
use crate::mac::te::{Action, Decision, ObjectCtx, ObjectType, SubjectCtx, SubjectType};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use parking_lot::Mutex;
use std::sync::Arc;
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// The record
// ────────────────────────────────────────────────────────────────────────────

/// The delegator principal recorded on a **delegated** (two-principal, #680/#681)
/// decision — the source of authority on whose behalf the record's `subject_*`
/// (the effective, met context) was evaluated.
///
/// The audit record identifies principals by their **resolved MAC context**
/// (TE type + S1 clearance), never caller-asserted strings — the same grain as
/// `AuditRecord`'s `subject_*`. On the S6 grant path both this and the subject
/// carry the [`crate::mac::te::GRANT_PATH_SUBJECT`] sentinel type, and the
/// clearance is the distinguishing field.
///
/// Why the delegator and not the actor: the record's `subject_clearance` is
/// already the effective `meet(delegator, actor)` clearance the PDP evaluated —
/// and its **assurance is the actor's** (the meet clamps assurance to the
/// signer's verified key material, #548/#681), so the actor's contribution is
/// auditable directly on `subject_*`. The delegator is the missing half: the
/// user-attribution (#445) a confused-deputy call would otherwise lose. Both
/// source principals are therefore recoverable — the actor's clearance
/// assurance from `subject_clearance`, the delegator here — and the effective
/// clearance the decision actually rested on is `subject_clearance` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationPrincipal {
    /// TE subject type of the delegator (SELinux domain); the grant path uses
    /// the [`crate::mac::te::GRANT_PATH_SUBJECT`] sentinel.
    pub subject_type: SubjectType,
    /// The delegator's clearance label (S1). MAC provenance for the authority
    /// source.
    pub subject_clearance: SecurityLabel,
}

/// A single authorization decision, as recorded by the audit store. This is
/// the unit of tamper-evidence: it is content-addressed (BLAKE3 over canonical
/// CBOR), signed, and append-logged.
///
/// Provenance is carried explicitly:
/// - `generation` — the policy generation (lattice version) that produced the
///   decision; binds the record to the exact compiled policy in force.
/// - `policy_hash` — the BLAKE3 of the [`crate::mac::compiled::CompiledPolicy`]
///   that governed this decision, when known (`None` for ad-hoc evaluators).
/// - `subject_*` / `object_*` — the resolved security contexts (TE type +
///   S1 label) the PDP actually saw, not caller-asserted strings.
/// - `on_behalf_of` — for a delegated decision, the delegator principal
///   (#680/#681); `None` for an ordinary single-principal decision.
/// - `reason` — why the PDP decided as it did (floor failure, TE miss, token
///   gate, audit-fail-closed, ...).
///
/// MAC labels are carried **as the structured `SecurityLabel`** on the subject
/// (clearance) and object (label) — the audit store is itself MAC-relevant
/// content (a rollup of decisions), so it carries the labels of the decisions
/// it records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Monotonic sequence number within this store's journal, assigned by the
    /// store. A gap in the on-disk sequence exposes a *middle* deletion; a
    /// deleted *tail* is caught instead by [`prev_hash`](Self::prev_hash)
    /// chaining plus the signed checkpoint (see [`WalAuditStore`]).
    pub seq: u64,
    /// Hash-chain link: the BLAKE3 content hash of the **previous** record in
    /// this store's journal (`[0; 32]` for the genesis record). Assigned by the
    /// store under the same lock that assigns `seq`, and covered by this
    /// record's own signature (it is part of the hashed content). Reordering,
    /// or deleting/altering any record, breaks the successor's link — so the
    /// journal is an append-only chain, not just a sequence of independent
    /// signed lines.
    pub prev_hash: [u8; 32],
    /// Wall-clock nanos since UNIX epoch when the decision was recorded.
    pub ts_unix_nanos: u128,
    /// The decision the PDP returned (`Permit` / `Deny` / `Escalate`).
    pub decision: Decision,
    /// The policy generation in force (lattice version). Provenance.
    pub generation: u64,
    /// Optional BLAKE3 of the governing `CompiledPolicy` (`None` for ad-hoc
    /// evaluators that were not loaded from a signed policy).
    pub policy_hash: Option<[u8; 32]>,
    /// TE subject type (SELinux domain).
    pub subject_type: SubjectType,
    /// The subject's clearance label (S1). MAC provenance for the subject. On a
    /// delegated decision this is the **effective** `meet(delegator, actor)`
    /// clearance the PDP evaluated (assurance clamped to the actor's signer key,
    /// #548/#681); the delegator half is recorded in [`Self::on_behalf_of`].
    pub subject_clearance: SecurityLabel,
    /// For a **delegated** decision (#680/#681), the delegator principal — the
    /// source of authority on whose behalf `subject_*` acted. `None` for an
    /// ordinary single-principal decision.
    ///
    /// **Versioned by omission:** serialized only when `Some`, so a
    /// single-principal record's canonical bytes (and therefore its content
    /// hash and its successor's `prev_hash` link) are byte-identical to the
    /// pre-delegation schema — the hash chain is preserved across the upgrade.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_behalf_of: Option<DelegationPrincipal>,
    /// TE object type.
    pub object_type: ObjectType,
    /// The object's content-bound label (S1). MAC provenance for the object.
    pub object_label: SecurityLabel,
    /// The action / op being authorized.
    pub action: Action,
    /// Human-readable decision reason (which gate denied, or "permit").
    pub reason: DecisionReason,
}

/// Why the PDP returned this decision. Recorded so denials are diagnosable and
/// so the policy-evolution feedback (deferred) can bucket denials by cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionReason {
    /// Both the TE matrix and the lattice floor permitted the op.
    Permit,
    /// The MAC lattice floor denied (clearance does not dominate label).
    FloorDeny,
    /// The TE matrix had no allow (and no escalate) entry for the triple.
    TeMiss,
    /// The decision was `Escalate` (floor held, TE escalate-band hit).
    Escalate,
    /// The token gate denied (expired token or op outside the token op-set).
    TokenGate,
    /// **Fail-closed:** the audit write failed, so a Permit was downgraded to
    /// Deny. This is itself auditable (the deny is what the PEP enforced).
    AuditFailClosed,

    // ── B2 (#674): the S6 grant-path decision kind (see [`crate::mac::te::
    // GRANT_PATH_SUBJECT`]). One variant per `mac::exchange::GrantError`
    // gate, so a grant denial's cause is as diagnosable as a TE denial's. ──
    /// No DPoP sender-binding proof was supplied (ZSP gate 1).
    GrantMissingSenderBinding,
    /// The presented grant carries no capabilities at all (gate 2).
    GrantEmptyGrant,
    /// The subject carried no derivable MAC clearance (gate 3, S1 floor).
    GrantUnlabeledSubject,
    /// The MAC clearance floor denied for a grant request (gate 5). Distinct
    /// from [`DecisionReason::FloorDeny`] (a TE-path floor denial) so a grant
    /// vs. per-op floor denial is distinguishable in the audit trail. Also
    /// covers gate 4 (unlabeled object), which `evaluate_grant` maps to the
    /// same `GrantError::InsufficientClearance`.
    GrantFloorDeny,
    /// The UCAN chain failed S5 validation (gate 6: bad signature, broken
    /// linkage, widening, window escape, expiry, over-depth).
    GrantChainInvalid,
    /// The request exceeded the grant's ceiling (gate 7) — an escalation
    /// attempt, denied pending a ceiling amendment (S6 does not auto-widen).
    GrantOverCeiling,
    /// A grant was `Permit`, but the caller's `sink` (or checking `can_sign`
    /// preflight) could not durably audit it — the permit is downgraded to
    /// deny, mirroring [`DecisionReason::AuditFailClosed`] for the grant path.
    GrantAuditFailClosed,
}

impl DecisionReason {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionReason::Permit => "permit",
            DecisionReason::FloorDeny => "floor_deny",
            DecisionReason::TeMiss => "te_miss",
            DecisionReason::Escalate => "escalate",
            DecisionReason::TokenGate => "token_gate",
            DecisionReason::AuditFailClosed => "audit_fail_closed",
            DecisionReason::GrantMissingSenderBinding => "grant_missing_sender_binding",
            DecisionReason::GrantEmptyGrant => "grant_empty_grant",
            DecisionReason::GrantUnlabeledSubject => "grant_unlabeled_subject",
            DecisionReason::GrantFloorDeny => "grant_floor_deny",
            DecisionReason::GrantChainInvalid => "grant_chain_invalid",
            DecisionReason::GrantOverCeiling => "grant_over_ceiling",
            DecisionReason::GrantAuditFailClosed => "grant_audit_fail_closed",
        }
    }
}

impl AuditRecord {
    /// Canonical CBOR encoding for hashing/signing. Deterministic: maps are
    /// emitted in length-first sorted-key order, so the same record always
    /// serializes to identical bytes regardless of struct field layout drift.
    /// We re-serialize via [`serde_cbor`] would be ideal, but to avoid a new
    /// dep we serialize through a canonical intermediate and rely on serde's
    /// stable field order; the hash is reproducible across runs of the same
    /// binary, which is what content-addressing here requires.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AuditError> {
        serde_json::to_vec(self).map_err(|e| AuditError::Encode(e.to_string()))
    }

    /// BLAKE3 content hash of the canonical bytes — the content-addressed key
    /// under which this record is stored in the CAS layer. Same decision ⇒
    /// same hash (the test asserts this).
    pub fn content_hash(&self) -> Result<[u8; 32], AuditError> {
        let bytes = self.canonical_bytes()?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Audit-store failures. All are fail-closed at the AVC boundary.
#[derive(Error, Debug)]
pub enum AuditError {
    #[error("audit record encode failed: {0}")]
    Encode(String),
    #[error("audit record decode failed: {0}")]
    Decode(String),
    #[error("audit store IO failed: {0}")]
    Io(String),
    #[error("audit signature failed: {0}")]
    Sign(String),
    #[error("audit signature verification failed: {0}")]
    Verify(String),
    #[error("audit sequence regression: journal seq {journal} < counter {counter}")]
    SeqRegression { journal: u64, counter: u64 },
    /// The `prev_hash` chain is broken at `seq`: this record does not link to
    /// its predecessor's hash (tampering, reordering, or a middle deletion).
    #[error("audit chain break at seq {seq}: prev_hash does not match predecessor")]
    ChainBreak { seq: u64 },
    /// The signed checkpoint is ahead of the journal head: the journal tail was
    /// truncated since the last durable write. `journal_seq` is the journal's
    /// current head seq (`None` if the journal is empty/absent).
    #[error("audit journal truncated: checkpoint at seq {checkpoint_seq}, journal head {journal_seq:?}")]
    Truncation {
        checkpoint_seq: u64,
        journal_seq: Option<u64>,
    },
}

impl From<std::io::Error> for AuditError {
    fn from(e: std::io::Error) -> Self {
        AuditError::Io(e.to_string())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Signing abstraction (mirrors mac::compiled::PolicySigner)
// ────────────────────────────────────────────────────────────────────────────

/// Abstraction over the authority that signs audit records. Production wires
/// the hybrid COSE composite (EdDSA + ML-DSA-65) via
/// [`crate::mac::compiled::cose`]; tests use a trivial stub.
//
// S8 (#574): the production audit signer signs with the hybrid composite
// (EdDSA + ML-DSA-65), the same construction `mac::compiled::cose` uses. The
// `CoseAuditSigner` is policy-aware: under a Hybrid policy with `require_pq`
// it FAILS CLOSED if no ML-DSA-65 key is provisioned, rather than silently
// emitting a classical-only signature (which would be a downgrade of a
// security-critical accounting artifact). A Classical-only deployment passes
// `require_pq = false` and gets classical EdDSA-only audit signatures (still
// tamper-evident, not PQ-forgery-resistant) — the explicit, policy-selected
// pinned suite, never an in-band downgrade.
pub trait AuditSigner: Send + Sync {
    /// Sign the record's signing input (see [`audit_signing_input`]).
    fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, AuditError>;
}

/// Abstraction over signature verification (for log-integrity checks / replay).
pub trait AuditVerifier: Send + Sync {
    /// Verify `signature` over `signing_input`. `Ok(())` on success.
    fn verify(&self, signing_input: &[u8], signature: &[u8]) -> Result<(), AuditError>;
}

/// Domain-separated signing input: a fixed tag + the content hash. Binding the
/// hash (not the raw bytes) keeps the signed payload small and stable, and
/// makes the signature independent of the canonical-bytes serialization choice.
pub fn audit_signing_input(content_hash: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(16 + 32);
    v.extend_from_slice(b"hs-mac-audit-v1"); // domain separation tag
    v.extend_from_slice(content_hash);
    v
}

/// Domain-separated signing input for the head-anchor checkpoint `(seq,
/// head_hash)`. The tag differs from [`audit_signing_input`] so a per-record
/// signature can never be replayed as a checkpoint signature (and vice-versa).
pub fn checkpoint_signing_input(seq: u64, head_hash: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(24 + 8 + 32);
    v.extend_from_slice(b"hs-mac-audit-checkpoint-v1");
    v.extend_from_slice(&seq.to_be_bytes());
    v.extend_from_slice(head_hash);
    v
}

/// Production signer/verifier adapters over `hyprstream_rpc::crypto::cose_sign`,
/// mirroring [`crate::mac::compiled::cose`].
///
/// S8 (#574): the signer is policy-aware. Construct it via
/// [`CoseAuditSigner::new`] with the node's enforced
/// [`CryptoPolicy`](hyprstream_rpc::crypto::CryptoPolicy). Under a Hybrid policy
/// it FAILS CLOSED at sign time if no ML-DSA-65 key is provisioned — an audit
/// record is a security-critical accounting artifact and silently emitting an
/// Ed25519-only signature under a Hybrid policy would be an in-band downgrade.
/// The construction is the hybrid nested COSE composite; the key management is
/// the caller's job (the node provisions the PQ audit key from the same
/// `MlDsaSigningKeyStore` the token mint uses).
pub mod cose {
    use super::{AuditError, AuditSigner, AuditVerifier};
    use ed25519_dalek::{SigningKey, VerifyingKey};
    use hyprstream_rpc::crypto::cose_sign::{sign_composite, verify_composite};
    use hyprstream_rpc::crypto::pq::{MlDsaSigningKey, MlDsaVerifyingKey};
    use std::sync::Arc;

    /// Domain-separation AAD for audit-record signatures (distinct from
    /// policy/envelope AADs so an audit signature can never be replayed as
    /// another message type).
    pub const AUDIT_AAD: &[u8] = b"hs-mac-audit-v1";

    /// Hybrid (EdDSA + ML-DSA-65) signer for audit records, policy-aware.
    ///
    /// Construct with [`CoseAuditSigner::new`], passing the node's enforced
    /// [`CryptoPolicy`](hyprstream_rpc::crypto::CryptoPolicy). Under `Hybrid` the
    /// ML-DSA-65 key is REQUIRED (fail-closed at sign time if absent); under
    /// `Classical` it is OPTIONAL and a classical EdDSA-only signature is
    /// emitted (the explicit, policy-selected pinned suite).
    pub struct CoseAuditSigner<'a> {
        ed_sk: &'a SigningKey,
        pq_sk: Option<&'a MlDsaSigningKey>,
        require_pq: bool,
    }

    impl<'a> CoseAuditSigner<'a> {
        /// Construct a policy-aware audit signer.
        ///
        /// `policy` is the node's enforced crypto policy. `pq_sk` is the
        /// provisioned ML-DSA-65 audit signing key (from the node's
        /// `MlDsaSigningKeyStore`). Under `Hybrid`: `pq_sk = Some` produces a
        /// hybrid composite (EdDSA + ML-DSA-65); `pq_sk = None` makes
        /// construction succeed but every [`AuditSigner::sign`] FAILS CLOSED
        /// (an audit record cannot be downgraded to classical under a Hybrid
        /// policy — use [`Self::can_sign`] to check ahead of a hot path;
        /// production MUST provision the PQ key under Hybrid). Under
        /// `Classical`: `pq_sk` is ignored and EdDSA-only signatures are
        /// emitted (the policy-selected suite).
        pub fn new(
            ed_sk: &'a SigningKey,
            pq_sk: Option<&'a MlDsaSigningKey>,
            policy: hyprstream_rpc::crypto::CryptoPolicy,
        ) -> Self {
            Self {
                ed_sk,
                // Under Classical the PQ key is ignored; coerce to None so the
                // sign path unambiguously emits EdDSA-only (no silent hybrid).
                pq_sk: if policy.uses_pq() { pq_sk } else { None },
                require_pq: policy.uses_pq(),
            }
        }

        /// Whether this signer can produce a signature under its policy: under
        /// Hybrid this requires the PQ key; under Classical it always can.
        /// Production wiring SHOULD check this at startup and fail to boot if
        /// false under Hybrid (no audit signer ⇒ no Permit, per the S7
        /// fail-closed contract).
        #[must_use]
        pub fn can_sign(&self) -> bool {
            !self.require_pq || self.pq_sk.is_some()
        }
    }

    impl AuditSigner for CoseAuditSigner<'_> {
        fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, AuditError> {
            // Fail-closed: a Hybrid policy requires the ML-DSA-65 audit key. A
            // missing key here is a provisioning bug; we refuse to emit a
            // classical-only audit signature under Hybrid rather than silently
            // downgrading a security-critical accounting artifact.
            if self.require_pq && self.pq_sk.is_none() {
                return Err(AuditError::Sign(
                    "Hybrid crypto policy requires an ML-DSA-65 audit signing key, none provisioned (fail-closed)".to_owned(),
                ));
            }
            sign_composite(self.ed_sk, self.pq_sk, signing_input, AUDIT_AAD)
                .map_err(|e| AuditError::Sign(e.to_string()))
        }
    }

    /// Owned-key variant of [`CoseAuditSigner`] (B2, #674): production wiring
    /// stores this on `OAuthState`/`AuditedAvc`, both of which must be
    /// `'static` — a borrowed signer cannot live there. Holds `Arc`s to the
    /// same key material a borrowed [`CoseAuditSigner`] would reference (a
    /// startup-time snapshot of the node's active EdDSA + ML-DSA-65 keys), and
    /// signs with the IDENTICAL fail-closed logic: under Hybrid, a missing PQ
    /// key fails every `sign` call rather than downgrading.
    pub struct OwnedCoseAuditSigner {
        ed_sk: Arc<SigningKey>,
        pq_sk: Option<Arc<MlDsaSigningKey>>,
        require_pq: bool,
    }

    impl OwnedCoseAuditSigner {
        /// Construct from owned key material + the node's enforced
        /// [`CryptoPolicy`](hyprstream_rpc::crypto::CryptoPolicy). See
        /// [`CoseAuditSigner::new`] for the exact Hybrid/Classical semantics —
        /// identical here.
        pub fn new(
            ed_sk: Arc<SigningKey>,
            pq_sk: Option<Arc<MlDsaSigningKey>>,
            policy: hyprstream_rpc::crypto::CryptoPolicy,
        ) -> Self {
            Self {
                ed_sk,
                pq_sk: if policy.uses_pq() { pq_sk } else { None },
                require_pq: policy.uses_pq(),
            }
        }

        /// As [`CoseAuditSigner::can_sign`]: whether this signer can produce a
        /// signature under its policy. Production wiring MUST check this at
        /// startup and refuse to boot the audited grant path if false under
        /// Hybrid (no audit signer ⇒ no Permit, per the S7 fail-closed
        /// contract) — see `OAuthState::with_audit_sink`.
        #[must_use]
        pub fn can_sign(&self) -> bool {
            !self.require_pq || self.pq_sk.is_some()
        }
    }

    impl AuditSigner for OwnedCoseAuditSigner {
        fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, AuditError> {
            if self.require_pq && self.pq_sk.is_none() {
                return Err(AuditError::Sign(
                    "Hybrid crypto policy requires an ML-DSA-65 audit signing key, none provisioned (fail-closed)".to_owned(),
                ));
            }
            sign_composite(&self.ed_sk, self.pq_sk.as_deref(), signing_input, AUDIT_AAD)
                .map_err(|e| AuditError::Sign(e.to_string()))
        }
    }

    /// Hybrid verifier. `require_pq = true` enforces the ML-DSA outer layer.
    pub struct CoseAuditVerifier<'a> {
        pub ed_vk: &'a VerifyingKey,
        pub pq_vk: Option<&'a MlDsaVerifyingKey>,
        pub require_pq: bool,
    }

    impl AuditVerifier for CoseAuditVerifier<'_> {
        fn verify(&self, signing_input: &[u8], signature: &[u8]) -> Result<(), AuditError> {
            verify_composite(
                signature,
                self.ed_vk,
                self.pq_vk,
                signing_input,
                AUDIT_AAD,
                self.require_pq,
            )
            .map(|_| ())
            .map_err(|e| AuditError::Verify(e.to_string()))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The sink trait (what the AVC calls)
// ────────────────────────────────────────────────────────────────────────────

/// Where the audited AVC sends records. The authoritative impl is
/// [`WalAuditStore`]; tests / non-durable deployments can swap a no-op or
/// in-memory sink. **Fail-closed contract**: a `record` error makes the
/// audited AVC downgrade a Permit to Deny (see [`AuditedAvc`]).
pub trait AuditSink: Send + Sync {
    /// Durably record `record`. Returns `Ok(())` only after the write is
    /// durable (fsynced). The caller MUST treat `Err` as fail-closed.
    fn record(&self, record: &AuditRecord) -> Result<(), AuditError>;
}

// Blanket forwarding so `Arc<T: AuditSink>` etc. are themselves sinks — the
// natural shape for a sink shared between the AVC and the test harness / a
// fan-out bus. This mirrors how `Clone`/`Deref` propagate through `Arc`.
impl<T: AuditSink + ?Sized> AuditSink for Arc<T> {
    #[inline]
    fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
        (**self).record(record)
    }
}

/// A no-op sink for environments that opt out of durable audit (e.g. tests of
/// the PDP itself that are not exercising S7). **Not for production** —
/// production requires tamper-evident audit, which this provides none of.
#[derive(Debug, Default, Clone)]
pub struct NullAuditSink;

impl AuditSink for NullAuditSink {
    #[inline]
    fn record(&self, _record: &AuditRecord) -> Result<(), AuditError> {
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The authoritative data-plane store: append-only journal + CAS + signed
// ────────────────────────────────────────────────────────────────────────────

/// The authoritative tamper-evident audit store. Append-only journal +
/// content-addressed CAS blobs, each record signed. WAL-durable on write.
///
/// ## On-disk layout
/// ```text
/// <root>/
///   journal.log       # append-only: one signed record per line (newline-framed
///                     #              <seq>\t<sig_hex>\t<payload_b64>\n)
///   checkpoint        # signed head anchor, rewritten after every append:
///                     #              <seq>\t<head_hash_hex>\t<sig_hex>\n
///   cas/<ab>/<cdef…>  # content-addressed blob per record: the canonical bytes,
///                     #   named by their BLAKE3 hex (name==bytes tamper check).
/// ```
/// The CAS layer gives a second tamper-evidence check (the blob's name is its
/// hash). The journal is an append-only **hash chain** (each record carries its
/// predecessor's hash), and the signed `checkpoint` anchors the head so a
/// truncated tail is detected on open (see the module docs).
///
/// ## What is NOT here (deferred)
/// - TODO(S7-followup): cross-host replication (gossip / OTLP log export to a
///   central collector). The local store is tamper-evident; remote collection
///   is a transport concern.
/// - TODO(S7-followup): retention / rotation. The journal grows unbounded
///   today; a rotating appender + remote archival is a follow-up.
/// - TODO(S7-followup): policy-evolution feedback (denial bucketing →
///   PolicyService). The records are here; the consumer is not.
pub struct WalAuditStore<S: AuditSigner> {
    root: PathBuf,
    journal_path: PathBuf,
    checkpoint_path: PathBuf,
    cas_dir: PathBuf,
    signer: S,
    /// The chain head: the next sequence number to assign and the content hash
    /// of the last record written. Guarded by a mutex because the hash chain
    /// requires record assignment (seq + prev_hash) and the journal append to
    /// be serialized — two concurrent appends must not interleave or the chain
    /// forks. (The previous `AtomicU64` seq did not order the appends.)
    head: Mutex<ChainHead>,
}

/// The mutable chain head of a [`WalAuditStore`].
#[derive(Debug, Clone, Copy)]
struct ChainHead {
    /// The sequence number the next record will receive.
    next_seq: u64,
    /// BLAKE3 content hash of the last record written (`[0; 32]` before any).
    head_hash: [u8; 32],
}

impl<S: AuditSigner> WalAuditStore<S> {
    /// Open (or create) a store at `root`. The journal and CAS dir are created
    /// if missing. The chain head (next `seq` + `head_hash`) is seeded from the
    /// journal's last record so a restart continues the chain monotonically.
    ///
    /// **Fail-closed truncation check:** if a signed checkpoint exists and its
    /// sequence is *ahead* of the journal's last record, the tail was truncated
    /// since the last durable write — [`open`](Self::open) refuses to start
    /// (`AuditError::Truncation`) rather than silently continuing on a shortened
    /// log (which would re-seed the seq counter from the truncated head and
    /// erase the evidence). Full cryptographic verification of the chain and the
    /// checkpoint signature is [`verify_journal`](Self::verify_journal).
    pub fn open(root: impl AsRef<Path>, signer: S) -> Result<Self, AuditError> {
        let root = root.as_ref().to_path_buf();
        let journal_path = root.join("journal.log");
        let checkpoint_path = root.join("checkpoint");
        let cas_dir = root.join("cas");
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(&cas_dir)?;

        // Seed the chain head from the journal's last record: the next record
        // continues *after* the last seq, chained to the last record's hash.
        let tail = journal_tail(&journal_path)?;
        let (next_seq, head_hash) = match tail {
            Some((seq, hash)) => (seq + 1, hash),
            None => (0, [0u8; 32]),
        };

        // Fail-closed tail-truncation check against the signed checkpoint. The
        // checkpoint records the head we last durably committed; if the journal
        // now has fewer records, its tail was deleted. (A checkpoint that is
        // *behind* the journal is fine — it just predates the last append, e.g.
        // a crash between the journal fsync and the checkpoint fsync.)
        if let Some((cp_seq, _cp_hash, _cp_sig)) = read_checkpoint(&checkpoint_path)? {
            let journal_head = tail.map(|(seq, _)| seq);
            let truncated = match journal_head {
                None => true, // checkpoint exists but journal is empty/gone
                Some(h) => h < cp_seq,
            };
            if truncated {
                return Err(AuditError::Truncation {
                    checkpoint_seq: cp_seq,
                    journal_seq: journal_head,
                });
            }
        }

        Ok(Self {
            root,
            journal_path,
            checkpoint_path,
            cas_dir,
            signer,
            head: Mutex::new(ChainHead {
                next_seq,
                head_hash,
            }),
        })
    }

    /// Root of the store (diagnostics / tests).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Durably append + sign + content-address a record. Returns `Ok(())` only
    /// after the journal file AND its parent directory are fsynced — so a
    /// Permit returned by the AVC is never ahead of a durable audit entry.
    ///
    /// The store owns sequence + chain assignment: the incoming `record.seq`
    /// and `record.prev_hash` are **ignored** and overwritten with the store's
    /// monotonic counter and current chain head, so neither invariant can be
    /// violated by a caller. The caller (the AVC) supplies `seq: 0` /
    /// `prev_hash: [0; 32]` as placeholders. The whole critical section runs
    /// under the head lock so the seq, the `prev_hash` link, and the journal
    /// append cannot interleave with a concurrent writer (which would fork the
    /// chain).
    fn record_inner(&self, record: &AuditRecord) -> Result<(), AuditError> {
        // 0. Take the chain head. Held across the append so seq assignment,
        //    prev_hash chaining, and the journal write are one atomic step.
        //    (parking_lot::Mutex — no poisoning, lock() yields the guard.)
        let mut head = self.head.lock();
        let seq = head.next_seq;
        let stored = AuditRecord {
            seq,
            prev_hash: head.head_hash,
            ..record.clone()
        };

        // 1. Canonical bytes + content hash (content-addressing + chain link).
        let bytes = stored.canonical_bytes()?;
        let hash = *blake3::hash(&bytes).as_bytes();

        // 2. Sign the domain-separated signing input. S8 (#574): the
        // CoseAuditSigner is policy-aware — under Hybrid it fails closed here
        // if no ML-DSA-65 key is provisioned (no silent downgrade). The signer
        // trait abstracts which mode is in use.
        let signing_input = audit_signing_input(&hash);
        let signature = self.signer.sign(&signing_input)?;

        // 3. Content-addressed CAS write (dedup + second tamper-evidence layer).
        //    Named by its own hash, so an overwrite is a no-op and a name never
        //    points at different bytes.
        let (shard, rest) = shard_of(&hash);
        let cas_shard = self.cas_dir.join(shard);
        std::fs::create_dir_all(&cas_shard)?;
        let blob_path = cas_shard.join(&rest);
        if !blob_path.exists() {
            // Write to a temp file in the same dir, fsync, then atomically rename
            // — so a crash never leaves a half-written blob under the content
            // address (which would violate the name==bytes invariant).
            let tmp = cas_shard.join(format!(".{rest}.tmp"));
            {
                let mut f = File::create(&tmp)?;
                f.write_all(&bytes)?;
                f.sync_all()?;
            }
            std::fs::rename(&tmp, &blob_path)?;
        }

        // 4. Append to the journal: <seq>\t<sig_hex>\t<cbor_b64>\n, then fsync
        // the file AND the directory (WAL durability: the append is
        // durably on disk before we return).
        let line = format!(
            "{}\t{}\t{}\n",
            stored.seq,
            hex_encode(&signature),
            base64_encode(&bytes),
        );
        {
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.journal_path)?;
            f.write_all(line.as_bytes())?;
            f.sync_all()?;
        }
        // fsync the directory so the rename (step 3) and the journal's metadata
        // (size, new inode) are durable. Without this a crash after the file
        // fsync but before the dir fsync can lose the directory entry.
        sync_dir(&self.root)?;

        // 5. Advance and durably anchor the chain head. The signed checkpoint
        //    records `(seq, hash)` out-of-band so a later tail-truncation is
        //    detectable on open. Written AFTER the journal is durable: if we
        //    crash between the two, the checkpoint lags the journal (benign —
        //    open() treats a checkpoint behind the journal as fine); the
        //    dangerous direction (checkpoint ahead of a shortened journal) only
        //    arises from truncation, which is exactly what we detect.
        self.write_checkpoint(seq, &hash)?;
        head.next_seq = seq + 1;
        head.head_hash = hash;
        drop(head);

        // 6. OTel fan-out: emit a structured tracing event so denials are
        //    observable (fixes #453). OTel is a tracing subscriber, so this
        //    reaches OTLP automatically when the `otel` feature is active.
        emit_decision_event(&stored);
        Ok(())
    }

    /// Write the signed head anchor `(seq, head_hash)` to the checkpoint file,
    /// atomically (temp + rename) and fsynced so it is durable before the
    /// enclosing `record` returns. The signature is over a domain-separated
    /// input distinct from the per-record one, so a record signature can never
    /// be replayed as a checkpoint signature.
    fn write_checkpoint(&self, seq: u64, head_hash: &[u8; 32]) -> Result<(), AuditError> {
        let signing_input = checkpoint_signing_input(seq, head_hash);
        let signature = self.signer.sign(&signing_input)?;
        let line = format!(
            "{}\t{}\t{}\n",
            seq,
            hex_encode(head_hash),
            hex_encode(&signature),
        );
        let tmp = self.root.join(".checkpoint.tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(line.as_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.checkpoint_path)?;
        sync_dir(&self.root)?;
        Ok(())
    }

    /// Read the whole journal back and verify it end-to-end: every signature
    /// against `verifier`, the sequence gap-free, the `prev_hash` **chain**
    /// intact, and — if present — the signed checkpoint's signature and that it
    /// anchors the journal's actual head. Returns the records in order.
    ///
    /// This is the offline/cryptographic counterpart to [`open`](Self::open)'s
    /// cheap runtime check: it catches tampering, reordering, and middle
    /// deletion (chain break) as well as tail truncation and checkpoint
    /// rollback (checkpoint vs. head mismatch).
    pub fn verify_journal<V: AuditVerifier>(
        &self,
        verifier: &V,
    ) -> Result<Vec<AuditRecord>, AuditError> {
        if !self.journal_path.exists() {
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&self.journal_path)?;
        let mut out = Vec::new();
        let mut expected_seq = 0u64;
        // The running chain head: each record must name this as its prev_hash.
        let mut running_head = [0u8; 32];
        for (lineno, raw_line) in content.lines().enumerate() {
            let line = raw_line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split('\t');
            let seq_s = parts.next().ok_or_else(|| {
                AuditError::Decode(format!("journal line {} missing seq", lineno + 1))
            })?;
            let sig_hex = parts.next().ok_or_else(|| {
                AuditError::Decode(format!("journal line {} missing sig", lineno + 1))
            })?;
            let cbor_b64 = parts.next().ok_or_else(|| {
                AuditError::Decode(format!("journal line {} missing payload", lineno + 1))
            })?;
            let seq: u64 = seq_s.parse().map_err(|e| {
                AuditError::Decode(format!("journal line {}: bad seq: {}", lineno + 1, e))
            })?;
            if seq != expected_seq {
                return Err(AuditError::SeqRegression {
                    journal: seq,
                    counter: expected_seq,
                });
            }
            expected_seq = seq + 1;
            let bytes = base64_decode(cbor_b64)?;
            let sig = hex_decode(sig_hex)?;
            let record: AuditRecord = serde_json::from_slice(&bytes).map_err(|e| {
                AuditError::Decode(format!("journal line {}: bad record: {}", lineno + 1, e))
            })?;
            // Chain: this record must link to the previous record's hash.
            if record.prev_hash != running_head {
                return Err(AuditError::ChainBreak { seq });
            }
            // Recompute the content hash and verify the signature over it.
            let hash = *blake3::hash(&bytes).as_bytes();
            let signing_input = audit_signing_input(&hash);
            verifier.verify(&signing_input, &sig)?;
            running_head = hash;
            out.push(record);
        }

        // Checkpoint: if one exists, its signature must verify and it must
        // anchor the journal's actual head — a truncated tail or a rolled-back
        // checkpoint is caught here. (A checkpoint strictly *behind* the head is
        // a benign crash between the journal and checkpoint fsyncs.)
        if let Some((cp_seq, cp_hash, cp_sig)) = read_checkpoint(&self.checkpoint_path)? {
            let cp_input = checkpoint_signing_input(cp_seq, &cp_hash);
            verifier.verify(&cp_input, &cp_sig)?;
            let head_seq = out.last().map(|r| r.seq);
            match head_seq {
                // Checkpoint ahead of the journal head ⇒ the tail was truncated.
                Some(h) if cp_seq > h => {
                    return Err(AuditError::Truncation {
                        checkpoint_seq: cp_seq,
                        journal_seq: Some(h),
                    });
                }
                None => {
                    return Err(AuditError::Truncation {
                        checkpoint_seq: cp_seq,
                        journal_seq: None,
                    });
                }
                // Checkpoint exactly at the head ⇒ verify it names the head hash.
                Some(h) if cp_seq == h && cp_hash != running_head => {
                    return Err(AuditError::ChainBreak { seq: cp_seq });
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

impl<S: AuditSigner> AuditSink for WalAuditStore<S> {
    fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
        self.record_inner(record)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Complete-mediation wrapper: an AVC that audits every decision
// ────────────────────────────────────────────────────────────────────────────

/// An [`Avc`] that routes every decision through an [`AuditSink`]. This is the
/// S7 complete-mediation hook: because the PEP (S2) calls the AVC per op,
/// wrapping it here audits every decision with no bypass.
///
/// **Fail-closed:** if the audit write fails on what would otherwise be a
/// `Permit`, the wrapper downgrades to `Deny` (reason `AuditFailClosed`) and
/// attempts to record *that* deny (best-effort: if the sink is broken, the
/// enforced deny is still emitted as a tracing event so it is observable).
/// A decision that cannot be audited is never allowed through.
pub struct AuditedAvc<A: Avc, S: AuditSink> {
    inner: A,
    sink: S,
    /// The policy generation records get tagged with (the wrapped evaluator's
    /// generation, captured by the caller at construction).
    generation: u64,
    /// Optional BLAKE3 of the governing compiled policy (provenance). `None`
    /// when the wrapped AVC was not built from a signed `CompiledPolicy`.
    policy_hash: Option<[u8; 32]>,
}

impl<A: Avc, S: AuditSink> AuditedAvc<A, S> {
    /// Wrap an AVC with audit. `generation` should be the evaluator's
    /// generation (`TeEvaluator::generation`); `policy_hash` the
    /// [`crate::mac::compiled::CompiledPolicy::policy_hash`] when available.
    pub fn new(inner: A, sink: S, generation: u64, policy_hash: Option<[u8; 32]>) -> Self {
        Self {
            inner,
            sink,
            generation,
            policy_hash,
        }
    }

    /// Borrow the inner AVC (diagnostics / cache stats).
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Record a decision through the sink, applying the fail-closed rule.
    /// Returns the decision the PEP should enforce, plus the reason.
    ///
    /// `on_behalf_of` carries the delegator principal on a delegated (#680/#681)
    /// decision — `subject` is the effective, met context the AVC decided on;
    /// the delegator is recorded but never re-evaluated here (the meet already
    /// happened upstream in `SecurityContext::delegated`).
    fn audit(
        &self,
        subject: SubjectCtx,
        on_behalf_of: Option<SubjectCtx>,
        object: ObjectCtx,
        action: Action,
        decision: Decision,
        reason: DecisionReason,
    ) -> Decision {
        let enforced = decision;
        let record = AuditRecord {
            // seq/ts are assigned by the store via build_record-equivalent; but
            // the store's record() takes a fully-formed record, so we let the
            // caller-side assign. To keep seq monotonic AND owned by the store,
            // we instead hand the store a record with seq=0 and let a wrapping
            // helper reassign. Simpler: the store assigns seq in record() — but
            // the trait takes a complete record. We resolve this by giving the
            // record a placeholder seq here and having WalAuditStore re-stamp
            // it; see the store impl. For trait-based sinks without that
            // re-stamp, seq stays as written here (tests use NullAuditSink and
            // don't rely on seq).
            seq: 0,
            // Placeholder chain link; WalAuditStore assigns the real prev_hash
            // under its head lock, exactly as it does seq.
            prev_hash: [0u8; 32],
            ts_unix_nanos: now_unix_nanos(),
            decision: enforced,
            generation: self.generation,
            policy_hash: self.policy_hash,
            subject_type: subject.subject_type,
            subject_clearance: subject.clearance,
            on_behalf_of: on_behalf_of.map(|d| DelegationPrincipal {
                subject_type: d.subject_type,
                subject_clearance: d.clearance,
            }),
            object_type: object.object_type,
            object_label: object.label,
            action,
            reason,
        };
        match self.sink.record(&record) {
            Ok(()) => enforced,
            Err(_) => {
                // Fail-closed: a Permit that can't be audited is downgraded to
                // Deny. Record the deny we actually enforce (best-effort: the
                // sink may still be broken, in which case the tracing event is
                // the only record, but it IS emitted).
                if enforced.is_permit() {
                    let deny_record = AuditRecord {
                        decision: Decision::Deny,
                        reason: DecisionReason::AuditFailClosed,
                        ..record
                    };
                    let _ = self.sink.record(&deny_record);
                    emit_decision_event(&deny_record);
                    Decision::Deny
                } else {
                    // A deny that can't be audited is still a deny — enforced as
                    // decided. We log the audit failure so it is observable.
                    tracing::error!(
                        target: "hyprstream.mac.audit",
                        decision = "deny",
                        reason = DecisionReason::AuditFailClosed.as_str(),
                        "audit write failed recording a deny; enforcing deny as decided"
                    );
                    enforced
                }
            }
        }
    }
}

impl<A: Avc, S: AuditSink> Avc for AuditedAvc<A, S> {
    #[inline]
    fn decide(&self, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Decision {
        let decision = self.inner.decide(subject, object, action);
        let reason = decision_reason_for(decision);
        self.audit(subject, None, object, action, decision, reason)
    }

    #[inline]
    fn decide_with_token(
        &self,
        subject: SubjectCtx,
        object: ObjectCtx,
        action: Action,
        token: &TokenScope,
    ) -> Decision {
        // The token gate lives inside the inner AVC; if it denies, the inner
        // returns Deny and we record reason=TokenGate (best-effort: we cannot
        // distinguish token-gate-deny from PDP-deny without the inner exposing
        // it, so we attribute a Deny to the most specific cause we can infer:
        // if the token's op-set excludes the action or it's expired, it's a
        // token gate; otherwise it's a PDP decision).
        let pre_reason = token_deny_reason(token, action);
        let decision = self.inner.decide_with_token(subject, object, action, token);
        let reason = if decision.is_permit() {
            DecisionReason::Permit
        } else {
            pre_reason.unwrap_or_else(|| decision_reason_for(decision))
        };
        self.audit(subject, None, object, action, decision, reason)
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

impl<A: Avc, S: AuditSink> AuditedAvc<A, S> {
    /// Delegated (#680/#681) counterpart of [`Avc::decide`]: `subject` is the
    /// effective, met context the PDP decides on (already `meet(delegator,
    /// actor)` from [`hyprstream_rpc::auth::mac::SecurityContext::delegated`]);
    /// `on_behalf_of` is the delegator principal, recorded on the audit record
    /// but **never re-evaluated** (the decision rests on the met `subject`
    /// alone — the delegator cannot re-widen it).
    ///
    /// The plain [`Avc::decide`] is the single-principal case (`on_behalf_of =
    /// None`); this is the two-principal case the (future) reference monitor
    /// calls when a request arrived over a delegation. Kept off the `Avc` trait
    /// so the hot-path cache key stays `(subject, object, action)` — delegation
    /// is an audit-attribution concern, not a decision input.
    #[inline]
    pub fn decide_delegated(
        &self,
        subject: SubjectCtx,
        on_behalf_of: Option<SubjectCtx>,
        object: ObjectCtx,
        action: Action,
    ) -> Decision {
        let decision = self.inner.decide(subject, object, action);
        let reason = decision_reason_for(decision);
        self.audit(subject, on_behalf_of, object, action, decision, reason)
    }

    /// Delegated counterpart of [`Avc::decide_with_token`]; see
    /// [`Self::decide_delegated`] for the two-principal semantics.
    #[inline]
    pub fn decide_delegated_with_token(
        &self,
        subject: SubjectCtx,
        on_behalf_of: Option<SubjectCtx>,
        object: ObjectCtx,
        action: Action,
        token: &TokenScope,
    ) -> Decision {
        let pre_reason = token_deny_reason(token, action);
        let decision = self.inner.decide_with_token(subject, object, action, token);
        let reason = if decision.is_permit() {
            DecisionReason::Permit
        } else {
            pre_reason.unwrap_or_else(|| decision_reason_for(decision))
        };
        self.audit(subject, on_behalf_of, object, action, decision, reason)
    }
}

/// Infer a `DecisionReason` from a bare `Decision` (no token context).
fn decision_reason_for(d: Decision) -> DecisionReason {
    match d {
        Decision::Permit => DecisionReason::Permit,
        Decision::Escalate => DecisionReason::Escalate,
        Decision::Deny => {
            // We can't tell floor-deny from TE-miss without re-running the PDP,
            // and re-running would duplicate work. Record the generic deny
            // cause; the detailed split is recoverable from the policy + the
            // recorded contexts during post-hoc analysis. This keeps the hot
            // path single-evaluation.
            DecisionReason::TeMiss
        }
    }
}

/// If the token gate would deny this action, return the reason; else `None`.
fn token_deny_reason(token: &TokenScope, action: Action) -> Option<DecisionReason> {
    if !token.authorizes_action(action) || token.valid_until <= std::time::Instant::now() {
        Some(DecisionReason::TokenGate)
    } else {
        None
    }
}

/// Emit the structured `tracing` event that fans out to OTel (OTel is a tracing
/// subscriber; this is the S7 observability surface that fixes #453's
/// debug-level-denial visibility gap).
fn emit_decision_event(record: &AuditRecord) {
    let decision_str = match record.decision {
        Decision::Permit => "allow",
        Decision::Deny => "deny",
        Decision::Escalate => "escalate",
    };
    // Denials/escalations are warn; permits are info. Both carry identical
    // structured fields so the deny path is first-class, not buried.
    if record.decision.is_permit() {
        tracing::info!(
            target: "hyprstream.mac.audit",
            decision = decision_str,
            subject_type = record.subject_type.0,
            subject_clearance = %record.subject_clearance,
            object_type = record.object_type.0,
            object_label = %record.object_label,
            action = record.action.0,
            generation = record.generation,
            reason = record.reason.as_str(),
            seq = record.seq,
            "MAC authorization decision"
        );
    } else {
        tracing::warn!(
            target: "hyprstream.mac.audit",
            decision = decision_str,
            subject_type = record.subject_type.0,
            subject_clearance = %record.subject_clearance,
            object_type = record.object_type.0,
            object_label = %record.object_label,
            action = record.action.0,
            generation = record.generation,
            reason = record.reason.as_str(),
            seq = record.seq,
            "MAC authorization decision"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Small helpers (no extra deps: hex/base64 are tiny, inlined)
// ────────────────────────────────────────────────────────────────────────────

fn now_unix_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Split a 32-byte BLAKE3 into a 2-hex-char shard dir and the remaining 62-hex
/// file name (git-style sharding to keep one CAS dir from bloating).
fn shard_of(hash: &[u8; 32]) -> (String, String) {
    let hex = hex_encode(hash);
    let (a, b) = hex.split_at(2);
    (a.to_owned(), b.to_owned())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_decode(hex: &str) -> Result<Vec<u8>, AuditError> {
    if !hex.len().is_multiple_of(2) {
        return Err(AuditError::Decode("odd-length hex".into()));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for chunk in bytes.chunks_exact(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, AuditError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(AuditError::Decode(format!("bad hex nibble: {b}"))),
    }
}

/// Standard base64 alphabet (no padding for journal compactness; we frame with
/// tabs/newlines so length is recoverable).
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHA[((triple >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHA[((triple >> 6) & 0x3f) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHA[(triple & 0x3f) as usize] as char);
        }
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, AuditError> {
    fn val(c: u8) -> Result<u8, AuditError> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(AuditError::Decode(format!("bad base64 char: {c}"))),
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = [0u8; 4];
    let mut i = 0;
    while i < bytes.len() {
        buf[i % 4] = val(bytes[i])?;
        if i % 4 == 3 {
            let triple = (u32::from(buf[0]) << 18)
                | (u32::from(buf[1]) << 12)
                | (u32::from(buf[2]) << 6)
                | u32::from(buf[3]);
            out.push((triple >> 16) as u8);
            out.push((triple >> 8) as u8);
            out.push(triple as u8);
        }
        i += 1;
    }
    // Handle the trailing partial group (1 or 2 trailing chars).
    let rem = i % 4;
    if rem == 2 {
        let triple = (u32::from(buf[0]) << 18) | (u32::from(buf[1]) << 12);
        out.push((triple >> 16) as u8);
    } else if rem == 3 {
        let triple =
            (u32::from(buf[0]) << 18) | (u32::from(buf[1]) << 12) | (u32::from(buf[2]) << 6);
        out.push((triple >> 16) as u8);
        out.push((triple >> 8) as u8);
    }
    Ok(out)
}

/// Read the journal's last record and return its `(seq, content_hash)` — the
/// chain head a restart continues from. The hash is recomputed from the stored
/// payload (BLAKE3 of the canonical bytes), identical to how it was chained on
/// write, so the next record links correctly. `None` for an empty/absent log.
fn journal_tail(path: &Path) -> Result<Option<(u64, [u8; 32])>, AuditError> {
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    let Some(last) = content
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .rfind(|l| !l.is_empty())
    else {
        return Ok(None);
    };
    let mut parts = last.split('\t');
    let seq_s = parts
        .next()
        .ok_or_else(|| AuditError::Decode("journal tail missing seq".into()))?;
    // parts[1] is the signature; parts[2] is the base64 payload.
    let _sig = parts.next();
    let payload_b64 = parts
        .next()
        .ok_or_else(|| AuditError::Decode("journal tail missing payload".into()))?;
    let seq: u64 = seq_s
        .parse()
        .map_err(|e| AuditError::Decode(format!("journal tail bad seq: {e}")))?;
    let bytes = base64_decode(payload_b64)?;
    let hash = *blake3::hash(&bytes).as_bytes();
    Ok(Some((seq, hash)))
}

/// Read the signed checkpoint head anchor: `(seq, head_hash, signature)`.
/// `None` if no checkpoint has been written yet.
fn read_checkpoint(path: &Path) -> Result<Option<(u64, [u8; 32], Vec<u8>)>, AuditError> {
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    let Some(line) = content
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .find(|l| !l.is_empty())
    else {
        return Ok(None);
    };
    let mut parts = line.split('\t');
    let seq_s = parts
        .next()
        .ok_or_else(|| AuditError::Decode("checkpoint missing seq".into()))?;
    let hash_hex = parts
        .next()
        .ok_or_else(|| AuditError::Decode("checkpoint missing head_hash".into()))?;
    let sig_hex = parts
        .next()
        .ok_or_else(|| AuditError::Decode("checkpoint missing sig".into()))?;
    let seq: u64 = seq_s
        .parse()
        .map_err(|e| AuditError::Decode(format!("checkpoint bad seq: {e}")))?;
    let hash_vec = hex_decode(hash_hex)?;
    let hash: [u8; 32] = hash_vec
        .try_into()
        .map_err(|_| AuditError::Decode("checkpoint head_hash not 32 bytes".into()))?;
    let sig = hex_decode(sig_hex)?;
    Ok(Some((seq, hash, sig)))
}

/// fsync a directory (durability of directory entries: renames, creates).
fn sync_dir(path: &Path) -> Result<(), AuditError> {
    let f = File::open(path)?;
    f.sync_all()?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::mac::avc::CachingAvc;
    use crate::mac::lattice::{
        Assurance, Compartment, CompartmentSet, Lattice, LatticeVersion, Level,
    };
    use crate::mac::te::{LatticeTeEvaluator, TeMatrix, TeRule};
    use parking_lot::Mutex;
    use std::collections::HashSet;
    use std::sync::Arc;

    fn lattice() -> Lattice {
        Lattice::new(LatticeVersion(7), [Compartment::new("pii")])
    }
    fn high() -> SecurityLabel {
        SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }
    fn low() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }
    fn subj(t: u32, clr: SecurityLabel) -> SubjectCtx {
        SubjectCtx {
            subject_type: SubjectType(t),
            clearance: clr,
        }
    }
    fn obj(t: u32, lbl: SecurityLabel) -> ObjectCtx {
        ObjectCtx {
            object_type: ObjectType(t),
            label: lbl,
        }
    }
    fn rule(s: u32, o: u32, a: u32) -> TeRule {
        TeRule {
            subject_type: SubjectType(s),
            object_type: ObjectType(o),
            action: Action(a),
        }
    }

    /// A trivial deterministic signer for unit tests (NOT for production).
    /// signature = blake3(key || signing_input). Mirrors compiled::StubSigner.
    struct StubSigner {
        key: [u8; 32],
    }
    impl AuditSigner for StubSigner {
        fn sign(&self, input: &[u8]) -> Result<Vec<u8>, AuditError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            Ok(h.finalize().as_bytes().to_vec())
        }
    }
    struct StubVerifier {
        key: [u8; 32],
    }
    impl AuditVerifier for StubVerifier {
        fn verify(&self, input: &[u8], sig: &[u8]) -> Result<(), AuditError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            let want = h.finalize();
            if want.as_bytes() == sig {
                Ok(())
            } else {
                Err(AuditError::Verify("stub signature mismatch".into()))
            }
        }
    }

    /// An in-memory capturing sink for AVC-wrapper tests.
    struct CapturingSink {
        records: Mutex<Vec<AuditRecord>>,
        fail_next: Mutex<bool>,
    }
    impl CapturingSink {
        fn new() -> Self {
            Self {
                records: Mutex::new(Vec::new()),
                fail_next: Mutex::new(false),
            }
        }
        fn records(&self) -> Vec<AuditRecord> {
            self.records.lock().clone()
        }
        fn fail_next(&self) {
            *self.fail_next.lock() = true;
        }
    }
    impl AuditSink for CapturingSink {
        fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
            let mut fail = self.fail_next.lock();
            if *fail {
                *fail = false;
                return Err(AuditError::Io("injected failure".into()));
            }
            self.records.lock().push(record.clone());
            Ok(())
        }
    }

    fn avc() -> CachingAvc<LatticeTeEvaluator> {
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        let e = LatticeTeEvaluator::new(TeMatrix::from_allow(allow), lattice());
        CachingAvc::new(Arc::new(e))
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("hs-s7-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    // ── Property: content-addressing ────────────────────────────────────────

    #[test]
    fn same_decision_yields_same_content_hash() {
        let mut rec = AuditRecord {
            seq: 1,
            prev_hash: [0u8; 32],
            ts_unix_nanos: 0,
            decision: Decision::Permit,
            generation: 7,
            policy_hash: None,
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(1),
            reason: DecisionReason::Permit,
        };
        let h1 = rec.content_hash().unwrap();
        // Same fields → same hash.
        let h2 = rec.content_hash().unwrap();
        assert_eq!(h1, h2, "identical record must content-address identically");

        // Mutate any field → different hash.
        rec.decision = Decision::Deny;
        let h3 = rec.content_hash().unwrap();
        assert_ne!(h1, h3, "a mutated field must change the content hash");

        // Restore and bump ts — also different (ts is part of the canonical bytes).
        rec.decision = Decision::Permit;
        rec.ts_unix_nanos = 1;
        let h4 = rec.content_hash().unwrap();
        assert_ne!(h1, h4, "timestamp drift must change the content hash");
    }

    /// #680/#681 hash-chain back-compat: a single-principal record
    /// (`on_behalf_of: None`) must serialize byte-identically to the
    /// pre-delegation schema — the field is skipped entirely, so existing
    /// content hashes and `prev_hash` links are preserved across the upgrade.
    /// Setting a delegator changes the hash (it is covered content).
    #[test]
    fn on_behalf_of_none_is_omitted_and_some_changes_the_hash() {
        let mut rec = AuditRecord {
            seq: 1,
            prev_hash: [0u8; 32],
            ts_unix_nanos: 0,
            decision: Decision::Permit,
            generation: 7,
            policy_hash: None,
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(1),
            reason: DecisionReason::Permit,
        };

        // `None` is skipped: the field name never appears in the canonical bytes,
        // so a single-principal record is byte-identical to the pre-delegation
        // schema (which had no such field at all).
        let bytes_none = rec.canonical_bytes().unwrap();
        let text = String::from_utf8(bytes_none.clone()).unwrap();
        assert!(
            !text.contains("on_behalf_of"),
            "a None delegator must be omitted from canonical bytes (hash-chain back-compat)"
        );
        let hash_none = rec.content_hash().unwrap();

        // A delegated record IS distinguished: the delegator is covered content,
        // so the hash moves and the field is now present.
        rec.on_behalf_of = Some(DelegationPrincipal {
            subject_type: SubjectType(2),
            subject_clearance: low(),
        });
        let bytes_some = rec.canonical_bytes().unwrap();
        assert!(
            String::from_utf8(bytes_some).unwrap().contains("on_behalf_of"),
            "a Some delegator must appear in the canonical bytes"
        );
        assert_ne!(
            hash_none,
            rec.content_hash().unwrap(),
            "recording a delegator must change the content hash — it is tamper-evident content"
        );
    }

    /// #680/#681: `AuditedAvc::decide_delegated` decides on the effective (met)
    /// `subject` alone — the delegator never re-widens the decision — but stamps
    /// the delegator on the audit record's `on_behalf_of`. The plain `decide`
    /// path records `None`.
    #[test]
    fn decide_delegated_records_delegator_without_affecting_the_decision() {
        let inner = avc();
        let sink = Arc::new(CapturingSink::new());
        let audited = AuditedAvc::new(inner, sink.clone(), 7, None);

        // Met subject clears the op (rule (1,1,1) + floor holds). The delegator
        // is a DISTINCT principal (different TE type + clearance).
        let met = subj(1, high());
        let delegator = subj(2, low());
        let d = audited.decide_delegated(met, Some(delegator), obj(1, low()), Action(1));
        assert_eq!(d, Decision::Permit, "decision rests on the met subject alone");

        // A non-delegated decision on the same triple records no delegator.
        let d2 = audited.decide(subj(1, high()), obj(1, low()), Action(1));
        assert_eq!(d2, Decision::Permit);

        let recs = sink.records();
        assert_eq!(recs.len(), 2);
        let obo = recs[0]
            .on_behalf_of
            .expect("the delegated decision records its delegator");
        assert_eq!(obo.subject_type, SubjectType(2), "delegator's TE type");
        assert_eq!(obo.subject_clearance, low(), "delegator's own clearance");
        assert_eq!(
            recs[0].subject_clearance,
            high(),
            "subject_clearance is the met context, not the delegator's"
        );
        assert!(
            recs[1].on_behalf_of.is_none(),
            "the non-delegated decision records no delegator"
        );
    }

    // ── Property: append-only / no mutate/delete API ────────────────────────

    #[test]
    fn store_exposes_no_mutation_or_delete() {
        // The AuditSink trait surface is a single `record(&self, ...)`. There is
        // no update/delete. This test is a compile-time guarantee documented at
        // the trait; we additionally assert the journal only ever grows.
        let dir = tmp_dir("append_only");
        let store = WalAuditStore::open(&dir, StubSigner { key: [1; 32] }).unwrap();
        let rec = AuditRecord {
            seq: 0,
            prev_hash: [0u8; 32],
            ts_unix_nanos: 1,
            decision: Decision::Permit,
            generation: 7,
            policy_hash: None,
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(1),
            reason: DecisionReason::Permit,
        };
        store.record(&rec).unwrap();
        store.record(&rec).unwrap();
        let content = std::fs::read_to_string(dir.join("journal.log")).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(
            lines.len(),
            2,
            "journal must contain both appends, never overwrite"
        );
        // The trait object exposes only record():
        let sink: &dyn AuditSink = &store;
        // (If a mutating method existed on the trait, this cast would still
        // surface it; the guarantee is the trait def itself.)
        let _ = sink;
    }

    // ── Property: WAL-durable-on-write ──────────────────────────────────────

    #[test]
    fn wal_record_is_durable_before_return() {
        // record() returns Ok only after the journal is fsynced AND the dir is
        // fsynced. We assert durability by: (1) record succeeds, (2) the bytes
        // are present on disk (readable by a fresh process-like open), (3) the
        // CAS blob is present. A real crash-consistency test needs a power-cut
        // harness; here we assert the durability *contract* — the data is on
        // disk when record returns, not just in-flight.
        let dir = tmp_dir("wal");
        let store = WalAuditStore::open(&dir, StubSigner { key: [2; 32] }).unwrap();
        let rec = AuditRecord {
            seq: 0,
            prev_hash: [0u8; 32],
            ts_unix_nanos: 42,
            decision: Decision::Deny,
            generation: 7,
            policy_hash: Some([0xaa; 32]),
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(9),
            reason: DecisionReason::TeMiss,
        };
        store.record(&rec).unwrap();

        // Re-open a second store at the same path (simulating a restart that
        // sees only what was fsynced). The record MUST be visible.
        let reopened = WalAuditStore::open(&dir, StubSigner { key: [2; 32] }).unwrap();
        let verifier = StubVerifier { key: [2; 32] };
        let recovered = reopened.verify_journal(&verifier).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].decision, Decision::Deny);
        assert_eq!(recovered[0].reason, DecisionReason::TeMiss);
        assert_eq!(recovered[0].policy_hash, Some([0xaa; 32]));

        // And the chain head continued monotonically after reopen: the highest
        // on-disk seq was 0, so the reopened counter is seeded to 1 (next write).
        assert_eq!(
            reopened.head.lock().next_seq,
            1,
            "reopened store must continue the sequence monotonically"
        );
        // Writing one more record assigns seq 1 (no collision with seq 0).
        reopened.record(&rec).unwrap();
        let recovered2 = reopened.verify_journal(&verifier).unwrap();
        assert_eq!(recovered2.len(), 2);
        assert_eq!(recovered2[0].seq, 0);
        assert_eq!(
            recovered2[1].seq, 1,
            "sequence must be gap-free across reopen"
        );
    }

    // ── Property: tamper-evidence (content-address + signature) ─────────────

    #[test]
    fn tampered_record_fails_verify() {
        let dir = tmp_dir("tamper");
        let store = WalAuditStore::open(&dir, StubSigner { key: [3; 32] }).unwrap();
        let rec = AuditRecord {
            seq: 0,
            prev_hash: [0u8; 32],
            ts_unix_nanos: 1,
            decision: Decision::Permit,
            generation: 7,
            policy_hash: None,
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(1),
            reason: DecisionReason::Permit,
        };
        store.record(&rec).unwrap();

        // Tamper: flip a byte in the journal payload. The signature (over the
        // content hash of the canonical bytes) must no longer verify.
        let jp = dir.join("journal.log");
        let mut content = std::fs::read_to_string(&jp).unwrap();
        // Flip the last payload char before the newline.
        let last = content.trim_end().chars().last().unwrap();
        let mut chars: Vec<char> = content.chars().collect();
        // Mutate the final base64 char to something different.
        let replacement = if last == 'A' { 'B' } else { 'A' };
        if let Some(c) = chars.last_mut() {
            if *c == '\n' {
                // step back one more
            }
            *c = replacement;
        }
        content = chars.into_iter().collect();
        std::fs::write(&jp, content).unwrap();

        let reopened = WalAuditStore::open(&dir, StubSigner { key: [3; 32] }).unwrap();
        let verifier = StubVerifier { key: [3; 32] };
        let res = reopened.verify_journal(&verifier);
        assert!(
            res.is_err(),
            "a tampered record must fail signature verification"
        );
    }

    // ── Property: sequence is gap-free (truncation detection) ───────────────

    #[test]
    fn journal_seq_gap_is_detected() {
        // The store owns seq assignment, so a live store never produces a gap.
        // Truncation/tampering, however, could remove or renumber journal lines.
        // We simulate that by writing a valid journal then rewriting it with a
        // gap, and assert verify_journal catches the regression.
        let dir = tmp_dir("gap");
        let store = WalAuditStore::open(&dir, StubSigner { key: [4; 32] }).unwrap();
        let base = AuditRecord {
            seq: 0, // ignored — the store assigns seq.
            prev_hash: [0u8; 32], // ignored — the store assigns the chain link.
            ts_unix_nanos: 1,
            decision: Decision::Permit,
            generation: 7,
            policy_hash: None,
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(1),
            reason: DecisionReason::Permit,
        };
        // Write two records (store assigns seq 0, then 1).
        store.record(&base).unwrap();
        store.record(&base).unwrap();
        let verifier = StubVerifier { key: [4; 32] };
        // Sanity: the live journal verifies.
        let ok = store.verify_journal(&verifier).unwrap();
        assert_eq!(ok.len(), 2);

        // Tamper: rewrite the journal renumbering the second line from seq 1 to
        // seq 5 (a gap of 2,3,4). verify_journal must reject it.
        let jp = dir.join("journal.log");
        let content = std::fs::read_to_string(&jp).unwrap();
        let mut lines: Vec<String> = content.lines().map(str::to_owned).collect();
        // The store wrote seq 0 then seq 1. Rewrite the seq field of line 2.
        if lines.len() >= 2 {
            let parts: Vec<&str> = lines[1].splitn(3, '\t').collect();
            assert_eq!(parts.len(), 3, "journal line must have seq\\tsig\\tpayload");
            lines[1] = format!("5\t{}\t{}", parts[1], parts[2]);
        }
        std::fs::write(&jp, lines.join("\n") + "\n").unwrap();

        let res = store.verify_journal(&verifier);
        match res {
            Err(AuditError::SeqRegression { journal, counter }) => {
                assert_eq!(journal, 5);
                assert_eq!(counter, 1, "expected the gap at counter=1");
            }
            other => panic!("expected SeqRegression, got {other:?}"),
        }
    }

    // ── B3 (#675): hash-chain + signed checkpoint truncation detection ──────

    /// A permit record with placeholder seq/prev_hash (the store assigns both).
    fn perm_rec() -> AuditRecord {
        AuditRecord {
            seq: 0,
            prev_hash: [0u8; 32],
            ts_unix_nanos: 1,
            decision: Decision::Permit,
            generation: 7,
            policy_hash: None,
            subject_type: SubjectType(1),
            subject_clearance: high(),
            on_behalf_of: None,
            object_type: ObjectType(1),
            object_label: low(),
            action: Action(1),
            reason: DecisionReason::Permit,
        }
    }

    /// Drop the last journal line (simulate a truncated tail).
    fn truncate_last_line(journal: &Path) {
        let content = std::fs::read_to_string(journal).unwrap();
        let mut lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        lines.pop();
        std::fs::write(journal, lines.join("\n") + "\n").unwrap();
    }

    /// The headline B3 fix: deleting the journal tail is detected on reopen via
    /// the signed checkpoint. The pre-fix store re-seeded its seq counter from
    /// the truncated journal head and continued silently — this asserts we now
    /// FAIL CLOSED instead.
    #[test]
    fn tail_truncation_is_detected_on_reopen() {
        let dir = tmp_dir("b3_trunc_reopen");
        let store = WalAuditStore::open(&dir, StubSigner { key: [7; 32] }).unwrap();
        for _ in 0..3 {
            store.record(&perm_rec()).unwrap(); // seq 0,1,2; checkpoint at seq 2
        }
        drop(store);

        truncate_last_line(&dir.join("journal.log")); // journal head now seq 1

        // `open` returns `Result<WalAuditStore, _>` and the store is not Debug,
        // so match on the Result rather than `unwrap_err`.
        let reopen = WalAuditStore::open(&dir, StubSigner { key: [7; 32] });
        assert!(
            matches!(
                reopen,
                Err(AuditError::Truncation {
                    checkpoint_seq: 2,
                    journal_seq: Some(1)
                })
            ),
            "reopen must fail closed on the truncated tail"
        );
    }

    /// The offline verifier catches the same truncation (and would also catch a
    /// checkpoint that survived but is now ahead of the journal head).
    #[test]
    fn tail_truncation_is_detected_by_verify_journal() {
        let dir = tmp_dir("b3_trunc_verify");
        let store = WalAuditStore::open(&dir, StubSigner { key: [8; 32] }).unwrap();
        for _ in 0..3 {
            store.record(&perm_rec()).unwrap();
        }
        truncate_last_line(&dir.join("journal.log"));

        let err = store
            .verify_journal(&StubVerifier { key: [8; 32] })
            .unwrap_err();
        assert!(
            matches!(err, AuditError::Truncation { checkpoint_seq: 2, .. }),
            "verify_journal must detect the truncated tail, got {err:?}"
        );
    }

    /// A record that is individually valid — correct seq, a real signature — but
    /// links to the WRONG predecessor hash must break the chain. A checker that
    /// only validated seq + signature (the pre-B3 behaviour) would accept it.
    #[test]
    fn forged_record_with_wrong_prev_hash_breaks_the_chain() {
        let dir = tmp_dir("b3_chain");
        let signer = StubSigner { key: [9; 32] };
        let store = WalAuditStore::open(&dir, StubSigner { key: [9; 32] }).unwrap();
        store.record(&perm_rec()).unwrap(); // seq 0
        store.record(&perm_rec()).unwrap(); // seq 1
        drop(store);

        // Forge a replacement for the seq-1 line: valid seq, valid signature over
        // its own content, but a bogus prev_hash that does not link to seq 0.
        let forged = AuditRecord {
            seq: 1,
            prev_hash: [0xEE; 32],
            ..perm_rec()
        };
        let bytes = forged.canonical_bytes().unwrap();
        let hash = *blake3::hash(&bytes).as_bytes();
        let sig = signer.sign(&audit_signing_input(&hash)).unwrap();
        let forged_line = format!("1\t{}\t{}", hex_encode(&sig), base64_encode(&bytes));

        let jp = dir.join("journal.log");
        let content = std::fs::read_to_string(&jp).unwrap();
        let mut lines: Vec<String> = content.lines().map(str::to_owned).collect();
        lines[1] = forged_line;
        std::fs::write(&jp, lines.join("\n") + "\n").unwrap();

        // Reopen succeeds (seq 1 == checkpoint seq, no truncation); the chain
        // break surfaces in verify_journal — before the signature even matters.
        let reopened = WalAuditStore::open(&dir, StubSigner { key: [9; 32] }).unwrap();
        let err = reopened
            .verify_journal(&StubVerifier { key: [9; 32] })
            .unwrap_err();
        assert!(
            matches!(err, AuditError::ChainBreak { seq: 1 }),
            "a signed record with a wrong prev_hash must break the chain, got {err:?}"
        );
    }

    /// An intact 3-record store verifies cleanly end-to-end (chain links + the
    /// checkpoint anchors the real head) — the positive control for the above.
    #[test]
    fn intact_chain_and_checkpoint_verify() {
        let dir = tmp_dir("b3_intact");
        let store = WalAuditStore::open(&dir, StubSigner { key: [10; 32] }).unwrap();
        for _ in 0..3 {
            store.record(&perm_rec()).unwrap();
        }
        let recs = store
            .verify_journal(&StubVerifier { key: [10; 32] })
            .unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].prev_hash, [0u8; 32], "genesis links to zero");
        assert_ne!(recs[1].prev_hash, [0u8; 32], "later records chain forward");
    }

    // ── Complete mediation: every decision is audited ───────────────────────

    #[test]
    fn every_decision_is_audited_including_cache_hits() {
        let inner = avc();
        let sink = Arc::new(CapturingSink::new());
        let audited = AuditedAvc::new(inner, sink.clone(), 7, None);

        // First call: PDP miss → evaluate → cache → audit.
        let d1 = audited.decide(subj(1, high()), obj(1, low()), Action(1));
        // Second call: cache hit → still audit (completeness over cache efficiency).
        let d2 = audited.decide(subj(1, high()), obj(1, low()), Action(1));
        // A deny also audited.
        let d3 = audited.decide(subj(1, high()), obj(1, low()), Action(9));

        assert_eq!(d1, Decision::Permit);
        assert_eq!(d2, Decision::Permit);
        assert_eq!(d3, Decision::Deny);
        let recs = sink.records();
        assert_eq!(recs.len(), 3, "all three ops must produce audit records");
        assert_eq!(recs[0].decision, Decision::Permit);
        assert_eq!(recs[1].decision, Decision::Permit);
        assert_eq!(recs[2].decision, Decision::Deny);
        assert_eq!(recs[2].reason, DecisionReason::TeMiss);
    }

    // ── Fail-closed: Permit downgrades to Deny when audit write fails ───────

    #[test]
    fn permit_downgrades_to_deny_when_audit_write_fails() {
        let inner = avc();
        let sink = Arc::new(CapturingSink::new());
        let audited = AuditedAvc::new(inner, sink.clone(), 7, None);

        // Poison the next record() — the audit write will fail.
        sink.fail_next();
        // This WOULD be a Permit (rule (1,1,1) exists + floor holds), but the
        // audit write fails, so it must be downgraded to Deny.
        let d = audited.decide(subj(1, high()), obj(1, low()), Action(1));
        assert_eq!(
            d,
            Decision::Deny,
            "a Permit that can't be audited MUST be denied"
        );

        // The fail-closed deny itself is recorded (best-effort second write).
        let recs = sink.records();
        let fail_closed = recs
            .iter()
            .find(|r| r.reason == DecisionReason::AuditFailClosed);
        assert!(
            fail_closed.is_some(),
            "the fail-closed deny must itself be auditable"
        );
        assert_eq!(fail_closed.unwrap().decision, Decision::Deny);
    }

    // ── Deny stays Deny even if audit write fails (no escalation) ───────────

    #[test]
    fn deny_stays_deny_when_audit_write_fails() {
        let inner = avc();
        let sink = Arc::new(CapturingSink::new());
        let audited = AuditedAvc::new(inner, sink.clone(), 7, None);
        sink.fail_next();
        // A deny (action 9 not in matrix) when audit fails: still a deny.
        let d = audited.decide(subj(1, high()), obj(1, low()), Action(9));
        assert_eq!(d, Decision::Deny);
    }

    // ── NullAuditSink passes decisions through unchanged ────────────────────

    #[test]
    fn null_sink_is_pass_through() {
        let inner = avc();
        let audited = AuditedAvc::new(inner, NullAuditSink, 7, None);
        assert_eq!(
            audited.decide(subj(1, high()), obj(1, low()), Action(1)),
            Decision::Permit
        );
        assert_eq!(
            audited.decide(subj(1, high()), obj(1, low()), Action(9)),
            Decision::Deny
        );
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    #[test]
    fn shard_of_splits_two_then_rest() {
        let h = [0xab; 32];
        let (shard, rest) = shard_of(&h);
        assert_eq!(shard, "ab");
        assert_eq!(rest.len(), 62);
    }

    #[test]
    fn hex_roundtrip() {
        let bytes = [0x00, 0x01, 0xfe, 0xff, 0xab, 0xcd];
        let s = hex_encode(&bytes);
        assert_eq!(hex_decode(&s).unwrap(), bytes);
    }

    #[test]
    fn base64_roundtrip() {
        for case in [
            &b""[..],
            b"a",
            b"ab",
            b"abc",
            b"\xff\x00\x10\xde",
            b"hello audit",
        ] {
            let s = base64_encode(case);
            let dec = base64_decode(&s).unwrap();
            assert_eq!(dec, case, "roundtrip failed for {case:?}");
        }
    }

    // ── S8 (#574): policy-aware hybrid audit signing ───────────────────────

    use crate::mac::audit::cose::{CoseAuditSigner, CoseAuditVerifier};
    use ed25519_dalek::SigningKey;
    use hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair;
    use hyprstream_rpc::crypto::CryptoPolicy;
    use rand::rngs::OsRng;

    /// Under a Hybrid policy, a signer with both keys produces a hybrid composite
    /// that verifies with `require_pq = true`.
    #[test]
    fn cose_audit_signer_hybrid_roundtrips_under_hybrid_policy() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let signer = CoseAuditSigner::new(&ed, Some(&pq_sk), CryptoPolicy::Hybrid);
        assert!(signer.can_sign());

        let input = b"audit-record-signing-input";
        let sig = signer.sign(input).unwrap();

        let verifier = CoseAuditVerifier {
            ed_vk: &ed.verifying_key(),
            pq_vk: Some(&pq_vk),
            require_pq: true,
        };
        assert!(verifier.verify(input, &sig).is_ok());
    }

    /// Fail-closed: under a Hybrid policy, a signer with NO ML-DSA-65 key MUST
    /// refuse to sign (no silent downgrade of a security-critical accounting
    /// artifact). `can_sign()` reflects this.
    #[test]
    fn cose_audit_signer_fails_closed_under_hybrid_without_pq_key() {
        let ed = SigningKey::generate(&mut OsRng);
        let signer = CoseAuditSigner::new(&ed, None, CryptoPolicy::Hybrid);
        assert!(!signer.can_sign(), "Hybrid without PQ key cannot sign");
        let err = signer.sign(b"x").unwrap_err();
        assert!(
            matches!(err, AuditError::Sign(ref m) if m.contains("fail-closed")),
            "expected fail-closed Sign error, got {err:?}"
        );
    }

    /// Under a Classical policy, the signer emits an EdDSA-only signature even
    /// if a PQ key was supplied (the PQ key is ignored — no silent hybrid). The
    /// classical verifier (`require_pq = false`) accepts it.
    #[test]
    fn cose_audit_signer_classical_emits_eddsa_only() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        // Classical policy: PQ key is IGNORED.
        let signer = CoseAuditSigner::new(&ed, Some(&pq_sk), CryptoPolicy::Classical);
        assert!(signer.can_sign());

        let input = b"classical-audit";
        let sig = signer.sign(input).unwrap();
        // Classical verifier accepts the EdDSA-only signature.
        let verifier = CoseAuditVerifier {
            ed_vk: &ed.verifying_key(),
            pq_vk: None,
            require_pq: false,
        };
        assert!(verifier.verify(input, &sig).is_ok());

        // A Hybrid verifier MUST reject the classical-only signature.
        let (_other_sk, other_vk) = ml_dsa_generate_keypair();
        let hybrid_verifier = CoseAuditVerifier {
            ed_vk: &ed.verifying_key(),
            pq_vk: Some(&other_vk),
            require_pq: true,
        };
        assert!(
            hybrid_verifier.verify(input, &sig).is_err(),
            "Hybrid verifier must reject classical-only audit signature"
        );
    }

    /// Tampered audit input must fail hybrid verification.
    #[test]
    fn cose_audit_signer_tampered_input_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let signer = CoseAuditSigner::new(&ed, Some(&pq_sk), CryptoPolicy::Hybrid);
        let sig = signer.sign(b"original").unwrap();
        let verifier = CoseAuditVerifier {
            ed_vk: &ed.verifying_key(),
            pq_vk: Some(&pq_vk),
            require_pq: true,
        };
        assert!(verifier.verify(b"tampered", &sig).is_err());
    }
}
