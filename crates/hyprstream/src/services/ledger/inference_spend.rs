//! Inference completion → single-phase ledger spend (issue #1264, epic #1064).
//!
//! At generation completion the inference service posts a **single-phase spend**
//! to the cell ledger: it debits the verified caller's `Available` account for
//! `prompt_tokens + generated_tokens` of the model's token unit, crediting the
//! cell-owned usage account, bound to a `transfer_id`. This is the accounting
//! *emit* — the durable accounting event the inference path previously never
//! produced, so inference had no quota-burn accounting.
//!
//! ## Native spend API
//!
//! The emit uses [`LedgerHandle::debit`] — the ledger crate's native single-phase
//! spend ([`hyprstream_ledger::LedgerBackend::debit`], overdraft-checked on the
//! debit side). It does **not** depend on `ResourceIntent` (#1065, not merged);
//! [`SpendInput`] is the minimal, completion-shaped record this seam needs and is
//! left for #1065 to generalize.
//!
//! ## `transfer_id` (#985)
//!
//! `Transfer.id` is **client-supplied** — it *is* the ledger's idempotency key
//! (the backend never mints one; the [`Outcome`](hyprstream_ledger::Outcome) for a
//! replayed id is returned verbatim). The admission path
//! ([`super::enforcer::LocalEnforcer::admit`]) mints it *at admission* via
//! `mint_transfer_id(grant_cid, nonce)` so the #985 spend-authorization can bind
//! to it. There is **no admission on the inference completion path yet** — the
//! two-phase reserve/post enforcer is gated off and not wired into inference
//! (see [`super`]'s module docs). So this emit mints the transfer id **at
//! completion** ([`completion_transfer_id`]), under a distinct domain tag so it
//! cannot collide with the admission path's id space. Cross-replay idempotency
//! (a retried request reproducing the same id) is a documented follow-up: it
//! requires threading the envelope `request_nonce` to completion; today the
//! per-stream `stream_id` is the entropy, so a retried generation mints a fresh
//! id and posts a fresh (idempotent-in-itself) spend.
//!
//! ## Fail-safe (#1264)
//!
//! A ledger/accounting failure MUST NOT break a generation that already produced
//! output. [`InferenceSpendEmitter::post_generation_spend`] therefore **never
//! returns a `Result`** the caller propagates as a stream failure: every outcome
//! maps to an observable [`SpendResult`] the caller only *records* (log / metric).
//! A declined spend (overdraft / unknown account) and a failed spend (actor down /
//! encoding) are both surfaced as a **content-free signal** and then the
//! generation proceeds — there is never a silent drop. The debit is best-effort
//! accounting; the generation already completed.
//!
//! "Per the local enforcer's policy": the completion single-phase spend has no
//! admission, so the in-memory [`super::CreditGate`] optimistic hold is not on
//! this path. The floor that declines an overdraft here is the ledger engine's
//! own overdraft check (`InsufficientBalance`) — the durable floor INV-1 upholds.

use hyprstream_ledger::{
    AccountId, Did, LedgerError, Outcome, Purpose, Transfer, TransferId, TransferResult, UnitId,
};

use super::handle::LedgerHandle;

/// The observable result of a completion spend. Never an error the caller raises
/// — the caller records it and continues (fail-safe, #1264).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpendResult {
    /// The spend posted (or a replay returned the same applied outcome).
    Posted {
        /// The transfer id the spend is bound to (the idempotency key).
        transfer_id: TransferId,
        /// Tokens debited (prompt + generated).
        amount: u128,
    },
    /// No spend was attempted: zero tokens to account, or not configured.
    Skipped(&'static str),
    /// The ledger floor declined the spend (overdraft / unknown account).
    /// Recorded; the generation is unaffected.
    Declined {
        /// The transfer id the (failed) spend was bound to.
        transfer_id: TransferId,
        /// The deterministic reason (content-free category).
        reason: SpendDecline,
    },
    /// The spend could not be posted (actor unavailable / encoding / misconfig).
    /// Fail-safe: the caller signals and continues.
    Failed {
        /// The transfer id, if one was minted before the failure.
        transfer_id: Option<TransferId>,
        /// The content-free reason category.
        reason: SpendFailure,
    },
}

/// The deterministic, content-free category of a *declined* completion spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpendDecline {
    /// The caller's account cannot cover the token amount (overdraft floor).
    InsufficientBalance,
    /// The caller's account (or the cell usage account) does not exist.
    UnknownAccount,
}

/// The content-free category of a *failed* completion spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpendFailure {
    /// The unit on the spend does not match an account's unit (misconfiguration).
    UnitMismatch,
    /// The holder and cell usage accounts resolve to the same id (misconfiguration).
    AccountsMustDiffer,
    /// The ledger actor is unavailable or reported an internal/encoding error.
    Internal,
}

impl SpendResult {
    /// A short, content-free label for logging/metrics (no subject, no prompt).
    pub fn category(&self) -> &'static str {
        match self {
            SpendResult::Posted { .. } => "posted",
            SpendResult::Skipped(_) => "skipped",
            SpendResult::Declined { .. } => "declined",
            SpendResult::Failed { .. } => "failed",
        }
    }
}

/// The minimal, completion-shaped record the emit consumes. Intentionally narrow:
/// the general `ResourceIntent` (#1065) is left to that issue.
#[derive(Debug, Clone)]
pub struct SpendInput<'a> {
    /// The verified caller's self-certifying pairwise DID (the account owner).
    pub owner_did: &'a str,
    /// Per-stream entropy → the deterministic transfer id (idempotency key).
    pub stream_id: &'a str,
    /// Prompt (prefill) tokens — from tokenization.
    pub prompt_tokens: u64,
    /// Generated (decode) tokens — from the stream completion count.
    pub generated_tokens: u64,
}

/// The single-phase completion spend emitter, bound to one cell ledger and one
/// inference token [`UnitId`] (the model-scoped token class; its `issuer` is the
/// cell identity). Cheap to clone (an [`LedgerHandle`] is an `mpsc::Sender`).
#[derive(Clone)]
pub struct InferenceSpendEmitter {
    handle: LedgerHandle,
    cell: Did,
    unit: UnitId,
}

impl std::fmt::Debug for InferenceSpendEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceSpendEmitter")
            .field("cell", &self.cell)
            .field("unit", &self.unit)
            .finish_non_exhaustive()
    }
}

impl InferenceSpendEmitter {
    /// Construct against a cell ledger handle, the cell identity (unit issuer),
    /// and the inference token unit (e.g. `resource_class = "inference.tokens"`).
    pub fn new(handle: LedgerHandle, cell: Did, unit: UnitId) -> Self {
        InferenceSpendEmitter { handle, cell, unit }
    }

    /// The cell ledger's service identity.
    pub fn cell(&self) -> &Did {
        &self.cell
    }

    /// The inference token unit spends are denominated in.
    pub fn unit(&self) -> &UnitId {
        &self.unit
    }

    /// Post the single-phase completion spend. Never errors: every outcome is an
    /// observable [`SpendResult`] (fail-safe, #1264).
    ///
    /// The caller's `Available` account is debited `prompt_tokens +
    /// generated_tokens`; the cell usage account is credited. The spend is bound
    /// to a completion-minted [`TransferId`] (see the module docs on #985).
    pub async fn post_generation_spend(&self, input: SpendInput<'_>) -> SpendResult {
        // Token counts are `u64`; the ledger denominates amounts in `u128` minor
        // units (token quanta). Widening is lossless; saturating add guards the
        // (unreachable) sum-overflow on a single generation.
        let amount: u128 =
            (input.prompt_tokens as u128).saturating_add(input.generated_tokens as u128);
        if amount == 0 {
            return SpendResult::Skipped("zero tokens");
        }

        let owner = Did(input.owner_did.to_owned());
        let debit = match holder_account(&self.cell, &owner, &self.unit) {
            Ok(id) => id,
            Err(_) => {
                // `AccountId::derive` only fails on an (unreachable) CBOR encode.
                return SpendResult::Failed {
                    transfer_id: None,
                    reason: SpendFailure::Internal,
                };
            }
        };
        let credit = match usage_account(&self.cell, &self.unit) {
            Ok(id) => id,
            Err(_) => {
                return SpendResult::Failed {
                    transfer_id: None,
                    reason: SpendFailure::Internal,
                };
            }
        };

        let transfer_id = completion_transfer_id(&self.cell, &owner, &self.unit, input.stream_id);
        let transfer = build_transfer(
            transfer_id,
            debit,
            credit,
            self.unit.clone(),
            amount,
            &owner,
            input.stream_id,
        );
        let outcome = self.handle.debit(transfer).await;
        classify(transfer_id, amount, outcome)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Pure helpers (no I/O) — unit-testable in isolation.
// ────────────────────────────────────────────────────────────────────────────

/// Derive the verified caller's spendable (`Available`) account id for `unit`.
fn holder_account(cell: &Did, owner: &Did, unit: &UnitId) -> Result<AccountId, LedgerError> {
    AccountId::derive(cell, owner, unit, &Purpose::Available)
}

/// Derive the cell-owned usage (credit) account id — the sink a spend credits.
fn usage_account(cell: &Did, unit: &UnitId) -> Result<AccountId, LedgerError> {
    AccountId::derive(cell, cell, unit, &Purpose::Available)
}

/// Deterministic completion transfer id: `blake3_128` over a domain-separated
/// encoding of `(cell, owner, unit, stream_id)`. Distinct domain tag from the
/// admission path's `mint_transfer_id` ⇒ the two id spaces never collide (#985).
fn completion_transfer_id(cell: &Did, owner: &Did, unit: &UnitId, stream_id: &str) -> TransferId {
    let mut h = blake3::Hasher::new();
    h.update(b"hs-ledger-inference-xfer-v1");
    h.update(cell.as_str().as_bytes());
    h.update(owner.as_str().as_bytes());
    h.update(unit.issuer.as_str().as_bytes());
    h.update(unit.resource_class.as_bytes());
    h.update(stream_id.as_bytes());
    let mut out = [0u8; 16];
    h.finalize_xof().fill(&mut out);
    TransferId(u128::from_be_bytes(out))
}

/// Build the single-phase spend [`Transfer`] for one completion. `user_data` is a
/// content address over `(owner, stream_id)` — opaque receipt correlation that
/// carries no prompt material.
fn build_transfer(
    id: TransferId,
    debit_account: AccountId,
    credit_account: AccountId,
    unit: UnitId,
    amount: u128,
    owner: &Did,
    stream_id: &str,
) -> Transfer {
    let mut ud = blake3::Hasher::new();
    ud.update(b"hs-ledger-inference-userdata-v1");
    ud.update(owner.as_str().as_bytes());
    ud.update(stream_id.as_bytes());
    let mut user_data = [0u8; 32];
    ud.finalize_xof().fill(&mut user_data);
    Transfer {
        id,
        debit_account,
        credit_account,
        unit,
        amount,
        // No grant on the completion path (no admission/authorization yet, #985
        // follow-up); the spend is an internal accounting emit.
        grant_cid: None,
        user_data,
    }
}

/// Map a backend [`Outcome`] to the fail-safe [`SpendResult`] the caller records.
fn classify(transfer_id: TransferId, amount: u128, outcome: Outcome) -> SpendResult {
    match outcome.result {
        Ok(TransferResult::Applied { posted, .. }) => {
            // Single-phase debit posts the full amount; `posted` is authoritative.
            SpendResult::Posted {
                transfer_id,
                amount: posted,
            }
        }
        // A debit never yields the other `Ok` variants, but treat any success as
        // posted rather than drop it (fail-loud, not fail-silent).
        Ok(_) => SpendResult::Posted {
            transfer_id,
            amount,
        },
        Err(LedgerError::InsufficientBalance { .. }) => SpendResult::Declined {
            transfer_id,
            reason: SpendDecline::InsufficientBalance,
        },
        Err(LedgerError::UnknownAccount(_)) => SpendResult::Declined {
            transfer_id,
            reason: SpendDecline::UnknownAccount,
        },
        Err(LedgerError::UnitMismatch { .. }) => SpendResult::Failed {
            transfer_id: Some(transfer_id),
            reason: SpendFailure::UnitMismatch,
        },
        Err(LedgerError::AccountsMustDiffer(_)) => SpendResult::Failed {
            transfer_id: Some(transfer_id),
            reason: SpendFailure::AccountsMustDiffer,
        },
        Err(LedgerError::ZeroAmount) => SpendResult::Skipped("zero amount at floor"),
        // Internal (actor down / encoding) and anything unexpected.
        Err(_) => SpendResult::Failed {
            transfer_id: Some(transfer_id),
            reason: SpendFailure::Internal,
        },
    }
}

/// Record a completion spend outcome as a content-free signal (log only). Called
/// by the inference completion path; never raises. "Content-free": the signal
/// carries no subject DID and no prompt material — only `stream_id`, the unit,
/// and the outcome category (#1264: never silently drop a spend).
pub fn observe_spend_result(result: &SpendResult, stream_id: &str, unit: &UnitId) {
    match result {
        SpendResult::Posted { amount, .. } => {
            tracing::debug!(
                stream_id,
                unit = unit.resource_class.as_str(),
                amount,
                category = result.category(),
                "ledger: posted inference token spend",
            );
        }
        SpendResult::Skipped(reason) => {
            tracing::debug!(
                stream_id,
                unit = unit.resource_class.as_str(),
                reason,
                category = result.category(),
                "ledger: inference spend skipped",
            );
        }
        SpendResult::Declined { reason, .. } => {
            tracing::warn!(
                stream_id,
                unit = unit.resource_class.as_str(),
                decline = ?reason,
                category = result.category(),
                "ledger: inference spend declined by floor (generation unaffected)",
            );
        }
        SpendResult::Failed { reason, .. } => {
            tracing::error!(
                stream_id,
                unit = unit.resource_class.as_str(),
                failure = ?reason,
                category = result.category(),
                "ledger: inference spend failed (fail-safe; generation unaffected)",
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::services::ledger::CoseCheckpointSigner;
    use std::sync::Arc;
    use hyprstream_ledger::{AccountSpec, IssueTransfer, LedgerBackend, MemLedger};

    fn unit() -> UnitId {
        UnitId {
            issuer: Did("did:web:cell.test".to_owned()),
            resource_class: "inference.tokens".to_owned(),
        }
    }

    fn did(s: &str) -> Did {
        Did(s.to_owned())
    }

    /// Open the issuer-liability + cell-usage accounts, plus one `Available`
    /// account per named holder, and issue `credit` to each holder. Returns the
    /// emitter and the spawned handle (for balance inspection).
    async fn fixture(holders: &[(&str, u128)]) -> (InferenceSpendEmitter, LedgerHandle) {
        let cell = did("did:web:cell.test");
        let mut backend = MemLedger::new(cell.clone());
        let issuer_liab =
            AccountId::derive(&cell, &unit().issuer, &unit(), &Purpose::IssuerLiability).unwrap();
        backend
            .open_account(AccountSpec::new(
                cell.clone(),
                unit().issuer.clone(),
                unit(),
                Purpose::IssuerLiability,
            ))
            .unwrap();
        backend
            .open_account(AccountSpec::new(
                cell.clone(),
                cell.clone(),
                unit(),
                Purpose::Available,
            ))
            .unwrap();

        for (idx, (h, credit)) in holders.iter().enumerate() {
            let holder = did(h);
            let acct = holder_account(&cell, &holder, &unit()).unwrap();
            backend
                .open_account(AccountSpec::new(
                    cell.clone(),
                    holder.clone(),
                    unit(),
                    Purpose::Available,
                ))
                .unwrap();
            backend
                .credit(IssueTransfer {
                    // Monotonic, holder-distinct idempotency id per issuance.
                    id: TransferId(1_000 + idx as u128),
                    issuer_liability: issuer_liab,
                    destination: acct,
                    unit: unit(),
                    amount: *credit,
                    grant_cid: None,
                    user_data: [0u8; 32],
                })
                .result
                .unwrap();
        }

        let signer = Arc::new(CoseCheckpointSigner::classical(
            cell.clone(),
            ed25519_dalek::SigningKey::from_bytes(&[1u8; 32]),
        ));
        let handle = LedgerHandle::spawn(Box::new(backend), signer);
        let emitter = InferenceSpendEmitter::new(handle.clone(), cell, unit());
        (emitter, handle)
    }

    async fn available(handle: &LedgerHandle, owner: &str) -> u128 {
        let cell = did("did:web:cell.test");
        let acct = holder_account(&cell, &did(owner), &unit()).unwrap();
        handle.balance(acct).await.unwrap().available
    }

    #[tokio::test]
    async fn two_subjects_post_to_own_accounts_only() {
        // Each holder starts with 1000 tokens; each spends 300. Only that
        // holder's account is debited; the other is untouched.
        let (emitter, handle) = fixture(&[("did:web:alice", 1000), ("did:web:bob", 1000)]).await;

        let a = emitter
            .post_generation_spend(SpendInput {
                owner_did: "did:web:alice",
                stream_id: "stream-a",
                prompt_tokens: 100,
                generated_tokens: 200,
            })
            .await;
        let b = emitter
            .post_generation_spend(SpendInput {
                owner_did: "did:web:bob",
                stream_id: "stream-b",
                prompt_tokens: 50,
                generated_tokens: 250,
            })
            .await;

        assert!(
            matches!(a, SpendResult::Posted { amount: 300, .. }),
            "{a:?}"
        );
        assert!(
            matches!(b, SpendResult::Posted { amount: 300, .. }),
            "{b:?}"
        );
        // Each debited exactly its own 300; the other untouched.
        assert_eq!(available(&handle, "did:web:alice").await, 700);
        assert_eq!(available(&handle, "did:web:bob").await, 700);
    }

    #[tokio::test]
    async fn overdraft_declined_per_local_enforcer_floor() {
        // The completion spend has no admission/CreditGate hold; the floor that
        // declines an overdraft is the ledger engine's own check
        // (`InsufficientBalance`) — the durable floor INV-1 upholds. Alice has
        // 10; she spends 100 → declined, balance untouched.
        let (emitter, handle) = fixture(&[("did:web:alice", 10)]).await;

        let res = emitter
            .post_generation_spend(SpendInput {
                owner_did: "did:web:alice",
                stream_id: "stream-over",
                prompt_tokens: 40,
                generated_tokens: 60,
            })
            .await;

        assert_eq!(
            res,
            SpendResult::Declined {
                transfer_id: completion_transfer_id(
                    &did("did:web:cell.test"),
                    &did("did:web:alice"),
                    &unit(),
                    "stream-over",
                ),
                reason: SpendDecline::InsufficientBalance,
            }
        );
        // No partial debit on a decline.
        assert_eq!(available(&handle, "did:web:alice").await, 10);
    }

    #[tokio::test]
    async fn missing_subject_account_fails_closed_no_spend_no_leak() {
        // Carol calls but has no account (no entitlement issued). The spend must
        // fail closed: no debit lands anywhere, and the outcome is a signal, not
        // a propagated error.
        let (emitter, handle) = fixture(&[("did:web:alice", 1000)]).await;

        let res = emitter
            .post_generation_spend(SpendInput {
                owner_did: "did:web:carol",
                stream_id: "stream-carol",
                prompt_tokens: 10,
                generated_tokens: 20,
            })
            .await;

        assert_eq!(
            res,
            SpendResult::Declined {
                transfer_id: completion_transfer_id(
                    &did("did:web:cell.test"),
                    &did("did:web:carol"),
                    &unit(),
                    "stream-carol",
                ),
                reason: SpendDecline::UnknownAccount,
            }
        );
        // Alice (the only funded account) is untouched — no leak to/from her.
        assert_eq!(available(&handle, "did:web:alice").await, 1000);
    }

    #[tokio::test]
    async fn zero_tokens_is_a_skip_not_a_spend() {
        let (emitter, handle) = fixture(&[("did:web:alice", 1000)]).await;
        let res = emitter
            .post_generation_spend(SpendInput {
                owner_did: "did:web:alice",
                stream_id: "stream-empty",
                prompt_tokens: 0,
                generated_tokens: 0,
            })
            .await;
        assert!(matches!(res, SpendResult::Skipped(_)));
        assert_eq!(available(&handle, "did:web:alice").await, 1000);
    }

    #[tokio::test]
    async fn same_stream_replays_to_same_outcome() {
        // The transfer id is deterministic in the stream id; a second post for
        // the same stream is a replay → the ledger returns the original applied
        // outcome verbatim (idempotency), and the holder is debited only once.
        let (emitter, handle) = fixture(&[("did:web:alice", 1000)]).await;
        let mk = || SpendInput {
            owner_did: "did:web:alice",
            stream_id: "stream-once",
            prompt_tokens: 100,
            generated_tokens: 100,
        };
        let first = emitter.post_generation_spend(mk()).await;
        let second = emitter.post_generation_spend(mk()).await;
        assert_eq!(first, second, "replay must yield the same SpendResult");
        assert!(matches!(first, SpendResult::Posted { amount: 200, .. }));
        assert_eq!(available(&handle, "did:web:alice").await, 800);
    }

    #[test]
    fn transfer_id_is_deterministic_and_stream_distinct() {
        let cell = did("did:web:cell.test");
        let alice = did("did:web:alice");
        let a = completion_transfer_id(&cell, &alice, &unit(), "s1");
        let a2 = completion_transfer_id(&cell, &alice, &unit(), "s1");
        let b = completion_transfer_id(&cell, &alice, &unit(), "s2");
        assert_eq!(a, a2, "same inputs ⇒ same id");
        assert_ne!(a, b, "different stream ⇒ different id");
        // Distinct from a different subject even on the same stream.
        let bob = did("did:web:bob");
        let c = completion_transfer_id(&cell, &bob, &unit(), "s1");
        assert_ne!(a, c, "different subject ⇒ different id");
    }

    #[test]
    fn build_transfer_carries_amount_and_no_grant() {
        let cell = did("did:web:cell.test");
        let owner = did("did:web:alice");
        let id = TransferId(7);
        let debit = holder_account(&cell, &owner, &unit()).unwrap();
        let credit = usage_account(&cell, &unit()).unwrap();
        let t = build_transfer(id, debit, credit, unit(), 250, &owner, "stream-x");
        assert_eq!(t.id, id);
        assert_eq!(t.amount, 250);
        assert!(
            t.grant_cid.is_none(),
            "completion spend carries no grant (#985)"
        );
        assert_ne!(
            t.user_data, [0u8; 32],
            "user_data is a real correlation hash"
        );
    }
}
