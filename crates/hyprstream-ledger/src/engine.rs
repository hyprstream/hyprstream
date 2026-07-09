//! The pure, backend-agnostic transfer state machine (plan §2a step 2, §2b).
//!
//! `stage` is a **total, default-deny** function of `(current state, op)` →
//! `(result, deltas)`. It performs no I/O and no mutation: it reads the current
//! state through [`StateView`] and returns the deltas a backend must apply
//! atomically inside its serialized commit. RocksLedger and MemLedger share this
//! module verbatim, which is what makes "same op sequence ⇒ same outcomes"
//! true by construction (backend-equivalence).
//!
//! There are no TOCTOU races here: within one cell ledger there is a single
//! writer, so every interleaving question reduces to an *ordering* question, and
//! the ordering is fixed by the sequence in which ops reach `stage`.

use crate::errors::LedgerError;
use crate::types::{
    Account, AccountId, AccountSpec, IssueTransfer, PendingReservation, PendingState, Purpose,
    Transfer, TransferId, TransferResult,
};

/// Minimum reservation timeout (plan §2b.6).
pub const MIN_TIMEOUT_S: u32 = 1;
/// Maximum reservation timeout — day-plus holds are tranches, not reservations.
pub const MAX_TIMEOUT_S: u32 = 24 * 60 * 60;

/// Read-only view of ledger state the engine needs. Backends implement this over
/// their own storage (a `BTreeMap` for MemLedger, RocksDB point-reads for
/// RocksLedger).
pub trait StateView {
    /// Look up an account by id.
    fn account(&self, id: AccountId) -> Option<&Account>;
    /// Look up a pending reservation by its phase-1 transfer id.
    fn pending(&self, id: TransferId) -> Option<&PendingReservation>;
    /// The ledger's logical commit clock (seconds). Deadlines are judged against
    /// this at apply time (plan §2b.5) — never a caller clock.
    fn now(&self) -> u64;
}

/// An op the engine can stage. The idempotency key (where present) is the
/// carried `TransferId`; `stage` never dedups — that is the commit loop's job
/// (it checks the outcome index before calling `stage`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Op {
    /// Create an account (boxed to keep the enum small). Idempotent by derived
    /// id; the backend's fast path returns an existing account without a journal
    /// entry, so this variant is only committed on real creation.
    OpenAccount(Box<AccountSpec>),
    /// Single-phase issuance (supply growth).
    Credit(IssueTransfer),
    /// Single-phase spend.
    Debit(Transfer),
    /// Phase-1 reservation.
    Reserve {
        /// The reserve transfer.
        transfer: Transfer,
        /// Timeout in seconds (validated against `[MIN,MAX]_TIMEOUT_S`).
        timeout_s: u32,
    },
    /// Phase-2 post. `amount: None` = full; `Some(p)` = partial (remainder released).
    Post {
        /// This post's own idempotency id.
        id: TransferId,
        /// The phase-1 reservation being resolved.
        pending: TransferId,
        /// Optional partial amount.
        amount: Option<u128>,
    },
    /// Phase-2 void (release the hold).
    Void {
        /// This void's own idempotency id.
        id: TransferId,
        /// The phase-1 reservation being cancelled.
        pending: TransferId,
    },
    /// Sweep-driven expiry of a single reservation (plan §2b.4). Emitted by the
    /// backend's `tick`, not by callers; it has no external idempotency id.
    Expire {
        /// The reservation to expire.
        pending: TransferId,
    },
}

impl Op {
    /// The idempotency key for this op, if it carries one. `Expire` is
    /// internally generated and has none.
    pub fn idempotency_id(&self) -> Option<TransferId> {
        // `Credit`/`Debit` carry different payload types, so their identical
        // bodies cannot be merged into one arm.
        #[allow(clippy::match_same_arms)]
        match self {
            Op::OpenAccount(_) | Op::Expire { .. } => None,
            Op::Credit(t) => Some(t.id),
            Op::Debit(t) => Some(t.id),
            Op::Reserve { transfer, .. } => Some(transfer.id),
            Op::Post { id, .. } | Op::Void { id, .. } => Some(*id),
        }
    }
}

/// A single state mutation the commit loop must apply atomically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delta {
    /// Upsert an account row (keyed by `account.id`).
    Account(Account),
    /// Upsert a pending reservation row (keyed by `reservation.transfer.id`).
    Pending(PendingReservation),
}

/// The output of [`stage`]: the business result plus the deltas to apply. On an
/// error result `deltas` is usually empty, but not always — an on-touch expiry
/// (plan §2b.4) both fails the second phase *and* releases the hold, so a failed
/// post can carry the expiry delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staged {
    /// The result recorded under the op's idempotency id.
    pub result: Result<TransferResult, LedgerError>,
    /// State mutations to commit atomically with the journal entry.
    pub deltas: Vec<Delta>,
}

impl Staged {
    fn err(e: LedgerError) -> Self {
        Staged {
            result: Err(e),
            deltas: Vec::new(),
        }
    }
}

/// Stage an op against the current state. Pure: no mutation, no I/O.
pub fn stage(view: &dyn StateView, op: &Op) -> Staged {
    match op {
        Op::OpenAccount(spec) => stage_open(view, spec),
        Op::Credit(t) => stage_credit(view, t),
        Op::Debit(t) => stage_debit(view, t),
        Op::Reserve {
            transfer,
            timeout_s,
        } => stage_reserve(view, transfer, *timeout_s),
        Op::Post {
            pending, amount, ..
        } => stage_second_phase(view, *pending, Phase::Post(*amount)),
        Op::Void { pending, .. } => stage_second_phase(view, *pending, Phase::Void),
        Op::Expire { pending } => stage_second_phase(view, *pending, Phase::Expire),
    }
}

fn stage_open(view: &dyn StateView, spec: &AccountSpec) -> Staged {
    let id = match spec.account_id() {
        Ok(id) => id,
        Err(e) => return Staged::err(e),
    };
    if let Some(existing) = view.account(id) {
        if existing.unit != spec.unit {
            return Staged::err(LedgerError::AccountUnitConflict { id });
        }
        // Idempotent no-op (the backend fast-paths this before staging, so this
        // arm is defensive).
        return Staged {
            result: Ok(TransferResult::Opened),
            deltas: Vec::new(),
        };
    }
    let account = Account::new(id, spec.unit.clone(), spec.purpose.clone(), spec.flags);
    Staged {
        result: Ok(TransferResult::Opened),
        deltas: vec![Delta::Account(account)],
    }
}

fn stage_credit(view: &dyn StateView, t: &IssueTransfer) -> Staged {
    if t.amount == 0 {
        return Staged::err(LedgerError::ZeroAmount);
    }
    let (liability, dest) = match (
        view.account(t.issuer_liability),
        view.account(t.destination),
    ) {
        (Some(l), Some(d)) => (l, d),
        (None, _) => return Staged::err(LedgerError::UnknownAccount(t.issuer_liability)),
        (_, None) => return Staged::err(LedgerError::UnknownAccount(t.destination)),
    };
    // The debit side must be an issuer-liability account (INV-1).
    if t.issuer_liability == t.destination {
        return Staged::err(LedgerError::AccountsMustDiffer(t.issuer_liability));
    }
    if liability.purpose != Purpose::IssuerLiability {
        return Staged::err(LedgerError::NotIssuerLiability(t.issuer_liability));
    }
    if let Err(e) = check_unit(liability, &t.unit) {
        return Staged::err(e);
    }
    if let Err(e) = check_unit(dest, &t.unit) {
        return Staged::err(e);
    }
    let mut liability = liability.clone();
    let mut dest = dest.clone();
    liability.debits_posted = liability.debits_posted.saturating_add(t.amount);
    dest.credits_posted = dest.credits_posted.saturating_add(t.amount);
    Staged {
        result: Ok(TransferResult::Issued),
        deltas: vec![Delta::Account(liability), Delta::Account(dest)],
    }
}

fn stage_debit(view: &dyn StateView, t: &Transfer) -> Staged {
    let (debit, credit) = match resolve_pair(view, t.debit_account, t.credit_account, &t.unit) {
        Ok(p) => p,
        Err(e) => return Staged::err(e),
    };
    if t.amount == 0 {
        return Staged::err(LedgerError::ZeroAmount);
    }
    if debit.is_debit_constrained() && t.amount > debit.available() {
        return Staged::err(LedgerError::InsufficientBalance {
            account: debit.id,
            needed: t.amount,
            available: debit.available(),
        });
    }
    let mut debit = debit.clone();
    let mut credit = credit.clone();
    debit.debits_posted = debit.debits_posted.saturating_add(t.amount);
    credit.credits_posted = credit.credits_posted.saturating_add(t.amount);
    Staged {
        result: Ok(TransferResult::Applied {
            posted: t.amount,
            released: 0,
        }),
        deltas: vec![Delta::Account(debit), Delta::Account(credit)],
    }
}

fn stage_reserve(view: &dyn StateView, t: &Transfer, timeout_s: u32) -> Staged {
    if !(MIN_TIMEOUT_S..=MAX_TIMEOUT_S).contains(&timeout_s) {
        return Staged::err(LedgerError::TimeoutOutOfBounds(timeout_s));
    }
    let (debit, credit) = match resolve_pair(view, t.debit_account, t.credit_account, &t.unit) {
        Ok(p) => p,
        Err(e) => return Staged::err(e),
    };
    if t.amount == 0 {
        return Staged::err(LedgerError::ZeroAmount);
    }
    // Overdraft check counts pending (plan §2b.1): `available` already subtracts
    // `debits_pending`, so back-to-back reserves cannot over-commit.
    if debit.is_debit_constrained() && t.amount > debit.available() {
        return Staged::err(LedgerError::InsufficientBalance {
            account: debit.id,
            needed: t.amount,
            available: debit.available(),
        });
    }
    let mut debit = debit.clone();
    let mut credit = credit.clone();
    debit.debits_pending = debit.debits_pending.saturating_add(t.amount);
    credit.credits_pending = credit.credits_pending.saturating_add(t.amount);
    let reservation = PendingReservation {
        transfer: t.clone(),
        deadline: view.now().saturating_add(timeout_s as u64),
        state: PendingState::Pending,
    };
    Staged {
        result: Ok(TransferResult::Reserved),
        deltas: vec![
            Delta::Account(debit),
            Delta::Account(credit),
            Delta::Pending(reservation),
        ],
    }
}

/// The three second-phase resolutions, unified because they share the
/// terminal-state and on-touch-expiry rules.
enum Phase {
    Post(Option<u128>),
    Void,
    Expire,
}

fn stage_second_phase(view: &dyn StateView, pending_id: TransferId, phase: Phase) -> Staged {
    let reservation = match view.pending(pending_id) {
        Some(r) => r,
        None => return Staged::err(LedgerError::UnknownPendingTransfer(pending_id)),
    };

    // Already terminal ⇒ deterministic loser error (plan §2b.5). The first
    // terminal commit won; this op observes that state.
    match reservation.state {
        PendingState::Pending => {}
        PendingState::Posted => {
            return Staged::err(LedgerError::PendingTransferAlreadyPosted(pending_id))
        }
        PendingState::Voided => {
            return Staged::err(LedgerError::PendingTransferAlreadyVoided(pending_id))
        }
        PendingState::Expired => {
            return Staged::err(LedgerError::PendingTransferExpired(pending_id))
        }
    }

    let t = &reservation.transfer;
    let debit = match view.account(t.debit_account) {
        Some(a) => a,
        None => return Staged::err(LedgerError::UnknownAccount(t.debit_account)),
    };
    let credit = match view.account(t.credit_account) {
        Some(a) => a,
        None => return Staged::err(LedgerError::UnknownAccount(t.credit_account)),
    };

    // On-touch expiry (plan §2b.4): if the deadline has passed by the logical
    // commit clock, expire the reservation *in this same commit* and fail the
    // second phase (unless the op itself is the expiry). This is what makes the
    // expiry-vs-post race deterministic: whichever commits first wins.
    let expired_now = view.now() >= reservation.deadline;
    if matches!(phase, Phase::Expire) || expired_now {
        let released = release(debit, credit, t.amount);
        let mut res = reservation.clone();
        res.state = PendingState::Expired;
        let deltas = vec![released.0, released.1, Delta::Pending(res)];
        // A caller's post/void that lost to expiry gets the deterministic error;
        // the sweep-driven `Expire` op reports success.
        let result = if matches!(phase, Phase::Expire) {
            Ok(TransferResult::Expired)
        } else {
            Err(LedgerError::PendingTransferExpired(pending_id))
        };
        return Staged { result, deltas };
    }

    match phase {
        Phase::Void => {
            let (d_delta, c_delta) = release(debit, credit, t.amount);
            let mut res = reservation.clone();
            res.state = PendingState::Voided;
            Staged {
                result: Ok(TransferResult::Voided),
                deltas: vec![d_delta, c_delta, Delta::Pending(res)],
            }
        }
        Phase::Post(amount) => {
            let reserved = t.amount;
            let post_amt = amount.unwrap_or(reserved);
            if post_amt == 0 {
                return Staged::err(LedgerError::ZeroAmount);
            }
            if post_amt > reserved {
                return Staged::err(LedgerError::PostExceedsReservation {
                    requested: post_amt,
                    reserved,
                });
            }
            let released = reserved - post_amt;
            let mut debit = debit.clone();
            let mut credit = credit.clone();
            // The full reserved amount leaves pending; `post_amt` becomes posted,
            // the remainder is simply released.
            debit.debits_pending = debit.debits_pending.saturating_sub(reserved);
            credit.credits_pending = credit.credits_pending.saturating_sub(reserved);
            debit.debits_posted = debit.debits_posted.saturating_add(post_amt);
            credit.credits_posted = credit.credits_posted.saturating_add(post_amt);
            let mut res = reservation.clone();
            res.state = PendingState::Posted;
            Staged {
                result: Ok(TransferResult::Applied {
                    posted: post_amt,
                    released,
                }),
                deltas: vec![
                    Delta::Account(debit),
                    Delta::Account(credit),
                    Delta::Pending(res),
                ],
            }
        }
        Phase::Expire => unreachable_expire(),
    }
}

/// Release a full pending hold on both sides (void / expire).
fn release(debit: &Account, credit: &Account, amount: u128) -> (Delta, Delta) {
    let mut debit = debit.clone();
    let mut credit = credit.clone();
    debit.debits_pending = debit.debits_pending.saturating_sub(amount);
    credit.credits_pending = credit.credits_pending.saturating_sub(amount);
    (Delta::Account(debit), Delta::Account(credit))
}

/// The `Phase::Expire` arm inside the non-expired branch is unreachable: the
/// expiry path is handled before the `match phase` above. We return a fail-closed
/// internal error rather than panic (clippy forbids `unreachable!` here anyway).
fn unreachable_expire() -> Staged {
    Staged::err(LedgerError::Internal(
        "expire reached the non-expired second-phase arm".to_owned(),
    ))
}

fn resolve_pair<'a>(
    view: &'a dyn StateView,
    debit_id: AccountId,
    credit_id: AccountId,
    unit: &crate::types::UnitId,
) -> Result<(&'a Account, &'a Account), LedgerError> {
    if debit_id == credit_id {
        return Err(LedgerError::AccountsMustDiffer(debit_id));
    }
    let debit = view
        .account(debit_id)
        .ok_or(LedgerError::UnknownAccount(debit_id))?;
    let credit = view
        .account(credit_id)
        .ok_or(LedgerError::UnknownAccount(credit_id))?;
    check_unit(debit, unit)?;
    check_unit(credit, unit)?;
    Ok((debit, credit))
}

fn check_unit(account: &Account, unit: &crate::types::UnitId) -> Result<(), LedgerError> {
    if &account.unit != unit {
        return Err(LedgerError::UnitMismatch {
            transfer: unit.clone(),
            account: account.unit.clone(),
        });
    }
    Ok(())
}
