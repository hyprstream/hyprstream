//! Property, idempotency, two-phase-safety and expiry-race tests against the
//! `MemLedger` oracle (plan §6.1). RocksLedger equivalence is a later work item;
//! MemLedger is exercised here as both SUT and reference.

// The workspace denies unwrap/expect even in tests; a test harness legitimately
// unwraps known-good values.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_ledger::engine::{MAX_TIMEOUT_S, MIN_TIMEOUT_S};
use hyprstream_ledger::{
    Account, AccountFlags, AccountSpec, Did, IssueTransfer, LedgerBackend, LedgerError, MemLedger,
    Purpose, TransferId, TransferResult, UnitId,
};
use proptest::prelude::*;

const K_SPENDABLE: usize = 3;

fn ledger_id() -> Did {
    Did("did:web:cell.test".to_owned())
}

fn unit() -> UnitId {
    UnitId {
        issuer: Did("did:web:issuer.test".to_owned()),
        resource_class: "gpu.h100.seconds".to_owned(),
    }
}

/// Build a ledger with an issuer-liability account (index 0) and `K_SPENDABLE`
/// spendable accounts. Returns their account ids indexed 0..=K.
fn fresh() -> (MemLedger, Vec<hyprstream_ledger::AccountId>) {
    let mut l = MemLedger::new(ledger_id());
    let mut ids = Vec::new();
    let liability = l
        .open_account(AccountSpec::new(
            ledger_id(),
            unit().issuer.clone(),
            unit(),
            Purpose::IssuerLiability,
        ))
        .unwrap();
    ids.push(liability.id);
    for i in 0..K_SPENDABLE {
        let a = l
            .open_account(AccountSpec::new(
                ledger_id(),
                Did(format!("did:key:owner{i}")),
                unit(),
                Purpose::Available,
            ))
            .unwrap();
        ids.push(a.id);
    }
    (l, ids)
}

/// Conservation (INV-1(c) / plan §6.1): across ALL accounts, per unit,
/// Σdebits_posted == Σcredits_posted AND Σdebits_pending == Σcredits_pending.
fn assert_conservation(l: &MemLedger) {
    let (mut dp, mut cp, mut dpend, mut cpend) = (0u128, 0u128, 0u128, 0u128);
    for a in l.accounts() {
        dp += a.debits_posted;
        cp += a.credits_posted;
        dpend += a.debits_pending;
        cpend += a.credits_pending;
    }
    assert_eq!(dp, cp, "posted debits != posted credits");
    assert_eq!(dpend, cpend, "pending debits != pending credits");
}

/// Two-phase safety (plan §6.1): no debit-constrained account ever lets
/// `debits_posted + debits_pending` exceed `credits_posted` — the overdraft floor
/// holds across every reserve→post/void/expire interleaving.
fn assert_no_overdraft(l: &MemLedger) {
    for a in l.accounts() {
        if a.is_debit_constrained() {
            let committed = a.debits_posted.saturating_add(a.debits_pending);
            assert!(
                committed <= a.credits_posted,
                "overdraft on {:?}: {}+{} > {}",
                a.id,
                a.debits_posted,
                a.debits_pending,
                a.credits_posted
            );
        }
    }
}

/// One generated action over the fixed account set.
#[derive(Debug, Clone)]
enum Action {
    Credit {
        dest: usize,
        amount: u64,
    },
    Debit {
        from: usize,
        to: usize,
        amount: u64,
    },
    Reserve {
        from: usize,
        to: usize,
        amount: u64,
        timeout: u32,
    },
    Post {
        which: usize,
        partial: Option<u64>,
    },
    Void {
        which: usize,
    },
    Tick,
    Advance {
        secs: u64,
    },
}

fn action_strategy() -> impl Strategy<Value = Action> {
    // Indices 1..=K_SPENDABLE are spendable accounts; 0 is the liability.
    let idx = 1usize..=K_SPENDABLE;
    prop_oneof![
        (idx.clone(), 1u64..1000).prop_map(|(dest, amount)| Action::Credit { dest, amount }),
        (idx.clone(), idx.clone(), 1u64..1000).prop_map(|(from, to, amount)| Action::Debit {
            from,
            to,
            amount
        }),
        (idx.clone(), idx.clone(), 1u64..1000, MIN_TIMEOUT_S..50u32).prop_map(
            |(from, to, amount, timeout)| Action::Reserve {
                from,
                to,
                amount,
                timeout
            }
        ),
        (0usize..8, proptest::option::of(1u64..1000))
            .prop_map(|(which, partial)| Action::Post { which, partial }),
        (0usize..8).prop_map(|which| Action::Void { which }),
        Just(Action::Tick),
        (1u64..60).prop_map(|secs| Action::Advance { secs }),
    ]
}

fn next_id(counter: &mut u128) -> TransferId {
    *counter += 1;
    TransferId(*counter)
}

fn ud() -> [u8; 32] {
    [0u8; 32]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Conservation + two-phase overdraft safety + inline idempotency over an
    /// arbitrary interleaving of every op kind.
    #[test]
    fn sequence_invariants(actions in proptest::collection::vec(action_strategy(), 0..60)) {
        let (mut l, ids) = fresh();
        let mut counter: u128 = 0;
        // Reservation ids in creation order, for Post/Void to reference.
        let mut reserves: Vec<TransferId> = Vec::new();

        assert_conservation(&l);
        assert_no_overdraft(&l);

        for action in actions {
            match action {
                Action::Credit { dest, amount } => {
                    let id = next_id(&mut counter);
                    let t = IssueTransfer {
                        id,
                        issuer_liability: ids[0],
                        destination: ids[dest],
                        unit: unit(),
                        amount: amount as u128,
                        grant_cid: None,
                        user_data: ud(),
                    };
                    let out1 = l.credit(t.clone());
                    // Inline idempotency: replay is transparent, no state change.
                    let before = snapshot(&l);
                    let out2 = l.credit(t);
                    prop_assert_eq!(&out1, &out2);
                    prop_assert_eq!(before, snapshot(&l));
                }
                Action::Debit { from, to, amount } => {
                    let id = next_id(&mut counter);
                    let t = mk_transfer(id, ids[from], ids[to], amount);
                    let out1 = l.debit(t.clone());
                    let before = snapshot(&l);
                    let out2 = l.debit(t);
                    prop_assert_eq!(&out1, &out2);
                    prop_assert_eq!(before, snapshot(&l));
                }
                Action::Reserve { from, to, amount, timeout } => {
                    let id = next_id(&mut counter);
                    let t = mk_transfer(id, ids[from], ids[to], amount);
                    let out1 = l.reserve(t.clone(), timeout);
                    reserves.push(id);
                    let before = snapshot(&l);
                    let out2 = l.reserve(t, timeout);
                    prop_assert_eq!(&out1, &out2);
                    prop_assert_eq!(before, snapshot(&l));
                }
                Action::Post { which, partial } => {
                    if reserves.is_empty() { continue; }
                    let pending = reserves[which % reserves.len()];
                    let id = next_id(&mut counter);
                    let out1 = l.post(id, pending, partial.map(|p| p as u128));
                    let before = snapshot(&l);
                    let out2 = l.post(id, pending, partial.map(|p| p as u128));
                    prop_assert_eq!(&out1, &out2);
                    prop_assert_eq!(before, snapshot(&l));
                }
                Action::Void { which } => {
                    if reserves.is_empty() { continue; }
                    let pending = reserves[which % reserves.len()];
                    let id = next_id(&mut counter);
                    let out1 = l.void(id, pending);
                    let before = snapshot(&l);
                    let out2 = l.void(id, pending);
                    prop_assert_eq!(&out1, &out2);
                    prop_assert_eq!(before, snapshot(&l));
                }
                Action::Tick => {
                    l.tick(&NoopSigner).unwrap();
                }
                Action::Advance { secs } => {
                    l.advance_clock(l.clock() + secs);
                }
            }
            assert_conservation(&l);
            assert_no_overdraft(&l);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Expiry-vs-post race determinism (plan §2b.5, §6.1): whatever the ordering
    /// of {advance, post, void, tick} after a reserve, you can NEVER post once the
    /// logical clock reached the deadline (no late post), the second phase
    /// succeeds at most once, and conservation/overdraft hold throughout.
    #[test]
    fn expiry_post_race(
        amount in 1u64..500,
        timeout in MIN_TIMEOUT_S..20u32,
        steps in proptest::collection::vec(0u8..4, 1..12),
    ) {
        let (mut l, ids) = fresh();
        let mut counter: u128 = 0;

        // Fund the debit account generously so the reserve always succeeds.
        let fund = next_id(&mut counter);
        l.credit(IssueTransfer {
            id: fund,
            issuer_liability: ids[0],
            destination: ids[1],
            unit: unit(),
            amount: (amount as u128) * 4,
            grant_cid: None,
            user_data: ud(),
        });

        // The reservation under test.
        let rid = next_id(&mut counter);
        let reserved_at = l.clock();
        let deadline = reserved_at + timeout as u64;
        let r = l.reserve(mk_transfer(rid, ids[1], ids[2], amount), timeout);
        prop_assert!(r.is_ok());

        let mut applied_count = 0usize;
        let mut voided_count = 0usize;

        for s in steps {
            match s {
                0 => l.advance_clock(l.clock() + 1),
                1 => {
                    let now = l.clock();
                    let out = l.post(next_id(&mut counter), rid, None);
                    if let Ok(TransferResult::Applied { .. }) = out.result {
                        // No late post: an Applied post is impossible at/after the deadline.
                        prop_assert!(now < deadline, "posted at now={} >= deadline={}", now, deadline);
                        applied_count += 1;
                    }
                }
                2 => {
                    let out = l.void(next_id(&mut counter), rid);
                    if let Ok(TransferResult::Voided) = out.result {
                        voided_count += 1;
                    }
                }
                _ => { l.tick(&NoopSigner).unwrap(); }
            }
            assert_conservation(&l);
            assert_no_overdraft(&l);
        }

        // Exactly-once second phase: at most one post AND at most one void, and
        // never both.
        prop_assert!(applied_count <= 1);
        prop_assert!(voided_count <= 1);
        prop_assert!(applied_count + voided_count <= 1);
    }
}

fn mk_transfer(
    id: TransferId,
    from: hyprstream_ledger::AccountId,
    to: hyprstream_ledger::AccountId,
    amount: u64,
) -> hyprstream_ledger::Transfer {
    hyprstream_ledger::Transfer {
        id,
        debit_account: from,
        credit_account: to,
        unit: unit(),
        amount: amount as u128,
        grant_cid: None,
        user_data: ud(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LedgerSnapshot {
    head_seq: u64,
    journal_len: usize,
    accounts: Vec<(u128, u128, u128, u128)>,
}

/// A stable snapshot of every account's balance view and journal position, for
/// idempotency no-state-change/no-journal-growth assertions.
fn snapshot(l: &MemLedger) -> LedgerSnapshot {
    let mut accounts: Vec<_> = l
        .accounts()
        .map(|a: &Account| {
            (
                a.id.0,
                a.debits_posted,
                a.credits_posted,
                a.debits_pending + a.credits_pending,
            )
        })
        .collect();
    accounts.sort_by_key(|t| t.0);
    LedgerSnapshot {
        head_seq: l.head().seq,
        journal_len: l.journal().len(),
        accounts,
    }
}

/// A no-op checkpoint signer for tick (which never actually checkpoints in the
/// skeleton).
struct NoopSigner;
impl hyprstream_ledger::CheckpointSigner for NoopSigner {
    fn sign(&self, _input: &[u8]) -> Result<Vec<u8>, hyprstream_ledger::LedgerError> {
        Ok(Vec::new())
    }
    fn ledger_id(&self) -> &Did {
        static ID: std::sync::OnceLock<Did> = std::sync::OnceLock::new();
        ID.get_or_init(|| Did("did:web:cell.test".to_owned()))
    }
}

/// Bounds sanity: the timeout band is what the plan specifies.
#[test]
fn timeout_bounds_are_plan_values() {
    assert_eq!(MIN_TIMEOUT_S, 1);
    assert_eq!(MAX_TIMEOUT_S, 24 * 60 * 60);
}

#[test]
fn issuance_rejects_available_account_even_with_empty_flags() {
    let mut l = MemLedger::new(ledger_id());
    let source = l
        .open_account(AccountSpec {
            ledger_id: ledger_id(),
            owner: Did("did:key:not-issuer-liability".to_owned()),
            unit: unit(),
            purpose: Purpose::Available,
            flags: AccountFlags::empty(),
        })
        .unwrap();
    let dest = l
        .open_account(AccountSpec::new(
            ledger_id(),
            Did("did:key:dest".to_owned()),
            unit(),
            Purpose::Available,
        ))
        .unwrap();

    let out = l.credit(IssueTransfer {
        id: TransferId(1),
        issuer_liability: source.id,
        destination: dest.id,
        unit: unit(),
        amount: 10,
        grant_cid: None,
        user_data: ud(),
    });

    assert_eq!(out.result, Err(LedgerError::NotIssuerLiability(source.id)));
}
