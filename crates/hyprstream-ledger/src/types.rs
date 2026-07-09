//! Core data model for the accounting plane (plan §2.0).
//!
//! Everything here is `no_std`-friendly in spirit (no I/O, no clock, no tokio) so
//! the whole crate builds for `wasm32-unknown-unknown`. Amounts are `u128` minor
//! units; **no floats anywhere** (INV grep-gate: `f32|f64` forbidden outside
//! benches).

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// A decentralized identifier, carried opaquely (`did:web` / `did:key` /
/// `did:at9p`). The ledger crate treats DIDs as pseudonymous strings and never
/// resolves or interprets them — per the #924 P0 constraint, **no legal-identity
/// fields** live in this model; a DID is the only principal handle.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Did(pub String);

impl Did {
    /// Borrow the DID string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Did {
    fn from(s: &str) -> Self {
        Did(s.to_owned())
    }
}

/// Opaque content identifier for a UCAN allocation grant. Kept as raw bytes so
/// this crate carries **no `hyprstream-pds` dependency** (D3 crate boundary): the
/// ledger only correlates a spend to its grant, it never parses the CID.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Cid(pub Vec<u8>);

/// A resource unit. INV-1(a): **the unit names its issuer.** Two units with
/// different issuers are different units, full stop — there is no bearer-token
/// representation of value that omits the issuer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UnitId {
    /// The resource authority — the liability holder for this unit.
    pub issuer: Did,
    /// e.g. `"gpu.h100.seconds"`.
    pub resource_class: String,
}

/// What an account is *for*. The purpose participates in the account id, so two
/// accounts of the same owner/unit but different purpose are distinct rows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Purpose {
    /// Spendable balance held by an owner.
    Available,
    /// Funds escrowed toward a peer cell (home side of a tranche, §2f).
    Escrow {
        /// The peer cell the escrow is pledged to.
        peer_cell: Did,
    },
    /// The issuer's own liability account — issuance debits this (INV-1). Grows
    /// as supply is issued; the "money supply" per unit *is* this balance.
    IssuerLiability,
    /// Remotely-funded spendable balance (remote side of a tranche, §2f).
    Remote {
        /// The home cell that pre-funded this balance.
        home_cell: Did,
    },
    /// A posted bond (Phase 3 slashing collateral); modelled as a ledger account
    /// so a slash is just a transfer to the injured party.
    Bond,
}

/// 128-bit account identity. Deterministic function of the canonical tuple, so
/// any party that knows `(ledger_id, owner, unit, purpose)` can derive it with
/// no allocation-time coordination, and it is TigerBeetle-compatible (TB account
/// ids are `u128`).
///
/// PLAN DECISION 1: `AccountId = blake3_128(encode(ledger_id, owner, unit,
/// purpose))`. The unit already names its issuer (INV-1), so the issuer is baked
/// into the id transitively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AccountId(pub u128);

impl AccountId {
    /// Derive the canonical account id.
    ///
    /// The hash input is a domain-separated, length-prefixed encoding of the
    /// tuple rather than raw DAG-CBOR. This keeps `derive` **infallible** (no
    /// serializer `Result` to unwrap) while remaining canonical: it is the single
    /// source function every backend calls, so RocksLedger / TigerBeetleLedger
    /// derive byte-identical ids. (Trait-surface decision — see PR notes.)
    pub fn derive(ledger_id: &Did, owner: &Did, unit: &UnitId, purpose: &Purpose) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"hs-ledger-account-id-v1");
        write_field(&mut h, ledger_id.0.as_bytes());
        write_field(&mut h, owner.0.as_bytes());
        write_field(&mut h, unit.issuer.0.as_bytes());
        write_field(&mut h, unit.resource_class.as_bytes());
        write_purpose(&mut h, purpose);
        let mut out = [0u8; 16];
        h.finalize_xof().fill(&mut out);
        AccountId(u128::from_be_bytes(out))
    }
}

/// Length-prefixed field write (8-byte BE length + bytes) — unambiguous framing
/// so no two distinct tuples can collide by concatenation boundary shifting.
fn write_field(h: &mut blake3::Hasher, bytes: &[u8]) {
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(bytes);
}

fn write_purpose(h: &mut blake3::Hasher, purpose: &Purpose) {
    match purpose {
        Purpose::Available => {
            h.update(&[0u8]);
        }
        Purpose::Escrow { peer_cell } => {
            h.update(&[1u8]);
            write_field(h, peer_cell.0.as_bytes());
        }
        Purpose::IssuerLiability => {
            h.update(&[2u8]);
        }
        Purpose::Remote { home_cell } => {
            h.update(&[3u8]);
            write_field(h, home_cell.0.as_bytes());
        }
        Purpose::Bond => {
            h.update(&[4u8]);
        }
    }
}

bitflags! {
    /// TigerBeetle-compatible account flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct AccountFlags: u32 {
        /// The normal case: a debit-constrained account may never let
        /// `debits_posted + debits_pending` exceed `credits_posted`. Issuer
        /// liability accounts clear this flag (they are the source of supply).
        const DEBITS_MUST_NOT_EXCEED_CREDITS = 0b0000_0001;
    }
}

impl AccountFlags {
    /// The default flags for a purpose: everything is debit-constrained except an
    /// issuer's own liability account.
    pub fn for_purpose(purpose: &Purpose) -> Self {
        match purpose {
            Purpose::IssuerLiability => AccountFlags::empty(),
            _ => AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS,
        }
    }
}

/// TigerBeetle-compatible account state (plan §2.0). The four counters are `u128`
/// minor units. `*_posted` only ever grow; `*_pending` rise on reserve and fall
/// on post/void/expire. Balances are differences of counters, never stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Account {
    /// Derived identity (see [`AccountId::derive`]).
    pub id: AccountId,
    /// Denormalized unit; checked to match on every transfer touching this account.
    pub unit: UnitId,
    /// Denormalized purpose; issuance checks this directly instead of inferring
    /// authority from flags.
    pub purpose: Purpose,
    /// Held-but-not-settled debits (outstanding reservations on the debit side).
    pub debits_pending: u128,
    /// Settled debits — monotonically increasing.
    pub debits_posted: u128,
    /// Held-but-not-settled credits.
    pub credits_pending: u128,
    /// Settled credits — monotonically increasing.
    pub credits_posted: u128,
    /// Behavioural flags (overdraft constraint, …).
    pub flags: AccountFlags,
}

impl Account {
    /// A freshly opened account with zeroed counters.
    pub fn new(id: AccountId, unit: UnitId, purpose: Purpose, flags: AccountFlags) -> Self {
        Account {
            id,
            unit,
            purpose,
            debits_pending: 0,
            debits_posted: 0,
            credits_pending: 0,
            credits_posted: 0,
            flags,
        }
    }

    /// Amount available to spend for a debit-constrained account: settled net
    /// credit minus everything already promised. Pending **credits** do NOT count
    /// (they may still void) — the TigerBeetle rule. Saturating so it can never
    /// underflow into a huge number.
    pub fn available(&self) -> u128 {
        self.credits_posted
            .saturating_sub(self.debits_posted)
            .saturating_sub(self.debits_pending)
    }

    /// Whether this account enforces the overdraft floor.
    pub fn is_debit_constrained(&self) -> bool {
        self.flags
            .contains(AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS)
    }

    /// A read-only projection of balances at a given journal sequence.
    pub fn view(&self, as_of_seq: u64) -> BalanceView {
        BalanceView {
            debits_pending: self.debits_pending,
            debits_posted: self.debits_posted,
            credits_pending: self.credits_pending,
            credits_posted: self.credits_posted,
            available: self.available(),
            as_of_seq,
        }
    }
}

/// Everything needed to open (or idempotently re-open) an account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSpec {
    /// The cell ledger this account lives in.
    pub ledger_id: Did,
    /// The owning principal (a pseudonymous DID; may be a group owner DID).
    pub owner: Did,
    /// The unit (names its issuer).
    pub unit: UnitId,
    /// What the account is for.
    pub purpose: Purpose,
    /// Flags; use [`AccountFlags::for_purpose`] for the sensible default.
    pub flags: AccountFlags,
}

impl AccountSpec {
    /// Convenience constructor deriving default flags from the purpose.
    pub fn new(ledger_id: Did, owner: Did, unit: UnitId, purpose: Purpose) -> Self {
        let flags = AccountFlags::for_purpose(&purpose);
        AccountSpec {
            ledger_id,
            owner,
            unit,
            purpose,
            flags,
        }
    }

    /// The derived id for this spec.
    pub fn account_id(&self) -> AccountId {
        AccountId::derive(&self.ledger_id, &self.owner, &self.unit, &self.purpose)
    }
}

/// Client-supplied 128-bit transfer id (plan §2c). **The id IS the idempotency
/// key**: replaying it returns the original outcome (including original errors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransferId(pub u128);

/// Single-phase issuance (INV-1): the only entry point that grows a unit's
/// supply. The debit side MUST be the issuer's `IssuerLiability` account for the
/// unit; the credit side is the destination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssueTransfer {
    /// Idempotency key.
    pub id: TransferId,
    /// The issuer's liability account (debit side; must be `Purpose::IssuerLiability`).
    pub issuer_liability: AccountId,
    /// Where the freshly-issued credit lands.
    pub destination: AccountId,
    /// The unit being issued (issuer must match the liability account owner).
    pub unit: UnitId,
    /// Amount in minor units; must be `> 0`.
    pub amount: u128,
    /// The grant this issuance backs, if any.
    pub grant_cid: Option<Cid>,
    /// Opaque correlation (e.g. request hash).
    pub user_data: [u8; 32],
}

/// A value movement between two accounts of the **same unit** (INV-1(b): one
/// `unit` field, never two — cross-issuer movement is two linked transfers).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transfer {
    /// Idempotency key (for `reserve` this is also the pending reservation id).
    pub id: TransferId,
    /// Account debited.
    pub debit_account: AccountId,
    /// Account credited.
    pub credit_account: AccountId,
    /// The unit; must equal both accounts' unit.
    pub unit: UnitId,
    /// Amount in minor units; must be `> 0`.
    pub amount: u128,
    /// The UCAN allocation this spend draws on.
    pub grant_cid: Option<Cid>,
    /// Opaque correlation.
    pub user_data: [u8; 32],
}

/// The lifecycle state of a two-phase reservation (plan §2b).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PendingState {
    /// Held, not yet resolved.
    Pending,
    /// Second phase committed as a post (full or partial).
    Posted,
    /// Second phase committed as a void.
    Voided,
    /// Deadline passed before a second phase committed.
    Expired,
}

impl PendingState {
    /// Whether the reservation has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        !matches!(self, PendingState::Pending)
    }
}

/// A tracked phase-1 reservation. Held in `CF(pending)` in RocksLedger; a
/// `BTreeMap` entry in MemLedger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingReservation {
    /// The reserve transfer (its `id` is the pending id).
    pub transfer: Transfer,
    /// Logical deadline (ledger clock seconds) after which the hold expires.
    pub deadline: u64,
    /// Current lifecycle state.
    pub state: PendingState,
}

/// What a successful op did — the positive half of an [`Outcome`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferResult {
    /// Supply issued (`credit`).
    Issued,
    /// A single-phase or posted two-phase settlement.
    Applied {
        /// Amount moved to posted.
        posted: u128,
        /// Amount released back from pending (partial-post remainder; `0` for
        /// single-phase and full posts).
        released: u128,
    },
    /// A phase-1 hold was placed (`reserve`).
    Reserved,
    /// A reservation was cancelled (`void`).
    Voided,
    /// A reservation expired (produced by the sweep or on-touch).
    Expired,
    /// An account was created (`open_account` via the journal).
    Opened,
}

/// The recorded result of an operation. Stored under the op's `TransferId`; a
/// replay returns this **verbatim** (PLAN DECISION 3 — original result, including
/// original errors).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// The business result: `Ok` with what happened, or the deterministic error.
    pub result: Result<TransferResult, crate::errors::LedgerError>,
    /// The journal sequence that recorded this outcome.
    pub seq: u64,
}

impl Outcome {
    /// Whether the op succeeded.
    pub fn is_ok(&self) -> bool {
        self.result.is_ok()
    }
}

/// Read-only balance projection returned by [`crate::LedgerBackend::balance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalanceView {
    /// Outstanding debit holds.
    pub debits_pending: u128,
    /// Settled debits.
    pub debits_posted: u128,
    /// Outstanding credit holds.
    pub credits_pending: u128,
    /// Settled credits.
    pub credits_posted: u128,
    /// Spendable amount (see [`Account::available`]).
    pub available: u128,
    /// The journal sequence this view reflects.
    pub as_of_seq: u64,
}
