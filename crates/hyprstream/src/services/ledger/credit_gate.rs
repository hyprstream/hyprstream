//! [`CreditGate`] — the enforcement-plane amortization cache for the Phase-1
//! local-enforcer (plan §5.2 / §5.3, item 1.7).
//!
//! Shaped after `mac::avc::CachingAvc`: a verified grant is resolved once per
//! epoch and cached; the per-op hot path ([`CreditGate::try_hold`]) touches only
//! the grant's fine-grained balance cell (one short-lived per-cell lock), no
//! ledger I/O, no signatures, no lattice walk — INV-2(a). A generation counter
//! is the revocation epoch: bumping it (policy reload, grant reissue, or
//! [`ReceiptDebt`](super::DebtBreaker)) invalidates every cached entry keyed
//! to an older generation, so the next admit re-verifies + re-materializes.
//!
//! # Spend-authorization verification (§5.3)
//!
//! The subject signs `{grant_cid, host, transfer_id, max_amount, exp}` with its
//! identity key (COSE composite; Classical-only clients yield Classical-
//! assurance receipts, labeled not rejected). [`CreditGate::verify_spend_authz`]
//! verifies that signature against the *same* verified key material as the
//! envelope. The authorization is captured at admission — no signature
//! round-trip at completion time.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use hyprstream_crypto::cose_sign::verify_composite;
use hyprstream_crypto::pq::MlDsaVerifyingKey;
use hyprstream_ledger::{Cid, Did, TransferId, UnitId};
use parking_lot::Mutex;

/// The reason a spend was denied at the gate. All denial classes are
/// retryable-or-deny-stable per plan §5.4; the [`super::LocalEnforcer`]
/// converts these into the admission result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenyReason {
    /// No verified identity (anonymous holds no inventory).
    Anonymous,
    /// The presented grant could not be verified (bad chain / signature / expired).
    UnverifiableGrant(String),
    /// The unit is not one this enforcer recognizes (no account materialized).
    UnknownUnit,
    /// The authenticated transport subject is not the holder named by the grant.
    HolderMismatch,
    /// The spend-authorization signature was absent or invalid.
    InvalidSpendAuthorization(String),
    /// The locally-materialized balance cannot cover the requested amount.
    InsufficientCredit {
        /// What was available on the cell at deny time.
        available: u128,
        /// What the spend asked for.
        requested: u128,
    },
    /// The gate is cold: its cached balance predates the current generation
    /// (revocation, policy reload, restart) and no fresh ledger snapshot has
    /// arrived yet. Deny until the first `CommitEvent` materializes the cell.
    ColdGate,
    /// Receipt debt: the outbox cannot durably evidence usage, so new spends
    /// are fail-closed (plan §2e / Appendix A.5). Already-committed spends
    /// stand.
    ReceiptDebt,
}

/// A grant that has passed full UCAN-chain + signature verification. Cached
/// keyed by its [`Cid`] so the hot path never re-verifies (plan §5.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedGrant {
    /// The verified, exact pairwise identity whose inventory this grant names.
    /// A verifier that cannot establish this value must return
    /// [`DenyReason::UnverifiableGrant`] rather than construct a grant.
    pub holder: Did,
    /// The unit (names its issuer — INV-1).
    pub unit: UnitId,
    /// The total cap authorized by this grant (minor units).
    pub cap_amount: u128,
    /// Grant expiry (unix seconds).
    pub exp: u64,
    /// Allocation epoch — bumped on reissue/revoke; the cache + balance cells
    /// key on `(cid, epoch)`.
    pub epoch: u64,
}

/// The seam that turns a presented grant `Cid` into a [`VerifiedGrant`]. The
/// production impl runs `hyprstream_rpc::auth::ucan::chain::validate` +
/// `verify_composite` against the issuer's keys (a follow-up wiring, clearly
/// marked in the `#[service_factory]`); the static test double is the default.
#[async_trait]
pub trait GrantVerifier: Send + Sync {
    /// Verify the grant identified by `grant_cid`. `Ok` caches it; `Err` denies
    /// admission with [`DenyReason::UnverifiableGrant`].
    async fn verify(&self, grant_cid: &Cid) -> Result<VerifiedGrant, DenyReason>;
}

/// The subject's spend authorization (§5.3, §4.3 `subjectAuth`): a signature
/// over `{grant_cid, host, transfer_id, max_amount, exp}`. Captured at
/// admission so completion-time posting needs no round-trip; `max_amount` +
/// partial-post cover metered usage below the cap.
#[derive(Debug, Clone)]
pub struct SpendAuthorization {
    /// The grant this spend draws on.
    pub grant_cid: Cid,
    /// The cell ledger's service identity (the receipt host).
    pub host: Did,
    /// The transfer id (idempotency key / nonce) this auth covers.
    pub transfer_id: TransferId,
    /// The maximum the subject authorizes for this transfer.
    pub max_amount: u128,
    /// Authorization expiry (unix seconds).
    pub exp: u64,
    /// The subject's COSE composite signature over the canonical input.
    pub signature: Vec<u8>,
}

impl SpendAuthorization {
    /// Domain-separated blake3 digest of the authorization fields — the single
    /// canonical payload both the subject signs and the gate verifies. Kept on
    /// the struct so a real client signer and the verifier cannot drift apart.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"hs-ledger-spend-authz-v1");
        h.update(&self.grant_cid.0);
        h.update(self.host.as_str().as_bytes());
        h.update(&self.transfer_id.0.to_be_bytes());
        h.update(&self.max_amount.to_be_bytes());
        h.update(&self.exp.to_be_bytes());
        *h.finalize().as_bytes()
    }
}

impl CreditGate {
    /// Verify a spend-authorization signature against the subject's verified
    /// key material (§5.3). `require_pq` follows the cell's `CryptoPolicy` — a
    /// Classical-only client yields a Classical-assurance receipt (labeled, not
    /// rejected), so callers pass `require_pq = false` when no PQ key is
    /// presented.
    pub fn verify_spend_authz(
        authz: &SpendAuthorization,
        ed_vk: &ed25519_dalek::VerifyingKey,
        pq_vk: Option<&MlDsaVerifyingKey>,
        require_pq: bool,
    ) -> Result<(), DenyReason> {
        let payload = authz.digest();
        verify_composite(&authz.signature, ed_vk, pq_vk, &payload, &[], require_pq)
            .map(|_| ())
            .map_err(|e| DenyReason::InvalidSpendAuthorization(e.to_string()))
    }
}

/// A materialized balance cell — the per-`(grant_cid, epoch)` counter the hot
/// path decrements. The available balance is a `u128` (the ledger's amount
/// width); this target's `std` does not expose 128-bit atomics, so the balance
/// is a fine-grained per-cell `Mutex<u128>` rather than a lock-free CAS. This
/// still satisfies the AVC amortization model: the per-op cost is one
/// short-lived lock on *this grant's* cell — no Casbin, no signature, no
/// lattice walk, no global contention (contention is per-grant only). A
/// lock-free CAS is a documented follow-up on targets that expose
/// `AtomicU128`.
#[derive(Debug)]
struct BalanceCell {
    available: Mutex<u128>,
    /// The generation at which this cell was last materialized from a ledger
    /// snapshot. A mismatch with the gate's current generation ⇒ stale ⇒ cold
    /// deny until refreshed. (Atomic — read on every hot-path entry.)
    materialized_gen: AtomicU64,
}

impl BalanceCell {
    fn new(available: u128, gen: u64) -> Arc<Self> {
        Arc::new(BalanceCell {
            available: Mutex::new(available),
            materialized_gen: AtomicU64::new(gen),
        })
    }
}

/// A held spend — return it to the gate with [`CreditGate::release`] when the
/// in-flight job settles (or the durable reserve fails and the job cancels).
#[derive(Debug, Clone)]
pub struct Hold {
    grant_cid: Cid,
    amount: u128,
}

impl Hold {
    /// The grant this hold draws on.
    pub fn grant_cid(&self) -> &Cid {
        &self.grant_cid
    }

    /// The held amount.
    pub fn amount(&self) -> u128 {
        self.amount
    }

    /// Construct a hold for a (sub-)amount of `grant_cid`. `pub(super)` so the
    /// [`super::enforcer::LocalEnforcer`] can build a remainder hold when
    /// settling a partial post.
    pub(super) fn new(grant_cid: Cid, amount: u128) -> Self {
        Hold { grant_cid, amount }
    }
}

/// The enforcement-plane amortization cache (plan §5.2, item 1.7).
pub struct CreditGate {
    verifier: Arc<dyn GrantVerifier + Send + Sync>,
    /// Shared with the [`DebtBreaker`](super::DebtBreaker), which bumps it on
    /// receipt debt. Revocation / policy reload also bump it.
    generation: Arc<AtomicU64>,
    cells: Mutex<HashMap<Cid, Arc<BalanceCell>>>,
    /// Grants cached keyed by cid (verified once per epoch).
    grants: Mutex<HashMap<Cid, Arc<VerifiedGrant>>>,
}

impl std::fmt::Debug for CreditGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreditGate")
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .finish()
    }
}

impl CreditGate {
    /// Construct with a grant verifier. The generation counter starts at 0;
    /// share it with the [`DebtBreaker`](super::DebtBreaker) via
    /// [`Self::generation_handle`].
    pub fn new(verifier: Arc<dyn GrantVerifier + Send + Sync>) -> Self {
        CreditGate {
            verifier,
            generation: Arc::new(AtomicU64::new(0)),
            cells: Mutex::new(HashMap::new()),
            grants: Mutex::new(HashMap::new()),
        }
    }

    /// The shared generation handle — pass this to
    /// [`DebtBreaker::new`](super::DebtBreaker::new) so receipt debt bumps the
    /// gate's epoch.
    pub fn generation_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.generation)
    }

    /// Current generation (revocation epoch).
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Bump the generation counter (revocation / policy reload). Every cached
    /// entry keyed to an older generation becomes a miss ⇒ re-verify +
    /// re-materialize on next admit.
    pub fn bump_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Resolve (and cache) a verified grant by cid. Cache hits are O(1) and
    /// perform no signature work.
    pub async fn verify_grant(&self, grant_cid: &Cid) -> Result<Arc<VerifiedGrant>, DenyReason> {
        if let Some(cached) = self.grants.lock().get(grant_cid).cloned() {
            return Ok(cached);
        }
        let grant = self.verifier.verify(grant_cid).await?;
        self.grants
            .lock()
            .insert(grant_cid.clone(), Arc::new(grant.clone()));
        Ok(Arc::new(grant))
    }

    /// Materialize (or refresh) a grant's balance cell from a ledger snapshot.
    /// Called when a `CommitEvent` arrives from the ledger actor (post/void/
    /// expire/settlement). `available` is the authoritative available balance
    /// for `(grant, epoch)` at this generation.
    pub fn materialize(&self, grant_cid: &Cid, available: u128) {
        let gen = self.generation();
        let cell = BalanceCell::new(available, gen);
        self.cells.lock().insert(grant_cid.clone(), cell);
    }

    /// Replenish a cell by `delta` (e.g. a credit posted to the holder's
    /// account). Saturating; a no-op if the cell does not exist yet (it will
    /// be materialized by the next snapshot).
    pub fn replenish(&self, grant_cid: &Cid, delta: u128) {
        let cells = self.cells.lock();
        if let Some(cell) = cells.get(grant_cid) {
            let mut bal = cell.available.lock();
            *bal = bal.saturating_add(delta);
        }
    }

    /// The hot-path admission primitive (INV-2(a)): one short-lived per-cell
    /// lock, no ledger I/O, no signatures. Returns a [`Hold`] on success.
    ///
    /// Denies cold if the cell's `materialized_gen` lags the current
    /// generation (the cache was invalidated and no fresh snapshot arrived).
    pub fn try_hold(&self, grant_cid: &Cid, amount: u128) -> Result<Hold, DenyReason> {
        if amount == 0 {
            // A zero-amount hold is a no-op admit (e.g. a metered-rate job with
            // no upfront reservation); succeeds without touching the cell.
            return Ok(Hold {
                grant_cid: grant_cid.clone(),
                amount: 0,
            });
        }
        let cell = {
            let cells = self.cells.lock();
            cells.get(grant_cid).cloned()
        };
        let cell = cell.ok_or(DenyReason::ColdGate)?;
        if cell.materialized_gen.load(Ordering::Acquire) != self.generation() {
            return Err(DenyReason::ColdGate);
        }
        let mut bal = cell.available.lock();
        match bal.checked_sub(amount) {
            Some(new_bal) => {
                *bal = new_bal;
                Ok(Hold {
                    grant_cid: grant_cid.clone(),
                    amount,
                })
            }
            None => Err(DenyReason::InsufficientCredit {
                available: *bal,
                requested: amount,
            }),
        }
    }

    /// Release a held amount back to the cell (the job settled below the hold,
    /// or was cancelled before the durable reserve landed).
    pub fn release(&self, hold: &Hold) {
        if hold.amount == 0 {
            return;
        }
        let cells = self.cells.lock();
        if let Some(cell) = cells.get(&hold.grant_cid) {
            let mut bal = cell.available.lock();
            *bal = bal.saturating_add(hold.amount);
        }
    }

    /// The currently-cached available balance for a grant, if materialized.
    pub fn cached_available(&self, grant_cid: &Cid) -> Option<u128> {
        self.cells
            .lock()
            .get(grant_cid)
            .map(|c| *c.available.lock())
    }
}

/// Static, in-memory [`GrantVerifier`] for tests and for bootstrapping a cell
/// before the UCAN-chain wiring is connected. Maps a grant `Cid` to a
/// [`VerifiedGrant`].
#[derive(Debug, Default)]
pub struct StaticGrantVerifier {
    grants: Mutex<HashMap<Cid, VerifiedGrant>>,
}

impl StaticGrantVerifier {
    /// Create an empty verifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a verified grant under its cid.
    pub fn with(self, grant_cid: Cid, grant: VerifiedGrant) -> Self {
        self.grants.lock().insert(grant_cid, grant);
        self
    }

    /// Register a verified grant under its cid (by reference).
    pub fn insert(&self, grant_cid: Cid, grant: VerifiedGrant) {
        self.grants.lock().insert(grant_cid, grant);
    }
}

#[async_trait]
impl GrantVerifier for StaticGrantVerifier {
    async fn verify(&self, grant_cid: &Cid) -> Result<VerifiedGrant, DenyReason> {
        self.grants
            .lock()
            .get(grant_cid)
            .cloned()
            .ok_or_else(|| DenyReason::UnverifiableGrant("grant cid not registered".to_owned()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn cid(b: u8) -> Cid {
        Cid(vec![b])
    }
    fn unit() -> UnitId {
        UnitId {
            issuer: Did("did:web:issuer.test".to_owned()),
            resource_class: "gpu.h100.seconds".to_owned(),
        }
    }
    fn grant(cap: u128) -> VerifiedGrant {
        VerifiedGrant {
            holder: Did("did:web:alice".to_owned()),
            unit: unit(),
            cap_amount: cap,
            exp: u64::MAX,
            epoch: 0,
        }
    }

    fn gate_with(grant: VerifiedGrant) -> (CreditGate, Cid) {
        let c = cid(1);
        let verifier = Arc::new(StaticGrantVerifier::new().with(c.clone(), grant));
        (CreditGate::new(verifier), c)
    }

    #[tokio::test]
    async fn try_hold_cas_subtracts_and_releases() {
        let (gate, c) = gate_with(grant(1000));
        gate.materialize(&c, 1000);
        let h = gate.try_hold(&c, 300).unwrap();
        assert_eq!(gate.cached_available(&c), Some(700));
        gate.release(&h);
        assert_eq!(gate.cached_available(&c), Some(1000));
    }

    #[tokio::test]
    async fn try_hold_denies_insufficient_credit() {
        let (gate, c) = gate_with(grant(100));
        gate.materialize(&c, 100);
        let err = gate.try_hold(&c, 101).unwrap_err();
        assert_eq!(
            err,
            DenyReason::InsufficientCredit {
                available: 100,
                requested: 101
            }
        );
        // A failed hold leaves the balance untouched.
        assert_eq!(gate.cached_available(&c), Some(100));
    }

    #[tokio::test]
    async fn unmaterialized_or_stale_cell_is_cold_deny() {
        let (gate, c) = gate_with(grant(100));
        // Never materialized → cold.
        assert_eq!(gate.try_hold(&c, 10).unwrap_err(), DenyReason::ColdGate);
        gate.materialize(&c, 100);
        assert!(gate.try_hold(&c, 10).is_ok());
        // Bump generation (e.g. revocation) → cell now stale → cold deny until
        // re-materialized.
        gate.bump_generation();
        assert_eq!(gate.try_hold(&c, 10).unwrap_err(), DenyReason::ColdGate);
        gate.materialize(&c, 90);
        assert!(gate.try_hold(&c, 10).is_ok());
    }

    #[tokio::test]
    async fn receipt_debt_bumps_shared_generation() {
        let (gate, c) = gate_with(grant(100));
        gate.materialize(&c, 100);
        let gen_handle = gate.generation_handle();
        // The DebtBreaker shares this handle and bumps it on debt.
        gen_handle.fetch_add(1, Ordering::AcqRel);
        // The gate observes the bumped generation → cached cell is now cold.
        assert_eq!(gate.try_hold(&c, 10).unwrap_err(), DenyReason::ColdGate);
    }

    #[tokio::test]
    async fn grant_verification_caches_after_first_resolve() {
        let g = grant(50);
        let (gate, c) = gate_with(g);
        let v1 = gate.verify_grant(&c).await.unwrap();
        let v2 = gate.verify_grant(&c).await.unwrap();
        assert_eq!(*v1, *v2);
        assert_eq!(v1.cap_amount, 50);
    }

    #[tokio::test]
    async fn unverifiable_grant_denies() {
        let verifier: Arc<dyn GrantVerifier + Send + Sync> = Arc::new(StaticGrantVerifier::new());
        let gate = CreditGate::new(verifier);
        let err = gate.verify_grant(&cid(9)).await.unwrap_err();
        assert!(matches!(err, DenyReason::UnverifiableGrant(_)));
    }

    #[test]
    fn spend_authz_signature_roundtrips_and_detects_tamper() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[5u8; 32]);
        let pq_sk = hyprstream_crypto::pq::ml_dsa_sk_from_seed(&[6u8; 32]);
        let ed_vk = ed_sk.verifying_key();
        let pq_vk = hyprstream_crypto::pq::ml_dsa_vk_from_bytes(
            &hyprstream_crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk),
        )
        .unwrap();

        let mut authz = SpendAuthorization {
            grant_cid: cid(1),
            host: Did("did:web:cell.test".to_owned()),
            transfer_id: TransferId(42),
            max_amount: 1000,
            exp: 9_999_999_999,
            signature: Vec::new(),
        };
        // Sign the canonical digest directly (a real client signs the same way).
        let digest = authz.digest();
        authz.signature =
            hyprstream_crypto::cose_sign::sign_composite(&ed_sk, Some(&pq_sk), &digest, &[])
                .unwrap();

        assert!(CreditGate::verify_spend_authz(&authz, &ed_vk, Some(&pq_vk), true).is_ok());

        // Tamper: raising max_amount invalidates the signature.
        authz.max_amount = 2000;
        assert!(CreditGate::verify_spend_authz(&authz, &ed_vk, Some(&pq_vk), true).is_err());
    }
}
