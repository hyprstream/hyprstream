//! [`LocalEnforcer`] — the Phase-1 admission contract (plan §5.1 / §5.4 / §5.5,
//! item 1.8 — the #761 realign).
//!
//! The scheduler stops owning quota and becomes:
//! *verify a presented capability → spend a credit → admit → evidence it.*
//! [`LocalEnforcer::admit`] is that contract. It is **reject-don't-queue**
//! (plan §5.4): every denial is a retryable `429`-shaped [`Rejection`], never a
//! promise to admit later. The admission result carries the `transfer_id` so
//! the completion path can post/void.
//!
//! ## Hold-then-reserve (plan §5.1)
//!
//! Admission is gated synchronously by the in-memory [`CreditGate`] (a CAS on
//! an atomic — INV-2(a)); the durable two-phase `reserve` against the ledger
//! happens *asynchronously after admit* via [`LocalEnforcer::reserve_durable`].
//! If the durable reserve then fails (e.g. a concurrent spend elsewhere in the
//! cell shrank the balance), the job is cancelled at its next preemption point
//! and **no receipt is owed** — the ledger's own overdraft check remains the
//! hard floor (INV-1 never violated; only scheduling optimism is).
//!
//! ## Wiring note (1.8)
//!
//! This implements the realigned admission *contract*. Plumbing it into the
//! live `hyprstream-workers` `SandboxPool::acquire` path is the physical
//! realign; it is gated on the open #761 group-authority decision (#921 open
//! decision 5) and lands behind the `ledger` flag, default off. The contract
//! here is what that path calls.

use std::sync::Arc;

use hyprstream_crypto::pq::MlDsaVerifyingKey;
use hyprstream_ledger::{
    AccountId, Cid, Did, LedgerError, Outcome, Purpose, Transfer, TransferId, UnitId,
};

use super::credit_gate::{CreditGate, DenyReason, Hold, SpendAuthorization, VerifiedGrant};
use super::handle::LedgerHandle;
use super::sink::DebtBreaker;
use super::LedgerConfig;

/// A retryable, `429`-shaped rejection (plan §5.4). `retry_after_secs` is a
/// hint for the handful of denial classes where backing off helps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    /// Why admission was denied.
    pub reason: DenyReason,
    /// Suggested retry delay, when applicable (insufficient credit / cold gate
    /// / receipt debt). `None` for hard denials (anonymous, unverifiable).
    pub retry_after_secs: Option<u64>,
}

impl Rejection {
    fn retryable(reason: DenyReason, retry_after: u64) -> Self {
        Rejection {
            reason,
            retry_after_secs: Some(retry_after),
        }
    }
    fn hard(reason: DenyReason) -> Self {
        Rejection {
            reason,
            retry_after_secs: None,
        }
    }
}

/// A spend admission request presented at the PEP.
#[derive(Debug, Clone)]
pub struct AdmissionRequest {
    /// The verified holder identity. `None` ⇒ anonymous ⇒ deny (plan §5.4).
    pub subject: Option<Did>,
    /// The presented allocation grant.
    pub grant_cid: Cid,
    /// The unit being spent (must match the grant's unit).
    pub unit: UnitId,
    /// Amount requested (minor units). `0` is a no-op admit (metered-rate job
    /// with no upfront reservation).
    pub amount: u128,
    /// Client nonce — the entropy that, with the grant cid, deterministically
    /// mints the [`TransferId`] (idempotency key).
    pub nonce: u128,
    /// The subject's spend authorization (§5.3). Required when
    /// [`LedgerConfig`] demands it.
    pub spend_authz: Option<SpendAuthorization>,
    /// The subject's verified Ed25519 key (same material the envelope verified).
    pub subject_ed_vk: Option<ed25519_dalek::VerifyingKey>,
    /// The subject's verified ML-DSA-65 key, if presented (Hybrid).
    pub subject_pq_vk: Option<MlDsaVerifyingKey>,
}

/// The result of admission.
#[derive(Debug)]
pub enum AdmissionResult {
    /// Admitted: a credit hold is placed and `transfer_id` minted. The durable
    /// reserve follows asynchronously.
    Admitted {
        /// The idempotency key / transfer id for the durable reserve.
        transfer_id: TransferId,
        /// The placed hold — return via [`LocalEnforcer::release`] or settle
        /// via post/void.
        hold: Hold,
        /// The verified grant the admission drew on.
        grant: Arc<VerifiedGrant>,
    },
    /// Rejected (reject-don't-queue, plan §5.4).
    Rejected(Rejection),
}

impl AdmissionResult {
    /// Whether admission succeeded.
    pub fn is_admitted(&self) -> bool {
        matches!(self, AdmissionResult::Admitted { .. })
    }
}

/// The Phase-1 local enforcer.
pub struct LocalEnforcer {
    gate: Arc<CreditGate>,
    handle: LedgerHandle,
    breaker: Arc<DebtBreaker>,
    cell_identity: Did,
    reserve_timeout_s: u32,
    require_pq: bool,
}

impl std::fmt::Debug for LocalEnforcer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEnforcer")
            .field("cell_identity", &self.cell_identity)
            .field("reserve_timeout_s", &self.reserve_timeout_s)
            .field("require_pq", &self.require_pq)
            .finish()
    }
}

impl LocalEnforcer {
    /// Construct with all collaborators. `breaker` shares its generation handle
    /// with `gate` (receipt debt bumps the gate's epoch).
    pub fn new(
        gate: Arc<CreditGate>,
        handle: LedgerHandle,
        breaker: Arc<DebtBreaker>,
        cell_identity: Did,
        config: &LedgerConfig,
    ) -> Self {
        LocalEnforcer {
            gate,
            handle,
            breaker,
            cell_identity,
            reserve_timeout_s: config.reserve_timeout_secs,
            require_pq: config.require_pq_signatures,
        }
    }

    /// The cell ledger's service identity.
    pub fn cell_identity(&self) -> &Did {
        &self.cell_identity
    }

    /// Derive the holder's spendable (`Available`) account id for `unit`.
    pub fn holder_account(&self, holder: &Did, unit: &UnitId) -> Result<AccountId, LedgerError> {
        AccountId::derive(&self.cell_identity, holder, unit, &Purpose::Available)
    }

    /// The Phase-1 admission contract (plan §5.1). `now` is the unix-second
    /// clock used only for grant-expiry (the balance CAS uses no clock).
    ///
    /// This is `async` only because grant verification is an async trait (the
    /// production UCAN path); on a cache hit it performs no I/O. The balance
    /// gate ([`CreditGate::try_hold`]) is a sync atomic CAS — INV-2(a).
    pub async fn admit(&self, req: &AdmissionRequest, now: u64) -> AdmissionResult {
        // 1. Fail-closed identity: anonymous holds no inventory ⇒ deny.
        let subject = match req.subject.as_ref() {
            Some(subject) => subject,
            None => return AdmissionResult::Rejected(Rejection::hard(DenyReason::Anonymous)),
        };

        // 2. Receipt debt ⇒ fail-closed for new spends (§2e). Committed spends
        //    stand; only new ones gate.
        if self.breaker.in_debt() {
            return AdmissionResult::Rejected(Rejection::retryable(
                DenyReason::ReceiptDebt,
                self.reserve_timeout_s.max(1) as u64,
            ));
        }

        // 3. Verify (and cache) the presented grant.
        let grant = match self.gate.verify_grant(&req.grant_cid).await {
            Ok(g) => g,
            Err(reason) => return AdmissionResult::Rejected(Rejection::hard(reason)),
        };
        if now >= grant.exp {
            return AdmissionResult::Rejected(Rejection::hard(DenyReason::UnverifiableGrant(
                "grant expired".to_owned(),
            )));
        }

        // 4. Bind the verified caller to the verified grant holder. Neither an
        //    absent holder nor a different (including root-equivalent) DID can
        //    authorize this spend: grants bind the exact pairwise subject.
        let holder = match &grant.holder {
            Some(holder) => holder,
            None => {
                return AdmissionResult::Rejected(Rejection::hard(DenyReason::UnverifiableGrant(
                    "grant has no holder".to_owned(),
                )));
            }
        };
        if subject != holder {
            return AdmissionResult::Rejected(Rejection::hard(DenyReason::UnverifiableGrant(
                "authenticated subject does not match grant holder".to_owned(),
            )));
        }

        // 5. Unknown unit ⇒ deny. The requested unit must match the grant's.
        if grant.unit != req.unit {
            return AdmissionResult::Rejected(Rejection::hard(DenyReason::UnknownUnit));
        }

        // 6. Spend-authorization verification (§5.3). Required by default; a
        //    Classical-only client passes no PQ key and is verified under the
        //    Classical policy (labeled, not rejected).
        if req.amount > 0 {
            match &req.spend_authz {
                Some(authz) => {
                    let ed_vk = match &req.subject_ed_vk {
                        Some(k) => k,
                        None => {
                            return AdmissionResult::Rejected(Rejection::hard(
                                DenyReason::InvalidSpendAuthorization(
                                    "no subject verifying key presented".to_owned(),
                                ),
                            ));
                        }
                    };
                    // The authz must bind to this grant and (if known) this cell.
                    if authz.grant_cid != req.grant_cid || authz.host != self.cell_identity {
                        return AdmissionResult::Rejected(Rejection::hard(
                            DenyReason::InvalidSpendAuthorization(
                                "authz not bound to this grant/host".to_owned(),
                            ),
                        ));
                    }
                    if req.amount > authz.max_amount {
                        return AdmissionResult::Rejected(Rejection::hard(
                            DenyReason::InvalidSpendAuthorization(
                                "requested amount exceeds authorized max".to_owned(),
                            ),
                        ));
                    }
                    let require_pq = self.require_pq && req.subject_pq_vk.is_some();
                    if let Err(reason) = CreditGate::verify_spend_authz(
                        authz,
                        ed_vk,
                        req.subject_pq_vk.as_ref(),
                        require_pq,
                    ) {
                        return AdmissionResult::Rejected(Rejection::hard(reason));
                    }
                }
                None => {
                    return AdmissionResult::Rejected(Rejection::hard(
                        DenyReason::InvalidSpendAuthorization(
                            "no spend authorization presented".to_owned(),
                        ),
                    ));
                }
            }
        }

        // 7. Mint the deterministic transfer id (idempotency key).
        let transfer_id = mint_transfer_id(&req.grant_cid, req.nonce);

        // 8. Balance gate: sync CAS on the materialized cell. Cold gate /
        //    insufficient credit ⇒ retryable reject, never queue.
        let hold = match self.gate.try_hold(&req.grant_cid, req.amount) {
            Ok(h) => h,
            Err(reason @ (DenyReason::ColdGate | DenyReason::InsufficientCredit { .. })) => {
                return AdmissionResult::Rejected(Rejection::retryable(
                    reason,
                    self.reserve_timeout_s.max(1) as u64,
                ));
            }
            // Defensive: other deny reasons are hard.
            Err(reason) => return AdmissionResult::Rejected(Rejection::hard(reason)),
        };

        AdmissionResult::Admitted {
            transfer_id,
            hold,
            grant,
        }
    }

    /// Durable two-phase reserve, fired asynchronously after a successful
    /// admit (plan §5.1, hold-then-reserve). Debits `debit_account` (the
    /// holder's Available) and credits `credit_account` (the cell usage
    /// account). On a failed outcome the hold is released and the caller must
    /// cancel the job — no receipt is owed.
    pub async fn reserve_durable(
        &self,
        admitted_transfer_id: TransferId,
        hold: &Hold,
        grant: &VerifiedGrant,
        debit_account: AccountId,
        credit_account: AccountId,
    ) -> Outcome {
        let transfer = Transfer {
            id: admitted_transfer_id,
            debit_account,
            credit_account,
            unit: grant.unit.clone(),
            amount: hold.amount(),
            grant_cid: Some(grant_cid_for_hold(hold)),
            user_data: [0u8; 32],
        };
        let outcome = self.handle.reserve(transfer, self.reserve_timeout_s).await;
        if !outcome.is_ok() {
            // The durable floor refused (e.g. concurrent spend shrank the
            // balance). Release the optimistic hold; the job must cancel.
            self.gate.release(hold);
        }
        outcome
    }

    /// Settle an admitted hold for `actual` usage (≤ the hold). Posts the
    /// reservation for `actual` and releases the remainder to the gate. Mints
    /// a fresh idempotency id for the post op.
    pub async fn post_actual(
        &self,
        admitted_transfer_id: TransferId,
        hold: &Hold,
        actual: u128,
    ) -> Outcome {
        let post_id = mint_post_id(admitted_transfer_id);
        let outcome = self
            .handle
            .post(post_id, admitted_transfer_id, Some(actual))
            .await;
        // Release the unused portion of the hold back to the gate regardless of
        // post outcome (on a failed/expired post the ledger has already released
        // the pending hold on its side; the gate's optimistic view must match).
        if actual < hold.amount() {
            let remainder = Hold::new(hold.grant_cid().clone(), hold.amount() - actual);
            self.gate.release(&remainder);
        }
        outcome
    }

    /// Cancel an admitted hold: void the reservation and release the full
    /// hold back to the gate.
    pub async fn void(&self, admitted_transfer_id: TransferId, hold: &Hold) -> Outcome {
        let void_id = mint_void_id(admitted_transfer_id);
        let outcome = self.handle.void(void_id, admitted_transfer_id).await;
        self.gate.release(hold);
        outcome
    }

    /// Release a hold without a ledger op (e.g. the durable reserve never
    /// landed because the job was cancelled first).
    pub fn release(&self, hold: &Hold) {
        self.gate.release(hold);
    }
}

// Small helpers kept as free functions so they are easy to unit-test in
// isolation from the actor plumbing.

fn grant_cid_for_hold(hold: &Hold) -> Cid {
    hold.grant_cid().clone()
}

/// Deterministic transfer id: `blake3_128("hs-ledger-xfer-v1" || grant_cid || nonce_be)`.
pub fn mint_transfer_id(grant_cid: &Cid, nonce: u128) -> TransferId {
    let mut h = blake3::Hasher::new();
    h.update(b"hs-ledger-xfer-id-v1");
    h.update(&grant_cid.0);
    h.update(&nonce.to_be_bytes());
    let mut out = [0u8; 16];
    h.finalize_xof().fill(&mut out);
    TransferId(u128::from_be_bytes(out))
}

fn mint_post_id(reserve: TransferId) -> TransferId {
    derived_id(b"hs-ledger-post-id-v1", reserve)
}

fn mint_void_id(reserve: TransferId) -> TransferId {
    derived_id(b"hs-ledger-void-id-v1", reserve)
}

fn derived_id(domain: &[u8], reserve: TransferId) -> TransferId {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&reserve.0.to_be_bytes());
    let mut out = [0u8; 16];
    h.finalize_xof().fill(&mut out);
    TransferId(u128::from_be_bytes(out))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::services::ledger::{
        CoseCheckpointSigner, CreditGate, DebtBreaker, StaticGrantVerifier,
    };
    use hyprstream_crypto::cose_sign::sign_composite;
    use hyprstream_crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes};
    use hyprstream_ledger::{LedgerBackend, MemLedger, Purpose};
    use rand::RngCore;

    fn unit() -> UnitId {
        UnitId {
            issuer: Did("did:web:issuer.test".to_owned()),
            resource_class: "gpu.h100.seconds".to_owned(),
        }
    }

    fn cid(b: u8) -> Cid {
        Cid(vec![b])
    }

    fn random_nonce() -> u128 {
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        u128::from_be_bytes(bytes)
    }

    /// Build a fully-wired enforcer over a fresh MemLedger, with `grant_cap`
    /// issued to `holder`'s Available account and materialized on the gate.
    /// Also opens a cell-owned usage account (the credit target for spends)
    /// and returns its id.
    async fn fixture(
        grant_cap: u128,
    ) -> (
        LocalEnforcer,
        Did,
        Cid,
        AccountId,
        AccountId,
        ed25519_dalek::SigningKey,
        hyprstream_crypto::pq::MlDsaSigningKey,
    ) {
        fixture_with_grant_holder(grant_cap, Some(Did("did:web:alice".to_owned()))).await
    }

    async fn fixture_with_grant_holder(
        grant_cap: u128,
        grant_holder: Option<Did>,
    ) -> (
        LocalEnforcer,
        Did,
        Cid,
        AccountId,
        AccountId,
        ed25519_dalek::SigningKey,
        hyprstream_crypto::pq::MlDsaSigningKey,
    ) {
        let cell = Did("did:web:cell.test".to_owned());
        let holder = Did("did:web:alice".to_owned());
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[1u8; 32]);
        let pq_sk = ml_dsa_sk_from_seed(&[2u8; 32]);

        // Backend + handle.
        let mut backend = MemLedger::new(cell.clone());
        let issuer_liab =
            AccountId::derive(&cell, &unit().issuer, &unit(), &Purpose::IssuerLiability).unwrap();
        let holder_acct = AccountId::derive(&cell, &holder, &unit(), &Purpose::Available).unwrap();
        let usage_acct = AccountId::derive(&cell, &cell, &unit(), &Purpose::Available).unwrap();
        backend
            .open_account(hyprstream_ledger::AccountSpec::new(
                cell.clone(),
                unit().issuer.clone(),
                unit(),
                Purpose::IssuerLiability,
            ))
            .unwrap();
        backend
            .open_account(hyprstream_ledger::AccountSpec::new(
                cell.clone(),
                holder.clone(),
                unit(),
                Purpose::Available,
            ))
            .unwrap();
        backend
            .open_account(hyprstream_ledger::AccountSpec::new(
                cell.clone(),
                cell.clone(),
                unit(),
                Purpose::Available,
            ))
            .unwrap();
        // Issue `grant_cap` from the issuer liability to the holder.
        backend
            .credit(hyprstream_ledger::IssueTransfer {
                id: TransferId(1),
                issuer_liability: issuer_liab,
                destination: holder_acct,
                unit: unit(),
                amount: grant_cap,
                grant_cid: Some(cid(1)),
                user_data: [0u8; 32],
            })
            .result
            .unwrap();

        let signer = Arc::new(CoseCheckpointSigner::classical(cell.clone(), ed_sk.clone()));
        let handle = LedgerHandle::spawn(Box::new(backend), signer);

        // Gate + breaker.
        let g = VerifiedGrant {
            holder: grant_holder,
            unit: unit(),
            cap_amount: grant_cap,
            exp: u64::MAX,
            epoch: 0,
        };
        let verifier = Arc::new(StaticGrantVerifier::new().with(cid(1), g));
        let gate = Arc::new(CreditGate::new(verifier));
        // Materialize the cell from the authoritative balance.
        let bal = handle.balance(holder_acct).await.unwrap();
        gate.materialize(&cid(1), bal.available);
        let cfg = LedgerConfig {
            enabled: true,
            require_pq_signatures: false,
            ..LedgerConfig::default()
        };
        let breaker = Arc::new(DebtBreaker::new(&cfg, gate.generation_handle()));
        let enforcer = LocalEnforcer::new(gate, handle, breaker, cell, &cfg);
        (
            enforcer,
            holder,
            cid(1),
            holder_acct,
            usage_acct,
            ed_sk,
            pq_sk,
        )
    }

    fn signed_authz(
        grant_cid: &Cid,
        host: &Did,
        transfer_nonce: u128,
        max_amount: u128,
        ed_sk: &ed25519_dalek::SigningKey,
        pq_sk: &hyprstream_crypto::pq::MlDsaSigningKey,
    ) -> (
        SpendAuthorization,
        ed25519_dalek::VerifyingKey,
        hyprstream_crypto::pq::MlDsaVerifyingKey,
        TransferId,
    ) {
        let transfer_id = mint_transfer_id(grant_cid, transfer_nonce);
        let mut authz = SpendAuthorization {
            grant_cid: grant_cid.clone(),
            host: host.clone(),
            transfer_id,
            max_amount,
            exp: u64::MAX,
            signature: Vec::new(),
        };
        let digest = authz.digest();
        authz.signature = sign_composite(ed_sk, Some(pq_sk), &digest, &[]).unwrap();
        let ed_vk = ed_sk.verifying_key();
        let pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(pq_sk)).unwrap();
        (authz, ed_vk, pq_vk, transfer_id)
    }

    #[tokio::test]
    async fn anonymous_subject_denied_hard() {
        let (enf, _holder, _cid, _debit, _credit, _, _) = fixture(1000).await;
        let req = AdmissionRequest {
            subject: None,
            grant_cid: cid(1),
            unit: unit(),
            amount: 10,
            nonce: 1,
            spend_authz: None,
            subject_ed_vk: None,
            subject_pq_vk: None,
        };
        match enf.admit(&req, 0).await {
            AdmissionResult::Rejected(r) => {
                assert_eq!(r.reason, DenyReason::Anonymous);
                assert!(r.retry_after_secs.is_none());
            }
            _ => panic!("anonymous must be denied"),
        }
    }

    #[tokio::test]
    async fn admitted_then_post_and_void_roundtrip() {
        let (enf, holder, grant_cid, debit, credit, ed_sk, pq_sk) = fixture(1000).await;
        let (authz, ed_vk, pq_vk, _tid) =
            signed_authz(&grant_cid, enf.cell_identity(), 7, 100, &ed_sk, &pq_sk);
        let req = AdmissionRequest {
            subject: Some(holder.clone()),
            grant_cid: grant_cid.clone(),
            unit: unit(),
            amount: 100,
            nonce: 7,
            spend_authz: Some(authz),
            subject_ed_vk: Some(ed_vk),
            subject_pq_vk: Some(pq_vk),
        };
        let (transfer_id, hold, grant) = match enf.admit(&req, 0).await {
            AdmissionResult::Admitted {
                transfer_id,
                hold,
                grant,
            } => (transfer_id, hold, grant),
            other => panic!("expected admit, got {other:?}"),
        };
        // Durable reserve against the holder's Available → a cell usage account.
        let res = enf
            .reserve_durable(transfer_id, &hold, &grant, debit, credit)
            .await;
        assert!(res.is_ok(), "reserve should succeed: {res:?}");

        // Post actual usage of 60 (partial), releasing 40 back.
        let post = enf.post_actual(transfer_id, &hold, 60).await;
        assert!(post.is_ok(), "post should succeed: {post:?}");
    }

    #[tokio::test]
    async fn different_subject_than_grant_holder_denied_hard() {
        let (enf, _holder, grant_cid, _debit, _credit, _ed_sk, _pq_sk) = fixture(1000).await;
        let attacker_ed_sk = ed25519_dalek::SigningKey::from_bytes(&[3u8; 32]);
        let attacker_pq_sk = ml_dsa_sk_from_seed(&[4u8; 32]);
        let nonce = random_nonce();
        let (authz, ed_vk, pq_vk, _) = signed_authz(
            &grant_cid,
            enf.cell_identity(),
            nonce,
            100,
            &attacker_ed_sk,
            &attacker_pq_sk,
        );
        let req = AdmissionRequest {
            subject: Some(Did("did:web:bob".to_owned())),
            grant_cid,
            unit: unit(),
            amount: 100,
            nonce,
            spend_authz: Some(authz),
            subject_ed_vk: Some(ed_vk),
            subject_pq_vk: Some(pq_vk),
        };

        match enf.admit(&req, 0).await {
            AdmissionResult::Rejected(r) => {
                assert_eq!(
                    r.reason,
                    DenyReason::UnverifiableGrant(
                        "authenticated subject does not match grant holder".to_owned()
                    )
                );
                assert!(r.retry_after_secs.is_none());
            }
            other => panic!("different subject must be denied, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn grant_without_holder_denied_hard() {
        let (enf, holder, grant_cid, _debit, _credit, ed_sk, pq_sk) =
            fixture_with_grant_holder(1000, None).await;
        let nonce = random_nonce();
        let (authz, ed_vk, pq_vk, _) =
            signed_authz(&grant_cid, enf.cell_identity(), nonce, 100, &ed_sk, &pq_sk);
        let req = AdmissionRequest {
            subject: Some(holder),
            grant_cid,
            unit: unit(),
            amount: 100,
            nonce,
            spend_authz: Some(authz),
            subject_ed_vk: Some(ed_vk),
            subject_pq_vk: Some(pq_vk),
        };

        match enf.admit(&req, 0).await {
            AdmissionResult::Rejected(r) => {
                assert_eq!(
                    r.reason,
                    DenyReason::UnverifiableGrant("grant has no holder".to_owned())
                );
                assert!(r.retry_after_secs.is_none());
            }
            other => panic!("grant without holder must be denied, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn insufficient_credit_rejects_retryable_not_queues() {
        let (enf, holder, grant_cid, _debit, _credit, ed_sk, pq_sk) = fixture(100).await;
        let (authz, ed_vk, pq_vk, _) =
            signed_authz(&grant_cid, enf.cell_identity(), 1, 500, &ed_sk, &pq_sk);
        let req = AdmissionRequest {
            subject: Some(holder),
            grant_cid,
            unit: unit(),
            amount: 500, // > available 100
            nonce: 1,
            spend_authz: Some(authz),
            subject_ed_vk: Some(ed_vk),
            subject_pq_vk: Some(pq_vk),
        };
        match enf.admit(&req, 0).await {
            AdmissionResult::Rejected(r) => {
                assert!(matches!(r.reason, DenyReason::InsufficientCredit { .. }));
                assert!(
                    r.retry_after_secs.is_some(),
                    "insufficient credit is retryable"
                );
            }
            _ => panic!("must reject"),
        }
    }

    #[tokio::test]
    async fn cold_gate_denies_until_materialized() {
        // Asserts the cold-gate contract at the gate level: an unmaterialized
        // grant denies ColdGate, then admits once materialized.
        let g = VerifiedGrant {
            holder: Some(Did("did:web:bob".to_owned())),
            unit: unit(),
            cap_amount: 100,
            exp: u64::MAX,
            epoch: 0,
        };
        let verifier = Arc::new(StaticGrantVerifier::new().with(cid(2), g));
        let gate = Arc::new(CreditGate::new(verifier));
        assert_eq!(
            gate.try_hold(&cid(2), 10).unwrap_err(),
            DenyReason::ColdGate
        );
        gate.materialize(&cid(2), 100);
        assert!(gate.try_hold(&cid(2), 10).is_ok());
    }

    #[test]
    fn transfer_id_is_deterministic_and_distinct_for_post_void() {
        let c = cid(1);
        let a = mint_transfer_id(&c, 5);
        let b = mint_transfer_id(&c, 5);
        assert_eq!(a, b, "same (cid,nonce) ⇒ same id");
        let other = mint_transfer_id(&c, 6);
        assert_ne!(a, other, "different nonce ⇒ different id");
        let post = mint_post_id(a);
        let void = mint_void_id(a);
        assert_ne!(post, void, "post and void ids must not collide");
        assert_ne!(post, a);
    }
}
