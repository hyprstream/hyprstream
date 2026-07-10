//! The ledger actor + its async facade [`LedgerHandle`] (plan §3.2 PLAN
//! DECISION 8, item 1.6).
//!
//! The [`hyprstream_ledger::LedgerBackend`] trait is deliberately synchronous
//! and single-writer. The service layer owns this actor — a `tokio` task that
//! holds the backend and a shared checkpoint signer, draining an `mpsc` of
//! [`LedgerCmd`]s and replying on per-call `oneshot` channels. The async
//! facade keeps the core crate tokio-free (the WASM requirement) and gives
//! the [`super::CreditGate`] / [`super::LocalEnforcer`] something to talk to
//! that is never on the hot admission path (INV-2): `admit` never awaits a
//! handle call; only the *durable reserve* does, after admission.
//!
//! Phase-1 groups commits one op per WriteBatch-equivalent (MemLedger has no
//! fsync). Batching to N=256 per batch is a later throughput item and does not
//! change the facade.

use std::sync::Arc;

use hyprstream_ledger::{
    Account, AccountId, AccountSpec, BalanceView, ChainHead, CheckpointSigner, IssueTransfer,
    LedgerBackend, LedgerError, OutboxItem, OutboxSeq, Outcome, SignedCheckpoint, TickReport,
    Transfer, TransferId,
};
use tokio::sync::{mpsc, oneshot};

/// A command queued to the ledger actor. Each carries its own reply channel.
#[derive(Debug)]
enum LedgerCmd {
    OpenAccount(AccountSpec, oneshot::Sender<Result<Account, LedgerError>>),
    Credit(IssueTransfer, oneshot::Sender<Outcome>),
    Debit(Transfer, oneshot::Sender<Outcome>),
    Reserve {
        transfer: Transfer,
        timeout_s: u32,
        reply: oneshot::Sender<Outcome>,
    },
    Post {
        id: TransferId,
        pending: TransferId,
        amount: Option<u128>,
        reply: oneshot::Sender<Outcome>,
    },
    Void {
        id: TransferId,
        pending: TransferId,
        reply: oneshot::Sender<Outcome>,
    },
    Balance(AccountId, oneshot::Sender<Result<BalanceView, LedgerError>>),
    Checkpoint(oneshot::Sender<Result<SignedCheckpoint, LedgerError>>),
    Tick(oneshot::Sender<Result<TickReport, LedgerError>>),
    OutboxPeek(usize, oneshot::Sender<Result<Vec<OutboxItem>, LedgerError>>),
    OutboxAck(OutboxSeq, oneshot::Sender<Result<(), LedgerError>>),
    Head(oneshot::Sender<ChainHead>),
}

/// Async facade over the single-writer ledger actor.
///
/// Cheap to clone (an `mpsc::Sender` handle). All methods return the
/// backend's result type verbatim; a dropped actor surface (the ledger
/// service shutting down) surfaces as a `RecvError` mapped to a fail-closed
/// [`LedgerError::Internal`].
#[derive(Clone)]
pub struct LedgerHandle {
    tx: mpsc::Sender<LedgerCmd>,
}

impl std::fmt::Debug for LedgerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LedgerHandle").finish()
    }
}

impl LedgerHandle {
    /// Spawn an actor owning `backend` + `signer` and return the facade. The
    /// actor runs until all handle clones drop or it panics (a panic is
    /// propagated as fail-closed errors on in-flight + future calls).
    pub fn spawn(
        backend: Box<dyn LedgerBackend + Send + 'static>,
        signer: Arc<dyn CheckpointSigner + Send + Sync>,
    ) -> Self {
        // A generous bound: the actor is the single writer and should never be
        // the bottleneck (the hot path does not call it). Backpressure here
        // would mean admission is outpacing durable accounting, which is a
        // config problem, not a queueing one.
        let (tx, mut rx) = mpsc::channel::<LedgerCmd>(1024);
        let handle = LedgerHandle { tx: tx.clone() };

        tokio::spawn(async move {
            let mut backend = backend;
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    LedgerCmd::OpenAccount(spec, r) => {
                        let _ = r.send(backend.open_account(spec));
                    }
                    LedgerCmd::Credit(t, r) => {
                        let _ = r.send(backend.credit(t));
                    }
                    LedgerCmd::Debit(t, r) => {
                        let _ = r.send(backend.debit(t));
                    }
                    LedgerCmd::Reserve {
                        transfer,
                        timeout_s,
                        reply,
                    } => {
                        let _ = reply.send(backend.reserve(transfer, timeout_s));
                    }
                    LedgerCmd::Post {
                        id,
                        pending,
                        amount,
                        reply,
                    } => {
                        let _ = reply.send(backend.post(id, pending, amount));
                    }
                    LedgerCmd::Void { id, pending, reply } => {
                        let _ = reply.send(backend.void(id, pending));
                    }
                    LedgerCmd::Balance(a, r) => {
                        let _ = r.send(backend.balance(a));
                    }
                    LedgerCmd::Checkpoint(r) => {
                        let _ = r.send(backend.checkpoint(signer.as_ref()));
                    }
                    LedgerCmd::Tick(r) => {
                        let _ = r.send(backend.tick(signer.as_ref()));
                    }
                    LedgerCmd::OutboxPeek(max, r) => {
                        let _ = r.send(backend.outbox_peek(max));
                    }
                    LedgerCmd::OutboxAck(up_to, r) => {
                        let _ = r.send(backend.outbox_ack(up_to));
                    }
                    LedgerCmd::Head(r) => {
                        let _ = r.send(backend.head());
                    }
                }
            }
            tracing::info!("ledger actor drained and exiting");
        });

        // Drop the actor's own sender clone is unnecessary — `handle.tx` is the
        // only retained sender; the actor exits when it drops.
        let _ = tx;
        handle
    }

    async fn call<R: std::fmt::Debug>(
        &self,
        make: impl FnOnce(oneshot::Sender<R>) -> LedgerCmd,
    ) -> Result<R, LedgerError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(make(tx))
            .await
            .map_err(|_| LedgerError::Internal("ledger actor unavailable".to_owned()))?;
        rx.await
            .map_err(|_| LedgerError::Internal("ledger actor dropped reply".to_owned()))
    }

    /// Idempotent account open.
    pub async fn open_account(&self, spec: AccountSpec) -> Result<Account, LedgerError> {
        self.call(|r| LedgerCmd::OpenAccount(spec, r)).await?
    }

    /// Single-phase issuance (INV-1).
    pub async fn credit(&self, t: IssueTransfer) -> Outcome {
        self.call(|r| LedgerCmd::Credit(t, r))
            .await
            .unwrap_or_else(|e| Outcome {
                result: Err(e),
                seq: 0,
            })
    }

    /// Single-phase spend.
    pub async fn debit(&self, t: Transfer) -> Outcome {
        self.call(|r| LedgerCmd::Debit(t, r))
            .await
            .unwrap_or_else(|e| Outcome {
                result: Err(e),
                seq: 0,
            })
    }

    /// Two-phase phase-1 hold.
    pub async fn reserve(&self, transfer: Transfer, timeout_s: u32) -> Outcome {
        self.call(|r| LedgerCmd::Reserve {
            transfer,
            timeout_s,
            reply: r,
        })
        .await
        .unwrap_or_else(|e| Outcome {
            result: Err(e),
            seq: 0,
        })
    }

    /// Two-phase phase-2 post (full or partial).
    pub async fn post(&self, id: TransferId, pending: TransferId, amount: Option<u128>) -> Outcome {
        self.call(|r| LedgerCmd::Post {
            id,
            pending,
            amount,
            reply: r,
        })
        .await
        .unwrap_or_else(|e| Outcome {
            result: Err(e),
            seq: 0,
        })
    }

    /// Two-phase phase-2 cancel.
    pub async fn void(&self, id: TransferId, pending: TransferId) -> Outcome {
        self.call(|r| LedgerCmd::Void {
            id,
            pending,
            reply: r,
        })
        .await
        .unwrap_or_else(|e| Outcome {
            result: Err(e),
            seq: 0,
        })
    }

    /// Read-only balance projection.
    pub async fn balance(&self, account: AccountId) -> Result<BalanceView, LedgerError> {
        self.call(|r| LedgerCmd::Balance(account, r)).await?
    }

    /// Force a signed checkpoint now.
    pub async fn checkpoint(&self) -> Result<SignedCheckpoint, LedgerError> {
        self.call(LedgerCmd::Checkpoint).await?
    }

    /// Housekeeping tick (expiry sweep + scheduled checkpoint).
    pub async fn tick(&self) -> Result<TickReport, LedgerError> {
        self.call(LedgerCmd::Tick).await?
    }

    /// Peek committed-but-unemitted proof-plane items.
    pub async fn outbox_peek(&self, max: usize) -> Result<Vec<OutboxItem>, LedgerError> {
        self.call(|r| LedgerCmd::OutboxPeek(max, r)).await?
    }

    /// Ack (drop) drained items up to `up_to`.
    pub async fn outbox_ack(&self, up_to: OutboxSeq) -> Result<(), LedgerError> {
        self.call(|r| LedgerCmd::OutboxAck(up_to, r)).await?
    }

    /// Current hash-chain head.
    pub async fn head(&self) -> Result<ChainHead, LedgerError> {
        self.call(LedgerCmd::Head).await
    }
}
