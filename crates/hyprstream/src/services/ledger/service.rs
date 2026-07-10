//! [`LedgerService`] — the `Spawnable` that owns the Phase-1 local-enforcer
//! runtime (item 1.6): it spawns the ledger actor ([`LedgerHandle`]), wires the
//! [`CreditGate`](super::CreditGate) + [`DebtBreaker`](super::DebtBreaker) +
//! [`LocalEnforcer`](super::LocalEnforcer), and runs the tick + receipt-emitter
//! loop until shutdown.
//!
//! Registered via `#[service_factory("ledger")]` in
//! `services::factories` (gated behind the `ledger` feature, default off). It
//! is a barrier-style `Spawnable` (like the event bus): no RPC endpoints of its
//! own in Phase-1 — the enforcer is consumed in-process by the scheduler PEP.

use std::sync::Arc;
use std::time::Duration;

use hyprstream_ledger::{CheckpointSigner, Did, LedgerBackend};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::Spawnable;
use tokio::sync::Notify;

use super::actor::run_emitter_loop;
use super::credit_gate::{CreditGate, GrantVerifier};
use super::enforcer::LocalEnforcer;
use super::handle::LedgerHandle;
use super::sink::{DebtBreaker, ReceiptSink};
use super::LedgerConfig;

/// The Phase-1 cellular-ledger local-enforcer service.
pub struct LedgerService {
    config: LedgerConfig,
    handle: LedgerHandle,
    gate: Arc<CreditGate>,
    breaker: Arc<DebtBreaker>,
    sink: Arc<dyn ReceiptSink>,
    cell_identity: Did,
}

impl std::fmt::Debug for LedgerService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LedgerService")
            .field("cell_identity", &self.cell_identity)
            .field("enabled", &self.config.enabled)
            .finish()
    }
}

impl LedgerService {
    /// Construct + spawn the actor. `backend` is moved into the actor task;
    /// `signer` is shared (used for checkpoint/tick). The grant verifier is
    /// injected so the production UCAN-chain wiring and tests can supply their
    /// own. The receipt sink defaults to [`super::sink::LoggingReceiptSink`]
    /// when the PDS emission path is not yet wired.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        config: LedgerConfig,
        backend: Box<dyn LedgerBackend + Send + 'static>,
        signer: Arc<dyn CheckpointSigner + Send + Sync>,
        verifier: Arc<dyn GrantVerifier + Send + Sync>,
        sink: Arc<dyn ReceiptSink>,
        cell_identity: Did,
    ) -> Self {
        let handle = LedgerHandle::spawn(backend, signer);
        let gate = Arc::new(CreditGate::new(verifier));
        let breaker = Arc::new(DebtBreaker::new(&config, gate.generation_handle()));
        LedgerService {
            config,
            handle,
            gate,
            breaker,
            sink,
            cell_identity,
        }
    }

    /// The async facade over the single-writer ledger actor.
    pub fn handle(&self) -> &LedgerHandle {
        &self.handle
    }

    /// The enforcement-plane amortization cache (the scheduler PEP reads this).
    pub fn gate(&self) -> &Arc<CreditGate> {
        &self.gate
    }

    /// The receipt-debt breaker (the scheduler PEP checks `in_debt()`).
    pub fn breaker(&self) -> &Arc<DebtBreaker> {
        &self.breaker
    }

    /// Build a [`LocalEnforcer`] against this service's collaborators. Cheap;
    /// construct per scheduler-PEP (or share the `Arc` it implies).
    pub fn enforcer(&self) -> LocalEnforcer {
        LocalEnforcer::new(
            Arc::clone(&self.gate),
            self.handle.clone(),
            Arc::clone(&self.breaker),
            self.cell_identity.clone(),
            &self.config,
        )
    }
}

impl Spawnable for LedgerService {
    fn name(&self) -> &str {
        "ledger"
    }

    fn registrations(&self) -> Vec<(hyprstream_rpc::registry::SocketKind, TransportConfig)> {
        // No RPC endpoints of its own in Phase-1 — the enforcer is consumed
        // in-process by the scheduler PEP.
        Vec::new()
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| RpcError::Other(e.to_string()))?;
        rt.block_on(async move {
            // Signal readiness immediately — the actor + gate are already
            // constructed in spawn(); the emitter loop is the only background
            // task, and it tolerates an empty/idle outbox.
            if let Some(ready) = on_ready {
                let _ = ready.send(());
            }
            let _ = hyprstream_rpc::notify::ready();
            let tick = Duration::from_secs(self.config.tick_interval_secs.max(1));
            run_emitter_loop(
                self.handle,
                self.sink,
                self.breaker,
                self.cell_identity,
                tick,
                shutdown,
            )
            .await;
            Ok::<(), RpcError>(())
        })
    }
}
