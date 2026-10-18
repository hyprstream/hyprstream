//! Private test-only, request-local observation; never changes authorization or cancellation.
use anyhow::{anyhow, ensure, Context, Result};
use hyprstream_inference::runtime::clef_decision::{DecisionBackboneObserver, DecisionCancel};
use hyprstream_rpc::service::EnvelopeContext;
use hyprstream_rpc_std::inference_client::DecisionRequest;
use parking_lot::{Condvar, Mutex};
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{sync_channel, Receiver, SyncSender},
    Arc, Weak,
};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub(crate) struct WorkerWitness {
    pub(crate) incarnation: String,
    pub(crate) generation: u64,
    pub(crate) model: String,
    pub(crate) layers: usize,
    total_checkpoints: usize,
    pub(crate) drain: Weak<AtomicBool>,
    pub(crate) artifact: Weak<crate::storage::pinned_model::PinnedModelArtifact>,
}

pub(crate) struct WorkerBinding {
    control: Weak<TestDecisionControl>,
    witness: WorkerWitness,
}
impl WorkerBinding {
    pub(crate) fn bind(
        control: Weak<TestDecisionControl>,
        incarnation: String,
        generation: u64,
        model: String,
        layers: usize,
        drain: &Arc<AtomicBool>,
        artifact: &Arc<crate::storage::pinned_model::PinnedModelArtifact>,
    ) -> Result<Self> {
        ensure!(
            (1..=128).contains(&layers),
            "T6 loaded layer count outside1..=128"
        );
        let total_checkpoints = layers
            .checked_add(6)
            .context("T6 checkpoint count overflow")?;
        ensure!(
            total_checkpoints <= 134,
            "T6 observer checkpoint bound exceeded"
        );
        let witness = WorkerWitness {
            incarnation,
            generation,
            model,
            layers,
            total_checkpoints,
            drain: Arc::downgrade(drain),
            artifact: Arc::downgrade(artifact),
        };
        let owner = control
            .upgrade()
            .context("T6 observer owner disappeared during spawn")?;
        let mut current = owner.worker.lock();
        if let Some(prior) = current.as_ref() {
            ensure!(
                witness.generation > prior.generation && witness.incarnation != prior.incarnation,
                "T6 duplicate/stale worker binding"
            );
        }
        *current = Some(witness.clone());
        drop(current);
        Ok(Self { control, witness })
    }

    pub(crate) fn attach(
        &self,
        ctx: &EnvelopeContext,
        request_id: u64,
        request: &DecisionRequest,
        cancel: DecisionCancel,
    ) -> DecisionCancel {
        let Some(owner) = self.control.upgrade() else {
            return cancel;
        };
        let mut armed = owner.armed.lock();
        let Some(expected) = armed.as_ref() else {
            return cancel;
        };
        let digest = request_digest(request);
        if expected.subject != ctx.subject().to_string()
            || Some(expected.tenant.as_str()) != ctx.verified_tenant()
            || expected.digest != digest
            || expected.ticket.worker.incarnation != self.witness.incarnation
            || expected.ticket.worker.generation != self.witness.generation
            || expected.ticket.worker.model != self.witness.model
        {
            owner.failed.store(true, Ordering::Release);
            return cancel;
        }
        let Some(expected) = armed.take() else {
            return cancel;
        };
        drop(armed);
        expected.ticket.request_id.lock().replace(request_id);
        let ticket = expected.ticket;
        cancel.with_backbone_observer(DecisionBackboneObserver::new(Arc::new(
            move |phase, deadline| {
                // Never hold an observer-control lock while waiting. Bounded,
                // nonblocking diagnostics cannot replace the real checker result.
                if ticket
                    .events
                    .try_send(Checkpoint {
                        phase,
                        deadline,
                        observed: Instant::now(),
                    })
                    .is_err()
                {
                    ticket.failed.store(true, Ordering::Release);
                }
                if phase == 5 {
                    let mut released = ticket.released.lock();
                    let watchdog = Instant::now() + Duration::from_secs(20);
                    while !*released {
                        let now = Instant::now();
                        if now >= watchdog {
                            ticket.failed.store(true, Ordering::Release);
                            break;
                        }
                        ticket.wake.wait_for(&mut released, watchdog - now);
                    }
                }
            },
        )))
    }
}

fn request_digest(request: &DecisionRequest) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update((request.clef_json.len() as u64).to_le_bytes());
    digest.update(request.clef_json.as_bytes());
    for value in [request.max_state_tokens, request.timeout_ms] {
        match value {
            Some(value) => {
                digest.update([1]);
                digest.update(value.to_le_bytes());
            }
            None => digest.update([0]),
        }
    }
    digest.finalize().into()
}

struct ArmedRequest {
    subject: String,
    tenant: String,
    digest: [u8; 32],
    ticket: Arc<RequestTicket>,
}

pub(crate) struct TestDecisionControl {
    armed: Mutex<Option<ArmedRequest>>,
    worker: Mutex<Option<WorkerWitness>>,
    failed: AtomicBool,
}
impl TestDecisionControl {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            armed: Mutex::new(None),
            worker: Mutex::new(None),
            failed: AtomicBool::new(false),
        })
    }
    pub(crate) fn current(&self) -> Result<WorkerWitness> {
        self.worker
            .lock()
            .clone()
            .context("T6 missing real worker binding")
    }
    pub(crate) fn arm(
        &self,
        subject: &str,
        tenant: &str,
        request: &DecisionRequest,
    ) -> Result<Arc<RequestTicket>> {
        self.check()?;
        let worker = self.current()?;
        let (events, receiver) = sync_channel(worker.total_checkpoints);
        let ticket = Arc::new(RequestTicket {
            worker,
            events,
            receiver: Mutex::new(receiver),
            failed: AtomicBool::new(false),
            released: Mutex::new(false),
            wake: Condvar::new(),
            request_id: Mutex::new(None),
        });
        let mut armed = self.armed.lock();
        ensure!(armed.is_none(), "T6 request arm already occupied");
        *armed = Some(ArmedRequest {
            subject: subject.to_owned(),
            tenant: tenant.to_owned(),
            digest: request_digest(request),
            ticket: Arc::clone(&ticket),
        });
        Ok(ticket)
    }
    pub(crate) fn check(&self) -> Result<()> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "T6 observer binding mismatch"
        );
        Ok(())
    }
    pub(crate) fn assert_consumed(&self) -> Result<()> {
        self.check()?;
        ensure!(self.armed.lock().is_none(), "T6 observer arm not consumed");
        Ok(())
    }
}

pub(crate) struct Checkpoint {
    pub(crate) phase: usize,
    pub(crate) deadline: Option<Instant>,
    pub(crate) observed: Instant,
}

pub(crate) struct RequestTicket {
    pub(crate) worker: WorkerWitness,
    events: SyncSender<Checkpoint>,
    receiver: Mutex<Receiver<Checkpoint>>,
    failed: AtomicBool,
    released: Mutex<bool>,
    wake: Condvar,
    request_id: Mutex<Option<u64>>,
}
impl RequestTicket {
    pub(crate) fn wait_partial_forward(&self) -> Result<Checkpoint> {
        let end = Instant::now() + Duration::from_secs(15);
        for expected in 1..=5 {
            let event = self
                .receiver
                .lock()
                .recv_timeout(end.saturating_duration_since(Instant::now()))
                .map_err(|e| anyhow!("T6 missing checkpoint{expected}: {e}"))?;
            ensure!(event.phase == expected, "T6 callback sequence mismatch");
            if expected == 5 {
                self.check()?;
                return Ok(event);
            }
        }
        Err(anyhow!("T6 no partial-forward witness"))
    }
    pub(crate) fn completed_success(&self) -> Result<()> {
        for expected in 1..=self.worker.total_checkpoints {
            let event = self
                .receiver
                .lock()
                .recv_timeout(Duration::from_secs(1))
                .map_err(|e| anyhow!("T6 missing successful checkpoint{expected}: {e}"))?;
            ensure!(
                event.phase == expected,
                "T6 successful callback sequence mismatch"
            );
        }
        ensure!(
            self.receiver.lock().try_recv().is_err(),
            "T6 extra successful callback beyond actual layer count"
        );
        self.check()
    }
    pub(crate) fn release(&self) {
        *self.released.lock() = true;
        self.wake.notify_all();
    }
    pub(crate) fn check(&self) -> Result<()> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "T6 callback overflow/watchdog failure"
        );
        ensure!(
            self.request_id.lock().is_some(),
            "T6 missing actual worker request binding"
        );
        Ok(())
    }
}

/// Independent controller owns this guard; all failure paths release the callback.
pub(crate) struct ReleaseOnDrop(pub(crate) Arc<RequestTicket>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}
