//! Subject-threaded 9P/VFS projection of the L1 CAS substrate (#813).
//!
//! The mount exposes CAS reads through the ordinary `Mount` surface, so access
//! goes through open/read/stat authorization instead of treating hashes as
//! bearer capabilities. A namespace may bind this mount wherever it chooses.
//!
//! ## Write-then-seal ingest (#814)
//!
//! Content addressing means the name (CID) is unknown until the bytes are
//! complete (Venti-style), so ingest is a *staging-then-seal* protocol rather
//! than an ordinary positioned write to a named file:
//!
//! 1. `create stage/new` (or `create /new` at the mount root) mints a
//!    per-Subject staging slot and returns a directory fid on `stage/<id>/`.
//!    Slots are quota'd per-slot and per-Subject (see [`StagingConfig`]).
//! 2. `Twrite` at offset into `stage/<id>/data` accumulates bytes
//!    (msize-chunked; resumable by offset; sparse gaps are zero-filled). Per-op
//!    `Mount` IO is unbounded — the 16 MB cap in
//!    [`hyprstream_vfs::namespace::MAX_READ_SIZE`] lives only on the `cat`/`echo`
//!    convenience loops, which bulk data must never use.
//! 3. Seal by writing `commit` to `stage/<id>/ctl` (**D-A**): the seal invokes
//!    [`CasSubstrate::put`] (CDC chunking to xorbs happens *below* the seal
//!    inside L1) and returns the manifest CID readback by *reading* `ctl`.
//!    Cluck-sealing cannot return the CID inline, which is why the seal is an
//!    explicit ctl command and the CID is read back separately.
//! 4. Abort is cluck-without-commit (or `abort` on ctl): staged bytes are
//!    discarded. Seal failure also converges to voided (fail-closed), matching
//!    the attested-resource saga convergence rule (#1064/#1066).
//!
//! The seal is the natural join point for [`crate::mac::ifc_join`]: any object
//! labels declared via `label` ctl commands are joined (LUB) and written into
//! the manifest. A slot with no declared label cannot seal; unlabeled staging is
//! transient only and is never admitted to the CAS.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use async_trait::async_trait;
use cas_serve::StoreError;
use hyprstream_rpc::Subject;
use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Lattice, Level, SecurityLabel};
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, ORDWR, OREAD, OTRUNC, OWRITE, Stat};
use parking_lot::Mutex;
use tokio::sync::Notify;
use tracing::{debug, warn};

use super::{CasError, CasSubstrate, DedupDomain};
use crate::mac::ifc_join;

const QTDIR: u8 = 0x80;
const QTFILE: u8 = 0x00;

/// Default per-slot staging cap (256 MiB). Staging buffers are in-memory today;
/// a spill-to-disk backing for truly huge ingests follows with the durable
/// resource saga (#1064).
pub const DEFAULT_STAGING_SLOT_QUOTA: usize = 256 * 1024 * 1024;
/// Default per-Subject aggregate cap across all open staging slots (1 GiB).
pub const DEFAULT_STAGING_SUBJECT_QUOTA: usize = 1024 * 1024 * 1024;

/// Object class addressed through [`CasMount`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasMountObjectKind {
    /// Full-file reconstruction by CID or legacy merkle.
    Object,
    /// Raw xorb bytes by xorb hash.
    Xorb,
    /// A write-then-seal staging slot (#814). `address` is the staging id.
    Stage,
}

/// Authorization request made before a CAS object is opened/read/stat'ed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasMountAuthzRequest<'a> {
    /// Which CAS object namespace is being accessed.
    pub kind: CasMountObjectKind,
    /// CID, legacy merkle, xorb hash, or staging id, exactly as walked.
    pub address: &'a str,
    /// Dedup domain the mount is serving.
    pub domain: &'a DedupDomain,
    /// Operation name for policy/audit (e.g. `open`, `read`, `stat`,
    /// `stage:create`, `stage:commit`).
    pub operation: &'static str,
}

/// Per-op authorization hook for [`CasMount`].
///
/// The default constructor uses [`DenyAllCasAuthorizer`]. The live XET read
/// route uses [`BootstrapCasAuthorizer`] (#1094 plane #1: explicit, audited
/// bootstrap grants, default-deny). As #699/#767 provenance and MAC plumbing
/// lands, production namespaces move to a real MAC authorizer; trusted local
/// single-tenant tools can explicitly opt into [`AllowAllCasAuthorizer`].
pub trait CasMountAuthorizer: Send + Sync {
    fn authorize(
        &self,
        caller: &Subject,
        request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError>;
}

/// Fail-closed authorizer used by [`CasMount::new`].
#[derive(Debug, Default)]
pub struct DenyAllCasAuthorizer;

impl CasMountAuthorizer for DenyAllCasAuthorizer {
    fn authorize(
        &self,
        caller: &Subject,
        request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError> {
        Err(MountError::PermissionDenied(format!(
            "CAS {} {} denied for {}: no CasMountAuthorizer installed",
            request.operation, request.address, caller
        )))
    }
}

/// Explicit allow authorizer for tests and trusted local single-tenant wiring.
#[derive(Debug, Default)]
pub struct AllowAllCasAuthorizer;

impl CasMountAuthorizer for AllowAllCasAuthorizer {
    fn authorize(
        &self,
        _caller: &Subject,
        _request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap-grant authorizer (#1094)
// ─────────────────────────────────────────────────────────────────────────────

/// Who a [`CasMountGrant`] applies to — the *subject breadth* axis the #1094
/// ratchet narrows.
///
/// Today there is exactly one breadth: the authenticated floor. Per-compartment
/// and per-subject variants arrive with #698 (authority-owned subject
/// clearances) and #699 (object provenance labels); they are deliberately not
/// modeled yet — see the ratchet path on [`BootstrapCasAuthorizer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasGrantSubject {
    /// Any authenticated (non-anonymous) [`Subject`]. Anonymous callers never
    /// match: the floor is *authenticated*, not bearerless.
    AnyAuthenticated,
}

impl CasGrantSubject {
    fn accepts(self, caller: &Subject) -> bool {
        match self {
            CasGrantSubject::AnyAuthenticated => !caller.is_anonymous(),
        }
    }
}

/// One explicit authorization grant — the unit of policy the
/// [`BootstrapCasAuthorizer`] evaluates (#1094).
///
/// A request matches a grant when the caller is accepted by `subject`, the
/// request's [`CasMountObjectKind`] is in `kinds`, and its operation string is
/// in `operations`. A request matching **no** grant is denied (fail-closed).
///
/// The shape mirrors the MAC decision it will become — subject breadth ×
/// object class × operation — so ratcheting toward the #698/#699 dominance
/// check (`subject.ctx ⊒ object.label` per op) narrows fields of this struct
/// instead of replacing an incompatible interim model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CasMountGrant {
    /// Stable audit name emitted with every decision the grant allows.
    pub name: &'static str,
    /// Subject breadth.
    pub subject: CasGrantSubject,
    /// Object kinds covered.
    pub kinds: &'static [CasMountObjectKind],
    /// Operations covered (matched against [`CasMountAuthzRequest::operation`]).
    pub operations: &'static [&'static str],
}

/// The day-one bootstrap grant: **"any authenticated subject may read xorb
/// X"** (#1094). Covers `open` + `read` on [`CasMountObjectKind::Xorb`] only —
/// `stat`, `obj/*` reads, and every `stage/*` operation stay denied.
pub const AUTHENTICATED_XORB_READ: CasMountGrant = CasMountGrant {
    name: "bootstrap:authenticated-xorb-read",
    subject: CasGrantSubject::AnyAuthenticated,
    kinds: &[CasMountObjectKind::Xorb],
    operations: &["open", "read"],
};

/// Enforcing authorizer for the live `GET /get_xorb/{hash}/` surface (#1094):
/// an explicit bootstrap grant list, default-deny, one audit event per decision.
///
/// This is plane #1 of #1091's R4b *compile-and-ratchet* MAC rollout: the route
/// flips to enforcing on day one behind [`AUTHENTICATED_XORB_READ`], making
/// today's implicit authenticated floor an explicit, auditable policy object
/// instead of a silent [`AllowAllCasAuthorizer`].
///
/// ## Ratchet path (grant breadth → 0)
///
/// The day-one grant is maximally broad on two axes: every authenticated
/// subject, every xorb address. Ratcheting is a deliberate, reviewable edit of
/// the grant list, sequenced with the MAC-activation prerequisites:
///
/// 1. **#699 lands provenance labels** — repo/compartment `security_label`s on
///    xorb-bearing manifests (carrier-(b)) give each request a real object
///    label. Grants can then narrow per-domain, then per-compartment.
/// 2. **#698 flows subject clearances** — authority-owned `Claims.clearance`
///    gives the caller a real `SecurityContext`; [`CasGrantSubject`] gains
///    clearance-bearing variants and the decision becomes dominance
///    (`subject.ctx ⊒ object.label`) per op.
/// 3. **Breadth 0** — with labels + clearances live, the grant list empties
///    (== [`DenyAllCasAuthorizer`]) and this type is retired in favor of the
///    MAC authorizer on the same [`CasMountAuthorizer`] seam.
///
/// That labeling/clearance work is **not** in scope here; only the seam and
/// the day-one grant are.
///
/// Audit: allows log at `debug`, denies at `warn` (denials are
/// security-relevant). Tamper-evident decision audit (#573,
/// `crate::mac::audit`) takes over when the MAC authorizer lands.
#[derive(Debug, Clone)]
pub struct BootstrapCasAuthorizer {
    grants: Vec<CasMountGrant>,
}

impl BootstrapCasAuthorizer {
    /// The day-one policy (#1094): exactly [`AUTHENTICATED_XORB_READ`].
    pub fn new() -> Self {
        Self {
            grants: vec![AUTHENTICATED_XORB_READ],
        }
    }

    /// An explicit grant list — the ratchet knob. Narrow the list as #699
    /// labels and #698 clearances land; an empty list denies everything
    /// (equivalent to [`DenyAllCasAuthorizer`]).
    pub fn with_grants(grants: Vec<CasMountGrant>) -> Self {
        Self { grants }
    }
}

impl Default for BootstrapCasAuthorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl CasMountAuthorizer for BootstrapCasAuthorizer {
    fn authorize(
        &self,
        caller: &Subject,
        request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError> {
        for grant in &self.grants {
            if grant.subject.accepts(caller)
                && grant.kinds.contains(&request.kind)
                && grant.operations.contains(&request.operation)
            {
                debug!(
                    subject = %caller,
                    grant = grant.name,
                    kind = ?request.kind,
                    address = request.address,
                    operation = request.operation,
                    "CAS bootstrap grant: allow"
                );
                return Ok(());
            }
        }
        warn!(
            subject = %caller,
            kind = ?request.kind,
            address = request.address,
            operation = request.operation,
            "CAS bootstrap grant: deny (no matching grant)"
        );
        Err(MountError::PermissionDenied(format!(
            "CAS {} {:?} {} denied for {}: no matching bootstrap grant (#1094)",
            request.operation, request.kind, request.address, caller
        )))
    }
}

/// Quota + lattice configuration for write-then-seal staging (#814).
///
/// Staging buffers are per-Subject and in-memory. Both quotas are hard floors:
/// a `Twrite` that would exceed either is rejected (the caller must commit or
/// abort to free the budget). [`StagingConfig::default`] is permissive enough
/// for ordinary model/adaptor ingest; production wiring (#1064/#1066 saga)
/// shrinks these from operator policy.
#[derive(Clone)]
pub struct StagingConfig {
    /// Max bytes a single staging slot may buffer.
    pub slot_quota_bytes: usize,
    /// Max total bytes across all of a subject's open (unsealed) slots.
    pub subject_quota_bytes: usize,
    /// Optional lattice for resolving compartment names in `label` ctl commands.
    /// When `None`, only level+assurance labels (no compartments) are accepted.
    pub lattice: Option<Arc<Lattice>>,
}

impl std::fmt::Debug for StagingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingConfig")
            .field("slot_quota_bytes", &self.slot_quota_bytes)
            .field("subject_quota_bytes", &self.subject_quota_bytes)
            .field("lattice", &self.lattice.is_some())
            .finish()
    }
}

impl Default for StagingConfig {
    fn default() -> Self {
        Self {
            slot_quota_bytes: DEFAULT_STAGING_SLOT_QUOTA,
            subject_quota_bytes: DEFAULT_STAGING_SUBJECT_QUOTA,
            lattice: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Staging slot + registry
// ─────────────────────────────────────────────────────────────────────────────

/// State machine for a staging slot. Stored as an atomic so the seal path can
/// CAS without holding a mutex across the async substrate `put`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SlotState {
    /// Accepting writes; not yet sealed.
    Open = 0,
    /// Seal in flight: a commit won the CAS and is mid-`put`. Concurrent
    /// commits/aborts/writes must reject.
    Sealing = 1,
    /// Sealed: bytes are in the substrate, `result` holds the CID.
    Sealed = 2,
    /// Voided (explicit abort, cluck-without-commit, or seal failure): bytes
    /// discarded. Terminal.
    Voided = 3,
}

impl SlotState {
    #[inline]
    const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Open,
            1 => Self::Sealing,
            2 => Self::Sealed,
            _ => Self::Voided,
        }
    }
}

/// Result of a seal, stashed for ctl readback (D-A: commit + CID readback).
type SealResult = Result<String, String>;

/// One in-flight write-then-seal ingest.
struct StagingSlot {
    owner: Subject,
    state: AtomicU8,
    buffer: Mutex<Vec<u8>>,
    /// Running LUB of caller-declared labels plus how many were declared.
    /// Folded incrementally (join is associative/commutative) so repeated
    /// `label` commands stay O(1) memory — declared labels must not bypass the
    /// byte quotas.
    declared_labels: Mutex<Option<(SecurityLabel, usize)>>,
    /// Signalled when a seal leaves `Sealing` (→ Sealed or Voided) so
    /// concurrent commits can suspend instead of spinning.
    seal_done: Notify,
    /// Set once the seal completes (Ok) or fails (Err). `Ok(cid)` after a
    /// successful commit; `Err(message)` after a seal failure that voided the
    /// slot. Read via ctl.
    result: Mutex<Option<SealResult>>,
    /// Outstanding fids (root/data/ctl) referencing this slot. The last cluck
    /// voids an Open slot and reaps the entry.
    outstanding: AtomicU64,
    slot_quota_bytes: usize,
}

impl StagingSlot {
    fn new(owner: Subject, slot_quota_bytes: usize) -> Self {
        Self {
            owner,
            state: AtomicU8::new(SlotState::Open as u8),
            buffer: Mutex::new(Vec::new()),
            declared_labels: Mutex::new(None),
            seal_done: Notify::new(),
            result: Mutex::new(None),
            outstanding: AtomicU64::new(0),
            slot_quota_bytes,
        }
    }

    #[inline]
    fn state(&self) -> SlotState {
        SlotState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Buffered byte count (the staged payload length so far).
    fn len(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Fold a caller-declared object label into the running LUB (joined at
    /// seal time). Rejected once the slot leaves `Open`: the state check runs
    /// under the label lock, and the seal reads the join under the same lock
    /// only after CAS'ing to `Sealing`, so a label accepted here is always
    /// visible to the seal.
    fn declare_label(&self, label: SecurityLabel) -> Result<(), MountError> {
        let mut declared = self.declared_labels.lock();
        if self.state() != SlotState::Open {
            return Err(MountError::InvalidArgument(
                "cannot declare labels on a slot that is no longer open".into(),
            ));
        }
        *declared = Some(match *declared {
            None => (label, 1),
            Some((current, n)) => (ifc_join(&[current, label]), n.saturating_add(1)),
        });
        Ok(())
    }

    /// Count of labels declared so far (for ctl status).
    fn label_count(&self) -> usize {
        self.declared_labels.lock().map_or(0, |(_, n)| n)
    }

    /// The IFC-joined object label to plumb into the substrate at seal. `None`
    /// is valid only while staging; the seal boundary rejects it before bytes
    /// are handed to CAS.
    fn joined_label(&self) -> Option<SecurityLabel> {
        self.declared_labels.lock().map(|(label, _)| label)
    }
}

struct StagingRegistry {
    slots: Mutex<HashMap<String, Arc<StagingSlot>>>,
    /// Per-subject total open-staged bytes (for the aggregate quota).
    subject_bytes: Mutex<HashMap<String, u64>>,
    next_id: AtomicU64,
}

impl StagingRegistry {
    fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            subject_bytes: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    fn mint(&self, owner: &Subject, slot_quota_bytes: usize) -> String {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let slot = Arc::new(StagingSlot::new(owner.clone(), slot_quota_bytes));
        // The create fid counts as one outstanding reference.
        slot.outstanding.store(1, Ordering::Release);
        self.slots.lock().insert(id.clone(), slot);
        id
    }

    fn get(&self, id: &str) -> Option<Arc<StagingSlot>> {
        self.slots.lock().get(id).cloned()
    }

    fn list_for(&self, owner: &Subject) -> Vec<String> {
        let slots = self.slots.lock();
        let mut ids: Vec<String> = slots
            .iter()
            .filter(|(_, s)| &s.owner == owner)
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        ids
    }

    /// Increment a slot's outstanding-fid count (a new walk/create referencing it).
    fn inc_outstanding(&self, slot: &Arc<StagingSlot>) {
        slot.outstanding.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement outstanding fids; on the last cluck, void an Open slot (bytes
    /// discarded — D-A "abort = cluck-without-commit") and reap it. Sealed
    /// slots are also reaped once unread (the content now lives in the
    /// substrate under its CID).
    fn cluck(&self, id: &str, slot: &Arc<StagingSlot>) {
        if slot.outstanding.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        // Last reference gone. CAS Open→Voided so an in-flight seal (Sealing)
        // is never clobbered; the byte count is read under the buffer lock
        // *after* winning the CAS so it pairs exactly with what was reserved.
        if slot
            .state
            .compare_exchange(
                SlotState::Open as u8,
                SlotState::Voided as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            // Cluck-without-commit → discard.
            let dropped = slot.buffer.lock().len() as u64;
            self.adjust_subject(&slot.owner, -(dropped as i64));
        }
        self.slots.lock().remove(id);
    }

    fn adjust_subject(&self, owner: &Subject, delta_bytes: i64) {
        let key = owner_key(owner);
        let mut map = self.subject_bytes.lock();
        // `get_mut` returns a short-lived borrow whose lifetime ends at the end
        // of this statement, so the subsequent `remove` on a zero total is
        // legal (no outstanding mutable borrow of `map`).
        let remaining = match map.get_mut(&key) {
            Some(v) => {
                if delta_bytes >= 0 {
                    *v = v.saturating_add(delta_bytes as u64);
                } else {
                    *v = v.saturating_sub((-delta_bytes) as u64);
                }
                *v
            }
            None => {
                if delta_bytes > 0 {
                    map.insert(key.clone(), delta_bytes as u64);
                    delta_bytes as u64
                } else {
                    0
                }
            }
        };
        if remaining == 0 {
            map.remove(&key);
        }
    }

    /// Atomically check-and-reserve `growth` bytes of aggregate staging budget
    /// under a single `subject_bytes` lock, so two concurrent writes cannot
    /// both observe the same usage and overshoot the quota. Returns the
    /// current usage on rejection.
    fn try_reserve(&self, owner: &Subject, growth: u64, limit: u64) -> Result<(), u64> {
        let key = owner_key(owner);
        let mut map = self.subject_bytes.lock();
        let current = map.get(&key).copied().unwrap_or(0);
        let new_total = current.saturating_add(growth);
        if new_total > limit {
            return Err(current);
        }
        if new_total > 0 {
            map.insert(key, new_total);
        }
        Ok(())
    }

    /// Current aggregate usage for a subject (test observability; production
    /// paths reserve via [`Self::try_reserve`]).
    #[cfg(test)]
    fn subject_usage(&self, owner: &Subject) -> u64 {
        self.subject_bytes
            .lock()
            .get(&owner_key(owner))
            .copied()
            .unwrap_or(0)
    }
}

fn owner_key(owner: &Subject) -> String {
    // Subjects validate to a safe charset (or are federated URLs validated at
    // JWT decode); either way this is a hash-map key, not a path component.
    owner.to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// CasMount
// ─────────────────────────────────────────────────────────────────────────────

/// CAS as a namespace mount.
///
/// Relative layout:
/// - `obj/{cid-or-legacy-merkle}` — full-file reconstruction.
/// - `xorb/{hash}` — raw xorb compatibility read.
/// - `stage/` — write-then-seal ingest (#814):
///   - `create stage/new` → `stage/<id>/` (per-Subject, quota'd).
///   - `stage/<id>/data` — bulk `Twrite`-at-offset target.
///   - `stage/<id>/ctl` — `commit` / `abort` / `label …`; read returns status
///     incl. the sealed CID.
#[derive(Clone)]
pub struct CasMount {
    substrate: CasSubstrate,
    domain: DedupDomain,
    authorizer: Arc<dyn CasMountAuthorizer>,
    staging: Arc<StagingRegistry>,
    staging_cfg: StagingConfig,
}

impl std::fmt::Debug for CasMount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CasMount")
            .field("domain", &self.domain)
            .field("staging_cfg", &self.staging_cfg)
            .finish_non_exhaustive()
    }
}

impl CasMount {
    /// Construct a fail-closed CAS mount over the given substrate/domain with
    /// default staging quotas.
    pub fn new(substrate: CasSubstrate, domain: DedupDomain) -> Self {
        Self::with_authorizer(substrate, domain, DenyAllCasAuthorizer)
    }

    /// Construct a fail-closed CAS mount over the default local dedup domain.
    pub fn local_default(substrate: CasSubstrate) -> Self {
        Self::new(substrate, DedupDomain::local_default())
    }

    /// Construct a CAS mount with an explicit authorizer and default staging.
    pub fn with_authorizer<A>(substrate: CasSubstrate, domain: DedupDomain, authorizer: A) -> Self
    where
        A: CasMountAuthorizer + 'static,
    {
        Self::with_authorizer_and_staging(substrate, domain, authorizer, StagingConfig::default())
    }

    /// Construct a CAS mount with an explicit authorizer and staging config.
    pub fn with_authorizer_and_staging<A>(
        substrate: CasSubstrate,
        domain: DedupDomain,
        authorizer: A,
        staging_cfg: StagingConfig,
    ) -> Self
    where
        A: CasMountAuthorizer + 'static,
    {
        Self {
            substrate,
            domain,
            authorizer: Arc::new(authorizer),
            staging: Arc::new(StagingRegistry::new()),
            staging_cfg,
        }
    }

    fn authorize(
        &self,
        caller: &Subject,
        kind: CasMountObjectKind,
        address: &str,
        operation: &'static str,
    ) -> Result<(), MountError> {
        self.authorizer.authorize(
            caller,
            CasMountAuthzRequest {
                kind,
                address,
                domain: &self.domain,
                operation,
            },
        )
    }

    async fn read_all(
        &self,
        kind: CasMountObjectKind,
        address: &str,
    ) -> Result<Vec<u8>, MountError> {
        match kind {
            CasMountObjectKind::Object => self.substrate.get(&self.domain, address).await,
            CasMountObjectKind::Xorb => self.substrate.read_xorb(&self.domain, address).await,
            CasMountObjectKind::Stage => {
                return Err(MountError::InvalidArgument(
                    "Stage kind is not a read target".into(),
                ));
            }
        }
        .map_err(map_cas_error)
    }

    /// Resolve a staging id to its slot, enforcing Subject ownership. Returns
    /// `NotFound` (not PermissionDenied) for foreign slots so existence does
    /// not leak across subjects.
    fn slot_for(&self, id: &str, caller: &Subject) -> Result<Arc<StagingSlot>, MountError> {
        let slot = self
            .staging
            .get(id)
            .ok_or_else(|| MountError::NotFound(format!("staging slot {id}")))?;
        if &slot.owner != caller {
            return Err(MountError::NotFound(format!("staging slot {id}")));
        }
        Ok(slot)
    }

    /// Run the seal: CAS Open→Sealing, take the buffered bytes, invoke the
    /// substrate, and record the CID (or void on failure). Idempotent — a
    /// second commit returns the same CID; concurrent commits serialize via CAS.
    async fn seal_slot(&self, slot: &Arc<StagingSlot>) -> Result<String, MountError> {
        // An unlabeled slot is a staging-only state. Reject before taking bytes
        // or transitioning to Sealing so the caller may label it and retry.
        if slot.joined_label().is_none() {
            return Err(MountError::PermissionDenied(
                "cannot seal an unlabeled staging slot; declare a label first".into(),
            ));
        }

        loop {
            match slot.state.compare_exchange(
                SlotState::Open as u8,
                SlotState::Sealing as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break, // won the commit race
                Err(actual) => match SlotState::from_u8(actual) {
                    SlotState::Sealed => {
                        // Idempotent commit: hand back the existing CID/error.
                        return match slot.result.lock().clone() {
                            Some(Ok(cid)) => Ok(cid),
                            Some(Err(e)) => Err(MountError::Io(e)),
                            None => Err(MountError::Io("sealed slot has no CID".into())),
                        };
                    }
                    SlotState::Sealing => {
                        // Another commit is mid-`put`. Suspend until it leaves
                        // Sealing (never spin — on a current-thread runtime a
                        // spin here would starve the in-flight seal). `enable`
                        // registers the waiter *before* the state re-check so
                        // a `notify_waiters` between the failed CAS and the
                        // await cannot be missed.
                        let notified = slot.seal_done.notified();
                        tokio::pin!(notified);
                        notified.as_mut().enable();
                        if slot.state() == SlotState::Sealing {
                            notified.await;
                        }
                        continue;
                    }
                    SlotState::Voided => {
                        // Surface a concurrent seal failure's message if any.
                        return match slot.result.lock().clone() {
                            Some(Err(e)) => Err(MountError::Io(e)),
                            _ => Err(MountError::InvalidArgument(
                                "cannot commit a voided staging slot".into(),
                            )),
                        };
                    }
                    SlotState::Open => continue,
                },
            }
        }

        // Won the CAS. Take the bytes + labels out from under their locks so we
        // don't hold them across the async substrate put.
        let bytes = std::mem::take(&mut *slot.buffer.lock());
        let joined = match slot.joined_label() {
            Some(label) => label,
            None => {
                // `declare_label` only adds labels and the state transition above
                // excludes concurrent declarations, so this is unreachable after
                // the preflight. Preserve fail-closed terminal behavior if that
                // invariant is ever changed.
                let msg = "cannot seal an unlabeled staging slot".to_owned();
                *slot.result.lock() = Some(Err(msg.clone()));
                slot.state.store(SlotState::Voided as u8, Ordering::Release);
                slot.seal_done.notify_waiters();
                self.staging
                    .adjust_subject(&slot.owner, -(bytes.len() as i64));
                return Err(MountError::PermissionDenied(msg));
            }
        };
        let put_result = self.substrate.put(&self.domain, &bytes, joined).await;

        match put_result {
            Ok(manifest) => {
                let cid = manifest.cid.clone();
                *slot.result.lock() = Some(Ok(cid.clone()));
                slot.state.store(SlotState::Sealed as u8, Ordering::Release);
                slot.seal_done.notify_waiters();
                // Bytes have left staging (now durable in the substrate).
                self.staging
                    .adjust_subject(&slot.owner, -(bytes.len() as i64));
                Ok(cid)
            }
            Err(err) => {
                let msg = err.to_string();
                *slot.result.lock() = Some(Err(msg.clone()));
                slot.state.store(SlotState::Voided as u8, Ordering::Release);
                slot.seal_done.notify_waiters();
                // Failure voids the slot; bytes are discarded.
                self.staging
                    .adjust_subject(&slot.owner, -(bytes.len() as i64));
                Err(map_cas_error(err))
            }
        }
    }

    /// Process one ctl command line. Returns bytes consumed (= line length).
    async fn ctl_command(
        &self,
        id: &str,
        slot: &Arc<StagingSlot>,
        line: &str,
        caller: &Subject,
    ) -> Result<(), MountError> {
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            None => Ok(()), // blank line
            Some("commit") => {
                self.authorize(caller, CasMountObjectKind::Stage, id, "stage:commit")?;
                self.seal_slot(slot).await.map(|_| ())
            }
            Some("abort") => {
                // CAS Open→Voided so an in-flight seal is never clobbered: a
                // commit that has already taken the bytes must not have its
                // Sealed result overwritten by a racing abort.
                match slot.state.compare_exchange(
                    SlotState::Open as u8,
                    SlotState::Voided as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        let dropped = {
                            let mut buf = slot.buffer.lock();
                            let n = buf.len() as u64;
                            buf.clear();
                            n
                        };
                        self.staging.adjust_subject(&slot.owner, -(dropped as i64));
                        Ok(())
                    }
                    Err(actual) => match SlotState::from_u8(actual) {
                        SlotState::Sealed => Err(MountError::InvalidArgument(
                            "cannot abort a sealed staging slot".into(),
                        )),
                        SlotState::Sealing => Err(MountError::InvalidArgument(
                            "cannot abort while a commit is sealing".into(),
                        )),
                        // Idempotent: aborting a voided slot is a no-op.
                        SlotState::Voided => Ok(()),
                        SlotState::Open => unreachable!("CAS from Open cannot fail with Open"),
                    },
                }
            }
            Some("label") => self.declare_label_cmd(slot, tokens.collect()),
            Some(cmd) => Err(MountError::InvalidArgument(format!(
                "unknown ctl command: {cmd}"
            ))),
        }
    }

    fn declare_label_cmd(
        &self,
        slot: &Arc<StagingSlot>,
        args: Vec<&str>,
    ) -> Result<(), MountError> {
        // label <level> <assurance> [compartment...]
        let mut it = args.into_iter();
        let level_s = it
            .next()
            .ok_or_else(|| MountError::InvalidArgument("label: missing level".into()))?;
        let assurance_s = it
            .next()
            .ok_or_else(|| MountError::InvalidArgument("label: missing assurance".into()))?;
        let level = parse_level(level_s)
            .ok_or_else(|| MountError::InvalidArgument(format!("unknown level: {level_s}")))?;
        let assurance = parse_assurance(assurance_s).ok_or_else(|| {
            MountError::InvalidArgument(format!("unknown assurance: {assurance_s}"))
        })?;
        let comps: Vec<&str> = it.collect();
        let label = if comps.is_empty() {
            SecurityLabel::new(level, assurance, CompartmentSet::EMPTY)
        } else {
            let lattice = self.staging_cfg.lattice.as_ref().ok_or_else(|| {
                MountError::InvalidArgument(
                    "label: compartments require a lattice on the mount".into(),
                )
            })?;
            let comps = comps
                .into_iter()
                .map(hyprstream_rpc::auth::mac::Compartment::new)
                .collect::<Vec<_>>();
            lattice
                .label(level, assurance, comps)
                .map_err(|e| MountError::InvalidArgument(format!("label: {e}")))?
        };
        slot.declare_label(label)
    }

    /// Render the ctl status text.
    fn ctl_status(&self, slot: &Arc<StagingSlot>) -> String {
        match slot.state() {
            SlotState::Open => {
                let labels = slot.label_count();
                format!(
                    "state=staging bytes={} labels={} quota={}\n",
                    slot.len(),
                    labels,
                    slot.slot_quota_bytes,
                )
            }
            SlotState::Sealing => "state=sealing\n".to_owned(),
            SlotState::Sealed => match slot.result.lock().clone() {
                Some(Ok(cid)) => format!("state=sealed cid={cid}\n"),
                Some(Err(e)) => format!("state=sealed error={e}\n"),
                None => "state=sealed\n".to_owned(),
            },
            SlotState::Voided => match slot.result.lock().clone() {
                Some(Err(e)) => format!("state=voided error={e}\n"),
                _ => "state=voided\n".to_owned(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CasFid {
    Root,
    ObjDir,
    XorbDir,
    StageDir,
    File {
        kind: CasMountObjectKind,
        address: String,
        opened: bool,
    },
    StageRoot {
        id: String,
        opened: bool,
    },
    /// `mode` is `Some(open-mode & 0x03)` once opened; reads require
    /// OREAD/ORDWR and writes OWRITE/ORDWR — a fid opened write-only must not
    /// read the staged bytes and vice versa.
    StageData {
        id: String,
        mode: Option<u8>,
    },
    StageCtl {
        id: String,
        mode: Option<u8>,
    },
}

/// Enforce that an opened stage fid's mode permits reading.
fn check_readable(mode: Option<u8>) -> Result<(), MountError> {
    match mode {
        None => Err(MountError::InvalidArgument("fid is not open".into())),
        Some(OREAD) | Some(ORDWR) => Ok(()),
        Some(_) => Err(MountError::PermissionDenied(
            "fid is not open for reading".into(),
        )),
    }
}

/// Enforce that an opened stage fid's mode permits writing.
fn check_writable(mode: Option<u8>) -> Result<(), MountError> {
    match mode {
        None => Err(MountError::InvalidArgument("fid is not open".into())),
        Some(OWRITE) | Some(ORDWR) => Ok(()),
        Some(_) => Err(MountError::PermissionDenied(
            "fid is not open for writing".into(),
        )),
    }
}

impl CasFid {
    /// If this fid pins a staging slot, return its id.
    fn slot_id(&self) -> Option<&str> {
        match self {
            CasFid::StageRoot { id, .. }
            | CasFid::StageData { id, .. }
            | CasFid::StageCtl { id, .. } => Some(id),
            _ => None,
        }
    }
}

#[async_trait]
impl Mount for CasMount {
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError> {
        let fid = match components {
            [] => CasFid::Root,
            ["obj"] => CasFid::ObjDir,
            ["xorb"] => CasFid::XorbDir,
            ["stage"] => CasFid::StageDir,
            ["obj", address] if !address.is_empty() => CasFid::File {
                kind: CasMountObjectKind::Object,
                address: (*address).to_owned(),
                opened: false,
            },
            ["xorb", hash] if !hash.is_empty() => CasFid::File {
                kind: CasMountObjectKind::Xorb,
                address: (*hash).to_owned(),
                opened: false,
            },
            ["stage", id] if !id.is_empty() => {
                let slot = self.slot_for(id, caller)?;
                self.staging.inc_outstanding(&slot);
                CasFid::StageRoot {
                    id: (*id).to_owned(),
                    opened: false,
                }
            }
            ["stage", id, "data"] if !id.is_empty() => {
                let slot = self.slot_for(id, caller)?;
                self.staging.inc_outstanding(&slot);
                CasFid::StageData {
                    id: (*id).to_owned(),
                    mode: None,
                }
            }
            ["stage", id, "ctl"] if !id.is_empty() => {
                let slot = self.slot_for(id, caller)?;
                self.staging.inc_outstanding(&slot);
                CasFid::StageCtl {
                    id: (*id).to_owned(),
                    mode: None,
                }
            }
            _ => return Err(MountError::NotFound(components.join("/"))),
        };
        Ok(Fid::new(fid))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError> {
        let inner = fid
            .downcast_mut::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;

        match inner {
            CasFid::Root | CasFid::ObjDir | CasFid::XorbDir | CasFid::StageDir => Ok(()),
            CasFid::File {
                kind,
                address,
                opened,
            } => {
                if mode & 0x03 != OREAD {
                    return Err(MountError::PermissionDenied(
                        "CAS obj/xorb files are read-only".into(),
                    ));
                }
                self.authorize(caller, *kind, address, "open")?;
                *opened = true;
                Ok(())
            }
            CasFid::StageRoot { opened, .. } => {
                *opened = true;
                Ok(())
            }
            CasFid::StageData { id, mode: m } | CasFid::StageCtl { id, mode: m } => {
                let _slot = self.slot_for(id, caller)?;
                *m = Some(mode & 0x03);
                Ok(())
            }
        }
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        match inner {
            CasFid::File {
                kind,
                address,
                opened,
            } => {
                if !opened {
                    return Err(MountError::InvalidArgument("fid is not open".into()));
                }
                self.authorize(caller, *kind, address, "read")?;
                let bytes = self.read_all(*kind, address).await?;
                Ok(slice_bytes(&bytes, offset, count))
            }
            CasFid::StageData { id, mode } => {
                check_readable(*mode)?;
                let slot = self.slot_for(id, caller)?;
                // Slice under the lock — never clone the whole staged buffer
                // (up to the slot quota) for a small positioned read.
                let bytes = slot.buffer.lock();
                Ok(slice_bytes(&bytes, offset, count))
            }
            CasFid::StageCtl { id, mode } => {
                check_readable(*mode)?;
                let slot = self.slot_for(id, caller)?;
                let status = self.ctl_status(&slot);
                Ok(slice_bytes(status.as_bytes(), offset, count))
            }
            CasFid::Root
            | CasFid::ObjDir
            | CasFid::XorbDir
            | CasFid::StageDir
            | CasFid::StageRoot { .. } => Err(MountError::IsDirectory(
                "cannot read a CAS directory as a file".into(),
            )),
        }
    }

    async fn write(
        &self,
        fid: &Fid,
        offset: u64,
        data: &[u8],
        caller: &Subject,
    ) -> Result<u32, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        match inner {
            CasFid::StageData { id, mode } => {
                check_writable(*mode)?;
                let slot = self.slot_for(id, caller)?;
                self.authorize(caller, CasMountObjectKind::Stage, id, "stage:write")?;

                let off = usize::try_from(offset)
                    .map_err(|_| MountError::InvalidArgument("write offset overflow".into()))?;
                let end = off
                    .checked_add(data.len())
                    .ok_or_else(|| MountError::InvalidArgument("write length overflow".into()))?;

                // Per-slot quota.
                if end > slot.slot_quota_bytes {
                    return Err(MountError::PermissionDenied(format!(
                        "staging slot quota exceeded: {end} > {}",
                        slot.slot_quota_bytes
                    )));
                }

                let mut buf = slot.buffer.lock();
                // Check the state *under the buffer lock*: the seal takes the
                // buffer under this lock only after CAS'ing to Sealing, and
                // abort/cluck clear it under this lock only after CAS'ing to
                // Voided — so Open observed here means these bytes are either
                // included in the seal or released with the void, never
                // written to a buffer that was already taken.
                if slot.state() != SlotState::Open {
                    return Err(MountError::InvalidArgument(format!(
                        "staging slot {id} is not open for writes (state={:?})",
                        slot.state()
                    )));
                }

                // Per-Subject aggregate quota: atomically check-and-reserve
                // the growth delta so concurrent writes can't both pass the
                // check and overshoot.
                let prev_len = buf.len();
                let growth = end.saturating_sub(prev_len);
                if growth > 0 {
                    if let Err(usage) = self.staging.try_reserve(
                        caller,
                        growth as u64,
                        self.staging_cfg.subject_quota_bytes as u64,
                    ) {
                        return Err(MountError::PermissionDenied(format!(
                            "subject staging quota exceeded: {} + {growth} > {}",
                            usage, self.staging_cfg.subject_quota_bytes
                        )));
                    }
                }

                if end > buf.len() {
                    buf.resize(off, 0); // sparse gap zero-fill
                    buf.extend_from_slice(data);
                } else {
                    buf[off..end].copy_from_slice(data);
                }
                drop(buf);
                Ok(data.len() as u32)
            }
            CasFid::StageCtl { id, mode } => {
                check_writable(*mode)?;
                let slot = self.slot_for(id, caller)?;
                let line = std::str::from_utf8(data)
                    .map_err(|_| MountError::InvalidArgument("ctl command is not UTF-8".into()))?;
                // Accept a single command per write (trailing newline trimmed).
                self.ctl_command(id, &slot, line.trim_end_matches(['\n', '\r']), caller)
                    .await?;
                Ok(data.len() as u32)
            }
            // Read-only object/xorb files and directories reject writes.
            _ => Err(MountError::NotSupported(
                "CAS mount writes go through stage/<id>/{data,ctl} (#814)".into(),
            )),
        }
    }

    async fn create(
        &self,
        fid: &mut Fid,
        _name: &str,
        _perm: u32,
        mode: u8,
        caller: &Subject,
    ) -> Result<Stat, MountError> {
        // `create` is honored on the mount root and on `stage/`. The caller's
        // name (conventionally "new") is ignored: the server mints a fresh
        // staging id so two concurrent ingests never collide.
        let inner = fid
            .downcast_mut::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        let is_stage_parent = matches!(inner, CasFid::Root | CasFid::StageDir);
        if !is_stage_parent {
            return Err(MountError::NotSupported(
                "create only supported on the CAS root or stage/ (#814)".into(),
            ));
        }

        self.authorize(caller, CasMountObjectKind::Stage, "new", "stage:create")?;

        let id = self.staging.mint(caller, self.staging_cfg.slot_quota_bytes);
        *inner = CasFid::StageRoot {
            id: id.clone(),
            opened: true,
        };
        // `mode` is the open mode the caller wants on the new staging dir; the
        // dir itself is walked into for data/ctl, so we don't enforce it here
        // beyond accepting the create. Mask off truncate/close bits.
        let _ = mode & !(OTRUNC);
        Ok(staging_dir_stat(&id))
    }

    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        match inner {
            CasFid::Root => Ok(vec![
                dir_entry("obj"),
                dir_entry("xorb"),
                dir_entry("stage"),
            ]),
            CasFid::ObjDir | CasFid::XorbDir => Ok(Vec::new()),
            CasFid::StageDir => {
                let ids = self.staging.list_for(caller);
                Ok(ids
                    .into_iter()
                    .map(|id| DirEntry {
                        name: id.clone(),
                        is_dir: true,
                        size: 0,
                        stat: Some(staging_dir_stat(&id)),
                    })
                    .collect())
            }
            CasFid::StageRoot { id, .. } => Ok(vec![
                DirEntry {
                    name: "data".to_owned(),
                    is_dir: false,
                    size: 0,
                    // The qid path must include the staging id so 9P clients do
                    // not alias `stage/<a>/data` with `stage/<b>/data` in a
                    // qid-keyed cache (qid is 9P's file identity).
                    stat: Some(staging_file_stat(id, "data", 0)),
                },
                DirEntry {
                    name: "ctl".to_owned(),
                    is_dir: false,
                    size: 0,
                    stat: Some(staging_file_stat(id, "ctl", 0)),
                },
            ]),
            other => Err(MountError::NotDirectory(
                other.slot_id().unwrap_or("file").to_owned(),
            )),
        }
    }

    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        match inner {
            CasFid::Root => Ok(dir_stat("", 0)),
            CasFid::ObjDir => Ok(dir_stat("obj", 1)),
            CasFid::XorbDir => Ok(dir_stat("xorb", 2)),
            CasFid::StageDir => Ok(dir_stat("stage", 3)),
            CasFid::File { kind, address, .. } => {
                self.authorize(caller, *kind, address, "stat")?;
                let len = self.read_all(*kind, address).await?.len() as u64;
                Ok(file_stat(address, len))
            }
            CasFid::StageRoot { id, .. } => Ok(staging_dir_stat(id)),
            CasFid::StageData { id, .. } => {
                let slot = self.slot_for(id, caller)?;
                Ok(staging_file_stat(id, "data", slot.len() as u64))
            }
            CasFid::StageCtl { id, .. } => {
                let slot = self.slot_for(id, caller)?;
                let len = self.ctl_status(&slot).len() as u64;
                Ok(staging_file_stat(id, "ctl", len))
            }
        }
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        let Ok(inner) = fid.downcast_into::<CasFid>() else {
            return;
        };
        if let Some(id) = inner.slot_id() {
            if let Some(slot) = self.staging.get(id) {
                self.staging.cluck(id, &slot);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn parse_level(s: &str) -> Option<Level> {
    match s.to_ascii_lowercase().as_str() {
        "public" => Some(Level::Public),
        "internal" => Some(Level::Internal),
        "confidential" => Some(Level::Confidential),
        "secret" => Some(Level::Secret),
        _ => None,
    }
}

fn parse_assurance(s: &str) -> Option<Assurance> {
    match s.to_ascii_lowercase().as_str() {
        "classical" => Some(Assurance::Classical),
        "pqhybrid" | "pq_hybrid" | "pq-hybrid" | "hybrid" => Some(Assurance::PqHybrid),
        _ => None,
    }
}

fn dir_entry(name: &str) -> DirEntry {
    DirEntry {
        name: name.to_owned(),
        is_dir: true,
        size: 0,
        stat: Some(dir_stat(name, hash_path(name))),
    }
}

fn dir_stat(name: &str, path: u64) -> Stat {
    Stat {
        qtype: QTDIR,
        version: 1,
        path,
        size: 0,
        name: name.to_owned(),
        mtime: 0,
    }
}

fn file_stat(name: &str, size: u64) -> Stat {
    Stat {
        qtype: QTFILE,
        version: 1,
        path: hash_path(name),
        size,
        name: name.to_owned(),
        mtime: 0,
    }
}

fn staging_dir_stat(id: &str) -> Stat {
    Stat {
        qtype: QTDIR,
        version: 1,
        path: hash_path(&format!("stage/{id}")),
        size: 0,
        name: id.to_owned(),
        mtime: 0,
    }
}

fn staging_file_stat(id: &str, name: &str, size: u64) -> Stat {
    Stat {
        qtype: QTFILE,
        version: 1,
        // The qid path includes the staging id so two slots' `data` (or `ctl`)
        // files never share a qid — qid is 9P's file identity, and a qid-keyed
        // client cache would otherwise alias `stage/<a>/data` with
        // `stage/<b>/data`. (Advisory hint only — see the qid-soundness note on
        // `hyprstream_vfs::Stat`; authz never keys on this.)
        path: hash_path(&format!("stage/{id}/{name}")),
        size,
        name: name.to_owned(),
        mtime: 0,
    }
}

fn slice_bytes(bytes: &[u8], offset: u64, count: u32) -> Vec<u8> {
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    let end = start.saturating_add(count as usize).min(bytes.len());
    bytes[start..end].to_vec()
}

fn map_cas_error(err: CasError) -> MountError {
    match err {
        CasError::Store(StoreError::NotFound(s)) => MountError::NotFound(s),
        CasError::Store(StoreError::InvalidHash(s)) | CasError::Cid(s) | CasError::Hex(s) => {
            MountError::InvalidArgument(s)
        }
        CasError::Manifest(s) => MountError::InvalidArgument(s),
        CasError::Store(e) => MountError::Io(e.to_string()),
        CasError::UnsupportedIngestAlgorithm(algo) => {
            MountError::InvalidArgument(format!("unsupported CAS algorithm: {algo:?}"))
        }
    }
}

fn hash_path(path: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in path.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_vfs::{ORDWR, OWRITE};
    use parking_lot::Mutex;

    fn label() -> SecurityLabel {
        SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY)
    }

    async fn declare_test_label(mount: &CasMount, ctl: &Fid, caller: &Subject) {
        mount
            .write(ctl, 0, b"label internal classical\n", caller)
            .await
            .unwrap();
    }

    #[derive(Default)]
    struct RecordingAuthorizer {
        deny: bool,
        calls: Mutex<Vec<(String, CasMountObjectKind, String, &'static str)>>,
    }

    impl RecordingAuthorizer {
        fn calls(&self) -> Vec<(String, CasMountObjectKind, String, &'static str)> {
            self.calls.lock().clone()
        }
    }

    impl CasMountAuthorizer for std::sync::Arc<RecordingAuthorizer> {
        fn authorize(
            &self,
            caller: &Subject,
            request: CasMountAuthzRequest<'_>,
        ) -> Result<(), MountError> {
            self.calls.lock().push((
                caller.to_string(),
                request.kind,
                request.address.to_owned(),
                request.operation,
            ));
            if self.deny {
                Err(MountError::PermissionDenied("denied by test".into()))
            } else {
                Ok(())
            }
        }
    }

    // ── read-side (existing) ───────────────────────────────────────────────

    #[tokio::test]
    async fn object_read_requires_authorizer_and_slices() {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let manifest = substrate
            .put(&domain, b"hello-cas-mount", label())
            .await
            .unwrap();
        let authz = std::sync::Arc::new(RecordingAuthorizer::default());
        let mount = CasMount::with_authorizer(substrate, domain, authz.clone());
        let caller = Subject::new("alice");

        let mut fid = mount.walk(&["obj", &manifest.cid], &caller).await.unwrap();
        mount.open(&mut fid, OREAD, &caller).await.unwrap();
        let out = mount.read(&fid, 6, 3, &caller).await.unwrap();

        assert_eq!(out, b"cas");
        assert_eq!(
            authz.calls(),
            vec![
                (
                    "alice".to_owned(),
                    CasMountObjectKind::Object,
                    manifest.cid.clone(),
                    "open"
                ),
                (
                    "alice".to_owned(),
                    CasMountObjectKind::Object,
                    manifest.cid,
                    "read"
                ),
            ]
        );
    }

    #[tokio::test]
    async fn default_authorizer_denies_even_when_hash_exists() {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let manifest = substrate
            .put(&domain, b"secret bytes", label())
            .await
            .unwrap();
        let mount = CasMount::new(substrate, domain);
        let caller = Subject::new("bob");

        let mut fid = mount
            .walk(&["obj", &manifest.merkle], &caller)
            .await
            .unwrap();
        let err = mount.open(&mut fid, OREAD, &caller).await.unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn xorb_read_uses_xorb_authorization_class() {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let manifest = substrate
            .put(&domain, b"xorb-backed-payload", label())
            .await
            .unwrap();
        let xorb = manifest.xorb_hashes.first().expect("xorb hash").clone();
        let authz = std::sync::Arc::new(RecordingAuthorizer::default());
        let mount = CasMount::with_authorizer(substrate, domain, authz.clone());
        let caller = Subject::new("alice");

        let mut fid = mount.walk(&["xorb", &xorb], &caller).await.unwrap();
        mount.open(&mut fid, OREAD, &caller).await.unwrap();
        let out = mount.read(&fid, 0, 1024, &caller).await.unwrap();

        assert!(!out.is_empty());
        assert_eq!(authz.calls()[0].1, CasMountObjectKind::Xorb);
        assert_eq!(authz.calls()[1].1, CasMountObjectKind::Xorb);
    }

    #[tokio::test]
    async fn root_lists_non_enumerating_cas_dirs() {
        let mount = CasMount::with_authorizer(
            CasSubstrate::new(tempfile::tempdir().unwrap().path()),
            DedupDomain::local_default(),
            AllowAllCasAuthorizer,
        );
        let caller = Subject::new("alice");
        let fid = mount.walk(&[], &caller).await.unwrap();
        let entries = mount.readdir(&fid, &caller).await.unwrap();
        let names: Vec<_> = entries.into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec!["obj", "xorb", "stage"]);
    }

    // ── bootstrap-grant authorizer (#1094) ─────────────────────────────────

    /// Build a substrate holding one xorb and return it with the xorb hash.
    async fn substrate_with_xorb(bytes: &[u8]) -> (tempfile::TempDir, CasSubstrate, String) {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let manifest = substrate
            .put(&DedupDomain::local_default(), bytes, label())
            .await
            .unwrap();
        let xorb = manifest.xorb_hashes.first().expect("xorb hash").clone();
        (dir, substrate, xorb)
    }

    #[tokio::test]
    async fn bootstrap_grant_allows_authenticated_xorb_read() {
        let (_dir, substrate, xorb) = substrate_with_xorb(b"bootstrap-xorb").await;
        let mount = CasMount::with_authorizer(
            substrate,
            DedupDomain::local_default(),
            BootstrapCasAuthorizer::new(),
        );
        let caller = Subject::new("alice");

        let mut fid = mount.walk(&["xorb", &xorb], &caller).await.unwrap();
        mount.open(&mut fid, OREAD, &caller).await.unwrap();
        let out = mount.read(&fid, 0, 1024, &caller).await.unwrap();

        assert!(!out.is_empty());
    }

    #[tokio::test]
    async fn bootstrap_grant_denies_anonymous_xorb_read() {
        let (_dir, substrate, xorb) = substrate_with_xorb(b"bootstrap-xorb").await;
        let mount = CasMount::with_authorizer(
            substrate,
            DedupDomain::local_default(),
            BootstrapCasAuthorizer::new(),
        );
        let caller = Subject::anonymous();

        let mut fid = mount.walk(&["xorb", &xorb], &caller).await.unwrap();
        let err = mount.open(&mut fid, OREAD, &caller).await.unwrap_err();

        assert!(matches!(err, MountError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn bootstrap_grant_denies_obj_reads_and_stage_ops() {
        let (_dir, substrate, xorb) = substrate_with_xorb(b"bootstrap-xorb").await;
        let manifest = substrate
            .put(&DedupDomain::local_default(), b"obj-bytes", label())
            .await
            .unwrap();
        let mount = CasMount::with_authorizer(
            substrate,
            DedupDomain::local_default(),
            BootstrapCasAuthorizer::new(),
        );
        let caller = Subject::new("alice");

        // obj/* reads are outside the day-one grant.
        let mut fid = mount.walk(&["obj", &manifest.cid], &caller).await.unwrap();
        let err = mount.open(&mut fid, OREAD, &caller).await.unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));

        // stat on a xorb is outside the grant (open + read only).
        let fid = mount.walk(&["xorb", &xorb], &caller).await.unwrap();
        let err = mount.stat(&fid, &caller).await.unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));

        // stage/* ingest is outside the grant.
        let mut root = mount.walk(&["stage"], &caller).await.unwrap();
        let err = mount
            .create(&mut root, "new", 0o600, OWRITE, &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
    }

    #[test]
    fn bootstrap_grant_list_ratchets_to_deny_all() {
        // The ratchet terminus: an empty grant list denies even the day-one
        // xorb read, equivalent to DenyAllCasAuthorizer.
        let authz = BootstrapCasAuthorizer::with_grants(Vec::new());
        let domain = DedupDomain::local_default();
        let request = CasMountAuthzRequest {
            kind: CasMountObjectKind::Xorb,
            address: "aa00",
            domain: &domain,
            operation: "read",
        };
        let err = authz
            .authorize(&Subject::new("alice"), request)
            .unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
    }

    #[test]
    fn bootstrap_day_one_grant_matches_exactly_xorb_open_read() {
        let grant = AUTHENTICATED_XORB_READ;
        assert_eq!(grant.subject, CasGrantSubject::AnyAuthenticated);
        assert_eq!(grant.kinds, &[CasMountObjectKind::Xorb]);
        assert_eq!(grant.operations, &["open", "read"]);
    }

    // ── write-then-seal (#814) ─────────────────────────────────────────────

    /// Build a mount with small quotas so overflow paths are testable without
    /// allocating hundreds of MiB.
    fn staging_mount(substrate: CasSubstrate, slot_quota: usize, subject_quota: usize) -> CasMount {
        CasMount::with_authorizer_and_staging(
            substrate,
            DedupDomain::local_default(),
            AllowAllCasAuthorizer,
            StagingConfig {
                slot_quota_bytes: slot_quota,
                subject_quota_bytes: subject_quota,
                lattice: None,
            },
        )
    }

    /// Create a staging slot and return `(id, root_fid)`. The root fid pins the
    /// slot (one outstanding reference); tests must `clunk` it — along with any
    /// walked data/ctl fids — for the slot to be reaped.
    async fn create_stage(mount: &CasMount, caller: &Subject) -> (String, Fid) {
        let mut root = mount.walk(&["stage"], caller).await.unwrap();
        let stat = mount
            .create(&mut root, "new", 0o600, OWRITE, caller)
            .await
            .unwrap();
        (stat.name, root)
    }

    async fn open_data_ctl(mount: &CasMount, id: &str, caller: &Subject) -> (Fid, Fid) {
        let mut data = mount.walk(&["stage", id, "data"], caller).await.unwrap();
        let mut ctl = mount.walk(&["stage", id, "ctl"], caller).await.unwrap();
        mount.open(&mut data, ORDWR, caller).await.unwrap();
        mount.open(&mut ctl, ORDWR, caller).await.unwrap();
        (data, ctl)
    }

    /// Read the ctl status text as an owned `String` (avoids temporary-value
    /// borrow errors from `from_utf8_lossy(&mount.read(...).await.unwrap())`).
    async fn ctl_str(mount: &CasMount, ctl: &Fid, caller: &Subject) -> String {
        String::from_utf8_lossy(&mount.read(ctl, 0, 1024, caller).await.unwrap()).into_owned()
    }

    #[tokio::test]
    async fn create_then_seal_returns_cid_readable_via_ctl() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;

        // Pre-seal status.
        let pre = mount.read(&ctl, 0, 1024, &caller).await.unwrap();
        assert!(String::from_utf8_lossy(&pre).contains("state=staging"));

        // Two positioned writes (resumable by offset).
        mount.write(&data, 0, b"hello ", &caller).await.unwrap();
        mount.write(&data, 6, b"world", &caller).await.unwrap();

        // Seal.
        declare_test_label(&mount, &ctl, &caller).await;
        let n = mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        assert_eq!(n, 7);

        let post = mount.read(&ctl, 0, 1024, &caller).await.unwrap();
        let status = String::from_utf8_lossy(&post);
        assert!(status.contains("state=sealed "), "got: {status}");
        let sealed_cid = status.split("cid=").nth(1).unwrap().trim().to_owned();
        assert!(!sealed_cid.is_empty());

        // The bytes are reconstructable by CID through the read side.
        let got = mount
            .substrate
            .get(&mount.domain, &sealed_cid)
            .await
            .unwrap();
        assert_eq!(got, b"hello world");

        // Cluck everything; sealed content remains in the substrate.
        mount.clunk(data, &caller).await;
        // Read the CID one more time before clucking ctl.
        let _ = mount.read(&ctl, 0, 1024, &caller).await.unwrap();
        mount.clunk(ctl, &caller).await;
        assert_eq!(
            mount
                .substrate
                .get(&mount.domain, &sealed_cid)
                .await
                .unwrap(),
            b"hello world"
        );
    }

    #[tokio::test]
    async fn sparse_write_zero_fills_gap() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;

        // Write at offset 5, leaving bytes 0..5 as a sparse gap.
        mount.write(&data, 5, b"xyz", &caller).await.unwrap();
        let buf = mount.read(&data, 0, 64, &caller).await.unwrap();
        assert_eq!(&buf[5..8], b"xyz");
        assert_eq!(&buf[..5], &[0, 0, 0, 0, 0]);

        declare_test_label(&mount, &ctl, &caller).await;
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        let status = ctl_str(&mount, &ctl, &caller).await;
        let cid = status.split("cid=").nth(1).unwrap().trim().to_owned();
        let reconstructed = mount.substrate.get(&mount.domain, &cid).await.unwrap();
        assert_eq!(reconstructed.len(), 8);
        assert_eq!(&reconstructed[..5], &[0, 0, 0, 0, 0]);
        assert_eq!(&reconstructed[5..], b"xyz");
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn cluck_without_commit_voids_and_discards() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount
            .write(&data, 0, b"never sealed", &caller)
            .await
            .unwrap();

        // Cluck every fid (root + data + ctl) without committing → the slot is
        // reaped (outstanding refcount hits zero while still Open).
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
        mount.clunk(root, &caller).await;

        // Walking the id again fails (NotFound).
        let err = mount
            .walk(&["stage", &id, "data"], &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::NotFound(_)));

        // And the stage dir no longer lists it.
        let stage = mount.walk(&["stage"], &caller).await.unwrap();
        let entries = mount.readdir(&stage, &caller).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn explicit_abort_discards_and_releases_quota() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 1 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount.write(&data, 0, &[0u8; 4096], &caller).await.unwrap();
        assert_eq!(mount.staging.subject_usage(&caller), 4096);

        // Abort via ctl.
        mount.write(&ctl, 0, b"abort\n", &caller).await.unwrap();
        let status = ctl_str(&mount, &ctl, &caller).await;
        assert!(status.contains("state=voided"), "got: {status}");

        // Subject quota released.
        assert_eq!(mount.staging.subject_usage(&caller), 0);

        // Cannot commit a voided slot.
        let err = mount
            .write(&ctl, 0, b"commit\n", &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::InvalidArgument(_)));
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn commit_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount.write(&data, 0, b"payload", &caller).await.unwrap();

        declare_test_label(&mount, &ctl, &caller).await;
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        let cid1 = {
            let s = ctl_str(&mount, &ctl, &caller).await;
            s.split("cid=").nth(1).unwrap().trim().to_owned()
        };

        // A second commit returns the same CID without re-ingesting.
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        let cid2 = {
            let s = ctl_str(&mount, &ctl, &caller).await;
            s.split("cid=").nth(1).unwrap().trim().to_owned()
        };
        assert_eq!(cid1, cid2);
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn slot_quota_rejects_oversize_write() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 8, 64);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        let err = mount
            .write(&data, 0, &[0u8; 16], &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn subject_quota_aggregates_across_slots() {
        let dir = tempfile::tempdir().unwrap();
        // slot=8, subject=12 → two slots can't both fill.
        let mount = staging_mount(CasSubstrate::new(dir.path()), 8, 12);
        let caller = Subject::new("alice");
        let (id1, _root1) = create_stage(&mount, &caller).await;
        let (id2, _root2) = create_stage(&mount, &caller).await;
        let (d1, c1) = open_data_ctl(&mount, &id1, &caller).await;
        let (d2, c2) = open_data_ctl(&mount, &id2, &caller).await;

        mount.write(&d1, 0, &[0u8; 8], &caller).await.unwrap();
        assert_eq!(mount.staging.subject_usage(&caller), 8);
        // Second slot's 8 bytes would bring the total to 16 > 12.
        let err = mount.write(&d2, 0, &[0u8; 8], &caller).await.unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
        assert_eq!(mount.staging.subject_usage(&caller), 8);
        for f in [d1, c1, d2, c2] {
            mount.clunk(f, &caller).await;
        }
    }

    #[tokio::test]
    async fn cross_subject_isolation_not_found_for_foreign_slot() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let alice = Subject::new("alice");
        let bob = Subject::new("bob");
        let (id, _root) = create_stage(&mount, &alice).await;

        // Bob cannot see alice's slot: NotFound (no existence leak).
        let err = mount.walk(&["stage", &id, "data"], &bob).await.unwrap_err();
        assert!(matches!(err, MountError::NotFound(_)));

        // And bob's stage listing is empty.
        let bob_stage = mount.walk(&["stage"], &bob).await.unwrap();
        assert!(mount.readdir(&bob_stage, &bob).await.unwrap().is_empty());

        // Alice still sees it.
        let alice_stage = mount.walk(&["stage"], &alice).await.unwrap();
        let entries = mount.readdir(&alice_stage, &alice).await.unwrap();
        assert_eq!(entries.len(), 1);

        let (d, c) = open_data_ctl(&mount, &id, &alice).await;
        mount.clunk(d, &alice).await;
        mount.clunk(c, &alice).await;
    }

    #[tokio::test]
    async fn seal_joins_declared_labels_into_manifest() {
        use hyprstream_rpc::auth::mac::{Compartment, LatticeVersion};
        let dir = tempfile::tempdir().unwrap();
        let lattice = Arc::new(Lattice::new(
            LatticeVersion(1),
            [Compartment::new("pii"), Compartment::new("finance")],
        ));
        let mount = CasMount::with_authorizer_and_staging(
            CasSubstrate::new(dir.path()),
            DedupDomain::local_default(),
            AllowAllCasAuthorizer,
            StagingConfig {
                slot_quota_bytes: 1 << 20,
                subject_quota_bytes: 4 << 20,
                lattice: Some(lattice.clone()),
            },
        );
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount
            .write(&data, 0, b"labeled-bytes", &caller)
            .await
            .unwrap();

        // Declare two labels; the seal must join them (LUB).
        mount
            .write(&ctl, 0, b"label internal classical pii\n", &caller)
            .await
            .unwrap();
        mount
            .write(&ctl, 0, b"label confidential pqhybrid finance\n", &caller)
            .await
            .unwrap();

        let status = ctl_str(&mount, &ctl, &caller).await;
        assert!(status.contains("labels=2"), "got: {status}");

        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        let cid = {
            let s = ctl_str(&mount, &ctl, &caller).await;
            s.split("cid=").nth(1).unwrap().trim().to_owned()
        };

        let pii = lattice
            .label(
                Level::Internal,
                Assurance::Classical,
                [Compartment::new("pii")],
            )
            .unwrap();
        let finance = lattice
            .label(
                Level::Confidential,
                Assurance::PqHybrid,
                [Compartment::new("finance")],
            )
            .unwrap();
        let joined = ifc_join(&[pii, finance]);
        let manifest = mount.substrate.manifest(&mount.domain, &cid).unwrap();
        assert_eq!(manifest.security_label, joined);
        assert!(manifest.security_label.can_access(&pii));
        assert!(manifest.security_label.can_access(&finance));
        assert!(manifest.security_label.compartments.contains(0));
        assert!(manifest.security_label.compartments.contains(1));
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn seal_without_a_label_fails_closed_before_cas_ingest() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount
            .write(&data, 0, b"must-not-seal", &caller)
            .await
            .unwrap();

        let err = mount
            .write(&ctl, 0, b"commit\n", &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
        assert!(
            ctl_str(&mount, &ctl, &caller)
                .await
                .contains("state=staging")
        );

        declare_test_label(&mount, &ctl, &caller).await;
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn create_rejected_under_obj_or_xorb() {
        let mount = staging_mount(
            CasSubstrate::new(tempfile::tempdir().unwrap().path()),
            1 << 20,
            4 << 20,
        );
        let caller = Subject::new("alice");
        let mut obj_dir = mount.walk(&["obj"], &caller).await.unwrap();
        let err = mount
            .create(&mut obj_dir, "new", 0o600, OWRITE, &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::NotSupported(_)));
    }

    #[tokio::test]
    async fn denyall_authorizer_blocks_create_and_commit() {
        let dir = tempfile::tempdir().unwrap();
        let mount = CasMount::new(CasSubstrate::new(dir.path()), DedupDomain::local_default());
        let caller = Subject::new("alice");
        let mut stage = mount.walk(&["stage"], &caller).await.unwrap();
        let err = mount
            .create(&mut stage, "new", 0o600, OWRITE, &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn unknown_ctl_command_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (_data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        let err = mount
            .write(&ctl, 0, b"frobnicate\n", &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::InvalidArgument(_)));
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn stage_fid_open_modes_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;

        // A write-only data fid must not read staged bytes.
        let mut wo = mount.walk(&["stage", &id, "data"], &caller).await.unwrap();
        mount.open(&mut wo, OWRITE, &caller).await.unwrap();
        mount.write(&wo, 0, b"secret", &caller).await.unwrap();
        let err = mount.read(&wo, 0, 64, &caller).await.unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));

        // A read-only data fid must not write.
        let mut ro = mount.walk(&["stage", &id, "data"], &caller).await.unwrap();
        mount.open(&mut ro, OREAD, &caller).await.unwrap();
        let err = mount.write(&ro, 0, b"x", &caller).await.unwrap_err();
        assert!(matches!(err, MountError::PermissionDenied(_)));
        assert_eq!(mount.read(&ro, 0, 64, &caller).await.unwrap(), b"secret");

        // An unopened fid is rejected outright.
        let unopened = mount.walk(&["stage", &id, "ctl"], &caller).await.unwrap();
        let err = mount.read(&unopened, 0, 64, &caller).await.unwrap_err();
        assert!(matches!(err, MountError::InvalidArgument(_)));

        for f in [wo, ro, unopened] {
            mount.clunk(f, &caller).await;
        }
    }

    #[tokio::test]
    async fn abort_after_seal_is_rejected_and_abort_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");

        // Abort after a successful commit must not clobber the sealed state.
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount
            .write(&data, 0, b"sealed-bytes", &caller)
            .await
            .unwrap();
        declare_test_label(&mount, &ctl, &caller).await;
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        let err = mount.write(&ctl, 0, b"abort\n", &caller).await.unwrap_err();
        assert!(matches!(err, MountError::InvalidArgument(_)));
        let status = ctl_str(&mount, &ctl, &caller).await;
        assert!(status.contains("state=sealed"), "got: {status}");
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;

        // A second abort of a voided slot is an idempotent no-op.
        let (id2, _root2) = create_stage(&mount, &caller).await;
        let (d2, c2) = open_data_ctl(&mount, &id2, &caller).await;
        mount.write(&c2, 0, b"abort\n", &caller).await.unwrap();
        mount.write(&c2, 0, b"abort\n", &caller).await.unwrap();
        let status = ctl_str(&mount, &c2, &caller).await;
        assert!(status.contains("state=voided"), "got: {status}");
        mount.clunk(d2, &caller).await;
        mount.clunk(c2, &caller).await;
    }

    #[tokio::test]
    async fn label_after_seal_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount.write(&data, 0, b"bytes", &caller).await.unwrap();
        declare_test_label(&mount, &ctl, &caller).await;
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();
        let err = mount
            .write(&ctl, 0, b"label secret pqhybrid\n", &caller)
            .await
            .unwrap_err();
        assert!(matches!(err, MountError::InvalidArgument(_)));
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn concurrent_commits_converge_on_one_cid() {
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount
            .write(&data, 0, b"raced-bytes", &caller)
            .await
            .unwrap();
        declare_test_label(&mount, &ctl, &caller).await;

        // Two commits racing on the same slot: exactly one seals, the other
        // waits on the seal (never spinning) and converges on the same CID.
        let (a, b) = tokio::join!(
            mount.write(&ctl, 0, b"commit\n", &caller),
            mount.write(&ctl, 0, b"commit\n", &caller),
        );
        a.unwrap();
        b.unwrap();
        let status = ctl_str(&mount, &ctl, &caller).await;
        assert!(status.contains("state=sealed cid="), "got: {status}");
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn commit_authorizes_with_staging_id_address() {
        let dir = tempfile::tempdir().unwrap();
        let authz = std::sync::Arc::new(RecordingAuthorizer::default());
        let mount = CasMount::with_authorizer_and_staging(
            CasSubstrate::new(dir.path()),
            DedupDomain::local_default(),
            authz.clone(),
            StagingConfig::default(),
        );
        let caller = Subject::new("alice");
        let (id, _root) = create_stage(&mount, &caller).await;
        let (data, ctl) = open_data_ctl(&mount, &id, &caller).await;
        mount
            .write(&data, 0, b"authz-bytes", &caller)
            .await
            .unwrap();
        declare_test_label(&mount, &ctl, &caller).await;
        mount.write(&ctl, 0, b"commit\n", &caller).await.unwrap();

        // The commit authorization must address the staging id, not "ctl".
        let commit_calls: Vec<_> = authz
            .calls()
            .into_iter()
            .filter(|(_, _, _, op)| *op == "stage:commit")
            .collect();
        assert_eq!(commit_calls.len(), 1);
        assert_eq!(commit_calls[0].2, id);
        mount.clunk(data, &caller).await;
        mount.clunk(ctl, &caller).await;
    }

    #[tokio::test]
    async fn staging_qids_are_distinct_per_slot_and_file() {
        // Regression guard for the qid-aliasing finding: a 9P client keys a
        // file cache on qid path, so two slots' `data` files (and `data` vs
        // `ctl`) MUST NOT share a qid. The path is hashed from
        // `stage/{id}/{name}`.
        let dir = tempfile::tempdir().unwrap();
        let mount = staging_mount(CasSubstrate::new(dir.path()), 1 << 20, 4 << 20);
        let caller = Subject::new("alice");
        let (id1, root1) = create_stage(&mount, &caller).await;
        let (id2, root2) = create_stage(&mount, &caller).await;

        // Stat each slot's data + ctl directly (no open needed for stat).
        let d1 = mount.walk(&["stage", &id1, "data"], &caller).await.unwrap();
        let c1 = mount.walk(&["stage", &id1, "ctl"], &caller).await.unwrap();
        let d2 = mount.walk(&["stage", &id2, "data"], &caller).await.unwrap();

        let d1_stat = mount.stat(&d1, &caller).await.unwrap();
        let c1_stat = mount.stat(&c1, &caller).await.unwrap();
        let d2_stat = mount.stat(&d2, &caller).await.unwrap();

        // Distinct slots' data files must not alias.
        assert_ne!(
            d1_stat.path, d2_stat.path,
            "stage/<a>/data and stage/<b>/data must have distinct qids"
        );
        // data vs ctl within the same slot must not alias either.
        assert_ne!(
            d1_stat.path, c1_stat.path,
            "stage/<id>/data and stage/<id>/ctl must have distinct qids"
        );
        // A qid of 0 would mean "unknown"; make sure these are real identities.
        assert_ne!(d1_stat.path, 0);
        assert_ne!(d2_stat.path, 0);

        // readdir on a staging root also surfaces the per-slot qids.
        let entries = mount.readdir(&root1, &caller).await.unwrap();
        let data_entry = entries.iter().find(|e| e.name == "data").unwrap();
        assert_eq!(data_entry.stat.as_ref().unwrap().path, d1_stat.path);

        for f in [d1, c1, d2, root1, root2] {
            mount.clunk(f, &caller).await;
        }
    }
}
