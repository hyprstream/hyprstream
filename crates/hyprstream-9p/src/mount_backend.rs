//! `MountBackend` — a [`Backend`] that exports an in-process VFS [`Mount`].
//!
//! This is the server-side adapter used by the UDS 9P listener (#506). It is
//! the local-mount counterpart to the `hyprstream` crate's capnp-RPC
//! `ModelBackend`: both implement the same [`Backend`] seam the [`Translator`]
//! dispatches to, so the accept/serve/fid-table machinery is shared verbatim —
//! only the object behind the `Backend` differs.
//!
//! ```text
//!   Wanix (p9kit.ClientFS) ──► UnixListener ──► Translator ──► MountBackend ──► dyn Mount
//!         9P2000.L wire            UDS            fid table      Subject-scoped     policy PEP
//! ```
//!
//! ## Subject threading (MAC-load-bearing)
//!
//! Every [`Mount`] method takes the verified caller [`Subject`]. The listener
//! serves exactly one tenant's export, so the `Subject` is fixed at construction
//! and threaded onto every op; the served mount remains the single policy
//! enforcement point. There is no path by which a 9P op reaches the mount without
//! the tenant's `Subject`.
//!
//! ## Fid mapping
//!
//! The translator allocates opaque `u32` 9P fids; a [`Mount`] hands back opaque
//! [`Fid`] handles from `walk`. `MountBackend` owns the `u32 → (path, Fid)`
//! mapping. Because a [`Mount::walk`] resolves a path from the mount root (not
//! relative to a source fid), each fid also remembers its absolute path so a
//! subsequent relative walk can re-resolve `parent_path + components`.

use std::sync::Arc;

use anyhow::{anyhow, Context as _, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, VfsOpContext};
use tokio::sync::{Mutex, OnceCell};

use crate::backend::{Backend, OpenResult, StatResult, WalkResult};
pub use crate::mac_seam::AttachAuthenticator as AttachAuthorizer;
use crate::mac_seam::VerifiedAttach;
use crate::msg::{self, Qid, ReaddirEntry};

/// Resolves the mount ticket a client presents in `Tattach.uname` to the
/// verified session [`Subject`] the export is scoped to.
///
/// This is the attach-time credential seam for transports where the caller
/// identity is not fixed at listener construction: the H1b `/9p` WebTransport
/// plane serves a cert-pinned mesh session over which many tenants could
/// attach, so the ticket rides `Tattach.uname` and is validated here — the
/// per-session analogue of the RPC `EnvelopeContext` (MAC interface policy:
/// "extend at attach time, never per-op"). The concrete impl lives in the
/// `hyprstream` crate (it owns the JWT/OAuth verification chain); this trait
/// keeps `hyprstream-9p` free of that dependency.
///
/// On denial, return a [`MountError`] — the translator maps it to an `Rlerror`
/// errno (e.g. [`MountError::PermissionDenied`] → `EACCES`).
/// Max bytes a single read/write returns. Kept below the translator's `MSG_SIZE`
/// (8 KiB) so an `Rread` carrying a full iounit plus its 9P header still fits the
/// negotiated msize.
const IOUNIT: u32 = 8 * 1024 - 64;

/// 9P qtype bit for a directory (`QTDIR`).
const QTDIR: u8 = 0x80;

/// Per-fid state: the absolute path it resolves to plus the live mount handle.
///
/// The handle lives behind an async `Mutex` so a mount op can be awaited without
/// holding a `DashMap` shard guard. `Option` because `clunk` takes ownership of
/// the [`Fid`] to hand back to the mount.
struct MountFidEntry {
    path: Vec<String>,
    handle: Mutex<Option<Fid>>,
}

/// A [`Backend`] backed by a single Subject-scoped VFS [`Mount`].
///
/// The session [`Subject`] is either fixed at construction ([`MountBackend::new`],
/// the UDS/vsock listeners) or resolved from the client's `Tattach.uname` ticket
/// at attach time ([`MountBackend::with_authorizer`], the H1b `/9p` WebTransport
/// plane). It lives behind a [`OnceCell`] so per-op code reads one bound value
/// either way; ops before a successful attach fail closed.
pub struct MountBackend {
    mount: Arc<dyn Mount>,
    subject: OnceCell<Subject>,
    /// The optional attachment context is fixed by the first successful
    /// attach. It is separate from `subject`: a verified identity alone is
    /// never elevated into Task authority.
    operation_context: OnceCell<Option<VfsOpContext>>,
    /// Present only for the attach-time path; resolves `uname` → `Subject`.
    authorizer: Option<Arc<dyn AttachAuthorizer>>,
    fids: DashMap<u32, Arc<MountFidEntry>>,
}

impl MountBackend {
    /// Wrap `mount` as the 9P export root for a `subject` fixed at construction.
    pub fn new(mount: Arc<dyn Mount>, subject: Subject) -> Self {
        let cell = OnceCell::new();
        // Infallible on a fresh cell.
        let _ = cell.set(subject);
        Self {
            mount,
            subject: cell,
            operation_context: OnceCell::new(),
            authorizer: None,
            fids: DashMap::new(),
        }
    }

    /// Wrap `mount` as the 9P export root whose session [`Subject`] is resolved
    /// from the `Tattach.uname` ticket by `authorizer` (H1b `/9p` WebTransport).
    ///
    /// The `Subject` is unbound until a successful [`Backend::attach`]; any op
    /// arriving before attach fails closed.
    pub fn with_authorizer(mount: Arc<dyn Mount>, authorizer: Arc<dyn AttachAuthorizer>) -> Self {
        Self {
            mount,
            subject: OnceCell::new(),
            operation_context: OnceCell::new(),
            authorizer: Some(authorizer),
            fids: DashMap::new(),
        }
    }

    /// The bound session [`Subject`], or an error if no attach has bound it yet
    /// (fail-closed: a 9P op must never reach the mount without a caller).
    fn caller(&self) -> Result<&Subject> {
        self.subject
            .get()
            .ok_or_else(|| anyhow!("9P op before Tattach: session Subject not bound"))
    }

    /// Validate a context supplied by a verified attach before session state
    /// is bound. This is deliberately not a conversion from the 9P MAC
    /// session: only `VerifiedAttach::try_new_with_operation_grant` can
    /// populate it from an independently-issued opaque grant.
    fn requested_operation_context(
        &self,
        verified: Option<&VerifiedAttach>,
    ) -> Result<Option<VfsOpContext>> {
        let context = verified
            .and_then(VerifiedAttach::operation_context)
            .cloned();
        if let (Some(verified), Some(context)) = (verified, context.as_ref()) {
            if context.subject() != verified.subject() {
                return Err(anyhow::Error::new(MountError::PermissionDenied(
                    "attachment operation context differs from verified attach subject".to_owned(),
                )));
            }
            context.ensure_current().map_err(anyhow::Error::new)?;
        }
        Ok(context)
    }

    /// Bind the optional context exactly once. Reattach is idempotent only for
    /// an exact copy of the original grant, including its scope set; context
    /// upgrades, downgrades, and replacements are all denied.
    fn bind_operation_context(&self, requested: Option<VfsOpContext>) -> Result<()> {
        // Check again immediately before binding. Revocation remains checked
        // at every effect by the retained fid context, closing the remaining
        // race after attach.
        if let Some(context) = requested.as_ref() {
            context.ensure_current().map_err(anyhow::Error::new)?;
        }
        match self.operation_context.set(requested) {
            Ok(()) => Ok(()),
            Err(attempted) => {
                let bound = self.operation_context.get().ok_or_else(|| {
                    anyhow!("bind attachment operation context: cell rejected without value")
                })?;
                let same = match (bound.as_ref(), attempted.as_ref()) {
                    (None, None) => true,
                    (Some(bound), Some(attempted)) => bound.same_grant(attempted),
                    _ => false,
                };
                if same {
                    Ok(())
                } else {
                    Err(anyhow::Error::new(MountError::PermissionDenied(
                        "reattach cannot upgrade, downgrade, or replace attachment operation context"
                            .to_owned(),
                    )))
                }
            }
        }
    }

    /// Walk through the context-aware trait seam when attach carried a Task
    /// grant. Generic mounts inherit `Mount`'s Subject-only default; mounts
    /// such as `ExecMount` override it to retain the context on the returned
    /// fid for later effects.
    async fn walk_mount(&self, components: &[&str]) -> Result<Fid> {
        match self.operation_context.get().and_then(Option::as_ref) {
            Some(context) => self
                .mount
                .walk_with_context(components, context)
                .await
                .map_err(anyhow::Error::new),
            None => self
                .mount
                .walk(components, self.caller()?)
                .await
                .map_err(anyhow::Error::new),
        }
    }

    /// Clone out the `Arc` for a fid, dropping the `DashMap` guard so the mount
    /// call can await without holding a shard lock.
    fn entry(&self, fid: u32) -> Result<Arc<MountFidEntry>> {
        self.fids
            .get(&fid)
            .map(|e| Arc::clone(&e))
            .ok_or_else(|| anyhow!("fid {fid} not walked"))
    }

    /// Build the leaf [`Qid`] for a mount handle by stat-ing it.
    async fn qid_of(&self, handle: &Fid) -> Result<Qid> {
        let st = self
            .mount
            .stat(handle, self.caller()?)
            .await
            .context("mount stat failed")?;
        Ok(Qid {
            qtype: st.qtype,
            version: st.version,
            path: st.path,
        })
    }
}

#[async_trait]
impl Backend for MountBackend {
    async fn attach(
        &self,
        uname: &str,
        aname: &str,
        verified: Option<VerifiedAttach>,
    ) -> Result<Option<VerifiedAttach>> {
        // Fixed-subject listeners have no authorizer; the Subject is already
        // bound and `uname`/`aname` are advisory (ignored, as on the UDS/vsock
        // paths).
        let Some(authorizer) = self.authorizer.as_ref() else {
            if let Some(verified) = verified {
                let existing = self
                    .subject
                    .get()
                    .ok_or_else(|| anyhow!("fixed 9P subject missing"))?;
                if existing != verified.subject() {
                    return Err(anyhow::Error::new(MountError::PermissionDenied(
                        "verified attach subject differs from fixed VFS subject".to_owned(),
                    )));
                }
                let context = self.requested_operation_context(Some(&verified))?;
                self.bind_operation_context(context)?;
                return Ok(Some(verified));
            }
            self.bind_operation_context(None)?;
            return Ok(None);
        };
        // Attach-time ticket path (H1b): resolve+narrow the Subject AND validate
        // the requested `aname` export against the ticket's namespace grant in
        // one authorization. A MountError here maps to an Rlerror errno
        // (PermissionDenied → EACCES) via the translator, before any fid exists.
        let verified = match verified {
            Some(verified) => verified,
            None => authorizer
                .authenticate(uname, aname)
                .await
                .map_err(anyhow::Error::new)?,
        };
        let subject = verified.subject().clone();
        let attempted_subject = subject.clone();
        let context = self.requested_operation_context(Some(&verified))?;
        // First attach wins; a second attach on the same connection must not
        // silently re-scope the session. Ignore a redundant identical set,
        // reject a conflicting one.
        match self.subject.set(subject) {
            Ok(()) => {
                self.bind_operation_context(context)?;
                Ok(Some(verified))
            }
            Err(_) => {
                let existing = self
                    .subject
                    .get()
                    .ok_or_else(|| anyhow!("bind session Subject: cell rejected without value"))?;
                if existing == &attempted_subject {
                    self.bind_operation_context(context)?;
                    Ok(Some(verified))
                } else {
                    Err(anyhow::Error::new(MountError::PermissionDenied(
                        "conflicting attach subject".to_owned(),
                    )))
                }
            }
        }
    }

    async fn walk(&self, fid: u32, newfid: u32, components: &[String]) -> Result<WalkResult> {
        // A Mount walk resolves from the root, so build the target's absolute
        // path as parent_path + components. An empty-components walk (attach /
        // clone) re-resolves the source fid's own path.
        let parent_path = self
            .fids
            .get(&fid)
            .map(|e| e.path.clone())
            .unwrap_or_default();
        let parent_len = parent_path.len();
        let mut new_path = parent_path;
        new_path.extend(components.iter().cloned());

        // Resolve each component independently so the 9P result carries one
        // QID for every object the new fid actually traversed. Returning only a
        // leaf QID while binding `newfid` to the complete path would let the
        // translator cache a shallower, incorrectly authorized name.
        let mut qids = Vec::with_capacity(components.len().max(1));
        let mut handle = None;
        let mut reached = Vec::with_capacity(components.len());
        for component_count in 1..=components.len() {
            let refs: Vec<&str> = new_path[..parent_len + component_count]
                .iter()
                .map(String::as_str)
                .collect();
            // On any failure past the first successful hop, `handle` still
            // holds the previously-resolved mount `Fid`. That `Fid` has no
            // `Drop` cleanup (see `hyprstream_vfs::Fid`) — for a remote mount
            // it is a live 9P fid held open on the peer — so it must be
            // explicitly clunked before returning the error, or every walk
            // that fails past its first component (e.g. `/a/b` where `a`
            // exists but `b` does not) leaks one backend handle.
            let next = match self.walk_mount(&refs).await {
                Ok(next) => next,
                Err(e) => {
                    if let Some(previous) = handle.take() {
                        self.mount.clunk(previous, self.caller()?).await;
                    }
                    return Err(anyhow::Error::new(e).context("mount walk failed"));
                }
            };
            let qid = match self.qid_of(&next).await {
                Ok(qid) => qid,
                Err(e) => {
                    self.mount.clunk(next, self.caller()?).await;
                    if let Some(previous) = handle.take() {
                        self.mount.clunk(previous, self.caller()?).await;
                    }
                    return Err(e);
                }
            };
            if let Some(previous) = handle.replace(next) {
                self.mount.clunk(previous, self.caller()?).await;
            }
            qids.push(qid);
            reached.push(components[component_count - 1].clone());
        }

        if components.is_empty() {
            let refs: Vec<&str> = new_path.iter().map(String::as_str).collect();
            let next = self.walk_mount(&refs).await.context("mount walk failed")?;
            qids.push(self.qid_of(&next).await?);
            handle = Some(next);
        }
        let handle = handle.ok_or_else(|| anyhow!("mount walk returned no handle"))?;
        self.fids.insert(
            newfid,
            Arc::new(MountFidEntry {
                path: new_path,
                handle: Mutex::new(Some(handle)),
            }),
        );
        Ok(WalkResult { qids, reached })
    }

    async fn open(&self, fid: u32, flags: u32) -> Result<OpenResult> {
        let entry = self.entry(fid)?;
        let mut guard = entry.handle.lock().await;
        let handle = guard
            .as_mut()
            .ok_or_else(|| anyhow!("open: fid {fid} is clunked"))?;
        let mode = lopen_flags_to_mode(flags);
        self.mount
            .open(handle, mode, self.caller()?)
            .await
            .context("mount open failed")?;
        let qid = self.qid_of(handle).await?;
        Ok(OpenResult {
            qid,
            iounit: IOUNIT,
        })
    }

    async fn read(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard
            .as_ref()
            .ok_or_else(|| anyhow!("read: fid {fid} is clunked"))?;
        self.mount
            .read(handle, offset, count, self.caller()?)
            .await
            .context("mount read failed")
    }

    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> Result<u32> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard
            .as_ref()
            .ok_or_else(|| anyhow!("write: fid {fid} is clunked"))?;
        self.mount
            .write(handle, offset, data, self.caller()?)
            .await
            .context("mount write failed")
    }

    async fn stat(&self, fid: u32) -> Result<StatResult> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard
            .as_ref()
            .ok_or_else(|| anyhow!("stat: fid {fid} is clunked"))?;
        let st = self
            .mount
            .stat(handle, self.caller()?)
            .await
            .context("mount stat failed")?;
        let is_dir = st.qtype & QTDIR != 0;
        let mode = if is_dir { 0o040755 } else { 0o100644 };
        Ok(StatResult {
            qid: Qid {
                qtype: st.qtype,
                version: st.version,
                path: st.path,
            },
            mode,
            size: st.size,
            mtime_sec: st.mtime,
        })
    }

    async fn readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        let entry = self.entry(fid)?;
        let entries = {
            let guard = entry.handle.lock().await;
            let handle = guard
                .as_ref()
                .ok_or_else(|| anyhow!("readdir: fid {fid} is clunked"))?;
            self.mount
                .readdir(handle, self.caller()?)
                .await
                .context("mount readdir failed")?
        };
        Ok(encode_dir_entries(&entries, offset, count))
    }

    async fn clunk(&self, fid: u32) -> Result<()> {
        // Drop local state and hand the mount handle back for release.
        if let Some((_, entry)) = self.fids.remove(&fid) {
            let handle = entry.handle.lock().await.take();
            if let Some(handle) = handle {
                self.mount.clunk(handle, self.caller()?).await;
            }
        }
        Ok(())
    }
}

/// Encode directory entries as a page of **standard 9P2000.L Rreaddir dirent
/// records** (`qid[13] · offset[8] · type[1] · name[s]`) via
/// [`msg::encode_readdir_page`].
///
/// This is the wire-faithful format a standard 9P client (Wanix `p9kit` over
/// `progrium/p9`) requires — not the hyprstream-internal
/// `name_len/name/is_dir/size` dialect. `offset` is a dirent cookie and `count`
/// a byte budget; records are packed whole (see `encode_readdir_page`).
///
/// Per-entry qid: a `DirEntry` may carry a `stat` with a real qid; when it does
/// we use it, otherwise we synthesize the qtype from `is_dir` with an unknown
/// (`version=0, path=0`) identity — sound per `Stat`'s qid invariant, and
/// sufficient because standard clients re-walk each name to stat it.
fn encode_dir_entries(entries: &[DirEntry], offset: u64, count: u32) -> Vec<u8> {
    let records: Vec<ReaddirEntry> = entries
        .iter()
        .map(|e| {
            let qid = match &e.stat {
                Some(st) => Qid {
                    qtype: st.qtype,
                    version: st.version,
                    path: st.path,
                },
                None => Qid {
                    qtype: if e.is_dir { QTDIR } else { 0 },
                    version: 0,
                    path: 0,
                },
            };
            ReaddirEntry {
                qid,
                name: e.name.clone(),
            }
        })
        .collect();
    msg::encode_readdir_page(&records, offset, count)
}

/// Map 9P2000.L `Tlopen` flags (Linux `O_*` bits) to a 9P open-mode byte
/// (`OREAD=0` / `OWRITE=1` / `ORDWR=2`). Only read/write intent is preserved;
/// `O_CREAT`/`O_TRUNC`/`O_APPEND` are advisory here. Mirrors the equivalent
/// mapping in the capnp-RPC `ModelBackend`.
fn lopen_flags_to_mode(flags: u32) -> u8 {
    const O_WRONLY: u32 = 0o1;
    const O_RDWR: u32 = 0o2;
    match flags & 0o3 {
        O_WRONLY => 1, // OWRITE
        O_RDWR => 2,   // ORDWR
        _ => 0,        // OREAD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lopen_flags_mapping() {
        assert_eq!(lopen_flags_to_mode(0), 0);
        assert_eq!(lopen_flags_to_mode(0o1), 1);
        assert_eq!(lopen_flags_to_mode(0o2), 2);
        assert_eq!(lopen_flags_to_mode(0o101), 1);
    }

    /// The attach-to-VFS bridge is tested here with a deliberately small
    /// attachment-aware mount instead of the Worker fake backend: this crate
    /// owns the direct `Backend::attach` / `dyn Mount` boundary, while
    /// `ExecMount` separately covers the same retained-fid context semantics
    /// in its focused Worker tests. The fake fid treats `write` as a ctl effect
    /// so this proves a post-attach revocation reaches the next 9P effect.
    #[tokio::test]
    async fn verified_attach_context_reaches_dyn_mount_and_fences_ctl_effects() {
        use async_trait::async_trait;
        use hyprstream_rpc::auth::mac::{
            CompartmentSet, Level, SecurityContext, VerifiedKeyMaterial,
        };
        use hyprstream_rpc::{AttachmentOperation, VerifiedAttachment};
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone)]
        struct AttachmentFid {
            context: Option<VfsOpContext>,
        }

        struct AttachmentAwareMount {
            legacy_walks: AtomicUsize,
            contextual_walks: AtomicUsize,
            ctl_effects: AtomicUsize,
        }

        #[async_trait]
        impl Mount for AttachmentAwareMount {
            async fn walk(
                &self,
                _components: &[&str],
                _caller: &Subject,
            ) -> Result<Fid, MountError> {
                self.legacy_walks.fetch_add(1, Ordering::SeqCst);
                Err(MountError::NotFound(
                    "attachment-bound task is hidden without an operation context".into(),
                ))
            }

            async fn walk_with_context(
                &self,
                _components: &[&str],
                context: &VfsOpContext,
            ) -> Result<Fid, MountError> {
                context.ensure_operation(hyprstream_rpc::AttachmentOperation::TaskRead)?;
                self.contextual_walks.fetch_add(1, Ordering::SeqCst);
                Ok(Fid::new(AttachmentFid {
                    context: Some(context.clone()),
                }))
            }

            async fn open(
                &self,
                _fid: &mut Fid,
                _mode: u8,
                _caller: &Subject,
            ) -> Result<(), MountError> {
                Ok(())
            }

            async fn read(
                &self,
                fid: &Fid,
                _offset: u64,
                _count: u32,
                _caller: &Subject,
            ) -> Result<Vec<u8>, MountError> {
                let context = fid
                    .downcast_ref::<AttachmentFid>()
                    .and_then(|fid| fid.context.as_ref())
                    .ok_or_else(|| MountError::NotFound("task is hidden".into()))?;
                context.ensure_operation(hyprstream_rpc::AttachmentOperation::TaskRead)?;
                Ok(b"visible through the verified attach".to_vec())
            }

            async fn write(
                &self,
                fid: &Fid,
                _offset: u64,
                data: &[u8],
                _caller: &Subject,
            ) -> Result<u32, MountError> {
                let context = fid
                    .downcast_ref::<AttachmentFid>()
                    .and_then(|fid| fid.context.as_ref())
                    .ok_or_else(|| MountError::NotFound("task ctl is hidden".into()))?;
                context.ensure_operation(hyprstream_rpc::AttachmentOperation::TaskSignal)?;
                self.ctl_effects.fetch_add(1, Ordering::SeqCst);
                Ok(data.len() as u32)
            }

            async fn readdir(
                &self,
                _fid: &Fid,
                _caller: &Subject,
            ) -> Result<Vec<DirEntry>, MountError> {
                Err(MountError::NotDirectory("not a directory".into()))
            }

            async fn stat(
                &self,
                _fid: &Fid,
                _caller: &Subject,
            ) -> Result<hyprstream_vfs::Stat, MountError> {
                Ok(hyprstream_vfs::Stat::unknown_qid(0, 0, "ctl".into(), 0))
            }

            async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
        }

        struct StaticGrantAuthorizer(VerifiedAttach);

        #[async_trait]
        impl AttachAuthorizer for StaticGrantAuthorizer {
            async fn authenticate(
                &self,
                _uname: &str,
                _aname: &str,
            ) -> Result<VerifiedAttach, MountError> {
                Ok(self.0.clone())
            }
        }

        fn verified_attach_for(
            attachment: &VerifiedAttachment,
            operations: &[AttachmentOperation],
        ) -> VerifiedAttach {
            let identity = VerifiedAttachIdentity::from_verified_credential(
                attachment.subject().name().expect("non-anonymous fixture"),
                "task-fixture-tenant",
            );
            let session = SessionContext::from_verified_clearance(
                identity.clone(),
                SecurityContext::new(
                    Level::Secret,
                    CompartmentSet::EMPTY,
                    VerifiedKeyMaterial::Classical,
                ),
            );
            let grant = attachment.for_test_operations(operations);
            VerifiedAttach::try_new_with_operation_grant(
                identity,
                attachment.subject().clone(),
                session,
                &grant,
            )
            .expect("fixture grant and verified attach share a subject")
        }

        let attachment =
            VerifiedAttachment::for_test_local_root(Subject::new("context-owner")).unwrap();
        let permitted = verified_attach_for(
            &attachment,
            &[
                AttachmentOperation::TaskRead,
                AttachmentOperation::TaskSignal,
            ],
        );
        let mount = Arc::new(AttachmentAwareMount {
            legacy_walks: AtomicUsize::new(0),
            contextual_walks: AtomicUsize::new(0),
            ctl_effects: AtomicUsize::new(0),
        });
        let backend = MountBackend::with_authorizer(
            mount.clone(),
            Arc::new(StaticGrantAuthorizer(permitted.clone())),
        );

        // An authorizer returns the independently-issued opaque grant as part
        // of its already-verified attach bundle. `MountBackend` retains it and
        // dispatches through `dyn Mount::walk_with_context`.
        backend.attach("ticket", "", None).await.unwrap();
        backend
            .walk(0, 1, &["task".to_owned(), "ctl".to_owned()])
            .await
            .unwrap();
        assert_eq!(mount.contextual_walks.load(Ordering::SeqCst), 2);
        assert_eq!(mount.legacy_walks.load(Ordering::SeqCst), 0);
        assert_eq!(
            backend.read(1, 0, 4096).await.unwrap(),
            b"visible through the verified attach"
        );
        assert_eq!(backend.write(1, 0, b"stop").await.unwrap(), 4);
        assert_eq!(mount.ctl_effects.load(Ordering::SeqCst), 1);

        // Reattach accepts an exact copy only. A same-attachment read-only
        // grant is not equivalent: scope changes are denied, not silently
        // applied to this established connection.
        backend
            .attach("ticket", "", Some(permitted.clone()))
            .await
            .unwrap();
        let narrowed = verified_attach_for(&attachment, &[AttachmentOperation::TaskRead]);
        assert!(backend.attach("ticket", "", Some(narrowed)).await.is_err());
        let identity = VerifiedAttachIdentity::from_verified_credential(
            "context-owner",
            "task-fixture-tenant",
        );
        let legacy = VerifiedAttach::try_new(
            identity.clone(),
            Subject::new("context-owner"),
            SessionContext::from_verified_clearance(
                identity,
                SecurityContext::new(
                    Level::Secret,
                    CompartmentSet::EMPTY,
                    VerifiedKeyMaterial::Classical,
                ),
            ),
        )
        .unwrap();
        assert!(backend.attach("ticket", "", Some(legacy)).await.is_err());

        // Revocation is observed by the context retained in the walked fid;
        // the next ctl effect is denied before the mount can perform it.
        attachment.revoke().unwrap();
        assert!(backend.attach("ticket", "", Some(permitted)).await.is_err());
        assert!(backend.write(1, 0, b"stop").await.is_err());
        assert_eq!(mount.ctl_effects.load(Ordering::SeqCst), 1);

        // A verified legacy attach has no delegated Task grant. It therefore
        // reaches only the generic Subject walk and the attachment-bound tree
        // stays hidden.
        let legacy_identity =
            VerifiedAttachIdentity::from_verified_credential("legacy-owner", "task-fixture-tenant");
        let legacy_attach = VerifiedAttach::try_new(
            legacy_identity.clone(),
            Subject::new("legacy-owner"),
            SessionContext::from_verified_clearance(
                legacy_identity,
                SecurityContext::new(
                    Level::Secret,
                    CompartmentSet::EMPTY,
                    VerifiedKeyMaterial::Classical,
                ),
            ),
        )
        .unwrap();
        let missing_context_mount = Arc::new(AttachmentAwareMount {
            legacy_walks: AtomicUsize::new(0),
            contextual_walks: AtomicUsize::new(0),
            ctl_effects: AtomicUsize::new(0),
        });
        let missing_context_backend = MountBackend::with_authorizer(
            missing_context_mount.clone(),
            Arc::new(StaticGrantAuthorizer(legacy_attach)),
        );
        missing_context_backend
            .attach("ticket", "", None)
            .await
            .unwrap();
        let upgrade_owner =
            VerifiedAttachment::for_test_local_root(Subject::new("legacy-owner")).unwrap();
        let attempted_upgrade =
            verified_attach_for(&upgrade_owner, &[AttachmentOperation::TaskRead]);
        assert!(missing_context_backend
            .attach("ticket", "", Some(attempted_upgrade))
            .await
            .is_err());
        assert!(missing_context_backend
            .walk(0, 1, &["task".to_owned()])
            .await
            .is_err());
        assert_eq!(
            missing_context_mount
                .contextual_walks
                .load(Ordering::SeqCst),
            0
        );
        assert_eq!(missing_context_mount.legacy_walks.load(Ordering::SeqCst), 1);
    }

    /// Regression: a multi-component walk that fails past its first hop must
    /// clunk every intermediate `Fid` it resolved before returning the error.
    /// `Fid` has no `Drop` cleanup — for a remote mount it is a live 9P fid
    /// held open on the peer — so a dropped-without-clunking handle leaks on
    /// every walk that ENOENTs past its first component (e.g. `/a/b` where
    /// `a` exists but `b` does not).
    #[tokio::test]
    async fn walk_clunks_intermediate_handle_on_mid_path_failure() {
        use async_trait::async_trait;
        use hyprstream_vfs::{DirEntry, Stat};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TrackedMount {
            clunks: AtomicUsize,
        }

        #[async_trait]
        impl Mount for TrackedMount {
            async fn walk(&self, components: &[&str], _c: &Subject) -> Result<Fid, MountError> {
                match components {
                    [] | ["a"] => Ok(Fid::new(components.len())),
                    ["a", "b"] => Err(MountError::NotFound("a/b".into())),
                    other => panic!("unexpected walk: {other:?}"),
                }
            }
            async fn open(&self, _f: &mut Fid, _m: u8, _c: &Subject) -> Result<(), MountError> {
                Err(MountError::PermissionDenied("open".into()))
            }
            async fn read(
                &self,
                _f: &Fid,
                _o: u64,
                _n: u32,
                _c: &Subject,
            ) -> Result<Vec<u8>, MountError> {
                Err(MountError::Io("read".into()))
            }
            async fn write(
                &self,
                _f: &Fid,
                _o: u64,
                _d: &[u8],
                _c: &Subject,
            ) -> Result<u32, MountError> {
                Err(MountError::NotSupported("write".into()))
            }
            async fn readdir(&self, _f: &Fid, _c: &Subject) -> Result<Vec<DirEntry>, MountError> {
                Err(MountError::NotDirectory("readdir".into()))
            }
            async fn stat(&self, f: &Fid, _c: &Subject) -> Result<Stat, MountError> {
                let depth = *f.downcast_ref::<usize>().unwrap_or(&0);
                Ok(Stat::unknown_qid(0, depth as u64, "x".into(), 0))
            }
            async fn clunk(&self, _f: Fid, _c: &Subject) {
                self.clunks.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mount = Arc::new(TrackedMount {
            clunks: AtomicUsize::new(0),
        });
        let backend = MountBackend::new(mount.clone(), Subject::new("tenant"));

        // Attach (empty walk) binds fid 0 to the root. `MountBackend::new`
        // fixes the Subject, so `uname`/`aname` are advisory here.
        backend.attach("tenant", "", None).await.unwrap();
        backend.fids.insert(
            0,
            Arc::new(MountFidEntry {
                path: vec![],
                handle: Mutex::new(Some(Fid::new(0usize))),
            }),
        );

        // Walk fid 0 -> newfid 1 via ["a", "b"]: "a" resolves (handle A is
        // held), "b" fails. Handle A must be clunked on this error path.
        let err = backend.walk(0, 1, &["a".to_owned(), "b".to_owned()]).await;
        assert!(err.is_err(), "walk into a missing leaf must fail");
        assert_eq!(
            mount.clunks.load(Ordering::SeqCst),
            1,
            "the intermediate handle for \"a\" must be clunked on the \"a/b\" failure, not leaked"
        );
    }

    #[test]
    fn encode_dir_entries_emits_standard_rreaddir_records() {
        use crate::msg::parse_readdir_entries;

        let entries = vec![
            DirEntry {
                name: "a".into(),
                is_dir: true,
                size: 0,
                stat: None,
            },
            DirEntry {
                name: "bb".into(),
                is_dir: false,
                size: 7,
                stat: None,
            },
        ];
        let full = encode_dir_entries(&entries, 0, u32::MAX);
        assert!(!full.is_empty());

        // Standard record layout: qid[13] · offset[8] · type[1] · name[2+len].
        // First entry "a" (a dir): qid.qtype = QTDIR at byte 0, dirent type at
        // byte 21 also QTDIR, name length u16=1 at bytes 22..24, 'a' at 24.
        assert_eq!(full[0], QTDIR); // qid.qtype
        assert_eq!(&full[13..21], &1u64.to_le_bytes()); // offset cookie = 1
        assert_eq!(full[21], QTDIR); // dirent type = dir
        assert_eq!(&full[22..24], &1u16.to_le_bytes()); // name len
        assert_eq!(full[24], b'a');

        // Round-trips through the standard client-side parser.
        let parsed = parse_readdir_entries(&full).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "a");
        assert!(parsed[0].qid.is_dir());
        assert_eq!(parsed[0].offset, 1);
        assert_eq!(parsed[1].name, "bb");
        assert!(!parsed[1].qid.is_dir());
        assert_eq!(parsed[1].offset, 2);

        // Cookie paging: offset past the last cookie yields nothing.
        assert!(encode_dir_entries(&entries, 2, u32::MAX).is_empty());
        // Resuming after cookie 1 yields only the second entry.
        let rest = parse_readdir_entries(&encode_dir_entries(&entries, 1, u32::MAX)).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].name, "bb");
    }
}
