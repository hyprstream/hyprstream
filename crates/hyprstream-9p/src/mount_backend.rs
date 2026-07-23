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
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError};
use tokio::sync::{Mutex, OnceCell};

use crate::backend::{Backend, OpenResult, StatResult, WalkResult};
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
#[async_trait]
pub trait AttachAuthorizer: Send + Sync {
    /// Validate the `uname` ticket and return the narrowed session [`Subject`].
    async fn authorize(&self, uname: &str) -> Result<Subject, MountError>;
}

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
        Self { mount, subject: cell, authorizer: None, fids: DashMap::new() }
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
        Ok(Qid { qtype: st.qtype, version: st.version, path: st.path })
    }
}

#[async_trait]
impl Backend for MountBackend {
    async fn attach(&self, uname: &str) -> Result<()> {
        // Fixed-subject listeners have no authorizer; the Subject is already
        // bound and `uname` is advisory (ignored, as on the UDS/vsock paths).
        let Some(authorizer) = self.authorizer.as_ref() else {
            return Ok(());
        };
        // Attach-time ticket path (H1b): resolve+narrow. A MountError here maps
        // to an Rlerror errno (PermissionDenied → EACCES) via the translator.
        let subject = authorizer.authorize(uname).await.map_err(anyhow::Error::new)?;
        let attempted_subject = subject.clone();
        // First attach wins; a second attach on the same connection must not
        // silently re-scope the session. Ignore a redundant identical set,
        // reject a conflicting one.
        match self.subject.set(subject) {
            Ok(()) => Ok(()),
            Err(_) => {
                let existing = self
                    .subject
                    .get()
                    .ok_or_else(|| anyhow!("bind session Subject: cell rejected without value"))?;
                if existing == &attempted_subject {
                    Ok(())
                } else {
                    Err(anyhow::Error::new(MountError::PermissionDenied(
                        "conflicting attach subject".to_owned(),
                    )))
                }
            }
        }
    }

    async fn walk(
        &self,
        fid: u32,
        newfid: u32,
        components: &[String],
    ) -> Result<WalkResult> {
        // A Mount walk resolves from the root, so build the target's absolute
        // path as parent_path + components. An empty-components walk (attach /
        // clone) re-resolves the source fid's own path.
        let parent_path = self.fids.get(&fid).map(|e| e.path.clone()).unwrap_or_default();
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
            let next = self.mount.walk(&refs, self.caller()?).await.context("mount walk failed")?;
            let qid = self.qid_of(&next).await?;
            if let Some(previous) = handle.replace(next) {
                self.mount.clunk(previous, self.caller()?).await;
            }
            qids.push(qid);
            reached.push(components[component_count - 1].clone());
        }

        if components.is_empty() {
            let refs: Vec<&str> = new_path.iter().map(String::as_str).collect();
            let next = self.mount.walk(&refs, self.caller()?).await.context("mount walk failed")?;
            qids.push(self.qid_of(&next).await?);
            handle = Some(next);
        }
        let handle = handle.ok_or_else(|| anyhow!("mount walk returned no handle"))?;
        self.fids.insert(
            newfid,
            Arc::new(MountFidEntry { path: new_path, handle: Mutex::new(Some(handle)) }),
        );
        Ok(WalkResult { qids, reached })
    }

    async fn open(&self, fid: u32, flags: u32) -> Result<OpenResult> {
        let entry = self.entry(fid)?;
        let mut guard = entry.handle.lock().await;
        let handle = guard.as_mut().ok_or_else(|| anyhow!("open: fid {fid} is clunked"))?;
        let mode = lopen_flags_to_mode(flags);
        self.mount
            .open(handle, mode, self.caller()?)
            .await
            .context("mount open failed")?;
        let qid = self.qid_of(handle).await?;
        Ok(OpenResult { qid, iounit: IOUNIT })
    }

    async fn read(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard.as_ref().ok_or_else(|| anyhow!("read: fid {fid} is clunked"))?;
        self.mount
            .read(handle, offset, count, self.caller()?)
            .await
            .context("mount read failed")
    }

    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> Result<u32> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard.as_ref().ok_or_else(|| anyhow!("write: fid {fid} is clunked"))?;
        self.mount
            .write(handle, offset, data, self.caller()?)
            .await
            .context("mount write failed")
    }

    async fn stat(&self, fid: u32) -> Result<StatResult> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard.as_ref().ok_or_else(|| anyhow!("stat: fid {fid} is clunked"))?;
        let st = self
            .mount
            .stat(handle, self.caller()?)
            .await
            .context("mount stat failed")?;
        let is_dir = st.qtype & QTDIR != 0;
        let mode = if is_dir { 0o040755 } else { 0o100644 };
        Ok(StatResult {
            qid: Qid { qtype: st.qtype, version: st.version, path: st.path },
            mode,
            size: st.size,
            mtime_sec: st.mtime,
        })
    }

    async fn readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        let entry = self.entry(fid)?;
        let entries = {
            let guard = entry.handle.lock().await;
            let handle = guard.as_ref().ok_or_else(|| anyhow!("readdir: fid {fid} is clunked"))?;
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
                Some(st) => Qid { qtype: st.qtype, version: st.version, path: st.path },
                None => Qid { qtype: if e.is_dir { QTDIR } else { 0 }, version: 0, path: 0 },
            };
            ReaddirEntry { qid, name: e.name.clone() }
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

    #[test]
    fn encode_dir_entries_emits_standard_rreaddir_records() {
        use crate::msg::parse_readdir_entries;

        let entries = vec![
            DirEntry { name: "a".into(), is_dir: true, size: 0, stat: None },
            DirEntry { name: "bb".into(), is_dir: false, size: 7, stat: None },
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
