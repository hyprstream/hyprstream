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

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount};
use tokio::sync::Mutex;

use crate::backend::{Backend, OpenResult, StatResult, WalkResult};
use crate::msg::{self, Qid, ReaddirEntry};

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
pub struct MountBackend {
    mount: Arc<dyn Mount>,
    subject: Subject,
    fids: DashMap<u32, Arc<MountFidEntry>>,
}

impl MountBackend {
    /// Wrap `mount` as the 9P export root for `subject`.
    pub fn new(mount: Arc<dyn Mount>, subject: Subject) -> Self {
        Self { mount, subject, fids: DashMap::new() }
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
            .stat(handle, &self.subject)
            .await
            .map_err(|e| anyhow!("mount stat failed: {e}"))?;
        Ok(Qid { qtype: st.qtype, version: st.version, path: st.path })
    }
}

#[async_trait]
impl Backend for MountBackend {
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
        let mut new_path = parent_path;
        new_path.extend(components.iter().cloned());

        let refs: Vec<&str> = new_path.iter().map(String::as_str).collect();
        let handle = self
            .mount
            .walk(&refs, &self.subject)
            .await
            .map_err(|e| anyhow!("mount walk failed: {e}"))?;

        // Mirror `ModelBackend::walk`: return the single leaf qid (the translator
        // records its qtype for the new fid; clients here walk one hop at a time).
        let qid = self.qid_of(&handle).await?;
        self.fids.insert(
            newfid,
            Arc::new(MountFidEntry { path: new_path, handle: Mutex::new(Some(handle)) }),
        );
        Ok(WalkResult { qids: vec![qid] })
    }

    async fn open(&self, fid: u32, flags: u32) -> Result<OpenResult> {
        let entry = self.entry(fid)?;
        let mut guard = entry.handle.lock().await;
        let handle = guard.as_mut().ok_or_else(|| anyhow!("open: fid {fid} is clunked"))?;
        let mode = lopen_flags_to_mode(flags);
        self.mount
            .open(handle, mode, &self.subject)
            .await
            .map_err(|e| anyhow!("mount open failed: {e}"))?;
        let qid = self.qid_of(handle).await?;
        Ok(OpenResult { qid, iounit: IOUNIT })
    }

    async fn read(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard.as_ref().ok_or_else(|| anyhow!("read: fid {fid} is clunked"))?;
        self.mount
            .read(handle, offset, count, &self.subject)
            .await
            .map_err(|e| anyhow!("mount read failed: {e}"))
    }

    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> Result<u32> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard.as_ref().ok_or_else(|| anyhow!("write: fid {fid} is clunked"))?;
        self.mount
            .write(handle, offset, data, &self.subject)
            .await
            .map_err(|e| anyhow!("mount write failed: {e}"))
    }

    async fn stat(&self, fid: u32) -> Result<StatResult> {
        let entry = self.entry(fid)?;
        let guard = entry.handle.lock().await;
        let handle = guard.as_ref().ok_or_else(|| anyhow!("stat: fid {fid} is clunked"))?;
        let st = self
            .mount
            .stat(handle, &self.subject)
            .await
            .map_err(|e| anyhow!("mount stat failed: {e}"))?;
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
                .readdir(handle, &self.subject)
                .await
                .map_err(|e| anyhow!("mount readdir failed: {e}"))?
        };
        Ok(encode_dir_entries(&entries, offset, count))
    }

    async fn clunk(&self, fid: u32) -> Result<()> {
        // Drop local state and hand the mount handle back for release.
        if let Some((_, entry)) = self.fids.remove(&fid) {
            let handle = entry.handle.lock().await.take();
            if let Some(handle) = handle {
                self.mount.clunk(handle, &self.subject).await;
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
