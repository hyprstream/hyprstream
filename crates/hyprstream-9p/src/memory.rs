//! In-memory [`Backend`] for standalone translator tests.
//!
//! Holds a flat map of absolute path → file contents plus a per-fid map of
//! open state (offset cursor, resolved path). The translator drives this
//! exactly as it would drive a `ModelBackend` (capnp RPC), so tests exercise
//! the full 9P codec + fid-table + dispatch path with no network.

use std::collections::HashMap;
use parking_lot::Mutex;

use async_trait::async_trait;

use crate::backend::{Backend, OpenResult, StatResult, WalkResult};
use crate::msg::Qid;

/// Path → bytes for a single open file. Fids reference these by allocated id.
struct FileEntry {
    bytes: Vec<u8>,
    qid: Qid,
}

/// Per-fid live state: which path it points at (None = root).
#[derive(Clone)]
struct FidLink {
    path: Option<String>,
    qid: Qid,
    is_dir: bool,
}

/// In-memory backend.
#[derive(Default)]
pub struct MemoryBackend {
    files: Mutex<HashMap<String, FileEntry>>,
    fids: Mutex<HashMap<u32, FidLink>>,
    next_path: Mutex<u64>,
}

impl MemoryBackend {
    /// Add a file at an absolute path (e.g. `/hello.txt`).
    pub fn add_file(&self, path: &str, contents: &[u8]) {
        let mut path_no_root = path.trim_start_matches('/');
        // Treat top-level files as single-component names.
        if let Some(stripped) = path_no_root.strip_prefix('/') {
            path_no_root = stripped;
        }
        let mut files = self.files.lock();
        let qid = Qid {
            qtype: 0, // QTFILE
            version: 1,
            path: {
                let mut np = self.next_path.lock();
                *np += 1;
                *np
            },
        };
        files.insert(
            path_no_root.to_owned(),
            FileEntry { bytes: contents.to_vec(), qid },
        );
    }

    fn root_qid() -> Qid {
        Qid { qtype: 0x80, version: 1, path: 0 } // QTDIR
    }
}

#[async_trait]
impl Backend for MemoryBackend {
    async fn walk(
        &self,
        _fid: u32,
        newfid: u32,
        components: &[String],
    ) -> anyhow::Result<WalkResult> {
        // Empty walk = clone/attach to root.
        if components.is_empty() {
            let link = FidLink { path: None, qid: Self::root_qid(), is_dir: true };
            self.fids.lock().insert(newfid, link);
            return Ok(WalkResult { qids: vec![Self::root_qid()], reached: Vec::new() });
        }

        let name = &components[0];
        let files = self.files.lock();
        let entry = files
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("no such file: {name}"))?;
        let qid = entry.qid.clone();
        drop(files);

        let link = FidLink { path: Some(name.clone()), qid: qid.clone(), is_dir: false };
        self.fids.lock().insert(newfid, link);
        Ok(WalkResult { qids: vec![qid], reached: vec![name.clone()] })
    }

    async fn open(&self, fid: u32, _flags: u32) -> anyhow::Result<OpenResult> {
        let fids = self.fids.lock();
        let link = fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("unknown fid {fid}"))?;
        Ok(OpenResult { qid: link.qid.clone(), iounit: 8 * 1024 })
    }

    async fn read(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>> {
        let fids = self.fids.lock();
        let link = fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("unknown fid {fid}"))?;
        if link.is_dir {
            // Directory read: emit standard 9P2000.L Rreaddir dirent records
            // (`qid[13] offset[8] type[1] name[s]`) — the same wire format the
            // Mount export path and any standard client (incl. this crate's own
            // `client.rs`, which parses via `parse_readdir_entries`) expect.
            let files = self.files.lock();
            let records: Vec<crate::msg::ReaddirEntry> = files
                .iter()
                .map(|(name, entry)| crate::msg::ReaddirEntry {
                    qid: entry.qid.clone(),
                    name: name.clone(),
                })
                .collect();
            drop(files);
            return Ok(crate::msg::encode_readdir_page(&records, offset, count));
        }
        let path = link.path.clone().unwrap_or_default();
        let files = self.files.lock();
        let entry = files
            .get(&path)
            .ok_or_else(|| anyhow::anyhow!("fid {fid} has no backing file"))?;
        let off = offset as usize;
        if off >= entry.bytes.len() {
            return Ok(Vec::new());
        }
        let end = (off + count as usize).min(entry.bytes.len());
        Ok(entry.bytes[off..end].to_vec())
    }

    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> anyhow::Result<u32> {
        let fids = self.fids.lock();
        let link = fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("unknown fid {fid}"))?;
        if link.is_dir {
            anyhow::bail!("cannot write to directory fid {fid}");
        }
        let path = link.path.clone().unwrap_or_default();
        drop(fids);

        let mut files = self.files.lock();
        let entry = files
            .get_mut(&path)
            .ok_or_else(|| anyhow::anyhow!("fid {fid} has no backing file"))?;
        let off = offset as usize;
        if off + data.len() > entry.bytes.len() {
            entry.bytes.resize(off + data.len(), 0);
        }
        entry.bytes[off..off + data.len()].copy_from_slice(data);
        Ok(data.len() as u32)
    }

    async fn stat(&self, fid: u32) -> anyhow::Result<StatResult> {
        let fids = self.fids.lock();
        let link = fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("unknown fid {fid}"))?;
        let (size, mode) = if link.is_dir {
            (0, 0o040755)
        } else {
            let path = link.path.clone().unwrap_or_default();
            let files = self.files.lock();
            let len = files.get(&path).map(|e| e.bytes.len() as u64).unwrap_or(0);
            (len, 0o100644)
        };
        Ok(StatResult { qid: link.qid.clone(), mode, size, mtime_sec: 0 })
    }

    async fn readdir(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>> {
        // Same encoding as read() for directories.
        self.read(fid, offset, count).await
    }

    async fn clunk(&self, fid: u32) -> anyhow::Result<()> {
        self.fids.lock().remove(&fid);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn walk_read_roundtrip() {
        let b = MemoryBackend::default();
        b.add_file("/greeting", b"hi");
        let w = b.walk(0, 1, &["greeting".into()]).await.unwrap();
        assert_eq!(w.qids.len(), 1);
        let o = b.open(1, 0).await.unwrap();
        assert_eq!(o.iounit, 8 * 1024);
        let d = b.read(1, 0, 16).await.unwrap();
        assert_eq!(&d, b"hi");
        b.clunk(1).await.unwrap();
    }

    #[tokio::test]
    async fn write_extends_file() {
        let b = MemoryBackend::default();
        b.add_file("/log", b"");
        b.walk(0, 1, &["log".into()]).await.unwrap();
        b.write(1, 0, b"abc").await.unwrap();
        let d = b.read(1, 0, 8).await.unwrap();
        assert_eq!(&d, b"abc");
    }
}
