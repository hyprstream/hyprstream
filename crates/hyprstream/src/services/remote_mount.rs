//! Remote model mount — bridges the VFS `Mount` trait to the generated `ModelClient` RPC.
//!
//! `RemoteModelMount` wraps a `ModelClient` and translates synchronous `Mount`
//! trait calls into async RPC requests over ZMQ. This allows the TUI shell's
//! VFS namespace to serve `/srv/model` by proxying 9P operations to the model
//! service, which in turn delegates to its local `FsHandler` / `SyntheticTree`.
//!
//! ## Fid management
//!
//! The mount allocates local fid numbers and tracks them in a `DashMap`. Each
//! local fid maps to a `RemoteFid` that stores the remote fid number returned
//! by walk, plus metadata (open mode, qtype) for stat/readdir synthesis.
//!
//! ## Timeout / graceful degradation
//!
//! A dedicated single-threaded tokio runtime (`self.rt`) drives the async
//! client. If the model service is unreachable, `block_on` will hang until
//! the ZMQ socket timeout fires (set on the underlying socket). Callers
//! see `MountError::Io("service unreachable: ...")`.

use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;

use crate::services::generated::model_client::{ModelFsClient, ModelClient};
use crate::services::types::QTDIR;

// ─────────────────────────────────────────────────────────────────────────────
// Remote fid state
// ─────────────────────────────────────────────────────────────────────────────

/// Per-fid state held locally.
#[derive(Clone, Debug)]
struct RemoteFidState {
    /// Fid number on the remote side.
    remote_fid: u32,
    /// Model reference this fid belongs to (e.g., "qwen3:main").
    model_ref: String,
    /// Qtype from the walk response (QTDIR or QTFILE).
    qtype: u8,
    /// Whether open() has been called.
    opened: bool,
}

/// Newtype stored inside the opaque `Fid`.
#[derive(Clone, Debug)]
struct RemoteFidKey(u32);

// ─────────────────────────────────────────────────────────────────────────────
// RemoteModelMount
// ─────────────────────────────────────────────────────────────────────────────

/// A `Mount` implementation that proxies 9P operations to the model service
/// via the generated `ModelClient` RPC.
pub struct RemoteModelMount {
    client: ModelClient,
    rt: tokio::runtime::Runtime,
    /// Local fid number → remote fid state.
    fids: DashMap<u32, RemoteFidState>,
    /// Monotonic fid allocator.
    next_fid: AtomicU32,
}

impl RemoteModelMount {
    /// Create a new remote mount wrapping the given model client.
    #[allow(clippy::expect_used)] // Runtime creation is infallible in practice
    pub fn new(client: ModelClient) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for RemoteModelMount");
        Self {
            client,
            rt,
            fids: DashMap::new(),
            next_fid: AtomicU32::new(1),
        }
    }

    /// Allocate a new local fid number.
    fn alloc_fid(&self) -> u32 {
        self.next_fid.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the scoped fs client for a model reference.
    fn fs_client(&self, model_ref: &str) -> ModelFsClient {
        self.client.fs(model_ref)
    }

    /// Map an `anyhow::Error` to a `MountError::Io`.
    fn map_err(e: anyhow::Error) -> MountError {
        let msg = e.to_string();
        if msg.contains("not found") || msg.contains("No such") {
            MountError::NotFound(msg)
        } else if msg.contains("permission denied") {
            MountError::PermissionDenied(msg)
        } else if msg.contains("not a directory") {
            MountError::NotDirectory(msg)
        } else if msg.contains("is a directory") {
            MountError::IsDirectory(msg)
        } else {
            MountError::Io(msg)
        }
    }
}

#[async_trait]
impl Mount for RemoteModelMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // First component is the model_ref (e.g., "qwen3:main"), rest is path within model tree.
        if components.is_empty() {
            return Err(MountError::InvalidArgument("empty path".into()));
        }

        let model_ref = components[0];
        let wnames: Vec<String> = components[1..].iter().map(std::string::ToString::to_string).collect();

        let local_fid = self.alloc_fid();
        let remote_newfid = local_fid; // Use same numbering for simplicity.

        let fs = self.fs_client(model_ref);
        let walk_req = crate::services::generated::model_client::NpWalk {
            fid: 0, // root
            newfid: remote_newfid,
            wnames,
        };

        let result = self.rt.block_on(fs.walk(&walk_req)).map_err(Self::map_err)?;

        // Determine qtype from the walk response qid.
        let qtype = result.qid.qtype;

        let state = RemoteFidState {
            remote_fid: remote_newfid,
            model_ref: model_ref.to_owned(),
            qtype,
            opened: false,
        };
        self.fids.insert(local_fid, state);

        Ok(Fid::new(RemoteFidKey(local_fid)))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let mut state = self.fids.get_mut(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let fs = self.fs_client(&state.model_ref);
        let open_req = crate::services::generated::model_client::NpOpen {
            fid: state.remote_fid,
            mode,
        };

        let _result = self.rt.block_on(fs.open(&open_req)).map_err(Self::map_err)?;
        state.opened = true;

        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let fs = self.fs_client(&state.model_ref);
        let read_req = crate::services::generated::model_client::NpRead {
            fid: state.remote_fid,
            offset,
            count,
        };

        let result = self.rt.block_on(fs.read(&read_req)).map_err(Self::map_err)?;
        Ok(result.data)
    }

    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let fs = self.fs_client(&state.model_ref);
        let write_req = crate::services::generated::model_client::NpWrite {
            fid: state.remote_fid,
            offset,
            data: data.to_vec(),
        };

        let result = self.rt.block_on(fs.write(&write_req)).map_err(Self::map_err)?;
        Ok(result.count)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        // 9P doesn't have a dedicated readdir — we read the directory fid and parse
        // the stat entries from the raw bytes. However, the model service's SyntheticTree
        // encodes directory listings in its read() response as newline-separated entries.
        //
        // For now, do a read at offset 0 and parse the response.
        // The model service returns stat-encoded directory entries.
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        if state.qtype != QTDIR {
            return Err(MountError::NotDirectory(format!("fid {} is not a directory", local_id)));
        }

        // Read directory data from remote.
        let fs = self.fs_client(&state.model_ref);
        let read_req = crate::services::generated::model_client::NpRead {
            fid: state.remote_fid,
            offset: 0,
            count: 65536, // Large enough for most directory listings.
        };

        let result = self.rt.block_on(fs.read(&read_req)).map_err(Self::map_err)?;

        // Parse stat entries from the raw 9P directory read data.
        // Directory reads return packed NpStat entries. Each entry is preceded
        // by a 2-byte little-endian size. We extract name + qtype from each.
        let entries = parse_dir_stats(&result.data);
        Ok(entries)
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let fs = self.fs_client(&state.model_ref);
        let stat_req = crate::services::generated::model_client::NpStatReq {
            fid: state.remote_fid,
        };

        let result = self.rt.block_on(fs.stat(&stat_req)).map_err(Self::map_err)?;

        // Convert the RPC RStat → VFS Stat.
        let np_stat = &result.stat;

        Ok(Stat {
            qtype: np_stat.qid.qtype,
            // Thread the wire qid version/path through instead of discarding
            // them. `nine.capnp` already carries these; we were flattening the
            // qid to qtype only. See the qid-soundness invariant on
            // `hyprstream_vfs::Stat` — these are advisory identity hints, not
            // yet a strong identity (content-CID qid lands in #387).
            version: np_stat.qid.version,
            path: np_stat.qid.path,
            size: np_stat.length,
            name: np_stat.name.clone(),
            mtime: np_stat.mtime as u64,
        })
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        let Some(key) = fid.downcast_ref::<RemoteFidKey>() else { return };
        let local_id = key.0;

        if let Some((_, state)) = self.fids.remove(&local_id) {
            let fs = self.fs_client(&state.model_ref);
            let clunk_req = crate::services::generated::model_client::NpClunk {
                fid: state.remote_fid,
            };

            // Best-effort clunk — ignore errors.
            let _ = self.rt.block_on(fs.clunk(&clunk_req));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Directory stat parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse packed 9P stat entries from a directory read response.
///
/// Each stat entry is: `[2-byte LE size][stat bytes]`.
/// We extract qid (type/version/path), length, mtime, and name for DirEntry.
fn parse_dir_stats(data: &[u8]) -> Vec<DirEntry> {
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + 2 <= data.len() {
        let entry_size = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + entry_size > data.len() {
            break;
        }

        let entry_data = &data[offset..offset + entry_size];
        offset += entry_size;

        // 9P stat layout. The outer 2-byte size has already been consumed; the
        // inner bytes are:
        //   type(2) dev(4) qid.type(1) qid.vers(4) qid.path(8)
        //   mode(4) atime(4) mtime(4) length(8)   = 39 bytes
        //   then name_len(2) name ...
        //
        // Offsets:
        //   qid.type  @ 6            (1 byte)
        //   qid.vers  @ 7..11        (u32 LE)
        //   qid.path  @ 11..19       (u64 LE)
        //   mode      @ 19..23
        //   atime     @ 23..27
        //   mtime     @ 27..31       (u32 LE)
        //   length    @ 31..39       (u64 LE)
        //   name_len  @ 39..41       (u16 LE)
        if entry_data.len() < 41 {
            continue;
        }

        let qtype = entry_data[6];
        let qvers = u32::from_le_bytes(entry_data[7..11].try_into().unwrap_or([0; 4]));
        let qpath = u64::from_le_bytes(entry_data[11..19].try_into().unwrap_or([0; 8]));
        let mtime = u32::from_le_bytes(entry_data[27..31].try_into().unwrap_or([0; 4])) as u64;
        let length = u64::from_le_bytes(entry_data[31..39].try_into().unwrap_or([0; 8]));

        let name_len = u16::from_le_bytes([entry_data[39], entry_data[40]]) as usize;
        if entry_data.len() < 41 + name_len {
            continue;
        }
        let name = String::from_utf8_lossy(&entry_data[41..41 + name_len]).to_string();

        entries.push(DirEntry {
            name: name.clone(),
            is_dir: qtype == QTDIR,
            size: length,
            stat: Some(Stat {
                qtype,
                version: qvers,
                path: qpath,
                size: length,
                name,
                mtime,
            }),
        });
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_dir_stats() {
        assert!(parse_dir_stats(&[]).is_empty());
    }

    #[test]
    fn parse_truncated_dir_stats() {
        // Size says 100 bytes but only 5 available — should not panic.
        let data = [100, 0, 0, 0, 0];
        assert!(parse_dir_stats(&data).is_empty());
    }

    /// Build one packed 9P stat entry (inner layout, no outer 2-byte size) and
    /// verify qid type/version/path, length, mtime, and name are all decoded.
    #[test]
    fn parse_dir_stats_decodes_qid_and_fields() {
        let qtype: u8 = 0x80; // QTDIR
        let qvers: u32 = 0x0A0B_0C0D;
        let qpath: u64 = 0x0102_0304_0506_0708;
        let mtime: u32 = 0x1122_3344;
        let length: u64 = 0x99AA_BBCC_DDDE_EEF0;
        let name = b"model-dir";

        // Inner stat: type(2) dev(4) qid.type(1) qid.vers(4) qid.path(8)
        //             mode(4) atime(4) mtime(4) length(8) name_len(2) name
        let mut inner = Vec::new();
        inner.extend_from_slice(&0u16.to_le_bytes()); // type
        inner.extend_from_slice(&0u32.to_le_bytes()); // dev
        inner.push(qtype);
        inner.extend_from_slice(&qvers.to_le_bytes());
        inner.extend_from_slice(&qpath.to_le_bytes());
        inner.extend_from_slice(&0u32.to_le_bytes()); // mode
        inner.extend_from_slice(&0u32.to_le_bytes()); // atime
        inner.extend_from_slice(&mtime.to_le_bytes());
        inner.extend_from_slice(&length.to_le_bytes());
        inner.extend_from_slice(&(name.len() as u16).to_le_bytes());
        inner.extend_from_slice(name);

        // Prepend the 2-byte LE entry size.
        let mut data = (inner.len() as u16).to_le_bytes().to_vec();
        data.extend_from_slice(&inner);

        let entries = parse_dir_stats(&data);
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.name, "model-dir");
        assert!(e.is_dir);
        assert_eq!(e.size, length);
        let stat = match e.stat.as_ref() {
            Some(s) => s,
            None => panic!("stat must be present on parsed dir entry"),
        };
        assert_eq!(stat.qtype, qtype);
        assert_eq!(stat.version, qvers, "qid version must be decoded, not flattened");
        assert_eq!(stat.path, qpath, "qid path must be decoded, not flattened");
        assert_eq!(stat.size, length);
        assert_eq!(stat.mtime, mtime as u64);
    }
}
