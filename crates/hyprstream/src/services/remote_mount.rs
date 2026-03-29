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

impl Mount for RemoteModelMount {
    fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // First component is the model_ref (e.g., "qwen3:main"), rest is path within model tree.
        if components.is_empty() {
            return Err(MountError::InvalidArgument("empty path".into()));
        }

        let model_ref = components[0];
        let wnames: Vec<String> = components[1..].iter().map(|s| s.to_string()).collect();

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
            model_ref: model_ref.to_string(),
            qtype,
            opened: false,
        };
        self.fids.insert(local_fid, state);

        Ok(Fid::new(RemoteFidKey(local_fid)))
    }

    fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
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

    fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
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

    fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
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

    fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
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

    fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
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
            size: np_stat.length,
            name: np_stat.name.clone(),
            mtime: np_stat.mtime as u64,
        })
    }

    fn clunk(&self, fid: Fid, _caller: &Subject) {
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
/// We extract the name and qtype for DirEntry conversion.
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

        // 9P stat layout (simplified):
        //   size(2) + type(2) + dev(4) + qid(13) + mode(4) + atime(4) + mtime(4) + length(8) +
        //   name_len(2) + name + uid_len(2) + uid + gid_len(2) + gid + muid_len(2) + muid
        //
        // But the outer size has already been consumed. The inner layout is:
        //   type(2) + dev(4) + qid.type(1) + qid.vers(4) + qid.path(8) + mode(4) + atime(4) + mtime(4) + length(8)
        //   = 39 bytes before name_len
        if entry_data.len() < 41 {
            continue;
        }

        let qtype = entry_data[6]; // qid.type at offset 6 (after type(2) + dev(4))
        let length = u64::from_le_bytes(entry_data[27..35].try_into().unwrap_or([0; 8]));

        // name_len at offset 39
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
                size: length,
                name,
                mtime: 0,
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
}
