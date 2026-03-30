//! Generic remote mount — bridges the VFS `Mount` trait to any `FsClient`.
//!
//! `RemoteMount<C>` wraps an `FsClient` implementation and translates the
//! synchronous `Mount` trait calls into async `fs_*` operations. This is the
//! generic replacement for the old `RemoteModelMount` which was hard-wired to
//! `ModelClient`.
//!
//! ## Fid management
//!
//! The mount allocates local fid numbers and tracks them in a `DashMap`. Each
//! local fid maps to a `RemoteFidState` that stores the remote fid number
//! plus metadata (open mode, qtype).

use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_rpc::{FsClient, Subject};

use crate::{DirEntry, Fid, Mount, MountError, Stat};

/// Qtype constant for directories (9P2000).
const QTDIR: u8 = 0x80;

// ─────────────────────────────────────────────────────────────────────────────
// Remote fid state
// ─────────────────────────────────────────────────────────────────────────────

/// Per-fid state held locally.
#[derive(Clone, Debug)]
struct RemoteFidState {
    /// Fid number on the remote side.
    remote_fid: u32,
    /// Qtype from the walk response (QTDIR or QTFILE).
    qtype: u8,
    /// Whether open() has been called.
    opened: bool,
}

/// Newtype stored inside the opaque `Fid`.
#[derive(Clone, Debug)]
struct RemoteFidKey(u32);

// ─────────────────────────────────────────────────────────────────────────────
// RemoteMount<C>
// ─────────────────────────────────────────────────────────────────────────────

/// A generic `Mount` implementation that proxies 9P operations to any
/// `FsClient` implementation.
///
/// This replaces the old `RemoteModelMount` which was coupled to the generated
/// `ModelClient` type. Now any service can provide a mount by implementing
/// `FsClient` (or wrapping its generated client in an adapter).
pub struct RemoteMount<C: FsClient> {
    client: C,
    /// Local fid number → remote fid state.
    fids: DashMap<u32, RemoteFidState>,
    /// Monotonic fid allocator.
    next_fid: AtomicU32,
}

impl<C: FsClient> RemoteMount<C> {
    /// Create a new remote mount wrapping the given client.
    pub fn new(client: C) -> Self {
        Self {
            client,
            fids: DashMap::new(),
            next_fid: AtomicU32::new(1),
        }
    }

    /// Allocate a new local fid number.
    fn alloc_fid(&self) -> u32 {
        self.next_fid.fetch_add(1, Ordering::Relaxed)
    }

    /// Map a `String` error to a `MountError`.
    fn map_err(e: String) -> MountError {
        if e.contains("not found") || e.contains("No such") {
            MountError::NotFound(e)
        } else if e.contains("permission denied") {
            MountError::PermissionDenied(e)
        } else if e.contains("not a directory") {
            MountError::NotDirectory(e)
        } else if e.contains("is a directory") {
            MountError::IsDirectory(e)
        } else {
            MountError::Io(e)
        }
    }
}

#[async_trait]
impl<C: FsClient> Mount for RemoteMount<C> {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        if components.is_empty() {
            return Err(MountError::InvalidArgument("empty path".into()));
        }

        let wnames: Vec<String> = components.iter().map(|s| s.to_string()).collect();

        let local_fid = self.alloc_fid();
        let remote_newfid = local_fid; // Use same numbering for simplicity.

        let result = self.client.fs_walk(wnames, remote_newfid)
            .await
            .map_err(Self::map_err)?;

        let state = RemoteFidState {
            remote_fid: remote_newfid,
            qtype: result.qtype,
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

        let _result = self.client.fs_open(state.remote_fid, mode)
            .await
            .map_err(Self::map_err)?;
        state.opened = true;

        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        self.client.fs_read(state.remote_fid, offset, count)
            .await
            .map_err(Self::map_err)
    }

    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        self.client.fs_write(state.remote_fid, offset, data.to_vec())
            .await
            .map_err(Self::map_err)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        if state.qtype != QTDIR {
            return Err(MountError::NotDirectory(format!("fid {} is not a directory", local_id)));
        }

        // Read raw directory data from the remote.
        let data = self.client.fs_readdir(state.remote_fid, 0, 65536)
            .await
            .map_err(Self::map_err)?;

        // Parse packed 9P stat entries.
        let entries = parse_dir_stats(&data);
        Ok(entries)
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let key = fid.downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self.fids.get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let result = self.client.fs_stat(state.remote_fid)
            .await
            .map_err(Self::map_err)?;

        Ok(Stat {
            qtype: result.qtype,
            size: result.size,
            name: result.name,
            mtime: result.mtime,
        })
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        let Some(key) = fid.downcast_ref::<RemoteFidKey>() else { return };
        let local_id = key.0;

        if let Some((_, state)) = self.fids.remove(&local_id) {
            // Best-effort clunk — ignore errors.
            let _ = self.client.fs_clunk(state.remote_fid).await;
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
