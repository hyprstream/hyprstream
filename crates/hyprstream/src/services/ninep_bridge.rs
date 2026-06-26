//! Shared sync→async bridge for 9P-style `Mount` implementations that proxy
//! to a Cap'n Proto RPC service (`nine.capnp` envelope).
//!
//! Both `RemoteModelMount` (1-level scope: first path component = model ref)
//! and `RemoteRegistryMount` (2-level scope: repo → worktree, with async UUID
//! resolution) translate synchronous VFS `Mount` trait calls into async RPC
//! requests by embedding a dedicated single-threaded tokio runtime and
//! `block_on`-ing each op. The runtime, the local-fid allocator, the
//! `DashMap<u32, S>` that tracks remote fid state, the `anyhow → MountError`
//! mapping, the opaque `RemoteFidKey` newtype stored in `Fid`, and the packed
//! 9P stat parser used for directory reads are identical across the two
//! mounts — only the RPC client type, the scope extraction, and which RPC
//! method names / directory byte format differ.
//!
//! This module factors out the shared pieces:
//!
//! - [`NinePBridge`]`<S>` — owns the runtime + fid map + fid allocator. Each
//!   mount holds one of these and supplies its own fid-state type `S`.
//! - [`map_err`] — `anyhow::Error → MountError` classification.
//! - [`RemoteFidKey`] — the opaque newtype stored inside `hyprstream_vfs::Fid`.
//! - [`parse_dir_stats`] — decode packed 9P stat entries from a directory
//!   `read()` response (the format the model service's `SyntheticTree` emits).
//!
//! What stays mount-specific:
//!
//! - the top-level `Mount` impl (walk's scope extraction differs: 1-level vs
//!   2-level with async `get_by_name` resolution),
//! - the request-struct construction (each generated client module has its
//!   own `NpWalk`/`NpOpen`/... types),
//! - the stat RPC method name (`stat` on `ModelFsClient`, `np_stat` on
//!   `WorktreeClient`),
//! - the directory-entry byte format (packed 9P stats for the model service
//!   vs `name_len+name+is_dir+size` for the registry worktree service).

use std::sync::atomic::{AtomicU32, Ordering};

use dashmap::DashMap;
use hyprstream_vfs::{DirEntry, Fid, MountError, Stat};

use crate::services::types::QTDIR;

// ─────────────────────────────────────────────────────────────────────────────
// Opaque fid key
// ─────────────────────────────────────────────────────────────────────────────

/// Newtype stored inside the opaque `hyprstream_vfs::Fid`.
///
/// Wraps the locally-allocated fid number so that subsequent `open`/`read`/
/// `write`/`readdir`/`stat`/`clunk` calls can look the remote fid state back
/// up in the bridge's `DashMap`.
#[derive(Clone, Debug)]
pub struct RemoteFidKey(pub u32);

// ─────────────────────────────────────────────────────────────────────────────
// Bridge: runtime + fid map + allocator
// ─────────────────────────────────────────────────────────────────────────────

/// Shared sync→async bridge plumbing for a 9P `Mount` proxying to a Cap'n
/// Proto RPC service.
///
/// Owns:
/// - a dedicated single-threaded tokio runtime (`rt`) used to `block_on` each
///   async RPC op from the sync `Mount` trait,
/// - a `DashMap<u32, S>` mapping locally-allocated fid numbers to per-fid
///   state `S` (the mount chooses what to store — typically the remote fid
///   number, the scoped RPC client, the walk-response qtype, and an
///   `opened` flag),
/// - a monotonic fid allocator (`next_fid`).
///
/// Each op in a `Mount` impl follows the same shape:
///
/// ```ignore
/// let local_id = fid_key(fid)?.0;
/// let state = bridge.get(local_id)?;      // or get_mut for open
/// // ...build request, bridge.rt().block_on(client.op(&req)).map_err(map_err)?...
/// ```
///
/// `fid_key` and `map_err` are module-level free functions (not methods on
/// the generic `NinePBridge<S>`) so call sites don't need to spell out the
/// fid-state type parameter `S`.
pub struct NinePBridge<S> {
    rt: tokio::runtime::Runtime,
    fids: DashMap<u32, S>,
    next_fid: AtomicU32,
}

impl<S> NinePBridge<S> {
    /// Construct a new bridge with a fresh single-threaded runtime and an
    /// empty fid map starting at fid 1 (fid 0 is reserved for the 9P root).
    #[allow(clippy::expect_used)] // Runtime creation is infallible in practice
    pub fn new(label: &str) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("failed to create tokio runtime for {label}: {e}"));
        Self {
            rt,
            fids: DashMap::new(),
            next_fid: AtomicU32::new(1),
        }
    }

    /// Allocate a new local fid number (monotonic, starts at 1).
    pub fn alloc_fid(&self) -> u32 {
        self.next_fid.fetch_add(1, Ordering::Relaxed)
    }

    /// Borrow the embedded runtime — used to `block_on` async RPC ops from
    /// the sync `Mount` trait.
    pub fn rt(&self) -> &tokio::runtime::Runtime {
        &self.rt
    }

    /// Insert fid state for a locally-allocated fid (called from `walk`).
    pub fn insert(&self, local_fid: u32, state: S) {
        self.fids.insert(local_fid, state);
    }

    /// Look up a shared reference to fid state.
    ///
    /// Convenience over `fids.get` that yields a `MountError::NotFound` for
    /// a stale/unknown local fid.
    pub fn get(&self, local_fid: u32) -> Result<dashmap::mapref::one::Ref<'_, u32, S>, MountError> {
        self.fids
            .get(&local_fid)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_fid)))
    }

    /// Look up a mutable reference to fid state (used by `open`, which flips
    /// the `opened` flag).
    pub fn get_mut(
        &self,
        local_fid: u32,
    ) -> Result<dashmap::mapref::one::RefMut<'_, u32, S>, MountError> {
        self.fids
            .get_mut(&local_fid)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_fid)))
    }

    /// Remove fid state (called from `clunk`). Returns the state so the
    /// caller can issue the best-effort remote clunk with the remote fid.
    pub fn remove(&self, local_fid: u32) -> Option<S> {
        self.fids.remove(&local_fid).map(|(_, s)| s)
    }
}

/// Downcast the opaque `Fid` to its `RemoteFidKey`, returning
/// `MountError::InvalidArgument` on a type mismatch. All `Mount` ops start
/// with this. Free-standing so callers don't need to name the bridge's
/// fid-state type parameter.
pub fn fid_key(fid: &Fid) -> Result<&RemoteFidKey, MountError> {
    fid.downcast_ref::<RemoteFidKey>()
        .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))
}

/// Classify an `anyhow::Error` (as surfaced by the generated RPC clients)
/// into a `MountError` by string-matching the common 9P / filesystem error
/// phrases. Identical for every backing service. Free-standing so callers
/// don't need to name the bridge's fid-state type parameter.
pub fn map_err(e: anyhow::Error) -> MountError {
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

// ─────────────────────────────────────────────────────────────────────────────
// Directory stat parsing (packed 9P stats)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse packed 9P stat entries from a directory `read()` response.
///
/// Each stat entry is: `[2-byte LE size][stat bytes]`. We extract qid
/// (type/version/path), length, mtime, and name for `DirEntry`.
///
/// This is the format the model service's `SyntheticTree` emits for directory
/// reads. The registry worktree service uses a different per-entry format
/// (`name_len+name+is_dir+size`) and parses it locally — see
/// `parse_worktree_dir_entries` in `remote_registry_mount.rs`.
pub fn parse_dir_stats(data: &[u8]) -> Vec<DirEntry> {
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
        assert_eq!(
            stat.version, qvers,
            "qid version must be decoded, not flattened"
        );
        assert_eq!(stat.path, qpath, "qid path must be decoded, not flattened");
        assert_eq!(stat.size, length);
        assert_eq!(stat.mtime, mtime as u64);
    }

    /// `map_err` classifies the common 9P/filesystem error phrases.
    #[test]
    fn map_err_classifies_common_phrases() {
        use hyprstream_vfs::MountError;

        let classify = |msg: &str| map_err(anyhow::anyhow!("{}", msg));

        assert!(matches!(
            classify("not found here"),
            MountError::NotFound(_)
        ));
        assert!(matches!(classify("No such file"), MountError::NotFound(_)));
        assert!(matches!(
            classify("permission denied"),
            MountError::PermissionDenied(_)
        ));
        assert!(matches!(
            classify("not a directory"),
            MountError::NotDirectory(_)
        ));
        assert!(matches!(
            classify("is a directory"),
            MountError::IsDirectory(_)
        ));
        assert!(matches!(classify("disk on fire"), MountError::Io(_)));
    }

    /// `NinePBridge` allocates fids monotonically from 1 and tracks state.
    #[test]
    fn bridge_allocates_and_tracks_fids() {
        let bridge = NinePBridge::<&'static str>::new("test");
        assert_eq!(bridge.alloc_fid(), 1);
        assert_eq!(bridge.alloc_fid(), 2);

        bridge.insert(1, "root");
        assert_eq!(bridge.get(1).map(|r| *r).unwrap_or("MISSING"), "root");
        assert!(bridge.get(99).is_err());

        {
            if let Ok(mut r) = bridge.get_mut(1) {
                *r = "changed";
            } else {
                panic!("fid 1 must be present for mutation");
            }
        }
        assert_eq!(bridge.get(1).map(|r| *r).unwrap_or("MISSING"), "changed");

        assert_eq!(bridge.remove(1), Some("changed"));
        assert_eq!(bridge.remove(1), None);
    }
}
