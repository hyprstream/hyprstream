//! Remote registry mount — bridges the VFS `Mount` trait to the registry
//! service's worktree filesystem via the generated `WorktreeClient` RPC.
//!
//! `RemoteRegistryMount` is the registry analogue of `RemoteModelMount`:
//! where `RemoteModelMount` proxies 9P operations to the model service's
//! per-`modelRef` synthetic tree (1 path level: first component = modelRef),
//! this mount proxies to the registry service's per-`worktree` real
//! filesystem — 2 path levels: first = repo name, second = worktree name,
//! the rest = in-worktree wnames. Repo names resolve to UUIDs via the
//! registry's `get_by_name` RPC.
//!
//! ## Real qids
//!
//! Now that #388 widened `vfs::Stat` with `version` + `path`, this mount
//! threads the wire qid `{qtype, version=ctime, path=ino}` from the registry
//! service's `RStat`/`RWalk` responses straight through — that is the whole
//! point of the qid-soundness work. See the invariant on
//! `hyprstream_vfs::Stat`: these are advisory hints, not yet a strong
//! identity (content-CID qid lands in #387).
//!
//! ## Fid management
//!
//! The mount allocates local fid numbers and tracks them in a `DashMap`.
//! Each local fid maps to a `RemoteFid` that stores the resolved
//! `WorktreeClient` (curried with `repo_id` + `worktree_name`), the remote
//! fid number returned by walk, plus qtype for stat/readdir synthesis.
//!
//! ## Timeout / graceful degradation
//!
//! A dedicated single-threaded tokio runtime (`self.rt`) drives the async
//! client. If the registry service is unreachable, `block_on` will hang
//! until the ZMQ socket timeout fires (set on the underlying socket).
//! Callers see `MountError::Io("service unreachable: ...")`.

use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;

use crate::services::generated::registry_client::{
    NpClunk, NpCreate, NpOpen, NpRead, NpStatReq, NpWalk, NpWrite, RegistryClient, WorktreeClient,
};
use crate::services::types::QTDIR;

// ─────────────────────────────────────────────────────────────────────────────
// Remote fid state
// ─────────────────────────────────────────────────────────────────────────────

/// Per-fid state held locally.
///
/// `worktree` is the curried `WorktreeClient` (already scoped to a specific
/// repo + worktree name) so subsequent open/read/write/stat/clunk calls hit
/// the right worktree without re-resolving the scope on every operation.
struct RemoteFidState {
    /// Fid number on the remote side.
    remote_fid: u32,
    /// Worktree client scoped to the resolved repo + worktree.
    worktree: WorktreeClient,
    /// Qtype from the walk response (QTDIR or QTFILE).
    qtype: u8,
    /// Whether open() has been called.
    opened: bool,
}

/// Newtype stored inside the opaque `Fid`.
#[derive(Clone, Debug)]
struct RemoteFidKey(u32);

// ─────────────────────────────────────────────────────────────────────────────
// RemoteRegistryMount
// ─────────────────────────────────────────────────────────────────────────────

/// A `Mount` implementation that proxies 9P operations to the registry
/// service's worktree filesystem via the generated `WorktreeClient` RPC.
///
/// Path layout under the mount point:
///   `/{repo_name}/{worktree_name}/<...rest...>`
///
/// `repo_name` is resolved to the registry UUID via `get_by_name` on the
/// first walk; `worktree_name` selects the worktree. The remainder is
/// forwarded to the worktree's 9P `walk` as `wnames`.
pub struct RemoteRegistryMount {
    client: RegistryClient,
    rt: tokio::runtime::Runtime,
    /// Local fid number → remote fid state.
    fids: DashMap<u32, RemoteFidState>,
    /// Monotonic fid allocator.
    next_fid: AtomicU32,
}

impl RemoteRegistryMount {
    /// Create a new remote registry mount wrapping the given registry client.
    #[allow(clippy::expect_used)] // Runtime creation is infallible in practice
    pub fn new(client: RegistryClient) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for RemoteRegistryMount");
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

    /// Resolve `repo_name` to a `WorktreeClient` scoped to `worktree_name`.
    ///
    /// Uses the registry's `get_by_name` to look up the repo UUID, then
    /// curries `repo(id).worktree(name)` — exactly the chain the worktree
    /// helpers (`worktree_helpers.rs`) and the CLI git handlers use.
    fn resolve_worktree(
        &self,
        repo_name: &str,
        worktree_name: &str,
    ) -> Result<WorktreeClient, MountError> {
        // NB: `RegistryClient` has an inherent `clone(&self, &CloneRequest)`
        // RPC method (it clones a git repo) that shadows `Clone::clone`.
        // Fully-qualify so we get the cheap Rc clone, not a repo clone RPC.
        let client = Clone::clone(&self.client);
        let repo_name = repo_name.to_owned();
        let worktree_name = worktree_name.to_owned();
        // block_on: the Mount trait is sync; the registry service is async
        // over ZMQ. Same pattern as RemoteModelMount.
        self.rt.block_on(async move {
            let tracked = client
                .get_by_name(&repo_name)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let repo = client.repo(&tracked.id);
            Ok(repo.worktree(&worktree_name))
        }).map_err(Self::map_err)
    }

    /// Map an `anyhow::Error` to a `MountError`.
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
impl Mount for RemoteRegistryMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // 2-level scope: first = repo name, second = worktree name, rest = wnames.
        if components.len() < 2 {
            return Err(MountError::InvalidArgument(format!(
                "registry walk needs at least /{{repo}}/{{worktree}}; got {:?}",
                components
            )));
        }

        let repo_name = components[0];
        let worktree_name = components[1];
        let wnames: Vec<String> = components[2..]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();

        let worktree = self.resolve_worktree(repo_name, worktree_name)?;

        let local_fid = self.alloc_fid();
        let remote_newfid = local_fid; // Use same numbering for simplicity.

        let walk_req = NpWalk {
            fid: 0, // root of the worktree
            newfid: remote_newfid,
            wnames,
        };

        let result = self
            .rt
            .block_on(worktree.walk(&walk_req))
            .map_err(Self::map_err)?;

        // qid.type from the walk response tells us dir vs file.
        let qtype = result.qid.qtype;

        let state = RemoteFidState {
            remote_fid: remote_newfid,
            worktree,
            qtype,
            opened: false,
        };
        self.fids.insert(local_fid, state);

        Ok(Fid::new(RemoteFidKey(local_fid)))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        let key = fid
            .downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let mut state = self
            .fids
            .get_mut(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let open_req = NpOpen {
            fid: state.remote_fid,
            mode,
        };

        let _result = self
            .rt
            .block_on(state.worktree.open(&open_req))
            .map_err(Self::map_err)?;
        state.opened = true;

        Ok(())
    }

    /// Create a file/dir under a walked *directory* fid, then open it.
    ///
    /// This is the writable-layer primitive that powers the union's copy-up
    /// (#394). The registry service's `handle_create` consumes the directory
    /// fid and replaces it with an opened fid on the new file, so on success we
    /// flip `opened = true` locally and thread the new file's qid back as the
    /// returned `Stat` (the union's copy-up caller uses it for logging only —
    /// it does not key authz on it; see the qid-soundness invariant on
    /// `hyprstream_vfs::Stat`).
    async fn create(
        &self,
        fid: &mut Fid,
        name: &str,
        perm: u32,
        mode: u8,
        _caller: &Subject,
    ) -> Result<Stat, MountError> {
        let key = fid
            .downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let mut state = self
            .fids
            .get_mut(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let create_req = NpCreate {
            fid: state.remote_fid,
            name: name.to_owned(),
            perm,
            mode,
        };

        let result = self
            .rt
            .block_on(state.worktree.create(&create_req))
            .map_err(Self::map_err)?;

        // The directory fid has been replaced on the server with the opened new
        // file. Mirror that transition locally so subsequent read/write/stat hit
        // the new file rather than a stale directory handle.
        state.qtype = result.qid.qtype;
        state.opened = true;

        Ok(Stat {
            qtype: result.qid.qtype,
            // Advisory identity hint only — see the qid-soundness invariant on
            // `hyprstream_vfs::Stat`. Content-CID qid lands in #387.
            version: result.qid.version,
            path: result.qid.path,
            size: 0,
            name: name.to_owned(),
            mtime: 0,
        })
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let key = fid
            .downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self
            .fids
            .get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let read_req = NpRead {
            fid: state.remote_fid,
            offset,
            count,
        };

        let result = self
            .rt
            .block_on(state.worktree.read(&read_req))
            .map_err(Self::map_err)?;
        Ok(result.data)
    }

    async fn write(
        &self,
        fid: &Fid,
        offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let key = fid
            .downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self
            .fids
            .get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let write_req = NpWrite {
            fid: state.remote_fid,
            offset,
            data: data.to_vec(),
        };

        let result = self
            .rt
            .block_on(state.worktree.write(&write_req))
            .map_err(Self::map_err)?;
        Ok(result.count)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        // The registry worktree service encodes directory listings in its
        // read() response using the per-entry format documented in
        // `WorktreeRequest.read`:
        //   name_len(u32) + name(utf8) + is_dir(u8) + size(u64)
        // (See `WorktreeClient::list_dir_path` in worktree_helpers.rs.)
        let key = fid
            .downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self
            .fids
            .get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        if state.qtype != QTDIR {
            return Err(MountError::NotDirectory(format!(
                "fid {} is not a directory",
                local_id
            )));
        }

        // Read directory data from the worktree service.
        let read_req = NpRead {
            fid: state.remote_fid,
            offset: 0,
            count: 65536, // Large enough for most directory listings.
        };

        let result = self
            .rt
            .block_on(state.worktree.read(&read_req))
            .map_err(Self::map_err)?;

        Ok(parse_worktree_dir_entries(&result.data))
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let key = fid
            .downcast_ref::<RemoteFidKey>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid type".into()))?;
        let local_id = key.0;

        let state = self
            .fids
            .get(&local_id)
            .ok_or_else(|| MountError::NotFound(format!("fid {} not found", local_id)))?;

        let stat_req = NpStatReq {
            fid: state.remote_fid,
        };

        let result = self
            .rt
            .block_on(state.worktree.np_stat(&stat_req))
            .map_err(Self::map_err)?;

        // Convert the RPC RStat → VFS Stat, threading the wire qid
        // version/path through. The registry service populates these from
        // filesystem metadata (qid_from_metadata: version=ctime, path=ino),
        // so this is a real qid, not a flattened one.
        let np_stat = &result.stat;

        Ok(Stat {
            qtype: np_stat.qid.qtype,
            // Thread the wire qid version/path through instead of discarding
            // them — that's the whole point of #388's widening. These are
            // advisory identity hints only; see the qid-soundness invariant
            // on `hyprstream_vfs::Stat`. Content-CID qid lands in #387.
            version: np_stat.qid.version,
            path: np_stat.qid.path,
            size: np_stat.length,
            name: np_stat.name.clone(),
            mtime: np_stat.mtime as u64,
        })
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        let Some(key) = fid.downcast_ref::<RemoteFidKey>() else {
            return;
        };
        let local_id = key.0;

        if let Some((_, state)) = self.fids.remove(&local_id) {
            let clunk_req = NpClunk {
                fid: state.remote_fid,
            };

            // Best-effort clunk — ignore errors.
            let _ = self.rt.block_on(state.worktree.clunk(&clunk_req));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Directory entry parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse directory entries from a registry worktree `read()` response.
///
/// The worktree service encodes directory entries in the documented per-entry
/// format (see `WorktreeRequest.read` annotation in `registry.capnp`):
///   `name_len(u32 LE) + name(utf8) + is_dir(u8) + size(u64 LE)`
///
/// This mirrors the parse loop in `WorktreeClient::list_dir_path`.
fn parse_worktree_dir_entries(data: &[u8]) -> Vec<DirEntry> {
    let mut entries = Vec::new();
    let mut cursor = 0;

    while cursor + 4 <= data.len() {
        let name_len = u32::from_le_bytes(
            data[cursor..cursor + 4].try_into().unwrap_or([0; 4]),
        ) as usize;
        cursor += 4;
        if cursor + name_len > data.len() {
            break;
        }
        let name =
            String::from_utf8_lossy(&data[cursor..cursor + name_len]).to_string();
        cursor += name_len;
        if cursor + 9 > data.len() {
            break;
        }
        let is_dir = data[cursor] != 0;
        cursor += 1;
        let size = u64::from_le_bytes(
            data[cursor..cursor + 8].try_into().unwrap_or([0; 8]),
        );
        cursor += 8;

        entries.push(DirEntry {
            name: name.clone(),
            is_dir,
            size,
            // Qid is unknown for dir entries read in bulk — the worktree
            // service's per-entry format carries only name/is_dir/size.
            // Callers needing a real qid can stat the child explicitly.
            stat: None,
        });
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_worktree_dir() {
        assert!(parse_worktree_dir_entries(&[]).is_empty());
    }

    #[test]
    fn parse_truncated_worktree_dir() {
        // name_len says 100 bytes but only 2 available — must not panic.
        let data = [100, 0, 0, 0, 0, 0];
        assert!(parse_worktree_dir_entries(&data).is_empty());
    }

    /// Build one worktree-format dir entry (name_len + name + is_dir + size)
    /// and verify it decodes. This is the format produced by
    /// `WorktreeClient::list_dir_path` per the registry.capnp annotation.
    #[test]
    fn parse_worktree_dir_decodes_entry() {
        let name = b"README.md";
        let is_dir: u8 = 0;
        let size: u64 = 1234;

        let mut data = Vec::new();
        data.extend_from_slice(&(name.len() as u32).to_le_bytes());
        data.extend_from_slice(name);
        data.push(is_dir);
        data.extend_from_slice(&size.to_le_bytes());

        let entries = parse_worktree_dir_entries(&data);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "README.md");
        assert!(!entries[0].is_dir);
        assert_eq!(entries[0].size, size);
        assert!(entries[0].stat.is_none());
    }

    #[test]
    fn parse_worktree_dir_decodes_directory_entry() {
        let name = b"src";
        let is_dir: u8 = 1;
        let size: u64 = 0;

        let mut data = Vec::new();
        data.extend_from_slice(&(name.len() as u32).to_le_bytes());
        data.extend_from_slice(name);
        data.push(is_dir);
        data.extend_from_slice(&size.to_le_bytes());

        let entries = parse_worktree_dir_entries(&data);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "src");
        assert!(entries[0].is_dir);
    }

    #[test]
    fn parse_worktree_dir_decodes_multiple_entries() {
        let mut data = Vec::new();

        // Entry 1: "src" directory.
        data.extend_from_slice(&(3u32).to_le_bytes());
        data.extend_from_slice(b"src");
        data.push(1);
        data.extend_from_slice(&0u64.to_le_bytes());

        // Entry 2: "README.md" file.
        data.extend_from_slice(&(9u32).to_le_bytes());
        data.extend_from_slice(b"README.md");
        data.push(0);
        data.extend_from_slice(&4096u64.to_le_bytes());

        let entries = parse_worktree_dir_entries(&data);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "src");
        assert!(entries[0].is_dir);
        assert_eq!(entries[1].name, "README.md");
        assert!(!entries[1].is_dir);
        assert_eq!(entries[1].size, 4096);
    }
}
