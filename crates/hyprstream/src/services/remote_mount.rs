//! Remote model mount — bridges the VFS `Mount` trait to the generated `ModelClient` RPC.
//!
//! `RemoteModelMount` wraps a `ModelClient` and translates synchronous `Mount`
//! trait calls into async RPC requests over ZMQ. This allows the TUI shell's
//! VFS namespace to serve `/srv/model` by proxying 9P operations to the model
//! service, which in turn delegates to its local `FsHandler` / `SyntheticTree`.
//!
//! ## Shared bridge
//!
//! The sync→async plumbing (embedded tokio runtime, fid allocator, fid map,
//! `anyhow → MountError` mapping, the opaque fid key, and the packed 9P stat
//! parser) is shared with `RemoteRegistryMount` via [`NinePBridge`] in
//! [`crate::services::ninep_bridge`]. Only the model-specific pieces live
//! here: the `ModelClient`/`ModelFsClient` types, the 1-level scope
//! extraction (first path component = model ref), and the `stat` (not
//! `np_stat`) RPC method name.
//!
//! ## Fid management
//!
//! The mount allocates local fid numbers and tracks them via the bridge's
//! `DashMap`. Each local fid maps to a `RemoteFidState` that stores the
//! remote fid number returned by walk, plus metadata (open mode, qtype) for
//! stat/readdir synthesis.
//!
//! ## Timeout / graceful degradation
//!
//! A dedicated single-threaded tokio runtime drives the async client. If the
//! model service is unreachable, `block_on` will hang until the ZMQ socket
//! timeout fires (set on the underlying socket). Callers see
//! `MountError::Io("service unreachable: ...")`.

use async_trait::async_trait;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};

use crate::services::generated::model_client::{ModelClient, ModelFsClient};
use crate::services::ninep_bridge::{fid_key, map_err, parse_dir_stats, NinePBridge, RemoteFidKey};
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

// ─────────────────────────────────────────────────────────────────────────────
// RemoteModelMount
// ─────────────────────────────────────────────────────────────────────────────

/// A `Mount` implementation that proxies 9P operations to the model service
/// via the generated `ModelClient` RPC.
pub struct RemoteModelMount {
    client: ModelClient,
    bridge: NinePBridge<RemoteFidState>,
}

impl RemoteModelMount {
    /// Create a new remote mount wrapping the given model client.
    pub fn new(client: ModelClient) -> Self {
        Self {
            client,
            bridge: NinePBridge::new("RemoteModelMount"),
        }
    }

    /// Get the scoped fs client for a model reference.
    fn fs_client(&self, model_ref: &str) -> ModelFsClient {
        self.client.fs(model_ref)
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
        let wnames: Vec<String> = components[1..]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();

        let local_fid = self.bridge.alloc_fid();
        let remote_newfid = local_fid; // Use same numbering for simplicity.

        let fs = self.fs_client(model_ref);
        let walk_req = crate::services::generated::model_client::NpWalk {
            fid: 0, // root
            newfid: remote_newfid,
            wnames,
        };

        let result = self
            .bridge
            .rt()
            .block_on(fs.walk(&walk_req))
            .map_err(map_err)?;

        // Determine qtype from the walk response qid.
        let qtype = result.qid.qtype;

        let state = RemoteFidState {
            remote_fid: remote_newfid,
            model_ref: model_ref.to_owned(),
            qtype,
            opened: false,
        };
        self.bridge.insert(local_fid, state);

        Ok(Fid::new(RemoteFidKey(local_fid)))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        let local_id = fid_key(fid)?.0;

        let mut state = self.bridge.get_mut(local_id)?;

        let fs = self.fs_client(&state.model_ref);
        let open_req = crate::services::generated::model_client::NpOpen {
            fid: state.remote_fid,
            mode,
        };

        let _result = self
            .bridge
            .rt()
            .block_on(fs.open(&open_req))
            .map_err(map_err)?;
        state.opened = true;

        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let local_id = fid_key(fid)?.0;

        let state = self.bridge.get(local_id)?;

        let fs = self.fs_client(&state.model_ref);
        let read_req = crate::services::generated::model_client::NpRead {
            fid: state.remote_fid,
            offset,
            count,
        };

        let result = self
            .bridge
            .rt()
            .block_on(fs.read(&read_req))
            .map_err(map_err)?;
        Ok(result.data)
    }

    async fn write(
        &self,
        fid: &Fid,
        offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let local_id = fid_key(fid)?.0;

        let state = self.bridge.get(local_id)?;

        let fs = self.fs_client(&state.model_ref);
        let write_req = crate::services::generated::model_client::NpWrite {
            fid: state.remote_fid,
            offset,
            data: data.to_vec(),
        };

        let result = self
            .bridge
            .rt()
            .block_on(fs.write(&write_req))
            .map_err(map_err)?;
        Ok(result.count)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        // 9P doesn't have a dedicated readdir — we read the directory fid and parse
        // the stat entries from the raw bytes. However, the model service's SyntheticTree
        // encodes directory listings in its read() response as newline-separated entries.
        //
        // For now, do a read at offset 0 and parse the response.
        // The model service returns stat-encoded directory entries.
        let local_id = fid_key(fid)?.0;

        let state = self.bridge.get(local_id)?;

        if state.qtype != QTDIR {
            return Err(MountError::NotDirectory(format!(
                "fid {} is not a directory",
                local_id
            )));
        }

        // Read directory data from remote.
        let fs = self.fs_client(&state.model_ref);
        let read_req = crate::services::generated::model_client::NpRead {
            fid: state.remote_fid,
            offset: 0,
            count: 65536, // Large enough for most directory listings.
        };

        let result = self
            .bridge
            .rt()
            .block_on(fs.read(&read_req))
            .map_err(map_err)?;

        // Parse stat entries from the raw 9P directory read data.
        // Directory reads return packed NpStat entries. Each entry is preceded
        // by a 2-byte little-endian size. We extract name + qtype from each.
        Ok(parse_dir_stats(&result.data))
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let local_id = fid_key(fid)?.0;

        let state = self.bridge.get(local_id)?;

        let fs = self.fs_client(&state.model_ref);
        let stat_req = crate::services::generated::model_client::NpStatReq {
            fid: state.remote_fid,
        };

        let result = self
            .bridge
            .rt()
            .block_on(fs.stat(&stat_req))
            .map_err(map_err)?;

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
        let local_id = match fid_key(&fid) {
            Ok(k) => k.0,
            Err(_) => return,
        };

        if let Some(state) = self.bridge.remove(local_id) {
            let fs = self.fs_client(&state.model_ref);
            let clunk_req = crate::services::generated::model_client::NpClunk {
                fid: state.remote_fid,
            };

            // Best-effort clunk — ignore errors.
            let _ = self.bridge.rt().block_on(fs.clunk(&clunk_req));
        }
    }
}
