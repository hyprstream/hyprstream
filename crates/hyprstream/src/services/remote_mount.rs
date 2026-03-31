//! Model-service `FsClient` adapter + convenience constructor.
//!
//! `ModelFsAdapter` implements the generic `FsClient` trait from hyprstream-rpc
//! by delegating to the generated `ModelFsClient` scoped RPC client. This lets
//! `hyprstream_vfs::RemoteMount<ModelFsAdapter>` serve `/srv/model` in the VFS
//! namespace without any model-specific code in the VFS crate.
//!
//! The first walk component is the model reference (e.g., "qwen3:main"), which
//! selects the `ModelFsClient` scope. The remaining components are forwarded as
//! 9P wnames.
//!
//! ## Legacy
//!
//! Previously this file contained `RemoteModelMount` which implemented `Mount`
//! directly with an inlined fid table. That logic now lives in the generic
//! `hyprstream_vfs::RemoteMount<C>`.

use async_trait::async_trait;
use hyprstream_rpc::{FsClient, FsOpenResult, FsStatResult, FsWalkResult};

use crate::services::generated::model_client::ModelClient;

// ─────────────────────────────────────────────────────────────────────────────
// ModelFsAdapter
// ─────────────────────────────────────────────────────────────────────────────

/// Adapts the generated `ModelClient` to the generic `FsClient` trait.
///
/// Walk components are split: the first element selects the model reference
/// (used to scope the `ModelFsClient`), the rest become 9P wnames. The model
/// reference is stored per-fid so that subsequent operations (open, read, etc.)
/// can re-obtain the scoped client.
///
/// Because `FsClient` methods only receive a fid number (not the model ref),
/// we store a default model ref at construction time. The walk implementation
/// extracts the model ref from the first wname when present.
pub struct ModelFsAdapter {
    client: ModelClient,
    /// Default model reference (e.g., "qwen3:main").
    /// Walk operations that include a model_ref as the first component override this.
    default_model_ref: String,
    /// Per-fid model reference tracking.
    fid_model_refs: dashmap::DashMap<u32, String>,
}

impl ModelFsAdapter {
    /// Create a new adapter wrapping the given model client.
    ///
    /// `default_model_ref` is used for fid operations where the model ref
    /// isn't embedded in the path (i.e., post-walk operations).
    pub fn new(client: ModelClient, default_model_ref: impl Into<String>) -> Self {
        Self {
            client,
            default_model_ref: default_model_ref.into(),
            fid_model_refs: dashmap::DashMap::new(),
        }
    }

    /// Convenience: create adapter with no default model ref.
    /// Walk operations must include the model ref as the first path component.
    pub fn new_unscoped(client: ModelClient) -> Self {
        Self::new(client, String::new())
    }

    /// Look up the model ref for a given fid, falling back to the default.
    fn model_ref_for_fid(&self, fid: u32) -> String {
        self.fid_model_refs
            .get(&fid)
            .map(|r| r.clone())
            .unwrap_or_else(|| self.default_model_ref.clone())
    }
}

#[async_trait]
impl FsClient for ModelFsAdapter {
    async fn fs_walk(&self, wnames: Vec<String>, newfid: u32) -> Result<FsWalkResult, String> {
        // First wname is the model reference; rest are 9P path components.
        if wnames.is_empty() {
            return Err("empty walk path".into());
        }

        let model_ref = &wnames[0];
        let inner_wnames: Vec<String> = wnames[1..].to_vec();

        // Track which model ref this fid belongs to.
        self.fid_model_refs.insert(newfid, model_ref.clone());

        let fs = self.client.fs(model_ref);
        let walk_req = crate::services::generated::model_client::NpWalk {
            fid: 0, // root
            newfid,
            wnames: inner_wnames,
        };

        let result = fs.walk(&walk_req).await.map_err(|e| e.to_string())?;
        Ok(FsWalkResult {
            qtype: result.qid.qtype,
        })
    }

    async fn fs_open(&self, fid: u32, mode: u8) -> Result<FsOpenResult, String> {
        let model_ref = self.model_ref_for_fid(fid);
        let fs = self.client.fs(&model_ref);
        let open_req = crate::services::generated::model_client::NpOpen { fid, mode };

        let result = fs.open(&open_req).await.map_err(|e| e.to_string())?;
        Ok(FsOpenResult {
            qtype: result.qid.qtype,
            iounit: result.iounit,
        })
    }

    async fn fs_read(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>, String> {
        let model_ref = self.model_ref_for_fid(fid);
        let fs = self.client.fs(&model_ref);
        let read_req = crate::services::generated::model_client::NpRead { fid, offset, count };

        let result = fs.read(&read_req).await.map_err(|e| e.to_string())?;
        Ok(result.data)
    }

    async fn fs_write(&self, fid: u32, offset: u64, data: Vec<u8>) -> Result<u32, String> {
        let model_ref = self.model_ref_for_fid(fid);
        let fs = self.client.fs(&model_ref);
        let write_req = crate::services::generated::model_client::NpWrite { fid, offset, data };

        let result = fs.write(&write_req).await.map_err(|e| e.to_string())?;
        Ok(result.count)
    }

    async fn fs_clunk(&self, fid: u32) -> Result<(), String> {
        let model_ref = self.model_ref_for_fid(fid);
        let fs = self.client.fs(&model_ref);
        let clunk_req = crate::services::generated::model_client::NpClunk { fid };

        fs.clunk(&clunk_req).await.map_err(|e| e.to_string())?;
        // Clean up our tracking.
        self.fid_model_refs.remove(&fid);
        Ok(())
    }

    async fn fs_stat(&self, fid: u32) -> Result<FsStatResult, String> {
        let model_ref = self.model_ref_for_fid(fid);
        let fs = self.client.fs(&model_ref);
        let stat_req = crate::services::generated::model_client::NpStatReq { fid };

        let result = fs.stat(&stat_req).await.map_err(|e| e.to_string())?;
        let np_stat = &result.stat;

        Ok(FsStatResult {
            qtype: np_stat.qid.qtype,
            size: np_stat.length,
            name: np_stat.name.clone(),
            mtime: np_stat.mtime as u64,
        })
    }

    async fn fs_readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>, String> {
        // Readdir is implemented as a regular read on a directory fid in 9P.
        self.fs_read(fid, offset, count).await
    }
}

/// Convenience type alias for the most common usage.
pub type RemoteModelMount = hyprstream_vfs::RemoteMount<ModelFsAdapter>;
