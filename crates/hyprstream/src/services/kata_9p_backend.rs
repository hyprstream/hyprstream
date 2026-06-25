//! `ModelBackend` — the capnp-RPC [`Backend`] for the Kata 9P translator.
//!
//! This is the production counterpart to the in-memory test backend. It turns
//! each 9P filesystem operation into a capnp RPC against the model service's
//! `fs` scope (`nine.capnp` envelope), via the generated `ModelClient`. The
//! flow is the mirror of [`RemoteModelMount`], which goes VFS → RPC:
//!
//! ```text
//!   Kata guest (virtio-9P) ──► Translator ──► ModelBackend ──► ModelClient.fs()
//!        9P2000.L wire            fid table      capnp RPC (nine.capnp)
//! ```
//!
//! The model service resolves the synthetic `/srv/model/<ref>/...` tree from
//! its `FsHandler`/`SyntheticTree`, exactly as it does for the in-process VFS.
//!
//! ## Fid numbering
//!
//! The translator allocates 9P fids on its side; we forward them unchanged as
//! the model service's remote fids (same one-to-one scheme `RemoteModelMount`
//! uses). The first walk component is the model reference; the rest is the
//! path within the model's synthetic tree.

use std::sync::atomic::AtomicU32;

use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_9p::backend::{Backend, OpenResult, StatResult, WalkResult};
use hyprstream_9p::msg::Qid;

use crate::services::generated::model_client::{ModelClient, ModelFsClient};
use crate::services::generated::model_client::{
    NpClunk, NpOpen, NpRead, NpStatReq, NpWalk, NpWrite, ROpen, RStat, RWalk,
};

/// Per-fid state cached on the backend side after a walk.
#[derive(Clone, Debug)]
struct ModelFidState {
    /// Model reference this fid belongs to (e.g. "qwen3:main").
    model_ref: String,
}

/// `Backend` impl that proxies 9P operations to the model service.
///
/// Holds a single-threaded tokio runtime to drive the async client, mirroring
/// `RemoteModelMount`. If the model service is unreachable, calls return
/// `anyhow::Error` and the translator surfaces an `Rlerror`.
pub struct ModelBackend {
    client: ModelClient,
    rt: tokio::runtime::Runtime,
    /// 9P fid → model_ref (resolved on walk).
    fids: DashMap<u32, ModelFidState>,
    /// Monotonic counter for locally-synthesized qid path values, used when
    /// the backend must fabricate a qid (e.g. root attach). Normally the qid
    /// arrives in the RPC response and this is unused.
    #[allow(dead_code)]
    next_path_id: AtomicU32,
}

impl ModelBackend {
    /// Wrap a `ModelClient`. The client must already be configured to talk to
    /// the model service (signing key, server verifying key, subject).
    #[allow(clippy::expect_used)] // runtime creation is infallible in practice
    pub fn new(client: ModelClient) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for ModelBackend");
        Self {
            client,
            rt,
            fids: DashMap::new(),
            next_path_id: AtomicU32::new(1),
        }
    }

    /// Build the scoped fs client for a model reference.
    fn fs_client(&self, model_ref: &str) -> ModelFsClient {
        self.client.fs(model_ref)
    }
}

fn qid_from_rpc(q: &crate::services::generated::model_client::Qid) -> Qid {
    Qid { qtype: q.qtype, version: q.version, path: q.path }
}

#[async_trait]
impl Backend for ModelBackend {
    async fn walk(
        &self,
        fid: u32,
        newfid: u32,
        components: &[String],
    ) -> anyhow::Result<WalkResult> {
        // Resolve model_ref: prefer a previously-walked parent fid, else take
        // the first walk component as the model_ref.
        let (model_ref, wnames): (String, Vec<String>) = if components.is_empty() {
            // Clone / attach — inherit model_ref from the source fid.
            let mr = self
                .fids
                .get(&fid)
                .map(|s| s.model_ref.clone())
                .ok_or_else(|| anyhow::anyhow!("walk: source fid {fid} has no model_ref"))?;
            (mr, Vec::new())
        } else {
            let model_ref = components[0].clone();
            let rest = components[1..].to_vec();
            (model_ref, rest)
        };

        // Forward the walk to the model service's fs scope.
        let fs = self.fs_client(&model_ref);
        let req = NpWalk { fid, newfid, wnames };
        let RWalk { qid } = self.rt.block_on(fs.walk(&req))?;
        self.fids.insert(newfid, ModelFidState { model_ref });

        Ok(WalkResult { qids: vec![qid_from_rpc(&qid)] })
    }

    async fn open(&self, fid: u32, flags: u32) -> anyhow::Result<OpenResult> {
        let state = self
            .fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("open: fid {fid} not walked"))?
            .clone();
        let fs = self.fs_client(&state.model_ref);
        // 9P2000.L lopen flags are Linux O_* bits; the model service's NpOpen
        // expects a 9P mode byte (OREAD=0, OWRITE=1, ORDWR=2). Truncate the
        // translation to the low bits for read/write intent.
        let mode = lopen_flags_to_mode(flags);
        let req = NpOpen { fid, mode };
        let ROpen { qid, iounit } = self.rt.block_on(fs.open(&req))?;
        Ok(OpenResult { qid: qid_from_rpc(&qid), iounit })
    }

    async fn read(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>> {
        let state = self
            .fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("read: fid {fid} not walked"))?
            .clone();
        let fs = self.fs_client(&state.model_ref);
        let req = NpRead { fid, offset, count };
        let result = self.rt.block_on(fs.read(&req))?;
        Ok(result.data)
    }

    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> anyhow::Result<u32> {
        let state = self
            .fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("write: fid {fid} not walked"))?
            .clone();
        let fs = self.fs_client(&state.model_ref);
        let req = NpWrite { fid, offset, data: data.to_vec() };
        let result = self.rt.block_on(fs.write(&req))?;
        Ok(result.count)
    }

    async fn stat(&self, fid: u32) -> anyhow::Result<StatResult> {
        let state = self
            .fids
            .get(&fid)
            .ok_or_else(|| anyhow::anyhow!("stat: fid {fid} not walked"))?
            .clone();
        let fs = self.fs_client(&state.model_ref);
        let req = NpStatReq { fid };
        let RStat { stat } = self.rt.block_on(fs.stat(&req))?;
        Ok(StatResult {
            qid: qid_from_rpc(&stat.qid),
            mode: stat.mode,
            size: stat.length,
            mtime_sec: stat.mtime as u64,
        })
    }

    async fn readdir(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>> {
        // The model service encodes directory listings in read() responses
        // (see RemoteModelMount::readdir): stat-encoded entries served via
        // NpRead on a directory fid. Reuse that path.
        self.read(fid, offset, count).await
    }

    async fn clunk(&self, fid: u32) -> anyhow::Result<()> {
        // Best-effort remote clunk; drop local state regardless.
        if let Some((_, state)) = self.fids.remove(&fid) {
            let fs = self.fs_client(&state.model_ref);
            let req = NpClunk { fid };
            let _ = self.rt.block_on(fs.clunk(&req));
        }
        Ok(())
    }
}

/// Map 9P2000.L Tlopen flags (Linux O_* bits) to a 9P open mode byte.
///
/// Only read/write intent is preserved; flags like O_TRUNC / O_APPEND are
/// advisory here since the model service's synthetic tree is mostly read-only.
fn lopen_flags_to_mode(flags: u32) -> u8 {
    const O_WRONLY: u32 = 0o1;
    const O_RDWR: u32 = 0o2;
    let acc = flags & 0o3;
    match acc {
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
        assert_eq!(lopen_flags_to_mode(0), 0); // OREAD
        assert_eq!(lopen_flags_to_mode(0o1), 1); // OWRITE
        assert_eq!(lopen_flags_to_mode(0o2), 2); // ORDWR
        assert_eq!(lopen_flags_to_mode(0o100), 0); // O_CREAT + OREAD → OREAD
        assert_eq!(lopen_flags_to_mode(0o101), 1); // O_CREAT + OWRITE → OWRITE
    }
}
