//! WanixMount — VFS Mount implementation backed by a 9P client over DMA.
//!
//! Mounts Wanix's filesystem into the hyprstream VFS namespace at `/wanix/`.
//! All filesystem operations are forwarded via 9P2000.L protocol over
//! SharedArrayBuffer DMA ring buffers.

#[cfg(target_arch = "wasm32")]
use async_trait::async_trait;

#[cfg(target_arch = "wasm32")]
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
#[cfg(target_arch = "wasm32")]
use hyprstream_rpc::Subject;

#[cfg(target_arch = "wasm32")]
use crate::client::{P9Client, P9Transport};

/// Fid state for WanixMount — holds the 9P fid number.
#[cfg(target_arch = "wasm32")]
struct P9FidState {
    fid: u32,
}

/// Mount that bridges to Wanix's Go 9P server via DMA.
///
/// Every VFS operation is translated to a 9P T-message, sent over the DMA
/// ring buffer, and the R-message response is parsed back.
#[cfg(target_arch = "wasm32")]
pub struct WanixMount<T: P9Transport> {
    client: P9Client<T>,
}

#[cfg(target_arch = "wasm32")]
impl<T: P9Transport> WanixMount<T> {
    /// Create a new WanixMount from a connected P9Client.
    pub fn new(client: P9Client<T>) -> Self {
        Self { client }
    }
}

#[cfg(target_arch = "wasm32")]
// SAFETY: wasm32 is single-threaded
unsafe impl<T: P9Transport> Send for WanixMount<T> {}
#[cfg(target_arch = "wasm32")]
unsafe impl<T: P9Transport> Sync for WanixMount<T> {}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
impl<T: P9Transport + 'static> Mount for WanixMount<T> {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let (fid, _qids) = self.client.walk(components).await
            .map_err(|e| MountError::NotFound(e.to_string()))?;
        Ok(Fid::new(P9FidState { fid }))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        let state = fid.downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.client.open(state.fid, mode as u32).await
            .map_err(|e| MountError::Io(e.to_string()))?;
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let state = fid.downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.client.read(state.fid, offset, count).await
            .map_err(|e| MountError::Io(e.to_string()))
    }

    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let state = fid.downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.client.write(state.fid, offset, data).await
            .map_err(|e| MountError::Io(e.to_string()))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid.downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let entries = self.client.readdir(state.fid, 0, 65536).await
            .map_err(|e| MountError::Io(e.to_string()))?;
        Ok(entries.into_iter().map(|e| DirEntry {
            name: e.name,
            is_dir: e.qid.is_dir(),
            size: 0,
            stat: None,
        }).collect())
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid.downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let (qid, mode, size, mtime) = self.client.getattr(state.fid).await
            .map_err(|e| MountError::Io(e.to_string()))?;
        Ok(Stat {
            qtype: qid.qtype,
            size,
            name: String::new(), // name comes from walk, not stat
            mtime,
        })
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        if let Some(state) = fid.downcast_ref::<P9FidState>() {
            let _ = self.client.clunk(state.fid).await;
        }
    }
}
