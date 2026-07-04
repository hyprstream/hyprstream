//! `Remote9pMount` — a native VFS [`Mount`] backed by a socket 9P client.
//!
//! This is the **client half** of the bidirectional 9P story (#708): where
//! [`MountBackend`](crate::mount_backend::MountBackend) exports a local VFS
//! [`Mount`] *as* a 9P server, `Remote9pMount` *imports* a remote 9P2000.L server
//! and presents it back as a [`Mount`] the host can
//! [`bind_mount`](hyprstream_vfs::Namespace) into its own namespace. Together
//! they let hyprstream both serve its namespace to a guest and consume the
//! guest's namespace — two independent 9P sessions over two sockets.
//!
//! It is the native, socket-transport analogue of the wasm-only
//! [`WanixMount`](crate::wanix_mount::WanixMount): the same `P9Client`-backed
//! `Mount` shape, but generic over a genuinely `Send + Sync`
//! [`SocketTransport`] rather than the browser DMA transport, and with no
//! `unsafe impl Send/Sync` (the wasm `WanixMount` needs one because its DMA
//! transport is only sound single-threaded). See the crate docs for why the two
//! are separate types rather than one renamed generic.
//!
//! ## Subject / fid semantics
//!
//! Every op threads the caller [`Subject`], exactly like every other VFS
//! [`Mount`]. Each `walk` returns an opaque [`Fid`] wrapping the remote 9P fid
//! number; `open`/`read`/`write`/`readdir`/`stat`/`clunk` forward to that fid.
//! The wrapped [`P9Client`] is held behind a single async `Mutex` so each
//! Mount op is one atomic T→R exchange on the connection — requests never
//! interleave on the wire.

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::Mutex;

use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};

use crate::client::{P9Client, P9Transport};
use crate::socket_transport::SocketTransport;

/// Fid state for a [`Remote9pMount`] — the remote 9P fid number.
struct P9FidState {
    fid: u32,
}

/// A VFS [`Mount`] that forwards every operation to a remote 9P2000.L server
/// over a socket [`P9Transport`].
///
/// Build one with [`connect_uds`](Remote9pMount::connect_uds) /
/// [`connect_tcp`](Remote9pMount::connect_tcp), or [`from_client`] with a
/// pre-connected [`P9Client`] over any transport.
///
/// [`from_client`]: Remote9pMount::from_client
pub struct Remote9pMount<T: P9Transport> {
    client: Mutex<P9Client<T>>,
}

impl<T: P9Transport> Remote9pMount<T> {
    /// Wrap an already-connected [`P9Client`] (version + attach handshake done).
    pub fn from_client(client: P9Client<T>) -> Self {
        Self {
            client: Mutex::new(client),
        }
    }
}

impl Remote9pMount<SocketTransport<UnixStream>> {
    /// Dial a remote 9P server on a Unix domain socket and attach its root.
    ///
    /// `uname`/`aname` are the 9P attach identity and export name (the server's
    /// [`Translator`](crate::translator::Translator) ignores them and serves its
    /// single Subject-scoped root, but they are sent for protocol conformance).
    pub async fn connect_uds(
        path: impl AsRef<Path>,
        uname: &str,
        aname: &str,
    ) -> Result<Self> {
        let transport = SocketTransport::connect_uds(path).await?;
        let client = P9Client::connect(transport, uname, aname).await?;
        Ok(Self::from_client(client))
    }
}

impl Remote9pMount<SocketTransport<TcpStream>> {
    /// Dial a remote 9P server over TCP and attach its root.
    pub async fn connect_tcp(
        addr: impl tokio::net::ToSocketAddrs,
        uname: &str,
        aname: &str,
    ) -> Result<Self> {
        let transport = SocketTransport::connect_tcp(addr).await?;
        let client = P9Client::connect(transport, uname, aname).await?;
        Ok(Self::from_client(client))
    }
}

#[async_trait]
impl<T> Mount for Remote9pMount<T>
where
    T: P9Transport + Send + Sync + 'static,
{
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let client = self.client.lock().await;
        let (fid, _qids) = client
            .walk(components)
            .await
            .map_err(|e| MountError::NotFound(e.to_string()))?;
        Ok(Fid::new(P9FidState { fid }))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        let f = fid
            .downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?
            .fid;
        let client = self.client.lock().await;
        client
            .open(f, mode as u32)
            .await
            .map_err(|e| MountError::Io(e.to_string()))?;
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let f = fid
            .downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?
            .fid;
        let client = self.client.lock().await;
        client
            .read(f, offset, count)
            .await
            .map_err(|e| MountError::Io(e.to_string()))
    }

    async fn write(
        &self,
        fid: &Fid,
        offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let f = fid
            .downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?
            .fid;
        let client = self.client.lock().await;
        client
            .write(f, offset, data)
            .await
            .map_err(|e| MountError::Io(e.to_string()))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let f = fid
            .downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?
            .fid;
        let client = self.client.lock().await;
        let entries = client
            .readdir(f, 0, 65536)
            .await
            .map_err(|e| MountError::Io(e.to_string()))?;
        Ok(entries
            .into_iter()
            .map(|e| DirEntry {
                name: e.name,
                is_dir: e.qid.is_dir(),
                size: 0,
                stat: None,
            })
            .collect())
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let f = fid
            .downcast_ref::<P9FidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?
            .fid;
        let client = self.client.lock().await;
        let (qid, _mode, size, mtime) = client
            .getattr(f)
            .await
            .map_err(|e| MountError::Io(e.to_string()))?;
        Ok(Stat {
            qtype: qid.qtype,
            version: qid.version,
            path: qid.path,
            size,
            name: String::new(), // name comes from walk, not stat
            mtime,
        })
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        if let Some(f) = fid.downcast_ref::<P9FidState>().map(|s| s.fid) {
            let client = self.client.lock().await;
            let _ = client.clunk(f).await;
        }
    }
}
