//! NinePServer — serves a Mount impl over TCP/Unix sockets using 9P2000.L.
//!
//! The server accepts connections and spawns a `Session` per client. Each
//! session has its own fid table and caller identity.

use std::sync::Arc;

use hyprstream_rpc::Subject;
use hyprstream_vfs::Mount;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{debug, info, warn};

use crate::protocol::{Tframe, Rframe};
use crate::session::Session;

/// Serves a `Mount` implementation over 9P2000.L on a TCP socket.
///
/// Each accepted connection gets its own `Session` with an isolated fid table.
/// The server runs until dropped or the listener is closed.
///
/// # Example (future use)
///
/// ```ignore
/// let mount: Arc<dyn Mount> = /* ... */;
/// let server = NinePServer::new(mount);
/// server.listen("127.0.0.1:5640").await?;
/// ```
pub struct NinePServer {
    mount: Arc<dyn Mount>,
}

impl NinePServer {
    /// Create a new 9P server wrapping a Mount implementation.
    pub fn new(mount: Arc<dyn Mount>) -> Self {
        Self { mount }
    }

    /// Listen for 9P connections on the given TCP address.
    ///
    /// Spawns a tokio task per connection. Runs until the listener errors
    /// or the returned future is dropped.
    pub async fn listen(&self, addr: &str) -> anyhow::Result<()> {
        let listener = TcpListener::bind(addr).await?;
        info!("9P server listening on {addr}");

        loop {
            let (stream, peer) = listener.accept().await?;
            let mount = Arc::clone(&self.mount);
            info!("9P connection from {peer}");

            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, mount).await {
                    warn!("9P connection from {peer} error: {e}");
                }
                debug!("9P connection from {peer} closed");
            });
        }
    }

    /// Access the underlying mount.
    pub fn mount(&self) -> &Arc<dyn Mount> {
        &self.mount
    }
}

/// Handle a single 9P client connection.
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    mount: Arc<dyn Mount>,
) -> anyhow::Result<()> {
    let peer_addr = stream.peer_addr()?.to_string();
    let caller = Subject::new(&peer_addr);
    let mut session = Session::new(mount, caller);

    let mut buf = vec![0u8; 64 * 1024];
    let mut pending = Vec::new();

    loop {
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            break; // EOF
        }

        pending.extend_from_slice(&buf[..n]);

        // Process all complete frames in the buffer.
        loop {
            let tframe = match Tframe::decode(&pending) {
                Ok(Some((frame, consumed))) => {
                    pending.drain(..consumed);
                    frame
                }
                Ok(None) => break, // need more data
                Err(e) => {
                    warn!("failed to decode Tframe: {e}");
                    // Try to skip past this frame using the size prefix.
                    if pending.len() >= 4 {
                        let size = u32::from_le_bytes([
                            pending[0], pending[1], pending[2], pending[3],
                        ]) as usize;
                        if size <= pending.len() {
                            pending.drain(..size);
                            continue;
                        }
                    }
                    break;
                }
            };

            let tag = tframe.tag;
            let rmsg = session.handle(tframe.msg);

            let rframe = Rframe { tag, msg: rmsg };
            let out = rframe.encode()?;
            stream.write_all(&out).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_vfs::{DirEntry, Fid as VfsFid, MountError, Stat as VfsStat};

    struct SimpleMnt;

    impl Mount for SimpleMnt {
        fn walk(&self, components: &[&str], _caller: &Subject) -> Result<VfsFid, MountError> {
            Ok(VfsFid::new(components.join("/")))
        }
        fn open(&self, _fid: &mut VfsFid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        fn read(&self, fid: &VfsFid, _offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let path = fid.downcast_ref::<String>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            if path == "hello" {
                Ok(b"world".to_vec())
            } else {
                Err(MountError::NotFound(path.clone()))
            }
        }
        fn write(&self, _fid: &VfsFid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }
        fn readdir(&self, _fid: &VfsFid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![DirEntry { name: "hello".into(), is_dir: false, size: 5, stat: None }])
        }
        fn stat(&self, _fid: &VfsFid, _caller: &Subject) -> Result<VfsStat, MountError> {
            Ok(VfsStat { qtype: 0, size: 5, name: "hello".into(), mtime: 0 })
        }
        fn clunk(&self, _fid: VfsFid, _caller: &Subject) {}
    }

    #[test]
    fn server_construction() {
        let mount: Arc<dyn Mount> = Arc::new(SimpleMnt);
        let _server = NinePServer::new(mount);
    }
}
