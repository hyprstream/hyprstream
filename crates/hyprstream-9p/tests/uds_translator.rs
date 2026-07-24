//! End-to-end integration test: the translator serving a Subject-scoped VFS
//! `Mount` as a 9P2000.L server over a Unix domain socket (#506).
//!
//! Binds a `UnixListener` on a temp path, exports a tiny in-test `Mount` via
//! `Translator::from_mount(mount, subject)`, then drives it from a raw
//! `UnixStream` using the 9P codec (version → attach → walk → lopen → read →
//! getattr → clunk). This proves the UDS accept loop, the transport-generic
//! `serve_connection`, and the `MountBackend` → `Mount` (Subject-threaded)
//! bridge all work together — the path a native Wanix `p9kit.ClientFS` uses.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use hyprstream_9p::msg::{self, Response};
use hyprstream_9p::{AccessDecider, Action, Translator};
use hyprstream_rpc::auth::mac::{ObjectRef, SecurityContext};
use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

struct TestDecider;
impl AccessDecider for TestDecider {
    fn check(&self, _: &SecurityContext, _: ObjectRef<'_>, _: Action) -> bool {
        true
    }
}

/// Opaque per-fid state for `TestMount`: the absolute path it resolves to.
struct TestFid {
    path: Vec<String>,
}

/// Minimal read-only `Mount` exposing a flat set of top-level files, so the
/// test has no dependency on the binary crate's synthetic tree.
struct TestMount {
    files: HashMap<String, Vec<u8>>,
}

#[async_trait]
impl Mount for TestMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let path: Vec<String> = components.iter().map(|s| (*s).to_owned()).collect();
        if path.is_empty() {
            return Ok(Fid::new(TestFid { path })); // root
        }
        let joined = path.join("/");
        if self.files.contains_key(&joined) {
            Ok(Fid::new(TestFid { path }))
        } else {
            Err(MountError::NotFound(joined))
        }
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let state = fid.downcast_ref::<TestFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let joined = state.path.join("/");
        let data = self.files.get(&joined).ok_or_else(|| MountError::NotFound(joined))?;
        let off = offset as usize;
        if off >= data.len() {
            return Ok(vec![]);
        }
        let end = (off + count as usize).min(data.len());
        Ok(data[off..end].to_vec())
    }

    async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        Err(MountError::NotSupported("read-only test mount".into()))
    }

    async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        Ok(self
            .files
            .keys()
            .map(|k| DirEntry { name: k.clone(), is_dir: false, size: 0, stat: None })
            .collect())
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid.downcast_ref::<TestFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        if state.path.is_empty() {
            return Ok(Stat { qtype: 0x80, version: 1, path: 0, size: 0, name: String::new(), mtime: 0 });
        }
        let joined = state.path.join("/");
        let size = self.files.get(&joined).map(|d| d.len() as u64).unwrap_or(0);
        Ok(Stat { qtype: 0, version: 1, path: 1, size, name: joined, mtime: 0 })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

/// Read one complete 9P message (length-prefixed) from `stream`.
async fn recv_message(stream: &mut UnixStream) -> Vec<u8> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).await.unwrap();
    let total = u32::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; total];
    buf[..4].copy_from_slice(&len);
    stream.read_exact(&mut buf[4..]).await.unwrap();
    buf
}

async fn rpc(stream: &mut UnixStream, req: Vec<u8>) -> Response {
    stream.write_all(&req).await.unwrap();
    let resp = recv_message(stream).await;
    let (_, parsed) = msg::parse_response(&resp).unwrap();
    parsed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uds_full_session() {
    let mut files = HashMap::new();
    files.insert("hello.txt".to_owned(), b"hello over uds".to_vec());
    let mount: Arc<dyn Mount> = Arc::new(TestMount { files });

    // Unique per-run socket path in the OS temp dir.
    let sock = std::env::temp_dir().join(format!(
        "hyprstream-9p-uds-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
    ));
    let _ = std::fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock).unwrap();

    let subject = Subject::new("tenant-a");
    let translator = Translator::from_mount(mount, subject, Arc::new(TestDecider));
    let server = tokio::spawn(async move {
        let _ = translator.serve_uds(listener).await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = UnixStream::connect(&sock).await.unwrap();

    // Version negotiation.
    match rpc(&mut client, msg::tversion(1, 4096, "9P2000.L")).await {
        Response::Version { version, .. } => assert_eq!(version, "9P2000.L"),
        other => panic!("expected Rversion, got {other:?}"),
    }

    // Attach: fid 0 = root (a directory).
    match rpc(&mut client, msg::tattach(2, 0, u32::MAX, "tenant-a", "/")).await {
        Response::Attach { qid } => assert!(qid.is_dir(), "root qid should be a dir"),
        other => panic!("expected Rattach, got {other:?}"),
    }

    // Walk root → fid 1 (the file).
    match rpc(&mut client, msg::twalk(3, 0, 1, &["hello.txt"])).await {
        Response::Walk { qids } => assert_eq!(qids.len(), 1),
        other => panic!("expected Rwalk, got {other:?}"),
    }

    // Open fid 1 for reading.
    match rpc(&mut client, msg::tlopen(4, 1, 0)).await {
        Response::Lopen { iounit, .. } => assert!(iounit > 0),
        other => panic!("expected Rlopen, got {other:?}"),
    }

    // Read the file content back.
    match rpc(&mut client, msg::tread(5, 1, 0, 256)).await {
        Response::Read { data } => assert_eq!(&data, b"hello over uds"),
        other => panic!("expected Rread, got {other:?}"),
    }

    // Stat (getattr): size matches.
    match rpc(&mut client, msg::tgetattr(6, 1, 0x7ff)).await {
        Response::Getattr { size, .. } => assert_eq!(size, b"hello over uds".len() as u64),
        other => panic!("expected Rgetattr, got {other:?}"),
    }

    // Clunk.
    match rpc(&mut client, msg::tclunk(7, 1)).await {
        Response::Clunk => {}
        other => panic!("expected Rclunk, got {other:?}"),
    }

    // Walking a missing file surfaces an Rlerror.
    match rpc(&mut client, msg::twalk(8, 0, 2, &["nope"])).await {
        Response::Error { .. } => {}
        other => panic!("expected Rlerror for missing file, got {other:?}"),
    }

    server.abort();
    let _ = std::fs::remove_file(&sock);
}
