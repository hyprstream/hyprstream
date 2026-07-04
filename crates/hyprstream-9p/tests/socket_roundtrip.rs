//! Round-trip integration test: the socket 9P **client** `Mount`
//! ([`Remote9pMount`]) against hyprstream's own server-side [`Translator`]
//! (`serve_mount_uds`), over a real Unix domain socket.
//!
//! This is the reusable-infra proof for #708 phase 1: it needs no Wanix. We
//! serve a small in-process [`SyntheticMount`] over a temp UDS, dial it with the
//! socket client Mount, and drive attach → walk → readdir → read (including a
//! nested walk and a stat) end-to-end, asserting the bytes round-trip through the
//! full 9P2000.L wire path (client codec ⇄ socket ⇄ translator ⇄ MountBackend).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use hyprstream_9p::{serve_mount_uds, Remote9pMount};
use hyprstream_rpc::Subject;
use hyprstream_vfs::{Mount, SyntheticMount, SyntheticNode};

#[tokio::test]
async fn socket_client_mount_round_trips_against_translator() {
    // ── Build a small synthetic tree to export ──────────────────────────────
    let root = SyntheticNode::dir()
        .with_child("hello.txt", SyntheticNode::file(b"hello world".to_vec()))
        .with_child(
            "dir",
            SyntheticNode::dir().with_child("nested.txt", SyntheticNode::file(b"deep".to_vec())),
        );
    let mount: Arc<dyn Mount> = Arc::new(SyntheticMount::new(root));
    let subject = Subject::new("tester");

    // Temp UDS path (unique dir, auto-cleaned on drop).
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("9p.sock");

    // ── Serve it over a UDS in the background via the hyprstream translator ──
    let server = tokio::spawn({
        let mount = Arc::clone(&mount);
        let subject = subject.clone();
        let sock = sock.clone();
        async move {
            // Returns only on listener error / shutdown; we abort it at the end.
            let _ = serve_mount_uds(mount, subject, sock).await;
        }
    });

    // Wait for the listener socket to appear.
    let mut ready = false;
    for _ in 0..200 {
        if sock.exists() {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(ready, "translator did not bind the UDS in time");

    // ── Dial it with the socket 9P *client* Mount (version + attach) ─────────
    let client_mount = Remote9pMount::connect_uds(&sock, "tester", "/")
        .await
        .expect("connect socket client mount");
    let caller = Subject::new("tester");

    // Walk the root, open it, readdir.
    let mut root_fid = client_mount.walk(&[], &caller).await.expect("walk root");
    client_mount
        .open(&mut root_fid, 0, &caller)
        .await
        .expect("open root dir");
    let entries = client_mount
        .readdir(&root_fid, &caller)
        .await
        .expect("readdir root");
    let mut names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
    names.sort();
    assert_eq!(names, vec!["dir".to_string(), "hello.txt".to_string()]);
    // The directory child must be reported as a directory (qid qtype carried
    // through the wire-faithful Rreaddir records).
    let dir_entry = entries.iter().find(|e| e.name == "dir").unwrap();
    assert!(dir_entry.is_dir, "`dir` should be reported as a directory");

    // Walk to the top-level file, open, read the full contents.
    let mut file_fid = client_mount
        .walk(&["hello.txt"], &caller)
        .await
        .expect("walk hello.txt");
    client_mount
        .open(&mut file_fid, 0, &caller)
        .await
        .expect("open hello.txt");
    let data = client_mount
        .read(&file_fid, 0, 1024, &caller)
        .await
        .expect("read hello.txt");
    assert_eq!(&data, b"hello world");

    // stat the file: size must match.
    let st = client_mount.stat(&file_fid, &caller).await.expect("stat");
    assert_eq!(st.size, 11);

    // Multi-component walk into the nested dir, open, read.
    let mut nested_fid = client_mount
        .walk(&["dir", "nested.txt"], &caller)
        .await
        .expect("walk dir/nested.txt");
    client_mount
        .open(&mut nested_fid, 0, &caller)
        .await
        .expect("open nested.txt");
    let nested = client_mount
        .read(&nested_fid, 0, 1024, &caller)
        .await
        .expect("read nested.txt");
    assert_eq!(&nested, b"deep");

    // Clunk everything we opened (best-effort; drives the clunk path).
    client_mount.clunk(root_fid, &caller).await;
    client_mount.clunk(file_fid, &caller).await;
    client_mount.clunk(nested_fid, &caller).await;

    server.abort();
    let _ = server.await;
}
