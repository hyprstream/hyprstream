//! Standalone 9P-over-UDS server used to prove interop with a real 9P2000.L
//! client (Wanix `p9kit.ClientFS`, i.e. the `progrium/p9` fork).
//!
//! Usage: `serve_mount_uds <socket-path>` — binds a `UnixListener` at the given
//! path and serves a small `SyntheticMount` (two files + a subdir with one file)
//! as its 9P root, forever. Kill with SIGINT/SIGTERM.
//!
//! This is a harness, not shipped functionality — it exercises `from_mount` /
//! `serve_mount_uds` exactly as #506's supervisor entry point does.

use std::sync::Arc;

use hyprstream_9p::translator::serve_mount_uds;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{Mount, SyntheticMount, SyntheticNode};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: serve_mount_uds <socket-path>");
    let _ = std::fs::remove_file(&path);

    // root/
    //   greeting.txt
    //   notes.md
    //   subdir/
    //     inner.txt
    let root = SyntheticNode::dir()
        .with_child(
            "greeting.txt",
            SyntheticNode::file(b"hello from hyprstream 9p\n".to_vec()),
        )
        .with_child("notes.md", SyntheticNode::file(b"# notes\n".to_vec()))
        .with_child(
            "subdir",
            SyntheticNode::dir().with_child(
                "inner.txt",
                SyntheticNode::file(b"inner file contents\n".to_vec()),
            ),
        );

    let mount: Arc<dyn Mount> = Arc::new(SyntheticMount::new(root));
    let subject = Subject::new("tenant-a");

    eprintln!("serving 9P mount at {path}");
    serve_mount_uds(
        mount,
        subject,
        Arc::new(hyprstream_9p::DenyAllDecider),
        &path,
    )
    .await
}
