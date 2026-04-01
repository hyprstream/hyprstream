//! VFS proxy for bridging sync callers to async Namespace operations.
//!
//! **Security invariant:** The [`VfsOp`] enum deliberately excludes
//! `mount`, `bind_mount`, and `unmount` operations. Scripts cannot
//! modify the namespace through the proxy — they can only read, write,
//! list, and ctl existing mount points.

use crate::namespace::Namespace;
use hyprstream_rpc::Subject;
use std::sync::Arc;

/// Request from sync caller to async VFS layer.
pub enum VfsOp {
    Cat { path: String },
    Ls { path: String },
    Echo { path: String, data: Vec<u8> },
    Ctl { path: String, cmd: Vec<u8> },
    MountPrefixes,
}

pub struct VfsRequest {
    pub op: VfsOp,
    pub reply: std::sync::mpsc::SyncSender<Result<Vec<u8>, String>>,
}

/// Spawn the async VFS proxy task. Returns a Sender for submitting requests.
///
/// The proxy task runs on the tokio runtime and processes VFS operations.
/// MUST be called from within a tokio runtime context.
///
/// Each request is executed in its own spawned task so that a panic in any
/// `Mount` method is caught as a `JoinError` instead of killing the proxy
/// loop and bricking every shell connected to this namespace.
pub fn spawn_vfs_proxy(
    ns: Arc<Namespace>,
    subject: Subject,
) -> tokio::sync::mpsc::Sender<VfsRequest> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<VfsRequest>(64);
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            let ns = Arc::clone(&ns);
            let subject = subject.clone();
            // Spawn each request in its own task. If a Mount method panics,
            // the JoinError is caught here instead of unwinding the proxy loop.
            let handle = tokio::spawn(async move {
                let result = match req.op {
                    VfsOp::Cat { ref path } => ns
                        .cat(path, &subject)
                        .await
                        .map_err(|e| e.to_string()),
                    VfsOp::Ls { ref path } => ns
                        .ls(path, &subject)
                        .await
                        .map(|entries| {
                            entries
                                .iter()
                                .map(|e| {
                                    if e.is_dir {
                                        format!("{}/", e.name)
                                    } else {
                                        e.name.clone()
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                                .into_bytes()
                        })
                        .map_err(|e| e.to_string()),
                    VfsOp::Echo { ref path, ref data } => ns
                        .echo(path, data, &subject)
                        .await
                        .map(|_| Vec::new())
                        .map_err(|e| e.to_string()),
                    VfsOp::Ctl { ref path, ref cmd } => ns
                        .ctl(path, cmd, &subject)
                        .await
                        .map_err(|e| e.to_string()),
                    VfsOp::MountPrefixes => {
                        let prefixes = ns.mount_prefixes();
                        Ok(prefixes.join("\n").into_bytes())
                    }
                };
                (result, req.reply)
            });
            // Await the per-request task: Ok means normal completion,
            // Err(JoinError) means the task panicked.
            match handle.await {
                Ok((result, reply)) => {
                    let _ = reply.send(result);
                }
                Err(join_err) => {
                    // Task panicked — proxy loop survives.
                    // The reply sender was moved into the panicked task and is
                    // now dropped, so the caller's recv() returns RecvError
                    // which builtins already handle as "VFS proxy did not respond".
                    eprintln!("VFS proxy: request panicked: {join_err}");
                }
            }
        }
    });
    tx
}
