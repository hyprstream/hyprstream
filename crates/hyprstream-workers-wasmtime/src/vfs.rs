//! VFS capability backing for the sandbox (#483 P2 — REAL, not a sketch).
//!
//! This is the proof the sandbox capability is real: a guest `vfs_*` call is
//! bridged, synchronously, to the canonical `hyprstream_vfs` [`Namespace`] through
//! the [`spawn_vfs_proxy`] seam, and is scoped to the [`Subject`] the `Sandbox` is
//! bound to — never to any identity the guest supplies.
//!
//! ## Wiring
//!
//! ```text
//!   guest env::vfs_cat(path)        -> VfsOp::Cat  { path }   -> Namespace::cat
//!   guest env::vfs_ls(path)         -> VfsOp::Ls   { path }   -> Namespace::ls
//!   guest env::vfs_echo(path,data)  -> VfsOp::Echo { path, data } -> Namespace::echo
//!   guest env::vfs_ctl(path,cmd)    -> VfsOp::Ctl  { path, cmd }  -> Namespace::ctl
//! ```
//!
//! The host fns (in `build_linker`) read `caller.data().subject` and submit a
//! [`VfsRequest`] whose `op` carries the Subject IMPLICITLY: the proxy task is
//! `spawn_vfs_proxy(ns, subject)` — i.e. each proxy is pinned to ONE Subject at
//! spawn time, so every request through it is executed as that Subject. The guest
//! has no way to name or forge an identity.
//!
//! ## Why path-based ops, not the P1 fid-based sketch
//!
//! The P1 `vfs.rs` sketched a fid-based `VfsCapability` (walk/open/read/write/…),
//! mirroring [`hyprstream_vfs::Mount`]. The CANONICAL sync seam, however, is the
//! proxy's [`VfsOp`] enum, which is PATH-based (`Cat`/`Ls`/`Echo`/`Ctl`) and
//! DELIBERATELY excludes `mount`/`bind`/`unmount` (its security invariant: scripts
//! cannot remount). We back the capability against that real seam rather than the
//! sketch, so the #483 capability is exactly the proxy's bounded surface.
//!
//! ## Sync-over-async bridge
//!
//! wasm host fns are synchronous; the `Namespace` is async. The proxy
//! ([`spawn_vfs_proxy`]) bridges: it returns a `tokio::sync::mpsc::Sender<VfsRequest>`
//! and each [`VfsRequest`] carries a `std::sync::mpsc::SyncSender` reply channel. A
//! host fn does `blocking_send` of the request, then blocks on the std reply
//! receiver — never touching the async runtime directly. (`blocking_send` requires
//! we are NOT on a tokio worker thread; the `Sandbox` is driven from a plain thread,
//! and the proxy task runs on a separate runtime — see the tests.)

use hyprstream_rpc::Subject;
use hyprstream_vfs::proxy::{VfsOp, VfsRequest};
use tokio::sync::mpsc::Sender as TokioSender;

/// A handle to a Subject-scoped VFS proxy.
///
/// Created by [`spawn_vfs_proxy`] (one per Subject) and stored in
/// [`crate::SandboxState`]. The host `vfs_*` fns submit ops over it and block on
/// the reply. The Subject is implicit in the proxy itself, so this handle does NOT
/// re-pass a Subject per call — which is exactly the property we want: the guest
/// cannot change identity.
#[derive(Clone)]
pub struct VfsProxyHandle {
    tx: TokioSender<VfsRequest>,
    /// The Subject this proxy is pinned to (kept for auditing / introspection only;
    /// it is NOT sent per-request — the proxy already runs as this Subject).
    subject: Subject,
}

/// Result of a VFS op exposed to the guest: a status tag + a UTF-8 payload.
///
/// `ok == true` -> payload is the result bytes; `ok == false` -> payload is the
/// error string. This is the shape the host fns serialise into the guest reply
/// buffer (status byte + payload).
pub struct VfsReply {
    pub ok: bool,
    pub body: Vec<u8>,
}

impl VfsProxyHandle {
    /// Wrap a proxy `Sender` pinned to `subject`.
    pub fn new(tx: TokioSender<VfsRequest>, subject: Subject) -> Self {
        Self { tx, subject }
    }

    /// The Subject every op through this handle runs as.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Submit one op and block on the reply (sync-over-async bridge).
    ///
    /// Returns `Err(String)` only for transport failures (proxy gone / no reply);
    /// VFS-level errors come back as `VfsReply { ok: false, .. }`.
    fn submit(&self, op: VfsOp) -> VfsReply {
        // std::sync::mpsc rendezvous channel for the reply (the proxy's reply type).
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        let req = VfsRequest {
            op,
            reply: reply_tx,
        };
        // `blocking_send`: the host fn is on a non-async thread; this parks until
        // the proxy task accepts the request. If the proxy is gone, surface it as
        // a VFS error rather than panicking the guest call.
        if self.tx.blocking_send(req).is_err() {
            return VfsReply {
                ok: false,
                body: b"vfs: proxy unavailable".to_vec(),
            };
        }
        match reply_rx.recv() {
            Ok(Ok(bytes)) => VfsReply {
                ok: true,
                body: bytes,
            },
            Ok(Err(msg)) => VfsReply {
                ok: false,
                body: msg.into_bytes(),
            },
            Err(_) => VfsReply {
                ok: false,
                body: b"vfs: proxy did not respond".to_vec(),
            },
        }
    }

    /// `cat` — read the whole contents of a file path.
    pub fn cat(&self, path: &str) -> VfsReply {
        self.submit(VfsOp::Cat {
            path: path.to_owned(),
        })
    }

    /// `ls` — list a directory path (newline-joined names, `name/` for dirs).
    pub fn ls(&self, path: &str) -> VfsReply {
        self.submit(VfsOp::Ls {
            path: path.to_owned(),
        })
    }

    /// `echo` — write `data` to a file path. Reply body is empty on success.
    pub fn echo(&self, path: &str, data: &[u8]) -> VfsReply {
        self.submit(VfsOp::Echo {
            path: path.to_owned(),
            data: data.to_vec(),
        })
    }

    /// `ctl` — write `cmd` to a ctl file path and read the response.
    pub fn ctl(&self, path: &str, cmd: &[u8]) -> VfsReply {
        self.submit(VfsOp::Ctl {
            path: path.to_owned(),
            cmd: cmd.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_vfs::proxy::spawn_vfs_proxy;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Namespace, Stat};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// In-memory `Mount` for the seam test: a flat file map under one prefix.
    /// Walk records the requested path; read returns the file (or the write buffer);
    /// write stores into the per-fid buffer. A `denied` Subject is refused at every
    /// op so we can prove Subject-scoping reaches the backend.
    struct MemMount {
        files: parking_lot::Mutex<HashMap<String, Vec<u8>>>,
    }

    struct MemFid {
        path: String,
        write_buf: parking_lot::Mutex<Vec<u8>>,
    }

    impl MemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self {
                files: parking_lot::Mutex::new(
                    files
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), v.to_vec()))
                        .collect(),
                ),
            }
        }

        fn check(caller: &Subject) -> Result<(), MountError> {
            if caller.name() == Some("denied") {
                return Err(MountError::PermissionDenied("denied subject".into()));
            }
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl Mount for MemMount {
        async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError> {
            Self::check(caller)?;
            Ok(Fid::new(MemFid {
                path: components.join("/"),
                write_buf: parking_lot::Mutex::new(Vec::new()),
            }))
        }
        async fn open(
            &self,
            _fid: &mut Fid,
            _mode: u8,
            caller: &Subject,
        ) -> Result<(), MountError> {
            Self::check(caller)
        }
        async fn read(
            &self,
            fid: &Fid,
            offset: u64,
            _count: u32,
            caller: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            Self::check(caller)?;
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let data = self
                .files
                .lock()
                .get(&inner.path)
                .cloned()
                .ok_or_else(|| MountError::NotFound(inner.path.clone()))?;
            let start = (offset as usize).min(data.len());
            Ok(data[start..].to_vec())
        }
        async fn write(
            &self,
            fid: &Fid,
            _offset: u64,
            data: &[u8],
            caller: &Subject,
        ) -> Result<u32, MountError> {
            Self::check(caller)?;
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            inner.write_buf.lock().extend_from_slice(data);
            // Persist so a later cat() sees it.
            self.files
                .lock()
                .insert(inner.path.clone(), inner.write_buf.lock().clone());
            Ok(data.len() as u32)
        }
        async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Self::check(caller)?;
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() {
                String::new()
            } else {
                format!("{}/", inner.path)
            };
            let mut entries = Vec::new();
            for key in self.files.lock().keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry {
                            name: rest.to_owned(),
                            is_dir: false,
                            size: 0,
                            stat: None,
                        });
                    }
                }
            }
            Ok(entries)
        }
        async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError> {
            Self::check(caller)?;
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat {
                qtype: 0,
                version: 0,
                path: 0,
                size: 0,
                name: inner.path.clone(),
                mtime: 0,
            })
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn make_ns() -> Arc<Namespace> {
        let mut ns = Namespace::new();
        ns.mount(
            "/config",
            Arc::new(MemMount::new(vec![("temperature", b"0.7")])),
        )
        .unwrap();
        Arc::new(ns)
    }

    /// An allowed Subject can cat/echo through the proxy-backed handle, and a
    /// `denied` Subject is refused at the Mount — proving the Subject reaches the
    /// backend. The proxy runs on its own tokio runtime; the handle is driven from
    /// a plain (non-async) thread, exactly like the wasm host fns.
    #[test]
    fn proxy_handle_is_subject_scoped() {
        // Dedicated runtime thread to host the proxy tasks.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let ns = make_ns();

        let alice = Subject::new("alice");
        let denied = Subject::new("denied");

        // spawn_vfs_proxy MUST run inside a runtime context.
        let alice_tx = rt.block_on(async { spawn_vfs_proxy(Arc::clone(&ns), alice.clone()) });
        let denied_tx = rt.block_on(async { spawn_vfs_proxy(Arc::clone(&ns), denied.clone()) });

        let alice_h = VfsProxyHandle::new(alice_tx, alice);
        let denied_h = VfsProxyHandle::new(denied_tx, denied);

        // Drive the handles from a plain thread (blocking_send forbidden on a
        // tokio worker; this mirrors the Sandbox's non-async driving thread).
        let result = std::thread::spawn(move || {
            // Allowed subject reads an existing file.
            let r = alice_h.cat("/config/temperature");
            assert!(r.ok, "alice cat should succeed");
            assert_eq!(r.body, b"0.7");

            // Allowed subject writes, then reads back (echo persists in the mount).
            let w = alice_h.echo("/config/temperature", b"0.9");
            assert!(w.ok, "alice echo should succeed");
            let r2 = alice_h.cat("/config/temperature");
            assert!(r2.ok);
            assert_eq!(r2.body, b"0.9");

            // Denied subject is refused at the backend (Subject reached the Mount).
            let d = denied_h.cat("/config/temperature");
            assert!(!d.ok, "denied subject must be refused");
            assert!(
                String::from_utf8_lossy(&d.body).contains("permission denied"),
                "got: {}",
                String::from_utf8_lossy(&d.body)
            );
        })
        .join();
        assert!(result.is_ok());
    }
}
