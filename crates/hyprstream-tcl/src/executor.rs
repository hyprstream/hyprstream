//! `TclExecutor` — `Send + Sync` handle to a `!Send` molt interpreter on a dedicated thread.
//!
//! The molt `Interp` uses `Rc` internally and is `!Send`. `TclExecutor` spawns a
//! dedicated OS thread that owns the interpreter and processes commands via a bounded
//! `mpsc::sync_channel`. The executor handle is `Send + Sync` (it only holds a
//! `SyncSender`) so it can be shared across async tasks and stored in service state.

use std::sync::mpsc;
use std::sync::Arc;

use hyprstream_vfs::{Namespace, Subject};

use crate::{TclCommand, TclShell};

/// A `Send + Sync` handle to a Tcl interpreter running on a dedicated thread.
///
/// Commands are sent via a bounded channel; the interpreter thread processes them
/// sequentially. Dropping the executor closes the channel, which causes the
/// interpreter thread to exit.
pub struct TclExecutor {
    tx: mpsc::SyncSender<TclCommand>,
}

impl TclExecutor {
    /// Spawn a `TclExecutor` on a dedicated thread.
    ///
    /// The interpreter runs in a loop processing commands until the sender is dropped.
    /// `ns` and `subject` define the VFS namespace and caller identity for builtins.
    /// `rt` is a tokio runtime handle used by builtins that call async VFS operations.
    pub fn spawn(ns: Arc<Namespace>, subject: Subject, rt: tokio::runtime::Handle) -> Self {
        let (tx, rx) = mpsc::sync_channel(64);
        std::thread::spawn(move || {
            let mut shell = TclShell::new(ns, subject, rt);
            while let Ok(cmd) = rx.recv() {
                shell.process_command(cmd);
            }
        });
        Self { tx }
    }

    /// Evaluate a Tcl script asynchronously.
    ///
    /// Sends the script to the interpreter thread and awaits the result.
    /// Uses `spawn_blocking` internally so the async runtime is not blocked
    /// while waiting for the interpreter to respond.
    pub async fn eval(&self, script: &str) -> Result<String, String> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(TclCommand::Eval {
                script: script.to_owned(),
                resp: resp_tx,
            })
            .map_err(|e| format!("executor channel closed: {e}"))?;

        // Use spawn_blocking to avoid blocking the async runtime on the std mpsc recv.
        let result = tokio::task::spawn_blocking(move || {
            resp_rx
                .recv()
                .map_err(|e| format!("executor response failed: {e}"))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {e}"))??;

        result
    }

    /// Evaluate a Tcl script synchronously (blocking the calling thread).
    pub fn eval_blocking(&self, script: &str) -> Result<String, String> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(TclCommand::Eval {
                script: script.to_owned(),
                resp: resp_tx,
            })
            .map_err(|e| format!("executor channel closed: {e}"))?;
        resp_rx
            .recv()
            .map_err(|e| format!("executor response failed: {e}"))?
    }

    /// Set the instruction limit on the interpreter.
    ///
    /// Takes effect for the next `eval` call. Set to 0 for unlimited
    /// (not recommended for untrusted input).
    pub fn set_instruction_limit(&self, limit: usize) {
        // Fire-and-forget: if the channel is closed, we silently ignore.
        let _ = self.tx.send(TclCommand::SetInstructionLimit { limit });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
    use std::collections::HashMap;

    /// Minimal in-memory mount for testing.
    struct MemMount {
        files: HashMap<String, Vec<u8>>,
    }

    impl MemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self {
                files: files
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v.to_vec()))
                    .collect(),
            }
        }
    }

    struct MemFid {
        path: String,
    }

    #[async_trait]
    impl Mount for MemMount {
        async fn walk(
            &self,
            components: &[&str],
            _caller: &Subject,
        ) -> Result<Fid, MountError> {
            let path = components.join("/");
            let exists = self.files.contains_key(&path)
                || self
                    .files
                    .keys()
                    .any(|k| k.starts_with(&format!("{path}/")));
            if !exists && !path.is_empty() {
                return Err(MountError::NotFound(path));
            }
            Ok(Fid::new(MemFid { path }))
        }

        async fn open(
            &self,
            _fid: &mut Fid,
            _mode: u8,
            _caller: &Subject,
        ) -> Result<(), MountError> {
            Ok(())
        }

        async fn read(
            &self,
            fid: &Fid,
            offset: u64,
            _count: u32,
            _caller: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() {
                        return Ok(vec![]);
                    }
                    Ok(data[start..].to_vec())
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(
            &self,
            _fid: &Fid,
            _offset: u64,
            _data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            Err(MountError::NotSupported("read-only".into()))
        }

        async fn readdir(
            &self,
            fid: &Fid,
            _caller: &Subject,
        ) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() {
                String::new()
            } else {
                format!("{}/", inner.path)
            };
            let mut entries = Vec::new();
            for key in self.files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry {
                            name: rest.to_owned(),
                            is_dir: false,
                            size: 0,
                            stat: None,
                        });
                    }
                } else if inner.path.is_empty() && !key.contains('/') {
                    entries.push(DirEntry {
                        name: key.clone(),
                        is_dir: false,
                        size: 0,
                        stat: None,
                    });
                }
            }
            Ok(entries)
        }

        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat {
                qtype: 0,
                size: 0,
                name: inner.path.clone(),
                mtime: 0,
            })
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn make_executor() -> TclExecutor {
        let rt = tokio::runtime::Handle::current();
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![
            ("temperature", b"0.7"),
            ("status", b"loaded"),
        ]));
        ns.mount("/config", mount).unwrap();
        TclExecutor::spawn(Arc::new(ns), Subject::new("test"), rt)
    }

    #[tokio::test]
    async fn eval_returns_result() {
        let exec = make_executor();
        let result = exec.eval("expr {2 + 3}").await.unwrap();
        assert_eq!(result, "5");
    }

    #[tokio::test]
    async fn eval_vfs_cat() {
        let exec = make_executor();
        let result = exec.eval("cat /config/temperature").await.unwrap();
        assert_eq!(result, "0.7");
    }

    #[tokio::test]
    async fn eval_error() {
        let exec = make_executor();
        let result = exec.eval("nonexistent_command").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[tokio::test]
    async fn eval_instruction_limit() {
        let exec = make_executor();
        exec.set_instruction_limit(500);
        // Give the set_instruction_limit message time to be processed.
        // We send a dummy eval first to ensure ordering.
        let _ = exec.eval("expr 1").await;
        let result = exec.eval("while {1} {}").await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("instruction limit"),
            "error should mention instruction limit"
        );
    }

    #[tokio::test]
    async fn concurrent_evals() {
        let exec = Arc::new(make_executor());
        let mut handles = Vec::new();
        for i in 0..10 {
            let exec = Arc::clone(&exec);
            handles.push(tokio::spawn(async move {
                let script = format!("expr {{{i} + 1}}");
                let result = exec.eval(&script).await.unwrap();
                assert_eq!(result, format!("{}", i + 1));
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn eval_blocking_works() {
        let exec = make_executor();
        let result = tokio::task::spawn_blocking(move || exec.eval_blocking("expr {7 * 6}"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result, "42");
    }

    #[tokio::test]
    async fn shutdown_clean() {
        let exec = make_executor();
        // Verify it works.
        let _ = exec.eval("expr 1").await.unwrap();
        // Drop the executor — the interpreter thread should exit.
        drop(exec);
        // No panic, no hang. The thread exits when the channel closes.
    }

    #[tokio::test]
    async fn eval_after_shutdown_errors() {
        let exec = make_executor();
        let _ = exec.eval("expr 1").await.unwrap();
        // Clone the sender before dropping (not possible — we need to test differently).
        // Instead, we test by dropping and trying eval_blocking from a new reference.
        // Since TclExecutor owns the sender, we can't clone it. Test the error path
        // by creating an executor and dropping it, then trying to use a stale reference.

        // We'll test this via the channel directly.
        let (tx, rx) = mpsc::sync_channel::<TclCommand>(1);
        drop(rx); // Simulate interpreter thread gone.
        let stale = TclExecutor { tx };
        let result = stale.eval("expr 1").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("executor channel closed") || err.contains("executor response failed"),
            "unexpected error: {err}"
        );
    }
}
