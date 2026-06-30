//! `/lang/python/` VFS mount backed by the wasm-sandboxed python guest.
//!
//! Exposes a `/lang/python` layout driven by the [`PyShell`](crate::python::PyShell)
//! over the wasm guest, so the interpreter is the sandboxed guest rather than a
//! native one.
//!
//! Layout:
//! - `/lang/python/eval`    — ctl: write a Python EXPRESSION, read `repr(result)`
//!   (with `\n---\n<stdout>` appended if it printed)
//! - `/lang/python/stdout`  — ctl: write Python STATEMENTS, read captured stdout
//! - `/lang/python/vars/`   — dynamic dir: non-dunder globals as readable files
//! - `/lang/python/defs/`   — dynamic dir: callable globals (functions/classes)
//!
//! ## Why a command channel
//!
//! A [`PyShell`](crate::python::PyShell) owns a long-lived wasmtime `Store`. When the
//! sandbox holds a VFS capability, its `vfs_*` host fns `blocking_send` on the proxy
//! — so the shell MUST be driven from a NON-async (plain) thread, never a tokio
//! worker. The async [`Mount`] methods therefore cannot call the shell directly:
//! [`PythonMount`] holds a `tokio::sync::mpsc::Sender<PyCommand>` and forwards each
//! request to a dedicated interpreter thread that owns the `PyShell` and replies
//! over a oneshot, keeping the shell off the async runtime.
//!
//! ## Daemon wiring
//!
//! Registering this mount into a running `/lang/python` namespace lives in the
//! torch-bound `hyprstream` crate, not here. To plug in:
//!
//! ```ignore
//! // In the daemon, where the per-session Namespace is built (alongside the Tcl
//! // `/lang/tcl` mount):
//! let sandbox = hyprstream_workers_wasmtime::Sandbox::from_bytes_for(GUEST_WASM, subject.clone())
//!     .with_vfs(hyprstream_workers_wasmtime::vfs::VfsProxyHandle::new(
//!         hyprstream_vfs::proxy::spawn_vfs_proxy(ns.clone(), subject.clone()),
//!         subject.clone(),
//!     ));
//! let mount = hyprstream_workers_wasmtime::python::mount::PythonMount::spawn(sandbox, PER_CALL_FUEL);
//! ns.mount("/lang/python", std::sync::Arc::new(mount))?;
//! ```
//!
//! The mount owns the interpreter driver thread and joins it on `Drop`, so the
//! daemon only needs to keep the `Arc<dyn Mount>` alive (in the Namespace).

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};
use parking_lot::Mutex;

use crate::python::{PyResult, PyShell};
use crate::Sandbox;

// ─────────────────────────────────────────────────────────────────────────────
// Commands sent from async Mount methods to the interpreter-owning thread.
// ─────────────────────────────────────────────────────────────────────────────

/// A request to the thread that owns the [`PyShell`].
enum PyCommand {
    Eval {
        code: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
    Exec {
        code: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
    ListVars {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    GetVar {
        name: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
    ListDefs {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    GetDef {
        name: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Fid types.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum PyFidKind {
    Root,
    Eval,
    Stdout,
    VarsDir,
    Var(String),
    DefsDir,
    Def(String),
}

/// Fid state. `result` buffers the response from a write so a later read returns it
/// (the Plan9 ctl pattern). `Mutex` for interior mutability through `&Fid`.
struct PyFid {
    kind: PyFidKind,
    result: Mutex<Vec<u8>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// The driver thread + mount.
// ─────────────────────────────────────────────────────────────────────────────

/// VFS mount exposing `/lang/python` over a wasm [`Sandbox`].
///
/// `Send + Sync` (it only holds a channel sender + the driver `JoinHandle`); the
/// `!`-async [`PyShell`] lives on the driver thread. Build with
/// [`PythonMount::spawn`].
///
/// On `Drop`, the mount FIRST drops the command sender (closing the channel so the
/// driver loop's `blocking_recv` returns `None` and the thread exits), THEN joins
/// the thread. Doing both here — rather than in a separate guard — avoids the
/// drop-ordering footgun where a separate guard would try to join while the sender
/// is still alive (the channel never closes → join hangs forever).
pub struct PythonMount {
    /// `Option` so `Drop` can take (and thus drop) it BEFORE joining the thread.
    tx: Option<tokio::sync::mpsc::Sender<PyCommand>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for PythonMount {
    fn drop(&mut self) {
        // Close the channel FIRST so the driver's blocking_recv returns None.
        self.tx.take();
        // Then join the driver thread (it will have exited its recv loop).
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl PythonMount {
    /// Spawn the interpreter driver thread over `sandbox` and return the mount.
    /// `per_call_fuel` is the fuel budget per `eval`/`exec`.
    ///
    /// The driver thread opens the [`PyShell`] (a non-async context, so the
    /// sandbox's `vfs_*` host fns may `blocking_send`) and serves `PyCommand`s until
    /// the channel closes (i.e. until the mount is dropped). The mount owns the
    /// thread and joins it on `Drop`.
    pub fn spawn(sandbox: Sandbox, per_call_fuel: u64) -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<PyCommand>(16);
        let handle = std::thread::Builder::new()
            .name("hyprstream-workers-wasmtime-pyshell".into())
            .spawn(move || {
                // Open the persistent shell on THIS (plain) thread.
                let mut shell: PyShell = match sandbox.open_shell(per_call_fuel) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("pyshell: failed to open: {e}");
                        return;
                    }
                };
                // blocking_recv: this is a dedicated OS thread, not a tokio worker.
                while let Some(cmd) = rx.blocking_recv() {
                    serve(&mut shell, cmd);
                }
            })
            .expect("spawn pyshell driver thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// The command sender (always `Some` until `Drop`).
    fn tx(&self) -> &tokio::sync::mpsc::Sender<PyCommand> {
        self.tx.as_ref().expect("mount sender present until drop")
    }

    async fn request<T>(
        &self,
        build: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> PyCommand,
    ) -> Result<T, MountError> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        self.tx()
            .send(build(resp_tx))
            .await
            .map_err(|_| MountError::Io("python interpreter gone".into()))?;
        resp_rx
            .await
            .map_err(|_| MountError::Io("python interpreter did not respond".into()))
    }
}

/// Serve one command against the shell (runs on the driver thread).
fn serve(shell: &mut PyShell, cmd: PyCommand) {
    match cmd {
        PyCommand::Eval { code, resp } => {
            let _ = resp.send(shell.eval(&code).unwrap_or_else(err_result));
        }
        PyCommand::Exec { code, resp } => {
            let _ = resp.send(shell.exec(&code).unwrap_or_else(err_result));
        }
        PyCommand::ListVars { resp } => {
            let _ = resp.send(shell.list_vars().unwrap_or_default());
        }
        PyCommand::GetVar { name, resp } => {
            let _ = resp.send(shell.get_var(&name).unwrap_or_else(err_result));
        }
        PyCommand::ListDefs { resp } => {
            let _ = resp.send(shell.list_defs().unwrap_or_default());
        }
        PyCommand::GetDef { name, resp } => {
            let _ = resp.send(shell.get_def(&name).unwrap_or_else(err_result));
        }
    }
}

/// A trapped/host-side error surfaces to the guest as a Python-level error string.
fn err_result(e: wasmtime::Error) -> PyResult {
    PyResult::Err(format!("sandbox: {e}"))
}

/// Render a [`PyResult`] for the `eval`/`stdout` ctl read.
fn render(r: PyResult) -> Vec<u8> {
    match r {
        PyResult::Ok(s) => s.into_bytes(),
        PyResult::None => Vec::new(),
        PyResult::Err(e) => format!("error: {e}").into_bytes(),
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Mount for PythonMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let kind = match components {
            [] => PyFidKind::Root,
            ["eval"] => PyFidKind::Eval,
            ["stdout"] => PyFidKind::Stdout,
            ["vars"] => PyFidKind::VarsDir,
            ["vars", name] => PyFidKind::Var((*name).to_owned()),
            ["defs"] => PyFidKind::DefsDir,
            ["defs", name] => PyFidKind::Def((*name).to_owned()),
            _ => return Err(MountError::NotFound(components.join("/"))),
        };
        Ok(Fid::new(PyFid {
            kind,
            result: Mutex::new(Vec::new()),
        }))
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
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
            .downcast_ref::<PyFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let data: Vec<u8> = match &inner.kind {
            PyFidKind::Eval | PyFidKind::Stdout => inner.result.lock().clone(),
            PyFidKind::Var(name) => {
                let r = self
                    .request(|resp| PyCommand::GetVar {
                        name: name.clone(),
                        resp,
                    })
                    .await?;
                match r {
                    PyResult::None => return Err(MountError::NotFound(format!("vars/{name}"))),
                    other => render(other),
                }
            }
            PyFidKind::Def(name) => {
                let r = self
                    .request(|resp| PyCommand::GetDef {
                        name: name.clone(),
                        resp,
                    })
                    .await?;
                match r {
                    PyResult::None => return Err(MountError::NotFound(format!("defs/{name}"))),
                    other => render(other),
                }
            }
            PyFidKind::Root | PyFidKind::VarsDir | PyFidKind::DefsDir => {
                return Err(MountError::IsDirectory("use readdir".into()))
            }
        };
        let start = offset as usize;
        if start >= data.len() {
            return Ok(Vec::new());
        }
        Ok(data[start..].to_vec())
    }

    async fn write(
        &self,
        fid: &Fid,
        _offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let inner = fid
            .downcast_ref::<PyFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        match &inner.kind {
            PyFidKind::Eval => {
                let code = String::from_utf8_lossy(data).into_owned();
                let r = self.request(|resp| PyCommand::Eval { code, resp }).await?;
                *inner.result.lock() = render(r);
                Ok(data.len() as u32)
            }
            PyFidKind::Stdout => {
                let code = String::from_utf8_lossy(data).into_owned();
                let r = self.request(|resp| PyCommand::Exec { code, resp }).await?;
                *inner.result.lock() = render(r);
                Ok(data.len() as u32)
            }
            _ => Err(MountError::NotSupported("read-only".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<PyFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let mk = |name: String, is_dir: bool| DirEntry {
            name,
            is_dir,
            size: 0,
            stat: None,
        };
        match &inner.kind {
            PyFidKind::Root => Ok(vec![
                mk("eval".into(), false),
                mk("stdout".into(), false),
                mk("vars".into(), true),
                mk("defs".into(), true),
            ]),
            PyFidKind::VarsDir => {
                let names = self.request(|resp| PyCommand::ListVars { resp }).await?;
                Ok(names.into_iter().map(|n| mk(n, false)).collect())
            }
            PyFidKind::DefsDir => {
                let names = self.request(|resp| PyCommand::ListDefs { resp }).await?;
                Ok(names.into_iter().map(|n| mk(n, false)).collect())
            }
            _ => Err(MountError::NotDirectory(format!("{:?}", inner.kind))),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<PyFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let (name, qtype) = match &inner.kind {
            PyFidKind::Root => ("python".to_owned(), 0x80u8),
            PyFidKind::Eval => ("eval".to_owned(), 0u8),
            PyFidKind::Stdout => ("stdout".to_owned(), 0u8),
            PyFidKind::VarsDir => ("vars".to_owned(), 0x80u8),
            PyFidKind::Var(n) => (n.clone(), 0u8),
            PyFidKind::DefsDir => ("defs".to_owned(), 0x80u8),
            PyFidKind::Def(n) => (n.clone(), 0u8),
        };
        Ok(Stat {
            qtype,
            size: 0,
            name,
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sandbox;
    use std::path::PathBuf;

    fn guest_wasm() -> Option<Vec<u8>> {
        if let Ok(p) = std::env::var("HYPRSTREAM_PYGUEST_WASM") {
            return std::fs::read(&p).ok();
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let guest_dir = manifest
            .parent()
            .map(|p| p.join("hyprstream-wasm-pyguest"))?;
        for profile in ["release", "debug"] {
            let c = guest_dir
                .join("target/wasm32-unknown-unknown")
                .join(profile)
                .join("hyprstream_wasm_pyguest.wasm");
            if c.exists() {
                if let Ok(b) = std::fs::read(&c) {
                    return Some(b);
                }
            }
        }
        None
    }

    /// CI guard: under CI a missing pyguest artifact is a hard failure (the mount
    /// guarantees must be exercised); locally it still skips.
    fn guest_wasm_or_ci_fail() -> Option<Vec<u8>> {
        match guest_wasm() {
            Some(wasm) => Some(wasm),
            None => {
                assert!(
                    std::env::var("CI").is_err(),
                    "mount test: pyguest wasm not built but running under CI — build the \
                     pyguest and set HYPRSTREAM_PYGUEST_WASM"
                );
                eprintln!(
                    "SKIP mount test: guest wasm not built (set CI=1 to make this a failure)"
                );
                None
            }
        }
    }

    /// End-to-end mount test against an in-memory VFS: write an expr to `eval`, read
    /// the result; exec sets a var, read it back under `vars/`; `readdir` lists the
    /// layout, driven by the wasm guest.
    #[test]
    fn mount_eval_vars_defs_over_guest() {
        let Some(wasm) = guest_wasm_or_ci_fail() else {
            return;
        };
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let subject = Subject::new("alice");
        let sandbox = Sandbox::from_bytes_for(&wasm, subject.clone()).expect("load");
        let mount = PythonMount::spawn(sandbox, 50_000_000_000);

        rt.block_on(async {
            // readdir root -> eval/stdout/vars/defs.
            let root = mount.walk(&[], &subject).await.unwrap();
            let entries = mount.readdir(&root, &subject).await.unwrap();
            let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains(&"eval"));
            assert!(names.contains(&"vars"));
            assert!(names.contains(&"defs"));

            // Write an expression to `eval`, read the repr back.
            let eval = mount.walk(&["eval"], &subject).await.unwrap();
            mount.write(&eval, 0, b"2 + 3", &subject).await.unwrap();
            let out = mount.read(&eval, 0, 4096, &subject).await.unwrap();
            assert_eq!(out, b"5", "eval should return repr 5");

            // exec sets a var via `stdout`, then read it under `vars/x`.
            let stdout = mount.walk(&["stdout"], &subject).await.unwrap();
            mount.write(&stdout, 0, b"x = 42", &subject).await.unwrap();

            let var_x = mount.walk(&["vars", "x"], &subject).await.unwrap();
            let out = mount.read(&var_x, 0, 4096, &subject).await.unwrap();
            assert_eq!(out, b"42", "vars/x should be the persisted value");

            // vars/ readdir includes x.
            let vars_dir = mount.walk(&["vars"], &subject).await.unwrap();
            let entries = mount.readdir(&vars_dir, &subject).await.unwrap();
            assert!(entries.iter().any(|e| e.name == "x"));

            // Define a function; defs/ lists it, vars/ shows it too.
            let stdout2 = mount.walk(&["stdout"], &subject).await.unwrap();
            mount
                .write(&stdout2, 0, b"def f():\n    return 1\n", &subject)
                .await
                .unwrap();
            let defs_dir = mount.walk(&["defs"], &subject).await.unwrap();
            let defs = mount.readdir(&defs_dir, &subject).await.unwrap();
            assert!(defs.iter().any(|e| e.name == "f"), "defs/ should list f");

            eprintln!("mount: eval/vars/defs over guest all green");
        });
    }
}
