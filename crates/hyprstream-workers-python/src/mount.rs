//! `/lang/python/` VFS mount backed by the wasm-sandboxed python guest.
//!
//! Exposes a `/lang/python` layout driven by a [`PythonShell`](crate::PythonShell)
//! over the wasm guest, so the interpreter is the sandboxed guest rather than a native
//! one.
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
//! A [`PythonShell`](crate::PythonShell) owns a long-lived wasmtime `Store`. When the
//! sandbox holds a VFS capability, its `vfs_*` host fns `blocking_send` on the proxy —
//! so the shell MUST be driven from a NON-async thread, never a tokio worker. The
//! async [`Mount`] methods therefore cannot call the shell directly: [`PythonMount`]
//! holds a `tokio::sync::mpsc::Sender<PyCommand>` and forwards each request to the
//! thread that owns the shell, which replies over a oneshot.
//!
//! This mirrors `hyprstream-workers-tcl`'s `TclMount`/`TclCommand`/`create_mount_channel`
//! model: build the channel with [`create_mount_channel`], mount
//! `PythonMount::new(tx)` at `/lang/python`, and have the shell owner drain the
//! `Receiver<PyCommand>` in its event loop (e.g. `ChatApp::tick()`), calling
//! [`PythonShell::process_command`](crate::PythonShell::process_command).
//!
//! Built on `hyprstream-vfs`'s `devfile` primitives (`ControlFile` for
//! `eval`/`stdout`, `DynamicDir` for `vars`/`defs`) rather than hand-rolling
//! the write→latch→read state machine — see `hyprstream_vfs::devfile` (#609).

use async_trait::async_trait;
use hyprstream_vfs::devfile::{ControlFile, DevFileState, DevFuture, DynamicDir};
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};

use crate::PyResult;

// ─────────────────────────────────────────────────────────────────────────────
// Commands sent from async Mount methods to the shell-owning thread.
// ─────────────────────────────────────────────────────────────────────────────

/// A request from [`PythonMount`] to the thread owning the [`PythonShell`](crate::PythonShell).
pub enum PyCommand {
    /// Evaluate a Python expression. Response: the rendered [`PyResult`].
    Eval {
        code: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
    /// Execute Python statements. Response: captured stdout as a [`PyResult`].
    Exec {
        code: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
    /// List non-dunder global variable names.
    ListVars {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// `repr` of one global variable (or `None` if absent).
    GetVar {
        name: String,
        resp: tokio::sync::oneshot::Sender<PyResult>,
    },
    /// List callable global names.
    ListDefs {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// `repr` of one callable global (or `None` if absent/not callable).
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

/// Fid state. `ctl_state` buffers the response from a write to `eval`/`stdout`
/// so a later read returns it (the Plan9 ctl pattern, via
/// `devfile::ControlFile`). Unused for non-ctl fid kinds.
struct PyFid {
    kind: PyFidKind,
    ctl_state: DevFileState,
}

// ─────────────────────────────────────────────────────────────────────────────
// devfile callback type aliases
// ─────────────────────────────────────────────────────────────────────────────

type PyCtl = ControlFile<Box<dyn Fn(Vec<u8>) -> DevFuture<'static, Result<Vec<u8>, MountError>> + Send + Sync>>;
type ListFn = Box<dyn Fn() -> DevFuture<'static, Vec<String>> + Send + Sync>;
type GetFn = Box<dyn Fn(String) -> DevFuture<'static, Result<Option<Vec<u8>>, MountError>> + Send + Sync>;
type FieldDir = DynamicDir<ListFn, GetFn>;

// ─────────────────────────────────────────────────────────────────────────────
// PythonMount
// ─────────────────────────────────────────────────────────────────────────────

/// VFS mount that proxies `/lang/python` requests to a [`PythonShell`](crate::PythonShell)
/// over a command channel.
///
/// Mount at `/lang/python` in the namespace. The shell owner must drain the
/// `Receiver<PyCommand>` returned by [`create_mount_channel`] in its event loop,
/// forwarding each command to the shell via
/// [`PythonShell::process_command`](crate::PythonShell::process_command).
pub struct PythonMount {
    eval: PyCtl,
    stdout: PyCtl,
    vars: FieldDir,
    defs: FieldDir,
}

impl PythonMount {
    /// Create a new `PythonMount` from the sending half of a mount channel.
    pub fn new(tx: tokio::sync::mpsc::Sender<PyCommand>) -> Self {
        let eval_tx = tx.clone();
        let eval_handler = move |data: Vec<u8>| -> DevFuture<'static, Result<Vec<u8>, MountError>> {
            let tx = eval_tx.clone();
            let code = String::from_utf8_lossy(&data).into_owned();
            Box::pin(async move {
                let r = request(&tx, |resp| PyCommand::Eval { code, resp }).await?;
                Ok(render(r))
            })
        };
        let eval = ControlFile::new(Box::new(eval_handler) as _);

        let stdout_tx = tx.clone();
        let stdout_handler = move |data: Vec<u8>| -> DevFuture<'static, Result<Vec<u8>, MountError>> {
            let tx = stdout_tx.clone();
            let code = String::from_utf8_lossy(&data).into_owned();
            Box::pin(async move {
                let r = request(&tx, |resp| PyCommand::Exec { code, resp }).await?;
                Ok(render(r))
            })
        };
        let stdout = ControlFile::new(Box::new(stdout_handler) as _);

        let vars_list_tx = tx.clone();
        let vars_get_tx = tx.clone();
        let vars = DynamicDir::new(
            Box::new(move || {
                let tx = vars_list_tx.clone();
                Box::pin(async move { request(&tx, |resp| PyCommand::ListVars { resp }).await.unwrap_or_default() })
                    as DevFuture<'static, _>
            }) as ListFn,
            Box::new(move |name: String| {
                let tx = vars_get_tx.clone();
                Box::pin(async move {
                    let r = request(&tx, |resp| PyCommand::GetVar { name, resp }).await?;
                    Ok(match r {
                        PyResult::None => None,
                        other => Some(render(other)),
                    })
                }) as DevFuture<'static, _>
            }) as GetFn,
        );

        let defs_list_tx = tx.clone();
        let defs_get_tx = tx;
        let defs = DynamicDir::new(
            Box::new(move || {
                let tx = defs_list_tx.clone();
                Box::pin(async move { request(&tx, |resp| PyCommand::ListDefs { resp }).await.unwrap_or_default() })
                    as DevFuture<'static, _>
            }) as ListFn,
            Box::new(move |name: String| {
                let tx = defs_get_tx.clone();
                Box::pin(async move {
                    let r = request(&tx, |resp| PyCommand::GetDef { name, resp }).await?;
                    Ok(match r {
                        PyResult::None => None,
                        other => Some(render(other)),
                    })
                }) as DevFuture<'static, _>
            }) as GetFn,
        );

        Self { eval, stdout, vars, defs }
    }
}

/// Send a command and await the shell response.
async fn request<T>(
    tx: &tokio::sync::mpsc::Sender<PyCommand>,
    build: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> PyCommand,
) -> Result<T, MountError> {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    tx.send(build(resp_tx))
        .await
        .map_err(|_| MountError::Io("python interpreter gone".into()))?;
    resp_rx
        .await
        .map_err(|_| MountError::Io("python interpreter did not respond".into()))
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
            ctl_state: DevFileState::new(),
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
        match &inner.kind {
            PyFidKind::Eval => Ok(self.eval.handle_read(&inner.ctl_state, offset)),
            PyFidKind::Stdout => Ok(self.stdout.handle_read(&inner.ctl_state, offset)),
            PyFidKind::Var(name) => self.vars.read(name, offset).await,
            PyFidKind::Def(name) => self.defs.read(name, offset).await,
            PyFidKind::Root | PyFidKind::VarsDir | PyFidKind::DefsDir => {
                Err(MountError::IsDirectory("use readdir".into()))
            }
        }
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
            PyFidKind::Eval => self.eval.handle_write(&inner.ctl_state, data).await,
            PyFidKind::Stdout => self.stdout.handle_write(&inner.ctl_state, data).await,
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
            PyFidKind::VarsDir => Ok(self.vars.readdir().await),
            PyFidKind::DefsDir => Ok(self.defs.readdir().await),
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

/// Create a `/lang/python` mount channel: the `Sender` goes to a
/// [`PythonMount::new`], the `Receiver` is drained by the shell owner (which forwards
/// each [`PyCommand`] to its [`PythonShell`](crate::PythonShell) via
/// [`process_command`](crate::PythonShell::process_command)).
pub fn create_mount_channel() -> (
    tokio::sync::mpsc::Sender<PyCommand>,
    tokio::sync::mpsc::Receiver<PyCommand>,
) {
    tokio::sync::mpsc::channel(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PythonShell;
    use hyprstream_workers_wasmtime::Sandbox;
    use std::path::PathBuf;

    fn guest_wasm() -> Option<Vec<u8>> {
        if let Ok(p) = std::env::var("HYPRSTREAM_PYGUEST_WASM") {
            return std::fs::read(&p).ok();
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let guest_dir = manifest
            .parent()
            .map(|p| p.join("hyprstream-workers-python-guest"))?;
        for profile in ["release", "debug"] {
            let c = guest_dir
                .join("target/wasm32-unknown-unknown")
                .join(profile)
                .join("hyprstream_workers_python_guest.wasm");
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
                eprintln!("SKIP mount test: guest wasm not built (set CI=1 to make this a failure)");
                None
            }
        }
    }

    /// End-to-end mount test against an in-memory VFS: write an expr to `eval`, read
    /// the result; exec sets a var, read it back under `vars/`; `readdir` lists the
    /// layout, driven by the wasm guest. The shell runs on a dedicated thread that
    /// drains the mount channel (mirroring the daemon owner loop).
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
        let (tx, mut rx) = create_mount_channel();

        // Shell owner thread: open the shell on a plain thread (vfs_* may
        // blocking_send) and drain the mount channel.
        let owner_wasm = wasm.clone();
        let owner_subject = subject.clone();
        let owner = std::thread::Builder::new()
            .name("test-pyshell-owner".into())
            .spawn(move || {
                let sandbox = Sandbox::from_bytes_for(&owner_wasm, owner_subject).expect("load");
                let mut shell = PythonShell::open(&sandbox, 50_000_000_000).expect("open shell");
                while let Some(cmd) = rx.blocking_recv() {
                    shell.process_command(cmd);
                }
            })
            .expect("spawn owner");

        let mount = PythonMount::new(tx);

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

            // Define a function; defs/ lists it.
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

        // Close the channel so the owner thread exits, then join.
        drop(mount);
        owner.join().unwrap();
    }
}
