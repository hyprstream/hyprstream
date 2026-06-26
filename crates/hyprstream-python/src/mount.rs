//! `/lang/python/` VFS mount — exposes Python interpreter state as files.
//!
//! `PythonShell` is `!Send` (RustPython uses `Rc`), so the mount cannot hold it
//! directly. `PythonMount` holds an `mpsc::Sender<PyCommand>` channel. Mount
//! methods send commands through the channel and block on a oneshot for the
//! response. The interpreter loop runs on the owning thread, which calls
//! `PythonShell::process_command()` for each received message.
//!
//! Layout:
//! - `/lang/python/eval`    — ctl: write Python expression, read `repr(result)\n---\nstdout`
//! - `/lang/python/stdout`  — ctl: write Python statements, read captured stdout
//! - `/lang/python/vars/`   — dynamic dir: non-dunder globals as readable files
//! - `/lang/python/defs/`   — dynamic dir: callable globals (functions/classes)

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};
use parking_lot::Mutex;

// ─────────────────────────────────────────────────────────────────────────────
// PyCommand — messages sent from mount threads to the interpreter owner
// ─────────────────────────────────────────────────────────────────────────────

/// A command sent from `PythonMount` to the thread owning the Python interpreter.
pub enum PyCommand {
    /// Evaluate a Python expression. Response: `Ok((repr, stdout))` or `Err(msg)`.
    Eval {
        code: String,
        resp: tokio::sync::oneshot::Sender<Result<(String, String), String>>,
    },
    /// Execute Python statements. Response: `Ok(stdout)` or `Err(msg)`.
    Exec {
        code: String,
        resp: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// List non-dunder global variable names.
    ListVars {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// Get the repr of a global variable.
    GetVar {
        name: String,
        resp: tokio::sync::oneshot::Sender<Option<String>>,
    },
    /// List callable global names (functions/classes).
    ListDefs {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// Get the repr of a callable global.
    GetDef {
        name: String,
        resp: tokio::sync::oneshot::Sender<Option<String>>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Fid types — internal state for each open file handle
// ─────────────────────────────────────────────────────────────────────────────

/// Which kind of file a fid refers to.
#[derive(Clone, Debug)]
enum PyFidKind {
    /// The root directory `/lang/python/`.
    Root,
    /// The `eval` ctl file (expression evaluator).
    Eval,
    /// The `stdout` ctl file (statement executor, captures stdout).
    Stdout,
    /// The `vars/` directory.
    VarsDir,
    /// A specific variable: `vars/<name>`.
    Var(String),
    /// The `defs/` directory.
    DefsDir,
    /// A specific definition: `defs/<name>`.
    Def(String),
}

/// Fid state for the Python mount.
///
/// `eval_result` stores the response from the interpreter after a write, so
/// a subsequent read can return it. Uses `Mutex` for interior mutability
/// through `&Fid` without unsafe (unlike the TCL mount's raw pointer cast).
struct PyFid {
    kind: PyFidKind,
    eval_result: Mutex<Vec<u8>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PythonMount
// ─────────────────────────────────────────────────────────────────────────────

/// VFS mount that proxies requests to a `!Send` Python interpreter via channels.
///
/// Mount at `/lang/python` in the namespace. The interpreter owner must poll the
/// `Receiver<PyCommand>` returned by [`create_mount_channel`] and call
/// `PythonShell::process_command()` for each message.
pub struct PythonMount {
    tx: tokio::sync::mpsc::Sender<PyCommand>,
}

impl PythonMount {
    /// Create a new `PythonMount` from the sending half of a mount channel.
    pub fn new(tx: tokio::sync::mpsc::Sender<PyCommand>) -> Self {
        Self { tx }
    }

    /// Send a command and await the interpreter response.
    async fn request<T>(
        &self,
        build: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> PyCommand,
    ) -> Result<T, MountError> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let cmd = build(resp_tx);
        self.tx
            .send(cmd)
            .await
            .map_err(|_| MountError::Io("python interpreter gone".into()))?;
        resp_rx
            .await
            .map_err(|_| MountError::Io("python interpreter did not respond".into()))
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
            _ => {
                return Err(MountError::NotFound(components.join("/")));
            }
        };
        Ok(Fid::new(PyFid {
            kind,
            eval_result: Mutex::new(Vec::new()),
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
            PyFidKind::Eval => {
                let buf = inner.eval_result.lock();
                buf.clone()
            }
            PyFidKind::Stdout => {
                let buf = inner.eval_result.lock();
                buf.clone()
            }
            PyFidKind::Var(name) => {
                let val = self
                    .request(|resp| PyCommand::GetVar { name: name.clone(), resp })
                    .await?;
                val.map(|s| s.into_bytes())
                    .ok_or_else(|| MountError::NotFound(format!("vars/{name}")))?
            }
            PyFidKind::Def(name) => {
                let val = self
                    .request(|resp| PyCommand::GetDef { name: name.clone(), resp })
                    .await?;
                val.map(|s| s.into_bytes())
                    .ok_or_else(|| MountError::NotFound(format!("defs/{name}")))?
            }
            PyFidKind::Root | PyFidKind::VarsDir | PyFidKind::DefsDir => {
                return Err(MountError::IsDirectory("use readdir".into()));
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
                let result = self
                    .request(|resp| PyCommand::Eval { code, resp })
                    .await?;
                let response_bytes = match result {
                    Ok((repr, stdout)) => {
                        if stdout.is_empty() {
                            repr.into_bytes()
                        } else if repr.is_empty() {
                            stdout.into_bytes()
                        } else {
                            format!("{repr}\n---\n{stdout}").into_bytes()
                        }
                    }
                    Err(e) => format!("error: {e}").into_bytes(),
                };
                *inner.eval_result.lock() = response_bytes;
                Ok(data.len() as u32)
            }
            PyFidKind::Stdout => {
                let code = String::from_utf8_lossy(data).into_owned();
                let result = self
                    .request(|resp| PyCommand::Exec { code, resp })
                    .await?;
                let response_bytes = match result {
                    Ok(stdout) => stdout.into_bytes(),
                    Err(e) => format!("error: {e}").into_bytes(),
                };
                *inner.eval_result.lock() = response_bytes;
                Ok(data.len() as u32)
            }
            _ => Err(MountError::NotSupported("read-only".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<PyFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match &inner.kind {
            PyFidKind::Root => Ok(vec![
                DirEntry { name: "eval".into(), is_dir: false, size: 0, stat: None },
                DirEntry { name: "stdout".into(), is_dir: false, size: 0, stat: None },
                DirEntry { name: "vars".into(), is_dir: true, size: 0, stat: None },
                DirEntry { name: "defs".into(), is_dir: true, size: 0, stat: None },
            ]),
            PyFidKind::VarsDir => {
                let names: Vec<String> =
                    self.request(|resp| PyCommand::ListVars { resp }).await?;
                Ok(names
                    .into_iter()
                    .map(|name| DirEntry { name, is_dir: false, size: 0, stat: None })
                    .collect())
            }
            PyFidKind::DefsDir => {
                let names: Vec<String> =
                    self.request(|resp| PyCommand::ListDefs { resp }).await?;
                Ok(names
                    .into_iter()
                    .map(|name| DirEntry { name, is_dir: false, size: 0, stat: None })
                    .collect())
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

        Ok(Stat::unknown_qid(qtype, 0, name, 0))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel constructor
// ─────────────────────────────────────────────────────────────────────────────

/// Create a bounded channel pair for the Python mount.
///
/// Returns `(sender, receiver)`. Pass the sender to [`PythonMount::new`] and
/// drive the receiver from the thread owning the `PythonShell`.
pub fn create_mount_channel() -> (
    tokio::sync::mpsc::Sender<PyCommand>,
    tokio::sync::mpsc::Receiver<PyCommand>,
) {
    tokio::sync::mpsc::channel(16)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Spawn a fake interpreter task that handles commands.
    fn spawn_fake_interp(
        mut rx: tokio::sync::mpsc::Receiver<PyCommand>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    PyCommand::Eval { code, resp } => {
                        let result = if code.contains("error") {
                            Err("test error".into())
                        } else {
                            Ok((format!("result: {code}"), String::new()))
                        };
                        let _ = resp.send(result);
                    }
                    PyCommand::Exec { code, resp } => {
                        let _ = resp.send(Ok(format!("stdout: {code}")));
                    }
                    PyCommand::ListVars { resp } => {
                        let _ = resp.send(vec!["x".into(), "y".into()]);
                    }
                    PyCommand::GetVar { name, resp } => {
                        let val = match name.as_str() {
                            "x" => Some("42".into()),
                            "y" => Some("hello".into()),
                            _ => None,
                        };
                        let _ = resp.send(val);
                    }
                    PyCommand::ListDefs { resp } => {
                        let _ = resp.send(vec!["my_func".into()]);
                    }
                    PyCommand::GetDef { name, resp } => {
                        let val = match name.as_str() {
                            "my_func" => Some("<function my_func>".into()),
                            _ => None,
                        };
                        let _ = resp.send(val);
                    }
                }
            }
        })
    }

    #[tokio::test]
    async fn walk_root() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&[], &subject).await.unwrap();
        let inner = fid.downcast_ref::<PyFid>().unwrap();
        assert!(matches!(inner.kind, PyFidKind::Root));
    }

    #[tokio::test]
    async fn walk_not_found() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let result = mount.walk(&["nonexistent"], &subject).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn stat_root_is_dir() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&[], &subject).await.unwrap();
        let st = mount.stat(&fid, &subject).await.unwrap();
        assert_eq!(st.qtype, 0x80);
        assert_eq!(st.name, "python");
    }

    #[tokio::test]
    async fn eval_write_then_read() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["eval"], &subject).await.unwrap();
        mount.open(&mut fid, 2, &subject).await.unwrap();

        let written = mount.write(&fid, 0, b"2+3", &subject).await.unwrap();
        assert_eq!(written, 3);

        let data = mount.read(&fid, 0, 4096, &subject).await.unwrap();
        assert_eq!(String::from_utf8(data).unwrap(), "result: 2+3");
    }

    #[tokio::test]
    async fn readdir_vars() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["vars"], &subject).await.unwrap();
        let entries = mount.readdir(&fid, &subject).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));
    }

    #[tokio::test]
    async fn readdir_defs() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["defs"], &subject).await.unwrap();
        let entries = mount.readdir(&fid, &subject).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "my_func");
    }

    #[tokio::test]
    async fn read_var() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = PythonMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["vars", "x"], &subject).await.unwrap();
        mount.open(&mut fid, 0, &subject).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject).await.unwrap();
        assert_eq!(String::from_utf8(data).unwrap(), "42");
    }
}
