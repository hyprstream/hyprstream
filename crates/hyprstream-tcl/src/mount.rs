//! `/lang/tcl/` VFS mount — exposes Tcl interpreter state as files.
//!
//! The molt `Interp` is `!Send` (uses `Rc`), so the mount cannot hold it directly.
//! Instead, `TclMount` holds a `SyncSender<TclCommand>` channel. Mount methods
//! send commands through the channel and block on a oneshot for the response.
//! The interpreter loop runs on the owning thread (ChatApp), which polls the
//! receiver in its `tick()` method.
//!
//! Layout:
//! - `/lang/tcl/eval`   — ctl file: write a Tcl script, read the result
//! - `/lang/tcl/vars/`  — dynamic dir: list Tcl variables as readable files
//! - `/lang/tcl/procs/` — dynamic dir: list defined procs as readable files

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};

// ─────────────────────────────────────────────────────────────────────────────
// TclCommand — messages sent from mount threads to the interpreter owner
// ─────────────────────────────────────────────────────────────────────────────

/// A command sent from `TclMount` to the thread owning the Tcl interpreter.
pub enum TclCommand {
    /// Evaluate a Tcl script. Response: `Ok(result)` or `Err(error_msg)`.
    Eval {
        script: String,
        resp: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// List all variable names in the current scope.
    ListVars {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// Get the value of a scalar variable.
    GetVar {
        name: String,
        resp: tokio::sync::oneshot::Sender<Option<String>>,
    },
    /// List all user-defined procedure names.
    ListProcs {
        resp: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// Get the body of a procedure.
    GetProc {
        name: String,
        resp: tokio::sync::oneshot::Sender<Option<String>>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Fid types — internal state for each open file handle
// ─────────────────────────────────────────────────────────────────────────────

/// Which kind of file a fid refers to.
#[derive(Clone, Debug)]
enum TclFidKind {
    /// The root directory `/lang/tcl/`.
    Root,
    /// The `eval` ctl file.
    Eval,
    /// The `vars/` directory.
    VarsDir,
    /// A specific variable: `vars/<name>`.
    Var(String),
    /// The `procs/` directory.
    ProcsDir,
    /// A specific procedure: `procs/<name>`.
    Proc(String),
}

/// Fid state for the Tcl mount.
struct TclFid {
    kind: TclFidKind,
    /// Buffer for ctl write→read pattern (eval).
    write_buf: Vec<u8>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TclMount
// ─────────────────────────────────────────────────────────────────────────────

/// VFS mount that proxies requests to a `!Send` Tcl interpreter via channels.
///
/// Mount at `/lang/tcl` in the namespace. The interpreter owner must poll the
/// `Receiver<TclCommand>` returned by [`create_mount_channel`] in its event
/// loop (e.g., `ChatApp::tick()`).
pub struct TclMount {
    tx: tokio::sync::mpsc::Sender<TclCommand>,
}

impl TclMount {
    /// Create a new `TclMount` from the sending half of a mount channel.
    pub fn new(tx: tokio::sync::mpsc::Sender<TclCommand>) -> Self {
        Self { tx }
    }

    /// Send a command and await the interpreter response.
    async fn request<T>(&self, build: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> TclCommand) -> Result<T, MountError> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let cmd = build(resp_tx);
        self.tx
            .send(cmd)
            .await
            .map_err(|_| MountError::Io("tcl interpreter gone".into()))?;
        resp_rx
            .await
            .map_err(|_| MountError::Io("tcl interpreter did not respond".into()))
    }
}

#[async_trait]
impl Mount for TclMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let kind = match components {
            [] => TclFidKind::Root,
            ["eval"] => TclFidKind::Eval,
            ["vars"] => TclFidKind::VarsDir,
            ["vars", name] => TclFidKind::Var((*name).to_owned()),
            ["procs"] => TclFidKind::ProcsDir,
            ["procs", name] => TclFidKind::Proc((*name).to_owned()),
            _ => {
                return Err(MountError::NotFound(components.join("/")));
            }
        };
        Ok(Fid::new(TclFid {
            kind,
            write_buf: Vec::new(),
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
            .downcast_ref::<TclFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let data = match &inner.kind {
            TclFidKind::Eval => {
                // Ctl pattern: after a write, read returns the result.
                if inner.write_buf.is_empty() {
                    // No script written yet — return empty.
                    Vec::new()
                } else {
                    inner.write_buf.clone()
                }
            }
            TclFidKind::Var(name) => {
                let val: Option<String> = self.request(|resp| TclCommand::GetVar {
                    name: name.clone(),
                    resp,
                }).await?;
                val.map(|s| s.into_bytes())
                    .ok_or_else(|| MountError::NotFound(format!("vars/{name}")))?
            }
            TclFidKind::Proc(name) => {
                let body: Option<String> = self.request(|resp| TclCommand::GetProc {
                    name: name.clone(),
                    resp,
                }).await?;
                body.map(|s| s.into_bytes())
                    .ok_or_else(|| MountError::NotFound(format!("procs/{name}")))?
            }
            TclFidKind::Root | TclFidKind::VarsDir | TclFidKind::ProcsDir => {
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
            .downcast_ref::<TclFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match &inner.kind {
            TclFidKind::Eval => {
                let script = String::from_utf8_lossy(data).into_owned();
                let result = self.request(|resp| TclCommand::Eval {
                    script,
                    resp,
                }).await?;
                let response_bytes = match result {
                    Ok(s) => s.into_bytes(),
                    Err(e) => format!("error: {e}").into_bytes(),
                };
                // Store in the fid's write_buf for subsequent read.
                // SAFETY: single fid, interior mutability needed for ctl pattern.
                // The Mount trait takes &self, but we need to store the response.
                let fid_ptr = inner as *const TclFid as *mut TclFid;
                unsafe {
                    (*fid_ptr).write_buf = response_bytes;
                }
                Ok(data.len() as u32)
            }
            _ => Err(MountError::NotSupported("read-only".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<TclFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match &inner.kind {
            TclFidKind::Root => Ok(vec![
                DirEntry {
                    name: "eval".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "vars".into(),
                    is_dir: true,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "procs".into(),
                    is_dir: true,
                    size: 0,
                    stat: None,
                },
            ]),
            TclFidKind::VarsDir => {
                let names: Vec<String> =
                    self.request(|resp| TclCommand::ListVars { resp }).await?;
                Ok(names
                    .into_iter()
                    .map(|name| DirEntry {
                        name,
                        is_dir: false,
                        size: 0,
                        stat: None,
                    })
                    .collect())
            }
            TclFidKind::ProcsDir => {
                let names: Vec<String> =
                    self.request(|resp| TclCommand::ListProcs { resp }).await?;
                Ok(names
                    .into_iter()
                    .map(|name| DirEntry {
                        name,
                        is_dir: false,
                        size: 0,
                        stat: None,
                    })
                    .collect())
            }
            _ => Err(MountError::NotDirectory(format!("{:?}", inner.kind))),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<TclFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let (name, qtype) = match &inner.kind {
            TclFidKind::Root => ("tcl".to_owned(), 0x80), // QTDIR
            TclFidKind::Eval => ("eval".to_owned(), 0),
            TclFidKind::VarsDir => ("vars".to_owned(), 0x80),
            TclFidKind::Var(n) => (n.clone(), 0),
            TclFidKind::ProcsDir => ("procs".to_owned(), 0x80),
            TclFidKind::Proc(n) => (n.clone(), 0),
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

// ─────────────────────────────────────────────────────────────────────────────
// Channel constructor
// ─────────────────────────────────────────────────────────────────────────────

/// Create a bounded channel pair for the Tcl mount.
///
/// Returns `(sender, receiver)`. Pass the sender to [`TclMount::new`] and
/// poll the receiver from the thread owning the `TclShell`.
pub fn create_mount_channel() -> (tokio::sync::mpsc::Sender<TclCommand>, tokio::sync::mpsc::Receiver<TclCommand>) {
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
        mut rx: tokio::sync::mpsc::Receiver<TclCommand>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    TclCommand::Eval { script, resp } => {
                        if script == "error" {
                            let _ = resp.send(Err("test error".into()));
                        } else {
                            let _ = resp.send(Ok(format!("result: {script}")));
                        }
                    }
                    TclCommand::ListVars { resp } => {
                        let _ = resp.send(vec!["x".into(), "y".into()]);
                    }
                    TclCommand::GetVar { name, resp } => {
                        let val = match name.as_str() {
                            "x" => Some("42".into()),
                            "y" => Some("hello".into()),
                            _ => None,
                        };
                        let _ = resp.send(val);
                    }
                    TclCommand::ListProcs { resp } => {
                        let _ = resp.send(vec!["myproc".into()]);
                    }
                    TclCommand::GetProc { name, resp } => {
                        let body = match name.as_str() {
                            "myproc" => Some("puts hello".into()),
                            _ => None,
                        };
                        let _ = resp.send(body);
                    }
                }
            }
        })
    }

    #[tokio::test]
    async fn walk_root() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&[], &subject).await.unwrap();
        let inner = fid.downcast_ref::<TclFid>().unwrap();
        assert!(matches!(inner.kind, TclFidKind::Root));
    }

    #[tokio::test]
    async fn walk_eval() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["eval"], &subject).await.unwrap();
        let inner = fid.downcast_ref::<TclFid>().unwrap();
        assert!(matches!(inner.kind, TclFidKind::Eval));
    }

    #[tokio::test]
    async fn walk_not_found() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let result = mount.walk(&["nonexistent"], &subject).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn eval_write_then_read() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["eval"], &subject).await.unwrap();
        mount.open(&mut fid, 2, &subject).await.unwrap();

        // Write a script.
        let written = mount.write(&fid, 0, b"expr {2+3}", &subject).await.unwrap();
        assert_eq!(written, 10);

        // Read back the result.
        let data = mount.read(&fid, 0, 4096, &subject).await.unwrap();
        assert_eq!(String::from_utf8(data).unwrap(), "result: expr {2+3}");
    }

    #[tokio::test]
    async fn eval_error() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["eval"], &subject).await.unwrap();
        mount.open(&mut fid, 2, &subject).await.unwrap();
        mount.write(&fid, 0, b"error", &subject).await.unwrap();

        let data = mount.read(&fid, 0, 4096, &subject).await.unwrap();
        assert_eq!(String::from_utf8(data).unwrap(), "error: test error");
    }

    #[tokio::test]
    async fn readdir_root() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&[], &subject).await.unwrap();
        let entries = mount.readdir(&fid, &subject).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"eval"));
        assert!(names.contains(&"vars"));
        assert!(names.contains(&"procs"));
    }

    #[tokio::test]
    async fn readdir_vars() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["vars"], &subject).await.unwrap();
        let entries = mount.readdir(&fid, &subject).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));
    }

    #[tokio::test]
    async fn read_var() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["vars", "x"], &subject).await.unwrap();
        mount.open(&mut fid, 0, &subject).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject).await.unwrap();
        assert_eq!(String::from_utf8(data).unwrap(), "42");
    }

    #[tokio::test]
    async fn read_var_not_found() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["vars", "z"], &subject).await.unwrap();
        mount.open(&mut fid, 0, &subject).await.unwrap();
        let result = mount.read(&fid, 0, 4096, &subject).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn readdir_procs() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["procs"], &subject).await.unwrap();
        let entries = mount.readdir(&fid, &subject).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "myproc");
    }

    #[tokio::test]
    async fn read_proc() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let mut fid = mount.walk(&["procs", "myproc"], &subject).await.unwrap();
        mount.open(&mut fid, 0, &subject).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject).await.unwrap();
        assert_eq!(String::from_utf8(data).unwrap(), "puts hello");
    }

    #[tokio::test]
    async fn stat_directory() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["vars"], &subject).await.unwrap();
        let st = mount.stat(&fid, &subject).await.unwrap();
        assert_eq!(st.qtype, 0x80); // QTDIR
        assert_eq!(st.name, "vars");
    }

    #[tokio::test]
    async fn stat_file() {
        let (tx, rx) = create_mount_channel();
        let _h = spawn_fake_interp(rx);
        let mount = TclMount::new(tx);
        let subject = Subject::anonymous();

        let fid = mount.walk(&["eval"], &subject).await.unwrap();
        let st = mount.stat(&fid, &subject).await.unwrap();
        assert_eq!(st.qtype, 0);
        assert_eq!(st.name, "eval");
    }
}
