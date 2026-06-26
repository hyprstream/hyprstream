//! RustPython shell for the hyprstream VFS namespace.
//!
//! Provides a `PythonShell` that wraps a RustPython interpreter with VFS
//! builtins (`cat`, `ls`, `write`, `ctl`, `help`, `mount`, `json_parse`)
//! injected into the Python builtins module. Dangerous builtins (`open`,
//! `exec`, `compile`, `breakpoint`) are removed after stdlib initialises.
//!
//! **Security**: All I/O goes through the VFS — no direct host filesystem access.
//!
//! `PythonShell` is `!Send`/`!Sync` (RustPython `VirtualMachine` uses `Rc`).
//! Must run in a `LocalSet` or single-threaded tokio runtime.

mod builtins;
pub mod mount;

use hyprstream_vfs::{Namespace, Subject};
use rustpython_vm::{
    builtins::PyStr,
    AsObject,
    Interpreter, Settings,
};
use std::sync::Arc;

pub use mount::{create_mount_channel, PyCommand, PythonMount};

/// Context carried to VFS builtins via a thread-local.
///
/// RustPython native functions take only `&VirtualMachine`; the namespace
/// and subject reach them through this thread-local rather than closure capture.
pub(crate) struct ShellContext {
    pub subject: Subject,
    pub namespace: Arc<Namespace>,
}

thread_local! {
    pub(crate) static SHELL_CTX: std::cell::RefCell<Option<ShellContext>> =
        const { std::cell::RefCell::new(None) };
}

/// A Python shell backed by the hyprstream VFS namespace.
///
/// Contains a RustPython `Interpreter` which is `!Send` (uses `Rc` internally).
/// Must be constructed and used on the same thread.
pub struct PythonShell {
    interp: Interpreter,
}

impl PythonShell {
    /// Create a new Python shell bound to the given caller identity and VFS namespace.
    pub fn new(subject: Subject, namespace: Arc<Namespace>) -> Self {
        let ctx = ShellContext { subject, namespace };
        SHELL_CTX.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });

        let settings = Settings::default();
        let interp = Interpreter::with_init(settings, |vm| {
            builtins::register_all(vm);
        });

        // stdlib is now available; remove dangerous builtins.
        interp.enter(|vm| {
            builtins::harden(vm);
        });

        Self { interp }
    }

    /// Evaluate a Python expression. Returns `Ok((result_repr, stdout))` or `Err(traceback)`.
    pub fn eval(&self, code: &str) -> Result<(String, String), String> {
        self.interp.enter(|vm| {
            let active = setup_stdout_capture(vm);

            let scope = vm.new_scope_with_builtins();
            let run_result = vm.run_block_expr(scope, code);

            let stdout = drain_stdout_capture(vm, active);

            match run_result {
                Ok(obj) => {
                    let repr = if vm.is_none(&obj) {
                        String::new()
                    } else {
                        obj.repr(vm)
                            .map(|s| s.as_str().to_owned())
                            .unwrap_or_else(|_| "<repr error>".to_owned())
                    };
                    Ok((repr, stdout))
                }
                Err(exc) => {
                    let msg = exc
                        .as_object()
                        .to_owned()
                        .str(vm)
                        .map(|s| s.as_str().to_owned())
                        .unwrap_or_else(|_| "<exception>".to_owned());
                    Err(msg)
                }
            }
        })
    }

    /// Execute a Python statement block. Returns `Ok(stdout)` or `Err(traceback)`.
    pub fn exec(&self, code: &str) -> Result<String, String> {
        self.interp.enter(|vm| {
            let active = setup_stdout_capture(vm);

            let scope = vm.new_scope_with_builtins();
            let run_result = vm.run_code_string(scope, code, "<hyprstream>".to_owned());

            let stdout = drain_stdout_capture(vm, active);

            match run_result {
                Ok(_) => Ok(stdout),
                Err(exc) => {
                    let msg = exc
                        .as_object()
                        .to_owned()
                        .str(vm)
                        .map(|s| s.as_str().to_owned())
                        .unwrap_or_else(|_| "<exception>".to_owned());
                    Err(msg)
                }
            }
        })
    }

    /// List all non-dunder global variable names.
    pub fn list_vars(&self) -> Vec<String> {
        self.interp.enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            scope
                .globals
                .into_iter()
                .filter_map(|(k, _v)| {
                    let name = k.downcast_ref::<PyStr>()?.as_str().to_owned();
                    if name.starts_with("__") && name.ends_with("__") {
                        return None;
                    }
                    Some(name)
                })
                .collect()
        })
    }

    /// Get the repr of a global variable by name, or `None` if not found.
    pub fn get_var(&self, name: &str) -> Option<String> {
        self.interp.enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            scope
                .globals
                .get_item(name, vm)
                .ok()?
                .repr(vm)
                .map(|s| s.as_str().to_owned())
                .ok()
        })
    }

    /// List all callable (function/class) names in the global scope.
    pub fn list_defs(&self) -> Vec<String> {
        self.interp.enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            scope
                .globals
                .into_iter()
                .filter_map(|(k, v)| {
                    let name = k.downcast_ref::<PyStr>()?.as_str().to_owned();
                    if name.starts_with("__") && name.ends_with("__") {
                        return None;
                    }
                    if v.is_callable() { Some(name) } else { None }
                })
                .collect()
        })
    }

    /// Get the repr of a callable definition by name, or `None` if not found/not callable.
    pub fn get_def(&self, name: &str) -> Option<String> {
        self.interp.enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            let obj = scope.globals.get_item(name, vm).ok()?;
            if !obj.is_callable() {
                return None;
            }
            obj.repr(vm).map(|s| s.as_str().to_owned()).ok()
        })
    }

    /// Process one command from the mount channel.
    pub fn process_command(&self, cmd: PyCommand) {
        match cmd {
            PyCommand::Eval { code, resp } => {
                let _ = resp.send(self.eval(&code));
            }
            PyCommand::Exec { code, resp } => {
                let _ = resp.send(self.exec(&code));
            }
            PyCommand::ListVars { resp } => {
                let _ = resp.send(self.list_vars());
            }
            PyCommand::GetVar { name, resp } => {
                let _ = resp.send(self.get_var(&name));
            }
            PyCommand::ListDefs { resp } => {
                let _ = resp.send(self.list_defs());
            }
            PyCommand::GetDef { name, resp } => {
                let _ = resp.send(self.get_def(&name));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stdout capture helpers
// ─────────────────────────────────────────────────────────────────────────────

// Python snippet: a minimal write-capturing stdout surrogate.
// Avoids `io.StringIO` (needs encodings) and uses a pure class.
const CAPTURE_SETUP: &str = r"
import sys as _sys
class __Capture:
    def __init__(self): self.__buf = []
    def write(self, s): self.__buf.append(str(s))
    def flush(self): pass
    def getvalue(self): return ''.join(self.__buf)
_sys.stdout = __Capture()
del _sys
";

const CAPTURE_DRAIN: &str = r"
import sys as _sys
__v = _sys.stdout.getvalue() if hasattr(_sys.stdout, 'getvalue') else ''
_sys.stdout = None
del _sys
__v
";

/// Redirect `sys.stdout` to a lightweight pure-Python capture object.
/// Returns `true` on success.
fn setup_stdout_capture(vm: &rustpython_vm::VirtualMachine) -> bool {
    let scope = vm.new_scope_with_builtins();
    vm.run_code_string(scope, CAPTURE_SETUP, "<capture-setup>".to_owned())
        .is_ok()
}

/// Read captured stdout and restore `sys.stdout` to `None`.
fn drain_stdout_capture(vm: &rustpython_vm::VirtualMachine, active: bool) -> String {
    if !active {
        return String::new();
    }
    let scope = vm.new_scope_with_builtins();
    vm.run_block_expr(scope, CAPTURE_DRAIN)
        .ok()
        .and_then(|obj| obj.str(vm).ok().map(|s| s.as_str().to_owned()))
        .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_types)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Namespace, Stat};
    use std::collections::HashMap;

    struct MemMount {
        files: HashMap<String, Vec<u8>>,
    }

    impl MemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self {
                files: files.into_iter().map(|(k, v)| (k.to_owned(), v.to_vec())).collect(),
            }
        }
    }

    struct MemFid {
        path: String,
        write_buf: parking_lot::Mutex<Vec<u8>>,
    }

    #[async_trait]
    impl Mount for MemMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            let exists = self.files.contains_key(&path)
                || self.files.keys().any(|k| k.starts_with(&format!("{path}/")));
            if !exists && !path.is_empty() {
                return Err(MountError::NotFound(path));
            }
            Ok(Fid::new(MemFid { path, write_buf: parking_lot::Mutex::new(Vec::new()) }))
        }
        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let wb = inner.write_buf.lock();
            if !wb.is_empty() {
                let start = offset as usize;
                return Ok(if start >= wb.len() { vec![] } else { wb[start..].to_vec() });
            }
            drop(wb);
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    Ok(if start >= data.len() { vec![] } else { data[start..].to_vec() })
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }
        async fn write(&self, fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            *inner.write_buf.lock() =
                format!("ok: {}", String::from_utf8_lossy(data)).into_bytes();
            Ok(data.len() as u32)
        }
        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() { String::new() } else { format!("{}/", inner.path) };
            let mut entries = Vec::new();
            for key in self.files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry { name: rest.to_owned(), is_dir: false, size: 0, stat: None });
                    }
                }
            }
            Ok(entries)
        }
        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn make_namespace() -> Arc<Namespace> {
        let mut ns = Namespace::new();
        let config = Arc::new(MemMount::new(vec![
            ("temperature", b"0.7"),
            ("status", b"loaded"),
        ]));
        ns.mount("/config", config).unwrap();
        let bin = Arc::new(MemMount::new(vec![("load", b"")]));
        ns.mount("/bin", bin).unwrap();
        Arc::new(ns)
    }

    fn make_shell() -> PythonShell {
        PythonShell::new(Subject::new("test"), make_namespace())
    }

    #[test]
    fn eval_expr() {
        let shell = make_shell();
        let (result, _stdout) = shell.eval("2 + 3").unwrap();
        assert_eq!(result, "5");
    }

    #[test]
    fn eval_str_concat() {
        let shell = make_shell();
        let (result, _stdout) = shell.eval("'hello' + ' world'").unwrap();
        assert_eq!(result, "'hello world'");
    }

    #[test]
    fn exec_captures_print() {
        let shell = make_shell();
        let stdout = shell.exec("print('hello world')").unwrap();
        assert!(stdout.contains("hello world"));
    }

    #[test]
    fn eval_error_zero_division() {
        let shell = make_shell();
        let err = shell.eval("1 / 0").unwrap_err();
        assert!(err.to_lowercase().contains("division") || err.contains("ZeroDivisionError"));
    }

    #[test]
    fn help_in_builtins() {
        let shell = make_shell();
        let (result, _) = shell.eval("help()").unwrap();
        // help() prints and returns None; result may be empty but stdout has the text
        let stdout = shell.exec("help()").unwrap();
        assert!(stdout.contains("cat") || result.is_empty() || result.contains("cat"));
    }

    #[test]
    fn dangerous_builtins_removed() {
        let shell = make_shell();
        let err = shell.eval("open('/etc/passwd')").unwrap_err();
        assert!(err.contains("NameError") || err.contains("open"));
    }
}
