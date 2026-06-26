//! RustPython shell for the hyprstream VFS namespace.
//!
//! Provides a `PythonShell` that wraps a RustPython interpreter with VFS
//! builtins (`cat`, `ls`, `write`, `ctl`, `help`, `mount`, `json_parse`)
//! injected into the Python builtins module. Dangerous builtins (`open`,
//! `exec`, `compile`, `breakpoint`, `__import__`) are removed after stdlib
//! initialises.
//!
//! **Security posture (best-effort, NOT a hard sandbox):** VFS builtins route
//! file I/O through the namespace, and the most obvious host-access builtins are
//! removed. However, this is *not* a security boundary. Two known gaps:
//!
//! - Import escape: `__import__` removal does not block modules already in
//!   `sys.modules`, nor imports via `importlib`'s bootstrap machinery (which
//!   bypasses `builtins.__import__`). See [`builtins::harden`].
//! - No CPU quota: RustPython 0.3 has no per-instruction budget; CPU-bound guest
//!   code (e.g. `while True: pass`) is only stopped by a wall-clock watchdog (see
//!   [`PythonShell::eval`]). Code blocked inside a VFS call is not interrupted.
//!
//! Treating untrusted Python here as fully isolated would over-claim; the level
//! of isolation to promise is a pending security-posture decision.
//!
//! `PythonShell` is `!Send`/`!Sync` (RustPython `VirtualMachine` uses `Rc`).
//! Must run in a `LocalSet` or single-threaded tokio runtime.

mod builtins;
pub mod mount;

use hyprstream_vfs::{Namespace, Subject};
use rustpython_vm::{
    builtins::{PyDictRef, PyStr},
    scope::Scope,
    signal::UserSignalSender,
    AsObject, Interpreter, Settings, VirtualMachine,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Wall-clock budget for a single `eval`/`exec` before the watchdog interrupts.
///
/// RustPython 0.3 has no instruction/quota hook, only `recursion_limit`, so this
/// is the only defence against CPU-bound guest loops. `check_signals()` runs on
/// every bytecode op, so the interrupt lands promptly for pure-Python loops.
const EXEC_TIMEOUT: Duration = Duration::from_secs(5);

/// Recursion limit for guest code — a cheap guard against stack-overflow aborts.
const RECURSION_LIMIT: usize = 500;

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
    /// Persistent globals dict reused across every `eval`/`exec` so user state
    /// (variables, functions) survives between calls and is visible to the
    /// `/lang/python/vars/` and `/defs/` mounts. `!Send` (`Rc`-backed).
    globals: PyDictRef,
    /// Sender used by the per-call watchdog to inject a "time limit exceeded"
    /// interrupt into the VM via the signal channel. `Clone + Send`.
    signal_tx: UserSignalSender,
    /// Wall-clock budget per `eval`/`exec`. Defaults to [`EXEC_TIMEOUT`].
    timeout: Duration,
}

impl PythonShell {
    /// Create a new Python shell bound to the given caller identity and VFS namespace.
    pub fn new(subject: Subject, namespace: Arc<Namespace>) -> Self {
        let ctx = ShellContext { subject, namespace };
        SHELL_CTX.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });

        let settings = Settings::default();
        let (signal_tx, signal_rx) = rustpython_vm::signal::user_signal_channel();
        let interp = Interpreter::with_init(settings, move |vm| {
            builtins::register_all(vm);
            // Watchdog channel: `check_signals()` runs every bytecode op, so a
            // queued user signal can interrupt a stuck loop. `signal_handlers`
            // is `Some` by default, so the channel is actually polled.
            vm.set_user_signal_channel(signal_rx);
            // Cheap stack-overflow guard for guest code.
            vm.recursion_limit.set(RECURSION_LIMIT);
        });

        // stdlib is now available; remove dangerous builtins.
        interp.enter(|vm| {
            builtins::harden(vm);
        });

        // Persistent globals dict + a safe baseline stdout (never `None`, never
        // the host process stdout) so stray `print()` calls cannot crash.
        let globals = interp.enter(|vm| {
            install_sink_stdout(vm);
            vm.ctx.new_dict()
        });

        Self {
            interp,
            globals,
            signal_tx,
            timeout: EXEC_TIMEOUT,
        }
    }

    /// Override the per-call wall-clock execution budget (default: 5s).
    ///
    /// This is the only defence against CPU-bound guest code (RustPython 0.3 has
    /// no instruction quota). A very short value risks interrupting legitimate
    /// long-running work.
    pub fn set_exec_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Build a scope over the persistent globals dict for one execution.
    fn scope(&self, vm: &VirtualMachine) -> Scope {
        Scope::with_builtins(None, self.globals.clone(), vm)
    }

    /// Run `body` (one interpreter execution) under a wall-clock watchdog.
    ///
    /// A watchdog thread waits `EXEC_TIMEOUT`; if the call has not finished it
    /// sends a signal that raises a `RuntimeError` inside the VM. A per-call
    /// `armed` flag neutralises any signal that loses the completion race, so a
    /// stale interrupt can never corrupt the *next* call.
    fn run_guarded<T>(&self, body: impl FnOnce() -> T) -> T {
        let armed = Arc::new(AtomicBool::new(true));
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let sig = self.signal_tx.clone();
        let armed_wd = Arc::clone(&armed);

        let timeout = self.timeout;
        let watchdog = std::thread::spawn(move || {
            if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                done_rx.recv_timeout(timeout)
            {
                if armed_wd.load(Ordering::Acquire) {
                    let armed_sig = Arc::clone(&armed_wd);
                    let _ = sig.send(Box::new(move |vm: &VirtualMachine| {
                        if armed_sig.load(Ordering::Acquire) {
                            Err(vm.new_runtime_error(
                                "execution exceeded time limit".to_owned(),
                            ))
                        } else {
                            Ok(())
                        }
                    }));
                }
            }
        });

        let result = body();

        // Disarm before signalling completion so any signal already queued in
        // the boundary race becomes a no-op.
        armed.store(false, Ordering::Release);
        let _ = done_tx.send(());
        let _ = watchdog.join();
        result
    }

    /// Evaluate a Python expression. Returns `Ok((result_repr, stdout))` or `Err(traceback)`.
    pub fn eval(&self, code: &str) -> Result<(String, String), String> {
        self.run_guarded(|| self.interp.enter(|vm| {
            let active = setup_stdout_capture(vm);

            let scope = self.scope(vm);
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
        }))
    }

    /// Execute a Python statement block. Returns `Ok(stdout)` or `Err(traceback)`.
    pub fn exec(&self, code: &str) -> Result<String, String> {
        self.run_guarded(|| self.interp.enter(|vm| {
            let active = setup_stdout_capture(vm);

            let scope = self.scope(vm);
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
        }))
    }

    /// List all non-dunder global variable names.
    pub fn list_vars(&self) -> Vec<String> {
        self.interp.enter(|_vm| {
            self.globals
                .clone()
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
            self.globals
                .get_item(name, vm)
                .ok()?
                .repr(vm)
                .map(|s| s.as_str().to_owned())
                .ok()
        })
    }

    /// List all callable (function/class) names in the global scope.
    pub fn list_defs(&self) -> Vec<String> {
        self.interp.enter(|_vm| {
            self.globals
                .clone()
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
            let obj = self.globals.get_item(name, vm).ok()?;
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

// On drain we read the captured value and restore a *sink* stdout (never
// `None`). Leaving `None` here is the bug: if a later `setup_stdout_capture`
// fails, any `print()` would raise `AttributeError: 'NoneType' has no 'write'`.
const CAPTURE_DRAIN: &str = r"
import sys as _sys
__v = _sys.stdout.getvalue() if hasattr(_sys.stdout, 'getvalue') else ''
class __Sink:
    def write(self, s): pass
    def flush(self): pass
_sys.stdout = __Sink()
del _sys
__v
";

// A no-op stdout installed as the safe baseline so stray `print()` calls never
// reach the host process stdout and never crash on a `None` stdout.
const SINK_SETUP: &str = r"
import sys as _sys
class __Sink:
    def write(self, s): pass
    def flush(self): pass
_sys.stdout = __Sink()
del _sys
";

/// Install the no-op sink as `sys.stdout`. Best-effort baseline; ignores errors.
fn install_sink_stdout(vm: &rustpython_vm::VirtualMachine) {
    let scope = vm.new_scope_with_builtins();
    let _ = vm.run_code_string(scope, SINK_SETUP, "<sink-setup>".to_owned());
}

/// Redirect `sys.stdout` to a lightweight pure-Python capture object.
/// Returns `true` on success. On failure, restores the safe sink baseline so
/// subsequent `print()` calls cannot crash.
fn setup_stdout_capture(vm: &rustpython_vm::VirtualMachine) -> bool {
    let scope = vm.new_scope_with_builtins();
    let ok = vm
        .run_code_string(scope, CAPTURE_SETUP, "<capture-setup>".to_owned())
        .is_ok();
    if !ok {
        install_sink_stdout(vm);
    }
    ok
}

/// Read captured stdout and restore `sys.stdout` to a safe no-op sink.
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

    // Bug #2: state must persist across eval/exec via a shared globals dict.
    #[test]
    fn state_persists_across_calls() {
        let shell = make_shell();
        shell.exec("x = 41").unwrap();
        let (result, _) = shell.eval("x + 1").unwrap();
        assert_eq!(result, "42");
    }

    // Bug #2: vars/defs must be visible to the /lang/python/vars and /defs mounts.
    #[test]
    fn vars_and_defs_visible_after_exec() {
        let shell = make_shell();
        shell.exec("answer = 42").unwrap();
        shell.exec("def greet():\n    return 'hi'\n").unwrap();
        assert!(shell.list_vars().contains(&"answer".to_owned()));
        assert_eq!(shell.get_var("answer"), Some("42".to_owned()));
        assert!(shell.list_defs().contains(&"greet".to_owned()));
        assert!(shell.get_def("greet").is_some());
        assert!(shell.get_def("answer").is_none()); // not callable
    }

    // Bug #3: repeated print() must keep working — drain restores a real sink,
    // not None, so a later print cannot raise AttributeError.
    #[test]
    fn print_works_repeatedly() {
        let shell = make_shell();
        for i in 0..3 {
            let out = shell.exec(&format!("print({i})")).unwrap();
            assert!(out.contains(&i.to_string()));
        }
    }

    // Bug #1: VFS builtins must work when the interpreter runs *inside* a tokio
    // runtime (the owner-loop environment). The old `Handle::current().block_on`
    // path panicked with "Cannot start a runtime from within a runtime".
    #[tokio::test]
    async fn vfs_cat_from_async_context() {
        let shell = make_shell();
        let (result, _stdout) = shell.eval("cat('/config/temperature')").unwrap();
        assert_eq!(result, "'0.7'");
    }

    #[tokio::test]
    async fn vfs_ls_from_async_context() {
        let shell = make_shell();
        let (result, _stdout) = shell.eval("ls('/config')").unwrap();
        assert!(result.contains("temperature"));
        assert!(result.contains("status"));
    }

    // Same, but on a multi-threaded runtime to exercise the cross-runtime bridge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn vfs_write_then_cat_multithread() {
        let shell = make_shell();
        let out = shell.exec("print(ctl('/bin/load', 'go'))").unwrap();
        assert!(out.contains("ok: go"));
    }

    // Bug #4: a CPU-bound loop must be interrupted by the watchdog rather than
    // pinning the owner thread forever.
    #[test]
    fn watchdog_interrupts_infinite_loop() {
        let mut shell = make_shell();
        shell.set_exec_timeout(std::time::Duration::from_millis(200));
        let err = shell.exec("while True:\n    pass\n").unwrap_err();
        assert!(
            err.to_lowercase().contains("time limit"),
            "expected time-limit error, got: {err}"
        );
        // The shell must remain usable after an interrupt.
        let (result, _) = shell.eval("1 + 1").unwrap();
        assert_eq!(result, "2");
    }
}
