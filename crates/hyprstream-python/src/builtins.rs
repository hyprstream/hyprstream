//! VFS-aware Python builtins for the hyprstream Python shell.
//!
//! Injects `cat`, `ls`, `write`, `ctl`, `help`, `mount`, `json_parse` into
//! Python's builtins module and removes dangerous defaults (`open`, `exec`,
//! `compile`, `breakpoint`).
//!
//! Each function calls back into the VFS namespace via the thread-local
//! `SHELL_CTX` set up in `PythonShell::new()`.
//!
//! VFS calls are synchronous bridges over the async namespace:
//! - Native: the async op is `spawn`ed onto a dedicated bridge runtime and the
//!   interpreter thread blocks on a plain channel for the result. We must NOT
//!   use `Handle::current().block_on()` here — the interpreter owner loop drives
//!   the VM from inside its own tokio runtime, and re-entering a runtime on the
//!   same thread panics with "Cannot start a runtime from within a runtime".
//! - WASM32: VFS builtins return a Python RuntimeError (not supported).

use crate::SHELL_CTX;
use rustpython_vm::{
    builtins::PyStr,
    convert::IntoObject,
    function::FuncArgs,
    PyObjectRef, PyResult, VirtualMachine,
};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Remove dangerous builtins. Called after stdlib is loaded inside `interp.enter`.
///
/// NOTE (isolation is best-effort, not a hard security boundary):
/// Removing `__import__` blocks *fresh* `import foo` statements (they look up
/// `builtins.__import__`), but it does NOT block modules already cached in
/// `sys.modules` at startup, nor `importlib`-driven imports (importlib's
/// bootstrap machinery does not go through `builtins.__import__`).
///
/// The amount of isolation we promise is a security-posture decision still
/// pending review (see crate-level docs). Do not treat this list as a sandbox.
pub(crate) fn harden(vm: &VirtualMachine) {
    let dict = vm.builtins.dict();
    for name in &["open", "exec", "compile", "breakpoint", "__import__"] {
        let _ = dict.del_item(*name, vm);
    }
}

/// Register VFS builtins. Called from `Interpreter::with_init` before stdlib.
pub(crate) fn register_all(vm: &VirtualMachine) {
    let register = |name: &'static str, f: fn(FuncArgs, &VirtualMachine) -> PyResult<PyObjectRef>| {
        let func = vm.new_function(name, f);
        let _ = vm.builtins.set_attr(name, func, vm);
    };

    register("cat", builtin_cat);
    register("ls", builtin_ls);
    register("write", builtin_write);
    register("ctl", builtin_ctl);
    register("help", builtin_help);
    register("mount", builtin_mount);
    register("json_parse", builtin_json_parse);
}

// ─────────────────────────────────────────────────────────────────────────────
// VFS bridge helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Borrow the namespace + subject from the thread-local context.
fn with_ctx<F, T>(vm: &VirtualMachine, f: F) -> PyResult<T>
where
    F: FnOnce(std::sync::Arc<hyprstream_vfs::Namespace>, hyprstream_vfs::Subject) -> T,
{
    SHELL_CTX.with(|cell| {
        let borrow = cell.borrow();
        let ctx = borrow
            .as_ref()
            .ok_or_else(|| vm.new_runtime_error("no shell context".to_owned()))?;
        Ok(f(std::sync::Arc::clone(&ctx.namespace), ctx.subject.clone()))
    })
}

/// Dedicated multi-threaded runtime used only to drive VFS futures for the
/// synchronous Python builtins.
///
/// The interpreter owner loop runs the (`!Send`) VM from inside its own tokio
/// runtime, so we cannot `block_on`/`block_in_place` on the owner thread. This
/// runtime has its own worker thread that polls the spawned VFS future while the
/// owner thread blocks on a std channel for the result.
#[cfg(not(target_arch = "wasm32"))]
fn vfs_bridge_runtime() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("hyprstream-python-vfs")
            .enable_all()
            .build()
            .expect("failed to build hyprstream-python VFS bridge runtime")
    })
}

/// Run an async VFS operation from a synchronous Python builtin.
///
/// Clones the namespace handle + subject out of the thread-local on the
/// interpreter-owner thread, then runs the future on the dedicated bridge
/// runtime (see [`vfs_bridge_runtime`]) and blocks for the result. The VFS
/// namespace is `Send + Sync`, so the future can be moved to another thread.
#[cfg(not(target_arch = "wasm32"))]
fn vfs_async<F, Fut, T>(vm: &VirtualMachine, f: F) -> PyResult<T>
where
    F: FnOnce(std::sync::Arc<hyprstream_vfs::Namespace>, hyprstream_vfs::Subject) -> Fut,
    Fut: std::future::Future<Output = Result<T, String>> + Send + 'static,
    T: Send + 'static,
{
    let (ns, sub) = with_ctx(vm, |ns, sub| (ns, sub))?;
    let fut = f(ns, sub);

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    vfs_bridge_runtime().spawn(async move {
        // Receiver may be gone only if the owner thread unwound; ignore.
        let _ = tx.send(fut.await);
    });
    let result = rx
        .recv()
        .map_err(|_| vm.new_runtime_error("VFS bridge task did not complete".to_owned()))?;
    result.map_err(|e| vm.new_runtime_error(e))
}

#[cfg(target_arch = "wasm32")]
fn vfs_async<F, Fut, T>(vm: &VirtualMachine, _f: F) -> PyResult<T>
where
    F: FnOnce(std::sync::Arc<hyprstream_vfs::Namespace>, hyprstream_vfs::Subject) -> Fut,
    Fut: std::future::Future<Output = Result<T, String>>,
{
    Err(vm.new_runtime_error(
        "VFS builtins are not available in browser (wasm32) builds".to_owned(),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// cat
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_cat(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let paths: Vec<String> = args
        .args
        .iter()
        .map(|a| {
            a.downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("cat: expected str path".to_owned()))
                .map(|s| s.as_str().to_owned())
        })
        .collect::<PyResult<_>>()?;

    if paths.is_empty() {
        return Err(vm.new_type_error("cat: expected at least one path argument".to_owned()));
    }

    let out = vfs_async(vm, move |ns, sub| async move {
        let mut result = String::new();
        for path in &paths {
            match ns.cat(path, &sub).await {
                Ok(bytes) => result.push_str(&String::from_utf8_lossy(&bytes)),
                Err(e) => return Err(format!("cat: {path}: {e}")),
            }
        }
        Ok(result)
    })?;

    Ok(vm.ctx.new_str(out).into_object())
}

// ─────────────────────────────────────────────────────────────────────────────
// ls
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_ls(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let path = match args.args.first() {
        Some(a) => a
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("ls: expected str path".to_owned()))?
            .as_str()
            .to_owned(),
        None => "/".to_owned(),
    };

    let out = vfs_async(vm, move |ns, sub| async move {
        ns.ls(&path, &sub)
            .await
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|e| if e.is_dir { format!("{}/", e.name) } else { e.name })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .map_err(|e| format!("ls: {path}: {e}"))
    })?;

    Ok(vm.ctx.new_str(out).into_object())
}

// ─────────────────────────────────────────────────────────────────────────────
// write
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_write(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    if args.args.len() < 2 {
        return Err(vm.new_type_error("write: expected (path, data)".to_owned()));
    }
    let path = args.args[0]
        .downcast_ref::<PyStr>()
        .ok_or_else(|| vm.new_type_error("write: path must be str".to_owned()))?
        .as_str()
        .to_owned();
    let data = args.args[1]
        .downcast_ref::<PyStr>()
        .ok_or_else(|| vm.new_type_error("write: data must be str".to_owned()))?
        .as_str()
        .as_bytes()
        .to_vec();

    vfs_async(vm, move |ns, sub| async move {
        ns.echo(&path, &data, &sub)
            .await
            .map_err(|e| format!("write: {path}: {e}"))
    })?;

    Ok(vm.ctx.none())
}

// ─────────────────────────────────────────────────────────────────────────────
// ctl
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_ctl(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    if args.args.len() < 2 {
        return Err(vm.new_type_error("ctl: expected (path, command)".to_owned()));
    }
    let path = args.args[0]
        .downcast_ref::<PyStr>()
        .ok_or_else(|| vm.new_type_error("ctl: path must be str".to_owned()))?
        .as_str()
        .to_owned();
    let cmd = args.args[1]
        .downcast_ref::<PyStr>()
        .ok_or_else(|| vm.new_type_error("ctl: command must be str".to_owned()))?
        .as_str()
        .as_bytes()
        .to_vec();

    let out = vfs_async(vm, move |ns, sub| async move {
        ns.ctl(&path, &cmd, &sub)
            .await
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|e| format!("ctl: {path}: {e}"))
    })?;

    Ok(vm.ctx.new_str(out).into_object())
}

// ─────────────────────────────────────────────────────────────────────────────
// mount
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_mount(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let specific_path = args
        .args
        .first()
        .map(|a| {
            a.downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("mount: expected str path".to_owned()))
                .map(|s| s.as_str().to_owned())
        })
        .transpose()?;

    let result = with_ctx(vm, |ns, _sub| {
        let prefixes: Vec<String> = ns.mount_prefixes().into_iter().map(str::to_owned).collect();
        if let Some(path) = specific_path {
            if prefixes.iter().any(|p| p == &path) {
                Ok(format!("mounted: {path}"))
            } else {
                Err(format!("mount: {path}: not mounted"))
            }
        } else {
            Ok(prefixes.join("\n"))
        }
    })?;
    let out = result.map_err(|e| vm.new_runtime_error(e))?;

    Ok(vm.ctx.new_str(out).into_object())
}

// ─────────────────────────────────────────────────────────────────────────────
// help
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_help(_args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let text = concat!(
        "hyprstream Python shell builtins:\n",
        "  cat(path, ...)    — read VFS file(s)\n",
        "  ls(path='/')      — list VFS directory\n",
        "  write(path, data) — write string to VFS file\n",
        "  ctl(path, cmd)    — send command to ctl file, read response\n",
        "  mount(path=None)  — list mount points or verify one\n",
        "  json_parse(text)  — parse JSON into Python objects\n",
        "  help()            — this message",
    );
    Ok(vm.ctx.new_str(text).into_object())
}

// ─────────────────────────────────────────────────────────────────────────────
// json_parse
// ─────────────────────────────────────────────────────────────────────────────

fn builtin_json_parse(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let text = args
        .args
        .first()
        .ok_or_else(|| vm.new_type_error("json_parse: expected one str argument".to_owned()))?
        .downcast_ref::<PyStr>()
        .ok_or_else(|| vm.new_type_error("json_parse: argument must be str".to_owned()))?
        .as_str()
        .to_owned();

    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| vm.new_value_error(format!("invalid JSON: {e}")))?;

    json_to_py(vm, &value)
}

fn json_to_py(vm: &VirtualMachine, value: &serde_json::Value) -> PyResult<PyObjectRef> {
    use serde_json::Value;
    match value {
        Value::Null => Ok(vm.ctx.none()),
        Value::Bool(b) => Ok(vm.ctx.new_bool(*b).into_object()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(vm.ctx.new_int(i).into_object())
            } else if let Some(f) = n.as_f64() {
                Ok(vm.ctx.new_float(f).into_object())
            } else {
                Ok(vm.ctx.new_str(n.to_string()).into_object())
            }
        }
        Value::String(s) => Ok(vm.ctx.new_str(s.clone()).into_object()),
        Value::Array(arr) => {
            let items: PyResult<Vec<PyObjectRef>> = arr.iter().map(|v| json_to_py(vm, v)).collect();
            Ok(vm.ctx.new_list(items?).into_object())
        }
        Value::Object(map) => {
            let dict = vm.ctx.new_dict();
            for (k, v) in map {
                let val = json_to_py(vm, v)?;
                dict.set_item(k.as_str(), val, vm)?;
            }
            Ok(dict.into_object())
        }
    }
}
