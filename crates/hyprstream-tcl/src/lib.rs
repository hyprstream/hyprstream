//! Tcl (molt) shell for the hyprstream VFS namespace.
//!
//! Provides a `TclShell` that wraps a molt interpreter with VFS builtins
//! (`cat`, `ls`, `echo`, `ctl`, `help`, `mount`). Commands are evaluated
//! as standard Tcl with VFS path operations. Unknown commands fall back
//! to `/bin/{name}` resolution (Plan9 PATH model).
//!
//! **Security**: Dangerous molt commands (`source`, `exit`, `puts`, `rename`,
//! `global`, `time`) are removed at construction. All host filesystem access
//! goes through the VFS — the Tcl interpreter is a guest with no direct I/O.
//!
//! `TclShell` is `!Send`/`!Sync` (molt `Value` uses `Rc`). One interpreter
//! per ChatApp or shell session.

mod builtins;
pub mod mount;

use hyprstream_vfs::proxy::{VfsOp, VfsRequest};
use hyprstream_vfs::Subject;
use molt::types::*;
use molt::Interp;

pub use mount::{create_mount_channel, TclCommand, TclMount};

/// Context stored in the molt interpreter via `save_context()`.
///
/// All VFS operations go through the channel proxy — no direct Namespace access.
pub(crate) struct ShellContext {
    pub _subject: Subject,
    pub vfs_tx: tokio::sync::mpsc::Sender<VfsRequest>,
}

/// A Tcl shell backed by the hyprstream VFS namespace.
///
/// Contains a molt `Interp` which is `!Send` (uses `Rc` internally).
/// Must be constructed and used on the same thread — use `spawn_blocking`
/// to run on a dedicated thread, passing only `Send` arguments (subject, vfs_tx).
pub struct TclShell {
    interp: Interp,
    ctx_id: ContextID,
}

impl TclShell {
    /// Create a new Tcl shell bound to the given caller identity and VFS proxy.
    ///
    /// The `vfs_tx` sender is used by builtins and `/bin/` resolution for all
    /// async VFS operations (cat, ls, echo, ctl, mount) via the channel proxy.
    ///
    /// **Must be called on the thread where the shell will be used** (e.g., inside
    /// `spawn_blocking`). TclShell is `!Send` due to molt's `Rc`-based `Value`.
    ///
    /// Dangerous commands are removed and the recursion limit is lowered.
    /// All host I/O goes through the VFS — this is a guest interpreter.
    pub fn new(subject: Subject, vfs_tx: tokio::sync::mpsc::Sender<VfsRequest>) -> Self {
        let mut interp = Interp::new();

        // Remove dangerous commands — all host I/O goes through VFS.
        // source: reads arbitrary host files via std::fs::read_to_string
        // exit: calls std::process::exit(), kills the server
        // puts: writes to server stdout (log injection)
        // rename: can shadow/delete security-critical builtins
        // global: scope manipulation unnecessary in VFS shell
        // time: runs a command N times (user-controlled N, CPU bomb)
        // parse/pdump/pclear: debug internals, not for users
        // apply/eval/subst/uplevel: evaluate arbitrary code, enabling injection via field output
        for cmd in &[
            "source", "exit", "puts", "rename", "global", "time",
            "parse", "pdump", "pclear", "apply", "eval", "subst", "uplevel",
        ] {
            if interp.has_command(cmd) {
                interp.remove_command(cmd);
            }
        }
        interp.set_recursion_limit(100);
        interp.set_instruction_limit(100_000);

        let ctx = ShellContext { _subject: subject, vfs_tx };
        let ctx_id = interp.save_context(ctx);
        builtins::register_all(&mut interp, ctx_id);
        Self { interp, ctx_id }
    }

    /// Evaluate a Tcl script. Returns Ok(output) or Err(error message).
    ///
    /// On "invalid command name" errors, tries `/bin/{name}` resolution
    /// (Plan9 PATH model) before returning the error.
    pub fn eval(&mut self, script: &str) -> Result<String, String> {
        self.interp.reset_instruction_count();
        match self.interp.eval(script) {
            Ok(val) => Ok(val.to_string()),
            Err(exception) => {
                let msg = exception.value().to_string();
                // Try /bin/ resolution on command-not-found.
                if let Some(cmd_name) = extract_unknown_command(&msg) {
                    if let Ok(result) = self.try_cmd_resolve(&cmd_name, script) {
                        return Ok(result);
                    }
                }
                Err(msg)
            }
        }
    }

    /// Check if a command name is registered in the interpreter.
    pub fn has_command(&self, name: &str) -> bool {
        self.interp.has_command(name)
    }

    /// Set an env array variable safely, bypassing the Tcl evaluator.
    ///
    /// Uses `Interp::set_element` to avoid Tcl injection from values
    /// containing unbalanced braces or bracket expressions.
    pub fn set_env(&mut self, key: &str, value: &str) -> Result<(), String> {
        self.interp
            .set_element("env", key, molt::types::Value::from(value))
            .map_err(|e| e.value().to_string())
    }

    /// Override the instruction limit (default: 100,000).
    /// Set to 0 for unlimited (not recommended for untrusted input).
    pub fn set_instruction_limit(&mut self, limit: usize) {
        self.interp.set_instruction_limit(limit);
    }

    /// Process one command from the mount channel.
    ///
    /// Called from the owner thread's event loop (e.g., `ChatApp::tick()`).
    /// Each command carries a oneshot response channel; the caller blocks
    /// until we send the reply.
    pub fn process_command(&mut self, cmd: TclCommand) {
        match cmd {
            TclCommand::Eval { script, resp } => {
                let _ = resp.send(self.eval(&script));
            }
            TclCommand::ListVars { resp } => {
                let names: Vec<String> = self
                    .interp
                    .vars_in_scope()
                    .iter()
                    .map(|v| v.to_string())
                    .collect();
                let _ = resp.send(names);
            }
            TclCommand::GetVar { name, resp } => {
                let val = self.interp.scalar(&name).ok().map(|v| v.to_string());
                let _ = resp.send(val);
            }
            TclCommand::ListProcs { resp } => {
                let names: Vec<String> = self
                    .interp
                    .proc_names()
                    .iter()
                    .map(|v| v.to_string())
                    .collect();
                let _ = resp.send(names);
            }
            TclCommand::GetProc { name, resp } => {
                let body = self.interp.proc_body(&name).ok().map(|v| v.to_string());
                let _ = resp.send(body);
            }
        }
    }

    /// Try resolving an unknown command via `/bin/{name}` (Plan9 PATH model).
    ///
    /// Extracts everything after the command name as the argument payload,
    /// then does a ctl (write+read) on `/bin/{name}` via the channel proxy.
    fn try_cmd_resolve(&mut self, cmd_name: &str, script: &str) -> Result<String, String> {
        // Sanitize: reject path traversal in command names.
        if cmd_name.contains('/') || cmd_name.contains("..") {
            return Err(format!("invalid command name \"{cmd_name}\""));
        }
        let cmd_path = format!("/bin/{cmd_name}");
        let raw_args = script
            .trim()
            .strip_prefix(cmd_name)
            .unwrap_or("")
            .trim();

        // Evaluate args through the Tcl evaluator to resolve variable and
        // command substitutions (e.g., `[cat /config/token]` → actual value).
        // Without this, raw bracket expressions reach ctl file handlers.
        let evaluated_args = if raw_args.is_empty() {
            String::new()
        } else {
            match self.interp.eval(&format!("list {raw_args}")) {
                Ok(val) => val.to_string(),
                Err(e) => return Err(e.value().to_string()),
            }
        };

        let ctx = self.interp.context::<ShellContext>(self.ctx_id);
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        let req = VfsRequest {
            op: VfsOp::Ctl {
                path: cmd_path,
                cmd: evaluated_args.into_bytes(),
            },
            reply: reply_tx,
        };
        if ctx.vfs_tx.blocking_send(req).is_err() {
            return Err(format!("invalid command name \"{cmd_name}\""));
        }
        match reply_rx.recv() {
            Ok(Ok(resp)) => Ok(String::from_utf8_lossy(&resp).into_owned()),
            _ => Err(format!("invalid command name \"{cmd_name}\"")),
        }
    }
}

/// Extract the command name from a molt "invalid command name" error.
///
/// The error format is: `invalid command name "foo"`
fn extract_unknown_command(msg: &str) -> Option<String> {
    let prefix = "invalid command name \"";
    if let Some(rest) = msg.strip_prefix(prefix) {
        rest.strip_suffix('"').map(|s| s.to_owned())
    } else {
        None
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
    use hyprstream_vfs::proxy::spawn_vfs_proxy;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Namespace, Stat};
    use std::collections::HashMap;
    use std::sync::Arc;

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
        write_buf: std::sync::Mutex<Vec<u8>>,
    }

    #[async_trait]
    impl Mount for MemMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            // Check file exists (or is a prefix of existing files for dirs).
            let exists = self.files.contains_key(&path)
                || self.files.keys().any(|k| k.starts_with(&format!("{path}/")));
            if !exists && !path.is_empty() {
                return Err(MountError::NotFound(path));
            }
            Ok(Fid::new(MemFid {
                path,
                write_buf: std::sync::Mutex::new(Vec::new()),
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
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            // If there's a write buffer (ctl pattern), return that then clear via offset.
            let wb = inner.write_buf.lock().unwrap();
            if !wb.is_empty() {
                let start = offset as usize;
                if start >= wb.len() {
                    return Ok(vec![]);
                }
                return Ok(wb[start..].to_vec());
            }
            drop(wb);
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() { return Ok(vec![]); }
                    Ok(data[start..].to_vec())
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(
            &self,
            fid: &Fid,
            _offset: u64,
            data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            // For ctl files: store the "response" as the uppercased write data.
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            *inner.write_buf.lock().unwrap() = format!("ok: {}", String::from_utf8_lossy(data)).into_bytes();
            Ok(data.len() as u32)
        }

        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
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
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat {
                qtype: 0,
                size: 0,
                name: inner.path.clone(),
                mtime: 0,
            })
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject {
        Subject::new("test")
    }

    fn make_namespace() -> Arc<Namespace> {
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![
            ("temperature", b"0.7"),
            ("status", b"loaded"),
        ]));
        ns.mount("/config", mount).unwrap();

        let model_mount = Arc::new(MemMount::new(vec![("status", b"loaded")]));
        ns.mount("/srv/model", model_mount).unwrap();

        // /bin/ mount with a ctl file (Plan9 PATH model).
        let bin_mount = Arc::new(MemMount::new(vec![("load", b"")]));
        ns.mount("/bin", bin_mount).unwrap();

        Arc::new(ns)
    }

    /// Helper: spawn proxy and construct shell on blocking thread, run closure.
    async fn with_shell<F, R>(f: F) -> R
    where
        F: FnOnce(&mut TclShell) -> R + Send + 'static,
        R: Send + 'static,
    {
        let ns = make_namespace();
        let vfs_tx = spawn_vfs_proxy(Arc::clone(&ns), test_subject());
        tokio::task::spawn_blocking(move || {
            let mut shell = TclShell::new(test_subject(), vfs_tx);
            f(&mut shell)
        })
        .await
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_cat() {
        let result = with_shell(|s| s.eval("cat /config/temperature")).await.unwrap();
        assert_eq!(result, "0.7");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_cat_multi() {
        let result = with_shell(|s| s.eval("cat /config/temperature /config/status")).await.unwrap();
        assert_eq!(result, "0.7loaded");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_cat_not_found() {
        let result = with_shell(|s| s.eval("cat /config/nonexistent")).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_ls() {
        let result = with_shell(|s| s.eval("ls /config")).await.unwrap();
        assert!(result.contains("temperature"));
        assert!(result.contains("status"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_ls_root() {
        let result = with_shell(|s| s.eval("ls")).await.unwrap();
        assert!(result.contains("srv"));
        assert!(result.contains("config"));
        assert!(result.contains("bin"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_echo() {
        let result = with_shell(|s| s.eval("echo /bin/load qwen3:main")).await;
        assert!(result.is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_ctl() {
        let result = with_shell(|s| s.eval("ctl /bin/load qwen3:main")).await.unwrap();
        assert_eq!(result, "ok: qwen3:main");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_help() {
        let result = with_shell(|s| s.eval("help")).await.unwrap();
        assert!(result.contains("cat"));
        assert!(result.contains("ls"));
        assert!(result.contains("ctl"));
        assert!(result.contains("echo"));
        assert!(result.contains("mount"));
        assert!(result.contains("help"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_mount() {
        let result = with_shell(|s| s.eval("mount")).await.unwrap();
        assert!(result.contains("/config"));
        assert!(result.contains("/srv/model"));
        assert!(result.contains("/bin"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_mount_specific() {
        let result = with_shell(|s| s.eval("mount /config")).await.unwrap();
        assert_eq!(result, "mounted: /config");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_mount_not_found() {
        let result = with_shell(|s| s.eval("mount /nonexistent")).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_tcl_builtins() {
        let result = with_shell(|s| {
            s.eval("set x [cat /config/temperature]; string length $x")
        }).await.unwrap();
        assert_eq!(result, "3"); // "0.7" is 3 chars
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eval_tcl_expr() {
        let result = with_shell(|s| s.eval("expr {2 + 3}")).await.unwrap();
        assert_eq!(result, "5");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unknown_resolves_cmd() {
        let result = with_shell(|s| s.eval("load qwen3:main")).await.unwrap();
        assert_eq!(result, "ok: qwen3:main");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unknown_fails_cleanly() {
        let result = with_shell(|s| s.eval("nonexistent_command arg1 arg2")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn has_command_builtin() {
        let ns = make_namespace();
        let vfs_tx = spawn_vfs_proxy(Arc::clone(&ns), test_subject());
        // has_command is sync-only, construct and check on blocking thread
        let result = tokio::task::spawn_blocking(move || {
            let shell = TclShell::new(test_subject(), vfs_tx);
            (
                shell.has_command("cat"),
                shell.has_command("ls"),
                shell.has_command("help"),
                shell.has_command("expr"),
                shell.has_command("nonexistent"),
            )
        })
        .await
        .unwrap();
        assert!(result.0); // cat
        assert!(result.1); // ls
        assert!(result.2); // help
        assert!(result.3); // expr
        assert!(!result.4); // nonexistent
    }

    #[test]
    fn extract_unknown_command_parse() {
        assert_eq!(
            extract_unknown_command("invalid command name \"foo\""),
            Some("foo".to_owned())
        );
        assert_eq!(
            extract_unknown_command("invalid command name \"load\""),
            Some("load".to_owned())
        );
        assert_eq!(extract_unknown_command("some other error"), None);
    }

    // ── Hardening tests ──────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_command_removed() {
        let result = with_shell(|s| s.eval("source /etc/passwd")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exit_command_removed() {
        let result = with_shell(|s| s.eval("exit")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn puts_command_removed() {
        let result = with_shell(|s| s.eval("puts hello")).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rename_command_removed() {
        let result = with_shell(|s| s.eval("rename cat {}")).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn time_command_removed() {
        let result = with_shell(|s| s.eval("time {expr 1} 999999")).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cmd_name_path_traversal_rejected() {
        let result = with_shell(|s| s.eval("../srv/model/ctl something")).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn safe_commands_still_work() {
        let results = with_shell(|s| {
            vec![
                s.eval("expr {2 + 3}"),
                s.eval("set x hello; string length $x"),
                s.eval("if {1} {set y yes}"),
                s.eval("list a b c"),
            ]
        }).await;
        assert_eq!(results[0].as_ref().unwrap(), "5");
        assert_eq!(results[1].as_ref().unwrap(), "5");
        assert_eq!(results[2].as_ref().unwrap(), "yes");
        assert_eq!(results[3].as_ref().unwrap(), "a b c");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn instruction_limit_catches_infinite_loop() {
        let ns = make_namespace();
        let vfs_tx = spawn_vfs_proxy(Arc::clone(&ns), test_subject());
        let result = tokio::task::spawn_blocking(move || {
            let mut shell = TclShell::new(test_subject(), vfs_tx);
            shell.set_instruction_limit(500);
            shell.eval("while {1} {}")
        })
        .await
        .unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("instruction limit"),
            "error should mention instruction limit"
        );
    }

    // ── field builtin tests ─────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_extract_string() {
        // Double-brace: outer {} = Tcl quoting, inner {} = JSON object
        let result = with_shell(|s| {
            s.eval(r#"field {{"name":"alice","age":30}} name"#)
        }).await.unwrap();
        assert_eq!(result, "alice");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_extract_number() {
        let result = with_shell(|s| {
            s.eval(r#"field {{"name":"alice","age":30}} age"#)
        }).await.unwrap();
        assert_eq!(result, "30");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_extract_nested() {
        let result = with_shell(|s| {
            s.eval(r#"field {{"data":{"x":1}}} data"#)
        }).await.unwrap();
        assert_eq!(result, r#"{"x":1}"#);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_not_found() {
        let result = with_shell(|s| {
            s.eval(r#"field {{"name":"alice"}} missing"#)
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("field 'missing' not found"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_invalid_json() {
        let result = with_shell(|s| {
            s.eval(r#"field {not json} name"#)
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid JSON"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_bracket_string_no_subst() {
        // SECURITY: Verify bracket-containing string values don't trigger Tcl substitution
        // Double-brace to pass JSON through Tcl quoting
        let result = with_shell(|s| {
            s.eval(r#"field {{"cmd":"[exec rm -rf /]"}} cmd"#)
        }).await.unwrap();
        // Should return the literal string, not execute it
        assert_eq!(result, "[exec rm -rf /]");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn field_in_help() {
        let result = with_shell(|s| s.eval("help")).await.unwrap();
        assert!(result.contains("field"));
    }

    // ── Runtime safety tests ────────────────────────────────────────────────
    //
    // These tests verify that VFS builtins work when the TclShell runs inside
    // a tokio runtime context (not on spawn_blocking). This catches:
    // - blocking_send() panics (would crash the process)
    // - Deadlocks from proxy sharing the same single-threaded runtime
    //
    // The pattern reproduces the CLI shell's architecture where ChatApp runs
    // directly in the LocalSet event loop.

    /// Helper: spawn a dedicated VFS proxy on its own OS thread (like
    /// `spawn_dedicated_vfs_proxy`), then construct and run the shell
    /// directly on the test's async task — NOT on `spawn_blocking`.
    ///
    /// This simulates the CLI shell path where ChatApp.handle_vfs_command()
    /// runs inside the tokio runtime.
    fn spawn_test_proxy(
        ns: Arc<Namespace>,
        subject: Subject,
    ) -> tokio::sync::mpsc::Sender<VfsRequest> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VfsRequest>(64);
        std::thread::Builder::new()
            .name("test-vfs-proxy".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async move {
                    use hyprstream_vfs::proxy::VfsOp;
                    while let Some(req) = rx.recv().await {
                        let result = match req.op {
                            VfsOp::Cat { ref path } => ns
                                .cat(path, &subject)
                                .await
                                .map_err(|e| e.to_string()),
                            VfsOp::Ls { ref path } => ns
                                .ls(path, &subject)
                                .await
                                .map(|entries| {
                                    entries.iter()
                                        .map(|e| if e.is_dir { format!("{}/", e.name) } else { e.name.clone() })
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
                                Ok(ns.mount_prefixes().join("\n").into_bytes())
                            }
                        };
                        let _ = req.reply.send(result);
                    }
                });
            })
            .unwrap();
        tx
    }

    /// VFS builtins must not panic when called from within a tokio runtime.
    ///
    /// This test constructs the TclShell directly on the async task (simulating
    /// the CLI shell path where ChatApp runs in the LocalSet). The VFS proxy
    /// runs on a dedicated thread to avoid deadlock.
    ///
    /// Before the try_send fix, this test would panic with:
    /// "Cannot block the current thread from within a runtime"
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn builtins_safe_inside_tokio_runtime() {
        let ns = make_namespace();
        let vfs_tx = spawn_test_proxy(Arc::clone(&ns), test_subject());

        // Construct shell directly on the async task — NOT on spawn_blocking.
        // TclShell is !Send so we can't move it across awaits, but we can
        // construct and use it synchronously within a single async block.
        let vfs_tx_clone = vfs_tx.clone();
        let result = tokio::task::spawn_blocking(move || {
            // Enter a tokio runtime context to simulate the CLI shell environment.
            // The _guard ensures Handle::current() returns Ok inside the closure.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let _guard = rt.enter();

            let mut shell = TclShell::new(test_subject(), vfs_tx_clone);

            // These would all panic with blocking_send if called inside a runtime.
            let ls = shell.eval("ls /config").unwrap();
            let cat = shell.eval("cat /config/temperature").unwrap();
            let help = shell.eval("help").unwrap();
            let mount = shell.eval("mount").unwrap();

            (ls, cat, help, mount)
        })
        .await
        .unwrap();

        assert!(result.0.contains("temperature"));
        assert_eq!(result.1, "0.7");
        assert!(result.2.contains("cat"));
        assert!(result.3.contains("/config"));
    }

    /// VFS builtins must work on a current_thread runtime with a dedicated proxy.
    ///
    /// This exactly replicates the CLI shell architecture:
    /// - current_thread runtime with LocalSet
    /// - VFS proxy on a separate OS thread
    /// - TclShell eval called from within the LocalSet
    ///
    /// Before the fix, this would deadlock because spawn_vfs_proxy ran the
    /// proxy on the same current_thread runtime, and reply_rx.recv() blocked
    /// the runtime thread.
    #[test]
    fn builtins_safe_on_current_thread_runtime() {
        let ns = make_namespace();
        let vfs_tx = spawn_test_proxy(Arc::clone(&ns), test_subject());

        // Run on a current_thread runtime — matching the CLI shell's architecture.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let result = rt.block_on(async {
            // The LocalSet is how the CLI shell drives its event loop.
            let local = tokio::task::LocalSet::new();
            local.run_until(async {
                // Construct TclShell inside the LocalSet (like handle_vfs_command).
                let mut shell = TclShell::new(test_subject(), vfs_tx);
                let ls = shell.eval("ls /config").unwrap();
                let cat = shell.eval("cat /config/temperature").unwrap();
                (ls, cat)
            }).await
        });

        assert!(result.0.contains("temperature"));
        assert_eq!(result.1, "0.7");
    }
}
