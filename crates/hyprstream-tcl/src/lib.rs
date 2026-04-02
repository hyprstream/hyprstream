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

use hyprstream_vfs::{Namespace, Subject};
use molt::types::*;
use molt::Interp;
use std::sync::Arc;

pub use mount::{create_mount_channel, TclCommand, TclMount};

/// Context stored in the molt interpreter via `save_context()`.
///
/// VFS operations are called directly via the Namespace — no proxy channel needed.
pub(crate) struct ShellContext {
    pub subject: Subject,
    pub namespace: Arc<Namespace>,
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
    /// Create a new Tcl shell bound to the given caller identity and VFS namespace.
    ///
    /// Builtins `.await` VFS operations directly on the namespace — no proxy channel.
    ///
    /// TclShell is `!Send` due to molt's `Rc`-based `Value`.
    /// Must be used within a `LocalSet` or single-threaded executor.
    ///
    /// Dangerous commands are removed and the recursion limit is lowered.
    /// All host I/O goes through the VFS — this is a guest interpreter.
    pub fn new(subject: Subject, namespace: Arc<Namespace>) -> Self {
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

        let ctx = ShellContext { subject, namespace };
        let ctx_id = interp.save_context(ctx);
        builtins::register_all(&mut interp, ctx_id);
        Self { interp, ctx_id }
    }

    /// Evaluate a Tcl script. Returns Ok(output) or Err(error message).
    ///
    /// On "invalid command name" errors, tries `/bin/{name}` resolution
    /// (Plan9 PATH model) before returning the error.
    pub async fn eval(&mut self, script: &str) -> Result<String, String> {
        self.interp.reset_instruction_count();
        match self.interp.eval(script).await {
            Ok(val) => Ok(val.to_string()),
            Err(exception) => {
                let msg = exception.value().to_string();
                // Try /bin/ resolution on command-not-found.
                if let Some(cmd_name) = extract_unknown_command(&msg) {
                    if let Ok(result) = self.try_cmd_resolve(&cmd_name, script).await {
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
    pub async fn process_command(&mut self, cmd: TclCommand) {
        match cmd {
            TclCommand::Eval { script, resp } => {
                let _ = resp.send(self.eval(&script).await);
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
    /// Two-phase resolution:
    /// 1. Try ctl (write+read) — for Rust-backed service actions
    /// 2. If ctl fails, try cat — for `.tcl` script files, then eval with args
    async fn try_cmd_resolve(&mut self, cmd_name: &str, script: &str) -> Result<String, String> {
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
            match self.interp.eval(&format!("list {raw_args}")).await {
                Ok(val) => val.to_string(),
                Err(e) => return Err(e.value().to_string()),
            }
        };

        // Get namespace and subject for direct VFS access.
        let ctx = self.interp.context::<ShellContext>(self.ctx_id);
        let namespace = Arc::clone(&ctx.namespace);
        let subject = ctx.subject.clone();

        // Phase 1: Try ctl (service actions — CtlFile nodes).
        if let Ok(resp) = namespace.ctl(&cmd_path, evaluated_args.as_bytes(), &subject).await {
            return Ok(String::from_utf8_lossy(&resp).into_owned());
        }

        // Phase 2: Try cat (script files — ReadFile nodes).
        // Read the script, prepend `set argv {args}`, and eval it.
        match namespace.cat(&cmd_path, &subject).await {
            Ok(script_bytes) => {
                let tool_script = String::from_utf8_lossy(&script_bytes);
                // Wrap: set argv before executing so the script can access args.
                let wrapped = format!("set argv {{{evaluated_args}}}\n{tool_script}");
                match self.interp.eval(&wrapped).await {
                    Ok(val) => Ok(val.to_string()),
                    Err(e) => Err(e.value().to_string()),
                }
            }
            Err(_) => Err(format!("invalid command name \"{cmd_name}\"")),
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
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Namespace, Stat};
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
        write_buf: std::sync::Mutex<Vec<u8>>,
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

        let bin_mount = Arc::new(MemMount::new(vec![("load", b"")]));
        ns.mount("/bin", bin_mount).unwrap();

        Arc::new(ns)
    }

    /// Helper: construct shell and run async closure within a LocalSet.
    async fn with_shell<F, Fut, R>(f: F) -> R
    where
        F: FnOnce(TclShell) -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let ns = make_namespace();
        let mut shell = TclShell::new(test_subject(), ns);
        f(shell).await
    }

    // Note: TclShell is !Send, so we use current_thread + LocalSet for all tests.

    #[tokio::test(flavor = "current_thread")]
    async fn eval_cat() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("cat /config/temperature").await
        }).await.unwrap();
        assert_eq!(result, "0.7");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_cat_multi() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("cat /config/temperature /config/status").await
        }).await.unwrap();
        assert_eq!(result, "0.7loaded");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_cat_not_found() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("cat /config/nonexistent").await
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_ls() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("ls /config").await
        }).await.unwrap();
        assert!(result.contains("temperature"));
        assert!(result.contains("status"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_ls_root() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("ls").await
        }).await.unwrap();
        assert!(result.contains("srv"));
        assert!(result.contains("config"));
        assert!(result.contains("bin"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_write() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("write /bin/load qwen3:main").await
        }).await;
        assert!(result.is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_ctl() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("ctl /bin/load qwen3:main").await
        }).await.unwrap();
        assert_eq!(result, "ok: qwen3:main");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_help() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("help").await
        }).await.unwrap();
        assert!(result.contains("cat"));
        assert!(result.contains("ls"));
        assert!(result.contains("ctl"));
        assert!(result.contains("write"));
        assert!(result.contains("mount"));
        assert!(result.contains("help"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_mount() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("mount").await
        }).await.unwrap();
        assert!(result.contains("/config"));
        assert!(result.contains("/srv/model"));
        assert!(result.contains("/bin"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_mount_specific() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("mount /config").await
        }).await.unwrap();
        assert_eq!(result, "mounted: /config");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_mount_not_found() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("mount /nonexistent").await
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_tcl_builtins() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("set x [cat /config/temperature]; string length $x").await
        }).await.unwrap();
        assert_eq!(result, "3");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eval_tcl_expr() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("expr {2 + 3}").await
        }).await.unwrap();
        assert_eq!(result, "5");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unknown_resolves_cmd() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("load qwen3:main").await
        }).await.unwrap();
        assert_eq!(result, "ok: qwen3:main");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unknown_fails_cleanly() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("nonexistent_command arg1 arg2").await
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[test]
    fn has_command_builtin() {
        let ns = make_namespace();
        let shell = TclShell::new(test_subject(), ns);
        assert!(shell.has_command("cat"));
        assert!(shell.has_command("ls"));
        assert!(shell.has_command("help"));
        assert!(shell.has_command("expr"));
        assert!(!shell.has_command("nonexistent"));
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

    #[tokio::test(flavor = "current_thread")]
    async fn source_command_removed() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("source /etc/passwd").await
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exit_command_removed() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("exit").await
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn puts_command_removed() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("puts hello").await
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rename_command_removed() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("rename cat {}").await
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn time_command_removed() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("time {expr 1} 999999").await
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cmd_name_path_traversal_rejected() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.eval("../srv/model/ctl something").await
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn safe_commands_still_work() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let results = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            vec![
                s.eval("expr {2 + 3}").await,
                s.eval("set x hello; string length $x").await,
                s.eval("if {1} {set y yes}").await,
                s.eval("list a b c").await,
            ]
        }).await;
        assert_eq!(results[0].as_ref().unwrap(), "5");
        assert_eq!(results[1].as_ref().unwrap(), "5");
        assert_eq!(results[2].as_ref().unwrap(), "yes");
        assert_eq!(results[3].as_ref().unwrap(), "a b c");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn instruction_limit_catches_infinite_loop() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut shell = TclShell::new(test_subject(), ns);
            shell.set_instruction_limit(500);
            shell.eval("while {1} {}").await
        }).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("instruction limit"),
            "error should mention instruction limit"
        );
    }

    // ── json builtin tests ──────────────────────────────────────────────────

    #[tokio::test(flavor = "current_thread")]
    async fn json_parse_object() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"dict get [json parse {{"name":"alice","age":30}}] name"#).await
        }).await.unwrap();
        assert_eq!(result, "alice");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_parse_number() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"dict get [json parse {{"name":"alice","age":30}}] age"#).await
        }).await.unwrap();
        assert_eq!(result, "30");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_parse_nested() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"dict get [dict get [json parse {{"data":{"x":1}}}] data] x"#).await
        }).await.unwrap();
        assert_eq!(result, "1");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_parse_invalid() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"json parse {not json}"#).await
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid JSON"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_parse_array() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"lindex [json parse {[1, 2, 3]}] 1"#).await
        }).await.unwrap();
        assert_eq!(result, "2");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_parse_bool_null() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"dict get [json parse {{"flag":true,"empty":null}}] flag"#).await
        }).await.unwrap();
        assert_eq!(result, "1");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_unknown_subcommand() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval(r#"json encode {something}"#).await
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown subcommand"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn json_in_help() {
        let ns = make_namespace();
        let local = tokio::task::LocalSet::new();
        let result = local.run_until(async {
            let mut s = TclShell::new(test_subject(), ns);
            s.eval("help").await
        }).await.unwrap();
        assert!(result.contains("json"));
    }
}
