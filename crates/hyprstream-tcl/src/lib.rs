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

use std::sync::Arc;

use hyprstream_vfs::{Namespace, Subject};
use molt::types::*;
use molt::Interp;

/// Context stored in the molt interpreter via `save_context()`.
pub(crate) struct VfsContext {
    pub ns: Arc<Namespace>,
    pub subject: Subject,
}

/// A Tcl shell backed by the hyprstream VFS namespace.
///
/// Contains a molt `Interp` which is `!Send` (uses `Rc` internally).
/// We implement `Send` because `TclShell` is constructed and then moved
/// to a single owner thread (ChatApp) where it stays for its lifetime.
/// It is never accessed from multiple threads concurrently.
pub struct TclShell {
    interp: Interp,
    ctx_id: ContextID,
}

// SAFETY: TclShell is constructed, moved once into ChatApp (which runs on a
// single thread), and never shared. The inner `Rc`s in molt::Value are only
// accessed from that thread. This is the same pattern as tch-rs tensors in
// the inference service (Thread spawner mode for !Send types).
unsafe impl Send for TclShell {}

impl TclShell {
    /// Create a new Tcl shell bound to the given VFS namespace and caller identity.
    ///
    /// Dangerous commands are removed and the recursion limit is lowered.
    /// All host I/O goes through the VFS — this is a guest interpreter.
    pub fn new(ns: Arc<Namespace>, subject: Subject) -> Self {
        let mut interp = Interp::new();

        // Remove dangerous commands — all host I/O goes through VFS.
        // source: reads arbitrary host files via std::fs::read_to_string
        // exit: calls std::process::exit(), kills the server
        // puts: writes to server stdout (log injection)
        // rename: can shadow/delete security-critical builtins
        // global: scope manipulation unnecessary in VFS shell
        // time: runs a command N times (user-controlled N, CPU bomb)
        // parse/pdump/pclear: debug internals, not for users
        for cmd in &[
            "source", "exit", "puts", "rename", "global", "time",
            "parse", "pdump", "pclear",
        ] {
            if interp.has_command(cmd) {
                interp.remove_command(cmd);
            }
        }
        interp.set_recursion_limit(100);

        let ctx = VfsContext { ns, subject };
        let ctx_id = interp.save_context(ctx);
        builtins::register_all(&mut interp, ctx_id);
        Self { interp, ctx_id }
    }

    /// Evaluate a Tcl script. Returns Ok(output) or Err(error message).
    ///
    /// On "invalid command name" errors, tries `/bin/{name}` resolution
    /// (Plan9 PATH model) before returning the error.
    pub fn eval(&mut self, script: &str) -> Result<String, String> {
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

    /// Try resolving an unknown command via `/bin/{name}` (Plan9 PATH model).
    ///
    /// Extracts everything after the command name as the argument payload,
    /// then does a ctl (write+read) on `/bin/{name}`.
    fn try_cmd_resolve(&mut self, cmd_name: &str, script: &str) -> Result<String, String> {
        // Sanitize: reject path traversal in command names.
        if cmd_name.contains('/') || cmd_name.contains("..") {
            return Err(format!("invalid command name \"{cmd_name}\""));
        }
        let ctx = self.interp.context::<VfsContext>(self.ctx_id);
        let cmd_path = format!("/bin/{cmd_name}");
        let args = script
            .trim()
            .strip_prefix(cmd_name)
            .unwrap_or("")
            .trim();
        match ctx.ns.ctl(&cmd_path, args.as_bytes(), &ctx.subject) {
            Ok(resp) => Ok(String::from_utf8_lossy(&resp).into_owned()),
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
    use hyprstream_vfs::{DirEntry, LocalFid, LocalMount, MountTarget, Stat};
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
        write_buf: Vec<u8>,
    }

    impl LocalMount for MemMount {
        fn walk(&self, components: &[&str], _caller: &Subject) -> Result<LocalFid, String> {
            let path = components.join("/");
            // Check file exists (or is a prefix of existing files for dirs).
            let exists = self.files.contains_key(&path)
                || self.files.keys().any(|k| k.starts_with(&format!("{path}/")));
            if !exists && !path.is_empty() {
                return Err(format!("not found: {path}"));
            }
            Ok(LocalFid::new(MemFid {
                path,
                write_buf: Vec::new(),
            }))
        }

        fn open(&self, _fid: &mut LocalFid, _mode: u8, _caller: &Subject) -> Result<(), String> {
            Ok(())
        }

        fn read(
            &self,
            fid: &LocalFid,
            _offset: u64,
            _count: u32,
            _caller: &Subject,
        ) -> Result<Vec<u8>, String> {
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
            // If there's a write buffer (ctl pattern), return that.
            if !inner.write_buf.is_empty() {
                return Ok(inner.write_buf.clone());
            }
            self.files
                .get(&inner.path)
                .cloned()
                .ok_or_else(|| format!("not found: {}", inner.path))
        }

        fn write(
            &self,
            fid: &LocalFid,
            _offset: u64,
            data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, String> {
            // For ctl files: store the "response" as the uppercased write data.
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
            // SAFETY: MemFid write_buf mutation — test-only, single-threaded.
            let fid_ptr = inner as *const MemFid as *mut MemFid;
            unsafe { (*fid_ptr).write_buf = format!("ok: {}", String::from_utf8_lossy(data)).into_bytes() };
            Ok(data.len() as u32)
        }

        fn readdir(&self, fid: &LocalFid, _caller: &Subject) -> Result<Vec<DirEntry>, String> {
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
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
                        });
                    }
                } else if inner.path.is_empty() && !key.contains('/') {
                    entries.push(DirEntry {
                        name: key.clone(),
                        is_dir: false,
                        size: 0,
                    });
                }
            }
            Ok(entries)
        }

        fn stat(&self, fid: &LocalFid, _caller: &Subject) -> Result<Stat, String> {
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
            Ok(Stat {
                qtype: 0,
                size: 0,
                name: inner.path.clone(),
                mtime: 0,
            })
        }

        fn clunk(&self, _fid: LocalFid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject {
        Subject::new("test")
    }

    fn make_shell() -> TclShell {
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![
            ("temperature", b"0.7"),
            ("status", b"loaded"),
        ]));
        ns.mount("/config", MountTarget::Local(mount)).unwrap();

        let model_mount = Arc::new(MemMount::new(vec![("status", b"loaded")]));
        ns.mount("/srv/model", MountTarget::Local(model_mount))
            .unwrap();

        // /bin/ mount with a ctl file (Plan9 PATH model).
        let bin_mount = Arc::new(MemMount::new(vec![("load", b"")]));
        ns.mount("/bin", MountTarget::Local(bin_mount)).unwrap();

        TclShell::new(Arc::new(ns), test_subject())
    }

    #[test]
    fn eval_cat() {
        let mut shell = make_shell();
        let result = shell.eval("cat /config/temperature").unwrap();
        assert_eq!(result, "0.7");
    }

    #[test]
    fn eval_cat_multi() {
        let mut shell = make_shell();
        let result = shell.eval("cat /config/temperature /config/status").unwrap();
        assert_eq!(result, "0.7loaded");
    }

    #[test]
    fn eval_cat_not_found() {
        let mut shell = make_shell();
        let result = shell.eval("cat /config/nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn eval_ls() {
        let mut shell = make_shell();
        let result = shell.eval("ls /config").unwrap();
        // Order may vary; check both entries are present.
        assert!(result.contains("temperature"));
        assert!(result.contains("status"));
    }

    #[test]
    fn eval_ls_root() {
        let mut shell = make_shell();
        let result = shell.eval("ls").unwrap();
        assert!(result.contains("srv"));
        assert!(result.contains("config"));
        assert!(result.contains("bin"));
    }

    #[test]
    fn eval_echo() {
        let mut shell = make_shell();
        // Echo just writes — shouldn't error.
        let result = shell.eval("echo /bin/load qwen3:main");
        assert!(result.is_ok());
    }

    #[test]
    fn eval_ctl() {
        let mut shell = make_shell();
        let result = shell.eval("ctl /bin/load qwen3:main").unwrap();
        assert_eq!(result, "ok: qwen3:main");
    }

    #[test]
    fn eval_help() {
        let mut shell = make_shell();
        let result = shell.eval("help").unwrap();
        assert!(result.contains("cat"));
        assert!(result.contains("ls"));
        assert!(result.contains("ctl"));
        assert!(result.contains("echo"));
        assert!(result.contains("mount"));
        assert!(result.contains("help"));
    }

    #[test]
    fn eval_mount() {
        let mut shell = make_shell();
        let result = shell.eval("mount").unwrap();
        assert!(result.contains("/config"));
        assert!(result.contains("/srv/model"));
        assert!(result.contains("/bin"));
    }

    #[test]
    fn eval_mount_specific() {
        let mut shell = make_shell();
        let result = shell.eval("mount /config").unwrap();
        assert_eq!(result, "mounted: /config");
    }

    #[test]
    fn eval_mount_not_found() {
        let mut shell = make_shell();
        let result = shell.eval("mount /nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn eval_tcl_builtins() {
        let mut shell = make_shell();
        // Use standard Tcl: set a variable from cat output, use string length.
        let result = shell
            .eval("set x [cat /config/temperature]; string length $x")
            .unwrap();
        assert_eq!(result, "3"); // "0.7" is 3 chars
    }

    #[test]
    fn eval_tcl_expr() {
        let mut shell = make_shell();
        let result = shell.eval("expr {2 + 3}").unwrap();
        assert_eq!(result, "5");
    }

    #[test]
    fn unknown_resolves_cmd() {
        let mut shell = make_shell();
        // "load" is not a registered command, but /bin/load exists.
        let result = shell.eval("load qwen3:main").unwrap();
        assert_eq!(result, "ok: qwen3:main");
    }

    #[test]
    fn unknown_fails_cleanly() {
        let mut shell = make_shell();
        let result = shell.eval("nonexistent_command arg1 arg2");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[test]
    fn has_command_builtin() {
        let shell = make_shell();
        assert!(shell.has_command("cat"));
        assert!(shell.has_command("ls"));
        assert!(shell.has_command("help"));
        assert!(shell.has_command("expr")); // standard Tcl
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

    #[test]
    fn source_command_removed() {
        let mut shell = make_shell();
        let result = shell.eval("source /etc/passwd");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[test]
    fn exit_command_removed() {
        let mut shell = make_shell();
        let result = shell.eval("exit");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid command name"));
    }

    #[test]
    fn puts_command_removed() {
        let mut shell = make_shell();
        let result = shell.eval("puts hello");
        assert!(result.is_err());
    }

    #[test]
    fn rename_command_removed() {
        let mut shell = make_shell();
        let result = shell.eval("rename cat {}");
        assert!(result.is_err());
    }

    #[test]
    fn time_command_removed() {
        let mut shell = make_shell();
        let result = shell.eval("time {expr 1} 999999");
        assert!(result.is_err());
    }

    #[test]
    fn cmd_name_path_traversal_rejected() {
        let mut shell = make_shell();
        // A command name with path separators should not resolve.
        let result = shell.eval("../srv/model/ctl something");
        assert!(result.is_err());
    }

    #[test]
    fn safe_commands_still_work() {
        let mut shell = make_shell();
        // Core Tcl still available.
        assert_eq!(shell.eval("expr {2 + 3}").unwrap(), "5");
        assert_eq!(shell.eval("set x hello; string length $x").unwrap(), "5");
        assert_eq!(shell.eval("if {1} {set y yes}").unwrap(), "yes");
        assert_eq!(shell.eval("list a b c").unwrap(), "a b c");
    }
}
