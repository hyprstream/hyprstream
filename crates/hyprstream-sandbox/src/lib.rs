//! Sandboxed Tcl execution with resource limits.
//!
//! `SandboxedShell` wraps a `TclShell` with an independently constructed
//! namespace. The namespace starts empty — the policy engine decides what
//! to mount based on the Subject's capabilities (principal of least
//! privilege). This crate does NOT make authorization decisions.
//!
//! This crate provides:
//! - **Resource controls**: instruction limits, recursion limits
//! - **Execution isolation**: the sandbox's namespace is independent,
//!   not a fork of any parent namespace
//!
//! # Usage
//!
//! ```ignore
//! // Policy engine builds the namespace for this subject.
//! let mut ns = Namespace::new();
//! // Only mount what the subject is authorized to see.
//! if policy.allows(&subject, "read", "/srv/model") {
//!     ns.mount("/srv/model", model_mount.clone()).unwrap();
//! }
//!
//! let mut sandbox = SandboxedShell::new(ns, subject, rt_handle);
//! sandbox.set_instruction_limit(10_000);
//! let result = sandbox.eval("cat /srv/model/status");
//! ```
//!
//! # Future: WASM isolation
//!
//! Full memory isolation via wasmtime (compiling molt to `wasm32-wasi`)
//! is planned. Landlock and microVM isolation via the WorkerService (Kata)
//! provide additional defense in depth for production workloads.

use std::sync::Arc;

use hyprstream_tcl::TclShell;
use hyprstream_vfs::{Namespace, Subject};

// ─────────────────────────────────────────────────────────────────────────────
// SandboxedShell
// ─────────────────────────────────────────────────────────────────────────────

/// A Tcl shell with an independently constructed namespace and resource limits.
///
/// The namespace is provided by the caller — the policy engine decides what
/// to mount. The sandbox starts with whatever the caller provides (which
/// could be empty for maximum restriction, or fully populated for trusted use).
///
/// `SandboxedShell` is `Send` (same safety argument as `TclShell`).
pub struct SandboxedShell {
    shell: TclShell,
}

impl SandboxedShell {
    /// Create a sandboxed shell with the given namespace.
    ///
    /// The caller constructs the namespace based on policy — only mounting
    /// what the subject is authorized to access. This ensures least privilege:
    /// the sandbox sees nothing unless explicitly granted.
    pub fn new(ns: Namespace, subject: Subject, rt: tokio::runtime::Handle) -> Self {
        let shell = TclShell::new(Arc::new(ns), subject, rt);
        Self { shell }
    }

    /// Evaluate a Tcl script within the sandbox.
    pub fn eval(&mut self, script: &str) -> Result<String, String> {
        self.shell.eval(script)
    }

    /// Set the instruction limit (resource control).
    ///
    /// Limits how many commands can be dispatched per eval. 0 = unlimited.
    /// Defense against infinite loops and CPU bombs.
    pub fn set_instruction_limit(&mut self, limit: usize) {
        self.shell.set_instruction_limit(limit);
    }

    /// Check if a command is available in this sandbox.
    pub fn has_command(&self, name: &str) -> bool {
        self.shell.has_command(name)
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
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
    use std::collections::HashMap;

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

    struct MemFid { path: String }

    #[async_trait]
    impl Mount for MemMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            let exists = self.files.contains_key(&path)
                || self.files.keys().any(|k| k.starts_with(&format!("{path}/")));
            if !exists && !path.is_empty() {
                return Err(MountError::NotFound(path));
            }
            Ok(Fid::new(MemFid { path }))
        }

        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }

        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() { return Ok(vec![]); }
                    Ok(data[start..].to_vec())
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(&self, _fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }

        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![])
        }

        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat { qtype: 0, size: 0, name: inner.path.clone(), mtime: 0 })
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject { Subject::new("test") }

    fn test_handle() -> tokio::runtime::Handle {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let handle = rt.handle().clone();
        std::mem::forget(rt);
        handle
    }

    #[test]
    fn empty_namespace_denies_everything() {
        let ns = Namespace::new();
        let mut sandbox = SandboxedShell::new(ns, test_subject(), test_handle());
        assert!(sandbox.eval("cat /srv/model/status").is_err());
        assert!(sandbox.eval("cat /config/temperature").is_err());
    }

    #[test]
    fn policy_grants_specific_access() {
        // Simulate policy: this subject can see /srv/model but not /config.
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![("status", b"loaded")])))
            .unwrap();
        // /config intentionally NOT mounted — least privilege.

        let mut sandbox = SandboxedShell::new(ns, test_subject(), test_handle());
        assert_eq!(sandbox.eval("cat /srv/model/status").unwrap(), "loaded");
        assert!(sandbox.eval("cat /config/temperature").is_err());
    }

    #[test]
    fn instruction_limit_enforced() {
        let ns = Namespace::new();
        let mut sandbox = SandboxedShell::new(ns, test_subject(), test_handle());
        sandbox.set_instruction_limit(100);
        let result = sandbox.eval("while {1} {}");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("instruction limit"));
    }

    #[test]
    fn dangerous_commands_removed() {
        let ns = Namespace::new();
        let mut sandbox = SandboxedShell::new(ns, test_subject(), test_handle());
        assert!(sandbox.eval("source /etc/passwd").is_err());
        assert!(sandbox.eval("exit").is_err());
        assert!(sandbox.eval("puts hello").is_err());
    }

    #[test]
    fn standard_tcl_works() {
        let ns = Namespace::new();
        let mut sandbox = SandboxedShell::new(ns, test_subject(), test_handle());
        assert_eq!(sandbox.eval("expr {2 + 3}").unwrap(), "5");
        assert_eq!(sandbox.eval("set x hello; string length $x").unwrap(), "5");
    }
}
