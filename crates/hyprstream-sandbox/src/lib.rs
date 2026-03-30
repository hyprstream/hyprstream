//! Trust-level-aware sandbox for untrusted Tcl execution.
//!
//! Wraps [`hyprstream_tcl::TclShell`] with configurable instruction limits
//! and a forked VFS namespace that removes sensitive mount points based on
//! the caller's trust level.
//!
//! # Trust levels
//!
//! | Level | Instruction limit | Namespace restrictions |
//! |------------|------------------|-------------------------------------------------|
//! | `Human` | 100,000 | Full namespace |
//! | `Agent` | 10,000 | `/private/` and `/env/` unmounted |
//! | `Federation`| 1,000 | Above + `/config/` and `/net/` unmounted |
//! | `Untrusted` | 500 | Only `/srv/` and `/bin/` remain |
//!
//! # Future: WASM isolation
//!
//! The current implementation uses a process-local `TclShell` with lowered
//! instruction limits and a restricted namespace. Full memory isolation via
//! wasmtime (compiling molt to `wasm32-wasi`) is planned but not yet
//! implemented — it requires:
//!
//! 1. Molt compiled to `wasm32-wasi` (no `std::fs`, no `std::process`)
//! 2. WASI imports for VFS operations (cat, ls, echo, ctl)
//! 3. Fuel metering and memory limits via wasmtime `Store` config
//! 4. `SharedMemoryMount` for zero-copy data exchange
//!
//! The `TrustLevel`-based approach gives us immediate value for LLM agent
//! sandboxing without the wasmtime compilation overhead.

use std::sync::Arc;

use hyprstream_tcl::TclShell;
use hyprstream_vfs::{Namespace, Subject};

// ─────────────────────────────────────────────────────────────────────────────
// Trust levels
// ─────────────────────────────────────────────────────────────────────────────

/// Trust level controls instruction limits and namespace visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustLevel {
    /// Interactive human user. Full namespace, 100K instruction limit.
    Human,
    /// LLM agent. Reduced limits, sensitive paths unmounted.
    Agent,
    /// Federated peer. Minimal access, tight limits.
    Federation,
    /// Fully untrusted (user-uploaded scripts, unknown origin).
    Untrusted,
}

impl TrustLevel {
    /// Instruction limit for this trust level.
    pub fn instruction_limit(self) -> usize {
        match self {
            TrustLevel::Human => 100_000,
            TrustLevel::Agent => 10_000,
            TrustLevel::Federation => 1_000,
            TrustLevel::Untrusted => 500,
        }
    }

    /// Path prefixes to unmount at this trust level.
    ///
    /// Each level includes the restrictions of higher trust levels.
    fn restricted_prefixes(self) -> &'static [&'static str] {
        match self {
            TrustLevel::Human => &[],
            TrustLevel::Agent => &["/private", "/env"],
            TrustLevel::Federation => &["/private", "/env", "/config", "/net"],
            TrustLevel::Untrusted => &["/private", "/env", "/config", "/net"],
        }
    }

    /// For `Untrusted`, we use an allowlist instead of a blocklist.
    /// Only these prefixes survive.
    fn allowed_prefixes(self) -> Option<&'static [&'static str]> {
        match self {
            TrustLevel::Untrusted => Some(&["/srv", "/bin"]),
            _ => None,
        }
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::Human => write!(f, "human"),
            TrustLevel::Agent => write!(f, "agent"),
            TrustLevel::Federation => write!(f, "federation"),
            TrustLevel::Untrusted => write!(f, "untrusted"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SandboxedShell
// ─────────────────────────────────────────────────────────────────────────────

/// A Tcl shell with trust-level-aware restrictions.
///
/// Forks the parent namespace to create an isolated view, then removes
/// mount points based on the trust level. The instruction limit is set
/// per trust level to prevent resource exhaustion.
///
/// `SandboxedShell` is `Send` (same safety argument as `TclShell`).
pub struct SandboxedShell {
    shell: TclShell,
    trust_level: TrustLevel,
}

impl SandboxedShell {
    /// Create a sandboxed shell by forking the given namespace.
    ///
    /// The fork is immediate and cheap (Arc clones of mount targets).
    /// Sensitive paths are removed from the forked namespace based on
    /// the trust level. The original namespace is not modified.
    ///
    /// The `rt` handle is used by the Tcl shell to `block_on()` async VFS operations.
    pub fn new(ns: &Namespace, subject: Subject, trust_level: TrustLevel, rt: tokio::runtime::Handle) -> Self {
        let mut forked = ns.fork();
        Self::apply_restrictions(&mut forked, trust_level);

        let mut shell = TclShell::new(Arc::new(forked), subject, rt);
        shell.set_instruction_limit(trust_level.instruction_limit());

        Self { shell, trust_level }
    }

    /// Evaluate a Tcl script within the sandbox.
    pub fn eval(&mut self, script: &str) -> Result<String, String> {
        self.shell.eval(script)
    }

    /// Get the trust level of this sandbox.
    pub fn trust_level(&self) -> TrustLevel {
        self.trust_level
    }

    /// Override the instruction limit (within the trust level's maximum).
    ///
    /// The limit is clamped to the trust level's default. Pass 0 to
    /// restore the trust level's default limit.
    pub fn set_instruction_limit(&mut self, limit: usize) {
        let max = self.trust_level.instruction_limit();
        let effective = if limit == 0 || limit > max { max } else { limit };
        self.shell.set_instruction_limit(effective);
    }

    /// Check if a command is available in this sandbox.
    pub fn has_command(&self, name: &str) -> bool {
        self.shell.has_command(name)
    }

    /// Apply namespace restrictions based on trust level.
    fn apply_restrictions(ns: &mut Namespace, trust_level: TrustLevel) {
        if let Some(allowed) = trust_level.allowed_prefixes() {
            // Allowlist mode: remove everything not in the allowed set.
            let prefixes: Vec<String> = ns
                .mount_prefixes()
                .iter()
                .map(|s| s.to_string())
                .collect();
            for prefix in &prefixes {
                let keep = allowed.iter().any(|a| prefix.starts_with(a));
                if !keep {
                    ns.unmount(prefix);
                }
            }
        } else {
            // Blocklist mode: remove restricted prefixes.
            for prefix in trust_level.restricted_prefixes() {
                ns.unmount(prefix);
            }
        }
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
            Ok(Fid::new(MemFid { path }))
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
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() {
                        return Ok(vec![]);
                    }
                    Ok(data[start..].to_vec())
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(
            &self,
            _fid: &Fid,
            _offset: u64,
            data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }

        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
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
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
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
        Subject::new("test-agent")
    }

    fn test_handle() -> tokio::runtime::Handle {
        // Create and leak a runtime for the test handle.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let handle = rt.handle().clone();
        std::mem::forget(rt);
        handle
    }

    /// Build a namespace with mounts at several prefixes for testing restrictions.
    fn make_namespace() -> Namespace {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![("status", b"loaded")])))
            .unwrap();
        ns.mount("/bin", Arc::new(MemMount::new(vec![("help", b"")])))
            .unwrap();
        ns.mount("/config", Arc::new(MemMount::new(vec![("temperature", b"0.7")])))
            .unwrap();
        ns.mount("/private", Arc::new(MemMount::new(vec![("key", b"secret")])))
            .unwrap();
        ns.mount("/env", Arc::new(MemMount::new(vec![("HOME", b"/home/test")])))
            .unwrap();
        ns.mount("/net", Arc::new(MemMount::new(vec![("peer-a", b"")])))
            .unwrap();
        ns
    }

    // ── Trust level unit tests ─────────────────────────────────────────────

    #[test]
    fn trust_level_instruction_limits() {
        assert_eq!(TrustLevel::Human.instruction_limit(), 100_000);
        assert_eq!(TrustLevel::Agent.instruction_limit(), 10_000);
        assert_eq!(TrustLevel::Federation.instruction_limit(), 1_000);
        assert_eq!(TrustLevel::Untrusted.instruction_limit(), 500);
    }

    #[test]
    fn trust_level_display() {
        assert_eq!(TrustLevel::Human.to_string(), "human");
        assert_eq!(TrustLevel::Agent.to_string(), "agent");
        assert_eq!(TrustLevel::Federation.to_string(), "federation");
        assert_eq!(TrustLevel::Untrusted.to_string(), "untrusted");
    }

    // ── Human trust level ──────────────────────────────────────────────────

    #[test]
    fn human_sees_all_mounts() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Human, test_handle());
        // Human can access everything.
        assert_eq!(sandbox.eval("cat /config/temperature").unwrap(), "0.7");
        assert_eq!(sandbox.eval("cat /private/key").unwrap(), "secret");
        assert_eq!(
            sandbox.eval("cat /env/HOME").unwrap(),
            "/home/test"
        );
        assert_eq!(sandbox.eval("cat /srv/model/status").unwrap(), "loaded");
    }

    // ── Agent trust level ──────────────────────────────────────────────────

    #[test]
    fn agent_cannot_see_private_or_env() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Agent, test_handle());
        // /private/ and /env/ should be unmounted.
        assert!(sandbox.eval("cat /private/key").is_err());
        assert!(sandbox.eval("cat /env/HOME").is_err());
        // /config/ and /srv/ still accessible.
        assert_eq!(sandbox.eval("cat /config/temperature").unwrap(), "0.7");
        assert_eq!(sandbox.eval("cat /srv/model/status").unwrap(), "loaded");
    }

    #[test]
    fn agent_instruction_limit_enforced() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Agent, test_handle());
        let result = sandbox.eval("while {1} {}");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("instruction limit"),
            "agent should hit instruction limit"
        );
    }

    // ── Federation trust level ─────────────────────────────────────────────

    #[test]
    fn federation_minimal_access() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Federation, test_handle());
        // Only /srv/ and /bin/ should be accessible (plus no /config/, /net/).
        assert!(sandbox.eval("cat /private/key").is_err());
        assert!(sandbox.eval("cat /env/HOME").is_err());
        assert!(sandbox.eval("cat /config/temperature").is_err());
        assert_eq!(sandbox.eval("cat /srv/model/status").unwrap(), "loaded");
    }

    #[test]
    fn federation_tight_instruction_limit() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Federation, test_handle());
        // 1000 instructions should still allow simple expressions.
        assert_eq!(sandbox.eval("expr {2 + 3}").unwrap(), "5");
        // But loops will hit the limit fast.
        let result = sandbox.eval("while {1} {}");
        assert!(result.is_err());
    }

    // ── Untrusted trust level ──────────────────────────────────────────────

    #[test]
    fn untrusted_allowlist_only() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Untrusted, test_handle());
        // Only /srv/ and /bin/ should survive.
        assert!(sandbox.eval("cat /private/key").is_err());
        assert!(sandbox.eval("cat /env/HOME").is_err());
        assert!(sandbox.eval("cat /config/temperature").is_err());
        assert!(sandbox.eval("cat /net/peer-a").is_err());
        // /srv/ still works.
        assert_eq!(sandbox.eval("cat /srv/model/status").unwrap(), "loaded");
    }

    #[test]
    fn untrusted_very_tight_limit() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Untrusted, test_handle());
        // 500 instructions: simple expr works, loop fails.
        assert_eq!(sandbox.eval("expr {1 + 1}").unwrap(), "2");
        assert!(sandbox.eval("while {1} {}").is_err());
    }

    // ── Forking isolation ──────────────────────────────────────────────────

    #[test]
    fn sandbox_does_not_modify_parent_namespace() {
        let ns = make_namespace();
        let original_prefixes = ns.mount_prefixes().len();

        // Create a restrictive sandbox.
        let _sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Untrusted, test_handle());

        // Parent namespace should be unmodified.
        assert_eq!(ns.mount_prefixes().len(), original_prefixes);
    }

    // ── Instruction limit clamping ─────────────────────────────────────────

    #[test]
    fn set_instruction_limit_clamped() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Agent, test_handle());
        // Trying to set higher than trust level's max: clamped.
        sandbox.set_instruction_limit(1_000_000);
        // Should still hit agent limit (10K), not 1M.
        let result = sandbox.eval("set i 0; while {$i < 50000} { incr i }");
        assert!(result.is_err());
    }

    #[test]
    fn set_instruction_limit_lower_works() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Human, test_handle());
        // Set lower than default: should be respected.
        sandbox.set_instruction_limit(100);
        let result = sandbox.eval("set i 0; while {$i < 1000} { incr i }");
        assert!(result.is_err());
    }

    #[test]
    fn set_instruction_limit_zero_restores_default() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Human, test_handle());
        sandbox.set_instruction_limit(10);
        // Zero restores default.
        sandbox.set_instruction_limit(0);
        // With 100K instructions, this loop should complete.
        let result = sandbox.eval("set i 0; while {$i < 100} { incr i }; set i");
        assert_eq!(result.unwrap(), "100");
    }

    // ── Dangerous commands still removed ───────────────────────────────────

    #[test]
    fn dangerous_commands_removed_in_sandbox() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Human, test_handle());
        assert!(sandbox.eval("source /etc/passwd").is_err());
        assert!(sandbox.eval("exit").is_err());
        assert!(sandbox.eval("puts hello").is_err());
    }

    // ── Standard Tcl still works ───────────────────────────────────────────

    #[test]
    fn standard_tcl_in_sandbox() {
        let ns = make_namespace();
        let mut sandbox = SandboxedShell::new(&ns, test_subject(), TrustLevel::Agent, test_handle());
        assert_eq!(sandbox.eval("expr {2 + 3}").unwrap(), "5");
        assert_eq!(
            sandbox.eval("set x hello; string length $x").unwrap(),
            "5"
        );
        assert_eq!(sandbox.eval("list a b c").unwrap(), "a b c");
    }
}
