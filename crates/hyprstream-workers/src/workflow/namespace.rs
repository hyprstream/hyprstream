//! Workflow namespace builder — constructs VFS namespaces for workflow runs.
//!
//! Each workflow run gets an isolated namespace populated with authorized
//! service mounts based on the runner's identity (Subject) and policy.
//!
//! # Pattern
//!
//! ```text
//! WorkflowService.dispatch()
//!     │
//!     └── build_workflow_namespace(subject, services)
//!             │
//!             ├── /srv/model   → RemoteMount<ModelFsAdapter>
//!             ├── /srv/tcl     → RemoteMount<TclFsAdapter>
//!             └── /srv/...     → additional service mounts
//!             │
//!             └── Namespace passed to:
//!                   ├── TclExecutor (for script eval)
//!                   └── VfsDaemon  (for worker VM virtio-fs)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::workflow::namespace::WorkflowNamespaceBuilder;
//! use hyprstream_vfs::{MountTarget, Namespace};
//!
//! let ns = WorkflowNamespaceBuilder::new()
//!     .mount_service("/srv/model", model_mount)
//!     .mount_service("/srv/tcl", tcl_mount)
//!     .build();
//!
//! // Pass to TclExecutor for script evaluation
//! let result = ns.cat("/srv/model/status", &subject).await?;
//!
//! // Or pass to VfsDaemon for worker VM access
//! let mut daemon = VfsDaemon::new(namespace_as_mount, socket_path);
//! daemon.start()?;
//! ```

use std::sync::Arc;

use hyprstream_vfs::{Mount, MountTarget, Namespace};

/// Builder for workflow-scoped VFS namespaces.
///
/// Constructs a `Namespace` with the service mounts that a workflow run
/// is authorized to access. Policy enforcement happens at build time:
/// only mounts explicitly added are available to the workflow.
///
/// This follows the Plan 9 `rfork(RFNAMEG)` pattern: the workflow gets
/// a forked namespace with only the services it needs.
pub struct WorkflowNamespaceBuilder {
    mounts: Vec<(String, MountTarget)>,
}

impl WorkflowNamespaceBuilder {
    /// Create a new builder with no mounts.
    pub fn new() -> Self {
        Self { mounts: Vec::new() }
    }

    /// Add a service mount at the given path prefix.
    ///
    /// Typically called with `/srv/{service_name}` prefixes.
    pub fn mount_service(mut self, prefix: impl Into<String>, mount: impl Mount + 'static) -> Self {
        self.mounts.push((prefix.into(), Arc::new(mount)));
        self
    }

    /// Add a pre-wrapped mount target at the given path prefix.
    pub fn mount_target(mut self, prefix: impl Into<String>, target: MountTarget) -> Self {
        self.mounts.push((prefix.into(), target));
        self
    }

    /// Build the namespace with all configured mounts.
    ///
    /// Returns `None` if any mount fails to register (should not happen
    /// with valid prefixes).
    pub fn build(self) -> Option<Namespace> {
        let mut ns = Namespace::new();
        for (prefix, target) in self.mounts {
            if ns.mount(&prefix, target).is_err() {
                return None;
            }
        }
        Some(ns)
    }
}

impl Default for WorkflowNamespaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{DirEntry, Fid, MountError, Stat};

    /// Minimal mock mount for testing.
    struct MockServiceMount {
        name: &'static str,
    }

    struct MockFid(String);

    #[async_trait]
    impl Mount for MockServiceMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(MockFid(components.join("/"))))
        }
        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        async fn read(&self, _fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let data = self.name.as_bytes();
            let start = offset as usize;
            if start >= data.len() {
                Ok(vec![])
            } else {
                Ok(data[start..].to_vec())
            }
        }
        async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(0)
        }
        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![DirEntry { name: "status".into(), is_dir: false, size: 0, stat: None }])
        }
        async fn stat(&self, _fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            Ok(Stat { qtype: 0, size: 0, name: String::new(), mtime: 0 })
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    #[tokio::test]
    async fn build_empty_namespace() {
        let ns = WorkflowNamespaceBuilder::new().build().unwrap();
        assert!(ns.mount_prefixes().is_empty());
    }

    #[tokio::test]
    async fn build_with_service_mounts() {
        let ns = WorkflowNamespaceBuilder::new()
            .mount_service("/srv/model", MockServiceMount { name: "model" })
            .mount_service("/srv/tcl", MockServiceMount { name: "tcl" })
            .build()
            .unwrap();

        let prefixes = ns.mount_prefixes();
        assert_eq!(prefixes.len(), 2);
        assert!(prefixes.contains(&"/srv/model"));
        assert!(prefixes.contains(&"/srv/tcl"));
    }

    #[tokio::test]
    async fn workflow_namespace_cat() {
        let ns = WorkflowNamespaceBuilder::new()
            .mount_service("/srv/model", MockServiceMount { name: "model" })
            .build()
            .unwrap();

        let caller = Subject::anonymous();
        let data = ns.cat("/srv/model/status", &caller).await.unwrap();
        assert_eq!(data, b"model");
    }

    #[tokio::test]
    async fn workflow_namespace_isolation() {
        // Build a namespace with only model access.
        let ns = WorkflowNamespaceBuilder::new()
            .mount_service("/srv/model", MockServiceMount { name: "model" })
            .build()
            .unwrap();

        let caller = Subject::anonymous();

        // Model is accessible.
        assert!(ns.cat("/srv/model/status", &caller).await.is_ok());

        // Tcl is NOT accessible (not mounted).
        assert!(ns.cat("/srv/tcl/eval", &caller).await.is_err());
    }

    #[tokio::test]
    async fn mount_target_variant() {
        let target: MountTarget = Arc::new(MockServiceMount { name: "shared" });
        let ns = WorkflowNamespaceBuilder::new()
            .mount_target("/srv/shared", target)
            .build()
            .unwrap();

        let caller = Subject::anonymous();
        let data = ns.cat("/srv/shared/status", &caller).await.unwrap();
        assert_eq!(data, b"shared");
    }
}
