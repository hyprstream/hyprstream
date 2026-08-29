//! `/exec` adapter root for the bounded clone-device spike.
//!
//! This maps Wanix's `#task/new/<kind>` convention (the adapter's `auto` form
//! is `#task/new/auto`) and its `cmd`/`ctl`/`fd`/`exit` files; it is not stock
//! tree identity and it does not provide a production Fersh broker.

use std::sync::Arc;

use async_trait::async_trait;
use hyprstream_rpc::AttachmentOperation;
use hyprstream_rpc_std::task::{NamespaceManifestDigest, TaskRuntimeConstraints};
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject, VfsOpContext};

use super::exec_mount::ExecMount;

/// Trusted policy result supplied to the clone device before allocation.
///
/// The device never parses manifest bytes. Its namespace digest is a
/// commitment only, not an effective/admitted `Namespace`; P12 must replace
/// this seam with derived/effective Namespace delivery.
#[derive(Clone, Debug)]
pub struct CloneAdmission {
    namespace_manifest: NamespaceManifestDigest,
    constraints: TaskRuntimeConstraints,
}

impl CloneAdmission {
    /// Construct a result selected by an injected trusted policy boundary.
    /// This is not namespace admission and does not mint Task authority.
    #[must_use]
    pub fn from_trusted_policy(
        namespace_manifest: NamespaceManifestDigest,
        constraints: TaskRuntimeConstraints,
    ) -> Self {
        Self {
            namespace_manifest,
            constraints,
        }
    }
}

/// Injection point for trusted clone policy. Identity, a 9P path, and clone
/// file bytes are deliberately not sufficient to create a policy result.
pub trait CloneAdmissionSource: Send + Sync {
    fn admit(&self, context: &VfsOpContext) -> Result<CloneAdmission, MountError>;
}

enum RootFid {
    Root,
    Clone {
        context: Option<VfsOpContext>,
        allocated: Option<String>,
    },
    Instances(Fid),
}

/// Root wrapper retaining the existing `ExecMount` API at `/exec/instances`.
pub struct ExecRootMount {
    instances: Arc<ExecMount>,
    admission: Arc<dyn CloneAdmissionSource>,
}

impl ExecRootMount {
    pub fn new(instances: Arc<ExecMount>, admission: Arc<dyn CloneAdmissionSource>) -> Self {
        Self {
            instances,
            admission,
        }
    }

    fn slice(data: &[u8], offset: u64, count: u32) -> Vec<u8> {
        let start = (offset as usize).min(data.len());
        let end = start.saturating_add(count as usize).min(data.len());
        data[start..end].to_vec()
    }

    async fn walk_inner(
        &self,
        components: &[&str],
        caller: &Subject,
        context: Option<&VfsOpContext>,
    ) -> Result<Fid, MountError> {
        match components {
            [] => Ok(Fid::new(RootFid::Root)),
            ["clone"] => Ok(Fid::new(RootFid::Clone {
                context: context.cloned(),
                allocated: None,
            })),
            ["instances", rest @ ..] => {
                let inner = match context {
                    Some(context) => self.instances.walk_with_context(rest, context).await?,
                    None => self.instances.walk(rest, caller).await?,
                };
                Ok(Fid::new(RootFid::Instances(inner)))
            }
            _ => Err(MountError::NotFound(components.join("/"))),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Mount for ExecRootMount {
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError> {
        self.walk_inner(components, caller, None).await
    }

    async fn walk_with_context(
        &self,
        components: &[&str],
        context: &VfsOpContext,
    ) -> Result<Fid, MountError> {
        self.walk_inner(components, context.subject(), Some(context))
            .await
    }

    async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError> {
        let root = fid
            .downcast_mut::<RootFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad exec root fid".into()))?;
        match root {
            RootFid::Clone { context, allocated } => {
                if mode != hyprstream_vfs::OREAD && mode != hyprstream_vfs::ORDWR {
                    return Err(MountError::PermissionDenied(
                        "clone must be opened for read".into(),
                    ));
                }
                let context = context.as_ref().ok_or_else(|| {
                    MountError::PermissionDenied(
                        "clone requires a verified attachment context".into(),
                    )
                })?;
                context.ensure_operation(AttachmentOperation::TaskSpawn)?;
                if allocated.is_none() {
                    let admission = self.admission.admit(context)?;
                    context.ensure_operation(AttachmentOperation::TaskSpawn)?;
                    *allocated = Some(
                        self.instances
                            .allocate_pending_task(
                                context,
                                admission.namespace_manifest,
                                admission.constraints,
                            )
                            .await?,
                    );
                }
                Ok(())
            }
            RootFid::Instances(inner) => self.instances.open(inner, mode, caller).await,
            RootFid::Root => Err(MountError::IsDirectory("use readdir".into())),
        }
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let root = fid
            .downcast_ref::<RootFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad exec root fid".into()))?;
        match root {
            RootFid::Clone {
                context: Some(context),
                allocated: Some(id),
            } => {
                // A clone id is non-bearer, but a revoked attachment still
                // may not continue the device conversation after allocation.
                context.ensure_operation(AttachmentOperation::TaskSpawn)?;
                Ok(Self::slice(format!("{id}\n").as_bytes(), offset, count))
            }
            RootFid::Clone {
                context: None,
                allocated: Some(_),
            } => Err(MountError::PermissionDenied(
                "clone id requires a verified attachment context".into(),
            )),
            RootFid::Clone {
                allocated: None, ..
            } => Err(MountError::InvalidArgument(
                "clone has not been opened".into(),
            )),
            RootFid::Instances(inner) => self.instances.read(inner, offset, count, caller).await,
            RootFid::Root => Err(MountError::IsDirectory("use readdir".into())),
        }
    }

    async fn write(
        &self,
        fid: &Fid,
        offset: u64,
        data: &[u8],
        caller: &Subject,
    ) -> Result<u32, MountError> {
        let root = fid
            .downcast_ref::<RootFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad exec root fid".into()))?;
        match root {
            RootFid::Instances(inner) => self.instances.write(inner, offset, data, caller).await,
            RootFid::Clone { .. } => Err(MountError::NotSupported(
                "clone input is not an admission channel".into(),
            )),
            RootFid::Root => Err(MountError::IsDirectory("use instances or clone".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let root = fid
            .downcast_ref::<RootFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad exec root fid".into()))?;
        match root {
            RootFid::Root => Ok(vec![
                DirEntry {
                    name: "clone".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "instances".into(),
                    is_dir: true,
                    size: 0,
                    stat: None,
                },
            ]),
            RootFid::Instances(inner) => self.instances.readdir(inner, caller).await,
            RootFid::Clone { .. } => Err(MountError::NotDirectory("clone".into())),
        }
    }

    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError> {
        let root = fid
            .downcast_ref::<RootFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad exec root fid".into()))?;
        match root {
            RootFid::Root => Ok(Stat::unknown_qid(0x80, 0, "exec".into(), 0)),
            RootFid::Clone { .. } => Ok(Stat::unknown_qid(0, 0, "clone".into(), 0)),
            RootFid::Instances(inner) => self.instances.stat(inner, caller).await,
        }
    }

    async fn clunk(&self, fid: Fid, caller: &Subject) {
        if let Ok(RootFid::Instances(inner)) = fid.downcast_into::<RootFid>() {
            self.instances.clunk(inner, caller).await;
        }
    }
}
