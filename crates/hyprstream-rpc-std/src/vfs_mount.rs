//! Generic VFS Mount driven by codegen — no per-service manual wrappers.
//!
//! `ServiceMount` uses the proc-macro-generated `dispatch()`, `schema_metadata()`,
//! and `render_doc()` functions to serve any service through the VFS. All
//! serialization, method routing, and documentation are codegen-driven.
//!
//! Mount points:
//!   /srv/{service}     → ServiceMount (ctl for mutations, cat for queries)
//!   /srv/{service}/doc → DocMount (man pages from schema annotations)

#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;

use crate::wasm_exports::RpcSession;

// ============================================================================
// Send+Sync wrapper for RpcSession on wasm32
// ============================================================================

struct SessionHandle(Arc<RpcSession>);

// SAFETY: wasm32 is single-threaded — no actual cross-thread sharing occurs.
unsafe impl Send for SessionHandle {}
unsafe impl Sync for SessionHandle {}

impl SessionHandle {
    fn session(&self) -> &RpcSession {
        &self.0
    }
}

impl Clone for SessionHandle {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

// ============================================================================
// Fid state
// ============================================================================

#[derive(Clone, Debug)]
struct VfsFidState {
    path: Vec<String>,
    opened: bool,
}

/// Apply offset+count slicing to a read result (9P semantics).
fn slice_read(data: Vec<u8>, offset: u64, count: u32) -> Vec<u8> {
    let start = (offset as usize).min(data.len());
    let end = (start + count as usize).min(data.len());
    data[start..end].to_vec()
}

// ============================================================================
// CtlResponseCache — stores write→read response for ctl pattern
// ============================================================================

struct CtlResponseCache(std::cell::RefCell<Option<Vec<u8>>>);

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for CtlResponseCache {}
unsafe impl Sync for CtlResponseCache {}

impl CtlResponseCache {
    fn new() -> Self {
        Self(std::cell::RefCell::new(None))
    }
    fn take(&self) -> Option<Vec<u8>> {
        self.0.borrow_mut().take()
    }
    fn set(&self, data: Vec<u8>) {
        *self.0.borrow_mut() = Some(data);
    }
}

/// Atomic-like counter for request IDs, Send+Sync on wasm32.
struct IdCounter(std::cell::Cell<u64>);
unsafe impl Send for IdCounter {}
unsafe impl Sync for IdCounter {}
impl IdCounter {
    fn new() -> Self { Self(std::cell::Cell::new(1)) }
    fn next(&self) -> u64 {
        let id = self.0.get();
        self.0.set(id + 1);
        id
    }
}

// ============================================================================
// GenericServiceMount — codegen-driven mount for any service
// ============================================================================

/// Trait for service-specific dispatch. Implemented via macro from generated code.
#[async_trait(?Send)]
trait ServiceDispatch: Send + Sync {
    async fn dispatch(&self, method: &str, args_json: &str, session: &RpcSession) -> Result<String, String>;
    fn metadata(&self) -> (&'static str, &'static [hyprstream_rpc::metadata::MethodMeta]);
}

/// Generic VFS mount driven by a ServiceDispatch implementation.
pub struct GenericServiceMount {
    session: SessionHandle,
    service: Box<dyn ServiceDispatch>,
    next_id: IdCounter,
    ctl_response: CtlResponseCache,
}

impl GenericServiceMount {
    pub fn new(session: Arc<RpcSession>, service: Box<dyn ServiceDispatch>) -> Self {
        Self {
            session: SessionHandle(session),
            service,
            next_id: IdCounter::new(),
            ctl_response: CtlResponseCache::new(),
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.next()
    }
}

#[async_trait(?Send)]
impl Mount for GenericServiceMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        Ok(Fid::new(VfsFidState {
            path: components.iter().map(|s| s.to_string()).collect(),
            opened: false,
        }))
    }

    async fn open(&self, fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        if let Some(state) = fid.downcast_mut::<VfsFidState>() {
            state.opened = true;
        }
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        // Check for ctl response first (from previous write)
        if let Some(resp) = self.ctl_response.take() {
            return Ok(slice_read(resp, offset, count));
        }

        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        if state.path.is_empty() {
            return Err(MountError::IsDirectory("service root is a directory".into()));
        }

        // cat /srv/{service}/{method} → dispatch query method with empty args
        let method = &state.path[0];
        let data = self.service.dispatch(method, "{}", self.session.session()).await
            .map_err(|e| MountError::Io(e))?;
        Ok(slice_read(data.into_bytes(), offset, count))
    }

    async fn write(&self, fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        // ctl pattern: data = "command {json_args}" or path contains command
        let data_str = std::str::from_utf8(data).unwrap_or("").trim();

        let (cmd, args_str) = if let Some(brace) = data_str.find('{') {
            let cmd_part = data_str[..brace].trim();
            let args_part = &data_str[brace..];
            if cmd_part.is_empty() {
                (state.path.first().map(|s| s.as_str()).unwrap_or(""), args_part)
            } else {
                (cmd_part, args_part)
            }
        } else {
            // No JSON — might be a simple command name
            let cmd = if data_str.is_empty() {
                state.path.first().map(|s| s.as_str()).unwrap_or("")
            } else {
                data_str
            };
            (cmd, "{}")
        };

        let result = self.service.dispatch(cmd, args_str, self.session.session()).await
            .map_err(|e| MountError::Io(e))?;
        let resp = result.into_bytes();
        let len = resp.len() as u32;
        self.ctl_response.set(resp);
        Ok(len)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        if state.path.is_empty() {
            // Root: list methods from metadata
            let (_, methods) = self.service.metadata();
            Ok(methods
                .iter()
                .filter(|m| !m.hidden)
                .map(|m| DirEntry {
                    name: m.name.to_owned(),
                    is_dir: m.is_scoped,
                    size: 0,
                    stat: None,
                })
                .collect())
        } else {
            Err(MountError::NotDirectory(state.path.join("/")))
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let (svc_name, _) = self.service.metadata();
        let name = state.path.last().map(|s| s.as_str()).unwrap_or(svc_name);
        Ok(Stat {
            qtype: if state.path.is_empty() { 0x80 } else { 0 },
            size: 0,
            name: name.to_string(),
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ============================================================================
// DocMount — serves man pages from compiled schema metadata
// ============================================================================

pub struct DocMount {
    render: fn(&[&str]) -> Option<String>,
    metadata: fn() -> (&'static str, &'static [hyprstream_rpc::metadata::MethodMeta]),
}

impl DocMount {
    pub fn new(
        render: fn(&[&str]) -> Option<String>,
        metadata: fn() -> (&'static str, &'static [hyprstream_rpc::metadata::MethodMeta]),
    ) -> Self {
        Self { render, metadata }
    }
}

#[async_trait(?Send)]
impl Mount for DocMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        Ok(Fid::new(VfsFidState {
            path: components.iter().map(|s| s.to_string()).collect(),
            opened: false,
        }))
    }

    async fn open(&self, fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        if let Some(state) = fid.downcast_mut::<VfsFidState>() {
            state.opened = true;
        }
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let path_refs: Vec<&str> = state.path.iter().map(|s| s.as_str()).collect();
        match (self.render)(&path_refs) {
            Some(text) => {
                let bytes = text.into_bytes();
                Ok(slice_read(bytes, offset, count))
            }
            None => Err(MountError::NotFound(state.path.join("/"))),
        }
    }

    async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        Err(MountError::NotSupported("docs are read-only".into()))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        if state.path.is_empty() {
            let (_, methods) = (self.metadata)();
            Ok(methods
                .iter()
                .filter(|m| !m.hidden)
                .map(|m| DirEntry {
                    name: m.name.to_owned(),
                    is_dir: m.is_scoped,
                    size: 0,
                    stat: None,
                })
                .collect())
        } else {
            Err(MountError::NotDirectory(state.path.join("/")))
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let name = state.path.last().map(|s| s.as_str()).unwrap_or("doc");
        Ok(Stat {
            qtype: if state.path.is_empty() { 0x80 } else { 0 },
            size: 0,
            name: name.to_string(),
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ============================================================================
// Per-service dispatch wrappers — thin adapters calling generated dispatch()
// ============================================================================

macro_rules! impl_service_dispatch {
    ($name:ident, $mod:path) => {
        struct $name;

        #[async_trait(?Send)]
        impl ServiceDispatch for $name {
            async fn dispatch(&self, method: &str, args_json: &str, session: &RpcSession) -> Result<String, String> {
                use $mod as svc;
                svc::dispatch(method, args_json, 0, |payload: Vec<u8>| async move {
                    // session.send() handles signing + envelope unwrapping
                    session.send(&payload).await
                        .map(|v| v)
                        .map_err(|e| format!("{e:?}"))
                }).await
            }

            fn metadata(&self) -> (&'static str, &'static [hyprstream_rpc::metadata::MethodMeta]) {
                use $mod as svc;
                svc::schema_metadata()
            }
        }
    };
}

impl_service_dispatch!(RegistryDispatch, crate::registry_client);
impl_service_dispatch!(ModelDispatch, crate::model_client);
impl_service_dispatch!(PolicyDispatch, crate::policy_client);
impl_service_dispatch!(McpDispatch, crate::mcp_client);
impl_service_dispatch!(InferenceDispatch, crate::inference_client);

// ============================================================================
// Builder — construct a Namespace with all service mounts
// ============================================================================

/// Build a VFS namespace with codegen-driven service mounts.
pub fn build_browser_namespace(
    registry_session: Arc<RpcSession>,
    model_session: Arc<RpcSession>,
) -> hyprstream_vfs::Namespace {
    let mut ns = hyprstream_vfs::Namespace::new();

    // Service mounts — all use GenericServiceMount with generated dispatch
    ns.mount("/srv/registry", Arc::new(GenericServiceMount::new(
        Arc::clone(&registry_session), Box::new(RegistryDispatch),
    ))).expect("mount /srv/registry");
    ns.mount("/srv/model", Arc::new(GenericServiceMount::new(
        Arc::clone(&model_session), Box::new(ModelDispatch),
    ))).expect("mount /srv/model");

    // Documentation mounts — generated at compile time from schema annotations
    ns.mount("/srv/registry/doc", Arc::new(DocMount::new(
        crate::registry_client::render_doc,
        crate::registry_client::schema_metadata,
    ))).expect("mount /srv/registry/doc");
    ns.mount("/srv/model/doc", Arc::new(DocMount::new(
        crate::model_client::render_doc,
        crate::model_client::schema_metadata,
    ))).expect("mount /srv/model/doc");
    ns.mount("/srv/inference/doc", Arc::new(DocMount::new(
        crate::inference_client::render_doc,
        crate::inference_client::schema_metadata,
    ))).expect("mount /srv/inference/doc");
    ns.mount("/srv/policy/doc", Arc::new(DocMount::new(
        crate::policy_client::render_doc,
        crate::policy_client::schema_metadata,
    ))).expect("mount /srv/policy/doc");
    ns.mount("/srv/mcp/doc", Arc::new(DocMount::new(
        crate::mcp_client::render_doc,
        crate::mcp_client::schema_metadata,
    ))).expect("mount /srv/mcp/doc");

    ns
}
