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
use hyprstream_rpc::rpc_client::RpcClient;

// ============================================================================
// RpcClient is already Send + Sync — no wrapper needed.
// ============================================================================

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

/// Result from service dispatch — normal response or stream setup.
pub enum ServiceDispatchResult {
    /// Normal JSON response string.
    Response(String),
    /// Streaming — JSON string of parsed StreamInfo + ephemeral keypair for ECDH.
    Stream {
        json: String,
        /// 64 bytes: [secret(32) | pubkey(32)]
        ephemeral_keypair: Vec<u8>,
    },
}

/// Trait for service-specific dispatch. Implemented via macro from generated code.
#[async_trait(?Send)]
trait ServiceDispatch: Send + Sync {
    async fn dispatch(&self, method: &str, args_json: &str, client: &dyn RpcClient) -> Result<ServiceDispatchResult, String>;
    fn metadata(&self) -> (&'static str, &'static [hyprstream_rpc::metadata::MethodMeta]);
}

/// Generic VFS mount driven by a ServiceDispatch implementation.
pub struct GenericServiceMount {
    client: Arc<dyn RpcClient>,
    service: Box<dyn ServiceDispatch>,
    next_id: IdCounter,
    ctl_response: CtlResponseCache,
    stream_registry: std::sync::Arc<crate::stream_mount::StreamRegistry>,
}

impl GenericServiceMount {
    pub fn new(
        client: Arc<dyn RpcClient>,
        service: Box<dyn ServiceDispatch>,
        stream_registry: std::sync::Arc<crate::stream_mount::StreamRegistry>,
    ) -> Self {
        Self {
            client,
            service,
            next_id: IdCounter::new(),
            ctl_response: CtlResponseCache::new(),
            stream_registry,
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
            // After ctl write→read, second read returns empty (EOF)
            // For plain cat on root, this is also correct (it's a directory)
            return Ok(Vec::new());
        }

        // cat /srv/{service}/{method} → dispatch query method with empty args
        let method = &state.path[0];
        match self.service.dispatch(method, "{}", self.client.as_ref()).await
            .map_err(|e| MountError::Io(e))? {
            ServiceDispatchResult::Response(json) => {
                Ok(slice_read(json.into_bytes(), offset, count))
            }
            ServiceDispatchResult::Stream { .. } => {
                Err(MountError::NotSupported("use ctl to start streams, then read from /stream/{topic}/data".into()))
            }
        }
    }

    async fn write(&self, fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        // ctl pattern: data = "command {json_args}" or "command json_args"
        // Tcl brace quoting strips outer {} so we may receive bare JSON fields.
        let data_str = std::str::from_utf8(data).unwrap_or("").trim();

        // Strip shell quotes from around JSON if present
        let stripped = data_str
            .trim_end_matches('\'')
            .trim_end_matches('"');

        let (cmd, args_owned);
        if let Some(brace) = stripped.find('{') {
            let cmd_part = stripped[..brace].trim().trim_end_matches('\'').trim_end_matches('"').trim();
            let args_part = &stripped[brace..];
            cmd = if cmd_part.is_empty() {
                state.path.first().map(|s| s.as_str()).unwrap_or("")
            } else {
                cmd_part
            };
            args_owned = args_part.to_owned();
        } else if stripped.contains(':') && stripped.contains('"') {
            // Looks like stripped JSON (Tcl brace quoting removed outer {})
            // Split on first whitespace to find command, rest is bare JSON
            let (c, rest) = stripped.split_once(char::is_whitespace).unwrap_or((stripped, ""));
            cmd = c.trim();
            // Re-wrap bare JSON fields with braces
            args_owned = format!("{{{}}}", rest.trim());
        } else {
            cmd = if stripped.is_empty() {
                state.path.first().map(|s| s.as_str()).unwrap_or("")
            } else {
                stripped
            };
            args_owned = "{}".to_owned();
        };
        let args_str = args_owned.as_str();

        let dispatch_result = self.service.dispatch(cmd, args_str, self.client.as_ref()).await
            .map_err(|e| MountError::Io(e))?;

        let resp = match dispatch_result {
            ServiceDispatchResult::Response(json) => json.into_bytes(),
            ServiceDispatchResult::Stream { json: stream_json, ephemeral_keypair } => {
                // Streaming response — use open_stream() to set up verified subscription
                let parsed: serde_json::Value = serde_json::from_str(&stream_json)
                    .map_err(|e| MountError::Io(format!("parse stream response: {e}")))?;

                // The dispatch already sent the streaming request and got StreamInfo back.
                // Now we need to open a stream handle. But open_stream() does the full
                // flow (send + ECDH + subscribe). Since dispatch already sent the request,
                // we need to do the ECDH + subscribe part manually using the StreamInfo.
                //
                // TODO: Refactor dispatch to return raw payload bytes so open_stream()
                // can handle the full flow. For now, use StreamHandle::open() directly.
                let inner = if let Some(obj) = parsed.as_object() {
                    if obj.len() == 1 {
                        obj.values().next().unwrap().clone()
                    } else {
                        parsed.clone()
                    }
                } else {
                    parsed.clone()
                };
                let info: hyprstream_rpc::stream_info::StreamInfo = serde_json::from_value(inner)
                    .map_err(|e| MountError::Io(format!("parse StreamInfo: {e}")))?;

                let client_secret = &ephemeral_keypair[..32];
                let client_pubkey = &ephemeral_keypair[32..64];
                let mut secret_32 = [0u8; 32];
                let mut pubkey_32 = [0u8; 32];
                secret_32.copy_from_slice(client_secret);
                pubkey_32.copy_from_slice(client_pubkey);

                // Open verified stream handle via RpcClient
                let handle = self.client.open_stream_from_info(info.clone(), secret_32, pubkey_32)
                    .await
                    .map_err(|e| MountError::Io(format!("open stream: {e}")))?;

                let topic = handle.stream_id().to_owned();

                self.stream_registry.register(topic.clone(), crate::stream_mount::StreamEntry {
                    handle: Some(handle),
                    owner: _caller.name().unwrap_or("anonymous").to_owned(),
                    bytes_received: 0,
                    blocks_received: 0,
                });

                let result = serde_json::json!({
                    "streamId": info.stream_id,
                    "topic": topic,
                });
                serde_json::to_string(&result).unwrap_or_default().into_bytes()
            }
        };
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
            async fn dispatch(&self, method: &str, args_json: &str, client: &dyn RpcClient) -> Result<ServiceDispatchResult, String> {
                use $mod as svc;

                // Generate ephemeral keypair upfront — used by streaming methods,
                // ignored by non-streaming (the send closure decides which transport to use).
                let keypair = hyprstream_rpc::wasm_api::generate_ephemeral_keypair()
                    .map_err(|e| format!("keypair: {e:?}"))?;
                let ephemeral_pubkey = keypair[32..64].to_vec();

                let result = svc::dispatch(method, args_json, 0, |payload: Vec<u8>| async move {
                    // Use call_streaming which injects ephemeral_pubkey into the
                    // RequestEnvelope. For non-streaming methods, the server ignores
                    // the pubkey — it only matters for streaming handshakes.
                    client.call_streaming(payload, {
                        let mut epk = [0u8; 32];
                        epk.copy_from_slice(&ephemeral_pubkey);
                        epk
                    }).await
                        .map_err(|e| format!("{e:?}"))
                }).await?;

                Ok(match result {
                    svc::DispatchResult::Response(json) => ServiceDispatchResult::Response(json),
                    svc::DispatchResult::Stream(json) => ServiceDispatchResult::Stream {
                        json,
                        ephemeral_keypair: keypair,
                    },
                })
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
    registry_client: Arc<dyn RpcClient>,
    model_client: Arc<dyn RpcClient>,
) -> (hyprstream_vfs::Namespace, Arc<crate::stream_mount::StreamRegistry>) {
    let stream_registry = Arc::new(crate::stream_mount::StreamRegistry::new());
    let mut ns = hyprstream_vfs::Namespace::new();

    // Service mounts — all use GenericServiceMount with generated dispatch
    ns.mount("/srv/registry", Arc::new(GenericServiceMount::new(
        Arc::clone(&registry_client), Box::new(RegistryDispatch), Arc::clone(&stream_registry),
    ))).expect("mount /srv/registry");
    ns.mount("/srv/model", Arc::new(GenericServiceMount::new(
        Arc::clone(&model_client), Box::new(ModelDispatch), Arc::clone(&stream_registry),
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

    // Stream mount — named pipes for active streaming data
    ns.mount("/stream", Arc::new(crate::stream_mount::StreamMount::new(
        Arc::clone(&stream_registry),
    ))).expect("mount /stream");

    (ns, stream_registry)
}
