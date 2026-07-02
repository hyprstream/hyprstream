//! Generic VFS Mount driven by codegen — no per-service manual wrappers.
//!
//! `ServiceMount` uses the proc-macro-generated `dispatch()`, `schema_metadata()`,
//! and `render_doc()` functions to serve any service through the VFS. All
//! serialization, method routing, and documentation are codegen-driven.
//!
//! Mount points:
//!   /srv/{service}     → ServiceMount (ctl for mutations, cat for queries)
//!   /srv/{service}/doc → DocMount (man pages from schema annotations)
//!   /wanix             → WanixMount (Wanix-native VFS via 9P2000.L over DMA;
//!                        optional — mounted only when running under Wanix,
//!                        see [`mount_wanix`]). #409/#391.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

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

/// Generate an ephemeral ECDH keypair as 64 bytes: `[secret_scalar(32) | pubkey(32)]`.
///
/// Same layout the streaming handshake expects (secret first, pubkey at `[32..64]`),
/// built directly from `DefaultKeyExchange` so it works on native and wasm alike.
fn ephemeral_keypair_64() -> Result<Vec<u8>, String> {
    use hyprstream_rpc::crypto::key_exchange::DefaultKeyExchange;
    use hyprstream_rpc::crypto::KeyExchange;

    let (secret, public) = DefaultKeyExchange::generate_keypair();
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(secret.scalar().as_bytes());
    out.extend_from_slice(&DefaultKeyExchange::pubkey_to_bytes(&public));
    if out.len() != 64 {
        return Err(format!("ephemeral keypair wrong length: {}", out.len()));
    }
    Ok(out)
}

// ============================================================================
// CtlResponseCache — stores write→read response for ctl pattern
// ============================================================================

struct CtlResponseCache(Mutex<Option<Vec<u8>>>);

impl CtlResponseCache {
    fn new() -> Self {
        Self(Mutex::new(None))
    }
    fn take(&self) -> Option<Vec<u8>> {
        self.0.lock().take()
    }
    fn set(&self, data: Vec<u8>) {
        *self.0.lock() = Some(data);
    }
}

/// Monotonic counter for request IDs.
struct IdCounter(AtomicU64);
impl IdCounter {
    fn new() -> Self { Self(AtomicU64::new(1)) }
    fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

// ============================================================================
// GenericServiceMount — codegen-driven mount for any service
// ============================================================================

/// Result from service dispatch — normal response or stream setup.
pub enum ServiceDispatchResult {
    /// Normal JSON response string.
    Response(String),
    /// Streaming — the VERIFIED-capnp `StreamInfo` library type (decoded +
    /// COSE-verified upstream, carried typed — NOT a JSON string to re-parse,
    /// #468) + ephemeral keypair for ECDH.
    Stream {
        info: hyprstream_rpc::stream_info::StreamInfo,
        /// 64 bytes: [secret(32) | pubkey(32)]
        ephemeral_keypair: Vec<u8>,
    },
}

/// Trait for service-specific dispatch. Implemented via macro from generated code.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait ServiceDispatch: Send + Sync {
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

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
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
            .map_err(MountError::Io)? {
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
            .map_err(MountError::Io)?;

        let resp = match dispatch_result {
            ServiceDispatchResult::Response(json) => json.into_bytes(),
            ServiceDispatchResult::Stream { info, ephemeral_keypair } => {
                // #468: `info` is the VERIFIED-capnp StreamInfo library type carried
                // typed through dispatch (decoded + COSE-verified upstream) — no
                // serde_json round-trip / re-parse at this boundary.
                //
                // The dispatch already sent the streaming request and got StreamInfo back.
                // Now we open a stream handle. open_stream() would do the full flow
                // (send + ECDH + subscribe), but dispatch already sent the request, so we
                // do the ECDH + subscribe part here using the verified StreamInfo.
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
        Ok(Stat::unknown_qid(
            if state.path.is_empty() { 0x80 } else { 0 },
            0,
            name.to_string(),
            0,
        ))
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

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
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
        Ok(Stat::unknown_qid(
            if state.path.is_empty() { 0x80 } else { 0 },
            0,
            name.to_string(),
            0,
        ))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ============================================================================
// Per-service dispatch wrappers — thin adapters calling generated dispatch()
// ============================================================================

// ============================================================================
// Auto-mount registry (#543, T4) — service → /srv/{name} mount, no manual wiring
// ============================================================================

/// Resolve an `Arc<dyn RpcClient>` for a service by name. Owns the signing key +
/// destination-key (trust store) resolution — kept as a boxed closure so
/// `hyprstream-rpc-std` stays free of the identity/config types.
#[cfg(not(target_arch = "wasm32"))]
type ClientDialer = Box<dyn Fn(&str) -> anyhow::Result<Arc<dyn RpcClient>> + Send + Sync>;

/// Context threaded into a [`MountFactory`] to build a service's mount.
///
/// A `MountFactory::construct` is a plain `fn` pointer (inventory constraint) and
/// therefore cannot capture the per-process transport/identity state a
/// [`GenericServiceMount`] needs — an `Arc<dyn RpcClient>` and the shared
/// [`StreamRegistry`](crate::stream_mount::StreamRegistry). `MountContext`
/// carries that state, mirroring how [`ServiceFactory`]'s `fn(&ServiceContext)`
/// threads spawn-time state. It resolves an `Arc<dyn RpcClient>` per service by
/// name via the caller-supplied `dial` closure (which owns the signing key +
/// trust-store lookup), so this crate stays free of identity/config types.
#[cfg(not(target_arch = "wasm32"))]
pub struct MountContext {
    dial: ClientDialer,
    /// Shared stream registry — all service mounts in one namespace share it so
    /// `/stream` named pipes are visible across services.
    stream_registry: Arc<crate::stream_mount::StreamRegistry>,
}

#[cfg(not(target_arch = "wasm32"))]
impl MountContext {
    /// Build a context from a per-service client dialer. The `stream_registry`
    /// is created fresh; use [`with_stream_registry`](Self::with_stream_registry)
    /// to share one across builders.
    pub fn new(
        dial: impl Fn(&str) -> anyhow::Result<Arc<dyn RpcClient>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            dial: Box::new(dial),
            stream_registry: Arc::new(crate::stream_mount::StreamRegistry::new()),
        }
    }

    /// Use an existing shared [`StreamRegistry`] instead of a fresh one.
    pub fn with_stream_registry(
        mut self,
        stream_registry: Arc<crate::stream_mount::StreamRegistry>,
    ) -> Self {
        self.stream_registry = stream_registry;
        self
    }

    /// The shared stream registry these mounts publish into.
    pub fn stream_registry(&self) -> &Arc<crate::stream_mount::StreamRegistry> {
        &self.stream_registry
    }

    /// Resolve an `Arc<dyn RpcClient>` for `service`.
    pub fn dial(&self, service: &str) -> anyhow::Result<Arc<dyn RpcClient>> {
        (self.dial)(service)
    }
}

/// Inventory entry that registers a service's generated mount.
///
/// Emitted by [`impl_service_dispatch!`] alongside the per-service dispatch
/// wrapper (the only site that knows both the service name and the concrete
/// `ServiceDispatch` type). Mirrors [`ServiceFactory`] for the mount surface:
/// implement the dispatch, get a `/srv/{name}` mount for free.
#[cfg(not(target_arch = "wasm32"))]
pub struct MountFactory {
    /// Service name; the default mount point is `/srv/{name}` (advisory
    /// convention — the auto-mount loop's prefix is overridable by the caller's
    /// later binds, not a contract).
    pub name: &'static str,
    /// Construct the service's `Mount` from a [`MountContext`].
    pub construct: fn(&MountContext) -> anyhow::Result<Arc<dyn Mount>>,
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(MountFactory);

/// Mount every inventory-registered service into `ns` at `/srv/{name}`.
///
/// This is the auto-wire path (#543): a service that `impl_service_dispatch!`s
/// gets a `/srv/{name}` mount with no per-service code here. `/srv/{name}` is
/// advisory convention; a caller may `bind_mount` a different prefix before or
/// after. A service whose `construct` fails (e.g. its transport can't be
/// resolved yet) is logged and skipped — one unavailable service does not block
/// the others.
///
/// Mounts land in `ns` directly; because [`Namespace::fork`] clones the mount
/// table by `Arc`, a namespace forked after this call inherits every auto-mount
/// (per-sandbox isolation is by *forking* the namespace, FS-D #365 — not by
/// re-running this loop).
#[cfg(not(target_arch = "wasm32"))]
pub fn mount_all_services(
    ns: &mut hyprstream_vfs::Namespace,
    ctx: &MountContext,
) -> usize {
    let mut mounted = 0;
    for factory in inventory::iter::<MountFactory> {
        match (factory.construct)(ctx) {
            Ok(mount) => {
                let prefix = format!("/srv/{}", factory.name);
                if let Err(e) = ns.mount(&prefix, mount) {
                    tracing::warn!(service = factory.name, %prefix, error = %e, "auto-mount failed");
                } else {
                    mounted += 1;
                }
            }
            Err(e) => {
                tracing::debug!(service = factory.name, error = %e, "service mount unavailable; skipping");
            }
        }
    }
    mounted
}

macro_rules! impl_service_dispatch {
    ($name:ident, $svc_name:literal, $mod:path) => {
        struct $name;

        // Auto-register a `/srv/{svc_name}` mount for this service (#543, T4).
        // This is the only site with both the service name and the concrete
        // dispatch type in scope, so the `MountFactory` constructor is emitted
        // here. `$svc_name` is the same name the service's `schema_metadata()`
        // reports and matches the `#[service_factory]` registration.
        #[cfg(not(target_arch = "wasm32"))]
        inventory::submit! {
            crate::vfs_mount::MountFactory {
                name: $svc_name,
                construct: |ctx: &crate::vfs_mount::MountContext| -> anyhow::Result<std::sync::Arc<dyn Mount>> {
                    let client = ctx.dial($svc_name)?;
                    Ok(std::sync::Arc::new(GenericServiceMount::new(
                        client,
                        Box::new($name),
                        std::sync::Arc::clone(ctx.stream_registry()),
                    )))
                },
            }
        }

        #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
        impl ServiceDispatch for $name {
            async fn dispatch(&self, method: &str, args_json: &str, client: &dyn RpcClient) -> Result<ServiceDispatchResult, String> {
                use $mod as svc;

                // Generate ephemeral keypair upfront — used by streaming methods,
                // ignored by non-streaming (the send closure decides which transport to use).
                let keypair = crate::vfs_mount::ephemeral_keypair_64()
                    .map_err(|e| format!("keypair: {e}"))?;
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
                    svc::DispatchResult::Stream(info) => ServiceDispatchResult::Stream {
                        info,
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

impl_service_dispatch!(RegistryDispatch, "registry", crate::registry_client);
impl_service_dispatch!(ModelDispatch, "model", crate::model_client);
impl_service_dispatch!(PolicyDispatch, "policy", crate::policy_client);
impl_service_dispatch!(McpDispatch, "mcp", crate::mcp_client);
impl_service_dispatch!(InferenceDispatch, "inference", crate::inference_client);

// ============================================================================
// Builder — construct a Namespace with all service mounts
// ============================================================================

/// Build a VFS namespace with codegen-driven service mounts.
///
/// Per #389 + #391 (Option 1: shared content model), this builder mounts the
/// same content trees as the native namespace builder
/// (`hyprstream::cli::shell_handlers`) so `/srv/registry` and `/srv/model`
/// resolve to the same spine-backed content in either context. The transport
/// leaf (DMA/SAB ring buffers here vs ZMQ in the native builder) is
/// correctly-scoped glue and is NOT part of the convergence contract — see
/// `hyprstream_vfs::STANDARD_NAMESPACE_PATHS`.
///
/// Note: `/worktree` is bind-mounted to `/srv/registry` for path-shape
/// parity with the native namespace. In the browser this exposes the
/// GenericServiceMount (ctl-style service-as-files: `ls`, `cat`, `ctl`);
/// in the native namespace the same path exposes the worktree filesystem
/// (`RemoteRegistryMount`, real qids). The convergence is at the path +
/// backing-service level; the access style differs by transport capability.
///
/// Browser-only: it wires the wasm transport clients into the browser namespace.
/// The `GenericServiceMount`/`DocMount`/`StreamMount` building blocks it uses are
/// target-agnostic; native callers compose them through the standard namespace
/// builder + `#[service_factory]` auto-mount (#543) instead.
#[cfg(target_arch = "wasm32")]
pub fn build_browser_namespace(
    registry_client: Arc<dyn RpcClient>,
    model_client: Arc<dyn RpcClient>,
) -> (hyprstream_vfs::Namespace, Arc<crate::stream_mount::StreamRegistry>) {
    let stream_registry = Arc::new(crate::stream_mount::StreamRegistry::new());
    let mut ns = hyprstream_vfs::Namespace::new();

    // Service mounts — all use GenericServiceMount with generated dispatch
    let registry_mount: Arc<GenericServiceMount> = Arc::new(GenericServiceMount::new(
        Arc::clone(&registry_client), Box::new(RegistryDispatch), Arc::clone(&stream_registry),
    ));
    ns.mount("/srv/registry", registry_mount.clone()).expect("mount /srv/registry");
    // `/worktree` aliases `/srv/registry` for path-shape parity with the
    // native namespace (see `hyprstream_vfs::STANDARD_NAMESPACE_PATHS`).
    ns.bind_mount("/worktree", registry_mount, hyprstream_vfs::BindFlag::After)
        .expect("bind mount /worktree");
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

    // `/wanix` is NOT mounted here: it needs a Wanix-provided SharedArrayBuffer
    // that the browser only has when running under Wanix. Wire it separately
    // via [`mount_wanix`] after this builder returns. See #409/#391.

    (ns, stream_registry)
}

// ============================================================================
// WanixMount wiring — #409/#391 (Wanix-native VFS access)
// ============================================================================

/// Mount the Wanix 9P filesystem into the namespace at `/wanix` (#409/#391).
///
/// `client` is a `P9Client` already connected over a `P9Transport` (e.g. the
/// `DmaTransport` SharedArrayBuffer bridge). Every VFS op under `/wanix` is
/// translated to 9P2000.L and forwarded — the Wanix half of the "browser
/// both-paths" model, riding the same wasm transport substrate as the service
/// mounts above.
///
/// This is the rpc-std seam that makes `WanixMount` reachable from the browser
/// namespace now that `hyprstream-9p` is a wasm32 dependency.
///
/// # Breaking change (#465)
///
/// This signature replaces the previous `async fn mount_wanix(ns, sab, uname,
/// aname) -> anyhow::Result<()>`, which performed the 9P version/attach handshake
/// internally over a `DmaTransport` built from a `SharedArrayBuffer`. The handshake
/// (and transport construction) now happen at the call site, so callers pass an
/// already-connected `P9Client<T>` and this function is **synchronous**, returning
/// [`hyprstream_vfs::NamespaceError`] instead of `anyhow::Result<()>`. Update call
/// sites: `mount_wanix(ns, &sab, uname, aname).await?` →
/// `let client = P9Client::connect(DmaTransport::new(&sab, true), uname, aname).await?;
/// mount_wanix(ns, client)?;`.
///
/// Wasm-only: `WanixMount`/`P9Client`/`DmaTransport` are SharedArrayBuffer-DMA
/// types compiled only for the browser target.
#[cfg(target_arch = "wasm32")]
pub fn mount_wanix<T>(
    ns: &mut hyprstream_vfs::Namespace,
    client: hyprstream_9p::client::P9Client<T>,
) -> Result<(), hyprstream_vfs::NamespaceError>
where
    T: hyprstream_9p::client::P9Transport + 'static,
{
    ns.mount(
        "/wanix",
        Arc::new(hyprstream_9p::wanix_mount::WanixMount::new(client)),
    )
}

// ============================================================================
// Compile-time Send + Sync guarantees (#539 T3)
// ============================================================================

/// The generic service→9p mounts must be `Send + Sync` to serve behind the
/// native (multi-threaded) `Mount` trait. These are compiler-enforced, not
/// assumed — the whole point of the native port.
#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assertions() {
        assert_send_sync::<GenericServiceMount>();
        assert_send_sync::<DocMount>();
        // The registry is the shared state behind the native mount; it must be
        // Send + Sync even though a `StreamEntry`'s `Box<dyn StreamHandle>` is
        // only `Send` (guarded by the registry's `Mutex`).
        assert_send_sync::<crate::stream_mount::StreamRegistry>();
    }
    let _ = assertions;
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod automount_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;
    use hyprstream_rpc::rpc_client::{CallOptions, RpcClient};
    use hyprstream_rpc::stream_consumer::StreamHandle;

    /// A stand-in `RpcClient` for the auto-mount wiring test. The end-to-end
    /// path exercised (`mount_all_services` → `walk` → `readdir` at the mount
    /// root) reads method names from the service's compiled `schema_metadata()`
    /// and never dials the client, so these methods only need to type-check.
    struct NoopClient;

    #[async_trait]
    impl RpcClient for NoopClient {
        async fn call(&self, _payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("NoopClient: no transport")
        }
        async fn call_with_options(
            &self,
            _payload: Vec<u8>,
            _options: CallOptions,
        ) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("NoopClient: no transport")
        }
        async fn call_streaming(
            &self,
            _payload: Vec<u8>,
            _ephemeral_pubkey: [u8; 32],
        ) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("NoopClient: no transport")
        }
        async fn open_stream(&self, _payload: Vec<u8>) -> anyhow::Result<Box<dyn StreamHandle>> {
            anyhow::bail!("NoopClient: no transport")
        }
        async fn open_stream_from_info(
            &self,
            _stream_info: hyprstream_rpc::stream_info::StreamInfo,
            _client_secret: [u8; 32],
            _client_pubkey: [u8; 32],
        ) -> anyhow::Result<Box<dyn StreamHandle>> {
            anyhow::bail!("NoopClient: no transport")
        }
        fn next_id(&self) -> u64 {
            0
        }
    }

    /// Every service that `impl_service_dispatch!`s registers a `MountFactory`,
    /// so `mount_all_services` yields a `/srv/{name}` mount with no per-service
    /// wiring here. Then `walk`/`readdir` works end-to-end through the
    /// auto-registered mount (listing the service's methods from schema
    /// metadata — the proof the generated mount is actually reachable).
    #[tokio::test]
    async fn auto_mount_registers_and_serves_srv_paths() {
        // At least the five `impl_service_dispatch!` services must be present.
        let registered: Vec<&str> = inventory::iter::<MountFactory>
            .into_iter()
            .map(|f| f.name)
            .collect();
        for expected in ["registry", "model", "policy", "mcp", "inference"] {
            assert!(
                registered.contains(&expected),
                "service {expected} must self-register a MountFactory; got {registered:?}"
            );
        }

        let ctx = MountContext::new(|_service| Ok(Arc::new(NoopClient) as Arc<dyn RpcClient>));
        let mut ns = hyprstream_vfs::Namespace::new();
        let mounted = mount_all_services(&mut ns, &ctx);
        assert_eq!(
            mounted,
            registered.len(),
            "every registered service must mount with the NoopClient dialer"
        );

        // The auto-mount landed at /srv/{name}.
        let prefixes = ns.mount_prefixes();
        assert!(prefixes.contains(&"/srv/registry"), "prefixes: {prefixes:?}");
        assert!(prefixes.contains(&"/srv/model"), "prefixes: {prefixes:?}");

        // End-to-end through the auto-registered mount: walk the mount root and
        // readdir it — the entries are the service's methods, served by the
        // generated GenericServiceMount with no per-service code.
        let subject = Subject::anonymous();
        let entries = ns
            .ls("/srv/registry", &subject)
            .await
            .expect("readdir /srv/registry through the auto-mounted service");
        assert!(
            !entries.is_empty(),
            "auto-mounted /srv/registry must expose the registry's methods"
        );
    }

    /// A namespace forked after auto-mount inherits every mount (the FS-D #365
    /// per-sandbox isolation model: fork the namespace, don't re-run the loop).
    #[tokio::test]
    async fn forked_namespace_inherits_auto_mounts() {
        let ctx = MountContext::new(|_service| Ok(Arc::new(NoopClient) as Arc<dyn RpcClient>));
        let mut ns = hyprstream_vfs::Namespace::new();
        mount_all_services(&mut ns, &ctx);

        let child = ns.fork();
        assert!(
            child.mount_prefixes().contains(&"/srv/registry"),
            "forked namespace must inherit /srv/registry; got {:?}",
            child.mount_prefixes()
        );
    }
}
