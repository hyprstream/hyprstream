//! MCP (Model Context Protocol) service — UUID-keyed tool registry
//!
//! This service provides an MCP-compliant interface for AI coding assistants
//! (Claude Code, Cursor, etc.) to interact with hyprstream via:
//! - HTTP/SSE transport (for web clients with streaming)
//! - ZMQ control plane (for internal service communication)
//!
//! # Architecture
//!
//! Tools are registered in a `HashMap<Uuid, ToolEntry>`, announced and called by UUID.
//! Sync tools return JSON via ZMQ REQ/REP. Streaming tools use `StreamHandle`
//! (DH + SUB + HMAC verify) to bridge ZMQ streams to MCP SSE.
//!
//! # Token Authentication
//!
//! The service reads `HYPRSTREAM_TOKEN` from environment (stdio) or
//! `Authorization: Bearer <token>` header (HTTP) and validates:
//! 1. JWT signature via Ed25519
//! 2. Token expiration
//! 3. Backend services enforce authorization via Casbin policies

use async_trait::async_trait;
use crate::services::{RegistryClient, PolicyClient};
use crate::services::generated::model_client::ModelClient;
use crate::services::generated::tui_client::TuiClient;
use http::header::AUTHORIZATION;
use crate::services::generated::mcp_client::{
    McpHandler, McpResponseVariant, ToolDefinition, ServiceStatus,
    ToolList, ServiceMetrics, CallTool, dispatch_mcp, serialize_response,
    ErrorInfo,
};
use crate::services::generated::policy_client::PolicyCheck;
use ed25519_dalek::{SigningKey, VerifyingKey};
use futures::future::BoxFuture;
use hyprstream_rpc::auth::jwt;
use hyprstream_service::ServiceContext;
use hyprstream_rpc::service::RequestService;
use hyprstream_rpc::moq_stream::MoqStreamHandle;
use hyprstream_rpc::streaming::StreamPayload;
use hyprstream_rpc::transport::TransportConfig;
use rmcp::{
    model::{
        CallToolRequestParams, CallToolResult, Content, JsonObject,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
        ToolAnnotations,
    },
    service::RequestContext,
    ErrorData, RoleServer, ServerHandler,
};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{trace, warn};
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════════
// Service Name
// ═══════════════════════════════════════════════════════════════════════════════

/// Service name for registration
pub const SERVICE_NAME: &str = "mcp";

/// UUID v5 namespace for deterministic tool UUIDs
const MCP_NS: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1,
    0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

// b07676df-be7f-56eb-9cd8-e91cfd689158 = UUID v5(MCP_NS, "mcp.refresh_tools")
const REFRESH_TOOLS_UUID: Uuid = Uuid::from_bytes([
    0xb0, 0x76, 0x76, 0xdf, 0xbe, 0x7f, 0x56, 0xeb,
    0x9c, 0xd8, 0xe9, 0x1c, 0xfd, 0x68, 0x91, 0x58,
]);

/// Normalize MCP tool arguments for backend deserialization.
///
/// MCP tool schemas expose snake_case parameter names with string types,
/// but the generated request structs use `#[serde(rename_all = "camelCase")]`
/// with native numeric types. This function:
/// 1. Converts object keys from snake_case to camelCase
/// 2. Coerces string values to numbers/booleans where possible
fn normalize_mcp_args(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let converted: serde_json::Map<String, Value> = map
                .into_iter()
                .map(|(k, v)| {
                    let camel = snake_to_camel(&k);
                    (camel, normalize_mcp_args(v))
                })
                .collect();
            Value::Object(converted)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(normalize_mcp_args).collect()),
        Value::String(ref s) if s.is_empty() => value,
        Value::String(ref s) => {
            if s == "true" { return Value::Bool(true); }
            if s == "false" { return Value::Bool(false); }
            if let Ok(n) = s.parse::<u64>() {
                return Value::Number(n.into());
            }
            if let Ok(n) = s.parse::<i64>() {
                return Value::Number(n.into());
            }
            if let Ok(n) = s.parse::<f64>() {
                if let Some(num) = serde_json::Number::from_f64(n) {
                    return Value::Number(num);
                }
            }
            value
        }
        other => other,
    }
}

fn snake_to_camel(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut capitalize_next = false;
    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.extend(ch.to_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for McpService
#[derive(Clone)]
pub struct McpConfig {
    /// Ed25519 public key for JWT verification
    pub verifying_key: VerifyingKey,
    /// Ed25519 signing key for creating RPC clients
    pub signing_key: SigningKey,
    /// RPC transport for control plane
    pub transport: TransportConfig,
    /// Service context for client construction (optional for backward compat)
    pub ctx: Option<Arc<ServiceContext>>,
    /// PolicyService verifying key — used to create the internal PolicyClient
    /// for peer key resolution and authorization checks.
    pub policy_verifying_key: VerifyingKey,
    /// Expected audience (resource URL) for future defense-in-depth
    pub expected_audience: Option<String>,
    /// JWT key source for verifying JWTs on ZMQ path (unified local + federated).
    pub jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Registry Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Handler return type — sync or streaming
pub enum ToolResult {
    /// Immediate JSON result (REQ/REP tools)
    Sync(Value),
    /// Streaming result — MoqStreamHandle encapsulates DH, moq subscribe, HMAC verification
    Stream(Box<MoqStreamHandle>),
}

/// Context passed to handler — carries auth + optional ServiceContext
pub struct ToolCallContext {
    pub args: Value,
    pub signing_key: SigningKey,
    /// Authenticated user string propagated to backend services
    pub user: String,
    /// ServiceContext for typed_client() / client() access (optional for backward compat)
    pub ctx: Option<Arc<ServiceContext>>,
    /// Bootstrap Policy client used only for Policy operations.
    pub policy_client: PolicyClient,
}

type ToolHandler = Arc<dyn Fn(ToolCallContext) -> BoxFuture<'static, anyhow::Result<ToolResult>> + Send + Sync>;

/// A registered tool
#[allow(dead_code)]
struct ToolEntry {
    uuid: Uuid,
    name: String,
    description: String,
    args_schema: Value,
    required_scope: String,
    streaming: bool,
    handler: ToolHandler,
}

/// UUID-keyed tool registry
pub struct ToolRegistry {
    by_uuid: HashMap<Uuid, ToolEntry>,
}

impl ToolRegistry {
    fn new() -> Self {
        Self {
            by_uuid: HashMap::new(),
        }
    }

    fn register(&mut self, entry: ToolEntry) {
        self.by_uuid.insert(entry.uuid, entry);
    }

    fn get(&self, uuid: &Uuid) -> Option<&ToolEntry> {
        self.by_uuid.get(uuid)
    }

    fn list(&self) -> impl Iterator<Item = &ToolEntry> {
        self.by_uuid.values()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Registration
// ═══════════════════════════════════════════════════════════════════════════════

/// Register all tools discovered from schema metadata.
///
/// Each service's schema_metadata() + scoped variants are iterated.
/// Scoped tools are discovered by recursively walking `scoped_client_tree()`.
/// Scope and streaming flags are read from MethodSchema.
fn register_schema_tools(reg: &mut ToolRegistry) {
    use crate::services::generated::{
        model_client, registry_client, policy_client, tui_client,
    };
    // Each service generates its own MethodSchema type, so we use a macro
    // to iterate each service's methods with the correct type.
    macro_rules! register_top_level {
        ($reg:expr, $schema_fn:expr) => {{
            let (service_name, methods) = $schema_fn;
            for method in methods {
                // FIX-7: respect $cliHidden — don't expose internal methods as MCP tools
                if method.hidden { continue; }
                let tool_name = format!("{service_name}.{}", method.name);
                let params: Vec<(&str, &str, bool, &str)> = method.params.iter()
                    .map(|p| (p.name, p.type_name, p.required, p.description))
                    .collect();
                let json_schema = params_to_json_schema(&params);
                let service_name = service_name.to_string();
                let method_name = method.name.to_string();
                let description = if method.description.is_empty() {
                    format!("{service_name}::{method_name}")
                } else {
                    method.description.to_string()
                };
                let required_scope = if !method.scope.is_empty() {
                    method.scope.to_string()
                } else {
                    format!("query:{service_name}:*")
                };

                if method.is_streaming {
                    register_streaming_tool($reg, &tool_name, description, json_schema, required_scope, service_name, method_name);
                } else {
                    register_sync_tool($reg, &tool_name, description, json_schema, required_scope, service_name, method_name);
                }
            }
        }};
    }

    register_top_level!(reg, model_client::schema_metadata());
    register_top_level!(reg, registry_client::schema_metadata());
    register_top_level!(reg, policy_client::schema_metadata());
    register_top_level!(reg, tui_client::schema_metadata());
    // Scoped tools: recursive tree walk for all services with nested scopes
    register_scoped_tools_recursive(reg, "registry", registry_client::scoped_client_tree(), "registry", &[]);
    register_scoped_tools_recursive(reg, "model", model_client::scoped_client_tree(), "model", &[]);
}

/// Accumulated scope info: (scope_name, field_name, capnp_type) for building the
/// `call_scoped_method` scope chain and injecting scope fields into JSON schemas.
type ScopeInfo = (&'static str, &'static str, &'static str); // (scope_name, field_name, type)

/// Recursively walk the scoped client tree, registering MCP tools at every level.
///
/// Each tree node represents a scope (e.g., `repo`, `worktree`, `ctl`) with a scope
/// field that gets injected into the JSON schema and extracted in the handler to build
/// the scope chain for `call_scoped_method`.
fn register_scoped_tools_recursive(
    reg: &mut ToolRegistry,
    service_name: &str,
    nodes: &'static [hyprstream_service::ScopedClientTreeNode],
    prefix: &str,
    parent_scopes: &[ScopeInfo],
) {
    for node in nodes {
        let new_prefix = format!("{}.{}", prefix, node.scope_name);
        let (_, _, methods) = (node.metadata_fn)();

        // Accumulate: parents + this node's (scope_name, field_name, type)
        let mut scopes: Vec<ScopeInfo> = parent_scopes.to_vec();
        if !node.scope_field.is_empty() {
            let field_type = match node.scope_field {
                "fid" => "UInt32",
                _ => "Text",
            };
            scopes.push((node.scope_name, node.scope_field, field_type));
        }

        for method in methods {
            // FIX-7: respect $cliHidden — don't expose internal methods as MCP tools
            if method.hidden { continue; }
            let tool_name = format!("{}.{}", new_prefix, method.name);

            // Build JSON schema: method params + all scope fields from ancestors
            let params: Vec<(&str, &str, bool, &str)> = method.params.iter()
                .map(|p| (p.name, p.type_name, p.required, p.description))
                .collect();
            let mut json_schema = params_to_json_schema(&params);
            if let Value::Object(ref mut map) = json_schema {
                let existing_props: Vec<String> = map.get("properties")
                    .and_then(|p| p.as_object())
                    .map(|p| p.keys().cloned().collect())
                    .unwrap_or_default();

                if let Some(Value::Object(ref mut props)) = map.get_mut("properties") {
                    for &(_, field_name, field_type) in &scopes {
                        // Skip scope fields that collide with method params
                        if existing_props.contains(&field_name.to_owned()) {
                            continue;
                        }
                        let json_type = match field_type {
                            "UInt8" | "UInt16" | "UInt32" | "UInt64" |
                            "Int8" | "Int16" | "Int32" | "Int64" => "integer",
                            "Float32" | "Float64" => "number",
                            "Bool" => "boolean",
                            _ => "string",
                        };
                        props.insert(field_name.into(), serde_json::json!({
                            "type": json_type,
                            "description": field_name,
                        }));
                    }
                }
                if let Some(Value::Array(ref mut req)) = map.get_mut("required") {
                    for (i, &(_, field_name, _)) in scopes.iter().enumerate() {
                        // Avoid duplicate required entries
                        let field_str = Value::String(field_name.into());
                        if !req.contains(&field_str) {
                            req.insert(i, field_str);
                        }
                    }
                }
            }

            let method_name = method.name.to_owned();
            let service = service_name.to_owned();
            let description = if method.description.is_empty() {
                format!("{}::{}", new_prefix, method.name)
            } else {
                method.description.to_owned()
            };
            let required_scope = if !method.scope.is_empty() {
                method.scope.to_owned()
            } else {
                format!("query:{}:*", service_name)
            };

            // Capture (scope_name, field_name) pairs for the handler closure
            let scope_pairs: Vec<(String, String)> = scopes.iter()
                .map(|&(scope_name, field_name, _)| (scope_name.to_owned(), field_name.to_owned()))
                .collect();

            if method.is_streaming {
                // Streaming scoped tool — uses call_scoped_streaming_method with DH key exchange
                reg.register(ToolEntry {
                    uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                    name: tool_name.clone(),
                    description,
                    args_schema: json_schema,
                    required_scope,
                    streaming: true,
                    handler: Arc::new(move |ctx| {
                        let method = method_name.clone();
                        let service = service.clone();
                        let scope_pairs = scope_pairs.clone();
                        Box::pin(async move {
                            // Build scope chain from args BEFORE moving ctx fields
                            // Args are already normalized to camelCase by normalize_mcp_args()
                            let scope_chain: Vec<(String, String)> = scope_pairs.iter()
                                .map(|(scope_name, field_name)| {
                                    let camel_key = snake_to_camel(field_name);
                                    let val_str = ctx.args.get(camel_key.as_str())
                                        .map(|v| match v {
                                            Value::String(s) => s.clone(),
                                            Value::Number(n) => n.to_string(),
                                            _ => v.to_string(),
                                        })
                                        .unwrap_or_default();
                                    (scope_name.clone(), val_str)
                                })
                                .collect();

                            let scope_refs: Vec<(&str, &str)> = scope_chain.iter()
                                .map(|(s, v)| (s.as_str(), v.as_str()))
                                .collect();

                            let (client_secret, client_pubkey) = hyprstream_rpc::generate_ephemeral_keypair();
                            let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

                            // #468: the streaming-method client returns the VERIFIED-capnp
                            // StreamInfo library type directly (no serde_json::Value round-trip).
                            let stream_info: hyprstream_rpc::stream_info::StreamInfo = match service.as_str() {
                                "registry" => {
                                    let client: RegistryClient = RegistryClient::from_resolver(
                                        ctx.signing_key, None,
                                    )?;
                                    client.call_scoped_streaming_method(&scope_refs, &method, &ctx.args, client_pubkey_bytes).await?
                                }
                                "model" => {
                                    let client = ModelClient::from_resolver(ctx.signing_key, None)?;
                                    client.call_scoped_streaming_method(&scope_refs, &method, &ctx.args, client_pubkey_bytes).await?
                                }
                                _ => anyhow::bail!("No scoped streaming dispatch for service: {service}"),
                            };

                            // #356: single networked reach shape (UDS-only resolves
                            // to the same-host fast path inside `networked`).
                            let DecodedStreamReach { dh_public, reach, broadcast_path } =
                                decode_stream_reach(stream_info)?;
                            // #321: derive_client_stream_keys yields the AEAD enc_key.
                            let (mac_key, enc_key, topic) = hyprstream_rpc::derive_client_stream_keys(
                                &client_secret, &client_pubkey_bytes, &dh_public,
                            )?;
                            // #358: MCP tool stream consumed live → direct-first; selection only reorders advertised reaches.
                            let qos = hyprstream_rpc::stream_info::StreamOpt::default();
                            let handle = MoqStreamHandle::networked(reach, &qos, broadcast_path, mac_key, enc_key, topic);

                            Ok(ToolResult::Stream(Box::new(handle)))
                        })
                    }),
                });
            } else {
                // Sync scoped tool — uses call_scoped_method
                reg.register(ToolEntry {
                    uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                    name: tool_name.clone(),
                    description,
                    args_schema: json_schema,
                    required_scope,
                    streaming: false,
                    handler: Arc::new(move |ctx| {
                        let method = method_name.clone();
                        let service = service.clone();
                        let scope_pairs = scope_pairs.clone();
                        Box::pin(async move {
                            // Build scope chain from args: [("repo", repo_id_val), ("worktree", name_val), ...]
                            // Args are already normalized to camelCase by normalize_mcp_args()
                            let scope_chain: Vec<(String, String)> = scope_pairs.iter()
                                .map(|(scope_name, field_name)| {
                                    let camel_key = snake_to_camel(field_name);
                                    let val_str = ctx.args.get(camel_key.as_str())
                                        .map(|v| match v {
                                            Value::String(s) => s.clone(),
                                            Value::Number(n) => n.to_string(),
                                            _ => v.to_string(),
                                        })
                                        .unwrap_or_default();
                                    (scope_name.clone(), val_str)
                                })
                                .collect();

                            let scope_refs: Vec<(&str, &str)> = scope_chain.iter()
                                .map(|(s, v)| (s.as_str(), v.as_str()))
                                .collect();

                            dispatch_scoped_call(&service, &scope_refs, &method, &ctx).await
                        })
                    }),
                });
            }
        }

        // Recurse into nested scopes
        register_scoped_tools_recursive(reg, service_name, node.nested, &new_prefix, &scopes);
    }
}

/// Dispatch a scoped method call to the appropriate service client.
///
/// Builds the service-specific client and calls `call_scoped_method` with the
/// scope chain (e.g., `[("repo", "abc-123"), ("worktree", "main")]`).
async fn dispatch_scoped_call(
    service: &str,
    scopes: &[(&str, &str)],
    method: &str,
    ctx: &ToolCallContext,
) -> anyhow::Result<ToolResult> {
    let result = match service {
        "registry" => {
            let client: RegistryClient = RegistryClient::from_resolver(
                ctx.signing_key.clone(), None,
            )?;
            client.call_scoped_method(scopes, method, &ctx.args).await?
        }
        "model" => {
            let client = ModelClient::from_resolver(ctx.signing_key.clone(), None)?;
            client.call_scoped_method(scopes, method, &ctx.args).await?
        }
        _ => anyhow::bail!("No scoped dispatch for service: {service}"),
    };
    Ok(ToolResult::Sync(result))
}

fn register_sync_tool(
    reg: &mut ToolRegistry,
    tool_name: &str,
    description: String,
    json_schema: Value,
    required_scope: String,
    service_name: String,
    method_name: String,
) {
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
        name: tool_name.to_owned(),
        description,
        args_schema: json_schema,
        required_scope,
        streaming: false,
        handler: Arc::new(move |ctx| {
            let service = service_name.clone();
            let method = method_name.clone();
            Box::pin(async move {
                let result = dispatch_schema_call(&service, &method, &ctx).await?;
                Ok(ToolResult::Sync(result))
            })
        }),
    });
}

fn register_streaming_tool(
    reg: &mut ToolRegistry,
    tool_name: &str,
    description: String,
    json_schema: Value,
    required_scope: String,
    service_name: String,
    method_name: String,
) {
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
        name: tool_name.to_owned(),
        description,
        args_schema: json_schema,
        required_scope,
        streaming: true,
        handler: Arc::new(move |ctx| {
            let service = service_name.clone();
            let method = method_name.clone();
            Box::pin(async move {
                let (client_secret, client_pubkey) = hyprstream_rpc::generate_ephemeral_keypair();
                let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

                // #468: verified-capnp StreamInfo returned directly (no serde_json round-trip).
                let stream_info: hyprstream_rpc::stream_info::StreamInfo = match service.as_str() {
                    "registry" => {
                        let client: RegistryClient = RegistryClient::from_resolver(
                            ctx.signing_key, None,
                        )?;
                        client.call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?
                    }
                    "model" => {
                        let client = ModelClient::from_resolver(ctx.signing_key, None)?;
                        client.call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?
                    }
                    "tui" => {
                        let client = TuiClient::from_resolver(ctx.signing_key, None)?;
                        client.call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?
                    }
                    _ => anyhow::bail!("No streaming support for service: {}", service),
                };

                // #356: single networked reach shape (UDS-only resolves to the
                // same-host fast path inside `networked`).
                let DecodedStreamReach { dh_public, reach, broadcast_path } =
                    decode_stream_reach(stream_info)?;
                // #321: derive_client_stream_keys yields the AEAD enc_key.
                let (mac_key, enc_key, topic) = hyprstream_rpc::derive_client_stream_keys(
                    &client_secret, &client_pubkey_bytes, &dh_public,
                )?;
                // #358: MCP tool stream consumed live → direct-first; selection only reorders advertised reaches.
                let qos = hyprstream_rpc::stream_info::StreamOpt::default();
                let handle = MoqStreamHandle::networked(reach, &qos, broadcast_path, mac_key, enc_key, topic);

                Ok(ToolResult::Stream(Box::new(handle)))
            })
        }),
    });
}

/// Convert method params to a JSON Schema for tool arguments.
///
/// Takes params as (name, type_name, required, description) tuples — works with any generated module's ParamSchema
/// since they all have the same layout.
fn params_to_json_schema(params: &[(&str, &str, bool, &str)]) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for &(name, type_name, is_required, description) in params {
        let json_type = match type_name {
            "Text" | "Data" => "string",
            "Bool" => "boolean",
            "UInt8" | "UInt16" | "UInt32" | "UInt64" |
            "Int8" | "Int16" | "Int32" | "Int64" => "integer",
            "Float32" | "Float64" => "number",
            t if t.starts_with("List(") => "array",
            _ => "string",
        };

        let mut param_schema = serde_json::Map::new();
        param_schema.insert("type".to_owned(), Value::String(json_type.to_owned()));
        if !description.is_empty() {
            param_schema.insert("description".to_owned(), Value::String(description.to_owned()));
        }

        properties.insert(name.to_owned(), Value::Object(param_schema));
        if is_required {
            required.push(Value::String(name.to_owned()));
        }
    }

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

/// The transport-resolved shape of a decoded streaming response (#356).
///
/// A single networked shape: every producer now advertises its moq reach via the
/// canonical `StreamInfo.announcedAt` list (or an empty list when UDS-only, which
/// `connect_moq_reach`/`MoqStreamHandle::networked` resolve to the same-host UDS
/// fast path from LOCAL config). There is no longer a wire-published UDS path.
struct DecodedStreamReach {
    dh_public: [u8; 32],
    reach: Vec<hyprstream_rpc::stream_info::Destination>,
    broadcast_path: String,
}

/// Decode a streaming response into its moq reach (#356).
///
/// Takes the VERIFIED-capnp `StreamInfo` library type as returned by the
/// streaming-method client (decoded + COSE-verified inside `call_streaming`,
/// `rpc_client.rs`) — NO `serde_json` round-trip (#468) — yielding the
/// native-capnp `announcedAt` reach list + `broadcastPath`.
/// Fails closed when the response carries no broadcast path or DH key.
fn decode_stream_reach(
    info: hyprstream_rpc::stream_info::StreamInfo,
) -> anyhow::Result<DecodedStreamReach> {
    if info.broadcast_path.is_empty() {
        anyhow::bail!(
            "missing broadcastPath in streaming response — server did not initialize moq transport"
        );
    }
    if info.dh_public == [0u8; 32] {
        anyhow::bail!("server did not provide DH public key for streaming");
    }
    Ok(DecodedStreamReach {
        dh_public: info.dh_public,
        reach: info.announced_at,
        broadcast_path: info.broadcast_path,
    })
}

/// Dispatch a method call to the appropriate generated client.
async fn dispatch_schema_call(service: &str, method: &str, ctx: &ToolCallContext) -> anyhow::Result<Value> {
    let signing_key = ctx.signing_key.clone();

    match service {
        "model" => {
            let client = ModelClient::from_resolver(signing_key, None)?;
            client.call_method(method, &ctx.args).await
        }
        "registry" => {
            let client: RegistryClient = RegistryClient::from_resolver(
                signing_key, None,
            )?;
            client.call_method(method, &ctx.args).await
        }
        "policy" => {
            ctx.policy_client.call_method(method, &ctx.args).await
        }
        "tui" => {
            let client = TuiClient::from_resolver(signing_key, None)?;
            client.call_method(method, &ctx.args).await
        }
        _ => anyhow::bail!("Unknown service: {service}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// McpService
// ═══════════════════════════════════════════════════════════════════════════════

/// MCP service implementation — UUID-keyed tool registry
pub struct McpService {
    /// UUID-keyed tool registry (RwLock for live refresh)
    registry: Arc<RwLock<ToolRegistry>>,
    /// Raw HYPRSTREAM_TOKEN from env (stdio transport — decoded per-request)
    stdio_token: Option<String>,
    /// Verifying key for JWT validation
    verifying_key: VerifyingKey,
    // === RequestService infrastructure ===
    transport: TransportConfig,
    signing_key: SigningKey,
    /// ServiceContext for typed_client() / client() access
    service_ctx: Option<Arc<ServiceContext>>,
    /// Expected audience for tokens (resource URL, for defense-in-depth)
    expected_audience: Option<String>,
    /// JWT key source for verifying JWTs on ZMQ path (unified local + federated).
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    /// Policy client for authorization checks (shared, avoids per-call socket creation)
    policy_client: PolicyClient,
}

impl McpService {
    /// Create a new McpService with JWT authentication
    pub fn new(config: McpConfig) -> anyhow::Result<Self> {
        let stdio_token = std::env::var("HYPRSTREAM_TOKEN").ok();

        let mut tool_reg = ToolRegistry::new();
        register_schema_tools(&mut tool_reg);

        tracing::info!(
            "McpService registered {} tools (all schema-discovered)",
            tool_reg.by_uuid.len(),
        );

        let policy_client = PolicyClient::for_local_bootstrap(
            config.signing_key.clone(),
            config.policy_verifying_key,
            None,
        )?;

        Ok(Self {
            registry: Arc::new(RwLock::new(tool_reg)),
            stdio_token,
            verifying_key: config.verifying_key,
            transport: config.transport,
            signing_key: config.signing_key,
            service_ctx: config.ctx,
            expected_audience: config.expected_audience,
            jwt_key_source: config.jwt_key_source,
            policy_client,
        })
    }

    /// Convert registry to rmcp Tool list (includes built-in refresh_tools meta-tool)
    fn tools_list(&self) -> Vec<Tool> {
        let reg = self.registry.read();
        let mut tools: Vec<Tool> = reg.list().map(|entry| {
            let schema: JsonObject = match &entry.args_schema {
                Value::Object(m) => m.clone(),
                _ => JsonObject::new(),
            };
            Tool {
                name: Cow::Owned(entry.uuid.to_string()),
                title: Some(entry.name.clone()),
                description: Some(Cow::Owned(entry.description.clone())),
                input_schema: Arc::new(schema),
                output_schema: None,
                annotations: Some(ToolAnnotations {
                    title: Some(entry.name.clone()),
                    read_only_hint: Some(entry.required_scope.starts_with("query:")),
                    destructive_hint: Some(!entry.required_scope.starts_with("query:")),
                    open_world_hint: Some(false),
                    // FIX-8: only query-scoped tools are idempotent; write/train/manage ops are not
                    idempotent_hint: Some(entry.required_scope.starts_with("query:")),
                }),
                icons: None,
                meta: None,
            }
        }).collect();

        tools.push(Tool {
            name: Cow::Owned(REFRESH_TOOLS_UUID.to_string()),
            title: Some("mcp.refresh_tools".to_owned()),
            description: Some(Cow::Borrowed(
                "Rebuild the tool registry from current service schemas and notify \
                 this session to re-fetch the tool list. Call after new services come \
                 online to discover their tools without reconnecting.",
            )),
            input_schema: Arc::new(JsonObject::new()),
            output_schema: None,
            annotations: Some(ToolAnnotations {
                title: Some("mcp.refresh_tools".to_owned()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                open_world_hint: Some(false),
                idempotent_hint: Some(true),
            }),
            icons: None,
            meta: None,
        });

        tools
    }

    /// Extract identity from validated middleware state or fall back to env/local.
    ///
    /// This is pure authentication (identity extraction). No audience checks,
    /// no scope filtering — ZMQ backends handle authorization via Casbin.
    ///
    /// Priority:
    /// 1. HTTP transport: use `AuthenticatedUser` from middleware (already validated)
    /// 2. Stdio/ZMQ transport (no HTTP Parts): use env var JWT claims or `"anonymous"`
    fn extract_user(&self, context: &RequestContext<RoleServer>) -> String {
        if let Some(parts) = context.extensions.get::<http::request::Parts>() {
            if let Some(auth_user) = parts.extensions.get::<crate::server::middleware::AuthenticatedUser>() {
                trace!("MCP HTTP auth: using validated identity for {}", auth_user.user);
                return auth_user.user.clone();
            }

            if parts.headers.contains_key(AUTHORIZATION) {
                warn!("MCP HTTP auth: Authorization header present but no AuthenticatedUser — middleware should have rejected");
            } else {
                trace!("MCP HTTP auth: no Authorization header, anonymous access");
            }
            "anonymous".to_owned()
        } else {
            trace!("MCP HTTP auth: no http::request::Parts in extensions (stdio/zmq transport)");
            match &self.stdio_token {
                Some(token) => {
                    match jwt::decode(token, &self.verifying_key, self.expected_audience.as_deref()) {
                        Ok(claims) => claims.sub.clone(),
                        Err(e) => {
                            warn!("MCP stdio auth: token decode failed ({}), downgrading to anonymous", e);
                            "anonymous".to_owned()
                        }
                    }
                }
                None => "anonymous".to_owned(),
            }
        }
    }

    /// Dispatch a tool call by UUID with a specific identity
    async fn dispatch_tool(&self, uuid: &Uuid, args: Value, user: String) -> Result<CallToolResult, ErrorData> {
        let handler = {
            let reg = self.registry.read();
            let entry = reg.get(uuid)
                .ok_or_else(|| ErrorData::invalid_request(format!("Unknown tool: {}", uuid), None))?;
            entry.handler.clone()
        };

        let ctx = ToolCallContext {
            args: normalize_mcp_args(args),
            signing_key: self.signing_key.clone(),
            user,
            ctx: self.service_ctx.clone(),
            policy_client: self.policy_client.clone(),
        };

        let result = handler(ctx).await
            .map_err(|e| ErrorData::internal_error(format!("Tool failed: {}", e), None))?;

        match result {
            ToolResult::Sync(value) => {
                Ok(CallToolResult::success(vec![Content::text(value.to_string())]))
            }
            ToolResult::Stream(mut handle) => {
                // Consume StreamHandle — DH, SUB, HMAC all handled internally
                let mut contents = Vec::new();
                while let Some(payload) = handle.recv_next().await
                    .map_err(|e| ErrorData::internal_error(format!("Stream error: {}", e), None))?
                {
                    match payload {
                        StreamPayload::Data(data) => {
                            contents.push(Content::text(
                                String::from_utf8_lossy(&data).to_string(),
                            ));
                        }
                        StreamPayload::Complete(meta) => {
                            contents.push(Content::text(
                                String::from_utf8_lossy(&meta).to_string(),
                            ));
                            break;
                        }
                        StreamPayload::Error(msg) => {
                            return Err(ErrorData::internal_error(msg, None));
                        }
                        StreamPayload::Tagged { .. } => {
                            // encrypted event payload, skip
                        }
                    }
                }
                Ok(CallToolResult::success(contents))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ServerHandler Implementation (manual — no macros)
// ═══════════════════════════════════════════════════════════════════════════════

impl ServerHandler for McpService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_tool_list_changed()
                .build(),
            server_info: rmcp::model::Implementation {
                name: "hyprstream".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                icons: None,
                title: None,
                website_url: None,
            },
            instructions: Some(
                "Hyprstream AI inference service. \
                 Connect via HTTP transport (url-based) for automatic OAuth authentication. \
                 For stdio transport, set HYPRSTREAM_TOKEN env var if needed."
                    .into(),
            ),
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        // Tool listing is public; authorization happens at call_tool time via ZMQ backends
        std::future::ready(Ok(ListToolsResult {
            meta: None,
            tools: self.tools_list(),
            next_cursor: None,
        }))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + Send + '_ {
        let user = self.extract_user(&context);
        async move {
            let uuid = Uuid::parse_str(&request.name)
                .map_err(|e| ErrorData::invalid_request(format!("Invalid UUID: {}", e), None))?;

            if uuid == REFRESH_TOOLS_UUID {
                let old_count;
                let new_count;
                {
                    let mut reg = self.registry.write();
                    old_count = reg.by_uuid.len();
                    let mut fresh = ToolRegistry::new();
                    register_schema_tools(&mut fresh);
                    new_count = fresh.by_uuid.len();
                    *reg = fresh;
                }
                let _ = context.peer.notify_tool_list_changed().await;
                tracing::info!("MCP refresh_tools: {} -> {} tools", old_count, new_count);
                return Ok(CallToolResult::success(vec![Content::text(
                    format!("Refreshed tool registry: {} tools (was {})", new_count, old_count),
                )]));
            }

            let args = match request.arguments {
                Some(map) => Value::Object(map),
                None => Value::Object(serde_json::Map::new()),
            };

            self.dispatch_tool(&uuid, args, user).await
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// McpHandler Implementation (generated trait)
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait::async_trait(?Send)]
impl McpHandler for McpService {
    async fn authorize(&self, ctx: &crate::services::EnvelopeContext, resource: &str, operation: &str) -> anyhow::Result<()> {
        let subject = ctx.subject().to_string();
        let result = self.policy_client.check(&PolicyCheck {
            subject: subject.clone(),
            domain: "*".to_owned(),
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        }).await;
        match result {
            Ok(allowed) => {
                if allowed {
                    Ok(())
                } else {
                    tracing::info!("MCP policy denied: sub={} obj={} act={}", subject, resource, operation);
                    anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
                }
            }
            Err(e) => {
                tracing::warn!("MCP policy check RPC error: sub={} obj={} act={} err={}", subject, resource, operation, e);
                anyhow::bail!("Unauthorized: {} cannot {} on {} (policy check error: {})", subject, operation, resource, e)
            }
        }
    }

    async fn handle_get_status(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        let loaded_model_count = {
            // Status check uses local identity (internal health check, no user context)
            let client = ModelClient::from_resolver(
                self.signing_key.clone(),
                None,
            )?;
            client.status(&crate::services::generated::model_client::StatusRequest { model_ref: String::new() }).await
                .map(|models| models.len() as u32)
                .unwrap_or(0)
        };

        Ok(McpResponseVariant::GetStatusResult(ServiceStatus {
            is_running: true,
            loaded_model_count,
            is_authenticated: self.stdio_token.is_some(),
            authenticated_user: self.stdio_token.as_ref()
                .and_then(|t| jwt::decode(t, &self.verifying_key, self.expected_audience.as_deref()).ok())
                .map(|c| c.sub)
                .unwrap_or_default(),
            scopes: vec![],  // Scopes no longer in JWT; authorization via Casbin
        }))
    }

    async fn handle_list_tools(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        let reg = self.registry.read();
        let tools: Vec<ToolDefinition> = reg.list().map(|entry| {
            ToolDefinition {
                name: entry.uuid.to_string(),
                description: entry.description.clone(),
                is_read_only: entry.required_scope.starts_with("query:"),
                is_destructive: !entry.required_scope.starts_with("query:"),
                required_scope: entry.required_scope.clone(),
                argument_schema: entry.args_schema.to_string(),
            }
        }).collect();

        Ok(McpResponseVariant::ListToolsResult(ToolList { tools }))
    }

    async fn handle_get_metrics(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        Ok(McpResponseVariant::GetMetricsResult(ServiceMetrics {
            total_calls: 0,
            calls_per_tool: Vec::new(),
            average_call_duration_ms: 0.0,
            uptime_seconds: 0.0,
        }))
    }

    async fn handle_call_tool(
        &self,
        ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
        data: &CallTool,
    ) -> anyhow::Result<McpResponseVariant> {
        let uuid = Uuid::parse_str(&data.tool_name)
            .map_err(|e| anyhow::anyhow!("Invalid tool UUID '{}': {}", data.tool_name, e))?;

        let args: Value = if data.arguments.is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&data.arguments)?
        };

        let user = ctx.user().to_owned();
        let result = self.dispatch_tool(&uuid, args, user).await;

        match result {
            Ok(call_result) => {
                use rmcp::model::RawContent;
                let text: String = call_result.content.iter()
                    .filter_map(|c| match &c.raw {
                        RawContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                Ok(McpResponseVariant::CallToolResult(crate::services::generated::mcp_client::ToolResult {
                    success: true,
                    result: text,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                Ok(McpResponseVariant::CallToolResult(crate::services::generated::mcp_client::ToolResult {
                    success: false,
                    result: "null".to_owned(),
                    error_message: format!("{}", e),
                }))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RequestService Implementation (internal control plane)
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait(?Send)]
impl RequestService for McpService {
    async fn handle_request(&self, ctx: &crate::services::EnvelopeContext, payload: &[u8]) -> anyhow::Result<(Vec<u8>, Option<crate::services::Continuation>)> {
        trace!(
            "McpService request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_mcp(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "mcp"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.jwt_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = McpResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}
