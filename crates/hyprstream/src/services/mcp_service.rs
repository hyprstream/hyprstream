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
use crate::services::{ModelZmqClient, GenRegistryClient, PolicyClient, InferenceZmqClient};
use http::header::AUTHORIZATION;
use crate::services::generated::mcp_client::{
    McpHandler, McpResponseVariant, ToolDefinition, ServiceStatus,
    ToolList, ServiceMetrics, CallTool, dispatch_mcp, serialize_response,
    ErrorInfo,
};
use ed25519_dalek::{SigningKey, VerifyingKey};
use futures::future::BoxFuture;
use hyprstream_rpc::auth::jwt;
use hyprstream_rpc::envelope::RequestIdentity;
use hyprstream_rpc::service::factory::ServiceContext;
use hyprstream_rpc::service::ZmqService;
use hyprstream_rpc::streaming::{StreamHandle, StreamPayload};
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
use tracing::trace;
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

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for McpService
#[derive(Clone)]
pub struct McpConfig {
    /// Ed25519 public key for JWT verification
    pub verifying_key: VerifyingKey,
    /// ZMQ context for backend clients
    pub zmq_context: Arc<zmq::Context>,
    /// Ed25519 signing key for creating ZMQ clients
    pub signing_key: SigningKey,
    /// ZMQ transport for control plane
    pub transport: TransportConfig,
    /// Service context for client construction (optional for backward compat)
    pub ctx: Option<Arc<ServiceContext>>,
    /// Expected audience (resource URL) for future defense-in-depth
    pub expected_audience: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Registry Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Handler return type — sync or streaming
pub enum ToolResult {
    /// Immediate JSON result (REQ/REP tools)
    Sync(Value),
    /// Streaming result — StreamHandle encapsulates DH, SUB, HMAC verification
    Stream(StreamHandle),
}

/// Context passed to handler — carries auth + ZMQ infra + optional ServiceContext
pub struct ToolCallContext {
    pub args: Value,
    pub signing_key: SigningKey,
    pub zmq_context: Arc<zmq::Context>,
    /// Authenticated identity propagated to backend services
    pub identity: RequestIdentity,
    /// ServiceContext for typed_client() / client() access (optional for backward compat)
    pub ctx: Option<Arc<ServiceContext>>,
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
/// Scope and streaming flags are read from MethodSchema.
fn register_schema_tools(reg: &mut ToolRegistry) {
    use crate::services::generated::{
        model_client, registry_client, policy_client, inference_client,
    };

    // Each service generates its own MethodSchema type, so we use a macro
    // to iterate each service's methods with the correct type.
    macro_rules! register_top_level {
        ($reg:expr, $schema_fn:expr) => {{
            let (service_name, methods) = $schema_fn;
            for method in methods {
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
    register_top_level!(reg, inference_client::schema_metadata());

    // Scoped: registry.repo.*
    {
        let (_, _, repo_methods) = registry_client::repo_schema_metadata();
        for method in repo_methods {
            let tool_name = format!("registry.repo.{}", method.name);

            let params: Vec<(&str, &str, bool, &str)> = method.params.iter()
                .map(|p| (p.name, p.type_name, p.required, p.description))
                .collect();
            let mut json_schema = params_to_json_schema(&params);
            if let Value::Object(ref mut map) = json_schema {
                if let Some(Value::Object(ref mut props)) = map.get_mut("properties") {
                    props.insert("repo_id".into(), serde_json::json!({"type": "string", "description": "Repository ID"}));
                }
                if let Some(Value::Array(ref mut req)) = map.get_mut("required") {
                    req.insert(0, Value::String("repo_id".into()));
                }
            }

            let method_name = method.name.to_string();
            let description = if method.description.is_empty() {
                format!("registry.repo::{}", method.name)
            } else {
                method.description.to_string()
            };
            let required_scope = if !method.scope.is_empty() {
                method.scope.to_string()
            } else {
                "query:registry:*".into()
            };
            reg.register(ToolEntry {
                uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                name: tool_name.clone(),
                description,
                args_schema: json_schema,
                required_scope,
                streaming: false,
                handler: Arc::new(move |ctx| {
                    let method = method_name.clone();
                    Box::pin(async move {
                        let repo_id = ctx.args["repo_id"].as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing repo_id"))?;
                        let client: GenRegistryClient = crate::services::core::create_service_client(
                            &hyprstream_rpc::registry::global().endpoint("registry", hyprstream_rpc::registry::SocketKind::Rep).to_zmq_string(),
                            ctx.signing_key, ctx.identity.clone(),
                        );
                        let gen_repo = client.repo(repo_id);
                        let result = gen_repo.call_method(&method, &ctx.args).await?;
                        Ok(ToolResult::Sync(result))
                    })
                }),
            });
        }
    }

    // Scoped: model.ttt.*, model.peft.*, model.infer.*
    {
        let scoped_metadata: &[(&str, fn() -> (&'static str, &'static str, &'static [model_client::MethodSchema]))] = &[
            ("ttt", model_client::ttt_schema_metadata),
            ("peft", model_client::peft_schema_metadata),
            ("infer", model_client::infer_schema_metadata),
        ];

        for &(scope_name, metadata_fn) in scoped_metadata {
            let (_, _, methods) = metadata_fn();

            for method in methods {
                let tool_name = format!("model.{scope_name}.{}", method.name);

                let params: Vec<(&str, &str, bool, &str)> = method.params.iter()
                    .map(|p| (p.name, p.type_name, p.required, p.description))
                    .collect();
                let mut json_schema = params_to_json_schema(&params);
                if let Value::Object(ref mut map) = json_schema {
                    if let Some(Value::Object(ref mut props)) = map.get_mut("properties") {
                        props.insert("model_ref".into(), serde_json::json!({"type": "string", "description": "Model reference"}));
                    }
                    if let Some(Value::Array(ref mut req)) = map.get_mut("required") {
                        req.insert(0, Value::String("model_ref".into()));
                    }
                }

                let method_name = method.name.to_string();
                let scope = scope_name.to_string();
                let description = if method.description.is_empty() {
                    format!("model.{scope_name}::{}", method.name)
                } else {
                    method.description.to_string()
                };
                let required_scope = if !method.scope.is_empty() {
                    method.scope.to_string()
                } else {
                    "query:model:*".into()
                };

                if method.is_streaming {
                    // Streaming handler: DH keypair + call_streaming_method on scoped client + StreamHandle
                    reg.register(ToolEntry {
                        uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                        name: tool_name.clone(),
                        description,
                        args_schema: json_schema,
                        required_scope,
                        streaming: true,
                        handler: Arc::new(move |ctx| {
                            let method = method_name.clone();
                            let scope = scope.clone();
                            Box::pin(async move {
                                let model_ref = ctx.args["model_ref"].as_str()
                                    .ok_or_else(|| anyhow::anyhow!("missing model_ref"))?;

                                let (client_secret, client_pubkey) = hyprstream_rpc::generate_ephemeral_keypair();
                                let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

                                let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
                                let stream_info_json = match scope.as_str() {
                                    "ttt" => client.gen.ttt(model_ref).call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?,
                                    "peft" => client.gen.peft(model_ref).call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?,
                                    "infer" => client.gen.infer(model_ref).call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?,
                                    _ => anyhow::bail!("Unknown scope: {}", scope),
                                };

                                let (stream_id, endpoint, server_pubkey) = parse_stream_info(&stream_info_json)?;

                                let handle = StreamHandle::new(
                                    &ctx.zmq_context,
                                    stream_id,
                                    &endpoint,
                                    &server_pubkey,
                                    &client_secret,
                                    &client_pubkey_bytes,
                                )?;

                                Ok(ToolResult::Stream(handle))
                            })
                        }),
                    });
                } else {
                    // Sync handler: call method via the appropriate scoped client
                    reg.register(ToolEntry {
                        uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                        name: tool_name.clone(),
                        description,
                        args_schema: json_schema,
                        required_scope,
                        streaming: false,
                        handler: Arc::new(move |ctx| {
                            let method = method_name.clone();
                            let scope = scope.clone();
                            Box::pin(async move {
                                let model_ref = ctx.args["model_ref"].as_str()
                                    .ok_or_else(|| anyhow::anyhow!("missing model_ref"))?;
                                let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
                                let result = match scope.as_str() {
                                    "ttt" => client.gen.ttt(model_ref).call_method(&method, &ctx.args).await?,
                                    "peft" => client.gen.peft(model_ref).call_method(&method, &ctx.args).await?,
                                    "infer" => client.gen.infer(model_ref).call_method(&method, &ctx.args).await?,
                                    _ => anyhow::bail!("Unknown scope: {}", scope),
                                };
                                Ok(ToolResult::Sync(result))
                            })
                        }),
                    });
                }
            }
        }
    }
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
        name: tool_name.to_string(),
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
        name: tool_name.to_string(),
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

                let stream_info_json = match service.as_str() {
                    "registry" => {
                        let client: GenRegistryClient = crate::services::core::create_service_client(
                            &hyprstream_rpc::registry::global().endpoint("registry", hyprstream_rpc::registry::SocketKind::Rep).to_zmq_string(),
                            ctx.signing_key, ctx.identity.clone(),
                        );
                        client.call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?
                    }
                    "model" => {
                        let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
                        client.gen.call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?
                    }
                    "inference" => {
                        let client = InferenceZmqClient::new(ctx.signing_key, ctx.identity.clone());
                        client.gen.call_streaming_method(&method, &ctx.args, client_pubkey_bytes).await?
                    }
                    _ => anyhow::bail!("No streaming support for service: {}", service),
                };

                let (stream_id, endpoint, server_pubkey) = parse_stream_info(&stream_info_json)?;

                let handle = StreamHandle::new(
                    &ctx.zmq_context,
                    stream_id,
                    &endpoint,
                    &server_pubkey,
                    &client_secret,
                    &client_pubkey_bytes,
                )?;

                Ok(ToolResult::Stream(handle))
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
            "UInt32" | "UInt64" | "Int32" | "Int64" => "integer",
            "Float32" | "Float64" => "number",
            t if t.starts_with("List(") => "array",
            _ => "string",
        };

        let mut param_schema = serde_json::Map::new();
        param_schema.insert("type".to_string(), Value::String(json_type.to_string()));
        if !description.is_empty() {
            param_schema.insert("description".to_string(), Value::String(description.to_string()));
        }

        properties.insert(name.to_string(), Value::Object(param_schema));
        if is_required {
            required.push(Value::String(name.to_string()));
        }
    }

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

/// Parse stream_id, endpoint, and server_pubkey from a streaming response JSON.
fn parse_stream_info(json: &Value) -> anyhow::Result<(String, String, Vec<u8>)> {
    let stream_id = json["stream_id"].as_str()
        .ok_or_else(|| anyhow::anyhow!("missing stream_id in streaming response"))?
        .to_string();
    let endpoint = json["endpoint"].as_str()
        .or_else(|| json["stream_endpoint"].as_str())
        .ok_or_else(|| anyhow::anyhow!("missing endpoint in streaming response"))?
        .to_string();
    let server_pubkey: Vec<u8> = json["server_pubkey"].as_array()
        .ok_or_else(|| anyhow::anyhow!("missing server_pubkey in streaming response"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u8)
        .collect();
    Ok((stream_id, endpoint, server_pubkey))
}

/// Dispatch a method call to the appropriate generated client.
async fn dispatch_schema_call(service: &str, method: &str, ctx: &ToolCallContext) -> anyhow::Result<Value> {
    let signing_key = ctx.signing_key.clone();
    let identity = ctx.identity.clone();

    match service {
        "model" => {
            let client = ModelZmqClient::new(signing_key, identity);
            client.gen.call_method(method, &ctx.args).await
        }
        "registry" => {
            let client: GenRegistryClient = crate::services::core::create_service_client(
                &hyprstream_rpc::registry::global().endpoint("registry", hyprstream_rpc::registry::SocketKind::Rep).to_zmq_string(),
                signing_key, identity,
            );
            client.call_method(method, &ctx.args).await
        }
        "policy" => {
            let client = PolicyClient::new(signing_key, identity);
            client.call_method(method, &ctx.args).await
        }
        "inference" => {
            let client = InferenceZmqClient::new(signing_key, identity);
            client.gen.call_method(method, &ctx.args).await
        }
        _ => anyhow::bail!("Unknown service: {service}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// McpService
// ═══════════════════════════════════════════════════════════════════════════════

/// MCP service implementation — UUID-keyed tool registry
pub struct McpService {
    /// UUID-keyed tool registry
    registry: Arc<ToolRegistry>,
    /// Raw HYPRSTREAM_TOKEN from env (stdio transport — decoded per-request)
    stdio_token: Option<String>,
    /// Verifying key for JWT validation
    verifying_key: VerifyingKey,
    // === ZmqService infrastructure ===
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
    /// ServiceContext for typed_client() / client() access
    service_ctx: Option<Arc<ServiceContext>>,
    /// Expected audience for tokens (resource URL, for future defense-in-depth)
    #[allow(dead_code)]
    expected_audience: Option<String>,
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

        Ok(Self {
            registry: Arc::new(tool_reg),
            stdio_token,
            verifying_key: config.verifying_key,
            context: config.zmq_context.clone(),
            transport: config.transport,
            signing_key: config.signing_key,
            service_ctx: config.ctx,
            expected_audience: config.expected_audience,
        })
    }

    /// Convert registry to rmcp Tool list
    fn tools_list(&self) -> Vec<Tool> {
        self.registry.list().map(|entry| {
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
                    idempotent_hint: Some(true),
                }),
                icons: None,
                meta: None,
            }
        }).collect()
    }

    /// Extract identity from HTTP Bearer token or fall back to env/local.
    ///
    /// This is pure authentication (identity extraction). No audience checks,
    /// no scope filtering — ZMQ backends handle authorization via Casbin.
    ///
    /// Priority:
    /// 1. HTTP transport: extract from `Authorization: Bearer <token>` header
    ///    - Valid token → `RequestIdentity::api_token(sub, "mcp")`
    ///    - Invalid/expired token → `RequestIdentity::anonymous()` (let ZMQ reject)
    ///    - No token → `RequestIdentity::anonymous()` (NOT local!)
    /// 2. Stdio/ZMQ transport (no HTTP Parts): use env var claims or `local()`
    fn extract_identity(&self, context: &RequestContext<RoleServer>) -> RequestIdentity {
        // Check for HTTP transport by looking for http::request::Parts in extensions
        if let Some(parts) = context.extensions.get::<http::request::Parts>() {
            // HTTP transport — extract Bearer token from Authorization header
            let token = parts
                .headers
                .get(AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|h| h.strip_prefix("Bearer ").map(str::trim));

            match token {
                Some(token) => {
                    match jwt::decode(token, &self.verifying_key) {
                        Ok(claims) => {
                            trace!("MCP HTTP auth: token validated for {}", claims.sub);
                            RequestIdentity::api_token(&claims.sub, "mcp")
                        }
                        Err(e) => {
                            trace!("MCP HTTP auth: token validation failed: {}", e);
                            RequestIdentity::anonymous()
                        }
                    }
                }
                None => {
                    trace!("MCP HTTP auth: no Bearer token, using anonymous");
                    RequestIdentity::anonymous()
                }
            }
        } else {
            // Stdio/ZMQ transport — decode env token per-request
            match &self.stdio_token {
                Some(token) => {
                    match jwt::decode(token, &self.verifying_key) {
                        Ok(claims) => RequestIdentity::api_token(&claims.sub, "mcp"),
                        Err(e) => {
                            trace!("MCP stdio auth: token invalid ({}), using local identity", e);
                            RequestIdentity::local()
                        }
                    }
                }
                None => RequestIdentity::local(),
            }
        }
    }

    /// Dispatch a tool call by UUID with a specific identity
    async fn dispatch_tool(&self, uuid: &Uuid, args: Value, identity: RequestIdentity) -> Result<CallToolResult, ErrorData> {
        let entry = self.registry.get(uuid)
            .ok_or_else(|| ErrorData::invalid_request(format!("Unknown tool: {}", uuid), None))?;

        let ctx = ToolCallContext {
            args,
            signing_key: self.signing_key.clone(),
            zmq_context: self.context.clone(),
            identity,
            ctx: self.service_ctx.clone(),
        };

        let result = (entry.handler)(ctx).await
            .map_err(|e| ErrorData::internal_error(format!("Tool failed: {}", e), None))?;

        match result {
            ToolResult::Sync(value) => {
                Ok(CallToolResult::success(vec![Content::text(value.to_string())]))
            }
            ToolResult::Stream(mut handle) => {
                // Consume StreamHandle — DH, SUB, HMAC all handled internally
                let mut contents = Vec::new();
                while let Some(payload) = handle.recv_next()
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
        let identity = self.extract_identity(&context);
        async move {
            let uuid = Uuid::parse_str(&request.name)
                .map_err(|e| ErrorData::invalid_request(format!("Invalid UUID: {}", e), None))?;

            let args = match request.arguments {
                Some(map) => Value::Object(map),
                None => Value::Object(serde_json::Map::new()),
            };

            self.dispatch_tool(&uuid, args, identity).await
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// McpHandler Implementation (generated trait)
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait::async_trait(?Send)]
impl McpHandler for McpService {
    async fn handle_get_status(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        let loaded_model_count = {
            // Status check uses local identity (internal health check, no user context)
            let client = ModelZmqClient::new(self.signing_key.clone(), RequestIdentity::local());
            client.list().await
                .map(|models| models.len() as u32)
                .unwrap_or(0)
        };

        Ok(McpResponseVariant::GetStatusResult(ServiceStatus {
            is_running: true,
            loaded_model_count,
            is_authenticated: self.stdio_token.is_some(),
            authenticated_user: self.stdio_token.as_ref()
                .and_then(|t| jwt::decode(t, &self.verifying_key).ok())
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
        let tools: Vec<ToolDefinition> = self.registry.list().map(|entry| {
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

        // ZMQ transport: use envelope identity (already authenticated by ZMQ layer)
        let identity = RequestIdentity::api_token(ctx.user(), "mcp");
        let result = self.dispatch_tool(&uuid, args, identity).await;

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
                    result: "null".to_string(),
                    error_message: format!("{}", e),
                }))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService Implementation (internal control plane)
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait(?Send)]
impl ZmqService for McpService {
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

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = McpResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_string(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}
