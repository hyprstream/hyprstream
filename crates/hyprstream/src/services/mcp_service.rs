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
//! 3. Scopes via `Claims.has_scope(&Scope)` for each tool call

use crate::services::{ModelZmqClient, RegistryClient, RegistryZmqClient, PolicyClient};
use crate::services::model::ModelLoadConfig;
use crate::services::traits::CloneOptions;
use crate::services::generated::mcp_client_gen::{
    McpHandler, McpResponseVariant, ToolDefinitionData, dispatch_mcp,
};
use ed25519_dalek::{SigningKey, VerifyingKey};
use futures::future::BoxFuture;
use hyprstream_rpc::auth::{jwt, Claims, Scope};
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

fn register_tools() -> ToolRegistry {
    let mut reg = ToolRegistry::new();

    // model.load
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, b"model.load"),
        name: "model.load".into(),
        description: "Load a model into memory for inference".into(),
        args_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "model_ref": {"type": "string", "description": "Model reference in format name:branch"},
                "device": {"type": "string", "description": "Device to load model on (cpu, cuda:N, rocm:N)"},
                "max_context": {"type": "number", "description": "Maximum context length for KV cache"}
            },
            "required": ["model_ref"]
        }),
        required_scope: "write:model:*".into(),
        streaming: false,
        handler: Arc::new(|ctx| Box::pin(async move {
            let model_ref = ctx.args["model_ref"].as_str()
                .ok_or_else(|| anyhow::anyhow!("missing model_ref"))?;
            let max_context = ctx.args["max_context"].as_u64().map(|v| v as usize);
            let config = ModelLoadConfig { max_context, kv_quant: None };
            let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
            let result = client.load(model_ref, Some(&config)).await?;
            Ok(ToolResult::Sync(serde_json::to_value(&result)?))
        })),
    });

    // model.list
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, b"model.list"),
        name: "model.list".into(),
        description: "List all models currently loaded in memory".into(),
        args_schema: serde_json::json!({"type": "object", "properties": {}}),
        required_scope: "read:model:*".into(),
        streaming: false,
        handler: Arc::new(|ctx| Box::pin(async move {
            let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
            let models = client.list().await?;
            let models_json: Vec<Value> = models.into_iter().map(|m| serde_json::json!({
                "model_ref": m.model_ref,
                "endpoint": m.endpoint,
                "loaded_at": m.loaded_at,
                "last_used": m.last_used,
            })).collect();
            Ok(ToolResult::Sync(Value::Array(models_json)))
        })),
    });

    // model.unload
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, b"model.unload"),
        name: "model.unload".into(),
        description: "Unload a model from memory to free resources".into(),
        args_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "model_ref": {"type": "string", "description": "Model reference to unload"}
            },
            "required": ["model_ref"]
        }),
        required_scope: "write:model:*".into(),
        streaming: false,
        handler: Arc::new(|ctx| Box::pin(async move {
            let model_ref = ctx.args["model_ref"].as_str()
                .ok_or_else(|| anyhow::anyhow!("missing model_ref"))?
                .to_owned();
            let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
            client.unload(&model_ref).await?;
            Ok(ToolResult::Sync(serde_json::json!({"unloaded": model_ref})))
        })),
    });

    // registry.list
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, b"registry.list"),
        name: "registry.list".into(),
        description: "List all models available in the registry".into(),
        args_schema: serde_json::json!({"type": "object", "properties": {}}),
        required_scope: "read:registry:*".into(),
        streaming: false,
        handler: Arc::new(|ctx| Box::pin(async move {
            let client = RegistryZmqClient::new(ctx.signing_key, ctx.identity.clone());
            let models = client.list_models().await?;
            let models_json: Vec<Value> = models.into_iter().map(|m| {
                let path_str = m.path.to_string_lossy().to_string();
                serde_json::json!({
                    "display_name": m.display_name,
                    "model": m.model,
                    "branch": m.branch,
                    "path": path_str,
                    "is_dirty": m.is_dirty,
                    "driver": m.driver,
                })
            }).collect();
            Ok(ToolResult::Sync(Value::Array(models_json)))
        })),
    });

    // model.status
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, b"model.status"),
        name: "model.status".into(),
        description: "Get detailed status information about a model".into(),
        args_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "model_ref": {"type": "string", "description": "Model reference to check status"}
            },
            "required": ["model_ref"]
        }),
        required_scope: "read:model:*".into(),
        streaming: false,
        handler: Arc::new(|ctx| Box::pin(async move {
            let model_ref = ctx.args["model_ref"].as_str()
                .ok_or_else(|| anyhow::anyhow!("missing model_ref"))?
                .to_owned();
            let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());
            let status = client.status(&model_ref).await?;
            Ok(ToolResult::Sync(serde_json::json!({
                "model_ref": model_ref,
                "loaded": status.loaded,
                "endpoint": status.endpoint,
            })))
        })),
    });

    // registry.clone (streaming)
    reg.register(ToolEntry {
        uuid: Uuid::new_v5(&MCP_NS, b"registry.clone"),
        name: "registry.clone".into(),
        description: "Clone a model repository from a URL (streaming progress)".into(),
        args_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Repository URL to clone"},
                "name": {"type": "string", "description": "Optional local name for the model"}
            },
            "required": ["url"]
        }),
        required_scope: "write:registry:*".into(),
        streaming: true,
        handler: Arc::new(|ctx| Box::pin(async move {
            let url = ctx.args["url"].as_str()
                .ok_or_else(|| anyhow::anyhow!("missing url"))?;
            let name = ctx.args["name"].as_str();

            // Generate ephemeral keypair for DH key exchange
            let (client_secret, client_pubkey) = hyprstream_rpc::generate_ephemeral_keypair();
            let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

            // Call backend — get StreamStartedInfo
            let client = RegistryZmqClient::new(ctx.signing_key, ctx.identity.clone());
            let stream_info = client.clone_stream(
                url,
                name,
                &CloneOptions::default(),
                Some(client_pubkey_bytes),
            ).await?;

            // StreamHandle encapsulates EVERYTHING: DH, SUB, HMAC verify
            let handle = StreamHandle::new(
                &ctx.zmq_context,
                stream_info.stream_id,
                &stream_info.endpoint,
                &stream_info.server_pubkey,
                &client_secret,
                &client_pubkey_bytes,
            )?;

            Ok(ToolResult::Stream(handle))
        })),
    });

    reg
}

/// Register tools discovered from schema metadata.
///
/// Each service's schema_metadata() + scoped variants are iterated.
/// Tools that overlap with existing hand-coded tools (by name) are skipped.
fn register_schema_tools(reg: &mut ToolRegistry, existing_names: &std::collections::HashSet<String>) {
    use crate::services::generated::{
        model_client, registry_client, policy_client, inference_client,
    };

    // Helper: extract (name, description, params) tuples from any generated ParamSchema
    macro_rules! extract_params {
        ($methods:expr) => {
            $methods.iter().map(|m| {
                let params: Vec<(&str, &str, bool, &str)> = m.params.iter()
                    .map(|p| (p.name, p.type_name, p.required, p.description))
                    .collect();
                (m.name, m.description, params)
            }).collect::<Vec<_>>()
        };
    }

    // Top-level methods from all services
    let all_schemas: &[(&str, Vec<(&str, &str, Vec<(&str, &str, bool, &str)>)>)] = &[
        { let (svc, methods) = model_client::schema_metadata(); (svc, extract_params!(methods)) },
        { let (svc, methods) = registry_client::schema_metadata(); (svc, extract_params!(methods)) },
        { let (svc, methods) = policy_client::schema_metadata(); (svc, extract_params!(methods)) },
        { let (svc, methods) = inference_client::schema_metadata(); (svc, extract_params!(methods)) },
    ];

    for (service_name, methods) in all_schemas {
        for (method_name, description, params) in methods {
            let tool_name = format!("{service_name}.{method_name}");
            if existing_names.contains(&tool_name) {
                continue;
            }

            let json_schema = params_to_json_schema(params);
            let service_name = service_name.to_string();
            let method_name = method_name.to_string();
            let description = if description.is_empty() {
                format!("{service_name}::{method_name}")
            } else {
                description.to_string()
            };

            reg.register(ToolEntry {
                uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                name: tool_name.clone(),
                description,
                args_schema: json_schema,
                required_scope: format!("read:{service_name}:*"),
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
    }

    // Scoped: registry.repo.*
    {
        let (_, _, repo_methods) = registry_client::repo_schema_metadata();
        for method in repo_methods {
            let tool_name = format!("registry.repo.{}", method.name);
            if existing_names.contains(&tool_name) {
                continue;
            }

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
            reg.register(ToolEntry {
                uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                name: tool_name.clone(),
                description,
                args_schema: json_schema,
                required_scope: "read:registry:*".into(),
                streaming: false,
                handler: Arc::new(move |ctx| {
                    let method = method_name.clone();
                    Box::pin(async move {
                        let repo_id = ctx.args["repo_id"].as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing repo_id"))?;
                        let client = RegistryZmqClient::new(ctx.signing_key, ctx.identity.clone());
                        let gen_repo = client.gen.repo(repo_id);
                        let result = gen_repo.call_method(&method, &ctx.args).await?;
                        Ok(ToolResult::Sync(result))
                    })
                }),
            });
        }
    }

    // Scoped: model.ttt.*, model.peft.*, model.infer.*
    {
        // Streaming methods per scope — return StreamInfo, need DH key exchange
        let streaming_methods: std::collections::HashMap<&str, &[&str]> = [
            ("ttt", &["train_stream"][..]),
            ("infer", &["generate_stream"][..]),
        ].into_iter().collect();

        let scoped_metadata: &[(&str, fn() -> (&'static str, &'static str, &'static [model_client::MethodSchema]))] = &[
            ("ttt", model_client::ttt_schema_metadata),
            ("peft", model_client::peft_schema_metadata),
            ("infer", model_client::infer_schema_metadata),
        ];

        for &(scope_name, metadata_fn) in scoped_metadata {
            let (_, _, methods) = metadata_fn();
            let scope_streaming = streaming_methods.get(scope_name).copied().unwrap_or(&[]);

            for method in methods {
                let tool_name = format!("model.{scope_name}.{}", method.name);
                if existing_names.contains(&tool_name) {
                    continue;
                }

                let is_streaming = scope_streaming.contains(&method.name);

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

                if is_streaming {
                    // Streaming handler: generate ephemeral keypair, call with pubkey, return StreamHandle
                    reg.register(ToolEntry {
                        uuid: Uuid::new_v5(&MCP_NS, tool_name.as_bytes()),
                        name: tool_name.clone(),
                        description,
                        args_schema: json_schema,
                        required_scope: "read:model:*".into(),
                        streaming: true,
                        handler: Arc::new(move |ctx| {
                            let method = method_name.clone();
                            let scope = scope.clone();
                            Box::pin(async move {
                                let model_ref = ctx.args["model_ref"].as_str()
                                    .ok_or_else(|| anyhow::anyhow!("missing model_ref"))?;

                                // Generate ephemeral keypair for DH key exchange
                                let (client_secret, client_pubkey) = hyprstream_rpc::generate_ephemeral_keypair();
                                let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

                                let client = ModelZmqClient::new(ctx.signing_key, ctx.identity.clone());

                                // Call the streaming method via manual wrapper (with ephemeral pubkey)
                                let stream_info = match (scope.as_str(), method.as_str()) {
                                    ("ttt", "train_stream") => {
                                        let input = ctx.args["input"].as_str().unwrap_or("");
                                        let gradient_steps = ctx.args["gradient_steps"].as_u64().unwrap_or(3) as u32;
                                        let learning_rate = ctx.args["learning_rate"].as_f64().unwrap_or(0.0) as f32;
                                        let auto_commit = ctx.args["auto_commit"].as_bool().unwrap_or(false);
                                        client.train_step_stream(
                                            model_ref, input, gradient_steps, learning_rate,
                                            auto_commit, client_pubkey_bytes,
                                        ).await?
                                    }
                                    ("infer", "generate_stream") => {
                                        let request = crate::services::mcp_service::parse_generation_request_from_args(&ctx.args)?;
                                        client.infer_stream(model_ref, &request, client_pubkey_bytes).await?
                                    }
                                    _ => anyhow::bail!("Unknown streaming method: {}.{}", scope, method),
                                };

                                let handle = StreamHandle::new(
                                    &ctx.zmq_context,
                                    stream_info.stream_id,
                                    &stream_info.endpoint,
                                    &stream_info.server_pubkey,
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
                        required_scope: "read:model:*".into(),
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

/// Convert method params to a JSON Schema for tool arguments.
///
/// Takes params as (name, type_name, required, description) tuples — works with any generated module's ParamSchema
/// since they all have the same layout.
fn params_to_json_schema(params: &[(&str, &str, bool, &str)]) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for &(name, type_name, is_required, description) in params {
        let json_type = match type_name {
            "Text" => "string",
            "Bool" => "boolean",
            "UInt32" | "UInt64" | "Int32" | "Int64" => "integer",
            "Float32" | "Float64" => "number",
            "Data" => "string",
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

/// Parse a GenerationRequest from MCP tool call JSON args.
pub(crate) fn parse_generation_request_from_args(args: &Value) -> anyhow::Result<crate::config::GenerationRequest> {
    use crate::config::{GenerationRequest, TemplatedPrompt};
    Ok(GenerationRequest {
        prompt: TemplatedPrompt::new(args["prompt"].as_str().unwrap_or("").to_string()),
        max_tokens: args["max_tokens"].as_u64().unwrap_or(256) as usize,
        temperature: args["temperature"].as_f64().unwrap_or(0.7) as f32,
        top_p: args["top_p"].as_f64().unwrap_or(0.9) as f32,
        top_k: args["top_k"].as_u64().map(|v| v as usize),
        repeat_penalty: args["repeat_penalty"].as_f64().unwrap_or(1.1) as f32,
        repeat_last_n: args["repeat_last_n"].as_u64().unwrap_or(64) as usize,
        stop_tokens: args["stop_tokens"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        seed: args["seed"].as_u64().map(|v| v as u32),
        images: Vec::new(),
        timeout: args["timeout_ms"].as_u64(),
        collect_metrics: false,
    })
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
            let client = RegistryZmqClient::new(signing_key, identity);
            client.gen.call_method(method, &ctx.args).await
        }
        "policy" => {
            let client = PolicyClient::new(signing_key, identity);
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
    /// UUID-keyed tool registry
    registry: Arc<ToolRegistry>,
    /// JWT claims extracted from token (if provided)
    claims: Option<Claims>,
    /// Verifying key for JWT validation
    #[allow(dead_code)]
    verifying_key: VerifyingKey,
    // === ZmqService infrastructure ===
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
    /// ServiceContext for typed_client() / client() access
    service_ctx: Option<Arc<ServiceContext>>,
}

impl McpService {
    /// Create a new McpService with JWT authentication
    pub fn new(config: McpConfig) -> anyhow::Result<Self> {
        let claims = match std::env::var("HYPRSTREAM_TOKEN") {
            Ok(token) => {
                jwt::decode(&token, &config.verifying_key)
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("Invalid HYPRSTREAM_TOKEN: {}", e))?
            }
            Err(_) => {
                tracing::warn!("No HYPRSTREAM_TOKEN provided - tools will require authentication");
                None
            }
        };

        if let Some(ref claims) = claims {
            tracing::info!(
                "McpService authenticated as: {} with scopes: {:?}",
                claims.sub,
                claims.scopes
            );
        }

        // Register hand-coded tools first, then discover schema-driven tools
        let mut tool_reg = register_tools();
        let existing_names: std::collections::HashSet<String> = tool_reg.list()
            .map(|e| e.name.clone())
            .collect();
        register_schema_tools(&mut tool_reg, &existing_names);

        tracing::info!(
            "McpService registered {} tools ({} hand-coded, {} schema-discovered)",
            tool_reg.by_uuid.len(),
            existing_names.len(),
            tool_reg.by_uuid.len() - existing_names.len(),
        );

        Ok(Self {
            registry: Arc::new(tool_reg),
            claims,
            verifying_key: config.verifying_key,
            context: config.zmq_context.clone(),
            transport: config.transport,
            signing_key: config.signing_key,
            service_ctx: config.ctx,
        })
    }

    /// Check if current claims have required scope
    fn check_scope(&self, required: &str) -> Result<(), ErrorData> {
        let scope = Scope::parse(required)
            .map_err(|e| ErrorData::internal_error(format!("Invalid scope: {}", e), None))?;

        match &self.claims {
            Some(claims) if claims.has_scope(&scope) => Ok(()),
            Some(claims) => Err(ErrorData::invalid_request(
                format!("Insufficient scope: {} (have: {:?})", required, claims.scopes),
                None,
            )),
            None => Err(ErrorData::invalid_request(
                "No authentication provided. Set HYPRSTREAM_TOKEN environment variable.".to_string(),
                None,
            )),
        }
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
                    read_only_hint: Some(entry.required_scope.starts_with("read:")),
                    destructive_hint: Some(!entry.required_scope.starts_with("read:")),
                    open_world_hint: Some(false),
                    idempotent_hint: Some(true),
                }),
                icons: None,
                meta: None,
            }
        }).collect()
    }

    /// Dispatch a tool call by UUID
    async fn dispatch_tool(&self, uuid: &Uuid, args: Value) -> Result<CallToolResult, ErrorData> {
        let entry = self.registry.get(uuid)
            .ok_or_else(|| ErrorData::invalid_request(format!("Unknown tool: {}", uuid), None))?;

        self.check_scope(&entry.required_scope)?;

        // Propagate authenticated identity to backend services (not local())
        let identity = match &self.claims {
            Some(claims) => RequestIdentity::api_token(&claims.sub, "mcp"),
            None => RequestIdentity::local(),
        };
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
                 Set HYPRSTREAM_TOKEN environment variable to enable tools. \
                 Get a token: hyprstream policy token create --user <user> --scopes '<scopes>' --expires 90d"
                    .into(),
            ),
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult {
            meta: None,
            tools: self.tools_list(),
            next_cursor: None,
        }))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + Send + '_ {
        async move {
            let uuid = Uuid::parse_str(&request.name)
                .map_err(|e| ErrorData::invalid_request(format!("Invalid UUID: {}", e), None))?;

            let args = match request.arguments {
                Some(map) => Value::Object(map),
                None => Value::Object(serde_json::Map::new()),
            };

            self.dispatch_tool(&uuid, args).await
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// McpHandler Implementation (generated trait)
// ═══════════════════════════════════════════════════════════════════════════════

impl McpHandler for McpService {
    fn handle_get_status(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        let loaded_model_count = tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                // Status check uses local identity (internal health check, no user context)
                let client = ModelZmqClient::new(self.signing_key.clone(), RequestIdentity::local());
                client.list().await
                    .map(|models| models.len() as u32)
                    .unwrap_or(0)
            })
        });

        let scopes: Vec<String> = self.claims
            .as_ref()
            .map(|c| c.scopes.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        Ok(McpResponseVariant::GetStatusResult {
            is_running: true,
            loaded_model_count,
            is_authenticated: self.claims.is_some(),
            authenticated_user: self.claims.as_ref()
                .map(|c| c.sub.clone())
                .unwrap_or_default(),
            scopes,
        })
    }

    fn handle_list_tools(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        let tools: Vec<ToolDefinitionData> = self.registry.list().map(|entry| {
            ToolDefinitionData {
                name: entry.uuid.to_string(),
                description: entry.description.clone(),
                is_read_only: entry.required_scope.starts_with("read:"),
                is_destructive: !entry.required_scope.starts_with("read:"),
                required_scope: entry.required_scope.clone(),
                argument_schema: entry.args_schema.to_string(),
            }
        }).collect();

        Ok(McpResponseVariant::ListToolsResult { tools })
    }

    fn handle_get_metrics(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
    ) -> anyhow::Result<McpResponseVariant> {
        Ok(McpResponseVariant::GetMetricsResult {
            total_calls: 0,
            calls_per_tool: Vec::new(),
            average_call_duration_ms: 0.0,
            uptime_seconds: 0.0,
        })
    }

    fn handle_call_tool(
        &self,
        _ctx: &crate::services::EnvelopeContext,
        _request_id: u64,
        tool_name: &str,
        arguments: &str,
        _caller_identity: &str,
    ) -> anyhow::Result<McpResponseVariant> {
        let uuid = Uuid::parse_str(tool_name)
            .map_err(|e| anyhow::anyhow!("Invalid tool UUID '{}': {}", tool_name, e))?;

        let args: Value = if arguments.is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(arguments)?
        };

        let result = tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(self.dispatch_tool(&uuid, args))
        });

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
                Ok(McpResponseVariant::CallToolResult {
                    success: true,
                    result: text,
                    error_message: String::new(),
                })
            }
            Err(e) => {
                Ok(McpResponseVariant::CallToolResult {
                    success: false,
                    result: "null".to_string(),
                    error_message: format!("{}", e),
                })
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService Implementation (internal control plane)
// ═══════════════════════════════════════════════════════════════════════════════

impl ZmqService for McpService {
    fn handle_request(&self, ctx: &crate::services::EnvelopeContext, payload: &[u8]) -> anyhow::Result<Vec<u8>> {
        trace!(
            "McpService request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_mcp(self, ctx, payload)
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
}
