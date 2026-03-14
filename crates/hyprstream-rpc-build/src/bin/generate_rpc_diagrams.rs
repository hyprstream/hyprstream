//! Generate Mermaid architecture diagrams from Cap'n Proto schema metadata
//! and auto-detected service dependencies.
//!
//! Usage:
//!   cargo run --bin generate-rpc-diagrams -- [--metadata-dir PATH] [--services-dir PATH] [-o PATH]
//!
//! Reads:
//!   - *_metadata.json files (from capnp build output) for method counts + scopes
//!   - services/factories.rs for service registrations, transport types, QUIC support
//!   - services/*.rs for client imports (auto-detected call graph)
// CLI binary — relaxed lints for a dev tool (not library code)
#![allow(clippy::all, clippy::pedantic, clippy::restriction, dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// ── Types ───────────────────────────────────────────────────────────────────

fn scope_name(ordinal: u64) -> &'static str {
    match ordinal {
        0 => "query",
        1 => "write",
        2 => "manage",
        3 => "infer",
        4 => "train",
        5 => "serve",
        6 => "context",
        7 => "subscribe",
        8 => "publish",
        _ => "unknown",
    }
}

fn title_case(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

#[derive(Debug)]
struct ServiceMeta {
    name: String,
    transport: Transport,
    methods: usize,
    scopes: BTreeMap<String, usize>,
    dependencies: BTreeSet<DepEdge>,
    has_quic: bool,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct DepEdge {
    target: String,
    client_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Transport {
    ZmqRep,
    Http,
    Proxy,
}

impl Transport {
    fn label(&self) -> &'static str {
        match self {
            Transport::ZmqRep => "ZMQ REQ/REP",
            Transport::Http => "HTTP",
            Transport::Proxy => "ZMQ Proxy",
        }
    }
}

// ── Client type → target service mapping ────────────────────────────────────

/// Maps a client/type name found in source to the service it connects to.
fn client_to_service(client_type: &str) -> Option<&'static str> {
    match client_type {
        "PolicyClient" => Some("policy"),
        "RegistryClient" => Some("registry"),
        "ModelZmqClient" => Some("model"),
        "InferenceZmqClient" => Some("inference"),
        "NotificationClient" => Some("notification"),
        "NotificationPublisher" => Some("notification"),
        "WorktreeClient" => Some("registry"),  // scoped sub-client of registry
        "RepositoryClient" => Some("registry"), // scoped sub-client of registry
        "CtlClient" => Some("registry"),        // scoped sub-client of registry
        "DiscoveryClient" => Some("discovery"),
        "WorkerClient" => Some("worker"),
        // Streaming: StreamChannel/StreamInfo indicate the service publishes
        // through StreamService (PUSH → XPUB relay)
        "StreamChannel" => Some("streams"),
        "StreamInfo" => Some("streams"),
        _ => None,
    }
}

/// All known client/type names to scan for.
const CLIENT_TYPES: &[&str] = &[
    "PolicyClient",
    "RegistryClient",
    "ModelZmqClient",
    "InferenceZmqClient",
    "NotificationClient",
    "NotificationPublisher",
    "WorktreeClient",
    "RepositoryClient",
    "CtlClient",
    "DiscoveryClient",
    "WorkerClient",
    // Streaming types — presence means service publishes via StreamService
    "StreamChannel",
    "StreamInfo",
];

// ── Source file → service name mapping ──────────────────────────────────────

/// Maps a source filename to the service it implements.
fn filename_to_service(filename: &str) -> Option<&'static str> {
    match filename {
        "factories.rs" => None, // handled separately
        "mod.rs" | "core.rs" | "generated.rs" => None, // infrastructure, not a service
        "worktree_helpers.rs" => None, // helper, not a service
        "registry.rs" => Some("registry"),
        "model.rs" => Some("model"),
        "inference.rs" => Some("inference"),
        "notification.rs" => Some("notification"),
        "discovery.rs" => Some("discovery"),
        "worker.rs" => Some("worker"),
        "mcp_service.rs" => Some("mcp"),
        "oai.rs" => Some("oai"),
        "flight.rs" => Some("flight"),
        "tui.rs" | "tui_service.rs" => Some("tui"),
        "stream.rs" => Some("streams"),
        "policy.rs" => Some("policy"),
        "editing.rs" => Some("editing"),
        _ => {
            // oauth/mod.rs, oauth/state.rs etc
            if filename.starts_with("oauth") {
                Some("oauth")
            } else {
                None
            }
        }
    }
}

// ── Method-level call graph types ────────────────────────────────────────────

/// A method defined in a service's schema.
#[derive(Debug, Clone)]
struct MethodDef {
    /// camelCase name from schema (e.g., "generateStream")
    schema_name: String,
    /// snake_case name as it appears in Rust code (e.g., "generate_stream")
    rust_name: String,
    /// Scope action (query, write, manage, infer, train, etc.)
    scope: String,
    /// Human description from $mcpDescription
    description: String,
    /// Which service this method belongs to
    service: String,
    /// Parent struct name (e.g., "ModelRequest", "RepositoryRequest")
    parent: String,
}

/// A detected method-level call from one caller to a target service method.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct MethodCall {
    /// Caller module (service name or "cli:handlers", "server:routes", etc.)
    caller: String,
    /// Target service
    target_service: String,
    /// Target method (snake_case)
    target_method: String,
    /// Source file path (relative)
    source_file: String,
    /// Line number
    line: usize,
}

/// camelCase → snake_case conversion (e.g., "generateStream" → "generate_stream")
fn to_snake_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

/// Parse all metadata files to build a method registry.
fn build_method_registry(metadata_dir: &Path) -> Vec<MethodDef> {
    let mut methods = Vec::new();

    let entries = match fs::read_dir(metadata_dir) {
        Ok(e) => e,
        Err(_) => return methods,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let is_metadata = path
            .file_name()
            .map(|n| n.to_string_lossy().ends_with("_metadata.json"))
            .unwrap_or(false);
        if !is_metadata {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(_) => continue,
        };

        let service = match json.get("service").and_then(|s| s.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };

        let structs = match json.get("structs").and_then(|s| s.as_array()) {
            Some(s) => s,
            None => continue,
        };

        for s in structs {
            let struct_name = s.get("name").and_then(|n| n.as_str()).unwrap_or("");

            for field in s.get("fields").and_then(|f| f.as_array()).into_iter().flatten() {
                let disc = field.get("discriminant").and_then(|d| d.as_u64()).unwrap_or(65535);
                if disc == 65535 {
                    continue; // not a union variant
                }

                let name = match field.get("name").and_then(|n| n.as_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };

                let mut scope = String::new();
                let mut description = String::new();

                for ann in field.get("annotations").and_then(|a| a.as_array()).into_iter().flatten() {
                    let ann_name = ann.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    if ann_name == "mcpScope" {
                        scope = ann.get("value")
                            .and_then(|v| {
                                v.as_object()
                                    .and_then(|o| o.get("enum_ordinal"))
                                    .and_then(|o| o.as_u64())
                                    .map(|ord| scope_name(ord).to_string())
                                    .or_else(|| v.as_str().map(String::from))
                            })
                            .unwrap_or_default();
                    }
                    if ann_name == "mcpDescription" {
                        description = ann.get("value")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                    }
                }

                methods.push(MethodDef {
                    rust_name: to_snake_case(&name),
                    schema_name: name,
                    scope,
                    description,
                    service: service.clone(),
                    parent: struct_name.to_string(),
                });
            }
        }
    }

    methods
}

/// Scan all Rust source files for method-level calls to known service methods.
fn scan_method_calls(
    src_root: &Path,
    method_registry: &[MethodDef],
) -> Vec<MethodCall> {
    let mut calls = Vec::new();

    // Generic Rust method names that produce too many false positives.
    // These are only matched when preceded by a known client variable pattern.
    const GENERIC_METHODS: &[&str] = &[
        "clone", "get", "list", "remove", "status", "check", "register",
        "load", "unload", "connect", "attach", "subscribe", "unsubscribe",
        "success", "repo", "worktree", "get_by_name",
    ];

    // Build a lookup: snake_case method name → Vec<MethodDef>
    // (multiple services might have same method name like "status", "list")
    let mut method_lookup: BTreeMap<String, Vec<&MethodDef>> = BTreeMap::new();
    for m in method_registry {
        method_lookup.entry(m.rust_name.clone()).or_default().push(m);
    }

    let generic_set: std::collections::HashSet<&str> = GENERIC_METHODS.iter().copied().collect();

    // Walk all .rs files under src_root
    walk_rs_files(src_root, &mut |path| {
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };

        let relative = path.strip_prefix(src_root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Determine caller identity from file path
        let caller = path_to_caller(&relative);
        if caller.is_empty() {
            return;
        }

        // Scan for patterns like:
        //   <ident>_client.<method>(    — cross-service call via named client
        //   <ident>.<method>(           — via client variable
        // We look for known method names preceded by a dot.
        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            // Skip comments
            if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("*") {
                continue;
            }

            for (method_name, defs) in &method_lookup {
                let is_generic = generic_set.contains(method_name.as_str());

                // Look for .<method_name>( pattern
                let pattern = format!(".{}(", method_name);
                if !trimmed.contains(&pattern) {
                    continue;
                }

                // For generic method names, require a client variable prefix
                // (e.g., "policy_client.check(" or "model_client.load(")
                if is_generic && !has_client_prefix(trimmed, method_name) {
                    continue;
                }

                // Disambiguate: figure out which client type is being called.
                if let Some(target_service) = resolve_call_target(trimmed, method_name, defs, &caller) {
                    calls.push(MethodCall {
                        caller: caller.clone(),
                        target_service,
                        target_method: method_name.clone(),
                        source_file: relative.clone(),
                        line: line_num + 1,
                    });
                }
            }
        }
    });

    // Deduplicate (same caller → same target method, keep first occurrence)
    calls.sort();
    calls.dedup_by(|a, b| {
        a.caller == b.caller && a.target_service == b.target_service && a.target_method == b.target_method
    });

    calls
}

/// Check if a line has a known client variable prefix before the method call.
/// Matches patterns like: policy_client.check(, model_client.load(, registry.list(
fn has_client_prefix(line: &str, method_name: &str) -> bool {
    let pattern = format!(".{}(", method_name);
    let pos = match line.find(&pattern) {
        Some(p) => p,
        None => return false,
    };

    let before = &line[..pos];
    let before_trimmed = before.trim_end();

    // Known client variable patterns
    const CLIENT_PREFIXES: &[&str] = &[
        "policy_client", "registry_client", "model_client", "worker_client",
        "inference_client", "notification_client", "discovery_client",
        "tui_client", "mcp_client",
        // Short forms used in some code
        "policy", "registry", "model", "worker", "inference",
        "notification", "discovery",
        // Scoped sub-clients
        "worktree", "repo",
    ];

    for prefix in CLIENT_PREFIXES {
        if before_trimmed.ends_with(prefix) {
            return true;
        }
    }
    // Also match `.gen.infer(...).method(` and `.gen.ttt(...).method(`
    // (these are internal dispatch, but we'll let resolve_call_target filter them)
    if before_trimmed.ends_with(')') {
        // Chained call like `.generate_stream(` after `.infer(model_ref)`
        return true;
    }
    false
}

/// Resolve which service a method call targets based on context.
fn resolve_call_target(
    line: &str,
    method_name: &str,
    defs: &[&MethodDef],
    caller: &str,
) -> Option<String> {
    // If only one service defines this method, it's unambiguous
    if defs.len() == 1 {
        let target = &defs[0].service;
        // Don't report self-calls
        if target == caller {
            return None;
        }
        return Some(target.clone());
    }

    // Disambiguate by looking at what precedes .<method>(
    // Common patterns:
    //   policy_client.check(     → policy
    //   registry_client.list(    → registry
    //   model_client.load(       → model
    //   self.gen.load(           → self (skip)
    //   worker_client.stop(      → worker

    let pattern = format!(".{}(", method_name);
    let pos = match line.find(&pattern) {
        Some(p) => p,
        None => return None,
    };

    let before = &line[..pos];
    let before_trimmed = before.trim_end();

    // Extract the variable/expression before the dot
    let var = before_trimmed.rsplit(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .next()
        .unwrap_or("");

    // self.gen calls are internal dispatch (not cross-service)
    if var.contains("self.gen") || var.ends_with(".gen") {
        return None;
    }

    // Match variable name to client type
    for def in defs {
        let target = &def.service;
        if target == caller {
            continue;
        }
        if var.contains("policy") && target == "policy" { return Some(target.clone()); }
        if var.contains("registry") && target == "registry" { return Some(target.clone()); }
        if var.contains("model") && target == "model" { return Some(target.clone()); }
        if var.contains("worker") && target == "worker" { return Some(target.clone()); }
        if var.contains("inference") && target == "inference" { return Some(target.clone()); }
        if var.contains("notif") && target == "notification" { return Some(target.clone()); }
        if var.contains("discovery") && target == "discovery" { return Some(target.clone()); }
    }

    // Fallback: if caller is known to depend on a specific service, prefer that
    None
}

/// Map a relative source path to a caller identity.
fn path_to_caller(relative: &str) -> String {
    // services/model.rs → model
    // services/mcp_service.rs → mcp
    // services/oauth/mod.rs → oauth
    // cli/handlers.rs → cli:handlers
    // cli/shell_handlers.rs → cli:shell
    // server/routes/openai.rs → server:openai
    // tui/service.rs → tui

    if relative.starts_with("services/") {
        let file = relative.trim_start_matches("services/");
        if file.starts_with("oauth") {
            return "oauth".to_string();
        }
        let stem = file.trim_end_matches(".rs")
            .trim_end_matches("/mod");
        return match stem {
            "factories" | "mod" | "core" | "generated" | "worktree_helpers" | "rpc_types" | "callback" => String::new(),
            "mcp_service" => "mcp".to_string(),
            "oai" => "oai".to_string(),
            other => other.to_string(),
        };
    }

    if relative.starts_with("cli/") {
        let stem = relative.trim_start_matches("cli/").trim_end_matches(".rs");
        return format!("cli:{}", stem.replace("_handlers", ""));
    }

    if relative.starts_with("server/") {
        let stem = relative.trim_start_matches("server/")
            .trim_start_matches("routes/")
            .trim_end_matches(".rs");
        return format!("server:{stem}");
    }

    if relative.starts_with("tui/") {
        return "tui".to_string();
    }

    String::new()
}

/// Recursively walk .rs files.
fn walk_rs_files(dir: &Path, callback: &mut dyn FnMut(&Path)) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, callback);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            callback(&path);
        }
    }
}

/// Generate the method-level call graph diagram.
fn generate_call_graph(calls: &[MethodCall], method_registry: &[MethodDef]) -> String {
    let mut out = String::new();

    // Build a method scope lookup for coloring
    let method_scope: BTreeMap<(&str, &str), &str> = method_registry.iter()
        .map(|m| ((m.service.as_str(), m.rust_name.as_str()), m.scope.as_str()))
        .collect();

    out.push_str("```mermaid\ngraph LR\n");

    // Group calls by caller → target_service
    let mut grouped: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
    for call in calls {
        grouped
            .entry((call.caller.as_str(), call.target_service.as_str()))
            .or_default()
            .push(call.target_method.as_str());
    }

    // Collect unique callers and target services as nodes
    let mut callers: BTreeSet<&str> = BTreeSet::new();
    let mut targets: BTreeSet<&str> = BTreeSet::new();
    for ((caller, target), _) in &grouped {
        callers.insert(caller);
        targets.insert(target);
    }

    // Caller subgraphs (group by category)
    let svc_callers: Vec<&&str> = callers.iter().filter(|c| !c.contains(':')).collect();
    let cli_callers: Vec<&&str> = callers.iter().filter(|c| c.starts_with("cli:")).collect();
    let server_callers: Vec<&&str> = callers.iter().filter(|c| c.starts_with("server:")).collect();

    if !svc_callers.is_empty() {
        out.push_str("\n    subgraph svc_callers [\"Services\"]\n");
        out.push_str("        direction TB\n");
        for c in &svc_callers {
            let id = caller_id(c);
            out.push_str(&format!("        {id}[\"{}\"]\n", title_case(c)));
        }
        out.push_str("    end\n");
    }

    if !cli_callers.is_empty() {
        out.push_str("\n    subgraph cli_callers [\"CLI\"]\n");
        out.push_str("        direction TB\n");
        for c in &cli_callers {
            let id = caller_id(c);
            let label = c.trim_start_matches("cli:");
            out.push_str(&format!("        {id}[\"cli:{label}\"]\n"));
        }
        out.push_str("    end\n");
    }

    if !server_callers.is_empty() {
        out.push_str("\n    subgraph server_callers [\"HTTP Routes\"]\n");
        out.push_str("        direction TB\n");
        for c in &server_callers {
            let id = caller_id(c);
            let label = c.trim_start_matches("server:");
            out.push_str(&format!("        {id}[\"server:{label}\"]\n"));
        }
        out.push_str("    end\n");
    }

    // Target service + method nodes
    for target in &targets {
        out.push_str(&format!("\n    subgraph {target}_methods [\"{} Methods\"]\n", title_case(target)));
        out.push_str("        direction TB\n");

        // Collect all methods called on this target
        let mut called_methods: BTreeSet<&str> = BTreeSet::new();
        for ((_, t), methods) in &grouped {
            if t == target {
                for m in methods {
                    called_methods.insert(m);
                }
            }
        }

        for method in &called_methods {
            let node_id = format!("{target}_{method}");
            let scope = method_scope.get(&(target, method)).copied().unwrap_or("");
            let scope_suffix = if scope.is_empty() { String::new() } else { format!(" [{scope}]") };
            out.push_str(&format!("        {node_id}[\"{method}{scope_suffix}\"]\n"));
        }
        out.push_str("    end\n");
    }

    // Edges
    out.push_str("\n    %% Method-level calls\n");
    for ((caller, target), methods) in &grouped {
        let caller_node = caller_id(caller);
        for method in methods {
            let target_node = format!("{target}_{method}");
            out.push_str(&format!("    {caller_node} --> {target_node}\n"));
        }
    }

    out.push_str("```\n");
    out
}

/// Generate a stable node ID from a caller string.
fn caller_id(caller: &str) -> String {
    caller.replace(':', "_").replace('.', "_")
}

// ── Schema metadata parsing ─────────────────────────────────────────────────

fn parse_metadata(path: &Path) -> Option<(String, usize, BTreeMap<String, usize>)> {
    let content = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let service = json.get("service")?.as_str()?.to_string();
    let mut methods = 0usize;
    let mut scopes = BTreeMap::new();

    for s in json.get("structs")?.as_array()? {
        for field in s.get("fields").and_then(|f| f.as_array()).into_iter().flatten() {
            for ann in field.get("annotations").and_then(|a| a.as_array()).into_iter().flatten() {
                if ann.get("name").and_then(|n| n.as_str()) == Some("mcpScope") {
                    methods += 1;
                    let scope = ann
                        .get("value")
                        .and_then(|v| {
                            v.as_object()
                                .and_then(|o| o.get("enum_ordinal"))
                                .and_then(|o| o.as_u64())
                                .map(|ord| scope_name(ord).to_string())
                                .or_else(|| v.as_str().map(String::from))
                        })
                        .unwrap_or_else(|| "unknown".into());
                    *scopes.entry(scope).or_insert(0) += 1;
                }
            }
        }
    }

    Some((service, methods, scopes))
}

// ── Factory scanning ────────────────────────────────────────────────────────

/// Parse factories.rs to extract service registrations with transport + QUIC.
fn scan_factories(factories_path: &Path) -> Vec<ServiceMeta> {
    let content = match fs::read_to_string(factories_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Warning: cannot read {}: {e}", factories_path.display());
            return Vec::new();
        }
    };

    let mut services = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Detect #[service_factory("name"...)]
        if line.starts_with("#[service_factory(") {
            if let Some(name) = extract_quoted(line) {
                // Scan the function body to determine transport + QUIC
                let (transport, has_quic) = scan_factory_body(&lines, i + 1);
                services.push(ServiceMeta {
                    name,
                    transport,
                    methods: 0,
                    scopes: BTreeMap::new(),
                    dependencies: BTreeSet::new(),
                    has_quic,
                });
            }
        }
        i += 1;
    }

    // Also scan factories.rs itself for client usage per factory function.
    // We split by factory boundaries and detect clients within each.
    scan_factory_dependencies(&content, &mut services);

    services
}

/// Extract the first quoted string from a line.
fn extract_quoted(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

/// Scan a factory function body to detect transport type and QUIC support.
fn scan_factory_body(lines: &[&str], start: usize) -> (Transport, bool) {
    let mut transport = Transport::ZmqRep;
    let mut has_quic = false;
    let mut brace_depth = 0;
    let mut in_body = false;

    for line in &lines[start..] {
        let trimmed = line.trim();

        for ch in trimmed.chars() {
            if ch == '{' { brace_depth += 1; in_body = true; }
            if ch == '}' { brace_depth -= 1; }
        }

        if in_body {
            if trimmed.contains("into_spawnable_quic") {
                has_quic = true;
            }
            if trimmed.contains("ProxyService") || trimmed.contains("XPubProxy")
                || trimmed.contains("PullToXPub") {
                transport = Transport::Proxy;
            }
            if trimmed.contains("OAIService") || trimmed.contains("ServerState")
                || trimmed.contains("FlightService") || trimmed.contains("OAuthService")
                || trimmed.contains("axum") || trimmed.contains("Axum")
                || trimmed.contains("FlightSqlService") {
                transport = Transport::Http;
            }

            if brace_depth == 0 {
                break;
            }
        }
    }

    (transport, has_quic)
}

/// Scan factories.rs to detect which clients each factory function creates.
fn scan_factory_dependencies(content: &str, services: &mut [ServiceMeta]) {
    // Split content into factory function blocks.
    // Each block starts at #[service_factory and ends at the next one (or EOF).
    let lines: Vec<&str> = content.lines().collect();
    let mut factory_ranges: Vec<(String, usize, usize)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if line.trim().starts_with("#[service_factory(") {
            if let Some(name) = extract_quoted(line.trim()) {
                // Close previous range
                if let Some(last) = factory_ranges.last_mut() {
                    last.2 = i;
                }
                factory_ranges.push((name, i, lines.len()));
            }
        }
    }

    for (name, start, end) in &factory_ranges {
        let block: String = lines[*start..*end].join("\n");
        let svc = match services.iter_mut().find(|s| &s.name == name) {
            Some(s) => s,
            None => continue,
        };

        for client_type in CLIENT_TYPES {
            if block.contains(client_type) {
                if let Some(target) = client_to_service(client_type) {
                    // Don't add self-dependency
                    if target != svc.name {
                        svc.dependencies.insert(DepEdge {
                            target: target.to_string(),
                            client_type: client_type.to_string(),
                        });
                    }
                }
            }
        }
    }
}

// ── Service source scanning ─────────────────────────────────────────────────

/// Scan services/*.rs files for client type usage to build call graph.
fn scan_service_sources(services_dir: &Path, services: &mut [ServiceMeta]) {
    let scan_dir = |dir: &Path| {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let mut results: Vec<(String, BTreeSet<DepEdge>)> = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                // Handle subdirectories like oauth/
                let dirname = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                let service_name = match dirname {
                    "oauth" => "oauth",
                    _ => continue,
                };

                // Scan all .rs files in the subdirectory
                let mut deps = BTreeSet::new();
                if let Ok(sub_entries) = fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        let sub_path = sub_entry.path();
                        if sub_path.extension().map(|e| e == "rs").unwrap_or(false) {
                            if let Ok(content) = fs::read_to_string(&sub_path) {
                                find_client_deps(&content, service_name, &mut deps);
                            }
                        }
                    }
                }
                if !deps.is_empty() {
                    results.push((service_name.to_string(), deps));
                }
                continue;
            }

            if path.extension().map(|e| e == "rs").unwrap_or(false) {
                let filename = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                let service_name = match filename_to_service(filename) {
                    Some(name) => name,
                    None => continue,
                };

                let content = match fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let mut deps = BTreeSet::new();
                find_client_deps(&content, service_name, &mut deps);

                if !deps.is_empty() {
                    results.push((service_name.to_string(), deps));
                }
            }
        }
        results
    };

    let mut all_deps = scan_dir(services_dir);

    // Also scan tui/ directory for TUI service dependencies
    let tui_dir = services_dir.parent().map(|p| p.join("tui"));
    if let Some(ref td) = tui_dir {
        if td.is_dir() {
            let mut tui_deps = BTreeSet::new();
            walk_rs_files(td, &mut |path| {
                if let Ok(content) = fs::read_to_string(path) {
                    find_client_deps(&content, "tui", &mut tui_deps);
                }
            });
            if !tui_deps.is_empty() {
                all_deps.push(("tui".to_string(), tui_deps));
            }
        }
    }

    // Merge discovered deps into service metadata
    for (svc_name, deps) in all_deps {
        if let Some(svc) = services.iter_mut().find(|s| s.name == svc_name) {
            for dep in deps {
                svc.dependencies.insert(dep);
            }
        }
    }
}

/// Find client type references in a source file.
fn find_client_deps(content: &str, service_name: &str, deps: &mut BTreeSet<DepEdge>) {
    for client_type in CLIENT_TYPES {
        if content.contains(client_type) {
            if let Some(target) = client_to_service(client_type) {
                if target != service_name {
                    deps.insert(DepEdge {
                        target: target.to_string(),
                        client_type: client_type.to_string(),
                    });
                }
            }
        }
    }
}

// ── Diagram generation ──────────────────────────────────────────────────────

fn generate_topology(services: &[ServiceMeta]) -> String {
    let mut out = String::new();

    out.push_str("```mermaid\n");
    out.push_str("---\nconfig:\n  layout: elk\n  elk:\n    mergeEdges: true\n    nodePlacementStrategy: NETWORK_SIMPLEX\n---\n");
    out.push_str("graph TD\n\n");

    // Group by transport
    let http: Vec<&ServiceMeta> = services.iter().filter(|s| s.transport == Transport::Http).collect();
    let zmq: Vec<&ServiceMeta> = services.iter().filter(|s| s.transport == Transport::ZmqRep).collect();
    let proxies: Vec<&ServiceMeta> = services.iter().filter(|s| s.transport == Transport::Proxy).collect();

    // HTTP subgraph
    if !http.is_empty() {
        out.push_str("    subgraph http [\"HTTP Services\"]\n");
        out.push_str("        direction LR\n");
        for s in &http {
            out.push_str(&format!("        {}[\"{}\"]\n", s.name, service_label(s)));
        }
        out.push_str("    end\n\n");
    }

    // ZMQ subgraph
    if !zmq.is_empty() {
        out.push_str("    subgraph zmq [\"ZMQ Services · encrypted RPC\"]\n");
        out.push_str("        direction LR\n");
        for s in &zmq {
            out.push_str(&format!("        {}[\"{}\"]\n", s.name, service_label(s)));
        }
        out.push_str("    end\n\n");
    }

    // Proxy subgraph
    if !proxies.is_empty() {
        out.push_str("    subgraph proxies [\"Proxy Services\"]\n");
        out.push_str("        direction LR\n");
        for s in &proxies {
            out.push_str(&format!("        {}[\"{}\"]\n", s.name, service_label(s)));
        }
        out.push_str("    end\n\n");
    }

    // Special: ModelService spawns InferenceService
    if services.iter().any(|s| s.name == "model") {
        out.push_str("    %% ModelService spawns InferenceService per loaded model\n");
        out.push_str("    model -->|\"spawns per model\"| inference\n\n");
    }

    // Dependency edges
    out.push_str("    %% Auto-detected service dependencies\n");
    for svc in services {
        // Group deps by target, collect client types
        let mut target_clients: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for dep in &svc.dependencies {
            target_clients.entry(dep.target.as_str())
                .or_default()
                .push(dep.client_type.as_str());
        }

        for (target, clients) in &target_clients {
            let is_stream = clients.iter().any(|c| *c == "StreamChannel" || *c == "StreamInfo");
            let is_policy = *target == "policy" && clients.iter().any(|c| *c == "PolicyClient");
            let has_other = clients.iter().any(|c| *c != "StreamChannel" && *c != "StreamInfo" && *c != "PolicyClient");

            // Policy: dashed line with "authorize" label
            if is_policy {
                out.push_str(&format!("    {} -.->|\"authorize\"| {}\n", svc.name, target));
            }

            // Streaming: thick arrow
            if is_stream {
                let stream_types: Vec<&str> = clients.iter()
                    .filter(|c| **c == "StreamChannel" || **c == "StreamInfo")
                    .copied()
                    .collect();
                let label = stream_types.join(", ");
                out.push_str(&format!("    {} ==>|\"{}\"| {}\n", svc.name, label, target));
            }

            // Other deps: normal arrow with client type labels
            if has_other {
                let other_clients: Vec<&str> = clients.iter()
                    .filter(|c| **c != "StreamChannel" && **c != "StreamInfo" && **c != "PolicyClient")
                    .copied()
                    .collect();
                if !other_clients.is_empty() {
                    let label = other_clients.join(", ");
                    out.push_str(&format!("    {} -->|\"{}\"| {}\n", svc.name, label, target));
                }
            }
        }
    }

    out.push_str("```\n");
    out
}

fn service_label(s: &ServiceMeta) -> String {
    let mut label = title_case(&s.name);
    if s.methods > 0 || s.has_quic {
        let mut parts = Vec::new();
        if s.methods > 0 {
            parts.push(format!("{} methods", s.methods));
        }
        if s.has_quic {
            parts.push("QUIC".to_string());
        }
        label.push_str(&format!("<br/>{}", parts.join(" · ")));
    }
    if !s.scopes.is_empty() {
        let scope_str: Vec<String> = s.scopes.iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect();
        label.push_str(&format!("<br/>{}", scope_str.join(" · ")));
    }
    label
}

fn generate_method_breakdown(services: &[ServiceMeta]) -> String {
    let mut out = String::new();
    out.push_str("```mermaid\ngraph LR\n\n");

    for s in services {
        if s.methods == 0 {
            continue;
        }

        let id = &s.name;
        out.push_str(&format!(
            "    subgraph {id}_svc [\"{} · {} · {} methods\"]\n",
            title_case(&s.name), s.transport.label(), s.methods
        ));
        out.push_str("        direction TB\n");

        for (scope, count) in &s.scopes {
            out.push_str(&format!("        {id}_{scope}[\"{scope}: {count}\"]\n"));
        }

        if s.has_quic {
            out.push_str(&format!("        {id}_quic[\"QUIC\"]\n"));
            out.push_str(&format!("        style {id}_quic fill:#2d8a4e,stroke:#333,color:#fff\n"));
        }

        out.push_str("    end\n\n");
    }

    out.push_str("```\n");
    out
}

fn generate_summary_table(services: &[ServiceMeta]) -> String {
    let mut out = String::new();
    out.push_str("| Service | Transport | QUIC | Methods | Scopes | Dependencies |\n");
    out.push_str("|---------|-----------|------|---------|--------|--------------|\n");

    for s in services {
        let quic = if s.has_quic { "yes" } else { "-" };
        let methods = if s.methods > 0 { s.methods.to_string() } else { "0".to_string() };
        let scopes = if s.scopes.is_empty() {
            "-".to_string()
        } else {
            s.scopes.iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(" · ")
        };
        let deps = if s.dependencies.is_empty() {
            "-".to_string()
        } else {
            let mut targets: BTreeSet<&str> = BTreeSet::new();
            for dep in &s.dependencies {
                targets.insert(&dep.target);
            }
            targets.into_iter().collect::<Vec<_>>().join(", ")
        };

        out.push_str(&format!(
            "| **{}** | {} | {} | {} | {} | {} |\n",
            title_case(&s.name), s.transport.label(), quic, methods, scopes, deps
        ));
    }

    out
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let matches = clap::Command::new("generate-rpc-diagrams")
        .about("Generate Mermaid architecture diagrams from Cap'n Proto metadata")
        .arg(clap::Arg::new("metadata-dir")
            .long("metadata-dir")
            .help("Directory containing *_metadata.json files")
            .default_value("target/rpc-metadata"))
        .arg(clap::Arg::new("services-dir")
            .long("services-dir")
            .help("Path to services/ source directory")
            .default_value("crates/hyprstream/src/services"))
        .arg(clap::Arg::new("src-root")
            .long("src-root")
            .help("Root source directory for method-level call scanning")
            .default_value("crates/hyprstream/src"))
        .arg(clap::Arg::new("output")
            .short('o')
            .long("output")
            .help("Output file path (default: stdout)"))
        .get_matches();

    let metadata_dir = PathBuf::from(matches.get_one::<String>("metadata-dir").unwrap());
    let services_dir = PathBuf::from(matches.get_one::<String>("services-dir").unwrap());
    let src_root = PathBuf::from(matches.get_one::<String>("src-root").unwrap());

    // 1. Scan factories for service registrations
    let factories_path = services_dir.join("factories.rs");
    let mut services = scan_factories(&factories_path);

    // 2. Merge schema metadata (method counts + scopes)
    if metadata_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&metadata_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let is_metadata = path
                    .file_name()
                    .map(|n| n.to_string_lossy().ends_with("_metadata.json"))
                    .unwrap_or(false);
                if !is_metadata { continue; }

                if let Some((name, methods, scopes)) = parse_metadata(&path) {
                    if let Some(svc) = services.iter_mut().find(|s| s.name == name) {
                        svc.methods = methods;
                        svc.scopes = scopes;
                    }
                }
            }
        }
    } else {
        eprintln!("Warning: metadata dir {} not found, method counts will be zero", metadata_dir.display());
    }

    // 3. Scan service sources for client dependencies
    scan_service_sources(&services_dir, &mut services);

    // Sort services by name
    services.sort_by(|a, b| a.name.cmp(&b.name));

    // 4. Build method registry and scan for method-level calls
    let method_registry = build_method_registry(&metadata_dir);
    let method_calls = scan_method_calls(&src_root, &method_registry);

    // 5. Generate output
    let mut output = String::new();
    output.push_str("# HyprStream RPC Service Architecture\n\n");
    output.push_str("<!-- Auto-generated by `cargo run --bin generate-rpc-diagrams` -->\n");
    output.push_str("<!-- Do not edit manually. Regenerate after schema changes. -->\n\n");

    output.push_str("## Service Topology\n\n");
    output.push_str(&generate_topology(&services));

    output.push_str("\n## Service Method Breakdown\n\n");
    output.push_str(&generate_method_breakdown(&services));

    if !method_calls.is_empty() {
        output.push_str("\n## Method-Level Call Graph\n\n");
        output.push_str(&generate_call_graph(&method_calls, &method_registry));
    }

    output.push_str("\n## Service Summary\n\n");
    output.push_str(&generate_summary_table(&services));

    // Write output
    if let Some(out_path) = matches.get_one::<String>("output") {
        match fs::write(out_path, &output) {
            Ok(_) => eprintln!("Written to {out_path}"),
            Err(e) => eprintln!("Error writing {out_path}: {e}"),
        }
    } else {
        print!("{output}");
    }

    // Print stats
    eprintln!("\n--- Stats ---");
    eprintln!("Services: {}", services.len());
    eprintln!("Dependency edges: {}", services.iter().map(|s| s.dependencies.len()).sum::<usize>());
    eprintln!("Schema methods: {}", method_registry.len());
    eprintln!("Method-level calls: {}", method_calls.len());
} 