//! VFS/9P projection generation (epic #539, T2).
//!
//! Emits a `vfs_nodes() -> (&str, &[VfsNode])` table per service — the 4th
//! codegen projection alongside client/handler/dispatch/metadata. Each node is
//! derived from a method's [`UnionVariant`] (the SAME data the MCP/metadata
//! surface reads) plus the `$vfs*` annotations captured in T1.
//!
//! The runtime mount engine (T3) consumes the table to serve `/srv/{service}`.
//! This module only emits the table + the read/write-vs-scope guard; it wires
//! no behavior.
//!
//! ## Mapping rules (method shape → node kind; `$vfsKind` overrides)
//!
//! | shape                          | default kind |
//! |--------------------------------|--------------|
//! | query, Void, name "list*"      | `Dir`        |
//! | query, Text/scalar arg         | `File` @ `{arg}` |
//! | query, Void                    | `File`       |
//! | query, struct (rich args)      | `Query`      |
//! | write / manage                 | `Ctl`        |
//! | streaming (any)                | `Stream`     |
//!
//! ## Guard (compile-time)
//!
//! A read-only kind (`File`/`Dir`/`Query`) on a write/manage scope, or a `Ctl`
//! on a query scope, is a contradiction — a 9P read must be side-effect-free.
//! [`validate_node`] returns the error; [`generate_mount`] turns it into a
//! `compile_error!`.

use proc_macro2::TokenStream;
use quote::quote;

use crate::resolve::ResolvedSchema;
use crate::schema::types::UnionVariant;
use crate::util::{to_snake_case, CapnpType};

/// The VFS node kind, mirrored from the runtime `hyprstream_rpc::metadata::VfsNodeKind`.
/// Kept as a plain derive-crate enum so inference/guard logic is unit-testable
/// without depending on the runtime crate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeKind {
    File,
    Dir,
    Ctl,
    Stream,
    Query,
}

impl NodeKind {
    /// Parse the `$vfsKind` enumerant name (as captured in `UnionVariant::vfs_kind`).
    fn parse(s: &str) -> Option<Self> {
        match s {
            "file" => Some(Self::File),
            "dir" => Some(Self::Dir),
            "ctl" => Some(Self::Ctl),
            "stream" => Some(Self::Stream),
            "query" => Some(Self::Query),
            _ => None,
        }
    }

    /// A read of this node must be side-effect-free (9P read invariant).
    fn is_read_only(self) -> bool {
        matches!(self, Self::File | Self::Dir | Self::Query)
    }

    /// The runtime-crate variant tokens for this kind.
    fn to_tokens(self) -> TokenStream {
        match self {
            Self::File => quote! { hyprstream_rpc::metadata::VfsNodeKind::File },
            Self::Dir => quote! { hyprstream_rpc::metadata::VfsNodeKind::Dir },
            Self::Ctl => quote! { hyprstream_rpc::metadata::VfsNodeKind::Ctl },
            Self::Stream => quote! { hyprstream_rpc::metadata::VfsNodeKind::Stream },
            Self::Query => quote! { hyprstream_rpc::metadata::VfsNodeKind::Query },
        }
    }
}

/// Whether the method's scope action denotes a mutation (write/manage) vs a read (query).
///
/// Scopes come from `$scope`/`$capability` (S3, #547). A `query` (or read-shaped)
/// scope must project to a read-only node; anything else is treated as a mutation.
/// Empty scope (a `$scopeExempt` method) is treated as read (least authority).
fn scope_is_mutation(scope: &str) -> bool {
    !matches!(scope, "" | "query" | "read")
}

/// Whether a method name denotes a listing (projects to a directory by default).
fn is_list_name(name: &str) -> bool {
    let snake = to_snake_case(name);
    snake == "list" || snake.starts_with("list_")
}

/// Infer the default node kind from a method's scope + argument shape + streaming.
///
/// `$vfsKind` (when present) overrides this — see [`resolve_kind`].
pub fn infer_kind(scope: &str, arg: &CapnpType, name: &str, is_streaming: bool) -> NodeKind {
    if is_streaming {
        return NodeKind::Stream;
    }
    if scope_is_mutation(scope) {
        return NodeKind::Ctl;
    }
    // Read-shaped (query/exempt): pick by argument shape.
    match arg {
        // A `list*` method with no args reads as a directory.
        CapnpType::Void if is_list_name(name) => NodeKind::Dir,
        // A struct / rich args → a synthetic query file (write spec, read result).
        // Void and single-scalar args both read as a file (the `{arg}` binding is
        // decided from arg shape in `infer_path`, not the kind).
        arg if is_query_file_arg(arg) => NodeKind::Query,
        _ => NodeKind::File,
    }
}

/// Whether a read-shaped method's argument is "rich" (a struct/list needing a
/// written query spec) rather than Void or a single bindable scalar.
fn is_query_file_arg(arg: &CapnpType) -> bool {
    matches!(
        arg,
        CapnpType::Struct(_)
            | CapnpType::Unknown(_)
            | CapnpType::ListText
            | CapnpType::ListData
            | CapnpType::ListPrimitive(_)
            | CapnpType::ListStruct(_)
            | CapnpType::Data
    )
}

/// Resolve the effective node kind: `$vfsKind` override, else inference.
///
/// Returns an `Err(message)` if `$vfsKind` names an unknown enumerant.
pub fn resolve_kind(
    v: &UnionVariant,
    arg: &CapnpType,
    is_streaming: bool,
) -> Result<NodeKind, String> {
    if v.vfs_kind.is_empty() {
        Ok(infer_kind(&v.scope, arg, &v.name, is_streaming))
    } else {
        NodeKind::parse(&v.vfs_kind).ok_or_else(|| {
            format!(
                "method `{}`: `$vfsKind(\"{}\")` is not a valid VFS node kind \
                 (expected one of file/dir/ctl/stream/query)",
                to_snake_case(&v.name),
                v.vfs_kind
            )
        })
    }
}

/// The compile-time guard: a node's kind must not contradict its scope.
///
/// - A write/manage scope must NOT project to a read-only node
///   (`file`/`dir`/`query`) — a mutation behind a side-effect-free 9P read.
/// - A query/read scope must NOT project to a `ctl` — a read that invokes a verb.
///
/// Returns `Err(message)` on contradiction, naming the offending method.
pub fn validate_node(method_snake: &str, scope: &str, kind: NodeKind) -> Result<(), String> {
    let mutation = scope_is_mutation(scope);
    if mutation && kind.is_read_only() {
        return Err(format!(
            "method `{method_snake}`: write/manage scope `{scope}` cannot project to a \
             read-only VFS node (`{kind:?}`) — a 9P read must be side-effect-free; \
             use `$vfsKind(ctl)` or `$vfsKind(stream)`"
        ));
    }
    if !mutation && matches!(kind, NodeKind::Ctl) {
        return Err(format!(
            "method `{method_snake}`: query scope `{scope}` cannot project to a `ctl` node \
             — a control-file write invokes the method; use a read kind \
             (`$vfsKind(file|dir|query)`) or give the method a write scope"
        ));
    }
    Ok(())
}

/// Infer the default mount-relative path for a node when `$vfsPath` is unset.
fn infer_path(kind: NodeKind, method_snake: &str, arg: &CapnpType) -> String {
    match kind {
        // A directory listing is the mount root by convention.
        NodeKind::Dir => ".".to_owned(),
        // A scalar-arg file binds the arg from the path segment.
        NodeKind::File if !matches!(arg, CapnpType::Void) => "{arg}".to_owned(),
        // A control file lands at the conventional `ctl` node.
        NodeKind::Ctl => "ctl".to_owned(),
        // Everything else (a Void read file like `health`, a stream, or a query
        // file) is served under the method's own name.
        NodeKind::Stream | NodeKind::Query | NodeKind::File => method_snake.to_owned(),
    }
}

/// Generate the per-service `vfs_nodes()` table.
///
/// Emits `pub fn vfs_nodes() -> (&'static str, &'static [VfsNode])`. On a
/// scope/kind contradiction (or bad `$vfsKind`), emits a `compile_error!`
/// naming the offending method instead.
pub fn generate_mount(service_name: &str, resolved: &ResolvedSchema) -> TokenStream {
    // Data-only schemas (no request variants) get no mount.
    if resolved.raw.request_variants.is_empty() {
        return TokenStream::new();
    }

    let scoped_names: Vec<&str> = resolved
        .raw
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    let mut node_entries: Vec<TokenStream> = Vec::new();

    for v in &resolved.raw.request_variants {
        // Scoped clients project as subdirectories — deferred to the nested pass
        // (their own node tables); skip here. `$vfsHidden` excludes a method.
        if scoped_names.contains(&v.name.as_str()) || v.vfs_hidden {
            continue;
        }

        let method_snake = to_snake_case(&v.name);
        let arg = resolved.resolve_type(&v.type_name).capnp_type.clone();
        let is_streaming = is_streaming_variant(&v.name, &resolved.raw.response_variants);

        // Resolve kind ($vfsKind override or inference) — bad annotation ⇒ compile_error.
        let kind = match resolve_kind(v, &arg, is_streaming) {
            Ok(k) => k,
            Err(msg) => return compile_error(&msg),
        };

        // The guard: kind must not contradict scope.
        if let Err(msg) = validate_node(&method_snake, &v.scope, kind) {
            return compile_error(&msg);
        }

        let path = if v.vfs_path.is_empty() {
            infer_path(kind, &method_snake, &arg)
        } else {
            v.vfs_path.clone()
        };

        let kind_tokens = kind.to_tokens();
        let scope = v.scope.as_str();
        let bulk = v.vfs_bulk;

        node_entries.push(quote! {
            hyprstream_rpc::metadata::VfsNode {
                method: #method_snake,
                path: #path,
                kind: #kind_tokens,
                scope: #scope,
                bulk: #bulk,
            }
        });
    }

    let doc = format!("Generated 9P/VFS node table for the {service_name} service (epic #539).");

    quote! {
        #[doc = #doc]
        pub fn vfs_nodes() -> (&'static str, &'static [hyprstream_rpc::metadata::VfsNode]) {
            static NODES: &[hyprstream_rpc::metadata::VfsNode] = &[
                #(#node_entries,)*
            ];
            (#service_name, NODES)
        }
    }
}

/// Emit a `compile_error!` carrying `msg`.
fn compile_error(msg: &str) -> TokenStream {
    quote! { ::core::compile_error!(#msg); }
}

/// Whether a request variant's response is a `StreamInfo` (a streaming method).
/// Mirrors `metadata::is_streaming_variant` for the non-scoped case.
fn is_streaming_variant(variant_name: &str, response_variants: &[UnionVariant]) -> bool {
    let expected = format!("{variant_name}Result");
    response_variants
        .iter()
        .find(|v| v.name == expected)
        .map(|v| v.type_name == "StreamInfo")
        .unwrap_or(false)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn infers_dir_for_void_list_query() {
        let k = infer_kind("query", &CapnpType::Void, "list", false);
        assert_eq!(k, NodeKind::Dir);
    }

    #[test]
    fn infers_file_for_void_query() {
        let k = infer_kind("query", &CapnpType::Void, "healthCheck", false);
        assert_eq!(k, NodeKind::File);
    }

    #[test]
    fn infers_parameterized_file_for_text_query() {
        let k = infer_kind("query", &CapnpType::Text, "getByName", false);
        assert_eq!(k, NodeKind::File);
        assert_eq!(infer_path(k, "get_by_name", &CapnpType::Text), "{arg}");
    }

    #[test]
    fn infers_query_for_struct_query() {
        let k = infer_kind("query", &CapnpType::Struct("SearchRequest".into()), "search", false);
        assert_eq!(k, NodeKind::Query);
    }

    #[test]
    fn infers_ctl_for_write() {
        let k = infer_kind("write", &CapnpType::Struct("CloneRequest".into()), "clone", false);
        assert_eq!(k, NodeKind::Ctl);
        assert_eq!(infer_path(k, "clone", &CapnpType::Struct("CloneRequest".into())), "ctl");
    }

    #[test]
    fn infers_stream_for_streaming() {
        let k = infer_kind("write", &CapnpType::Struct("CloneRequest".into()), "cloneStream", true);
        assert_eq!(k, NodeKind::Stream);
    }

    #[test]
    fn guard_rejects_write_scope_on_read_only_node() {
        // A write method forced to a read-only `file` must be rejected.
        let err = validate_node("clone", "write", NodeKind::File).unwrap_err();
        assert!(err.contains("clone"), "error names the method: {err}");
        assert!(err.contains("side-effect-free"), "error explains why: {err}");
    }

    #[test]
    fn guard_rejects_query_scope_on_ctl_node() {
        let err = validate_node("get_by_name", "query", NodeKind::Ctl).unwrap_err();
        assert!(err.contains("get_by_name"));
        assert!(err.contains("ctl"));
    }

    #[test]
    fn guard_permits_consistent_pairings() {
        assert!(validate_node("clone", "write", NodeKind::Ctl).is_ok());
        assert!(validate_node("list", "query", NodeKind::Dir).is_ok());
        assert!(validate_node("get", "query", NodeKind::File).is_ok());
        assert!(validate_node("search", "query", NodeKind::Query).is_ok());
        // A scope-exempt method (empty scope) is treated as read — ctl still rejected.
        assert!(validate_node("ping", "", NodeKind::File).is_ok());
        assert!(validate_node("ping", "", NodeKind::Ctl).is_err());
    }

    #[test]
    fn bad_vfs_kind_annotation_is_reported() {
        let v = UnionVariant {
            name: "clone".into(),
            type_name: "CloneRequest".into(),
            description: String::new(),
            scope: "write".into(),
            scope_exempt: false,
            cli_hidden: false,
            doc_example: String::new(),
            vfs_path: String::new(),
            vfs_kind: "bogus".into(),
            vfs_bulk: false,
            vfs_hidden: false,
        };
        let err = resolve_kind(&v, &CapnpType::Struct("CloneRequest".into()), false).unwrap_err();
        assert!(err.contains("bogus"));
        assert!(err.contains("clone"));
    }

    #[test]
    fn vfs_kind_override_wins_over_inference() {
        // A query method explicitly annotated as a query file (rich-args override).
        let v = UnionVariant {
            name: "getByName".into(),
            type_name: "Text".into(),
            description: String::new(),
            scope: "query".into(),
            scope_exempt: false,
            cli_hidden: false,
            doc_example: String::new(),
            vfs_path: String::new(),
            vfs_kind: "query".into(),
            vfs_bulk: false,
            vfs_hidden: false,
        };
        let k = resolve_kind(&v, &CapnpType::Text, false).expect("valid kind");
        assert_eq!(k, NodeKind::Query);
    }

    // ── End-to-end token generation over a fixture service ───────────────────

    use crate::schema::types::{ParsedSchema, StructDef};

    fn variant(name: &str, type_name: &str, scope: &str) -> UnionVariant {
        UnionVariant {
            name: name.into(),
            type_name: type_name.into(),
            description: String::new(),
            scope: scope.into(),
            scope_exempt: false,
            cli_hidden: false,
            doc_example: String::new(),
            vfs_path: String::new(),
            vfs_kind: String::new(),
            vfs_bulk: false,
            vfs_hidden: false,
        }
    }

    fn empty_struct(name: &str) -> StructDef {
        StructDef {
            name: name.into(),
            fields: vec![],
            has_union: false,
            domain_type: None,
            origin_file: None,
            data_words: 0,
            pointer_words: 0,
            discriminant_count: 0,
            discriminant_offset: 0,
            union_arms: vec![],
        }
    }

    fn schema_with(request_variants: Vec<UnionVariant>, structs: Vec<StructDef>) -> ParsedSchema {
        ParsedSchema {
            request_variants,
            response_variants: vec![],
            structs,
            scoped_clients: vec![],
            enums: vec![],
            request_struct: None,
            response_struct: None,
        }
    }

    #[test]
    fn generates_a_sensible_node_table() {
        // A registry-like service: list (dir), getByName (param file), health
        // (void read file), clone (ctl).
        let schema = schema_with(
            vec![
                variant("list", "Void", "query"),
                variant("getByName", "Text", "query"),
                variant("healthCheck", "Void", "query"),
                variant("clone", "CloneRequest", "write"),
            ],
            vec![empty_struct("CloneRequest")],
        );
        let resolved = ResolvedSchema::from(&schema);
        let out = generate_mount("registry", &resolved).to_string();

        assert!(out.contains("fn vfs_nodes"), "emits vfs_nodes(): {out}");
        // list → Dir at "."
        assert!(out.contains("VfsNodeKind :: Dir"), "list is a Dir: {out}");
        // getByName → File at "{arg}"
        assert!(out.contains("\"{arg}\""), "param file binds {{arg}}: {out}");
        // clone → Ctl at "ctl"
        assert!(out.contains("VfsNodeKind :: Ctl"), "clone is a Ctl: {out}");
        assert!(out.contains("\"ctl\""), "ctl node path: {out}");
        // health → File node present
        assert!(out.contains("\"health_check\""), "health file path: {out}");
        // No compile_error emitted for a consistent schema.
        assert!(!out.contains("compile_error"), "no guard error: {out}");
    }

    #[test]
    fn contradiction_emits_compile_error() {
        // A write method forced to a read-only `file` via $vfsKind must fail the build.
        let mut bad = variant("clone", "CloneRequest", "write");
        bad.vfs_kind = "file".into();
        let schema = schema_with(vec![bad], vec![empty_struct("CloneRequest")]);
        let resolved = ResolvedSchema::from(&schema);
        let out = generate_mount("registry", &resolved).to_string();

        assert!(out.contains("compile_error"), "emits compile_error: {out}");
        assert!(out.contains("clone"), "names the offending method: {out}");
    }

    #[test]
    fn vfs_hidden_excludes_a_method() {
        let mut hidden = variant("internalPing", "Void", "query");
        hidden.vfs_hidden = true;
        let schema = schema_with(
            vec![variant("list", "Void", "query"), hidden],
            vec![],
        );
        let resolved = ResolvedSchema::from(&schema);
        let out = generate_mount("registry", &resolved).to_string();

        assert!(out.contains("\"list\"") || out.contains("VfsNodeKind :: Dir"));
        assert!(!out.contains("internal_ping"), "hidden method excluded: {out}");
    }

    #[test]
    fn data_only_schema_emits_no_mount() {
        let schema = schema_with(vec![], vec![empty_struct("SomeData")]);
        let resolved = ResolvedSchema::from(&schema);
        let out = generate_mount("dataonly", &resolved).to_string();
        assert!(out.is_empty(), "data-only schema has no mount: {out}");
    }
}
