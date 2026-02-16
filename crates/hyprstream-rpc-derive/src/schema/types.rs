//! Type definitions for parsed Cap'n Proto schemas.

#[derive(Debug)]
pub struct ParsedSchema {
    /// Request union variants
    pub request_variants: Vec<UnionVariant>,
    /// Response union variants
    pub response_variants: Vec<UnionVariant>,
    /// Struct definitions referenced by variants
    pub structs: Vec<StructDef>,
    /// Scoped clients detected from nested union patterns
    pub scoped_clients: Vec<ScopedClient>,
    /// Enum definitions parsed from schema
    pub enums: Vec<EnumDef>,
}

#[derive(Debug)]
pub struct UnionVariant {
    pub name: String,
    pub type_name: String,
    pub description: String,
    /// MCP scope override (e.g., "write:model:*"). Empty string means use default.
    pub scope: String,
    /// Whether this method is hidden from CLI (internal-only).
    pub cli_hidden: bool,
}

#[derive(Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub has_union: bool,
    /// Domain type path from `$domainType` annotation (e.g., "runtime::VersionResponse").
    /// When set, the generated client returns this type via `FromCapnp::read_from()`.
    pub domain_type: Option<String>,
    /// Origin file stem for imported types (e.g., `Some("streaming")` for types from streaming.capnp).
    /// `None` means the type is local to the service's own schema file.
    pub origin_file: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub type_name: String,
    pub description: String,
    /// From `$fixedSize(N)` annotation: generates `[u8; N]` instead of `Vec<u8>` for Data fields.
    pub fixed_size: Option<u32>,
}

#[derive(Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<(String, u32)>,
    /// Origin file stem for imported types (e.g., `Some("streaming")` for types from streaming.capnp).
    /// `None` means the type is local to the service's own schema file.
    pub origin_file: Option<String>,
}

/// A scoped client detected from nested union patterns in the schema.
#[derive(Debug)]
pub struct ScopedClient {
    /// Factory method name on parent client (e.g., "repo", "session")
    pub factory_name: String,
    /// Generated client struct name (e.g., "RepositoryClient", "ModelSessionClient")
    pub client_name: String,
    /// Curried scope fields from the inner struct
    pub scope_fields: Vec<FieldDef>,
    /// Inner union variants from the request struct
    pub inner_request_variants: Vec<UnionVariant>,
    /// Inner union variants from the response struct
    pub inner_response_variants: Vec<UnionVariant>,
    /// Cap'n Proto module name for the inner response struct (snake_case)
    pub capnp_inner_response: String,
    /// Nested scoped clients detected within this scope (e.g., Fs within Repository)
    pub nested_clients: Vec<ScopedClient>,
}
