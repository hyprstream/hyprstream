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
}

#[derive(Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub has_union: bool,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub type_name: String,
    pub description: String,
}

#[derive(Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<(String, u32)>,
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
    /// Nested scoped clients detected within this scope (3rd level, e.g., Fs within Repository)
    pub nested_clients: Vec<ScopedClient>,
    /// Parent scope fields (empty for top-level scoped clients)
    pub parent_scope_fields: Vec<FieldDef>,
    /// Parent factory name (None for top-level scoped clients)
    pub parent_factory_name: Option<String>,
    /// Parent capnp inner response module (None for top-level scoped clients)
    pub parent_capnp_inner_response: Option<String>,
}
