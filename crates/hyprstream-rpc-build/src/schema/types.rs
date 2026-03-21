//! Type definitions for parsed Cap'n Proto schemas.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
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
    /// Root request struct (wire format needed for builder codegen).
    /// `None` when parsed from text (no CGR wire format info).
    pub request_struct: Option<StructDef>,
    /// Root response struct (wire format needed for parser codegen).
    /// `None` when parsed from text (no CGR wire format info).
    pub response_struct: Option<StructDef>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnionVariant {
    pub name: String,
    pub type_name: String,
    pub description: String,
    /// MCP scope override (e.g., "write:model:*"). Empty string means use default.
    pub scope: String,
    /// Whether this method is hidden from CLI (internal-only).
    pub cli_hidden: bool,
}

#[derive(Debug, Serialize, Deserialize)]
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
    /// Number of data words in the struct section (from CGR).
    /// Used by TypeScript codegen for correct wire format offsets.
    pub data_words: u16,
    /// Number of pointer words in the struct section (from CGR).
    pub pointer_words: u16,
    /// Discriminant count (number of union members).
    pub discriminant_count: u16,
    /// Offset of the discriminant in the data section (in u16 units).
    pub discriminant_offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub type_name: String,
    pub description: String,
    /// From `$fixedSize(N)` annotation: generates `[u8; N]` instead of `Vec<u8>` for Data fields.
    pub fixed_size: Option<u32>,
    /// From `$optional` annotation: field is optional in MCP tool schemas and uses
    /// type-appropriate zero-value defaults when absent at runtime.
    pub optional: bool,
    /// Slot offset from CGR (field::slot::get_offset()).
    /// For data section fields, this is the word-level offset.
    /// For pointer section fields, this is the pointer index.
    pub slot_offset: u32,
    /// Which section this field lives in (data section or pointer section).
    pub section: FieldSection,
    /// Discriminant value for union fields (0xFFFF means non-union).
    pub discriminant_value: u16,
    /// From `$serdeRename("...")` annotation: generates `#[serde(rename = "...")]` on the Rust field.
    pub serde_rename: Option<String>,
}

/// Which section of a Cap'n Proto struct a field resides in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldSection {
    /// Data section (fixed-size scalars: bool, ints, floats, enum ordinals)
    #[default]
    Data,
    /// Pointer section (variable-length: Text, Data, List, Struct)
    Pointer,
    /// Group (inlined, no slot)
    Group,
}

impl StructDef {
    /// Fields that are NOT union members (discriminant_value == 0xFFFF).
    ///
    /// Use this for Rust codegen which should not generate `get_*()` / `set_*()`
    /// calls for union members (Cap'n Proto Rust uses `which()` for those).
    /// TypeScript codegen should use `.fields` directly since it needs all fields
    /// including union members for wire format slot offsets and discriminant values.
    pub fn non_union_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| f.discriminant_value == 0xFFFF)
    }

    /// Fields that ARE union members (discriminant_value != 0xFFFF).
    ///
    /// These are the variant arms of a union within the struct.
    /// Each has a distinct `discriminant_value` that identifies which arm is active.
    pub fn union_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| f.discriminant_value != 0xFFFF)
    }

    /// True if this struct is a pure union (has_union and no non-union fields).
    pub fn is_pure_union(&self) -> bool {
        self.has_union && self.non_union_fields().count() == 0
    }

    /// Returns the inner type name `T` if this struct is a valid `Option<T>` wrapper.
    ///
    /// Validates that the struct is a pure union with exactly two variants:
    /// `none @0 :Void` and `some @1 :T`.
    ///
    /// This is more reliable than checking `name.starts_with("Option")` alone,
    /// because it validates the actual structure rather than just the name.
    pub fn option_inner_type(&self) -> Option<&str> {
        if !self.is_pure_union() {
            return None;
        }
        if self.discriminant_count != 2 {
            return None;
        }
        let mut union_fields: Vec<&FieldDef> = self.union_fields().collect();
        if union_fields.len() != 2 {
            return None;
        }
        union_fields.sort_by_key(|f| f.discriminant_value);
        let none_field = union_fields[0];
        let some_field = union_fields[1];
        if none_field.name != "none" || none_field.type_name != "Void" {
            return None;
        }
        if some_field.name != "some" {
            return None;
        }
        Some(&some_field.type_name)
    }
}

/// Check if a Cap'n Proto type name is a primitive (not a struct reference).
///
/// Returns true for all scalar types, `Void`, `Text`, `Data`, and `List(...)`.
/// Includes `Void` — callers that need to exclude it (e.g., setter dispatch)
/// should additionally check `type_name != "Void"`.
pub fn is_primitive_capnp_type(type_name: &str) -> bool {
    matches!(
        type_name,
        "Void"
            | "Bool"
            | "Text"
            | "Data"
            | "UInt8"
            | "UInt16"
            | "UInt32"
            | "UInt64"
            | "Int8"
            | "Int16"
            | "Int32"
            | "Int64"
            | "Float32"
            | "Float64"
    ) || type_name.starts_with("List(")
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<(String, u32)>,
    /// Origin file stem for imported types (e.g., `Some("streaming")` for types from streaming.capnp).
    /// `None` means the type is local to the service's own schema file.
    pub origin_file: Option<String>,
}

/// A scoped client detected from nested union patterns in the schema.
#[derive(Debug, Serialize, Deserialize)]
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
