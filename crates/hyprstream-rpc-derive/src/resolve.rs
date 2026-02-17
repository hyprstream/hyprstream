//! Resolution layer between parsing and codegen.
//!
//! `ResolvedSchema` is built once from a `ParsedSchema` and provides O(1) lookups
//! for structs, enums, and response variants, plus cached name conversions and
//! precomputed type classifications.

use std::collections::{HashMap, HashSet};

use proc_macro2::Ident;
use quote::format_ident;

use crate::schema::types::*;
use crate::util::{to_pascal_case, to_snake_case, CapnpType};

/// Cached name conversions for a single identifier string.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResolvedName {
    pub raw: String,
    pub snake: String,
    pub pascal: String,
    pub snake_ident: Ident,
    pub pascal_ident: Ident,
}

/// Cached type classification for a single type_name string.
#[derive(Debug, Clone)]
pub struct ResolvedType {
    pub capnp_type: CapnpType,
    pub rust_owned: String,
    pub rust_param: String,
    pub is_by_ref: bool,
    pub is_numeric: bool,
}

/// Indexed, precomputed view of a `ParsedSchema`.
///
/// Built once in `generate_service()`, consumed by all codegen modules.
pub struct ResolvedSchema<'a> {
    pub raw: &'a ParsedSchema,
    pub struct_index: HashMap<String, usize>,
    pub enum_index: HashMap<String, usize>,
    #[allow(dead_code)]
    pub response_index: HashMap<String, usize>,
    names: HashMap<String, ResolvedName>,
    types: HashMap<String, ResolvedType>,
}

impl<'a> ResolvedSchema<'a> {
    /// Build a resolved schema from a parsed schema.
    pub fn from(schema: &'a ParsedSchema) -> Self {
        // Build indexes
        let struct_index: HashMap<String, usize> = schema
            .structs
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.clone(), i))
            .collect();

        let enum_index: HashMap<String, usize> = schema
            .enums
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.clone(), i))
            .collect();

        let response_index: HashMap<String, usize> = schema
            .response_variants
            .iter()
            .enumerate()
            .map(|(i, v)| (v.name.clone(), i))
            .collect();

        // Collect all unique name strings that need conversion
        let mut name_strings = HashSet::new();
        // Collect all unique type_name strings that need classification
        let mut type_strings = HashSet::new();

        // Walk request/response variants
        for v in &schema.request_variants {
            name_strings.insert(v.name.clone());
            type_strings.insert(v.type_name.clone());
        }
        for v in &schema.response_variants {
            name_strings.insert(v.name.clone());
            type_strings.insert(v.type_name.clone());
        }

        // Walk structs and their fields
        for s in &schema.structs {
            name_strings.insert(s.name.clone());
            // Struct names are also valid type references (e.g., when used in
            // List(StructName) or as field types in other structs)
            type_strings.insert(s.name.clone());
            for f in &s.fields {
                name_strings.insert(f.name.clone());
                type_strings.insert(f.type_name.clone());
            }
        }

        // Walk enums
        for e in &schema.enums {
            name_strings.insert(e.name.clone());
            // Enum names are also valid type references
            type_strings.insert(e.name.clone());
            for (vname, _) in &e.variants {
                name_strings.insert(vname.clone());
            }
        }

        // Walk scoped clients recursively
        fn collect_from_scoped(
            sc: &ScopedClient,
            names: &mut HashSet<String>,
            types: &mut HashSet<String>,
        ) {
            names.insert(sc.factory_name.clone());
            names.insert(sc.client_name.clone());
            for f in &sc.scope_fields {
                names.insert(f.name.clone());
                types.insert(f.type_name.clone());
            }
            for v in &sc.inner_request_variants {
                names.insert(v.name.clone());
                types.insert(v.type_name.clone());
            }
            for v in &sc.inner_response_variants {
                names.insert(v.name.clone());
                types.insert(v.type_name.clone());
            }
            for nested in &sc.nested_clients {
                collect_from_scoped(nested, names, types);
            }
        }
        for sc in &schema.scoped_clients {
            collect_from_scoped(sc, &mut name_strings, &mut type_strings);
        }

        // Pre-resolve all names
        let names: HashMap<String, ResolvedName> = name_strings
            .into_iter()
            .map(|raw| {
                let snake = to_snake_case(&raw);
                let pascal = to_pascal_case(&raw);
                let snake_ident = format_ident!("{}", snake);
                let pascal_ident = format_ident!("{}", pascal);
                (
                    raw.clone(),
                    ResolvedName {
                        raw,
                        snake,
                        pascal,
                        snake_ident,
                        pascal_ident,
                    },
                )
            })
            .collect();

        // Pre-resolve all types using indexed lookups
        let types: HashMap<String, ResolvedType> = type_strings
            .into_iter()
            .map(|type_name| {
                let ct = classify_with_indexes(&type_name, &struct_index, &enum_index);
                let rust_owned = ct.rust_owned_type();
                let rust_param = ct.rust_param_type();
                let is_by_ref = ct.is_by_ref();
                let is_numeric = ct.is_numeric();
                (
                    type_name,
                    ResolvedType {
                        capnp_type: ct,
                        rust_owned,
                        rust_param,
                        is_by_ref,
                        is_numeric,
                    },
                )
            })
            .collect();

        Self {
            raw: schema,
            struct_index,
            enum_index,
            response_index,
            names,
            types,
        }
    }

    /// O(1) struct lookup by name.
    pub fn find_struct(&self, name: &str) -> Option<&'a StructDef> {
        self.struct_index.get(name).map(|&i| &self.raw.structs[i])
    }

    /// O(1) enum lookup by name.
    pub fn find_enum(&self, name: &str) -> Option<&'a EnumDef> {
        self.enum_index.get(name).map(|&i| &self.raw.enums[i])
    }

    /// O(1) response variant lookup by name.
    #[allow(dead_code)]
    pub fn find_response(&self, name: &str) -> Option<&'a UnionVariant> {
        self.response_index
            .get(name)
            .map(|&i| &self.raw.response_variants[i])
    }

    /// O(1) cached name lookup. Panics if name was not seen during schema walk.
    pub fn name(&self, raw: &str) -> &ResolvedName {
        self.names
            .get(raw)
            .unwrap_or_else(|| panic!("ResolvedSchema: name '{}' not collected during resolution", raw))
    }

    /// O(1) cached type lookup. Panics if type was not seen during schema walk.
    pub fn resolve_type(&self, type_name: &str) -> &ResolvedType {
        self.types.get(type_name).unwrap_or_else(|| {
            panic!(
                "ResolvedSchema: type '{}' not collected during resolution",
                type_name
            )
        })
    }

    /// O(1) check if a name refers to a known struct.
    pub fn is_struct(&self, name: &str) -> bool {
        self.struct_index.contains_key(name)
    }

    /// O(1) check if a name refers to a known enum.
    #[allow(dead_code)]
    pub fn is_enum(&self, name: &str) -> bool {
        self.enum_index.contains_key(name)
    }
}

/// Classify a Cap'n Proto type name using HashMap indexes instead of linear scans.
///
/// This duplicates the classification logic from `CapnpType::classify()` but uses
/// O(1) HashMap lookups instead of O(n) `iter().any()`. The public `CapnpType::classify()`
/// API in `util.rs` is unchanged (used by parser and tests).
fn classify_with_indexes(
    type_name: &str,
    struct_index: &HashMap<String, usize>,
    enum_index: &HashMap<String, usize>,
) -> CapnpType {
    match type_name {
        "Void" => CapnpType::Void,
        "Bool" => CapnpType::Bool,
        "Text" => CapnpType::Text,
        "Data" => CapnpType::Data,
        "UInt8" => CapnpType::UInt8,
        "UInt16" => CapnpType::UInt16,
        "UInt32" => CapnpType::UInt32,
        "UInt64" => CapnpType::UInt64,
        "Int8" => CapnpType::Int8,
        "Int16" => CapnpType::Int16,
        "Int32" => CapnpType::Int32,
        "Int64" => CapnpType::Int64,
        "Float32" => CapnpType::Float32,
        "Float64" => CapnpType::Float64,
        t if t.starts_with("List(") && t.ends_with(')') => {
            let inner = &t[5..t.len() - 1];
            match inner {
                "Text" => CapnpType::ListText,
                "Data" => CapnpType::ListData,
                _ => {
                    let inner_type = classify_with_indexes(inner, struct_index, enum_index);
                    if inner_type.is_numeric() {
                        CapnpType::ListPrimitive(Box::new(inner_type))
                    } else {
                        CapnpType::ListStruct(inner.to_owned())
                    }
                }
            }
        }
        t => {
            if enum_index.contains_key(t) {
                CapnpType::Enum(t.to_owned())
            } else if struct_index.contains_key(t) {
                CapnpType::Struct(t.to_owned())
            } else {
                CapnpType::Unknown(t.to_owned())
            }
        }
    }
}
