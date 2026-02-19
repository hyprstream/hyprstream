//! Utility functions for name conversion and type mapping.

use proc_macro2::TokenStream;
use quote::quote;

use crate::schema::types::{EnumDef, StructDef};

// ─────────────────────────────────────────────────────────────────────────────
// CapnpType — Centralized type classification
// ─────────────────────────────────────────────────────────────────────────────

/// Classified Cap'n Proto type for codegen dispatch.
///
/// Centralizes the type classification that was previously done via string
/// matching in 19+ match expressions across the codegen modules. Use
/// `CapnpType::classify()` to create from a raw type name string.
#[derive(Debug, Clone, PartialEq)]
pub enum CapnpType {
    Void,
    Bool,
    Text,
    Data,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    ListText,
    ListData,
    /// List of a primitive numeric type (e.g., `List(Int64)` → `ListPrimitive(Int64)`).
    ListPrimitive(Box<CapnpType>),
    /// List of a named struct type (e.g., `List(ModelInfo)` → `ListStruct("ModelInfo")`).
    ListStruct(String),
    /// A known struct type from the schema.
    Struct(String),
    /// A known enum type from the schema.
    Enum(String),
    /// Unknown/fallback type (not a primitive, struct, or enum in the schema).
    Unknown(String),
}

impl CapnpType {
    /// Classify a Cap'n Proto type name, resolving structs and enums from schema context.
    pub fn classify(type_name: &str, structs: &[StructDef], enums: &[EnumDef]) -> Self {
        match type_name {
            "Void" => Self::Void,
            "Bool" => Self::Bool,
            "Text" => Self::Text,
            "Data" => Self::Data,
            "UInt8" => Self::UInt8,
            "UInt16" => Self::UInt16,
            "UInt32" => Self::UInt32,
            "UInt64" => Self::UInt64,
            "Int8" => Self::Int8,
            "Int16" => Self::Int16,
            "Int32" => Self::Int32,
            "Int64" => Self::Int64,
            "Float32" => Self::Float32,
            "Float64" => Self::Float64,
            t if t.starts_with("List(") && t.ends_with(')') => {
                let inner = &t[5..t.len() - 1];
                match inner {
                    "Text" => Self::ListText,
                    "Data" => Self::ListData,
                    _ => {
                        let inner_type = Self::classify(inner, structs, enums);
                        if inner_type.is_numeric() {
                            Self::ListPrimitive(Box::new(inner_type))
                        } else {
                            Self::ListStruct(inner.to_owned())
                        }
                    }
                }
            }
            t => {
                if enums.iter().any(|e| e.name == t) {
                    Self::Enum(t.to_owned())
                } else if structs.iter().any(|s| s.name == t) {
                    Self::Struct(t.to_owned())
                } else {
                    Self::Unknown(t.to_owned())
                }
            }
        }
    }

    /// Classify without schema context (primitives and lists only; structs/enums become Unknown).
    pub fn classify_primitive(type_name: &str) -> Self {
        Self::classify(type_name, &[], &[])
    }

    /// The owned Rust type string (e.g., `"String"`, `"u32"`, `"Vec<String>"`).
    pub fn rust_owned_type(&self) -> String {
        match self {
            Self::Void => "()".into(),
            Self::Bool => "bool".into(),
            Self::Text => "String".into(),
            Self::Data | Self::Unknown(_) => "Vec<u8>".into(),
            Self::UInt8 => "u8".into(),
            Self::UInt16 => "u16".into(),
            Self::UInt32 => "u32".into(),
            Self::UInt64 => "u64".into(),
            Self::Int8 => "i8".into(),
            Self::Int16 => "i16".into(),
            Self::Int32 => "i32".into(),
            Self::Int64 => "i64".into(),
            Self::Float32 => "f32".into(),
            Self::Float64 => "f64".into(),
            Self::ListText => "Vec<String>".into(),
            Self::ListData => "Vec<Vec<u8>>".into(),
            Self::ListPrimitive(inner) => format!("Vec<{}>", inner.rust_owned_type()),
            Self::ListStruct(inner) => format!("Vec<{inner}>"),
            Self::Struct(name) => name.clone(),
            Self::Enum(name) => format!("{name}Enum"),
        }
    }

    /// The Rust parameter type string (borrowed where appropriate).
    pub fn rust_param_type(&self) -> String {
        match self {
            Self::Text | Self::Enum(_) => "&str".into(),
            Self::Data => "&[u8]".into(),
            Self::ListText => "&[String]".into(),
            Self::ListData => "&[Vec<u8>]".into(),
            Self::ListPrimitive(inner) => format!("&[{}]", inner.rust_owned_type()),
            Self::ListStruct(inner) => format!("&[{inner}]"),
            _ => self.rust_owned_type(),
        }
    }

    /// Whether function arguments of this type should be passed by reference in dispatch.
    pub fn is_by_ref(&self) -> bool {
        matches!(
            self,
            Self::Text
                | Self::Data
                | Self::ListText
                | Self::ListData
                | Self::ListPrimitive(_)
                | Self::ListStruct(_)
                | Self::Enum(_)
        )
    }

    /// Whether this is a numeric/value type (Bool included — Copy semantics).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Bool
                | Self::UInt8
                | Self::UInt16
                | Self::UInt32
                | Self::UInt64
                | Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::Float32
                | Self::Float64
        )
    }

}

// ─────────────────────────────────────────────────────────────────────────────
// Name conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert snake_case or camelCase to PascalCase.
pub fn to_pascal_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut capitalize_next = true;

    for c in s.chars() {
        if c == '_' || c == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }

    result
}

/// Convert PascalCase or camelCase to snake_case.
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 && !chars[i - 1].is_ascii_uppercase() {
                result.push('_');
            }
            if i > 0
                && chars[i - 1].is_ascii_uppercase()
                && i + 1 < chars.len()
                && chars[i + 1].is_ascii_lowercase()
            {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }

    result
}

/// Convert PascalCase to snake_case matching Cap'n Proto's Rust codegen naming.
///
/// Unlike `to_snake_case` (which groups consecutive uppercase letters like `DNS` → `dns`),
/// capnp inserts `_` before every uppercase letter: `DNSConfig` → `d_n_s_config`.
pub fn to_capnp_module_name(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 8);
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(c.to_ascii_lowercase());
    }
    result
}

/// Convert a Rust type string into a TokenStream for use in quote!.
pub fn rust_type_tokens(type_str: &str) -> TokenStream {
    let ts: TokenStream = type_str.parse().unwrap_or_else(|_| quote! { Vec<u8> });
    ts
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::types::*;

    fn test_structs() -> Vec<StructDef> {
        vec![StructDef {
            name: "ModelInfo".into(),
            fields: vec![
                FieldDef { name: "name".into(), type_name: "Text".into(), description: String::new(), fixed_size: None, optional: false },
                FieldDef { name: "size".into(), type_name: "UInt64".into(), description: String::new(), fixed_size: None, optional: false },
            ],
            has_union: false,
            domain_type: None,
            origin_file: None,
        }]
    }

    fn test_enums() -> Vec<EnumDef> {
        vec![EnumDef {
            name: "Status".into(),
            variants: vec![("active".into(), 0), ("inactive".into(), 1)],
            origin_file: None,
        }]
    }

    #[test]
    fn classify_primitives() {
        assert_eq!(CapnpType::classify_primitive("Void"), CapnpType::Void);
        assert_eq!(CapnpType::classify_primitive("Bool"), CapnpType::Bool);
        assert_eq!(CapnpType::classify_primitive("Text"), CapnpType::Text);
        assert_eq!(CapnpType::classify_primitive("Data"), CapnpType::Data);
        assert_eq!(CapnpType::classify_primitive("UInt32"), CapnpType::UInt32);
        assert_eq!(CapnpType::classify_primitive("Float64"), CapnpType::Float64);
    }

    #[test]
    fn classify_lists() {
        assert_eq!(CapnpType::classify_primitive("List(Text)"), CapnpType::ListText);
        assert_eq!(CapnpType::classify_primitive("List(Data)"), CapnpType::ListData);
        assert_eq!(
            CapnpType::classify_primitive("List(ModelInfo)"),
            CapnpType::ListStruct("ModelInfo".into())
        );
    }

    #[test]
    fn classify_list_primitives() {
        assert_eq!(
            CapnpType::classify_primitive("List(Int64)"),
            CapnpType::ListPrimitive(Box::new(CapnpType::Int64))
        );
        assert_eq!(
            CapnpType::classify_primitive("List(UInt8)"),
            CapnpType::ListPrimitive(Box::new(CapnpType::UInt8))
        );
        assert_eq!(
            CapnpType::classify_primitive("List(Bool)"),
            CapnpType::ListPrimitive(Box::new(CapnpType::Bool))
        );
        assert_eq!(
            CapnpType::classify_primitive("List(Float64)"),
            CapnpType::ListPrimitive(Box::new(CapnpType::Float64))
        );
        // Verify owned types
        assert_eq!(
            CapnpType::ListPrimitive(Box::new(CapnpType::Int64)).rust_owned_type(),
            "Vec<i64>"
        );
        assert_eq!(
            CapnpType::ListPrimitive(Box::new(CapnpType::UInt8)).rust_owned_type(),
            "Vec<u8>"
        );
        // Verify param types
        assert_eq!(
            CapnpType::ListPrimitive(Box::new(CapnpType::Int64)).rust_param_type(),
            "&[i64]"
        );
        // Verify is_by_ref
        assert!(CapnpType::ListPrimitive(Box::new(CapnpType::Int64)).is_by_ref());
    }

    #[test]
    fn classify_with_schema_context() {
        let structs = test_structs();
        let enums = test_enums();
        assert_eq!(
            CapnpType::classify("ModelInfo", &structs, &enums),
            CapnpType::Struct("ModelInfo".into())
        );
        assert_eq!(
            CapnpType::classify("Status", &structs, &enums),
            CapnpType::Enum("Status".into())
        );
        assert_eq!(
            CapnpType::classify("Unknown", &structs, &enums),
            CapnpType::Unknown("Unknown".into())
        );
    }

    #[test]
    fn rust_owned_types() {
        assert_eq!(CapnpType::Text.rust_owned_type(), "String");
        assert_eq!(CapnpType::UInt32.rust_owned_type(), "u32");
        assert_eq!(CapnpType::ListText.rust_owned_type(), "Vec<String>");
        assert_eq!(
            CapnpType::ListStruct("ModelInfo".into()).rust_owned_type(),
            "Vec<ModelInfo>"
        );
        assert_eq!(CapnpType::Struct("ModelInfo".into()).rust_owned_type(), "ModelInfo");
        assert_eq!(CapnpType::Enum("Status".into()).rust_owned_type(), "StatusEnum");
    }

    #[test]
    fn rust_param_types() {
        assert_eq!(CapnpType::Text.rust_param_type(), "&str");
        assert_eq!(CapnpType::Data.rust_param_type(), "&[u8]");
        assert_eq!(CapnpType::UInt32.rust_param_type(), "u32");
        assert_eq!(CapnpType::ListText.rust_param_type(), "&[String]");
        assert_eq!(CapnpType::Enum("Status".into()).rust_param_type(), "&str");
    }

    #[test]
    fn is_by_ref() {
        assert!(CapnpType::Text.is_by_ref());
        assert!(CapnpType::Data.is_by_ref());
        assert!(CapnpType::ListText.is_by_ref());
        assert!(!CapnpType::Bool.is_by_ref());
        assert!(!CapnpType::UInt32.is_by_ref());
        assert!(!CapnpType::Void.is_by_ref());
    }

    #[test]
    fn to_pascal_case_works() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("helloWorld"), "HelloWorld");
        assert_eq!(to_pascal_case("policy"), "Policy");
    }

    #[test]
    fn to_snake_case_works() {
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("helloWorld"), "hello_world");
        assert_eq!(to_snake_case("HTTPRequest"), "http_request");
    }

}
