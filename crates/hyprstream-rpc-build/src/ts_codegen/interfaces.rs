//! Generate TypeScript interfaces from Cap'n Proto struct and enum definitions.

use hyprstream_rpc_build::schema::types::{EnumDef, FieldSection, ParsedSchema, StructDef};
use hyprstream_rpc_build::util::to_camel_case;

use super::capnp_to_ts_type;

/// Generate all interface and type declarations for a service.
pub fn generate_interfaces(out: &mut String, schema: &ParsedSchema) {
    // 1. Enum types
    for e in &schema.enums {
        emit_enum(out, e);
    }

    // 2. Struct interfaces (data types referenced by request/response variants)
    for s in &schema.structs {
        emit_struct_interface(out, s);
    }
}

/// Emit a TypeScript string union type from an enum definition.
fn emit_enum(out: &mut String, e: &EnumDef) {
    out.push_str(&format!("export type {} =", e.name));
    for (i, (name, _ord)) in e.variants.iter().enumerate() {
        if i == 0 {
            out.push('\n');
        }
        let sep = if i + 1 < e.variants.len() { "" } else { ";" };
        out.push_str(&format!("  | '{}'{}\n", to_camel_case(name), sep));
    }
    out.push('\n');
}

/// Emit a TypeScript interface from a struct definition.
///
/// Only non-union fields are included. Structs that are pure union envelopes
/// (no non-union fields except possibly in scoped patterns) get an interface
/// with just the non-union fields if any.
fn emit_struct_interface(out: &mut String, s: &StructDef) {
    let non_union_fields: Vec<_> = s.non_union_fields().collect();

    // Pure union envelopes — emit as a discriminated union type alias
    if non_union_fields.is_empty() && s.has_union {
        let union_fields: Vec<_> = s.union_fields().collect();
        if !union_fields.is_empty() {
            out.push_str(&format!(
                "export type {} = {{ variant: string; data: unknown }};\n\n",
                s.name
            ));
        }
        return;
    }

    out.push_str(&format!("export interface {} {{\n", s.name));
    for f in &non_union_fields {
        let ts = capnp_to_ts_type(&f.type_name);
        let opt = if f.optional { "?" } else { "" };
        // Struct pointer fields are nullable (getStruct can return null).
        // Option* types already encode absence as `undefined` — don't add `| null`.
        let nullable = if matches!(f.section, FieldSection::Pointer)
            && !matches!(f.type_name.as_str(), "Text" | "Data")
            && !f.type_name.starts_with("List(")
            && !f.type_name.starts_with("Option")
        {
            " | null"
        } else {
            ""
        };
        out.push_str(&format!(
            "  {}{}: {}{};\n",
            to_camel_case(&f.name),
            opt,
            ts,
            nullable
        ));
    }
    out.push_str("}\n\n");
}
