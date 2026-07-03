//! Generate TypeScript interfaces from Cap'n Proto struct and enum definitions.

use hyprstream_rpc_build::schema::types::{
    EnumDef, FieldDef, FieldSection, ParsedSchema, StructDef,
};
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

/// Emit a TypeScript interface for an inline union `group` arm's leaf fields.
///
/// Mirrors the non-union-field emission in [`emit_struct_interface`]: struct
/// pointers are nullable, Option*/Text/Data/List are not.
fn emit_group_interface(out: &mut String, name: &str, leaves: &[FieldDef]) {
    out.push_str(&format!("export interface {name} {{\n"));
    for leaf in leaves {
        let ts = capnp_to_ts_type(&leaf.type_name);
        let nullable = if matches!(leaf.section, FieldSection::Pointer)
            && !matches!(leaf.type_name.as_str(), "Text" | "Data")
            && !leaf.type_name.starts_with("List(")
            && !leaf.type_name.starts_with("Option")
        {
            " | null"
        } else {
            ""
        };
        out.push_str(&format!(
            "  {}: {}{};\n",
            to_camel_case(&leaf.name),
            ts,
            nullable
        ));
    }
    out.push_str("}\n\n");
}

/// Emit a TypeScript interface from a struct definition.
///
/// Only non-union fields are included. Structs that are pure union envelopes
/// (no non-union fields except possibly in scoped patterns) get an interface
/// with just the non-union fields if any.
fn emit_struct_interface(out: &mut String, s: &StructDef) {
    let non_union_fields: Vec<_> = s.non_union_fields().collect();

    // Pure union envelopes — emit as a typed discriminated union type alias.
    //
    // Each arm mirrors exactly what the generated parser returns for that
    // discriminant (`{ variant: '<name>', data: <typed> }`), plus the parser's
    // default-case `{ variant: 'unknown'; data: null }`. This replaces the
    // former degenerate `{ variant: string; data: unknown }` stub.
    if non_union_fields.is_empty() && s.has_union {
        let union_fields: Vec<_> = s.union_fields().collect();
        if !union_fields.is_empty() {
            // Synthesize a named interface for each inline `group` arm. Its leaf
            // fields live in this struct's own sections, so the arm's `data` is
            // that leaf shape (see `group_arm_leaves`). Emit before the union
            // type so the alias can reference it.
            for f in &union_fields {
                if let Some(leaves) = super::group_arm_leaves(s, f) {
                    emit_group_interface(
                        out,
                        &super::group_arm_type_name(&s.name, &f.name),
                        leaves,
                    );
                }
            }
            out.push_str(&format!("export type {} =\n", s.name));
            for f in &union_fields {
                // Use the raw field name to match the generated parser/builder,
                // which key the discriminated `variant` on the capnp field name.
                out.push_str(&format!(
                    "  | {{ variant: '{}'; data: {} }}\n",
                    f.name,
                    super::union_arm_data_type(s, f)
                ));
            }
            // Fallback arm matching the parser's `default` case.
            out.push_str("  | { variant: 'unknown'; data: null };\n\n");
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
