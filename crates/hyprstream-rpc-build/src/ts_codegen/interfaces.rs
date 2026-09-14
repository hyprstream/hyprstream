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
        emit_struct_interface(out, s, schema);
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
/// Union-having structs emit as typed discriminated-union aliases (see
/// [`super::emits_union_alias`]): each arm mirrors exactly what the generated
/// parser returns for that discriminant (`{ variant: '<name>', data: <typed> }`,
/// with any non-union fields shared into every arm), plus the parser's
/// default-case `{ variant: 'unknown'; data: null }`. Mixed structs (named
/// fields + union) that are referenced data types must take this path — a
/// named-fields-only interface would not declare the `variant`/`data`
/// properties the generated request serializer switches on (#1616).
/// Service/scoped envelopes and plain structs get a named-fields interface.
fn emit_struct_interface(out: &mut String, s: &StructDef, schema: &ParsedSchema) {
    if super::emits_union_alias(schema, s) {
        let union_fields: Vec<_> = s.union_fields().collect();
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
        let shared_fields = super::shared_union_fields(s);
        let arms: Vec<(String, String)> = union_fields
            .iter()
            // Use the raw field name to match the generated parser/builder,
            // which key the discriminated `variant` on the capnp field name.
            .map(|f| (f.name.clone(), super::union_arm_data_type(s, f)))
            .collect();
        super::emit_union_alias(out, &s.name, &shared_fields, &arms);
        return;
    }

    let non_union_fields: Vec<_> = s.non_union_fields().collect();

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
