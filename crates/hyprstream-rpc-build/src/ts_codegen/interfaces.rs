//! Generate TypeScript interfaces from Cap'n Proto struct and enum definitions.

use hyprstream_rpc_build::schema::types::{EnumDef, FieldDef, FieldSection, ParsedSchema, StructDef};
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

    // Pure union envelopes — emit as a typed discriminated union type alias.
    //
    // Each arm mirrors exactly what the generated parser returns for that
    // discriminant (`{ variant: '<name>', data: <typed> }`), plus the parser's
    // default-case `{ variant: 'unknown'; data: null }`. This replaces the
    // former degenerate `{ variant: string; data: unknown }` stub.
    if non_union_fields.is_empty() && s.has_union {
        let union_fields: Vec<_> = s.union_fields().collect();
        if !union_fields.is_empty() {
            out.push_str(&format!("export type {} =\n", s.name));
            for f in &union_fields {
                // Use the raw field name to match the generated parser/builder,
                // which key the discriminated `variant` on the capnp field name.
                out.push_str(&format!(
                    "  | {{ variant: '{}'; data: {} }}\n",
                    f.name,
                    union_variant_data_type(f)
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

/// Map a union-member field to the TypeScript type carried in the `data` slot.
///
/// Mirrors the per-variant `data` expression produced by the parser emitters:
/// - `Void`              → `undefined`
/// - scalars / `Bool`    → the TS scalar (`number` / `bigint` / `boolean`)
/// - `Text`              → `string`
/// - `Data`              → `Uint8Array`
/// - `List(Text)`        → `string[]`
/// - `List(Struct)`      → `Inner[]` (elements are parsed into objects)
/// - struct              → `TypeName | null` (the parser returns `null` on a null pointer)
///
/// Keeping this aligned with the parser guarantees the typed discriminated union
/// is sound against the runtime values callers actually receive.
fn union_variant_data_type(f: &FieldDef) -> String {
    let tn = f.type_name.as_str();
    if tn == "Void" {
        return "undefined".to_owned();
    }
    if tn.starts_with("List(") {
        // capnp_to_ts_type already maps List(T) → T[].
        return capnp_to_ts_type(tn);
    }
    if super::is_primitive(tn) || tn == "Text" || tn == "Data" {
        return capnp_to_ts_type(tn);
    }
    // Struct (or enum) reference. Struct pointers can be null at runtime; enums
    // are data-section and never null, but the only data-section non-scalar union
    // payloads are enums, which `capnp_to_ts_type` maps to their type name.
    if matches!(f.section, FieldSection::Pointer) {
        format!("{} | null", capnp_to_ts_type(tn))
    } else {
        capnp_to_ts_type(tn)
    }
}
