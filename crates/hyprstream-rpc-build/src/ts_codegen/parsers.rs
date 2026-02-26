//! Generate TypeScript response parser functions.
//!
//! For each service, generates a `parse{Service}Response` function that reads
//! the response union discriminant and returns typed data.
//!
//! Scoped response variants (structs that themselves contain unions) are
//! parsed recursively, returning nested `{ variant, data }` objects.

use hyprstream_rpc_build::schema::types::{FieldDef, FieldSection, ParsedSchema, ScopedClient, StructDef};
use hyprstream_rpc_build::util::{to_camel_case, to_pascal_case};

use super::{bool_bit_index, capnp_to_ts_type, data_byte_offset, is_data_scalar};

/// Generate response parser types and functions for a service.
pub fn generate_parsers(out: &mut String, service_name: &str, schema: &ParsedSchema) {
    let resp_struct = match &schema.response_struct {
        Some(s) => s,
        None => return,
    };

    let pascal = to_pascal_case(service_name);

    let non_union_fields: Vec<&FieldDef> = resp_struct
        .fields
        .iter()
        .filter(|f| f.discriminant_value == 0xFFFF)
        .collect();

    let disc_byte_off = resp_struct.discriminant_offset * 2;

    // Build a map of factory_name → ScopedClient for scoped response detection
    let scoped_map: Vec<(&str, &ScopedClient)> = schema
        .scoped_clients
        .iter()
        .map(|sc| (sc.factory_name.as_str(), sc))
        .collect();

    // Response result interface
    out.push_str(&format!(
        "export interface {pascal}ResponseResult {{\n"
    ));
    for f in &non_union_fields {
        out.push_str(&format!(
            "  {}: {};\n",
            to_camel_case(&f.name),
            capnp_to_ts_type(&f.type_name)
        ));
    }
    out.push_str("  variant: string;\n");
    out.push_str("  data: unknown;\n");
    out.push_str("}\n\n");

    // Parser function
    out.push_str(&format!(
        "function parse{pascal}Response(bytes: Uint8Array): {pascal}ResponseResult {{\n"
    ));
    out.push_str(&format!(
        "  const reader = new CapnpReader(bytes, {}, {});\n",
        resp_struct.data_words, resp_struct.pointer_words
    ));

    for f in &non_union_fields {
        emit_reader_field(out, "reader", f, &to_camel_case(&f.name), "  ");
    }

    out.push_str(&format!(
        "  const _disc = reader.getUint16({disc_byte_off});\n"
    ));
    out.push_str("  switch (_disc) {\n");

    for variant in &schema.response_variants {
        let variant_field = resp_struct
            .fields
            .iter()
            .find(|f| f.name == variant.name && f.discriminant_value != 0xFFFF);

        let disc_value = variant_field.map(|f| f.discriminant_value).unwrap_or(0);

        out.push_str(&format!("    case {disc_value}: // {}\n", variant.name));

        // Check if this variant is a scoped response (matches "{factory}Result")
        let is_scoped = variant
            .name
            .strip_suffix("Result")
            .and_then(|base| scoped_map.iter().find(|(name, _)| *name == base))
            .map(|(_, sc)| *sc);

        if let Some(sc) = is_scoped {
            // Scoped response: read inner struct and parse its union
            emit_scoped_response_parse(
                out,
                "reader",
                variant_field,
                sc,
                schema,
                &non_union_fields,
                &variant.name,
            );
        } else if variant.type_name == "Void" {
            emit_return(out, &non_union_fields, &variant.name, "undefined");
        } else if variant.type_name == "Bool" {
            if let Some(vf) = variant_field {
                let byte_off = data_byte_offset(vf);
                let bit = bool_bit_index(vf);
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getBool({byte_off}, {bit})"),
                );
            }
        } else if is_data_scalar(&variant.type_name) {
            if let Some(vf) = variant_field {
                let byte_off = data_byte_offset(vf);
                let method = super::getter_method(&variant.type_name);
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.{method}({byte_off})"),
                );
            }
        } else if variant.type_name == "Text" {
            if let Some(vf) = variant_field {
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getText({})", vf.slot_offset),
                );
            }
        } else if variant.type_name == "Data" {
            if let Some(vf) = variant_field {
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getData({})", vf.slot_offset),
                );
            }
        } else if variant.type_name.starts_with("List(Text") {
            if let Some(vf) = variant_field {
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getTextList({})", vf.slot_offset),
                );
            }
        } else {
            // Struct type
            if let Some(vf) = variant_field {
                let struct_def = schema.structs.iter().find(|s| s.name == variant.type_name);
                if let Some(sd) = struct_def {
                    emit_struct_read(out, "reader", vf, sd, &non_union_fields, &variant.name);
                } else {
                    emit_return(out, &non_union_fields, &variant.name, "null");
                }
            }
        }
    }

    // Default case
    let default_fields: Vec<String> = non_union_fields
        .iter()
        .map(|f| to_camel_case(&f.name))
        .collect();
    let fields_str = default_fields.join(", ");
    let comma = if fields_str.is_empty() { "" } else { ", " };
    out.push_str("    default:\n");
    out.push_str(&format!(
        "      return {{ {fields_str}{comma}variant: 'unknown', data: null }};\n"
    ));
    out.push_str("  }\n");
    out.push_str("}\n\n");
}

/// Emit code to parse a scoped response variant (inner union within a struct).
fn emit_scoped_response_parse(
    out: &mut String,
    reader_var: &str,
    variant_field: Option<&FieldDef>,
    sc: &ScopedClient,
    schema: &ParsedSchema,
    non_union_fields: &[&FieldDef],
    variant_name: &str,
) {
    let Some(vf) = variant_field else {
        emit_return(out, non_union_fields, variant_name, "null");
        return;
    };

    // Find the inner response struct
    let resp_variant_name = format!("{}Result", sc.factory_name);
    let outer_resp_variant = schema
        .response_variants
        .iter()
        .find(|v| v.name == resp_variant_name);
    let inner_resp_type = outer_resp_variant
        .map(|v| v.type_name.as_str())
        .unwrap_or("");
    let inner_resp_struct = schema.structs.iter().find(|s| s.name == inner_resp_type);

    let Some(irs) = inner_resp_struct else {
        emit_return(out, non_union_fields, variant_name, "null");
        return;
    };

    out.push_str(&format!(
        "    {{\n      const _inner = {reader_var}.getStruct({}, {}, {});\n",
        vf.slot_offset, irs.data_words, irs.pointer_words
    ));

    let resp_fields: Vec<String> = non_union_fields
        .iter()
        .map(|f| to_camel_case(&f.name))
        .collect();
    let fields_str = resp_fields.join(", ");
    let comma = if fields_str.is_empty() { "" } else { ", " };

    out.push_str(&format!(
        "      if (!_inner) return {{ {fields_str}{comma}variant: '{variant_name}', data: null }};\n"
    ));

    // Read inner discriminant
    let inner_disc_byte_off = irs.discriminant_offset * 2;
    out.push_str(&format!(
        "      const _innerDisc = _inner.getUint16({inner_disc_byte_off});\n"
    ));
    out.push_str("      switch (_innerDisc) {\n");

    for inner_variant in &sc.inner_response_variants {
        let inner_field = irs
            .fields
            .iter()
            .find(|f| f.name == inner_variant.name && f.discriminant_value != 0xFFFF);

        let inner_disc_value = inner_field.map(|f| f.discriminant_value).unwrap_or(0);

        out.push_str(&format!(
            "        case {inner_disc_value}: // {}\n",
            inner_variant.name
        ));

        let inner_data = emit_inner_variant_read(inner_field, &inner_variant.type_name, schema);

        out.push_str(&format!(
            "          return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {inner_data} }} }};\n",
            inner_variant.name
        ));
    }

    // Also handle nested scoped response variants
    for nc in &sc.nested_clients {
        let nested_resp_name = format!("{}Result", nc.factory_name);
        let nested_resp = sc
            .inner_response_variants
            .iter()
            .find(|v| v.name == nested_resp_name);

        // If this was already handled above, skip
        // (nested client variants are filtered from inner_response_variants in CGR reader)
        // They won't appear in the loop above, so we handle them here.
        if nested_resp.is_some() {
            // Already emitted in the inner_response_variants loop
            continue;
        }

        // Check if there's a matching field in the inner response struct
        let nested_field = irs
            .fields
            .iter()
            .find(|f| f.name == nested_resp_name && f.discriminant_value != 0xFFFF);

        if let Some(nf) = nested_field {
            out.push_str(&format!(
                "        case {}: // {} (nested scope)\n",
                nf.discriminant_value, nested_resp_name
            ));
            out.push_str(&format!(
                "          return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: null }} }};\n",
                nested_resp_name
            ));
        }
    }

    out.push_str("        default:\n");
    out.push_str(&format!(
        "          return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: 'unknown', data: null }} }};\n"
    ));
    out.push_str("      }\n");
    out.push_str("    }\n");
}

/// Generate an inline expression to read an inner variant's data.
fn emit_inner_variant_read(
    field: Option<&FieldDef>,
    type_name: &str,
    schema: &ParsedSchema,
) -> String {
    let Some(f) = field else {
        return "null".into();
    };

    match type_name {
        "Void" => "undefined".into(),
        "Bool" => {
            let byte_off = data_byte_offset(f);
            let bit = bool_bit_index(f);
            format!("_inner.getBool({byte_off}, {bit})")
        }
        "Text" => format!("_inner.getText({})", f.slot_offset),
        "Data" => format!("_inner.getData({})", f.slot_offset),
        t if is_data_scalar(t) => {
            let byte_off = data_byte_offset(f);
            let method = super::getter_method(type_name);
            format!("_inner.{method}({byte_off})")
        }
        t if t.starts_with("List(Text") => {
            format!("_inner.getTextList({})", f.slot_offset)
        }
        _ => {
            // Struct — read its fields as an object
            let struct_def = schema.structs.iter().find(|s| s.name == type_name);
            if let Some(sd) = struct_def {
                let fields: Vec<String> = sd
                    .fields
                    .iter()
                    .filter(|sf| sf.discriminant_value == 0xFFFF)
                    .map(|sf| {
                        let camel = to_camel_case(&sf.name);
                        let read = match sf.section {
                            FieldSection::Data => {
                                let byte_off = data_byte_offset(sf);
                                if sf.type_name == "Bool" {
                                    let bit = bool_bit_index(sf);
                                    format!("_is.getBool({byte_off}, {bit})")
                                } else {
                                    let method = super::getter_method(&sf.type_name);
                                    format!("_is.{method}({byte_off})")
                                }
                            }
                            FieldSection::Pointer => {
                                let method = super::getter_method(&sf.type_name);
                                format!("_is.{method}({})", sf.slot_offset)
                            }
                            FieldSection::Group => "undefined".into(),
                        };
                        format!("{camel}: {read}")
                    })
                    .collect();
                format!(
                    "(() => {{ const _is = _inner.getStruct({}, {}, {}); return _is ? {{ {} }} : null; }})()",
                    f.slot_offset, sd.data_words, sd.pointer_words, fields.join(", ")
                )
            } else {
                "null".into()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn emit_reader_field(
    out: &mut String,
    reader_var: &str,
    field: &FieldDef,
    local_name: &str,
    indent: &str,
) {
    match field.section {
        FieldSection::Data => {
            let byte_off = data_byte_offset(field);
            if field.type_name == "Bool" {
                let bit = bool_bit_index(field);
                out.push_str(&format!(
                    "{indent}const {local_name} = {reader_var}.getBool({byte_off}, {bit});\n"
                ));
            } else {
                let method = super::getter_method(&field.type_name);
                out.push_str(&format!(
                    "{indent}const {local_name} = {reader_var}.{method}({byte_off});\n"
                ));
            }
        }
        FieldSection::Pointer => {
            let method = super::getter_method(&field.type_name);
            out.push_str(&format!(
                "{indent}const {local_name} = {reader_var}.{method}({});\n",
                field.slot_offset
            ));
        }
        FieldSection::Group => {
            out.push_str(&format!(
                "{indent}const {local_name} = undefined; // group\n"
            ));
        }
    }
}

fn emit_return(
    out: &mut String,
    non_union_fields: &[&FieldDef],
    variant_name: &str,
    data_expr: &str,
) {
    let fields: Vec<String> = non_union_fields
        .iter()
        .map(|f| to_camel_case(&f.name))
        .collect();
    let fields_str = fields.join(", ");
    let comma = if fields_str.is_empty() { "" } else { ", " };
    out.push_str(&format!(
        "      return {{ {fields_str}{comma}variant: '{variant_name}', data: {data_expr} }};\n"
    ));
}

fn emit_struct_read(
    out: &mut String,
    reader_var: &str,
    ptr_field: &FieldDef,
    struct_def: &StructDef,
    non_union_fields: &[&FieldDef],
    variant_name: &str,
) {
    out.push_str(&format!(
        "    {{\n      const _s = {reader_var}.getStruct({}, {}, {});\n",
        ptr_field.slot_offset, struct_def.data_words, struct_def.pointer_words
    ));

    let visible_fields: Vec<&FieldDef> = struct_def
        .fields
        .iter()
        .filter(|f| f.discriminant_value == 0xFFFF)
        .collect();

    out.push_str("      const _data = _s ? {\n");
    for sf in &visible_fields {
        let camel = to_camel_case(&sf.name);
        let read_expr = match sf.section {
            FieldSection::Data => {
                let byte_off = data_byte_offset(sf);
                if sf.type_name == "Bool" {
                    let bit = bool_bit_index(sf);
                    format!("_s.getBool({byte_off}, {bit})")
                } else {
                    let method = super::getter_method(&sf.type_name);
                    format!("_s.{method}({byte_off})")
                }
            }
            FieldSection::Pointer => {
                let method = super::getter_method(&sf.type_name);
                format!("_s.{method}({})", sf.slot_offset)
            }
            FieldSection::Group => "undefined".into(),
        };
        out.push_str(&format!("        {camel}: {read_expr},\n"));
    }
    out.push_str("      } : null;\n");

    let response_fields: Vec<String> = non_union_fields
        .iter()
        .map(|f| to_camel_case(&f.name))
        .collect();
    let fields_str = response_fields.join(", ");
    let comma = if fields_str.is_empty() { "" } else { ", " };
    out.push_str(&format!(
        "      return {{ {fields_str}{comma}variant: '{variant_name}', data: _data }};\n"
    ));
    out.push_str("    }\n");
}
