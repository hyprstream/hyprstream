//! Generate TypeScript response parser functions.
//!
//! For each service, generates a `parse{Service}Response` function that reads
//! the response union discriminant and returns typed data.
//!
//! Scoped response variants (structs that themselves contain unions) are
//! parsed recursively, returning nested `{ variant, data }` objects.

use hyprstream_rpc_build::schema::types::{
    FieldDef, FieldSection, ParsedSchema, ScopedClient, StructDef,
};
use hyprstream_rpc_build::util::{to_camel_case, to_pascal_case};

use super::{bool_bit_index, capnp_to_ts_type, data_byte_offset, is_data_scalar};

/// Generate response parser types and functions for a service.
pub fn generate_parsers(out: &mut String, service_name: &str, schema: &ParsedSchema) {
    let resp_struct = match &schema.response_struct {
        Some(s) => s,
        None => return,
    };

    let pascal = to_pascal_case(service_name);

    let non_union_fields: Vec<&FieldDef> = resp_struct.non_union_fields().collect();

    let disc_byte_off = resp_struct.discriminant_offset * 2;

    // Build a map of factory_name → ScopedClient for scoped response detection
    let scoped_map: Vec<(&str, &ScopedClient)> = schema
        .scoped_clients
        .iter()
        .map(|sc| (sc.factory_name.as_str(), sc))
        .collect();

    // Response result type — a typed discriminated union when the response is a
    // union envelope, so callers narrow on `result.variant`. Each arm mirrors the
    // parser's per-variant `return` (shared non-union fields + `{ variant, data }`).
    if resp_struct.has_union {
        let shared_fields: Vec<String> = non_union_fields
            .iter()
            .map(|f| format!("{}: {}", to_camel_case(&f.name), capnp_to_ts_type(&f.type_name)))
            .collect();
        let arms: Vec<(String, String)> = schema
            .response_variants
            .iter()
            .map(|v| {
                let data_ty = resp_struct
                    .fields
                    .iter()
                    .find(|f| f.name == v.name && f.discriminant_value != 0xFFFF)
                    .map(super::union_variant_data_type)
                    .unwrap_or_else(|| "unknown".to_owned());
                (v.name.clone(), data_ty)
            })
            .collect();
        emit_result_union_alias(out, &format!("{pascal}ResponseResult"), &shared_fields, &arms);
    } else {
        // Data-only response (no union) — plain interface, unchanged shape.
        out.push_str(&format!("export interface {pascal}ResponseResult {{\n"));
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
    }

    // Parser function
    out.push_str(&format!(
        "function parse{pascal}Response(bytes: Uint8Array): {pascal}ResponseResult {{\n"
    ));
    out.push_str(&format!(
        "  const reader = new CapnpReader(bytes, {}, {});\n",
        resp_struct.data_words, resp_struct.pointer_words
    ));

    for f in &non_union_fields {
        emit_reader_field(out, "reader", f, &to_camel_case(&f.name), "  ", schema);
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
        } else if let Some(inner) = super::extract_list_inner_type(&variant.type_name) {
            // List(Struct) variant
            if let Some(vf) = variant_field {
                if let Some(sd) = schema.structs.iter().find(|s| s.name == inner) {
                    emit_struct_list_read(
                        out,
                        "reader",
                        vf,
                        sd,
                        &non_union_fields,
                        &variant.name,
                        schema,
                    );
                } else {
                    emit_return(out, &non_union_fields, &variant.name, "[]");
                }
            }
        } else {
            // Struct type
            if let Some(vf) = variant_field {
                let struct_def = schema.structs.iter().find(|s| s.name == variant.type_name);
                if let Some(sd) = struct_def {
                    emit_struct_read(
                        out,
                        "reader",
                        vf,
                        sd,
                        &non_union_fields,
                        &variant.name,
                        schema,
                    );
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
        "      return {{ {fields_str}{comma}variant: 'unknown' as const, data: null }};\n"
    ));
    out.push_str("  }\n");
    out.push_str("}\n\n");
}

// ---------------------------------------------------------------------------
// Standalone struct parsers
// ---------------------------------------------------------------------------

/// Generate standalone parser functions for all structs in a schema.
///
/// For union-bearing structs: `parse{StructName}(bytes): {StructName}Result`
/// (discriminated union return with `{ variant, data }` fields).
///
/// For non-union structs: `parse{StructName}(bytes): StructName`
/// (returns the struct interface directly).
///
/// Skips Option* wrapper structs and service request/response structs.
pub fn generate_struct_parsers(out: &mut String, schema: &ParsedSchema) {
    for sd in &schema.structs {
        if sd.option_inner_type().is_some() {
            continue;
        }
        // Skip service request/response structs (handled by generate_parsers)
        if schema
            .request_struct
            .as_ref()
            .is_some_and(|rs| rs.name == sd.name)
            || schema
                .response_struct
                .as_ref()
                .is_some_and(|rs| rs.name == sd.name)
        {
            continue;
        }

        if !sd.has_union {
            generate_plain_struct_parser(out, sd, schema);
            continue;
        }

        let non_union_fields: Vec<&FieldDef> = sd.non_union_fields().collect();
        let union_fields: Vec<&FieldDef> = sd.union_fields().collect();

        let disc_byte_off = sd.discriminant_offset * 2;

        // Result type — typed discriminated union so callers narrow on `.variant`.
        // One arm per union variant (data type mirroring the parser output) plus a
        // shared block of non-union fields and the parser's `unknown` default arm.
        let shared_fields: Vec<String> = non_union_fields
            .iter()
            .map(|f| {
                let ts_type = capnp_to_ts_type(&f.type_name);
                // Struct pointer fields may be null when the capnp pointer is null.
                let ts_type = if f.section == FieldSection::Pointer
                    && !f.type_name.starts_with("List(")
                    && f.type_name != "Text"
                    && f.type_name != "Data"
                    && !super::is_primitive(&f.type_name)
                {
                    format!("{ts_type} | null")
                } else {
                    ts_type
                };
                format!("{}: {}", to_camel_case(&f.name), ts_type)
            })
            .collect();
        let arms: Vec<(String, String)> = union_fields
            .iter()
            .map(|f| (f.name.clone(), super::union_variant_data_type(f)))
            .collect();
        emit_result_union_alias(out, &format!("{}Result", sd.name), &shared_fields, &arms);

        // Parser function
        out.push_str(&format!(
            "export function parse{}(bytes: Uint8Array): {}Result {{\n",
            sd.name, sd.name
        ));
        out.push_str(&format!(
            "  const reader = new CapnpReader(bytes, {}, {});\n",
            sd.data_words, sd.pointer_words
        ));

        // Read non-union fields
        for f in &non_union_fields {
            emit_reader_field(out, "reader", f, &to_camel_case(&f.name), "  ", schema);
        }

        // Read discriminant
        out.push_str(&format!(
            "  const _disc = reader.getUint16({disc_byte_off});\n"
        ));
        out.push_str("  switch (_disc) {\n");

        for variant in &union_fields {
            out.push_str(&format!(
                "    case {}: // {}\n",
                variant.discriminant_value, variant.name
            ));

            if variant.type_name == "Void" {
                emit_return(out, &non_union_fields, &variant.name, "undefined");
            } else if variant.type_name == "Bool" {
                let byte_off = data_byte_offset(variant);
                let bit = bool_bit_index(variant);
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getBool({byte_off}, {bit})"),
                );
            } else if is_data_scalar(&variant.type_name) {
                let byte_off = data_byte_offset(variant);
                let method = super::getter_method(&variant.type_name);
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.{method}({byte_off})"),
                );
            } else if variant.type_name == "Text" {
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getText({})", variant.slot_offset),
                );
            } else if variant.type_name == "Data" {
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getData({})", variant.slot_offset),
                );
            } else if variant.type_name.starts_with("List(Text") {
                emit_return(
                    out,
                    &non_union_fields,
                    &variant.name,
                    &format!("reader.getTextList({})", variant.slot_offset),
                );
            } else if let Some(inner) = super::extract_list_inner_type(&variant.type_name) {
                if let Some(isd) = schema.structs.iter().find(|s| s.name == inner) {
                    emit_struct_list_read(
                        out,
                        "reader",
                        variant,
                        isd,
                        &non_union_fields,
                        &variant.name,
                        schema,
                    );
                } else {
                    emit_return(out, &non_union_fields, &variant.name, "[]");
                }
            } else {
                // Struct type
                if let Some(struct_def) =
                    schema.structs.iter().find(|s| s.name == variant.type_name)
                {
                    emit_struct_read(
                        out,
                        "reader",
                        variant,
                        struct_def,
                        &non_union_fields,
                        &variant.name,
                        schema,
                    );
                } else {
                    emit_return(out, &non_union_fields, &variant.name, "null");
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
            "      return {{ {fields_str}{comma}variant: 'unknown' as const, data: null }};\n"
        ));
        out.push_str("  }\n");
        out.push_str("}\n\n");
    }
}

/// Generate a parser function for a non-union struct.
///
/// `export function parse{StructName}(bytes: Uint8Array): StructName`
fn generate_plain_struct_parser(out: &mut String, sd: &StructDef, schema: &ParsedSchema) {
    out.push_str(&format!(
        "export function parse{}(bytes: Uint8Array): {} {{\n",
        sd.name, sd.name
    ));
    out.push_str(&format!(
        "  const reader = new CapnpReader(bytes, {}, {});\n",
        sd.data_words, sd.pointer_words
    ));

    // Gensym counter: _f{n} locals are always valid identifiers regardless of field name.
    // This avoids reserved-word collisions (function, arguments, ...) and name-capture
    // bugs (a field named "reader" would shadow the CapnpReader local above).
    let mut field_pairs: Vec<(String, String)> = Vec::new();
    for (n, f) in sd.fields.iter().enumerate() {
        let camel = to_camel_case(&f.name);
        let local = format!("_f{n}");
        emit_reader_field(out, "reader", f, &local, "  ", schema);
        field_pairs.push((camel, local));
    }

    let props: Vec<String> = field_pairs
        .iter()
        .map(|(key, local)| format!("{key}: {local}"))
        .collect();
    out.push_str(&format!("  return {{ {} }};\n", props.join(", ")));
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

    // Handle nested scoped response variants by recursively parsing
    for nc in &sc.nested_clients {
        let nested_resp_name = format!("{}Result", nc.factory_name);

        // Skip if already handled in inner_response_variants loop
        if sc
            .inner_response_variants
            .iter()
            .any(|v| v.name == nested_resp_name)
        {
            continue;
        }

        // Find the matching discriminant field
        let nested_field = irs
            .fields
            .iter()
            .find(|f| f.name == nested_resp_name && f.discriminant_value != 0xFFFF);

        let Some(nf) = nested_field else { continue };

        // Find the nested response struct to know its layout
        let nested_resp_struct = schema.structs.iter().find(|s| s.name == nf.type_name);

        out.push_str(&format!(
            "        case {}: // {} (nested scope)\n",
            nf.discriminant_value, nested_resp_name
        ));

        if let Some(nrs) = nested_resp_struct {
            // Read the nested struct and recursively parse its union
            out.push_str(&format!(
                "        {{\n          const _nested = _inner.getStruct({}, {}, {});\n",
                nf.slot_offset, nrs.data_words, nrs.pointer_words
            ));
            out.push_str(&format!(
                "          if (!_nested) return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: null }} }};\n",
                nested_resp_name
            ));

            // Read nested discriminant and switch
            let nested_disc_byte_off = nrs.discriminant_offset * 2;
            out.push_str(&format!(
                "          const _nestedDisc = _nested.getUint16({nested_disc_byte_off});\n"
            ));
            out.push_str("          switch (_nestedDisc) {\n");

            // Emit each variant of the nested response
            for nv in &nc.inner_response_variants {
                let nv_field = nrs
                    .fields
                    .iter()
                    .find(|f| f.name == nv.name && f.discriminant_value != 0xFFFF);
                let nv_disc = nv_field.map(|f| f.discriminant_value).unwrap_or(0);
                let nv_data =
                    emit_inner_variant_read_from("_nested", nv_field, &nv.type_name, schema, 0);
                out.push_str(&format!("            case {nv_disc}: // {}\n", nv.name));
                out.push_str(&format!(
                    "              return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {{ variant: '{}', data: {nv_data} }} }} }};\n",
                    nested_resp_name, nv.name
                ));
            }

            // Handle nested-nested clients (e.g., worktree → ctl)
            for nnc in &nc.nested_clients {
                let nn_resp_name = format!("{}Result", nnc.factory_name);
                if nc
                    .inner_response_variants
                    .iter()
                    .any(|v| v.name == nn_resp_name)
                {
                    continue;
                }
                let nn_field = nrs
                    .fields
                    .iter()
                    .find(|f| f.name == nn_resp_name && f.discriminant_value != 0xFFFF);
                let Some(nnf) = nn_field else { continue };
                let nn_struct = schema.structs.iter().find(|s| s.name == nnf.type_name);

                out.push_str(&format!(
                    "            case {}: // {} (nested scope)\n",
                    nnf.discriminant_value, nn_resp_name
                ));

                if let Some(nns) = nn_struct {
                    out.push_str(&format!(
                        "            {{\n              const _deep = _nested.getStruct({}, {}, {});\n",
                        nnf.slot_offset, nns.data_words, nns.pointer_words
                    ));
                    out.push_str(&format!(
                        "              if (!_deep) return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {{ variant: '{}', data: null }} }} }};\n",
                        nested_resp_name, nn_resp_name
                    ));
                    let nn_disc_off = nns.discriminant_offset * 2;
                    out.push_str(&format!(
                        "              const _deepDisc = _deep.getUint16({nn_disc_off});\n"
                    ));
                    out.push_str("              switch (_deepDisc) {\n");
                    for nnv in &nnc.inner_response_variants {
                        let nnv_field = nns
                            .fields
                            .iter()
                            .find(|f| f.name == nnv.name && f.discriminant_value != 0xFFFF);
                        let nnv_disc = nnv_field.map(|f| f.discriminant_value).unwrap_or(0);
                        let nnv_data = emit_inner_variant_read_from(
                            "_deep",
                            nnv_field,
                            &nnv.type_name,
                            schema,
                            0,
                        );
                        out.push_str(&format!(
                            "                case {nnv_disc}: // {}\n",
                            nnv.name
                        ));
                        out.push_str(&format!(
                            "                  return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {{ variant: '{}', data: {{ variant: '{}', data: {nnv_data} }} }} }} }};\n",
                            nested_resp_name, nn_resp_name, nnv.name
                        ));
                    }
                    out.push_str(&format!(
                        "                default:\n                  return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {{ variant: '{}', data: {{ variant: 'unknown', data: null }} }} }} }};\n",
                        nested_resp_name, nn_resp_name
                    ));
                    out.push_str("              }\n");
                    out.push_str("            }\n");
                } else {
                    out.push_str(&format!(
                        "              return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {{ variant: '{}', data: null }} }} }};\n",
                        nested_resp_name, nn_resp_name
                    ));
                }
            }

            out.push_str(&format!(
                "            default:\n              return {{ {fields_str}{comma}variant: '{variant_name}', data: {{ variant: '{}', data: {{ variant: 'unknown', data: null }} }} }};\n",
                nested_resp_name
            ));
            out.push_str("          }\n");
            out.push_str("        }\n");
        } else {
            // Can't find nested struct — fall back to null
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
///
/// `reader_var` is the variable name of the reader to use (e.g., "_inner", "_el").
/// `depth` is threaded through to `emit_struct_pointer_expr` to avoid `_p{n}` TDZ collisions.
fn emit_inner_variant_read_from(
    reader_var: &str,
    field: Option<&FieldDef>,
    type_name: &str,
    schema: &ParsedSchema,
    depth: usize,
) -> String {
    let Some(f) = field else {
        return "null".into();
    };

    match type_name {
        "Void" => "undefined".into(),
        "Bool" => {
            let byte_off = data_byte_offset(f);
            let bit = bool_bit_index(f);
            format!("{reader_var}.getBool({byte_off}, {bit})")
        }
        "Text" => format!("{reader_var}.getText({})", f.slot_offset),
        "Data" => format!("{reader_var}.getData({})", f.slot_offset),
        t if is_data_scalar(t) => {
            let byte_off = data_byte_offset(f);
            let method = super::getter_method(type_name);
            format!("{reader_var}.{method}({byte_off})")
        }
        t if t.starts_with("List(Text") => {
            format!("{reader_var}.getTextList({})", f.slot_offset)
        }
        t if t.starts_with("List(") => {
            // List(Struct) — read as mapped array
            let inner = super::extract_list_inner_type(t).unwrap_or("");
            if let Some(sd) = schema.structs.iter().find(|s| s.name == inner) {
                emit_struct_list_field_expr(reader_var, f.slot_offset, sd, schema)
            } else {
                "[]".into()
            }
        }
        _ => {
            // Struct — read its fields as an object via getStruct
            if let Some(sd) = schema.structs.iter().find(|s| s.name == type_name) {
                emit_struct_pointer_expr(reader_var, f.slot_offset, sd, schema, depth)
            } else {
                "null".into()
            }
        }
    }
}

/// Convenience wrapper using `_inner` as the reader variable (for scoped response parsing).
fn emit_inner_variant_read(
    field: Option<&FieldDef>,
    type_name: &str,
    schema: &ParsedSchema,
) -> String {
    emit_inner_variant_read_from("_inner", field, type_name, schema, 0)
}

// ---------------------------------------------------------------------------
// List(Struct) helpers
// ---------------------------------------------------------------------------

/// Emit an inline IIFE that reads a struct pointer and returns an object literal.
///
/// Uses `_p{depth}` variable names to avoid temporal dead zone collisions in nested IIFEs.
fn emit_struct_pointer_expr(
    reader_var: &str,
    slot: u32,
    sd: &StructDef,
    schema: &ParsedSchema,
    depth: usize,
) -> String {
    if depth > 4 {
        return "null".into(); // guard against hypothetical recursive schemas
    }
    let var_name = format!("_p{depth}");

    if sd.has_union {
        // Union struct pointer — generate discriminant switch inside an IIFE
        let disc_byte_off = sd.discriminant_offset * 2;

        let non_union_fields: Vec<&FieldDef> = sd.non_union_fields().collect();
        let union_fields: Vec<&FieldDef> = sd.union_fields().collect();

        let mut s = format!(
            "(() => {{ const {var_name} = {reader_var}.getStruct({slot}, {}, {}); if (!{var_name}) return null;\n",
            sd.data_words, sd.pointer_words
        );

        // Read non-union fields
        let mut nuf_names = Vec::new();
        for f in &non_union_fields {
            let camel = to_camel_case(&f.name);
            let read = emit_struct_element_field_read_inner(&var_name, f, schema, depth + 1);
            s.push_str(&format!("  const {camel} = {read};\n"));
            nuf_names.push(camel);
        }

        let nuf_str = nuf_names.join(", ");
        let nuf_comma = if nuf_str.is_empty() { "" } else { ", " };

        s.push_str(&format!(
            "  const _disc = {var_name}.getUint16({disc_byte_off});\n"
        ));
        s.push_str("  switch (_disc) {\n");

        for uf in &union_fields {
            s.push_str(&format!("    case {}: {{\n", uf.discriminant_value));
            let data_expr =
                emit_inner_variant_read_from(&var_name, Some(uf), &uf.type_name, schema, depth + 1);
            s.push_str(&format!(
                "      return {{ {nuf_str}{nuf_comma}variant: '{}' as const, data: {data_expr} }};\n",
                uf.name
            ));
            s.push_str("    }\n");
        }

        s.push_str(&format!(
            "    default: return {{ {nuf_str}{nuf_comma}variant: 'unknown' as const, data: null }};\n"
        ));
        s.push_str("  }\n");
        s.push_str("})()");
        s
    } else {
        let fields: Vec<String> = sd
            .non_union_fields()
            .map(|sf| {
                let camel = to_camel_case(&sf.name);
                let read = emit_struct_element_field_read_inner(&var_name, sf, schema, depth + 1);
                format!("{camel}: {read}")
            })
            .collect();
        format!(
            "(() => {{ const {var_name} = {reader_var}.getStruct({slot}, {}, {}); return {var_name} ? {{ {} }} : null; }})()",
            sd.data_words, sd.pointer_words, fields.join(", ")
        )
    }
}

/// Generate an inline read expression for a single struct field (thin wrapper).
fn emit_struct_element_field_read(
    reader_var: &str,
    field: &FieldDef,
    schema: &ParsedSchema,
) -> String {
    emit_struct_element_field_read_inner(reader_var, field, schema, 0)
}

/// Emit a read expression for a pointer-section field.
///
/// Handles: List(Text), List(Data), List(Struct), Struct, Text, Data.
fn emit_pointer_read_expr(
    reader_var: &str,
    field: &FieldDef,
    schema: &ParsedSchema,
    depth: usize,
) -> String {
    match super::extract_list_inner_type(&field.type_name) {
        Some("Text") => {
            format!("{reader_var}.getTextList({})", field.slot_offset)
        }
        Some("Data") => {
            format!("{reader_var}.getDataList({})", field.slot_offset)
        }
        Some(inner) => {
            // List(Struct) — look up struct def, map elements.
            // For primitive lists (List(Int64), List(UInt32), etc.) we don't yet have
            // a runtime getter, so emit a typed empty array to satisfy TypeScript.
            if let Some(isd) = schema.structs.iter().find(|s| s.name == inner) {
                emit_struct_list_field_expr(reader_var, field.slot_offset, isd, schema)
            } else {
                let elem_ts = super::capnp_to_ts_type(inner);
                format!("([] as {elem_ts}[])")
            }
        }
        None => {
            // Non-list pointer: Struct, Text, Data, or unknown
            if let Some(sd) = schema.structs.iter().find(|s| s.name == field.type_name) {
                if let Some(inner_name) = sd.option_inner_type() {
                    emit_option_read_expr(reader_var, field, sd, inner_name, schema)
                } else {
                    emit_struct_pointer_expr(reader_var, field.slot_offset, sd, schema, depth)
                }
            } else if field.type_name == "Text" || field.type_name == "Data" {
                let method = super::getter_method(&field.type_name);
                format!("{reader_var}.{method}({})", field.slot_offset)
            } else {
                // Unknown pointer type (AnyPointer, Interface, etc.)
                "null".into()
            }
        }
    }
}

/// Generate an inline read expression for a single struct field.
/// Handles data scalars, pointers (Text, Data, List(Text), List(Struct), Struct), and groups.
fn emit_struct_element_field_read_inner(
    reader_var: &str,
    field: &FieldDef,
    schema: &ParsedSchema,
    depth: usize,
) -> String {
    match field.section {
        FieldSection::Data => {
            if field.type_name == "Void" {
                return "undefined".into();
            }
            let byte_off = data_byte_offset(field);
            if field.type_name == "Bool" {
                let bit = bool_bit_index(field);
                format!("{reader_var}.getBool({byte_off}, {bit})")
            } else if !super::is_data_scalar(&field.type_name) {
                // Enum type — stored as UInt16 in data section
                if let Some(ed) = schema.enums.iter().find(|e| e.name == field.type_name) {
                    super::emit_enum_getter_expr(reader_var, byte_off, &field.type_name, ed)
                } else {
                    format!("{reader_var}.getUint16({byte_off})")
                }
            } else {
                let method = super::getter_method(&field.type_name);
                format!("{reader_var}.{method}({byte_off})")
            }
        }
        FieldSection::Pointer => emit_pointer_read_expr(reader_var, field, schema, depth),
        FieldSection::Group => "undefined".into(),
    }
}

/// Generate an inline expression that reads a List(Struct) and maps elements to objects.
///
/// For non-union structs: maps to `{ field1, field2, ... }`.
/// For union structs: maps to `{ variant, data, ...nonUnionFields }` via discriminant switch.
fn emit_struct_list_field_expr(
    reader_var: &str,
    slot: u32,
    sd: &StructDef,
    schema: &ParsedSchema,
) -> String {
    if sd.has_union {
        return emit_union_struct_list_field_expr(reader_var, slot, sd, schema);
    }

    let fields: Vec<String> = sd
        .non_union_fields()
        .map(|f| {
            let camel = to_camel_case(&f.name);
            let read = emit_struct_element_field_read_inner("_el", f, schema, 0);
            format!("{camel}: {read}")
        })
        .collect();
    format!(
        "{reader_var}.getStructList({}, {}, {}).map(_el => ({{ {} }}))",
        slot,
        sd.data_words,
        sd.pointer_words,
        fields.join(", ")
    )
}

/// Generate a multi-line expression for reading a List(UnionStruct).
///
/// Each element is parsed as `{ variant: string; data: unknown }` with optional
/// non-union fields (for mixed structs like TuiFrame).
fn emit_union_struct_list_field_expr(
    reader_var: &str,
    slot: u32,
    sd: &StructDef,
    schema: &ParsedSchema,
) -> String {
    let disc_byte_off = sd.discriminant_offset * 2;

    let non_union_fields: Vec<&FieldDef> = sd.non_union_fields().collect();
    let union_fields: Vec<&FieldDef> = sd.union_fields().collect();

    let mut s = format!(
        "{reader_var}.getStructList({slot}, {}, {}).map(_el => {{\n",
        sd.data_words, sd.pointer_words
    );

    // Read non-union fields
    let mut nuf_names = Vec::new();
    for f in &non_union_fields {
        let camel = to_camel_case(&f.name);
        let read = emit_struct_element_field_read_inner("_el", f, schema, 0);
        s.push_str(&format!("    const {camel} = {read};\n"));
        nuf_names.push(camel);
    }

    let nuf_str = nuf_names.join(", ");
    let nuf_comma = if nuf_str.is_empty() { "" } else { ", " };

    s.push_str(&format!(
        "    const _disc = _el.getUint16({disc_byte_off});\n"
    ));
    s.push_str("    switch (_disc) {\n");

    for uf in &union_fields {
        s.push_str(&format!(
            "      case {}: // {}\n",
            uf.discriminant_value, uf.name
        ));
        let data_expr = emit_inner_variant_read_from("_el", Some(uf), &uf.type_name, schema, 0);
        // `as const` narrows the literal so the mapped array types as the typed
        // discriminated union (e.g. StreamPayload[]) rather than widening to
        // `{ variant: string; ... }[]`.
        s.push_str(&format!(
            "        return {{ {nuf_str}{nuf_comma}variant: '{}' as const, data: {data_expr} }};\n",
            uf.name
        ));
    }

    s.push_str(&format!(
        "      default:\n        return {{ {nuf_str}{nuf_comma}variant: 'unknown' as const, data: null }};\n"
    ));
    s.push_str("    }\n");
    s.push_str("  })");
    s
}

/// Emit a top-level List(Struct) variant read wrapped in a block.
fn emit_struct_list_read(
    out: &mut String,
    reader_var: &str,
    ptr_field: &FieldDef,
    struct_def: &StructDef,
    non_union_fields: &[&FieldDef],
    variant_name: &str,
    schema: &ParsedSchema,
) {
    let expr = emit_struct_list_field_expr(reader_var, ptr_field.slot_offset, struct_def, schema);
    out.push_str(&format!("    {{\n      const _items = {expr};\n"));
    emit_return(out, non_union_fields, variant_name, "_items");
    out.push_str("    }\n");
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
    schema: &ParsedSchema,
) {
    match field.section {
        FieldSection::Data => {
            if field.type_name == "Void" {
                out.push_str(&format!("{indent}const {local_name} = undefined;\n"));
                return;
            }
            let byte_off = data_byte_offset(field);
            if field.type_name == "Bool" {
                let bit = bool_bit_index(field);
                out.push_str(&format!(
                    "{indent}const {local_name} = {reader_var}.getBool({byte_off}, {bit});\n"
                ));
            } else if !super::is_data_scalar(&field.type_name) {
                // Enum type — stored as UInt16 in data section
                if let Some(ed) = schema.enums.iter().find(|e| e.name == field.type_name) {
                    let expr =
                        super::emit_enum_getter_expr(reader_var, byte_off, &field.type_name, ed);
                    out.push_str(&format!("{indent}const {local_name} = {expr};\n"));
                } else {
                    out.push_str(&format!(
                        "{indent}const {local_name} = {reader_var}.getUint16({byte_off});\n"
                    ));
                }
            } else {
                let method = super::getter_method(&field.type_name);
                out.push_str(&format!(
                    "{indent}const {local_name} = {reader_var}.{method}({byte_off});\n"
                ));
            }
        }
        FieldSection::Pointer => {
            let read_expr = emit_pointer_read_expr(reader_var, field, schema, 0);
            out.push_str(&format!("{indent}const {local_name} = {read_expr};\n"));
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
    // `as const` narrows the discriminant literal so the return matches the
    // corresponding arm of the typed `…Result` discriminated union.
    out.push_str(&format!(
        "      return {{ {fields_str}{comma}variant: '{variant_name}' as const, data: {data_expr} }};\n"
    ));
}

/// Emit a typed discriminated-union `type` alias for a parser result.
///
/// Shape (one arm per union variant plus the parser's `unknown` default case):
/// ```text
/// export type {name} =
///   | { {shared}variant: '{v}'; data: {ty} }
///   | { {shared}variant: 'unknown'; data: null };
/// ```
/// `shared_fields` are the non-union field declarations (`name: type`) repeated in
/// every arm — matching what every parser `return` carries alongside `variant`/`data`.
/// Keeping the arm `data` types aligned with the parser's per-variant output lets
/// callers narrow on `result.variant`.
fn emit_result_union_alias(
    out: &mut String,
    name: &str,
    shared_fields: &[String],
    arms: &[(String, String)],
) {
    let shared = shared_fields.join("; ");
    let shared_prefix = if shared.is_empty() {
        String::new()
    } else {
        format!("{shared}; ")
    };
    out.push_str(&format!("export type {name} =\n"));
    for (variant, data_ty) in arms {
        out.push_str(&format!(
            "  | {{ {shared_prefix}variant: '{variant}'; data: {data_ty} }}\n"
        ));
    }
    out.push_str(&format!(
        "  | {{ {shared_prefix}variant: 'unknown'; data: null }};\n\n"
    ));
}

fn emit_struct_read(
    out: &mut String,
    reader_var: &str,
    ptr_field: &FieldDef,
    struct_def: &StructDef,
    non_union_fields: &[&FieldDef],
    variant_name: &str,
    schema: &ParsedSchema,
) {
    out.push_str(&format!(
        "    {{\n      const _s = {reader_var}.getStruct({}, {}, {});\n",
        ptr_field.slot_offset, struct_def.data_words, struct_def.pointer_words
    ));

    let visible_fields: Vec<&FieldDef> = struct_def.non_union_fields().collect();

    out.push_str("      const _data = _s ? {\n");
    for sf in &visible_fields {
        let camel = to_camel_case(&sf.name);
        let read_expr = emit_struct_element_field_read("_s", sf, schema);
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

/// Generate an inline expression that reads an Option* wrapper struct pointer.
///
/// Returns `T | undefined`: reads the struct pointer, checks the discriminant,
/// and returns the inner value if discriminant == 1 (some), else `undefined`.
fn emit_option_read_expr(
    reader_var: &str,
    field: &FieldDef,
    opt_struct: &StructDef,
    inner_type_name: &str,
    schema: &ParsedSchema,
) -> String {
    let disc_byte_off = opt_struct.discriminant_offset * 2;

    let read_val = if inner_type_name == "Text" {
        "_optR.getText(0)".to_owned()
    } else if inner_type_name == "Data" {
        "_optR.getData(0)".to_owned()
    } else {
        // Data-section scalar or enum: find the 'some' field to get its byte offset
        let some_field = opt_struct.fields.iter().find(|f| f.name == "some");
        if let Some(sf) = some_field {
            let some_byte_off = super::data_byte_offset(sf);
            // Enum types are stored as UInt16 — generate array lookup + cast
            if let Some(ed) = schema.enums.iter().find(|e| e.name == inner_type_name) {
                super::emit_enum_getter_expr("_optR", some_byte_off, inner_type_name, ed)
            } else if inner_type_name == "Bool" {
                let bit = super::bool_bit_index(sf);
                format!("_optR.getBool({some_byte_off}, {bit})")
            } else {
                let getter = super::getter_method(inner_type_name);
                format!("_optR.{getter}({some_byte_off})")
            }
        } else {
            "undefined".to_owned()
        }
    };

    format!(
        "(() => {{ const _optR = {reader_var}.getStruct({}, {}, {}); \
         if (!_optR) return undefined; \
         return _optR.getUint16({disc_byte_off}) === 1 ? {read_val} : undefined; }})()",
        field.slot_offset, opt_struct.data_words, opt_struct.pointer_words
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::path::Path;

    use hyprstream_rpc_build::schema::parse_from_cgr_path;
    use hyprstream_rpc_build::schema::types::ParsedSchema;

    /// A union struct with a shared non-union field plus Void / Text / scalar arms.
    const UNION_SCHEMA: &str = r#"
@0xd00dfeedd00dfeed;

struct Payload {
  seq @0 :UInt32;
  union {
    none @1 :Void;
    text @2 :Text;
    count @3 :UInt64;
  }
}
"#;

    /// A service-shaped schema: `{Pascal}Request`/`{Pascal}Response` unions so the
    /// CGR reader populates `response_struct` and the top-level `generate_parsers`
    /// path runs.
    const SERVICE_SCHEMA: &str = r#"
@0xbeefcafebeefcafe;

struct PayloadRequest {
  union {
    ping @0 :Void;
    echo @1 :Text;
  }
}

struct PayloadResponse {
  seq @0 :UInt32;
  union {
    ok @1 :Text;
    fail @2 :UInt64;
  }
}
"#;

    /// Compile `schema_src` to a CGR keyed on `name` and parse it. Uses a
    /// per-(pid, name) temp dir so parallel tests don't collide.
    fn parse_schema(name: &str, schema_src: &str) -> ParsedSchema {
        let tmp = std::env::temp_dir().join(format!(
            "hyprstream_ts_{name}_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let capnp_path = tmp.join(format!("{name}.capnp"));
        std::fs::write(&capnp_path, schema_src).expect("write capnp");

        let cgr_path = tmp.join(format!("{name}.cgr"));
        capnpc::CompilerCommand::new()
            .src_prefix(&tmp)
            .file(&capnp_path)
            .raw_code_generator_request_path(&cgr_path)
            .run()
            .expect("compile capnp to CGR");

        let parsed = parse_from_cgr_path(Path::new(&cgr_path), name).expect("parse CGR");
        let _ = std::fs::remove_dir_all(&tmp);
        parsed
    }

    #[test]
    fn struct_parser_emits_typed_discriminated_union() {
        let schema = parse_schema("structfix", UNION_SCHEMA);
        let mut out = String::new();
        super::generate_struct_parsers(&mut out, &schema);

        // The result type must be a typed discriminated union — NOT the old
        // `interface … { variant: string; data: unknown }` stub.
        assert!(
            out.contains("export type PayloadResult ="),
            "expected a typed union alias, got:\n{out}"
        );
        assert!(
            !out.contains("variant: string;"),
            "must not emit the degenerate `variant: string` stub:\n{out}"
        );
        assert!(
            !out.contains("data: unknown;"),
            "must not emit the degenerate `data: unknown` stub:\n{out}"
        );

        // Each arm carries the shared non-union field and a typed `data` slot.
        assert!(
            out.contains("| { seq: number; variant: 'none'; data: undefined }"),
            "Void arm shape wrong:\n{out}"
        );
        assert!(
            out.contains("| { seq: number; variant: 'text'; data: string }"),
            "Text arm shape wrong:\n{out}"
        );
        assert!(
            out.contains("| { seq: number; variant: 'count'; data: bigint }"),
            "scalar arm shape wrong:\n{out}"
        );
        // Plus the parser's default `unknown` arm.
        assert!(
            out.contains("| { seq: number; variant: 'unknown'; data: null };"),
            "fallback arm shape wrong:\n{out}"
        );

        // Parser `return`s narrow the discriminant with `as const` so they match
        // the union arms.
        assert!(
            out.contains("variant: 'text' as const"),
            "variant return missing `as const`:\n{out}"
        );
        assert!(
            out.contains("variant: 'unknown' as const"),
            "default return missing `as const`:\n{out}"
        );
    }

    #[test]
    fn top_level_response_parser_emits_typed_discriminated_union() {
        let schema = parse_schema("payload", SERVICE_SCHEMA);
        let mut out = String::new();
        super::generate_parsers(&mut out, "payload", &schema);

        // Top-level `{Service}ResponseResult` is now a typed union, not the
        // `interface … { variant: string; data: unknown }` stub.
        assert!(
            out.contains("export type PayloadResponseResult ="),
            "expected a typed union alias, got:\n{out}"
        );
        assert!(
            !out.contains("variant: string;"),
            "must not emit the degenerate `variant: string` stub:\n{out}"
        );
        assert!(
            !out.contains("data: unknown;"),
            "must not emit the degenerate `data: unknown` stub:\n{out}"
        );
        assert!(
            out.contains("| { seq: number; variant: 'ok'; data: string }"),
            "Text arm shape wrong:\n{out}"
        );
        assert!(
            out.contains("| { seq: number; variant: 'fail'; data: bigint }"),
            "scalar arm shape wrong:\n{out}"
        );
        assert!(
            out.contains("| { seq: number; variant: 'unknown'; data: null };"),
            "fallback arm shape wrong:\n{out}"
        );
        assert!(
            out.contains("variant: 'ok' as const"),
            "variant return missing `as const`:\n{out}"
        );
    }
}
