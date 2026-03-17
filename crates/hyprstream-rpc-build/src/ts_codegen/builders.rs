//! Generate TypeScript request builder functions.
//!
//! For each request union variant, generates a function that constructs
//! a Cap'n Proto binary message with correct wire format offsets.
//!
//! Scoped clients get builder functions that construct the full chain:
//! root → init inner struct → set scope fields → set discriminant → payload.

use hyprstream_rpc_build::schema::types::{EnumDef, FieldDef, FieldSection, ParsedSchema, ScopedClient, StructDef};
use hyprstream_rpc_build::util::{to_camel_case, to_pascal_case};

use super::{bool_bit_index, capnp_to_ts_type, data_byte_offset, is_data_scalar, is_primitive};

/// Generate all request builder functions for a service.
pub fn generate_builders(out: &mut String, service_name: &str, schema: &ParsedSchema) {
    let req_struct = match &schema.request_struct {
        Some(s) => s,
        None => return,
    };

    let pascal = to_pascal_case(service_name);

    // Non-union fields from the root request struct (e.g., `id`)
    let non_union_fields: Vec<&FieldDef> = req_struct
        .fields
        .iter()
        .filter(|f| f.discriminant_value == 0xFFFF)
        .collect();

    // Discriminant byte offset (discriminant_offset is in u16 units)
    let disc_byte_off = req_struct.discriminant_offset * 2;

    // Filter out scoped variants (they get separate builders)
    let scoped_names: Vec<&str> = schema
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Top-level (non-scoped) builder functions
    for variant in &schema.request_variants {
        if scoped_names.contains(&variant.name.as_str()) {
            continue;
        }

        let variant_field = req_struct
            .fields
            .iter()
            .find(|f| f.name == variant.name && f.discriminant_value != 0xFFFF);

        let disc_value = variant_field.map(|f| f.discriminant_value).unwrap_or(0);

        let is_void = variant.type_name == "Void";
        let is_prim = is_primitive(&variant.type_name) && !is_void;
        let payload_struct = if !is_void && !is_prim {
            schema.structs.iter().find(|s| s.name == variant.type_name)
        } else {
            None
        };

        let fn_name = format!("build{pascal}Request_{}", variant.name);

        // Parameters: non-union fields + optional payload
        let mut params = Vec::new();
        for f in &non_union_fields {
            params.push(format!(
                "{}: {}",
                to_camel_case(&f.name),
                capnp_to_ts_type(&f.type_name)
            ));
        }
        if !is_void {
            if let Some(ps) = payload_struct {
                params.push(format!("p: {}", ps.name));
            } else if is_prim {
                params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
            }
        }

        out.push_str(&format!(
            "function {fn_name}({}): Uint8Array {{\n",
            params.join(", ")
        ));
        // Estimate slack for pointer targets: 16 words (128 B) per pointer field.
        // grow() handles overflow automatically via wasm.alloc/dealloc.
        let extra_words = std::cmp::max(8usize, req_struct.pointer_words as usize * 16);
        out.push_str(&format!(
            "  const msg = CapnpArena.intoWasm({}, {}, {extra_words});\n",
            req_struct.data_words, req_struct.pointer_words
        ));

        // Per-function gensym counter — unique IDs for list builder variables
        let mut counter = 0u32;

        // Set non-union fields
        for f in &non_union_fields {
            emit_field_setter(out, "msg", f, &to_camel_case(&f.name), "  ", &schema.enums, &schema.structs, &mut counter);
        }

        // Set discriminant
        out.push_str(&format!(
            "  msg.setUint16({disc_byte_off}, {disc_value}); // {}\n",
            variant.name
        ));

        // Set variant payload
        if is_void {
            // Nothing
        } else if is_prim {
            if let Some(vf) = variant_field {
                emit_field_setter(out, "msg", vf, "p", "  ", &schema.enums, &schema.structs, &mut counter);
            }
        } else if let Some(ps) = payload_struct {
            if let Some(vf) = variant_field {
                emit_struct_init(out, vf, ps, "msg", "p", "  ", schema, &mut counter);
            }
        }

        out.push_str("  return msg.finish();\n");
        out.push_str("}\n\n");
    }

    // Scoped builder functions (recursive for nested clients)
    for sc in &schema.scoped_clients {
        generate_scoped_builders(
            out,
            service_name,
            schema,
            sc,
            &[],
            req_struct,
            &non_union_fields,
        );
    }
}

/// Generate builder functions for a scoped client's methods.
///
/// `ancestors` tracks the scoped client chain from outermost to the parent of `sc`.
/// Each ancestor's scope fields + discriminant are written into the chain.
#[allow(clippy::too_many_arguments)]
fn generate_scoped_builders(
    out: &mut String,
    service_name: &str,
    schema: &ParsedSchema,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    root_struct: &StructDef,
    root_non_union_fields: &[&FieldDef],
) {
    let pascal = to_pascal_case(service_name);

    // Build the function name prefix from the ancestor chain
    let mut name_chain = String::new();
    for anc in ancestors {
        name_chain.push_str(&format!("_{}", anc.factory_name));
    }
    name_chain.push_str(&format!("_{}", sc.factory_name));

    // Filter out nested scoped variants (they get their own builders)
    let nested_names: Vec<&str> = sc
        .nested_clients
        .iter()
        .map(|nc| nc.factory_name.as_str())
        .collect();

    // All scope fields from ancestors + current
    let all_scope_fields: Vec<&FieldDef> = ancestors
        .iter()
        .flat_map(|a| a.scope_fields.iter())
        .chain(sc.scope_fields.iter())
        .collect();

    for variant in &sc.inner_request_variants {
        if nested_names.contains(&variant.name.as_str()) {
            continue;
        }

        let is_void = variant.type_name == "Void";
        let is_prim = is_primitive(&variant.type_name) && !is_void;
        let payload_struct = if !is_void && !is_prim {
            schema.structs.iter().find(|s| s.name == variant.type_name)
        } else {
            None
        };

        let fn_name = format!("build{pascal}Request{name_chain}_{}", variant.name);

        // Parameters: root non-union fields + all scope fields + optional payload
        let mut params = Vec::new();
        for f in root_non_union_fields {
            params.push(format!(
                "{}: {}",
                to_camel_case(&f.name),
                capnp_to_ts_type(&f.type_name)
            ));
        }
        for f in &all_scope_fields {
            params.push(format!(
                "{}: {}",
                to_camel_case(&f.name),
                capnp_to_ts_type(&f.type_name)
            ));
        }
        if !is_void {
            if let Some(ps) = payload_struct {
                params.push(format!("p: {}", ps.name));
            } else if is_prim {
                params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
            }
        }

        out.push_str(&format!(
            "function {fn_name}({}): Uint8Array {{\n",
            params.join(", ")
        ));
        let extra_words = std::cmp::max(8usize, root_struct.pointer_words as usize * 16);
        out.push_str(&format!(
            "  const msg = CapnpArena.intoWasm({}, {}, {extra_words});\n",
            root_struct.data_words, root_struct.pointer_words
        ));

        // Per-function gensym counter — unique IDs for list builder variables
        let mut counter = 0u32;

        // Set root non-union fields
        for f in root_non_union_fields {
            emit_field_setter(out, "msg", f, &to_camel_case(&f.name), "  ", &schema.enums, &schema.structs, &mut counter);
        }

        // Build the chain: root → ancestors → current → method
        // Level 0: root → first scope (sc or first ancestor)
        let chain: Vec<&ScopedClient> = ancestors.iter().copied().chain(std::iter::once(sc)).collect();

        let mut builder_var = "msg".to_owned();
        for (i, level) in chain.iter().enumerate() {
            // Find the variant field in the parent struct for this level
            let (parent_disc_byte_off, variant_disc_value, variant_slot_offset) =
                find_chain_variant_info(i, &chain, root_struct, schema);

            // Set parent discriminant for this level's factory
            out.push_str(&format!(
                "  {builder_var}.setUint16({parent_disc_byte_off}, {variant_disc_value}); // {}\n",
                level.factory_name
            ));

            // Init the inner struct at the variant's pointer slot
            let inner_struct_name = find_inner_struct_name(i, &chain, root_struct, schema);
            let inner_struct = schema.structs.iter().find(|s| s.name == inner_struct_name);

            if let Some(is) = inner_struct {
                let var_name = format!("_s{i}");
                out.push_str(&format!(
                    "  const {var_name} = {builder_var}.initStruct({variant_slot_offset}, {}, {});\n",
                    is.data_words, is.pointer_words
                ));

                // Set scope fields on this level's struct
                for sf in &level.scope_fields {
                    emit_field_setter(out, &var_name, sf, &to_camel_case(&sf.name), "  ", &schema.enums, &schema.structs, &mut counter);
                }

                builder_var = var_name;
            }
        }

        // Now set the method discriminant on the innermost struct
        let _innermost_sc = chain.last().expect("chain must be non-empty");
        let inner_struct_name =
            find_inner_struct_name(chain.len() - 1, &chain, root_struct, schema);
        let inner_struct = schema.structs.iter().find(|s| s.name == inner_struct_name);

        if let Some(is) = inner_struct {
            let inner_disc_byte_off = is.discriminant_offset * 2;
            let method_field = is
                .fields
                .iter()
                .find(|f| f.name == variant.name && f.discriminant_value != 0xFFFF);

            if let Some(mf) = method_field {
                out.push_str(&format!(
                    "  {builder_var}.setUint16({inner_disc_byte_off}, {}); // {}\n",
                    mf.discriminant_value, variant.name
                ));

                // Set method payload
                if is_void {
                    // Nothing
                } else if is_prim {
                    emit_field_setter(out, &builder_var, mf, "p", "  ", &schema.enums, &schema.structs, &mut counter);
                } else if let Some(ps) = payload_struct {
                    emit_struct_init(out, mf, ps, &builder_var, "p", "  ", schema, &mut counter);
                }
            }
        }

        out.push_str("  return msg.finish();\n");
        out.push_str("}\n\n");
    }

    // Recurse for nested scoped clients
    let mut new_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    new_ancestors.push(sc);
    for nc in &sc.nested_clients {
        generate_scoped_builders(
            out,
            service_name,
            schema,
            nc,
            &new_ancestors,
            root_struct,
            root_non_union_fields,
        );
    }
}

/// Find the discriminant byte offset, discriminant value, and slot offset
/// for the variant that leads to `chain[level]` from its parent.
fn find_chain_variant_info(
    level: usize,
    chain: &[&ScopedClient],
    root_struct: &StructDef,
    schema: &ParsedSchema,
) -> (u32, u16, u32) {
    if level == 0 {
        // Parent is the root request struct
        let disc_byte_off = root_struct.discriminant_offset * 2;
        let variant_field = root_struct
            .fields
            .iter()
            .find(|f| f.name == chain[0].factory_name && f.discriminant_value != 0xFFFF);
        let disc_value = variant_field.map(|f| f.discriminant_value).unwrap_or(0);
        let slot_offset = variant_field.map(|f| f.slot_offset).unwrap_or(0);
        (disc_byte_off, disc_value, slot_offset)
    } else {
        // Parent is the inner struct of chain[level-1]
        let parent_inner_name =
            find_inner_struct_name(level - 1, chain, root_struct, schema);
        let parent_struct = schema
            .structs
            .iter()
            .find(|s| s.name == parent_inner_name);
        if let Some(ps) = parent_struct {
            let disc_byte_off = ps.discriminant_offset * 2;
            let variant_field = ps
                .fields
                .iter()
                .find(|f| f.name == chain[level].factory_name && f.discriminant_value != 0xFFFF);
            let disc_value = variant_field.map(|f| f.discriminant_value).unwrap_or(0);
            let slot_offset = variant_field.map(|f| f.slot_offset).unwrap_or(0);
            (disc_byte_off, disc_value, slot_offset)
        } else {
            (0, 0, 0)
        }
    }
}

/// Find the inner struct type name for `chain[level]`.
///
/// The inner struct name comes from the variant's type_name in the parent struct.
fn find_inner_struct_name(
    level: usize,
    chain: &[&ScopedClient],
    root_struct: &StructDef,
    schema: &ParsedSchema,
) -> String {
    if level == 0 {
        // Look up the variant type in the root request struct
        let variant_field = root_struct
            .fields
            .iter()
            .find(|f| f.name == chain[0].factory_name && f.discriminant_value != 0xFFFF);
        variant_field
            .map(|f| f.type_name.clone())
            .unwrap_or_default()
    } else {
        // Look up the variant type in chain[level-1]'s inner struct
        let parent_inner_name =
            find_inner_struct_name(level - 1, chain, root_struct, schema);
        let parent_struct = schema
            .structs
            .iter()
            .find(|s| s.name == parent_inner_name);
        if let Some(ps) = parent_struct {
            let variant_field = ps
                .fields
                .iter()
                .find(|f| f.name == chain[level].factory_name && f.discriminant_value != 0xFFFF);
            variant_field
                .map(|f| f.type_name.clone())
                .unwrap_or_default()
        } else {
            String::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone struct builders
// ---------------------------------------------------------------------------

/// Generate standalone builder functions for all structs in a schema.
///
/// For union-bearing structs: one exported function per union variant:
/// `build{StructName}_{variant}(nonUnionFields..., payload?): Uint8Array`.
///
/// For non-union structs: one exported function:
/// `build{StructName}(p: StructName): Uint8Array`.
///
/// Skips Option* wrapper structs and service request/response structs.
pub fn generate_struct_builders(out: &mut String, schema: &ParsedSchema) {
    for sd in &schema.structs {
        if sd.option_inner_type().is_some() {
            continue;
        }
        // Skip service request/response structs (handled by generate_builders)
        if is_service_struct(sd, schema) {
            continue;
        }

        if !sd.has_union {
            generate_plain_struct_builder(out, sd, schema);
            continue;
        }

        let non_union_fields: Vec<&FieldDef> = sd
            .fields
            .iter()
            .filter(|f| f.discriminant_value == 0xFFFF)
            .collect();

        let union_fields: Vec<&FieldDef> = sd
            .fields
            .iter()
            .filter(|f| f.discriminant_value != 0xFFFF)
            .collect();

        let disc_byte_off = sd.discriminant_offset * 2;

        for variant in &union_fields {
            let is_void = variant.type_name == "Void";
            let is_prim = is_primitive(&variant.type_name) && !is_void;
            let payload_struct = if !is_void && !is_prim {
                schema.structs.iter().find(|s| s.name == variant.type_name)
            } else {
                None
            };

            let fn_name = format!("build{}_{}", sd.name, variant.name);

            // Parameters: non-union fields + optional payload
            let mut params = Vec::new();
            for f in &non_union_fields {
                params.push(format!(
                    "{}: {}",
                    to_camel_case(&f.name),
                    capnp_to_ts_type(&f.type_name)
                ));
            }
            if !is_void {
                if let Some(ps) = payload_struct {
                    params.push(format!("p: {}", ps.name));
                } else if is_prim {
                    params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
                }
            }

            out.push_str(&format!(
                "export function {fn_name}({}): Uint8Array {{\n",
                params.join(", ")
            ));
            let extra_words = std::cmp::max(8usize, sd.pointer_words as usize * 16);
            out.push_str(&format!(
                "  const msg = CapnpArena.intoWasm({}, {}, {extra_words});\n",
                sd.data_words, sd.pointer_words
            ));

            // Per-function gensym counter — unique IDs for list builder variables
            let mut counter = 0u32;

            // Set non-union fields
            for f in &non_union_fields {
                emit_field_setter(
                    out,
                    "msg",
                    f,
                    &to_camel_case(&f.name),
                    "  ",
                    &schema.enums,
                    &schema.structs,
                    &mut counter,
                );
            }

            // Set discriminant
            out.push_str(&format!(
                "  msg.setUint16({disc_byte_off}, {}); // {}\n",
                variant.discriminant_value, variant.name
            ));

            // Set variant payload
            if is_void {
                // Nothing
            } else if is_prim {
                emit_field_setter(
                    out,
                    "msg",
                    variant,
                    "p",
                    "  ",
                    &schema.enums,
                    &schema.structs,
                    &mut counter,
                );
            } else if let Some(ps) = payload_struct {
                emit_struct_init(out, variant, ps, "msg", "p", "  ", schema, &mut counter);
            }

            out.push_str("  return msg.finish();\n");
            out.push_str("}\n\n");
        }
    }
}

/// Generate a single builder function for a non-union struct.
///
/// `export function build{StructName}(p: StructName): Uint8Array`
fn generate_plain_struct_builder(out: &mut String, sd: &StructDef, schema: &ParsedSchema) {
    let fn_name = format!("build{}", sd.name);

    out.push_str(&format!(
        "export function {fn_name}(p: {}): Uint8Array {{\n",
        sd.name
    ));
    let extra_words = std::cmp::max(8usize, sd.pointer_words as usize * 16);
    out.push_str(&format!(
        "  const msg = CapnpArena.intoWasm({}, {}, {extra_words});\n",
        sd.data_words, sd.pointer_words
    ));

    let mut counter = 0u32;
    for f in &sd.fields {
        let raw_expr = format!("p.{}", to_camel_case(&f.name));
        // Optional fields carry `T | undefined`; coalesce to the capnp default so
        // TypeScript strict mode doesn't reject passing `T | undefined` to a setter.
        let value_expr = if f.optional {
            let default = if f.section == FieldSection::Data
                && !is_data_scalar(&f.type_name)
                && f.type_name != "Bool"
            {
                schema
                    .enums
                    .iter()
                    .find(|e| e.name == f.type_name)
                    .and_then(|e| e.variants.first())
                    .map(|(name, _)| format!("'{}'", to_camel_case(name)))
                    .unwrap_or_else(|| "0".into())
            } else {
                super::default_value_expr(&f.type_name).to_owned()
            };
            format!("{raw_expr} ?? {default}")
        } else {
            raw_expr
        };
        emit_field_setter(
            out,
            "msg",
            f,
            &value_expr,
            "  ",
            &schema.enums,
            &schema.structs,
            &mut counter,
        );
    }

    out.push_str("  return msg.finish();\n");
    out.push_str("}\n\n");
}

/// Check if a struct is the service request or response struct (handled by generate_builders).
fn is_service_struct(sd: &StructDef, schema: &ParsedSchema) -> bool {
    schema
        .request_struct
        .as_ref()
        .is_some_and(|rs| rs.name == sd.name)
        || schema
            .response_struct
            .as_ref()
            .is_some_and(|rs| rs.name == sd.name)
}

// ---------------------------------------------------------------------------
// Field emission helpers
// ---------------------------------------------------------------------------

/// Emit a setter call for a field on a builder variable.
///
/// `counter` is a per-function gensym counter. Each List(Struct) field claims the
/// next ID (`_sl{n}`, `_sv{n}`, `_si{n}`) and increments the counter, guaranteeing
/// unique variable names across all nesting levels within a single builder function.
fn emit_field_setter(
    out: &mut String,
    builder_var: &str,
    field: &FieldDef,
    value_expr: &str,
    indent: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    counter: &mut u32,
) {
    match field.section {
        FieldSection::Data => {
            let byte_off = data_byte_offset(field);
            if field.type_name == "Bool" {
                let bit = bool_bit_index(field);
                out.push_str(&format!(
                    "{indent}{builder_var}.setBool({byte_off}, {bit}, {value_expr});\n"
                ));
            } else if is_data_scalar(&field.type_name) {
                if let Some(method) = super::setter_method(&field.type_name) {
                    out.push_str(&format!(
                        "{indent}{builder_var}.{method}({byte_off}, {value_expr});\n"
                    ));
                }
            } else {
                // Enum type — stored as UInt16, needs ordinal conversion
                let enum_def = enums.iter().find(|e| e.name == field.type_name);
                if let Some(ed) = enum_def {
                    let variants: Vec<String> = ed
                        .variants
                        .iter()
                        .map(|(name, _)| format!("'{}'", to_camel_case(name)))
                        .collect();
                    out.push_str(&format!(
                        "{indent}{builder_var}.setUint16({byte_off}, [{}].indexOf({value_expr}));\n",
                        variants.join(", ")
                    ));
                } else {
                    out.push_str(&format!(
                        "{indent}{builder_var}.setUint16({byte_off}, {value_expr} as unknown as number);\n"
                    ));
                }
            }
        }
        FieldSection::Pointer => {
            if let Some(method) = super::setter_method(&field.type_name) {
                out.push_str(&format!(
                    "{indent}{builder_var}.{method}({}, {value_expr});\n",
                    field.slot_offset
                ));
            } else if let Some(inner) = super::extract_list_inner_type(&field.type_name) {
                // List(Struct) — init struct list and set each element's fields.
                // Claim a unique ID from the per-function counter so nested lists at
                // the same capnp pointer slot never shadow each other (TDZ collision).
                if let Some(sd) = structs.iter().find(|s| s.name == inner) {
                    let id = *counter;
                    *counter += 1;
                    let sl = format!("_sl{id}");
                    let sv = format!("_sv{id}");
                    let si = format!("_si{id}");
                    out.push_str(&format!("{indent}{{\n"));
                    out.push_str(&format!(
                        "{indent}  const {sl} = {builder_var}.initStructList({}, {value_expr}.length, {}, {});\n",
                        field.slot_offset, sd.data_words, sd.pointer_words
                    ));
                    out.push_str(&format!(
                        "{indent}  {value_expr}.forEach(({sv}, {si}) => {{\n"
                    ));
                    // Set non-union fields
                    for sf in &sd.fields {
                        if sf.discriminant_value != 0xFFFF {
                            continue;
                        }
                        let inner_builder = format!("{sl}[{si}]");
                        let inner_value = format!("{sv}.{}", to_camel_case(&sf.name));
                        emit_field_setter(
                            out,
                            &inner_builder,
                            sf,
                            &inner_value,
                            &format!("{indent}    "),
                            enums,
                            structs,
                            counter,
                        );
                    }
                    // Handle union variants in list elements
                    if sd.has_union {
                        let inner_builder = format!("{sl}[{si}]");
                        let inner_indent = format!("{indent}    ");
                        emit_union_element_setter(
                            out,
                            sd,
                            &inner_builder,
                            &sv,
                            &inner_indent,
                            enums,
                            structs,
                            counter,
                        );
                    }
                    out.push_str(&format!("{indent}  }});\n"));
                    out.push_str(&format!("{indent}}}\n"));
                } else {
                    out.push_str(&format!(
                        "{indent}// {}: {} — struct definition not found\n",
                        to_camel_case(&field.name),
                        field.type_name
                    ));
                }
            } else if let Some(sd) = structs.iter().find(|s| s.name == field.type_name) {
                if let Some(inner_name) = sd.option_inner_type() {
                    emit_option_setter(out, field, sd, inner_name, builder_var, value_expr, indent);
                } else {
                    // Non-option struct pointer — init and set fields.
                    // Use gensym counter for the local variable to avoid TDZ collisions
                    // when the same slot_offset appears in nested or sibling struct pointers.
                    let sp_id = *counter;
                    *counter += 1;
                    let sp_var = format!("_sp{sp_id}");
                    out.push_str(&format!(
                        "{indent}if ({value_expr} != null) {{\n"
                    ));
                    out.push_str(&format!(
                        "{indent}  const {sp_var} = {builder_var}.initStruct({}, {}, {});\n",
                        field.slot_offset, sd.data_words, sd.pointer_words
                    ));
                    for sf in &sd.fields {
                        if sf.discriminant_value != 0xFFFF {
                            continue;
                        }
                        let val_expr = format!("{value_expr}.{}", to_camel_case(&sf.name));
                        emit_field_setter(
                            out,
                            &sp_var,
                            sf,
                            &val_expr,
                            &format!("{indent}  "),
                            enums,
                            structs,
                            counter,
                        );
                    }
                    out.push_str(&format!("{indent}}}\n"));
                }
            } else {
                // Unsupported pointer type — skip with comment
                out.push_str(&format!(
                    "{indent}// {}: {} — not yet supported by runtime\n",
                    to_camel_case(&field.name),
                    field.type_name
                ));
            }
        }
        FieldSection::Group => {} // groups are inlined, not handled here
    }
}

/// Emit code to init a sub-struct and set its non-union fields.
///
/// For optional fields (field.optional = true), adds `?? defaultValue` coalescing
/// so TypeScript strict mode doesn't reject `T | undefined` where `T` is expected.
fn emit_struct_init(
    out: &mut String,
    ptr_field: &FieldDef,
    struct_def: &StructDef,
    builder_var: &str,
    param_name: &str,
    indent: &str,
    schema: &ParsedSchema,
    counter: &mut u32,
) {
    out.push_str(&format!(
        "{indent}const _ps = {builder_var}.initStruct({}, {}, {});\n",
        ptr_field.slot_offset, struct_def.data_words, struct_def.pointer_words
    ));

    for f in &struct_def.fields {
        if f.discriminant_value != 0xFFFF {
            continue;
        }
        let raw_expr = format!("{param_name}.{}", to_camel_case(&f.name));
        let value_expr = if f.optional {
            // For enum types (Data section, non-scalar), default to the first variant string
            let default = if f.section == FieldSection::Data
                && !is_data_scalar(&f.type_name)
                && f.type_name != "Bool"
            {
                // Enum type — default to first variant name or '0' if not found
                schema
                    .enums
                    .iter()
                    .find(|e| e.name == f.type_name)
                    .and_then(|e| e.variants.first())
                    .map(|(name, _)| format!("'{}'", to_camel_case(name)))
                    .unwrap_or_else(|| "0".into())
            } else {
                super::default_value_expr(&f.type_name).to_owned()
            };
            format!("{raw_expr} ?? {default}")
        } else {
            raw_expr
        };
        emit_field_setter(out, "_ps", f, &value_expr, indent, &schema.enums, &schema.structs, counter);
    }
}

/// Emit code to write an Option* wrapper struct at a pointer slot.
///
/// Generates: if value !== undefined, init the option struct and set the discriminant + inner value.
/// Else: leave the pointer null (cap'n proto default = none).
fn emit_option_setter(
    out: &mut String,
    field: &FieldDef,
    opt_struct: &StructDef,
    inner_type_name: &str,
    builder_var: &str,
    value_expr: &str,
    indent: &str,
) {
    let disc_byte_off = opt_struct.discriminant_offset * 2;

    out.push_str(&format!("{indent}if ({value_expr} !== undefined && {value_expr} !== null) {{\n"));
    out.push_str(&format!(
        "{indent}  const _opt = {builder_var}.initStruct({}, {}, {});\n",
        field.slot_offset, opt_struct.data_words, opt_struct.pointer_words
    ));
    out.push_str(&format!("{indent}  _opt.setUint16({disc_byte_off}, 1);\n")); // some discriminant = 1

    // For pointer-type inners (Text, Data), the 'some' field lives in the pointer section at index 0.
    // For data-type scalars, use the slot_offset * type_size as byte offset.
    if inner_type_name == "Text" {
        out.push_str(&format!("{indent}  _opt.setText(0, {value_expr});\n"));
    } else if inner_type_name == "Data" {
        out.push_str(&format!("{indent}  _opt.setData(0, {value_expr});\n"));
    } else if let Some(method) = super::setter_method(inner_type_name) {
        // Find the 'some' field slot offset for computing the byte offset
        let some_field = opt_struct.fields.iter().find(|f| f.name == "some");
        if let Some(sf) = some_field {
            let some_byte_off = super::data_byte_offset(sf);
            out.push_str(&format!("{indent}  _opt.{method}({some_byte_off}, {value_expr});\n"));
        }
    }
    // else: unsupported inner type (nested struct, etc.) — leave as-is with discriminant set

    out.push_str(&format!("{indent}}}\n"));
    out.push_str(&format!("{indent}// else: null pointer = none (cap'n proto default)\n"));
}

/// Emit a switch statement that writes union discriminant + variant payload for a list element.
///
/// Used when a List(Struct) field contains a union struct (e.g., `List(StreamPayload)`).
/// The element parameter is `{ variant: string; data: unknown }`.
fn emit_union_element_setter(
    out: &mut String,
    sd: &StructDef,
    builder_var: &str,
    value_var: &str,
    indent: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    counter: &mut u32,
) {
    let disc_byte_off = sd.discriminant_offset * 2;
    let union_fields: Vec<&FieldDef> = sd
        .fields
        .iter()
        .filter(|f| f.discriminant_value != 0xFFFF)
        .collect();

    out.push_str(&format!("{indent}switch ({value_var}.variant) {{\n"));
    for uf in &union_fields {
        let ci = format!("{indent}  "); // case indent
        let bi = format!("{indent}    "); // body indent
        out.push_str(&format!("{ci}case '{}':\n", uf.name));
        out.push_str(&format!(
            "{bi}{builder_var}.setUint16({disc_byte_off}, {}); // {}\n",
            uf.discriminant_value, uf.name
        ));

        if uf.type_name == "Void" {
            // Nothing to set
        } else if uf.section == FieldSection::Data && is_data_scalar(&uf.type_name) {
            let byte_off = data_byte_offset(uf);
            if let Some(method) = super::setter_method(&uf.type_name) {
                out.push_str(&format!(
                    "{bi}{builder_var}.{method}({byte_off}, {value_var}.data as {});\n",
                    capnp_to_ts_type(&uf.type_name)
                ));
            }
        } else if uf.type_name == "Text" {
            out.push_str(&format!(
                "{bi}{builder_var}.setText({}, {value_var}.data as string);\n",
                uf.slot_offset
            ));
        } else if uf.type_name == "Data" {
            out.push_str(&format!(
                "{bi}{builder_var}.setData({}, {value_var}.data as Uint8Array);\n",
                uf.slot_offset
            ));
        } else if let Some(ps) = structs.iter().find(|s| s.name == uf.type_name) {
            // Struct variant — init sub-struct and set its non-union fields
            out.push_str(&format!("{bi}{{\n"));
            out.push_str(&format!(
                "{bi}  const _ups = {builder_var}.initStruct({}, {}, {});\n",
                uf.slot_offset, ps.data_words, ps.pointer_words
            ));
            out.push_str(&format!(
                "{bi}  const _upd = {value_var}.data as {};\n",
                ps.name
            ));
            for sf in &ps.fields {
                if sf.discriminant_value != 0xFFFF {
                    continue;
                }
                let val_expr = format!("_upd.{}", to_camel_case(&sf.name));
                emit_field_setter(out, "_ups", sf, &val_expr, &format!("{bi}  "), enums, structs, counter);
            }
            out.push_str(&format!("{bi}}}\n"));
        }

        out.push_str(&format!("{bi}break;\n"));
    }
    out.push_str(&format!("{indent}  default: break;\n"));
    out.push_str(&format!("{indent}}}\n"));
}
