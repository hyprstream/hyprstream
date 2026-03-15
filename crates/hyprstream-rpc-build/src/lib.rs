//! Shared Cap'n Proto build helpers for annotation extraction and TypeScript codegen.
//!
//! This crate provides:
//! - Schema types (`ParsedSchema`, `StructDef`, `FieldDef`, etc.) shared between
//!   the proc-macro derive crate and the TypeScript codegen binary
//! - CGR (CodeGeneratorRequest) parsing with full wire format info
//! - Metadata JSON extraction for proc-macro annotation merging
//! - TypeScript codegen from parsed schemas (via the `hyprstream-ts-codegen` binary)

#![allow(clippy::print_stdout)] // cargo:warning= directives require println!

pub mod schema;
pub mod util;

use std::path::Path;

/// Compile Cap'n Proto schemas with CGR + metadata extraction.
///
/// For each schema name, compiles `{schema_dir}/{name}.capnp` via capnpc,
/// saves the raw CGR to `{out_dir}/{name}.cgr`, and extracts annotation
/// metadata to `{out_dir}/{name}_metadata.json`.
///
/// `import_paths` are passed to capnpc for resolving `using import` directives.
pub fn compile_schemas(
    schema_dir: &Path,
    out_dir: &Path,
    import_paths: &[&Path],
    names: &[&str],
) {
    for name in names {
        let path = schema_dir.join(format!("{name}.capnp"));
        if !path.exists() {
            continue;
        }

        let cgr_path = out_dir.join(format!("{name}.cgr"));
        let metadata_path = out_dir.join(format!("{name}_metadata.json"));

        let mut cmd = capnpc::CompilerCommand::new();
        cmd.src_prefix(schema_dir);
        for ip in import_paths {
            cmd.import_path(ip);
        }
        cmd.file(&path)
            .raw_code_generator_request_path(&cgr_path)
            .run()
            .unwrap_or_else(|e| panic!("Failed to compile {name}.capnp: {e}"));

        if let Err(e) = parse_schema_and_extract_annotations(&cgr_path, &metadata_path, name) {
            println!("cargo:warning=Failed to parse schema for {name}: {e}");
        }
    }
}

/// Parse a CodeGeneratorRequest CGR file and extract schema metadata with annotations.
///
/// Writes a JSON file at `output_path` containing structs, enums, and annotations
/// extracted from the compiled Cap'n Proto schema.
pub fn parse_schema_and_extract_annotations(
    cgr_path: &Path,
    output_path: &Path,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use capnp::message::ReaderOptions;
    use capnp::schema_capnp::{field, node};
    use capnp::serialize;

    // Read the CodeGeneratorRequest binary
    let file = std::fs::File::open(cgr_path)?;
    let reader = serialize::read_message(file, ReaderOptions::new())?;

    // Parse as CodeGeneratorRequest (defined in schema.capnp)
    let cgr = reader.get_root::<capnp::schema_capnp::code_generator_request::Reader>()?;

    // Extract all nodes (structs, enums, etc.)
    let nodes = cgr.get_nodes()?;

    // Build annotation name map: annotation node ID -> short name
    let mut annotation_names: std::collections::HashMap<u64, String> =
        std::collections::HashMap::new();
    for i in 0..nodes.len() {
        let node = nodes.get(i);
        if let Ok(capnp::schema_capnp::node::Annotation(_)) = node.which() {
            let dn = node.get_display_name()?.to_str()?.to_owned();
            let short = dn.rsplit(':').next().unwrap_or(&dn).to_owned();
            annotation_names.insert(node.get_id(), short);
        }
    }

    let mut structs = Vec::new();
    let mut enums = Vec::new();
    let mut annotations_map = serde_json::Map::new();

    // Iterate through nodes and extract structure + annotations
    for i in 0..nodes.len() {
        let node = nodes.get(i);
        let node_id = node.get_id();
        let display_name = node.get_display_name()?.to_str()?.to_owned();

        // Extract node-level annotations
        let node_annotations = extract_annotations(node.get_annotations()?, &annotation_names)?;
        if !node_annotations.is_empty() {
            annotations_map.insert(display_name.clone(), serde_json::json!(node_annotations));
        }

        // Process based on node type
        match node.which()? {
            node::Struct(struct_reader) => {
                let mut fields = Vec::new();
                let struct_fields = struct_reader.get_fields()?;

                for j in 0..struct_fields.len() {
                    let field = struct_fields.get(j);
                    let field_name = field.get_name()?.to_str()?.to_owned();
                    let discriminant = field.get_discriminant_value();

                    // Extract field-level annotations
                    let field_annotations =
                        extract_annotations(field.get_annotations()?, &annotation_names)?;

                    // Get field type name
                    let type_name = match field.which()? {
                        field::Slot(slot) => extract_type_name(slot.get_type()?),
                        field::Group(_) => "Group".to_owned(),
                    };

                    fields.push(serde_json::json!({
                        "name": field_name,
                        "type": type_name,
                        "discriminant": discriminant,
                        "annotations": field_annotations,
                    }));
                }

                structs.push(serde_json::json!({
                    "name": display_name,
                    "id": node_id,
                    "fields": fields,
                }));
            }
            node::Enum(enum_reader) => {
                let mut variants = Vec::new();
                let enumerants = enum_reader.get_enumerants()?;

                for j in 0..enumerants.len() {
                    let enumerant = enumerants.get(j);
                    let variant_name = enumerant.get_name()?.to_str()?.to_owned();
                    let code_order = enumerant.get_code_order();

                    variants.push(serde_json::json!({
                        "name": variant_name,
                        "value": code_order,
                    }));
                }

                enums.push(serde_json::json!({
                    "name": display_name,
                    "id": node_id,
                    "variants": variants,
                }));
            }
            _ => {
                // Ignore interfaces, consts, etc. for now
            }
        }
    }

    let metadata = serde_json::json!({
        "service": name,
        "structs": structs,
        "enums": enums,
        "annotations": annotations_map,
    });

    // Write metadata JSON
    std::fs::write(output_path, serde_json::to_string_pretty(&metadata)?)?;

    Ok(())
}

/// Extract annotations from a list and return as JSON array.
///
/// Each annotation includes its `id`, resolved `name`, and `value`.
/// Supports Text, Void, Bool, and Enum annotation values.
fn extract_annotations(
    annotations: capnp::struct_list::Reader<capnp::schema_capnp::annotation::Owned>,
    annotation_names: &std::collections::HashMap<u64, String>,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();

    for i in 0..annotations.len() {
        let ann = annotations.get(i);
        let ann_id = ann.get_id();
        let name = annotation_names.get(&ann_id).cloned().unwrap_or_default();

        if let Ok(value) = ann.get_value() {
            match extract_value_json(value) {
                Some(val) => {
                    result.push(serde_json::json!({
                        "id": ann_id,
                        "name": name,
                        "value": val,
                    }));
                }
                None => {
                    // Void annotations (e.g., $cliHidden) -- presence is the value
                    result.push(serde_json::json!({
                        "id": ann_id,
                        "name": name,
                        "value": true,
                    }));
                }
            }
        }
    }

    Ok(result)
}

/// Extract a JSON-compatible value from a Cap'n Proto Value.
///
/// Returns Some for Text, Bool, Enum(ordinal), None for Void/unsupported.
fn extract_value_json(value: capnp::schema_capnp::value::Reader) -> Option<serde_json::Value> {
    use capnp::schema_capnp::value;

    match value.which().ok()? {
        value::Text(text_reader) => Some(serde_json::Value::String(
            text_reader.ok()?.to_str().ok()?.to_owned(),
        )),
        value::Bool(b) => Some(serde_json::Value::Bool(b)),
        value::Uint32(n) => Some(serde_json::Value::Number(n.into())),
        value::Enum(ordinal) => Some(serde_json::json!({"enum_ordinal": ordinal})),
        _ => None,
    }
}

/// Extract a human-readable type name from a Type.
fn extract_type_name(type_reader: capnp::schema_capnp::type_::Reader) -> String {
    use capnp::schema_capnp::type_;

    match type_reader.which() {
        Ok(type_::Void(())) => "Void".to_owned(),
        Ok(type_::Bool(())) => "Bool".to_owned(),
        Ok(type_::Int8(())) => "Int8".to_owned(),
        Ok(type_::Int16(())) => "Int16".to_owned(),
        Ok(type_::Int32(())) => "Int32".to_owned(),
        Ok(type_::Int64(())) => "Int64".to_owned(),
        Ok(type_::Uint8(())) => "UInt8".to_owned(),
        Ok(type_::Uint16(())) => "UInt16".to_owned(),
        Ok(type_::Uint32(())) => "UInt32".to_owned(),
        Ok(type_::Uint64(())) => "UInt64".to_owned(),
        Ok(type_::Float32(())) => "Float32".to_owned(),
        Ok(type_::Float64(())) => "Float64".to_owned(),
        Ok(type_::Text(())) => "Text".to_owned(),
        Ok(type_::Data(())) => "Data".to_owned(),
        Ok(type_::List(list_type)) => {
            if let Ok(element_type) = list_type.get_element_type() {
                format!("List({})", extract_type_name(element_type))
            } else {
                "List".to_owned()
            }
        }
        Ok(type_::Enum(_)) => "Enum".to_owned(),
        Ok(type_::Struct(_)) => "Struct".to_owned(),
        Ok(type_::Interface(_)) => "Interface".to_owned(),
        Ok(type_::AnyPointer(_)) => "AnyPointer".to_owned(),
        Err(_) => "Unknown".to_owned(),
    }
}
