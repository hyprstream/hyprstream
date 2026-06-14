//! Shared Cap'n Proto build helpers for annotation extraction and TypeScript codegen.
//!
//! This crate provides:
//! - Schema types (`ParsedSchema`, `StructDef`, `FieldDef`, etc.) shared between
//!   the proc-macro derive crate and the TypeScript codegen binary
//! - CGR (CodeGeneratorRequest) parsing with full wire format info
//! - Metadata JSON extraction for proc-macro annotation merging
//! - TypeScript codegen from parsed schemas (via the `hyprstream-ts-codegen` binary)

#![allow(clippy::print_stdout)] // cargo:warning= directives require println!

pub mod backend;
pub mod schema;
pub mod util;

// Generated `StreamContract` (#213/#216) types, used to decode the struct-valued
// `$streamPolicy` annotation value (see `build.rs`). Crate-private; the generated
// `streaming_capnp` references its import as `crate::annotations_capnp`, so both modules
// live at the crate root.
mod annotations_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/annotations_capnp.rs"));
}
mod streaming_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/streaming_capnp.rs"));
}

use std::path::Path;

/// Compile Cap'n Proto schemas with CGR + metadata extraction.
///
/// For each schema name, compiles `{schema_dir}/{name}.capnp` via capnpc,
/// saves the raw CGR to `{out_dir}/{name}.cgr`, and extracts annotation
/// metadata to `{out_dir}/{name}_metadata.json`.
///
/// `import_paths` are passed to capnpc for resolving `using import` directives.
pub fn compile_schemas(schema_dir: &Path, out_dir: &Path, import_paths: &[&Path], names: &[&str]) {
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
            // Struct-valued annotations need the generated typed reader (#216) — capnp has
            // no CGR-runtime dynamic decode. `$streamPolicy` (:StreamContract) is the one
            // such annotation; decode it structurally. Any other struct annotation, or a
            // decode failure, falls through to presence-only so we never silently corrupt.
            let val = if name == "streamPolicy" {
                match value.which() {
                    Ok(capnp::schema_capnp::value::Struct(ptr)) => match extract_stream_contract(ptr) {
                        Ok(v) => Some(v),
                        // Fail-closed: an invalid/contradictory $streamPolicy aborts the build
                        // rather than silently degrading (#213).
                        Err(e) => panic!("invalid $streamPolicy annotation: {e}"),
                    },
                    _ => extract_value_json(value),
                }
            } else {
                extract_value_json(value)
            };
            match val {
                Some(v) => result.push(serde_json::json!({
                    "id": ann_id,
                    "name": name,
                    "value": v,
                })),
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

/// Decode a `$streamPolicy` annotation value (a `StreamContract`, #213/#216) into
/// structured JSON for the codegen backends. Uses the generated typed reader — capnp-rust
/// cannot build a `StructSchema` from a raw CGR, so generic dynamic decode isn't possible
/// (see `build.rs`). Each axis is a union; group variants carry their own params.
fn extract_stream_contract(
    ptr: capnp::any_pointer::Reader,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use crate::streaming_capnp::{backpressure, completion, delivery, ordering, retention, stream_contract};

    let c: stream_contract::Reader = ptr.get_as()?;

    // Fail-closed validation (#213): reject contradictory axis combinations at build time.
    // A returned Err is treated as fatal by the caller (the build aborts) — an invalid
    // policy must never silently pass through to codegen.
    let lossy = matches!(c.get_backpressure()?.which()?, backpressure::DropOldest(_));
    let reliable = matches!(c.get_delivery()?.which()?, delivery::AtLeastOnce(_));
    if lossy && reliable {
        return Err("streamPolicy: backpressure=dropOldest contradicts delivery=atLeastOnce \
                    (a stream cannot be lossless and also drop under load)"
            .into());
    }
    let terminal = matches!(c.get_completion()?.which()?, completion::Terminal(()));
    let at_most_once = matches!(c.get_delivery()?.which()?, delivery::AtMostOnce(()));
    if terminal && at_most_once {
        return Err("streamPolicy: completion=terminal contradicts delivery=atMostOnce \
                    (truncation detection requires reliable delivery)"
            .into());
    }

    let ordering = match c.get_ordering()?.which()? {
        ordering::Ordered(()) => serde_json::json!({ "ordered": true }),
        ordering::Unordered(u) => serde_json::json!({
            "unordered": { "replayWindow": u.get_replay_window() }
        }),
    };
    let delivery = match c.get_delivery()?.which()? {
        delivery::AtMostOnce(()) => serde_json::json!({ "atMostOnce": true }),
        delivery::AtLeastOnce(a) => serde_json::json!({
            "atLeastOnce": { "dedupWindow": a.get_dedup_window(), "resumable": a.get_resumable() }
        }),
    };
    let completion = match c.get_completion()?.which()? {
        completion::Terminal(()) => "terminal",
        completion::None(()) => "none",
    };
    let retention = match c.get_retention()?.which()? {
        retention::LiveOnly(()) => serde_json::json!({ "liveOnly": true }),
        retention::Groups(n) => serde_json::json!({ "groups": n }),
        retention::Seconds(n) => serde_json::json!({ "seconds": n }),
    };
    let backpressure = match c.get_backpressure()?.which()? {
        backpressure::Block(()) => serde_json::json!({ "block": true }),
        backpressure::DropOldest(d) => serde_json::json!({
            "dropOldest": { "highWater": d.get_high_water() }
        }),
    };

    Ok(serde_json::json!({
        "ordering": ordering,
        "delivery": delivery,
        "completion": completion,
        "retention": retention,
        "backpressure": backpressure,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::items_after_test_module)]
mod stream_contract_tests {
    use super::extract_stream_contract;
    use crate::streaming_capnp::stream_contract;

    /// Round-trips a `StreamContract` (ordered / atLeastOnce{256,true} / none / groups=256
    /// / block) through the typed decoder and asserts the JSON the codegen will consume —
    /// covering void variants, a group with params, and the scalar-bearing retention variant.
    #[test]
    fn stream_contract_decodes_to_json() {
        let mut msg = capnp::message::Builder::new_default();
        {
            let mut c = msg.init_root::<stream_contract::Builder>();
            c.reborrow().init_ordering().set_ordered(());
            {
                let mut a = c.reborrow().init_delivery().init_at_least_once();
                a.set_dedup_window(256);
                a.set_resumable(true);
            }
            c.reborrow().init_completion().set_none(());
            c.reborrow().init_retention().set_groups(256);
            c.reborrow().init_backpressure().set_block(());
        }
        let root = msg
            .get_root_as_reader::<capnp::any_pointer::Reader>()
            .unwrap();
        let json = extract_stream_contract(root).unwrap();

        assert_eq!(json["ordering"]["ordered"], serde_json::json!(true));
        assert_eq!(json["delivery"]["atLeastOnce"]["dedupWindow"], serde_json::json!(256));
        assert_eq!(json["delivery"]["atLeastOnce"]["resumable"], serde_json::json!(true));
        assert_eq!(json["completion"], serde_json::json!("none"));
        assert_eq!(json["retention"]["groups"], serde_json::json!(256));
        assert_eq!(json["backpressure"]["block"], serde_json::json!(true));
    }

    /// A contradictory policy (atLeastOnce + dropOldest = lossless yet drops) must be
    /// rejected by the fail-closed validation, not silently decoded.
    #[test]
    fn contradictory_policy_is_rejected() {
        let mut msg = capnp::message::Builder::new_default();
        {
            let mut c = msg.init_root::<stream_contract::Builder>();
            c.reborrow().init_ordering().set_ordered(());
            {
                let mut a = c.reborrow().init_delivery().init_at_least_once();
                a.set_dedup_window(64);
                a.set_resumable(true);
            }
            c.reborrow().init_completion().set_none(());
            c.reborrow().init_retention().set_groups(64);
            c.reborrow().init_backpressure().init_drop_oldest().set_high_water(1024);
        }
        let root = msg
            .get_root_as_reader::<capnp::any_pointer::Reader>()
            .unwrap();
        let err = extract_stream_contract(root).unwrap_err().to_string();
        assert!(err.contains("dropOldest") && err.contains("atLeastOnce"), "got: {err}");
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
