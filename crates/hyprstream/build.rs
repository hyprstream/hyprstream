//! Build script for Hyprstream

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    // Capture git info for version string
    capture_git_info();

    // Compile Cap'n Proto schemas
    compile_capnp_schemas();

    // If using Python PyTorch or download-libtorch, tch-rs handles libtorch setup
    if env::var("LIBTORCH_USE_PYTORCH").is_ok() || env::var("LIBTORCH").is_err() {
        // tch-rs will handle libtorch setup
        return;
    }

    let libtorch_path = match env::var("LIBTORCH") {
        Ok(path) => path,
        Err(_) => return, // Early return, should not happen due to check above
    };

    // Validate libtorch exists
    let libtorch_dir = Path::new(&libtorch_path);
    if !libtorch_dir.exists() {
        panic!("libtorch directory not found at {libtorch_path}");
    }

    // Configure linking
    println!("cargo:rustc-link-search=native={libtorch_path}/lib");
    println!("cargo:rustc-env=LIBTORCH_STATIC=0");
    println!("cargo:rustc-env=LIBTORCH_BYPASS_VERSION_CHECK=1");
}

fn compile_capnp_schemas() {
    let schema_dir = Path::new("schema");
    let out_dir = env::var("OUT_DIR").unwrap();

    // Skip if schema directory doesn't exist
    if !schema_dir.exists() {
        return;
    }

    // Note: common.capnp (identity, envelope) is in hyprstream-rpc crate
    for name in ["events", "inference", "registry", "policy", "model", "mcp"] {
        let path = schema_dir.join(format!("{name}.capnp"));
        if path.exists() {
            let cgr_path = Path::new(&out_dir).join(format!("{name}.cgr"));
            let metadata_path = Path::new(&out_dir).join(format!("{name}_metadata.json"));

            // 1. Compile to Rust AND save raw CodeGeneratorRequest
            capnpc::CompilerCommand::new()
                .src_prefix("schema")
                .file(&path)
                .raw_code_generator_request_path(&cgr_path)  // ← Save CGR!
                .run()
                .unwrap_or_else(|e| panic!("failed to compile {name}.capnp: {e}"));

            // 2. Parse CGR and extract schema metadata with annotations
            if let Err(e) = parse_schema_and_extract_annotations(&cgr_path, &metadata_path, name) {
                eprintln!("Warning: Failed to parse schema for {name}: {e}");
                eprintln!("Falling back to text parsing (annotations not available)");
            }
        }
    }
}

/// Parse CodeGeneratorRequest and extract annotations using capnp crate.
fn parse_schema_and_extract_annotations(
    cgr_path: &Path,
    output_path: &Path,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use capnp::message::ReaderOptions;
    use capnp::serialize;
    use capnp::schema_capnp::{node, field};

    // Read the CodeGeneratorRequest binary
    let file = std::fs::File::open(cgr_path)?;
    let reader = serialize::read_message(file, ReaderOptions::new())?;

    // Parse as CodeGeneratorRequest (defined in schema.capnp)
    let cgr = reader.get_root::<capnp::schema_capnp::code_generator_request::Reader>()?;

    // Extract all nodes (structs, enums, etc.)
    let nodes = cgr.get_nodes()?;

    let mut structs = Vec::new();
    let mut enums = Vec::new();
    let mut annotations_map = serde_json::Map::new();

    // Iterate through nodes and extract structure + annotations
    for i in 0..nodes.len() {
        let node = nodes.get(i);
        let node_id = node.get_id();
        let display_name = node.get_display_name()?.to_str()?.to_string();

        // Extract node-level annotations
        let node_annotations = extract_annotations(node.get_annotations()?)?;
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
                    let field_name = field.get_name()?.to_str()?.to_string();
                    let discriminant = field.get_discriminant_value();

                    // Extract field-level annotations
                    let field_annotations = extract_annotations(field.get_annotations()?)?;

                    // Get field type name
                    let type_name = match field.which()? {
                        field::Slot(slot) => {
                            extract_type_name(slot.get_type()?)
                        }
                        field::Group(_) => "Group".to_string(),
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
                    let variant_name = enumerant.get_name()?.to_str()?.to_string();
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
    println!("cargo:warning=Extracted schema metadata for {name} to {}", output_path.display());

    Ok(())
}

/// Extract annotations from a list and return as JSON array.
fn extract_annotations(
    annotations: capnp::struct_list::Reader<capnp::schema_capnp::annotation::Owned>
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();

    for i in 0..annotations.len() {
        let ann = annotations.get(i);
        let ann_id = ann.get_id();

        // Try to extract annotation value (typically Text for our annotations)
        if let Ok(value) = ann.get_value() {
            if let Ok(text_value) = extract_text_from_value(value) {
                result.push(serde_json::json!({
                    "id": ann_id,
                    "value": text_value,
                }));
            }
        }
    }

    Ok(result)
}

/// Extract text value from a Cap'n Proto Value (if it's a Text type).
fn extract_text_from_value(
    value: capnp::schema_capnp::value::Reader
) -> Result<String, Box<dyn std::error::Error>> {
    use capnp::schema_capnp::value;

    match value.which()? {
        value::Text(text_reader) => Ok(text_reader?.to_str()?.to_string()),
        _ => Err("Not a text value".into()),
    }
}

/// Extract a human-readable type name from a Type.
fn extract_type_name(
    type_reader: capnp::schema_capnp::type_::Reader
) -> String {
    use capnp::schema_capnp::type_;

    match type_reader.which() {
        Ok(type_::Void(())) => "Void".to_string(),
        Ok(type_::Bool(())) => "Bool".to_string(),
        Ok(type_::Int8(())) => "Int8".to_string(),
        Ok(type_::Int16(())) => "Int16".to_string(),
        Ok(type_::Int32(())) => "Int32".to_string(),
        Ok(type_::Int64(())) => "Int64".to_string(),
        Ok(type_::Uint8(())) => "UInt8".to_string(),
        Ok(type_::Uint16(())) => "UInt16".to_string(),
        Ok(type_::Uint32(())) => "UInt32".to_string(),
        Ok(type_::Uint64(())) => "UInt64".to_string(),
        Ok(type_::Float32(())) => "Float32".to_string(),
        Ok(type_::Float64(())) => "Float64".to_string(),
        Ok(type_::Text(())) => "Text".to_string(),
        Ok(type_::Data(())) => "Data".to_string(),
        Ok(type_::List(list_type)) => {
            if let Ok(element_type) = list_type.get_element_type() {
                format!("List({})", extract_type_name(element_type))
            } else {
                "List".to_string()
            }
        }
        Ok(type_::Enum(_)) => "Enum".to_string(),
        Ok(type_::Struct(_)) => "Struct".to_string(),
        Ok(type_::Interface(_)) => "Interface".to_string(),
        Ok(type_::AnyPointer(_)) => "AnyPointer".to_string(),
        Err(_) => "Unknown".to_string(),
    }
}

/// Capture git info and export as environment variables for the build
///
/// Exports:
/// - GIT_SHA: 7-char commit SHA (e.g., "abc1234")
/// - GIT_BRANCH: sanitized branch name (e.g., "main", "feature-auth")
/// - GIT_DIRTY: "true" or "false"
fn capture_git_info() {
    // Get commit SHA (short)
    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();

    // Get branch name
    let branch = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .map(|b| if b == "HEAD" { String::new() } else { b }) // Detached HEAD
        .unwrap_or_default();

    // Check if worktree is dirty
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    // Sanitize branch name for filesystem safety
    let sanitized_branch = sanitize_git_ref(&branch);

    // Export individual components as environment variables
    println!("cargo:rustc-env=GIT_SHA={}", sha);
    println!("cargo:rustc-env=GIT_BRANCH={}", sanitized_branch);
    println!("cargo:rustc-env=GIT_DIRTY={}", dirty);

    // Build complete version string
    let cargo_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_owned());
    let build_version = if sha.is_empty() {
        cargo_version
    } else {
        let mut v = format!("{}+", cargo_version);
        if !sanitized_branch.is_empty() {
            v.push_str(&sanitized_branch);
            v.push('.');
        }
        v.push('g');
        v.push_str(&sha);
        if dirty {
            v.push_str(".dirty");
        }
        v
    };
    println!("cargo:rustc-env=BUILD_VERSION={}", build_version);
}

/// Sanitize a git ref name for safe use in filesystem paths
///
/// - Lowercase
/// - Replace `/`, `.`, ` ` with `-`
/// - Remove other special characters
/// - Collapse multiple `-` into one
/// - Trim leading/trailing `-`
/// - Limit to 50 characters
fn sanitize_git_ref(ref_name: &str) -> String {
    let sanitized: String = ref_name
        .to_lowercase()
        .chars()
        .map(|c| match c {
            'a'..='z' | '0'..='9' | '-' | '_' => c,
            _ => '-',
        })
        .collect();

    // Collapse multiple dashes and trim
    let mut result = String::with_capacity(sanitized.len());
    let mut last_was_dash = true; // Start true to trim leading dashes
    for c in sanitized.chars().take(50) {
        if c == '-' {
            if !last_was_dash {
                result.push(c);
                last_was_dash = true;
            }
        } else {
            result.push(c);
            last_was_dash = false;
        }
    }

    // Trim trailing dash
    if result.ends_with('-') {
        result.pop();
    }

    result
}
