//! Python code generation backend (skeleton).
//!
//! Provides `PythonBackend` which implements `CodegenBackend` with Python type
//! mappings.  A complete Python implementation is out of scope; this skeleton
//! validates that the `CodegenBackend` interface compiles correctly with a
//! non-TypeScript implementation.

#![allow(dead_code)]

use hyprstream_rpc_build::backend::CodegenBackend;
use hyprstream_rpc_build::schema::types::{FieldDef, ParsedSchema};

/// Python code generation backend (skeleton).
pub struct PythonBackend;

impl CodegenBackend for PythonBackend {
    fn map_type(&self, capnp_type: &str) -> String {
        match capnp_type {
            "Void" => "None".into(),
            "Bool" => "bool".into(),
            "UInt8" | "UInt16" | "UInt32" | "UInt64"
            | "Int8" | "Int16" | "Int32" | "Int64" => "int".into(),
            "Float32" | "Float64" => "float".into(),
            "Text" => "str".into(),
            "Data" => "bytes".into(),
            _ if capnp_type.starts_with("List(") => {
                let inner = &capnp_type[5..capnp_type.len() - 1];
                format!("list[{}]", self.map_type(inner))
            }
            _ => capnp_type.to_owned(),
        }
    }

    fn setter_method(&self, _type_name: &str) -> Option<String> {
        // Not yet implemented
        None
    }

    fn getter_expr(&self, reader_var: &str, field: &FieldDef) -> String {
        // Not yet implemented
        format!("{reader_var}.{}", field.name)
    }

    fn default_value(&self, capnp_type: &str) -> String {
        match capnp_type {
            "Void" => "None".into(),
            "Bool" => "False".into(),
            "Text" => "\"\"".into(),
            "Data" => "b\"\"".into(),
            _ if capnp_type.starts_with("List(") => "[]".into(),
            _ if capnp_type.contains("Int") || capnp_type.contains("Float") => "0".into(),
            _ => "None".into(),
        }
    }

    fn generate_service_file(&self, service_name: &str, _schema: &ParsedSchema) -> String {
        // Not yet implemented — placeholder comment
        format!("# Generated for {service_name} — Python codegen not yet implemented\n")
    }

    fn runtime_file(&self) -> (String, String) {
        // Not yet implemented
        (
            "capnp.py".to_owned(),
            "# Cap'n Proto Python runtime — not yet implemented\n".to_owned(),
        )
    }

    fn file_extension(&self) -> &str {
        "py"
    }
}
