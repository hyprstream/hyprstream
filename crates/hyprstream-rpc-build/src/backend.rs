//! Language-agnostic code generation backend trait.
//!
//! Defines the `CodegenBackend` trait that each code generator backend must implement.
//! The TypeScript backend wraps the existing `ts_codegen` emitters; additional backends
//! (e.g., Python, Rust client stubs) can implement the trait independently.

use crate::schema::types::{FieldDef, ParsedSchema};

/// Backend trait for multi-language code generation from Cap'n Proto schemas.
///
/// Implementations map Cap'n Proto types and wire format to a target language's
/// type system, serialization API, and file layout.  The language-agnostic
/// `generate_all` driver calls these methods to produce output files.
pub trait CodegenBackend {
    /// Map a Cap'n Proto type name to the target language's type.
    ///
    /// E.g., `"Text"` → `"string"` for TypeScript, `"str"` for Python.
    fn map_type(&self, capnp_type: &str) -> String;

    /// Return the setter method name on a builder for this type, if the runtime
    /// supports a direct method call.  Returns `None` for types handled specially
    /// (struct pointers, enum ordinals, etc.).
    fn setter_method(&self, type_name: &str) -> Option<String>;

    /// Return an expression that reads a field's value from a reader variable.
    fn getter_expr(&self, reader_var: &str, field: &FieldDef) -> String;

    /// Return the default value expression for a type (used for optional field coalescing).
    fn default_value(&self, type_name: &str) -> String;

    /// Generate the complete source file content for a service schema.
    fn generate_service_file(&self, service_name: &str, schema: &ParsedSchema) -> String;

    /// Return `(filename, content)` for the shared runtime file included in all output.
    ///
    /// E.g., TypeScript returns `("capnp.ts", <wire-format runtime>)`.
    fn runtime_file(&self) -> (String, String);

    /// Return the file extension for generated service files (e.g., `"ts"`, `"py"`).
    fn file_extension(&self) -> &str;

    /// Return `(filename, content)` for an optional index/manifest file.
    ///
    /// Returns `None` if this backend does not generate an index file.
    /// TypeScript overrides this to produce `index.ts` with re-exports.
    fn index_file(
        &self,
        _all_names: &[String],
        _client_names: &[String],
    ) -> Option<(String, String)> {
        None
    }
}
