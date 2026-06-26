//! Service documentation helper for VFS mount integration.
//!
//! `DocFs` wraps a generated `render_doc()` function and provides
//! VFS-compatible helpers for serving documentation at `/doc/` paths.

use crate::metadata::MethodMeta;

/// Helper for serving service documentation through VFS paths.
///
/// Services delegate `["doc", ...]` walk/read/readdir operations to this helper.
/// Documentation is generated at compile time by the proc macro from Cap'n Proto
/// schema annotations (`$mcpDescription`, `$paramDescription`, `$docExample`).
pub struct DocFs {
    /// The generated `render_doc` function from the proc macro.
    render: fn(&[&str]) -> Option<String>,
    /// The generated `schema_metadata` function for directory listing.
    metadata: fn() -> (&'static str, &'static [MethodMeta]),
}

impl DocFs {
    /// Create a new DocFs from generated functions.
    pub fn new(
        render: fn(&[&str]) -> Option<String>,
        metadata: fn() -> (&'static str, &'static [MethodMeta]),
    ) -> Self {
        Self { render, metadata }
    }

    /// Check if a doc path is valid.
    pub fn exists(&self, path: &[&str]) -> bool {
        (self.render)(path).is_some()
    }

    /// Read documentation content for a path.
    pub fn read(&self, path: &[&str]) -> Option<Vec<u8>> {
        (self.render)(path).map(String::into_bytes)
    }

    /// List available documentation entries at a path.
    ///
    /// At the root, lists all non-hidden methods from `schema_metadata()`.
    pub fn list_entries(&self, path: &[&str]) -> Vec<String> {
        match path {
            [] | [""] => {
                let (_, methods) = (self.metadata)();
                methods
                    .iter()
                    .filter(|m| !m.hidden)
                    .map(|m| m.name.to_owned())
                    .collect()
            }
            _ => {
                // For scoped paths, we could enumerate but for now
                // just return empty — the render function handles validation.
                Vec::new()
            }
        }
    }
}
