//! Core serialization traits for Cap'n Proto.

use anyhow::Result;

/// Trait for serializing Rust types to Cap'n Proto.
///
/// Types implementing this trait can be written to a Cap'n Proto builder.
/// Use `#[derive(ToCapnp)]` to generate implementations automatically.
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::prelude::*;
///
/// #[derive(ToCapnp)]
/// #[capnp(my_schema_capnp::my_request)]
/// pub struct MyRequest {
///     pub name: String,
///     pub count: u32,
/// }
/// ```
pub trait ToCapnp {
    /// The Cap'n Proto builder type for this struct.
    type Builder<'a>;

    /// Write this struct's fields to the builder.
    fn write_to(&self, builder: &mut Self::Builder<'_>);
}

/// Trait for deserializing Cap'n Proto to Rust types.
///
/// Types implementing this trait can be read from a Cap'n Proto reader.
/// Use `#[derive(FromCapnp)]` to generate implementations automatically.
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::prelude::*;
///
/// #[derive(FromCapnp)]
/// #[capnp(my_schema_capnp::my_response)]
/// pub struct MyResponse {
///     pub result: String,
///     pub success: bool,
/// }
/// ```
pub trait FromCapnp: Sized {
    /// The Cap'n Proto reader type for this struct.
    type Reader<'a>;

    /// Read this struct's fields from the reader.
    fn read_from(reader: Self::Reader<'_>) -> Result<Self>;
}
