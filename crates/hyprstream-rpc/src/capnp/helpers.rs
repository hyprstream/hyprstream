//! Helper functions for Cap'n Proto serialization.

use anyhow::Result;
use capnp::message::Builder;
use capnp::serialize;

/// Serialize a Cap'n Proto message to bytes using a closure.
///
/// This wraps the common pattern of creating a message builder,
/// setting up the root, and serializing to a Vec<u8>.
///
/// # Example
///
/// ```ignore
/// let bytes = serialize_message(|builder| {
///     let mut req = builder.init_root::<my_capnp::request::Builder>();
///     req.set_id(42);
///     request.write_to(&mut req);
/// })?;
/// ```
pub fn serialize_message<F>(setup: F) -> Result<Vec<u8>>
where
    F: FnOnce(&mut Builder<capnp::message::HeapAllocator>),
{
    let mut message = Builder::new_default();
    setup(&mut message);

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;
    Ok(bytes)
}

/// Build a standard error response.
///
/// Many RPC services share a common error format. This helper
/// creates a consistent error response structure.
pub fn build_error_response(request_id: u64, message: &str, code: &str) -> Vec<u8> {
    // This is a generic helper - actual implementation depends on schema
    // Services will use their own error builders but can use this pattern
    let mut msg = Builder::new_default();

    // Generic error structure - services override with their specific schema
    let _ = (request_id, message, code, &mut msg);

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &msg).unwrap_or_default();
    bytes
}
