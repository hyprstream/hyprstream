//! Cap'n Proto serialization traits and helpers.

mod traits;
mod helpers;
mod error;

pub use traits::{FromCapnp, ToCapnp};
pub use helpers::{serialize_message, build_error_response};
pub use error::{CapnpResultExt, RpcError};
