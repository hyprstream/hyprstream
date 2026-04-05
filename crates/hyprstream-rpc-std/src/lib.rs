//! Standard hyprstream service schemas and generated clients.
//!
//! This crate contains the Cap'n Proto service protocol definitions and
//! generated client types for all standard hyprstream services (model,
//! registry, inference, policy, mcp, etc.).
//!
//! MIT licensed. Depends only on hyprstream-rpc (also MIT).
//! Compiles to native and wasm32.

#![allow(dead_code, unused_imports)]

// ============================================================================
// Re-export shared capnp modules from hyprstream-rpc so that generated code
// using `crate::common_capnp`, `crate::streaming_capnp`, etc. resolves
// ============================================================================

pub use hyprstream_rpc::common_capnp;
pub use hyprstream_rpc::streaming_capnp;
pub use hyprstream_rpc::annotations_capnp;
pub use hyprstream_rpc::optional_capnp;
pub use hyprstream_rpc::events_capnp;
pub use hyprstream_rpc::nine_capnp;

// ============================================================================
// Cap'n Proto generated modules — service schemas
// ============================================================================

pub mod inference_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/inference_capnp.rs"));
}

pub mod model_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/model_capnp.rs"));
}

pub mod registry_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/registry_capnp.rs"));
}

pub mod policy_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/policy_capnp.rs"));
}

pub mod mcp_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/mcp_capnp.rs"));
}

pub mod metrics_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/metrics_capnp.rs"));
}

pub mod notification_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/notification_capnp.rs"));
}

pub mod service_events_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/service_events_capnp.rs"));
}

pub mod chat_core_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/chat_core_capnp.rs"));
}

// ============================================================================
// Generated service clients (from proc macro)
// These include server-side handler traits + ZMQ transport code that
// requires native deps. Gate to non-wasm32 for now.
// TODO: proc macro flag for client-only generation (no server handler code)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub mod inference_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("inference");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod model_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("model");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod registry_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("registry");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod policy_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("policy");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod mcp_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("mcp");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod notification_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("notification");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod metrics_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_service!("metrics");
}

// ============================================================================
// WASM exports (browser only) — TODO after clients work
// ============================================================================

// #[cfg(target_arch = "wasm32")]
// pub mod wasm_exports;
