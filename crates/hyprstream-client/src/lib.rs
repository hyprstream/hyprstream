//! Hyprstream RPC client — lightweight crate that compiles to wasm32.
//!
//! Uses the `generate_rpc_service!` proc macro to produce typed clients
//! for each service. On native, these use ZMQ transport. On wasm32, they
//! use WebTransport via web_sys.
//!
//! This crate deliberately avoids heavy deps (torch, zmq FFI, tokio) so
//! it can compile to wasm32-unknown-unknown for browser use.

// Re-export common types from hyprstream-rpc
pub use hyprstream_rpc::crypto::{SigningKey, VerifyingKey};
pub use hyprstream_rpc::envelope;
pub use hyprstream_rpc::zmtp_framing;

// Generated service clients via proc macro
// Each module contains: XxxClient struct with typed methods
pub mod model_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_client;
    hyprstream_rpc_derive::generate_rpc_service!("model", types_crate = hyprstream_rpc);
}

pub mod registry_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_client;
    hyprstream_rpc_derive::generate_rpc_service!("registry", types_crate = hyprstream_rpc);
}

pub mod policy_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_client;
    hyprstream_rpc_derive::generate_rpc_service!("policy", types_crate = hyprstream_rpc);
}

pub mod mcp_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_client;
    hyprstream_rpc_derive::generate_rpc_service!("mcp", types_crate = hyprstream_rpc);
}

pub mod inference_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_client;
    hyprstream_rpc_derive::generate_rpc_service!("inference", types_crate = hyprstream_rpc);
}

// WebTransport transport (wasm32 only)
#[cfg(target_arch = "wasm32")]
pub use hyprstream_rpc::web_transport;

// WASM-bindgen exports (wasm32 only)
#[cfg(target_arch = "wasm32")]
pub mod wasm_exports;
