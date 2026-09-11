//! Server dispatch generated against the canonical rpc-std contracts.
//!
//! The public clients/data types come from `hyprstream-rpc-std`; this AGPL crate
//! owns only handler traits and dispatch glue.

pub mod worker_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::worker_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "worker",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub mod workflow_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::workflow_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "workflow",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}
