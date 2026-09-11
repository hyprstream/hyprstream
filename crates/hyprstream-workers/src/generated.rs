//! Server dispatch generated against the canonical rpc-std contracts.
//!
//! The public clients/data types come from `hyprstream-rpc-std`; this AGPL crate
//! owns only handler traits and dispatch glue; callers import clients and wire
//! data directly from the canonical modules.

pub mod worker_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    hyprstream_rpc_derive::generate_rpc_server!(
        "worker",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub mod workflow_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    hyprstream_rpc_derive::generate_rpc_server!(
        "workflow",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}
