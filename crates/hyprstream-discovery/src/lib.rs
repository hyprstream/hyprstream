//! Service discovery and endpoint resolution for hyprstream.
//!
//! This crate provides the `DiscoveryService` — the authoritative source
//! for service endpoint resolution in a hyprstream cluster. It wraps
//! the `EndpointRegistry` from `hyprstream-rpc` and exposes it as both
//! a local `Resolver` implementation and a Cap'n Proto RPC service.
//!
//! # Architecture
//!
//! ```text
//! hyprstream-rpc          trait Resolver + SocketKind + EndpointRegistry
//!     ↑
//! hyprstream-discovery    DiscoveryService (wraps registry, serves RPC)
//!     ↑
//! hyprstream              factory creates DiscoveryService, injects AuthorizationProvider
//! ```

// Re-export shared capnp modules so generated code's `crate::annotations_capnp` resolves
pub use hyprstream_rpc::annotations_capnp;

// Cap'n Proto generated module
pub mod discovery_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/discovery_capnp.rs"));
}

// Generated RPC service code (clients, handlers, dispatch)
pub mod generated {
    pub mod discovery_client {
        #![allow(dead_code, unused_imports, unused_variables)]
        #![allow(clippy::all)]
        hyprstream_rpc_derive::generate_rpc_service!("discovery", scope_handlers);
    }
}

mod service;

// Re-export key types
pub use hyprstream_rpc::resolver::Resolver;
pub use hyprstream_rpc::registry::SocketKind;
pub use service::{AuthorizationProvider, DiscoveryService};

// Re-export generated types that consumers need
pub use generated::discovery_client::{
    DiscoveryClient, DiscoveryHandler, DiscoveryResponseVariant,
    ErrorInfo, ServiceList, ServiceSummary, ServiceEndpoints, EndpointInfo,
    PingInfo, AuthMetadata, AuthMetadataList,
    dispatch_discovery, serialize_response,
};
