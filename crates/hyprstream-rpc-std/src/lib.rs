//! Standard hyprstream service schemas and generated clients.
//!
//! This crate contains the Cap'n Proto service protocol definitions and
//! generated client types for all standard hyprstream services (model,
//! registry, inference, policy, mcp, etc.).
//!
//! The generated service client/data surface is Apache-2.0.  The
//! `pay_client` module is a license-preserving re-export of the separately
//! MIT-licensed `hyprstream-pay` protocol.  No AGPL service implementation is
//! in this dependency graph. Compiles to native and wasm32.

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

pub mod oauth_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/oauth_capnp.rs"));
}

pub mod worker_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/worker_capnp.rs"));
}

pub mod workflow_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/workflow_capnp.rs"));
}

pub mod discovery_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/discovery_capnp.rs"));
}

pub mod tui_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/tui_capnp.rs"));
}

pub mod compositor_ipc_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/compositor_ipc_capnp.rs"));
}

// ============================================================================
// Generated client types (from proc macro)
// Client-only: data structs, response enums, metadata. No server handlers.
// Compiles to all targets including wasm32.
// ============================================================================

pub mod inference_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("inference");
}

pub mod model_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("model");
}

pub mod registry_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("registry");
}

pub mod policy_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("policy");
}

pub mod mcp_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("mcp");
}


pub mod metrics_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("metrics");
}

pub mod oauth_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("oauth");
}

pub mod worker_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("worker");
}

pub mod workflow_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("workflow");
}

pub mod discovery_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("discovery");
}

pub mod tui_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("tui");
}

/// Data contracts for compositor/ChatApp IPC. This schema has no RPC request
/// union, so the generated module contains only serializable data types.
pub mod compositor_ipc_types {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("compositor_ipc");
}

/// MIT settlement/tariff protocol surface.
///
/// This is a license-preserving re-export of `hyprstream-pay`; the AGPL
/// settlement service remains in `hyprstream::services::pay` and is not a
/// dependency of this SDK crate.
pub mod pay_client {
    pub use hyprstream_pay::{
        attestation, capability, types, ATTESTATION_V1_TAG, ALL_SCOPES, IssueRequest,
        IssueResponse, PayError, SettlementAttestation, SettlementIssuer, TariffProvider,
        TariffQuote, TariffRequest, UnitRef,
    };
}

/// Stable, discoverable imports for third-party Rust consumers.  Service
/// implementations may re-export data types for compatibility, but the public
/// client implementations live only in this Apache-2.0 crate.
pub mod prelude {
    pub use crate::{
        discovery_client, inference_client, mcp_client, metrics_client, model_client,
        oauth_client, pay_client, policy_client, registry_client, tui_client, worker_client,
        workflow_client, compositor_ipc_types,
    };
    pub use hyprstream_rpc::{CallOptions, FromCapnp, RpcClient, RpcClientProvider, ToCapnp};
}

// Compile-time smoke test for the split codegen boundary.  The test module
// deliberately generates a server surface for an existing standard schema
// while importing its contracts from this crate; it ensures the server macro
// cannot accidentally grow a second client/data implementation.
#[cfg(test)]
mod server_codegen_smoke {
    pub mod model_server {
        hyprstream_rpc_derive::generate_rpc_server!("model", types_crate = crate, scope_handlers);
    }
}

#[cfg(test)]
mod contract_compat_tests {
    /// Schema IDs are wire identities. Keeping these assertions next to the
    /// canonical generated modules catches an accidental schema replacement
    /// before a client/service release can drift apart.
    #[test]
    fn first_party_schema_ids_are_stable() {
        use capnp::traits::HasTypeId;

        assert_eq!(
            <crate::worker_capnp::worker_request::Reader<'static> as HasTypeId>::TYPE_ID,
            0xbd3a_6a4e_7920_5f9c
        );
        assert_eq!(
            <crate::workflow_capnp::workflow_request::Reader<'static> as HasTypeId>::TYPE_ID,
            0x90a9_1c80_c0f0_7c3d
        );
        assert_eq!(
            <crate::discovery_capnp::discovery_request::Reader<'static> as HasTypeId>::TYPE_ID,
            0xecbd_7dea_6f0a_0715
        );
        assert_eq!(
            <crate::tui_capnp::tui_request::Reader<'static> as HasTypeId>::TYPE_ID,
            0x9e27_3fdc_e270_b381
        );
        assert_eq!(
            <crate::compositor_ipc_capnp::compositor_ipc_in::Reader<'static> as HasTypeId>::TYPE_ID,
            0x8c6b_4203_2547_6ee5
        );
    }
}

// ============================================================================
// WASM exports (browser only)
// ============================================================================

// Generic codegen-driven service→9p projection (`ServiceMount`) + stream pipes.
// Native and wasm both: the same generated dispatch drives the file surface on
// either target. (#539 T3)
pub mod vfs_mount;
pub mod stream_mount;
#[cfg(target_arch = "wasm32")]
pub mod wasm_exports;
#[cfg(target_arch = "wasm32")]
pub mod wasm_rpc_client;
#[cfg(target_arch = "wasm32")]
pub mod browser_session;
// Pure framing/reach-parsing for the moq worker — host-testable (not wasm-gated).
pub mod moq_frame;
#[cfg(target_arch = "wasm32")]
pub mod moq_worker;
#[cfg(target_arch = "wasm32")]
pub mod moq_wt_session;

// Phase 2: iroh peer identity + pkarr helpers exported to JavaScript.
#[cfg(target_arch = "wasm32")]
pub mod iroh_exports;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod did_field_domain_type_tests {
    use crate::mcp_client::CallTool;
    use hyprstream_rpc::identity::Did;
    use hyprstream_rpc::{serialize_message, FromCapnp, ToCapnp};

    /// Field-level `$domainType("hyprstream_rpc::identity::Did")` on
    /// `CallTool.callerIdentity` must generate a `Did` newtype field (not `String`),
    /// and the value must round-trip through the capnp wire (which stays `Text`).
    #[test]
    fn call_tool_caller_identity_is_did_and_roundtrips() {
        let original = CallTool {
            tool_name: "model_list".to_owned(),
            arguments: "{}".to_owned(),
            caller_identity: Did::new("did:key:z6MkcallerExample".to_owned()),
        };
        // Type-level proof: the codegen emitted a `Did` field, not a `String`.
        let _typecheck: &Did = &original.caller_identity;

        let bytes = serialize_message(|msg| {
            let mut b = msg.init_root::<crate::mcp_capnp::call_tool::Builder>();
            original.write_to(&mut b);
        })
        .expect("serialize");

        let reader =
            capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
                .expect("read message");
        let root = reader
            .get_root::<crate::mcp_capnp::call_tool::Reader>()
            .expect("root reader");
        let back = CallTool::read_from(root).expect("read_from");

        assert_eq!(back.caller_identity, original.caller_identity);
        assert_eq!(back.caller_identity.as_str(), "did:key:z6MkcallerExample");
        assert!(back.caller_identity.is_did_key());
        assert_eq!(back.tool_name, "model_list");
        assert_eq!(back.arguments, "{}");
    }
}
