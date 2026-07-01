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

pub mod oauth_capnp {
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/oauth_capnp.rs"));
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

pub mod notification_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc_std;
    hyprstream_rpc_derive::generate_rpc_client!("notification");
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

// ============================================================================
// WASM exports (browser only)
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub mod vfs_mount;
#[cfg(target_arch = "wasm32")]
pub mod stream_mount;
#[cfg(target_arch = "wasm32")]
pub mod wasm_exports;
#[cfg(target_arch = "wasm32")]
pub mod wasm_rpc_client;
#[cfg(target_arch = "wasm32")]
pub mod moq_worker;

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
