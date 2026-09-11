//! RPC surface used by the AGPL daemon.
//!
//! Public contracts and clients are generated once by the Apache-2.0
//! `hyprstream-rpc-std` crate.  This module invokes the server-only generator
//! for the daemon's handlers and re-exports the canonical client surface.

pub mod model_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::model_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "model",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );

    /// Send the canonical Model health request and retain its typed response.
    ///
    /// The generated `health_check` convenience method turns `ErrorInfo` into
    /// an untyped error. OAI startup needs to distinguish one authenticated
    /// Model denial from local transport and response-verification failures,
    /// so this narrow helper keeps the method-bound call and parses the
    /// already-verified response bytes without changing shared code generation.
    pub(crate) async fn verified_health_check_response(
        client: &ModelClient,
    ) -> anyhow::Result<ModelResponseVariant> {
        const HEALTH_CHECK_METHOD: u16 = 3;

        let request_id = client.next_id();
        let payload = hyprstream_rpc::serialize_message(|message| {
            let mut request = message.init_root::<hyprstream_rpc_std::model_capnp::model_request::Builder>();
            request.set_id(request_id);
            request.set_health_check(());
        })?;
        let response = client
            .call_with_method(HEALTH_CHECK_METHOD, payload)
            .await?;
        ModelClient::parse_response(&response)
    }
}

pub mod registry_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::registry_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "registry",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub mod policy_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::policy_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "policy",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub mod inference_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::inference_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "inference",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub mod mcp_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::mcp_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "mcp",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub use hyprstream_discovery::generated::discovery_client;


pub mod tui_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::tui_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "tui",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

pub mod metrics_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::metrics_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "metrics",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}

// worker_client — use hyprstream_workers::generated::worker_client instead.
// workflow_client — use hyprstream_workers::generated::workflow_client instead.

pub mod oauth_client {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    pub use hyprstream_rpc_std::oauth_client::*;
    hyprstream_rpc_derive::generate_rpc_server!(
        "oauth",
        types_crate = hyprstream_rpc_std,
        scope_handlers,
    );
}
