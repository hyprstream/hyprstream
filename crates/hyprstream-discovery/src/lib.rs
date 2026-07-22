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

extern crate self as hyprstream_discovery;

// Re-export shared capnp modules so generated code's `crate::*_capnp` resolves
pub use hyprstream_rpc::annotations_capnp;
pub use hyprstream_rpc::common_capnp;

// Cap'n Proto generated module
pub mod discovery_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(
        clippy::all,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::match_same_arms
    )]
    #![allow(
        clippy::semicolon_if_nothing_returned,
        clippy::doc_markdown,
        clippy::indexing_slicing
    )]
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
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
#[cfg(not(target_arch = "wasm32"))]
mod checkpointed_pds;

/// #893 (at9p D1) — `did:at9p` capsule resolver: turns a GATE-verified capsule
/// into a dialable `TransportConfig::iroh` (sibling to
/// `hyprstream_rpc::service_entry::decode_iroh`). Injectable at the
/// `Resolver::set_global` / network-profile resolver seam (#873).
pub mod at9p_resolver;

/// #896 (at9p D4) — `did:web` / `did:key` → `did:at9p` aliasing resolver:
/// resolves a classical DID to its authoritative `did:at9p` identity via mutual
/// `alsoKnownAs` attestation + the GATE pipeline. Sibling to `at9p_resolver`.
pub mod at9p_alias;

/// #1136 — explicit DID-anchored deployment trust source. Resolves the public
/// `(did:at9p, did:web)` pin pair without weakening the OS-owned-file source.
#[cfg(not(target_arch = "wasm32"))]
pub mod did_anchored;

/// #524 P1 — placement directory record ingestion + in-process index.
///
/// Polls `RecordResolver::resolve_repo` for a bootstrap set of node DIDs,
/// walks the returned CAR's MST, and decodes `ai.hyprstream.placement.*`
/// records (node / workload / group / groupItem) into a queryable index:
/// label-selector evaluation, resource matching, and bidirectional-consent
/// group membership. Consumed by `handle_query_candidates`.
pub mod placement_index;

/// #523 P0 / #524 / #628 — Scheduling Substrate.
///
/// The shared `LabelSelector` / `ResourceRequest` / `PlacementCandidate`
/// vocabulary + a small `filter → rank → select` helper + one `SelectionReport`
/// explain shape + a `CapabilitySource` contract, so every scheduling surface
/// (placement `queryCandidates`, the `SandboxPool` engine, `CellRouter` routing,
/// backend `selection.rs`) composes without duplicating filter/rank/explain
/// logic or capability truth.
///
/// **Not a framework**: free functions + plain types. See the module doc.
pub mod scheduling;

/// K4c (#787) — the placement→PodSpec vocabulary map (`k8s` feature).
///
/// Pure translation of a placement decision (in the [`scheduling`] substrate
/// vocabulary) into the Kubernetes `PodSpec` scheduling fields — `nodeSelector`
/// / `affinity` / `tolerations` / `resources` / `priorityClassName`. The
/// two-level-scheduling seam: hyprstream decides *what runs with what
/// constraints*, Kubernetes decides *which node*. No cluster, no kube client.
#[cfg(feature = "k8s")]
pub mod podspec;

// Re-export key types
pub use hyprstream_rpc::registry::SocketKind;
pub use hyprstream_rpc::resolver::Resolver;
#[cfg(not(target_arch = "wasm32"))]
pub use did_anchored::{DeploymentTrustSource, DidAnchors};
pub use service::{
    bootstrap_deployment_process, deployment_registry_verifier, resolve_and_authenticate_did_anchors,
    AuthorizationProvider, DiscoveryService, RecordCarData, RecordResolver, RegistryDeploymentVerifier,
    production_browser_currentness_verifier, production_browser_provisioning,
    production_rpc_client,
};

// Re-export generated types that consumers need
pub use generated::discovery_client::{
    dispatch_discovery, serialize_response, AuthMetadata, AuthMetadataList, DiscoveryClient,
    DiscoveryHandler, DiscoveryResponseVariant, EndpointInfo, ErrorInfo, GetRecordRequest,
    PingInfo, RecordCar, ServiceAnnouncement, ServiceEndpoints, ServiceList, ServiceSummary,
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod did_field_domain_type_tests {
    use super::*;
    use generated::discovery_client::{EnvelopeKeyset, RegisterEnvelopeKeysetRequest};
    use hyprstream_rpc::identity::Did;
    use hyprstream_rpc::{serialize_message, FromCapnp, ToCapnp};

    /// Field-level `$domainType("hyprstream_rpc::identity::Did")` on
    /// `EnvelopeKeyset.serviceDid` must generate a `Did` newtype field (not `String`),
    /// and the value must round-trip through the capnp wire (which stays `Text`).
    #[test]
    fn envelope_keyset_service_did_is_did_and_roundtrips() {
        let original = EnvelopeKeyset {
            service_did: Did::new("did:web:registry.example".to_owned()),
            cose_keyset_cbor: vec![1, 2, 3, 4],
            fetched_at: 1234,
        };
        // Type-level proof: the codegen emitted a `Did` field, not a `String`.
        let _typecheck: &Did = &original.service_did;

        let bytes = serialize_message(|msg| {
            let mut b = msg.init_root::<discovery_capnp::envelope_keyset::Builder>();
            original.write_to(&mut b);
        })
        .expect("serialize");

        let reader =
            capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
                .expect("read message");
        let root = reader
            .get_root::<discovery_capnp::envelope_keyset::Reader>()
            .expect("root reader");
        let back = EnvelopeKeyset::read_from(root).expect("read_from");

        assert_eq!(back.service_did, original.service_did);
        assert_eq!(back.service_did.as_str(), "did:web:registry.example");
        assert!(back.service_did.is_did_web());
        assert_eq!(back.cose_keyset_cbor, vec![1, 2, 3, 4]);
        assert_eq!(back.fetched_at, 1234);
    }

    /// Same for the request struct's `serviceDid` field.
    #[test]
    fn register_envelope_keyset_request_service_did_roundtrips() {
        let original = RegisterEnvelopeKeysetRequest {
            service_did: Did::new("did:key:z6Mkexample".to_owned()),
            cose_keyset_cbor: vec![9, 9, 9],
        };
        let _typecheck: &Did = &original.service_did;

        let bytes = serialize_message(|msg| {
            let mut b =
                msg.init_root::<discovery_capnp::register_envelope_keyset_request::Builder>();
            original.write_to(&mut b);
        })
        .expect("serialize");

        let reader =
            capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
                .expect("read message");
        let root = reader
            .get_root::<discovery_capnp::register_envelope_keyset_request::Reader>()
            .expect("root reader");
        let back = RegisterEnvelopeKeysetRequest::read_from(root).expect("read_from");

        assert_eq!(back.service_did.as_str(), "did:key:z6Mkexample");
        assert!(back.service_did.is_did_key());
    }
}
