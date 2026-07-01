//! RPC infrastructure for hyprstream services.
//!
//! This crate provides:
//! - `ToCapnp` / `FromCapnp` traits and derive macros for Cap'n Proto serialization
//! - ZMTP framing over UDS/QUIC/iroh transports
//! - Service dispatch helpers
//! - Ed25519 signed envelopes for request authentication
//! - DH key exchange + HMAC for streaming response authentication
//!
//! # Security Model
//!
//! All RPC messages are wrapped in signed envelopes:
//! - **Transport layer**: TLS 1.3 (QUIC) or UDS peer credentials (IPC)
//! - **Application layer**: Ed25519 signatures (survives message forwarding)
//!
//! Streaming responses use HMAC-SHA256 derived from DH shared secrets.
//!
//! # Feature Flags
//!
//! - Default: Ristretto255 for DH key exchange (prime-order group, no cofactor issues)
//! - `fips`: ECDH P-256 for FIPS 140-2 compliance
//!
//! # Example
//!
//! ```ignore
//! use hyprstream_rpc::prelude::*;
//!
//! #[derive(FromCapnp)]
//! #[capnp(my_schema_capnp::my_response)]
//! pub struct MyResponse {
//!     pub result: String,
//! }
//!
//! // Client method becomes a one-liner
//! async fn get_result(&self) -> Result<MyResponse> {
//!     self.rpc.call(GetResultRequest).await
//! }
//! ```

// ============================================================================
// Modules available on ALL targets (including wasm32)
// ============================================================================

// Cap'n Proto generated modules
pub mod common_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/common_capnp.rs"));
}

pub mod events_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/events_capnp.rs"));
}

pub mod nine_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/nine_capnp.rs"));
}

pub mod streaming_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/streaming_capnp.rs"));
}

pub mod annotations_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/annotations_capnp.rs"));
}

pub mod optional_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/optional_capnp.rs"));
}

pub mod common_types {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc;
    hyprstream_rpc_derive::generate_rpc_service!("common");
}

/// Code-generated streaming data types (#273): `StreamInfo`, `StreamOpt`, and the
/// five QoS axis unions (`Ordering`/`Delivery`/`Completion`/`Retention`/
/// `OverflowPolicy`). Re-exported through `stream_info` (the canonical hub) so
/// service codegen and call sites keep resolving `hyprstream_rpc::stream_info::*`.
///
/// NOTE: this module also generates the wire types (`StreamBlock`, `StreamPayload`,
/// `StreamControl`, etc.). Those are NOT re-exported — `streaming.rs` remains the
/// authoritative hand-written implementation for the wire path; the generated
/// duplicates live here unused (`#![allow(dead_code)]`).
pub mod streaming_types {
    #![allow(dead_code, unused_imports, unused_variables)]
    #![allow(clippy::all)]
    extern crate self as hyprstream_rpc;
    hyprstream_rpc_derive::generate_rpc_service!("streaming");
}

pub mod capnp;
pub mod cid;
pub mod crypto;
pub mod envelope;
pub mod error;
pub mod platform;
pub mod rpc_client;
pub mod stream_info;
pub mod zmtp_framing;

// ============================================================================
// Cross-platform modules (available on all targets including wasm32)
// ============================================================================

/// Schema introspection metadata types — used by proc macro codegen on all targets.
pub mod metadata {
    pub use crate::_metadata::*;
}
mod _metadata;

// ============================================================================
// Transport traits (available on ALL targets)
pub mod transport_traits;
pub mod stream_consumer;

// Native-only modules (not compiled for wasm32)
// ============================================================================

pub mod auth;
#[cfg(not(target_arch = "wasm32"))]
pub mod registry;
#[cfg(not(target_arch = "wasm32"))]
pub mod resolver;
#[cfg(not(target_arch = "wasm32"))]
pub mod service;
#[cfg(not(target_arch = "wasm32"))]
pub mod streaming;
#[cfg(not(target_arch = "wasm32"))]
pub mod stream_provenance;
#[cfg(not(target_arch = "wasm32"))]
pub mod moq_stream;
#[cfg(not(target_arch = "wasm32"))]
pub mod moq_authz;
#[cfg(not(target_arch = "wasm32"))]
pub mod moq_event;
#[cfg(not(target_arch = "wasm32"))]
pub mod events;
#[cfg(not(target_arch = "wasm32"))]
pub mod latch;
#[cfg(not(target_arch = "wasm32"))]
pub mod transport;
#[cfg(not(target_arch = "wasm32"))]
pub mod dial;
#[cfg(not(target_arch = "wasm32"))]
pub mod service_entry;
// Shared `did:key` (Ed25519) codec — compiled on all targets so the native
// `did_web` resolver and the wasm32 `iroh_peer` identity helpers share one
// implementation (#475). `did_web` re-exports its public fns for compatibility.
pub mod did_key;
#[cfg(not(target_arch = "wasm32"))]
pub mod did_web;
#[cfg(not(target_arch = "wasm32"))]
pub mod admission;
#[cfg(not(target_arch = "wasm32"))]
pub mod notify;
#[cfg(not(target_arch = "wasm32"))]
pub mod paths;
#[cfg(not(target_arch = "wasm32"))]
pub mod socket;

// ============================================================================
// WASM-bindgen API (wasm32 target only)
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub mod wasm_api;
#[cfg(target_arch = "wasm32")]
pub mod web_transport;
// #409 Path A: browser counterpart to native `dial`. Compiles only on wasm32;
// see `dial_wasm.rs` for why this is a separate module rather than an arm in
// native `dial()` (which is itself `#[cfg(not(target_arch = "wasm32"))]`).
#[cfg(target_arch = "wasm32")]
pub mod dial_wasm;
// Phase 2: iroh peer identity + pkarr helpers for wasm32. Adds iroh as a
// first-class wasm32 dep — browser gets own NodeId, did:key conversion,
// and native pkarr lookup. Full dial_iroh_reach() is Phase 3.
#[cfg(target_arch = "wasm32")]
pub mod iroh_peer;

// ============================================================================
// Re-exports available on ALL targets
// ============================================================================

pub use capnp::{serialize_message, FromCapnp, ToCapnp};
pub use rpc_client::{CallOptions, RequestBuilder, RpcClient, RpcClientImpl};
pub use transport_traits::{PublishSink, Signer, Transport};
pub mod identity;
pub mod node_identity;
#[cfg(not(target_arch = "wasm32"))]
pub mod federated_identity;
pub mod signer;
pub use stream_info::StreamInfo;
pub use crypto::{
    generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes,
    DefaultKeyExchange, KeyExchange, SharedSecret, SigningKey, StreamHmacState, VerifyingKey,
};

#[cfg(not(feature = "fips"))]
pub use crypto::{generate_ephemeral_keypair, ristretto_dh, RistrettoPublic, RistrettoSecret};

pub use envelope::{
    unwrap_and_verify,
    Authorization, EnvelopeVerification, FederatedToken, InMemoryNonceCache, KeyedPqTrustStore,
    NonceCache, PqTrustStore, RequestEnvelope, ResponseEnvelope, SignedEnvelope, Subject,
    TokenClaims, UnwrapOptions, MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
};
#[cfg(not(target_arch = "wasm32"))]
pub use envelope::unwrap_envelope;
pub use error::{EnvelopeError, EnvelopeResult, Result, RpcError};

// ============================================================================
// Native-only re-exports
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub use hyprstream_rpc_derive::{authorize, service_factory, FromCapnp, ToCapnp};

#[cfg(not(target_arch = "wasm32"))]
pub use resolver::Resolver;

#[cfg(not(target_arch = "wasm32"))]
pub use registry::SocketKind;

#[cfg(not(target_arch = "wasm32"))]
pub use service::{Continuation, EnvelopeContext, Spawnable, ServiceHandle, RequestService};

#[cfg(not(target_arch = "wasm32"))]
pub use streaming::{
    ChannelProgressReporter, derive_client_stream_keys,
    progress_channel, ProgressUpdate, ResponseStream, StreamChannel, StreamContext,
    StreamPayload, StreamPayloadData, StreamVerifier,
};

// ============================================================================
// Prelude (native only — too many native-only types)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub mod prelude {
    pub use crate::{
        // Serialization
        serialize_message, FromCapnp, ToCapnp,
        // Crypto
        generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes,
        DefaultKeyExchange, KeyExchange, SharedSecret, SigningKey, StreamHmacState, VerifyingKey,
        // Envelope
        unwrap_envelope, unwrap_and_verify, Authorization, EnvelopeVerification,
        InMemoryNonceCache, NonceCache, RequestEnvelope, ResponseEnvelope,
        SignedEnvelope, Subject, UnwrapOptions,
        MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
        // Error
        EnvelopeError, EnvelopeResult, Result, RpcError,
        // Service (transport)
        EnvelopeContext, ServiceHandle, RequestService,
        // Streaming
        StreamContext,
    };

    #[cfg(not(feature = "fips"))]
    pub use crate::{generate_ephemeral_keypair, ristretto_dh, RistrettoPublic, RistrettoSecret};

    // Registry (with renamed imports for convenience)
    pub use crate::registry::{
        global as registry_global, init as registry_init, try_global as registry_try_global,
        EndpointMode, EndpointRegistry, ServiceEntry, ServiceRegistration,
    };

    // Transport
    pub use crate::transport::TransportConfig;
}

// ============================================================================
// Systemd helpers (native only)
// ============================================================================

#[cfg(all(not(target_arch = "wasm32"), feature = "systemd"))]
pub fn has_systemd() -> bool {
    systemd::daemon::booted().unwrap_or(false)
}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "systemd")))]
pub fn has_systemd() -> bool {
    false
}
