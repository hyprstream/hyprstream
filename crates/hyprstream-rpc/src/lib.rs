//! RPC infrastructure for hyprstream services.
//!
//! This crate provides:
//! - `ToCapnp` / `FromCapnp` traits and derive macros for Cap'n Proto serialization
//! - ZMQ transport implementation
//! - Service dispatch helpers
//! - Ed25519 signed envelopes for request authentication
//! - DH key exchange + HMAC for streaming response authentication
//!
//! # Security Model
//!
//! All ZMQ messages are wrapped in signed envelopes:
//! - **Transport layer**: CURVE encryption (TCP only)
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

pub mod capnp;
pub mod crypto;
pub mod envelope;
pub mod error;
pub mod platform;
pub mod zmtp_framing;

// ============================================================================
// Native-only modules (not compiled for wasm32)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub mod auth;
#[cfg(not(target_arch = "wasm32"))]
pub mod registry;
#[cfg(not(target_arch = "wasm32"))]
pub mod service;
#[cfg(not(target_arch = "wasm32"))]
pub mod streaming;
#[cfg(not(target_arch = "wasm32"))]
pub mod transport;
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

// ============================================================================
// Re-exports available on ALL targets
// ============================================================================

pub use capnp::{serialize_message, FromCapnp, ToCapnp};
pub use crypto::{
    generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes, ChainedStreamHmac,
    DefaultKeyExchange, HmacKey, KeyExchange, SharedSecret, SigningKey, VerifyingKey,
};

#[cfg(not(feature = "fips"))]
pub use crypto::{generate_ephemeral_keypair, ristretto_dh, RistrettoPublic, RistrettoSecret};

pub use envelope::{
    unwrap_and_verify, InMemoryNonceCache, NonceCache, RequestEnvelope, RequestIdentity,
    ResponseEnvelope, SignedEnvelope, Subject, MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
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
pub use service::{Continuation, EnvelopeContext, RequestLoop, ServiceClient, ServiceHandle, ZmqClient, ZmqService};

#[cfg(not(target_arch = "wasm32"))]
pub use streaming::{
    ChannelProgressReporter, forward_progress_to_stream, progress_channel,
    ProgressUpdate, ResponseStream, StreamChannel, StreamContext, StreamHandle,
    StreamPayload, StreamPublisher, StreamVerifier,
};

#[cfg(not(target_arch = "wasm32"))]
pub use service::spawner::{
    ProcessBackend, ProcessConfig, ProcessKind, ProcessSpawner,
    ProxyService, ServiceKind, ServiceMode, ServiceSpawner,
    Spawnable, SpawnedProcess, SpawnedService, SpawnerBackend, StandaloneBackend,
    SystemdBackend,
};

#[cfg(not(target_arch = "wasm32"))]
pub use service::{detect_service_manager, ServiceManager, StandaloneManager};

#[cfg(not(target_arch = "wasm32"))]
pub use service::metadata::{MethodMeta, ParamMeta};

#[cfg(all(not(target_arch = "wasm32"), feature = "systemd"))]
pub use service::SystemdManager;

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
        ChainedStreamHmac, DefaultKeyExchange, HmacKey, KeyExchange, SharedSecret,
        SigningKey, VerifyingKey,
        // Envelope
        unwrap_envelope, unwrap_and_verify, InMemoryNonceCache, NonceCache, RequestEnvelope,
        RequestIdentity, ResponseEnvelope, SignedEnvelope, Subject,
        MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
        // Error
        EnvelopeError, EnvelopeResult, Result, RpcError,
        // Service
        EnvelopeContext, RequestLoop, ServiceClient, ServiceHandle, ZmqClient, ZmqService,
        // Streaming
        StreamContext, StreamPublisher,
        // Spawner
        ProcessBackend, ProcessConfig, ProcessKind, ProcessSpawner,
        ProxyService, ServiceKind, ServiceMode, ServiceSpawner,
        Spawnable, SpawnedProcess, SpawnedService, SpawnerBackend, StandaloneBackend,
        SystemdBackend,
        // Service manager
        detect_service_manager, ServiceManager, StandaloneManager,
    };

    #[cfg(not(feature = "fips"))]
    pub use crate::{generate_ephemeral_keypair, ristretto_dh, RistrettoPublic, RistrettoSecret};

    #[cfg(feature = "systemd")]
    pub use crate::SystemdManager;

    // Registry (with renamed imports for convenience)
    pub use crate::registry::{
        global as registry_global, init as registry_init, try_global as registry_try_global,
        EndpointMode, EndpointRegistry, ServiceEntry, ServiceRegistration,
    };

    // Transport
    pub use crate::transport::{AsyncTransport, Transport, TransportConfig};
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
