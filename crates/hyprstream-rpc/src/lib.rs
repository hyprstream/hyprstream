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

// Cap'n Proto generated modules
// Note: Generated code uses unwrap/expect internally - this is standard capnp-rust behavior
// These modules are excluded from clippy checks via cargo configuration
pub mod common_capnp {
    #![allow(dead_code, unused_imports, clippy::unwrap_used, clippy::expect_used)]
    include!(concat!(env!("OUT_DIR"), "/common_capnp.rs"));
}

pub mod events_capnp {
    #![allow(dead_code, unused_imports, clippy::unwrap_used, clippy::expect_used)]
    include!(concat!(env!("OUT_DIR"), "/events_capnp.rs"));
}

pub mod streaming_capnp {
    #![allow(dead_code, unused_imports, clippy::unwrap_used, clippy::expect_used)]
    include!(concat!(env!("OUT_DIR"), "/streaming_capnp.rs"));
}

pub mod auth;
pub mod capnp;
pub mod crypto;
pub mod envelope;
pub mod error;
pub mod registry;
pub mod service;
pub mod streaming;
pub mod transport;

/// Prelude for convenient imports.
///
/// Re-exports the most commonly used types for `use hyprstream_rpc::prelude::*;`
pub mod prelude {
    // Re-export from crate root (DRY - single source of truth)
    pub use crate::{
        // Serialization
        serialize_message, FromCapnp, ToCapnp,
        // Crypto
        generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes,
        ChainedStreamHmac, DefaultKeyExchange, HmacKey, KeyExchange, SharedSecret,
        SigningKey, StreamHmac, VerifyingKey,
        // Envelope
        unwrap_envelope, InMemoryNonceCache, NonceCache, RequestEnvelope, RequestIdentity,
        ResponseEnvelope, SignedEnvelope, MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
        // Error
        EnvelopeError, EnvelopeResult, Result, RpcError,
        // Service
        EnvelopeContext, RequestLoop, ServiceHandle, ZmqClient, ZmqService,
        // Streaming
        StreamContext, StreamPublisher,
        // Spawner
        ProcessBackend, ProcessConfig, ProcessKind, ProcessSpawner,
        ProxyService, ServiceKind, ServiceMode, ServiceSpawner,
        Spawnable, SpawnedProcess, SpawnedService, SpawnerBackend, StandaloneBackend,
        SystemdBackend,
        // Service manager
        detect_service_manager, ServiceManager, StandaloneManager,
        // Derive macros (FromCapnp, ToCapnp already imported above as traits)
        rpc_method,
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

// Utility modules (merged from hyprstream-utils)
pub mod notify;
pub mod paths;
pub mod socket;

// Re-export key types at crate root
pub use capnp::{serialize_message, FromCapnp, ToCapnp};
pub use crypto::{
    generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes, ChainedStreamHmac,
    DefaultKeyExchange, HmacKey, KeyExchange, SharedSecret, SigningKey, StreamHmac, VerifyingKey,
};

#[cfg(not(feature = "fips"))]
pub use crypto::{generate_ephemeral_keypair, ristretto_dh, RistrettoPublic, RistrettoSecret};
pub use envelope::{
    unwrap_envelope, InMemoryNonceCache, NonceCache, RequestEnvelope, RequestIdentity,
    ResponseEnvelope, SignedEnvelope, MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
};
pub use error::{EnvelopeError, EnvelopeResult, Result, RpcError};
pub use hyprstream_rpc_derive::{authorize, register_scopes, rpc_method, service_factory, FromCapnp, ToCapnp};
pub use service::{EnvelopeContext, RequestLoop, ServiceHandle, ZmqClient, ZmqService};
pub use streaming::{
    ChannelProgressReporter, forward_progress_to_stream, progress_channel,
    ProgressUpdate, ResponseStream, StreamChannel, StreamContext, StreamPublisher,
};
pub use service::spawner::{
    ProcessBackend, ProcessConfig, ProcessKind, ProcessSpawner,
    ProxyService, ServiceKind, ServiceMode, ServiceSpawner,
    Spawnable, SpawnedProcess, SpawnedService, SpawnerBackend, StandaloneBackend,
    SystemdBackend,
};

// Service manager re-exports
pub use service::{detect_service_manager, ServiceManager, StandaloneManager};
#[cfg(feature = "systemd")]
pub use service::SystemdManager;

/// Check if system booted with systemd
#[cfg(feature = "systemd")]
pub fn has_systemd() -> bool {
    systemd::daemon::booted().unwrap_or(false)
}

#[cfg(not(feature = "systemd"))]
pub fn has_systemd() -> bool {
    false
}
