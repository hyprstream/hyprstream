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
//! - Default: X25519 for DH key exchange
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
#[allow(dead_code)]
#[allow(unused_imports)]
pub mod common_capnp {
    include!(concat!(env!("OUT_DIR"), "/common_capnp.rs"));
}

#[allow(dead_code)]
#[allow(unused_imports)]
pub mod events_capnp {
    include!(concat!(env!("OUT_DIR"), "/events_capnp.rs"));
}

pub mod capnp;
pub mod crypto;
pub mod envelope;
pub mod error;
pub mod service;
pub mod transport;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::capnp::{serialize_message, FromCapnp, ToCapnp};
    pub use crate::crypto::{
        generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes,
        ChainedStreamHmac, DefaultKeyExchange, HmacKey, KeyExchange, SharedSecret, SigningKey,
        StreamHmac, VerifyingKey,
    };
    pub use crate::envelope::{
        InMemoryNonceCache, NonceCache, RequestEnvelope, RequestIdentity, ResponseEnvelope,
        SignedEnvelope, MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
    };

    #[cfg(not(feature = "fips"))]
    pub use crate::crypto::{
        ed25519_to_x25519_pubkey, ed25519_to_x25519_secret, generate_ephemeral_keypair,
    };
    pub use crate::error::{EnvelopeError, EnvelopeResult};
    pub use crate::service::{
        EnvelopeContext, RpcService, ServiceHandle, ServiceRunner, ZmqClient, ZmqService,
    };
    pub use crate::transport::{AsyncTransport, Transport, TransportConfig};

    // Re-export derive macros
    pub use hyprstream_rpc_derive::{rpc_method, FromCapnp, ToCapnp};
}

// Re-export key types at crate root
pub use capnp::{serialize_message, FromCapnp, ToCapnp};
pub use crypto::{
    generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes, ChainedStreamHmac,
    DefaultKeyExchange, HmacKey, KeyExchange, SharedSecret, SigningKey, StreamHmac, VerifyingKey,
};

#[cfg(not(feature = "fips"))]
pub use crypto::{ed25519_to_x25519_pubkey, ed25519_to_x25519_secret, generate_ephemeral_keypair};
pub use envelope::{
    InMemoryNonceCache, NonceCache, RequestEnvelope, RequestIdentity, ResponseEnvelope,
    SignedEnvelope, MAX_CLOCK_SKEW_MS, MAX_TIMESTAMP_AGE_MS,
};
pub use error::{EnvelopeError, EnvelopeResult};
pub use hyprstream_rpc_derive::{rpc_method, FromCapnp, ToCapnp};
pub use service::{EnvelopeContext, ServiceHandle, ServiceRunner, ZmqClient, ZmqService};
