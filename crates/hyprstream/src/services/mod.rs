//! Service layer for hyprstream
//!
//! This module provides ZMQ-based services for inference and registry operations.
//! Services use the REQ/REP pattern and Cap'n Proto for serialization.
//!
//! # Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `ServiceRunner` verifies Ed25519 signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.casbin_subject()` for policy checks
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  hyprstream/src/services/                                   │
//! │  ├── core.rs      ← ZmqService trait, runners, clients     │
//! │  ├── traits.rs    ← RegistryClient, RepositoryClient traits│
//! │  ├── registry.rs  ← Registry service (REP) + client (REQ)  │
//! │  └── inference.rs ← Inference service (REP) + client (REQ) │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::services::{EnvelopeContext, ServiceRunner, ZmqService};
//! use hyprstream_rpc::prelude::*;
//!
//! // Define a service
//! struct MyService;
//!
//! impl ZmqService for MyService {
//!     fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
//!         // ctx.identity is already verified
//!         println!("Request from: {}", ctx.casbin_subject());
//!         Ok(vec![])
//!     }
//!
//!     fn name(&self) -> &str {
//!         "my-service"
//!     }
//! }
//!
//! // Run the service with server's verifying key (waits for socket bind)
//! let runner = ServiceRunner::new("inproc://my-service", server_pubkey);
//! let handle = runner.run(MyService).await?;
//!
//! // Connect a client (signing is automatic)
//! let client = ZmqClient::new("inproc://my-service", signing_key, identity);
//! let response = client.call(payload).await?;
//!
//! // Stop the service
//! handle.stop().await;
//! ```

mod core;
mod traits;
pub mod inference;
pub mod policy;
pub mod registry;
pub mod rpc_types;

pub use core::{
    EnvelopeContext, ServiceClient, ServiceHandle, ServiceRunner, ZmqClient, ZmqService,
};

pub use traits::{
    RegistryClient, RegistryServiceError, RemoteInfo, RepositoryClient, WorktreeInfo,
};

pub use inference::{
    InferenceService, InferenceZmqClient, INFERENCE_ENDPOINT, INFERENCE_STREAM_ENDPOINT,
};
pub use registry::{
    RegistryOps, RegistryService, RegistryZmq, RegistryZmqClient, RepositoryZmqClient,
    REGISTRY_ENDPOINT,
};
pub use policy::{PolicyService, PolicyZmqClient, POLICY_ENDPOINT};
