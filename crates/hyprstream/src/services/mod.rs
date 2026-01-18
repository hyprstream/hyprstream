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
//! use crate::services::{EnvelopeContext, ZmqService};
//! use hyprstream_rpc::prelude::*;
//! use hyprstream_rpc::service::{as_spawnable, InprocManager, ServiceManager};
//! use hyprstream_rpc::transport::TransportConfig;
//! use std::sync::Arc;
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
//! // Run the service using unified pattern
//! let service = MyService;
//! let transport = TransportConfig::inproc("my-service");
//! let context = Arc::new(zmq::Context::new());
//! let spawnable = as_spawnable(service, transport, context, verifying_key);
//! let manager = InprocManager::new();
//! let handle = manager.spawn(Box::new(spawnable)).await?;
//!
//! // Connect a client (signing is automatic)
//! let client = ZmqClient::new("inproc://my-service", signing_key, identity);
//! let response = client.call(payload, None).await?;
//!
//! // Stop the service
//! handle.stop().await;
//! ```

mod core;
mod traits;
pub mod callback;
pub mod inference;
pub mod model;
pub mod policy;
pub mod registry;
pub mod rpc_types;
pub mod stream;
pub mod worker;

pub use core::{
    EnvelopeContext, ServiceRunner, ZmqClient, ZmqService,
};

pub use traits::{
    RegistryClient, RegistryServiceError, RemoteInfo, RepositoryClient, WorktreeInfo,
};

pub use inference::{InferenceService, InferenceZmqClient, INFERENCE_ENDPOINT};
pub use registry::{
    RegistryOps, RegistryService, RegistryZmq, RegistryZmqClient, RepositoryZmqClient,
};
pub use policy::{PolicyService, PolicyZmqClient};
pub use model::{
    LoadedModelInfo, ModelHealthInfo, ModelService, ModelServiceConfig, ModelStatusInfo,
    ModelZmqClient, MODEL_ENDPOINT,
};
pub use stream::StreamService;
pub use worker::WorkerClient;
pub use callback::{CallbackRouter, Instance};
