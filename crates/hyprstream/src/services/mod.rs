//! Service layer for hyprstream
//!
//! This module provides ZMQ-based services for inference and registry operations.
//! Services use the REQ/REP pattern and Cap'n Proto for serialization.
//!
//! # Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `RequestLoop` verifies Ed25519 signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.subject()` for policy checks and resource isolation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  hyprstream/src/services/                                   │
//! │  ├── core.rs      ← ZmqService trait, runners, clients     │
//! │  ├── types.rs     ← Shared types (FsDirEntry, ModelInfo, etc.)│
//! │  ├── registry.rs  ← Registry service (REP) + client (REQ)  │
//! │  └── inference.rs ← Inference service (REP) + client (REQ) │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! Services implement `ZmqService` with infrastructure methods and are automatically
//! `Spawnable` via blanket impl:
//!
//! ```rust,ignore
//! use crate::services::{EnvelopeContext, ZmqService};
//! use hyprstream_rpc::prelude::*;
//! use hyprstream_rpc::service::{InprocManager, ServiceManager, Spawnable};
//! use hyprstream_rpc::transport::TransportConfig;
//! use std::sync::Arc;
//!
//! // Define a service with infrastructure
//! struct MyService {
//!     context: Arc<zmq::Context>,
//!     transport: TransportConfig,
//!     verifying_key: VerifyingKey,
//! }
//!
//! impl ZmqService for MyService {
//!     fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<Continuation>)> {
//!         // ctx.identity is already verified
//!         println!("Request from: {}", ctx.subject());
//!         Ok((vec![], None))
//!     }
//!
//!     fn name(&self) -> &str { "my-service" }
//!     fn context(&self) -> &Arc<zmq::Context> { &self.context }
//!     fn transport(&self) -> &TransportConfig { &self.transport }
//!     fn verifying_key(&self) -> VerifyingKey { self.verifying_key }
//! }
//!
//! // Services are directly Spawnable - no wrapping needed!
//! let service = MyService { context, transport, verifying_key };
//! let manager = InprocManager::new();
//! let handle = manager.spawn(Box::new(service)).await?;
//!
//! // Connect a client (signing is automatic)
//! let client = ZmqClient::new("inproc://my-service", context, signing_key, identity);
//! let response = client.call(payload, None).await?;
//!
//! // Stop the service
//! handle.stop().await;
//! ```

mod core;
mod types;
pub mod contained_root;
pub mod callback;
pub mod factories;
pub mod flight;
pub mod generated;
pub mod inference;
pub mod mcp_service;
pub mod model;
pub mod oauth;
pub mod oai;
pub mod policy;
pub mod registry;
pub mod rpc_types;
pub mod stream;
pub mod worker;

pub use core::{
    CallOptions, Continuation, EnvelopeContext, ZmqClient, ZmqService,
    create_service_client,
};

// Generated client types — the public API
pub use generated::registry_client::{
    RegistryClient as GenRegistryClient,
    RepositoryClient, WorktreeClient,
    TrackedRepository as GenTrackedRepository,
    WorktreeInfo as GenWorktreeInfo,
    RepositoryStatus as GenRepositoryStatus,
    FsStatResponse, FsDirEntryInfo, RemoteInfo,
    SeekWhenceEnum,
};

// Remaining domain types
pub use types::{MAX_FDS_GLOBAL, MAX_FDS_PER_CLIENT, MAX_FS_IO_SIZE};

pub use inference::{InferenceService, InferenceZmqClient, INFERENCE_ENDPOINT};
pub use registry::RegistryService;
pub use policy::PolicyService;
pub use generated::policy_client::PolicyClient;
pub use model::{
    LoadedModelInfo, ModelHealthInfo, ModelService, ModelServiceConfig, ModelStatusInfo,
    ModelZmqClient, MODEL_ENDPOINT,
};
pub use stream::StreamService;
pub use worker::{WorkerZmqClient, WorkflowZmqClient, build_authorize_fn};
pub use oauth::OAuthService;
pub use oai::OAIService;
pub use flight::FlightService;
pub use callback::{CallbackRouter, Instance};
pub use mcp_service::{McpConfig, McpService};
