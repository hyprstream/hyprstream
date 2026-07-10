//! Service layer for hyprstream
//!
//! This module provides services for inference and registry operations.
//! Services use Cap'n Proto for serialization via moq-net + UDS transport.
//!
//! # Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `process_request` verifies Ed25519 signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.subject()` for policy checks and resource isolation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  hyprstream/src/services/                                   │
//! │  ├── core.rs      ← RequestService trait re-exports           │
//! │  ├── types.rs     ← Shared types (FsDirEntry, ModelInfo, etc.)│
//! │  ├── registry.rs  ← Registry service + client               │
//! │  └── inference.rs ← Inference service + client              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! Services implement `RequestService` and are automatically `Spawnable` via blanket impl:
//!
//! ```rust,ignore
//! use crate::services::{EnvelopeContext, RequestService};
//! use hyprstream_rpc::prelude::*;
//! use hyprstream_rpc::transport::TransportConfig;
//!
//! struct MyService {
//!     transport: TransportConfig,
//!     signing_key: SigningKey,
//! }
//!
//! impl RequestService for MyService {
//!     async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<Continuation>)> {
//!         println!("Request from: {}", ctx.subject());
//!         Ok((vec![], None))
//!     }
//!
//!     fn name(&self) -> &str { "my-service" }
//!     fn transport(&self) -> &TransportConfig { &self.transport }
//!     fn signing_key(&self) -> SigningKey { self.signing_key.clone() }
//! }
//!
//! let service = MyService { transport, signing_key };
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
mod worktree_helpers;
pub use worktree_helpers::StatResult;
// contained_root replaced by hyprstream-containedfs crate
pub mod discovery;
pub mod editing;
pub mod factories;
pub mod flight;
pub mod generated;
#[cfg(feature = "oci-image")]
pub mod image_substrate;
pub mod inference;
pub mod mcp_service;
pub mod metrics;
pub mod ninep_bridge;
pub mod model;
pub mod oauth;
pub mod oai;
pub mod policy;
pub mod registry;
pub mod router;
pub mod xet;
pub mod xet_provenance;
pub mod fs;
pub mod namespace_builder;
pub mod remote_mount;
pub mod remote_registry_mount;
pub mod kata_9p_backend;
pub mod rpc_types;
pub mod typed;
pub mod worker;

/// Phase-1 cellular-ledger local-enforcer (epic #922, #925). Gated behind the
/// `ledger` cargo feature, default off — the scheduler quota path is unchanged
/// until an operator opts in.
#[cfg(feature = "ledger")]
pub mod ledger;

pub use core::{
    Continuation, EnvelopeContext, RequestService,
};

// Generated client types — the public API
pub use generated::registry_client::{
    RegistryClient,
    RepositoryClient, WorktreeClient, CtlClient,
    TrackedRepository as GenTrackedRepository,
    WorktreeInfo as GenWorktreeInfo,
    RepositoryStatus as GenRepositoryStatus,
    RemoteInfo,
    RWalk, ROpen, RRead, RWrite, RStat,
    NpStat as NpStatData, Qid as QidData,
    FileStatus, LogEntry, ValidationResult, FileInfo,
    DocFormat,
};

// Remaining domain types
pub use types::{
    MAX_FDS_GLOBAL, MAX_FDS_PER_CLIENT, MAX_FS_IO_SIZE,
    DEFAULT_IOUNIT, MAX_IOUNIT,
    QTDIR, QTFILE, OREAD, OWRITE, ORDWR, OTRUNC, ORCLOSE, DMDIR,
    FsDirEntryInfo,
};
// Open mode constants also re-exported from hyprstream-vfs::mount for VFS consumers.

pub use inference::{InferenceService, InferenceServiceConfig, INFERENCE_ENDPOINT};
pub use generated::inference_client::InferenceClient;
pub use registry::RegistryService;
pub use policy::PolicyService;
pub use generated::policy_client::PolicyClient;
pub use model::{ModelService, ModelServiceConfig, MODEL_ENDPOINT};
pub use generated::model_client::ModelClient;
pub use worker::build_authorize_fn;
pub use namespace_builder::{build_standard_namespace, StandardNamespaceConfig};
pub use hyprstream_workers::runtime::WorkerClient;
pub use oauth::OAuthService;
pub use oai::OAIService;
pub use xet::{XetService, XetState};
#[cfg(feature = "oci-image")]
pub use image_substrate::{
    create_image_substrate_router, ImageSubstratePolicy, ImageSubstrateState,
};
pub use flight::FlightService;
pub use discovery::DiscoveryService;
pub use generated::discovery_client::DiscoveryClient;
pub use mcp_service::{McpConfig, McpService};
pub use metrics::MetricsService;
