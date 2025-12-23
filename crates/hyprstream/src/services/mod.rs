//! Service layer for hyprstream
//!
//! This module provides ZMQ-based services for inference and registry operations.
//! Services use the REQ/REP pattern and Cap'n Proto for serialization.
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
//! use crate::services::{ServiceRunner, ZmqService};
//!
//! // Define a service
//! struct MyService;
//!
//! impl ZmqService for MyService {
//!     fn handle_request(&self, request: &[u8]) -> Result<Vec<u8>> {
//!         // Handle the request
//!         Ok(vec![])
//!     }
//!
//!     fn name(&self) -> &str {
//!         "my-service"
//!     }
//! }
//!
//! // Run the service
//! let runner = ServiceRunner::new("inproc://my-service");
//! let handle = runner.run(MyService);
//!
//! // Connect a client
//! let client = AsyncServiceClient::new("inproc://my-service");
//! let response = client.call(request).await?;
//!
//! // Stop the service
//! handle.stop().await;
//! ```

mod core;
mod traits;
pub mod inference;
pub mod registry;

pub use core::{
    AsyncServiceClient, ServiceClient, ServiceHandle, ServiceRunner, ZmqService,
};

pub use traits::{
    RegistryClient, RegistryServiceError, RemoteInfo, RepositoryClient, WorktreeInfo,
};

pub use inference::{
    InferenceService, InferenceZmqClient, INFERENCE_ENDPOINT, INFERENCE_STREAM_ENDPOINT,
};
pub use registry::{RegistryService, RegistryZmqClient, RepositoryZmqClient, REGISTRY_ENDPOINT};
