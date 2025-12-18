//! Registry service abstraction for in-process and future IPC access.
//!
//! This module provides a service-oriented interface to Git2DB, allowing
//! components to share a single registry through client handles.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              RegistryClient (trait)              │
//! └─────────────────────────────────────────────────┘
//!                         ▲
//!                         │
//!               ┌─────────┴─────────┐
//!               │    LocalClient    │
//!               │  (mpsc channels)  │
//!               └───────────────────┘
//!                         │
//!                         ▼
//! ┌─────────────────────────────────────────────────┐
//! │              LocalService                        │
//! │  registry: Git2DB                               │
//! │  (runs as tokio task)                           │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use git2db::service::{LocalService, RegistryClient};
//! use std::sync::Arc;
//!
//! // Start the service (returns client handle)
//! let client = LocalService::start("/path/to/registry").await?;
//! let client: Arc<dyn RegistryClient> = Arc::new(client);
//!
//! // Use the client
//! let repos = client.list().await?;
//!
//! // Clone shares the same underlying service
//! let client2 = client.clone();
//! ```

mod client;
mod local;
mod request;

pub use client::{RegistryClient, ServiceError};
pub use local::{LocalClient, LocalService};
pub use request::RegistryRequest;
