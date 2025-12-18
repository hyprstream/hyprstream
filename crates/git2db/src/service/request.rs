//! Request types for the registry service.
//!
//! These types define the message format for communication between
//! LocalClient and LocalService via channels.

use crate::{RepoId, TrackedRepository};
use std::path::PathBuf;
use tokio::sync::oneshot;

use super::ServiceError;

/// Request types for the registry service.
///
/// All variants include a oneshot reply channel for returning results.
/// All responses return owned data (no lifetimes) to work across async boundaries.
pub enum RegistryRequest {
    /// List all tracked repositories.
    List {
        reply: oneshot::Sender<Result<Vec<TrackedRepository>, ServiceError>>,
    },

    /// Get repository by ID.
    Get {
        id: RepoId,
        reply: oneshot::Sender<Result<Option<TrackedRepository>, ServiceError>>,
    },

    /// Get repository by name.
    GetByName {
        name: String,
        reply: oneshot::Sender<Result<Option<TrackedRepository>, ServiceError>>,
    },

    /// Clone a repository from URL.
    Clone {
        url: String,
        name: Option<String>,
        reply: oneshot::Sender<Result<RepoId, ServiceError>>,
    },

    /// Register an existing repository.
    Register {
        id: RepoId,
        name: Option<String>,
        path: PathBuf,
        reply: oneshot::Sender<Result<(), ServiceError>>,
    },

    /// Upsert: update if exists, create if not.
    Upsert {
        name: String,
        url: String,
        reply: oneshot::Sender<Result<RepoId, ServiceError>>,
    },

    /// Remove a repository from the registry.
    Remove {
        id: RepoId,
        reply: oneshot::Sender<Result<(), ServiceError>>,
    },

    /// Health check request.
    HealthCheck {
        reply: oneshot::Sender<Result<(), ServiceError>>,
    },
}

impl std::fmt::Debug for RegistryRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::List { .. } => write!(f, "RegistryRequest::List"),
            Self::Get { id, .. } => write!(f, "RegistryRequest::Get {{ id: {:?} }}", id),
            Self::GetByName { name, .. } => {
                write!(f, "RegistryRequest::GetByName {{ name: {:?} }}", name)
            }
            Self::Clone { url, name, .. } => {
                write!(f, "RegistryRequest::Clone {{ url: {:?}, name: {:?} }}", url, name)
            }
            Self::Register { id, name, path, .. } => {
                write!(
                    f,
                    "RegistryRequest::Register {{ id: {:?}, name: {:?}, path: {:?} }}",
                    id, name, path
                )
            }
            Self::Upsert { name, url, .. } => {
                write!(f, "RegistryRequest::Upsert {{ name: {:?}, url: {:?} }}", name, url)
            }
            Self::Remove { id, .. } => write!(f, "RegistryRequest::Remove {{ id: {:?} }}", id),
            Self::HealthCheck { .. } => write!(f, "RegistryRequest::HealthCheck"),
        }
    }
}
