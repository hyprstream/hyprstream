//! Request types for the registry service.
//!
//! These types define the message format for communication between
//! LocalClient and LocalService via channels.

use crate::{RepoId, TrackedRepository};
use std::path::PathBuf;
use tokio::sync::oneshot;

use super::client::WorktreeInfo;
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

    // === Repository Operations ===

    /// Create a worktree for a repository.
    CreateWorktree {
        repo_id: RepoId,
        worktree_path: PathBuf,
        branch: String,
        reply: oneshot::Sender<Result<PathBuf, ServiceError>>,
    },

    /// List worktrees for a repository.
    ListWorktrees {
        repo_id: RepoId,
        reply: oneshot::Sender<Result<Vec<WorktreeInfo>, ServiceError>>,
    },

    /// Get worktree path for a branch.
    WorktreePath {
        repo_id: RepoId,
        branch: String,
        reply: oneshot::Sender<Result<Option<PathBuf>, ServiceError>>,
    },

    /// Create a branch.
    CreateBranch {
        repo_id: RepoId,
        name: String,
        from: Option<String>,
        reply: oneshot::Sender<Result<(), ServiceError>>,
    },

    /// Checkout a branch or ref.
    Checkout {
        repo_id: RepoId,
        ref_spec: String,
        reply: oneshot::Sender<Result<(), ServiceError>>,
    },

    /// Get the default branch.
    DefaultBranch {
        repo_id: RepoId,
        reply: oneshot::Sender<Result<String, ServiceError>>,
    },

    /// List all branches.
    ListBranches {
        repo_id: RepoId,
        reply: oneshot::Sender<Result<Vec<String>, ServiceError>>,
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
            Self::CreateWorktree {
                repo_id,
                worktree_path,
                branch,
                ..
            } => {
                write!(
                    f,
                    "RegistryRequest::CreateWorktree {{ repo_id: {:?}, path: {:?}, branch: {:?} }}",
                    repo_id, worktree_path, branch
                )
            }
            Self::ListWorktrees { repo_id, .. } => {
                write!(f, "RegistryRequest::ListWorktrees {{ repo_id: {:?} }}", repo_id)
            }
            Self::WorktreePath { repo_id, branch, .. } => {
                write!(
                    f,
                    "RegistryRequest::WorktreePath {{ repo_id: {:?}, branch: {:?} }}",
                    repo_id, branch
                )
            }
            Self::CreateBranch {
                repo_id,
                name,
                from,
                ..
            } => {
                write!(
                    f,
                    "RegistryRequest::CreateBranch {{ repo_id: {:?}, name: {:?}, from: {:?} }}",
                    repo_id, name, from
                )
            }
            Self::Checkout { repo_id, ref_spec, .. } => {
                write!(
                    f,
                    "RegistryRequest::Checkout {{ repo_id: {:?}, ref_spec: {:?} }}",
                    repo_id, ref_spec
                )
            }
            Self::DefaultBranch { repo_id, .. } => {
                write!(f, "RegistryRequest::DefaultBranch {{ repo_id: {:?} }}", repo_id)
            }
            Self::ListBranches { repo_id, .. } => {
                write!(f, "RegistryRequest::ListBranches {{ repo_id: {:?} }}", repo_id)
            }
        }
    }
}
