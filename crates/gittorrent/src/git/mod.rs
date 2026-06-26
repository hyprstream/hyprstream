//! Git protocol utilities for SHA256 object operations

pub mod objects;
pub mod protocol;
pub mod remote;
pub mod remote_helper;
pub mod repository;
pub mod transport;

pub use objects::*;
pub use protocol::*;
pub use remote::*;
pub use repository::Repository;
pub use transport::{GittorrentTransportFactory, register_gittorrent_transport};

use std::path::PathBuf;
use crate::Result;

/// Execute a blocking git2 operation on a repository opened from the given path.
///
/// `git2::Repository` is `!Send`, so we pass an owned `PathBuf`, open a fresh
/// repository inside `spawn_blocking`, and return owned results.
pub(crate) async fn with_repo_blocking<F, T>(path: PathBuf, f: F) -> Result<T>
where
    F: FnOnce(git2::Repository) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let repo = git2::Repository::open(&path)?;
        f(repo)
    })
    .await
    .map_err(|e| crate::Error::other(format!("Task join error: {e}")))?
}