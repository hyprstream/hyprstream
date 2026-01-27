//! ImageClient trait and ZMQ client
//!
//! Client-side trait for CRI ImageService (`runtime.v1`) for future kubelet compatibility.

use crate::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// CRI-aligned ImageClient trait
///
/// Client-side interface for CRI ImageService (`runtime.v1`) for future kubelet/crictl compatibility.
#[async_trait]
pub trait ImageClient: Send + Sync {
    /// List images matching filter
    async fn list_images(&self, filter: Option<&ImageFilter>) -> Result<Vec<Image>>;

    /// Get image status
    async fn image_status(
        &self,
        image: &ImageSpec,
        verbose: bool,
    ) -> Result<ImageStatusResponse>;

    /// Pull an image from registry
    async fn pull_image(
        &self,
        image: &ImageSpec,
        auth: Option<&AuthConfig>,
    ) -> Result<String>;

    /// Remove an image
    async fn remove_image(&self, image: &ImageSpec) -> Result<()>;

    /// Get filesystem info for images
    async fn image_fs_info(&self) -> Result<Vec<FilesystemUsage>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Image specification
#[derive(Debug, Clone, Default)]
pub struct ImageSpec {
    /// Image reference (e.g., "docker.io/library/alpine:latest")
    pub image: String,
    /// Annotations
    pub annotations: HashMap<String, String>,
    /// Runtime handler hint
    pub runtime_handler: String,
}

/// Image filter for listing
#[derive(Debug, Clone, Default)]
pub struct ImageFilter {
    /// Image spec to match
    pub image: Option<ImageSpec>,
}

/// Image information
#[derive(Debug, Clone)]
pub struct Image {
    /// Image ID (digest)
    pub id: String,
    /// Repository tags
    pub repo_tags: Vec<String>,
    /// Repository digests
    pub repo_digests: Vec<String>,
    /// Size in bytes
    pub size: u64,
    /// UID the image is configured to run as
    pub uid: Option<i64>,
    /// Username the image is configured to run as
    pub username: String,
    /// Image spec
    pub spec: Option<ImageSpec>,
    /// Whether image is pinned (won't be GC'd)
    pub pinned: bool,
}

/// Image status response
#[derive(Debug, Clone)]
pub struct ImageStatusResponse {
    /// Image info (None if not found)
    pub image: Option<Image>,
    /// Additional info (if verbose)
    pub info: HashMap<String, String>,
}

/// Authentication configuration
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// Authentication token
    pub auth: String,
    /// Server address
    pub server_address: String,
    /// Identity token
    pub identity_token: String,
    /// Registry token
    pub registry_token: String,
}

/// Filesystem usage information
#[derive(Debug, Clone)]
pub struct FilesystemUsage {
    /// Timestamp
    pub timestamp: i64,
    /// Filesystem ID
    pub fs_id: FilesystemIdentifier,
    /// Used bytes
    pub used_bytes: u64,
    /// Inodes used
    pub inodes_used: u64,
}

/// Filesystem identifier
#[derive(Debug, Clone)]
pub struct FilesystemIdentifier {
    /// Mountpoint
    pub mountpoint: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// ZMQ Client Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// ZMQ client for ImageClient
pub struct ImageZmq {
    _endpoint: String,
}

impl ImageZmq {
    /// Create a new ImageZmq client
    pub fn new(endpoint: &str) -> Self {
        Self {
            _endpoint: endpoint.to_owned(),
        }
    }
}

#[async_trait]
impl ImageClient for ImageZmq {
    async fn list_images(&self, _filter: Option<&ImageFilter>) -> Result<Vec<Image>> {
        todo!("Implement ZMQ call")
    }

    async fn image_status(
        &self,
        _image: &ImageSpec,
        _verbose: bool,
    ) -> Result<ImageStatusResponse> {
        todo!("Implement ZMQ call")
    }

    async fn pull_image(
        &self,
        _image: &ImageSpec,
        _auth: Option<&AuthConfig>,
    ) -> Result<String> {
        todo!("Implement ZMQ call")
    }

    async fn remove_image(&self, _image: &ImageSpec) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn image_fs_info(&self) -> Result<Vec<FilesystemUsage>> {
        todo!("Implement ZMQ call")
    }
}
