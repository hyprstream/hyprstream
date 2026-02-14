//! ImageClient trait and ZMQ client
//!
//! Client-side trait for CRI ImageService (`runtime.v1`) for future kubelet compatibility.
//! Uses generated `*Data` types from the Cap'n Proto schema directly.

use crate::error::Result;
use async_trait::async_trait;

use crate::generated::worker_client::{
    ImageSpec, ImageInfo, ImageStatusResult,
    AuthConfig, FilesystemUsage,
};

/// CRI-aligned ImageClient trait
///
/// Client-side interface for CRI ImageService (`runtime.v1`) for future kubelet/crictl compatibility.
/// Uses generated `*Data` types directly — no domain type wrappers.
#[async_trait]
pub trait ImageClient: Send + Sync {
    /// List images matching filter
    async fn list_images(&self, filter: Option<&ImageSpec>) -> Result<Vec<ImageInfo>>;

    /// Get image status
    async fn image_status(
        &self,
        image: &ImageSpec,
        verbose: bool,
    ) -> Result<ImageStatusResult>;

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
