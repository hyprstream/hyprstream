//! Feature-invariant image-store trait + inventory registration.
//!
//! The `ImageStore` trait is the seam between the CRI `ImageHandler` RPC surface
//! (generated, always compiled) and concrete image backends (`RafsStore` today,
//! future plain-OCI / stargz / squashfs backends). It mirrors the five CRI image
//! ops the handler dispatches to.
//!
//! A trait object keeps `WorkerService` feature-invariant: it holds
//! `Option<Arc<dyn ImageStore>>` (always compiled) and the handler dispatches
//! through it with a single fallback error path. The only `#[cfg(feature)]` is
//! on the concrete `RafsStore` impl + its `inventory::submit!`, which live in
//! `image/store.rs` next to the nydus deps they pull.
//!
//! **Pluggability:** a new image backend is a new `impl ImageStore` + one
//! `inventory::submit!{ ImageBackendRegistration }`. Zero changes to
//! `WorkerService`. Same pattern as `SandboxBackend` / `BackendRegistration`
//! (`runtime/selection.rs`).

use std::sync::Arc;

use async_trait::async_trait;

use crate::generated::worker_client::{
    AuthConfig, FilesystemUsage, ImageInfo, ImageStatusResult,
};

/// The image-store operations the CRI `ImageHandler` RPC surface dispatches to.
///
/// `RafsStore` satisfies this structurally (its existing public methods match).
/// The trait is `Send + Sync` so it can be held as `Arc<dyn ImageStore>` on the
/// long-lived `WorkerService`.
#[async_trait]
pub trait ImageStore: Send + Sync {
    /// List available images.
    async fn list_images(&self) -> anyhow::Result<Vec<ImageInfo>>;

    /// Get image details for `image_ref`.
    async fn image_status(
        &self,
        image_ref: &str,
        verbose: bool,
    ) -> anyhow::Result<ImageStatusResult>;

    /// Pull `image_ref` from a registry, optionally with `auth`. Returns the
    /// resolved image id (content address).
    async fn pull_with_auth(
        &self,
        image_ref: &str,
        auth: Option<&AuthConfig>,
    ) -> anyhow::Result<String>;

    /// Remove a pulled image.
    async fn remove_image(&self, image_ref: &str) -> anyhow::Result<()>;

    /// Filesystem usage for the store's backing storage (for CRI `fsInfo`).
    async fn fs_info(&self) -> anyhow::Result<Vec<FilesystemUsage>>;
}

/// Inventory-based registration for image backends (mirrors
/// `BackendRegistration` in `runtime/selection.rs`).
///
/// A backend submits one of these next to its `impl ImageStore`; the worker
/// service startup resolves the highest-priority available registration and
/// constructs the store. When no backend is compiled in (e.g. a build without
/// `oci-image`), there is no registration and the slot is `None`.
#[derive(Clone)]
pub struct ImageBackendRegistration {
    /// Backend name (e.g. `"rafs"`). Selectable for explicit `image_backend`
    /// config; today there is only one per build.
    pub name: &'static str,
    /// Construct the store from an `ImageConfig`. Boxed because the config is
    /// owned by the caller and construction may fail.
    pub construct: fn(
        &crate::config::ImageConfig,
    ) -> anyhow::Result<Arc<dyn ImageStore>>,
}

inventory::collect!(ImageBackendRegistration);
