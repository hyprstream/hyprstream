//! CRI ImageClient implementation with Dragonfly-native blob fetching
//!
//! Provides Kubernetes CRI-aligned APIs for managing container images.
//! Backed by Nydus RAFS for chunk-level CAS deduplication.
//!
//! # Architecture
//!
//! ```text
//! ImageClient (CRI-aligned, client-side interface)
//!     │
//!     ├── list_images()     → List available images
//!     ├── image_status()    → Get image details
//!     ├── pull_image()      → Pull from registry (Dragonfly P2P)
//!     └── remove_image()    → Remove image
//!     │
//!     └── RafsStore
//!           │
//!           ├── ManifestFetcher    (HTTP - manifests only)
//!           ├── nydus-storage      (Dragonfly-native blob fetch)
//!           │
//!           ├── blobs/             (layer blobs)
//!           ├── bootstrap/         (RAFS metadata)
//!           ├── cache/             (nydus blob cache)
//!           └── refs/              (tag symlinks)
//! ```
//!
//! # Dragonfly Integration
//!
//! When `ImageConfig::dragonfly_peer` is set, all blob fetches route through
//! the Dragonfly peer for P2P distribution. This happens transparently via
//! nydus-storage's `blob_redirected_host` configuration.

// The nydus-bound submodules (client/store/rafs_builder/image_fs) require the
// `oci-image` feature (they pull nydus + fuse-backend-rs). `manifest` (pure
// HTTP) and `store_trait` (the trait seam) are nydus-free and always compiled —
// the trait surface stays feature-invariant so `WorkerService` can hold
// `Option<Arc<dyn ImageStore>>` with no cfg mirror (#646).
#[cfg(feature = "oci-image")]
mod client;
mod manifest;
// `pub(crate)` so sibling-module tests (e.g. runtime::sandbox_fs, FS-D #365)
// can synthesize RAFS images. The builder fn stays crate-internal.
#[cfg(feature = "oci-image")]
pub(crate) mod rafs_builder;
// #633 (spine of #508): the universal mountable OCI/RAFS image filesystem
// service — `ImageFs`, an `FsMount` = OverlayFs(in-process RAFS lower +
// writable upper). Native-only (the overlay/RAFS FileSystem stack is Linux).
#[cfg(all(feature = "oci-image", not(target_arch = "wasm32")))]
mod image_fs;
// The image-store trait + inventory registration seam — ALWAYS compiled
// (feature-invariant). Concrete backends (RafsStore under `oci-image`) impl
// `ImageStore` and `inventory::submit!` an `ImageBackendRegistration`.
pub mod store_trait;
#[cfg(feature = "oci-image")]
mod store;
// #652 (follow-on to #633/#641): projects `RafsStore`'s on-disk CAS as a
// 9P/VFS `Mount` — the storage-layer namespace surface, distinct from
// `ImageFs` (the composed filesystem a guest mounts). Only needs `RafsStore`
// itself, not the fuse/overlay stack `image_fs` pulls in, but both are gated
// on `oci-image` since that's what makes `RafsStore` exist at all.
#[cfg(feature = "oci-image")]
mod store_mount;

pub use crate::generated::worker_client::{
    ImageSpec, ImageInfo, ImageStatusResult,
    AuthConfig, FilesystemUsage, FilesystemIdentifier,
};
pub use manifest::{ImageReference, ManifestFetcher, ManifestResult, OciManifest};
#[cfg(all(feature = "oci-image", not(target_arch = "wasm32")))]
pub use image_fs::{image_fs_for, image_fs_from_rafs, ImageFs};
#[cfg(feature = "oci-image")]
pub use store::{GcStats, ImageMetadata, RafsStore};
#[cfg(feature = "oci-image")]
pub use store_mount::RafsStoreMount;
// The trait seam + inventory registration are always available; only the
// concrete `RafsStore` impl + its `submit!` require `oci-image`.
pub use store_trait::{ImageBackendRegistration, ImageStore};
