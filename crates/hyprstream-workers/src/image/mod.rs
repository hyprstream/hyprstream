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

mod client;
mod manifest;
mod store;

pub use client::{
    AuthConfig, FilesystemIdentifier, FilesystemUsage, Image, ImageFilter, ImageClient,
    ImageSpec, ImageStatusResponse, ImageZmq,
};
pub use manifest::{ImageReference, ManifestFetcher, ManifestResult, OciManifest};
pub use store::{GcStats, ImageMetadata, RafsStore};
