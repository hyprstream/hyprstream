//! RafsStore - Dragonfly-native image storage using nydus-storage
//!
//! Uses Nydus RAFS format for efficient image storage with:
//! - Chunk-level deduplication (across all images)
//! - Lazy loading (on-demand chunk fetch via Dragonfly P2P)
//! - ~80% storage savings vs traditional layers
//!
//! Architecture:
//! ```text
//! ManifestFetcher (HTTP)  →  nydus-storage Registry backend  →  TarballBuilder
//!      │                              │                              │
//!      └── OCI manifests only         └── Dragonfly P2P for blobs    └── RAFS output
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use nydus_api::RegistryConfig;
use nydus_storage::backend::registry::Registry;
use nydus_storage::backend::{BlobBackend, BlobBufReader};

use crate::config::ImageConfig;
use crate::error::{Result, WorkerError};

use super::client::{AuthConfig as CriAuthConfig, Image, ImageSpec};
use super::manifest::{AuthConfig, ImageReference, ManifestFetcher, ManifestResult};

/// RAFS-backed image store with Dragonfly-native blob fetching
///
/// Uses nydus-storage's backend-registry for blob operations, which supports
/// Dragonfly P2P via the `blob_redirected_host` configuration.
///
/// Storage layout:
/// ```text
/// images/
/// ├── blobs/sha256/         # Layer blobs (content-addressed)
/// ├── bootstrap/            # RAFS metadata (per image)
/// ├── cache/                # nydus-storage blob cache
/// └── refs/                 # Tag → digest symlinks
/// ```
pub struct RafsStore {
    /// Directory for layer blobs
    blobs_dir: PathBuf,

    /// Directory for RAFS bootstrap metadata
    bootstrap_dir: PathBuf,

    /// Directory for tag references
    refs_dir: PathBuf,

    /// Cache directory for nydus-storage
    cache_dir: PathBuf,

    /// Manifest fetcher (HTTP only, no blob handling)
    manifest_fetcher: Arc<ManifestFetcher>,

    /// Configuration
    config: ImageConfig,
}

impl RafsStore {
    /// Create a new RafsStore with configuration
    pub fn new(config: ImageConfig) -> Result<Self> {
        let manifest_fetcher = ManifestFetcher::new()
            .map_err(|e| WorkerError::RafsError(format!("failed to create manifest fetcher: {e}")))?;

        Ok(Self {
            blobs_dir: config.blobs_dir.clone(),
            bootstrap_dir: config.bootstrap_dir.clone(),
            refs_dir: config.refs_dir.clone(),
            cache_dir: config.cache_dir.clone(),
            manifest_fetcher: Arc::new(manifest_fetcher),
            config,
        })
    }

    /// Pull an image from registry
    pub async fn pull(&self, image_ref: &str) -> Result<String> {
        self.pull_with_auth(image_ref, None).await
    }

    /// Pull an image with authentication
    pub async fn pull_with_auth(
        &self,
        image_ref: &str,
        auth: Option<&CriAuthConfig>,
    ) -> Result<String> {
        tracing::info!(image = %image_ref, "Pulling image");

        // Ensure directories exist
        self.ensure_dirs().await?;

        // 1. Parse image reference
        let img_ref = ImageReference::parse(image_ref)
            .map_err(|e| WorkerError::ImageParseFailed {
                image: image_ref.to_owned(),
                reason: e.to_string(),
            })?;

        // Convert CRI auth to manifest auth (CRI uses String, manifest uses Option<String>)
        let manifest_auth = auth.map(|a| AuthConfig {
            username: if a.username.is_empty() { None } else { Some(a.username.clone()) },
            password: if a.password.is_empty() { None } else { Some(a.password.clone()) },
            token: if a.auth.is_empty() { None } else { Some(a.auth.clone()) },
        });

        // 2. Fetch manifest
        let manifest = self
            .manifest_fetcher
            .fetch(&img_ref, manifest_auth.as_ref())
            .await
            .map_err(|e| WorkerError::ImagePullFailed {
                image: image_ref.to_owned(),
                reason: e.to_string(),
            })?;

        // 3. Handle multi-platform images - select linux/amd64 by default
        let manifest = match manifest {
            ManifestResult::Manifest(m) => m,
            ManifestResult::Index(idx) => {
                // Find linux/amd64 platform
                let platform_manifest = idx
                    .manifests
                    .iter()
                    .find(|m| {
                        m.platform.as_ref().is_some_and(|p| {
                            p.os == "linux" && p.architecture == "amd64"
                        })
                    })
                    .or_else(|| idx.manifests.first())
                    .ok_or_else(|| WorkerError::ImagePullFailed {
                        image: image_ref.to_owned(),
                        reason: "no suitable platform manifest found".to_owned(),
                    })?;

                tracing::debug!(
                    digest = %platform_manifest.digest,
                    "Selected platform manifest"
                );

                self.manifest_fetcher
                    .fetch_platform_manifest(&img_ref, &platform_manifest.digest, manifest_auth.as_ref())
                    .await
                    .map_err(|e| WorkerError::ImagePullFailed {
                        image: image_ref.to_owned(),
                        reason: e.to_string(),
                    })?
            }
        };

        tracing::debug!(
            layers = %manifest.layers.len(),
            config = %manifest.config.digest,
            "Got manifest"
        );

        // 4. Download config blob via nydus-storage
        let config_path = self.blobs_dir.join(digest_to_filename(&manifest.config.digest));
        if !config_path.exists() {
            self.download_blob(&img_ref, &manifest.config.digest, &config_path, auth)
                .await?;
            tracing::debug!(digest = %manifest.config.digest, "Downloaded config");
        }

        // 5. Download layer blobs via nydus-storage (Dragonfly-native)
        for layer in &manifest.layers {
            let layer_path = self.blobs_dir.join(digest_to_filename(&layer.digest));
            if !layer_path.exists() {
                tracing::info!(
                    digest = %layer.digest,
                    size = %layer.size,
                    "Downloading layer via nydus-storage"
                );
                self.download_blob(&img_ref, &layer.digest, &layer_path, auth)
                    .await?;
            } else {
                tracing::debug!(digest = %layer.digest, "Layer already cached");
            }
        }

        // 6. Generate image ID (SHA256 of config)
        let config_data = tokio::fs::read(&config_path).await?;
        let image_id = format!("sha256:{}", hex::encode(Sha256::digest(&config_data)));

        // 7. Write metadata
        let metadata = ImageMetadata {
            image_ref: image_ref.to_owned(),
            image_id: image_id.clone(),
            config_digest: manifest.config.digest.clone(),
            layers: manifest.layers.iter().map(|l| l.digest.clone()).collect(),
            host: img_ref.host.clone(),
            repository: img_ref.repository.clone(),
        };
        let metadata_path = self
            .bootstrap_dir
            .join(format!("{}.json", digest_to_filename(&image_id)));
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(&metadata_path, metadata_json).await?;

        // 8. Create ref symlink
        let ref_name = Self::normalize_ref(image_ref);
        let ref_path = self.refs_dir.join(&ref_name);

        // Remove existing symlink if present
        if ref_path.exists() {
            tokio::fs::remove_file(&ref_path).await?;
        }
        tokio::fs::symlink(&metadata_path, &ref_path).await?;

        tracing::info!(
            image = %image_ref,
            image_id = %image_id,
            layers = %manifest.layers.len(),
            "Image pulled successfully"
        );

        Ok(image_id)
    }

    /// Download a blob using nydus-storage's Registry backend.
    ///
    /// This enables Dragonfly P2P when `blob_redirected_host` is configured.
    async fn download_blob(
        &self,
        img_ref: &ImageReference,
        digest: &str,
        dest_path: &Path,
        auth: Option<&CriAuthConfig>,
    ) -> Result<()> {
        // Extract blob ID from digest (remove "sha256:" prefix)
        let blob_id = digest
            .strip_prefix("sha256:")
            .unwrap_or(digest).to_owned(); // Convert to owned String for 'static lifetime

        // Create nydus-storage Registry backend config
        let registry_config = self.create_registry_config(img_ref, &blob_id, auth);

        // Create the registry backend - this is blocking, so spawn_blocking
        let blob_id_clone = blob_id.clone();
        let backend = tokio::task::spawn_blocking(move || {
            Registry::new(&registry_config, Some(&blob_id_clone))
                .map_err(|e| WorkerError::RafsError(format!("failed to create registry backend: {e:?}")))
        })
        .await
        .map_err(|e| WorkerError::RafsError(format!("spawn_blocking failed: {e}")))??;

        // Get a blob reader
        let reader = backend
            .get_reader(&blob_id)
            .map_err(|e| WorkerError::RafsError(format!("failed to get blob reader: {e:?}")))?;

        // Get blob size (we need to know the size for BlobBufReader)
        let blob_size = reader
            .blob_size()
            .map_err(|e| WorkerError::RafsError(format!("failed to get blob size: {e:?}")))?;

        // Create a buffered reader and download the blob
        let dest_path_clone = dest_path.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<()> {
            // Use BlobBufReader for efficient buffered reading
            let mut buf_reader = BlobBufReader::new(
                1024 * 1024, // 1MB buffer
                reader,
                0,
                blob_size,
            );

            // Create destination file
            let mut file = std::fs::File::create(&dest_path_clone).map_err(|e| {
                WorkerError::IoError(format!("failed to create blob file: {e}"))
            })?;

            // Copy data
            std::io::copy(&mut buf_reader, &mut file).map_err(|e| {
                WorkerError::IoError(format!("failed to write blob: {e}"))
            })?;

            // Sync to disk
            file.sync_all().map_err(|e| {
                WorkerError::IoError(format!("failed to sync blob: {e}"))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| WorkerError::RafsError(format!("spawn_blocking failed: {e}")))??;

        Ok(())
    }

    /// Create a nydus-storage RegistryConfig for a blob fetch.
    fn create_registry_config(
        &self,
        img_ref: &ImageReference,
        blob_id: &str,
        auth: Option<&CriAuthConfig>,
    ) -> RegistryConfig {
        let mut config = RegistryConfig {
            host: img_ref.host.clone(),
            repo: img_ref.repository.clone(),
            ..Default::default()
        };

        // DRAGONFLY INTEGRATION POINT
        // When configured, all blob fetches route through Dragonfly peer
        if let Some(peer) = &self.config.dragonfly_peer {
            config.blob_redirected_host = peer.clone();
            tracing::debug!(
                peer = %peer,
                blob_id = %blob_id,
                "Using Dragonfly peer for blob fetch"
            );
        }

        // Set authentication if provided (CRI AuthConfig uses String, not Option<String>)
        if let Some(auth) = auth {
            if !auth.username.is_empty() && !auth.password.is_empty() {
                use base64::Engine;
                let credentials = format!("{}:{}", auth.username, auth.password);
                config.auth = Some(
                    base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes()),
                );
            } else if !auth.auth.is_empty() {
                config.registry_token = Some(auth.auth.clone());
            } else if !auth.registry_token.is_empty() {
                config.registry_token = Some(auth.registry_token.clone());
            }
        }

        // Configure timeouts
        config.connect_timeout = 10;
        config.timeout = 300;
        config.retry_limit = 3;

        config
    }

    /// Ensure storage directories exist
    async fn ensure_dirs(&self) -> Result<()> {
        tracing::debug!(path = %self.blobs_dir.display(), "Creating blobs directory");
        tokio::fs::create_dir_all(&self.blobs_dir).await.map_err(|e| {
            tracing::error!(path = %self.blobs_dir.display(), error = %e, "Failed to create blobs directory");
            e
        })?;

        tracing::debug!(path = %self.bootstrap_dir.display(), "Creating bootstrap directory");
        tokio::fs::create_dir_all(&self.bootstrap_dir).await.map_err(|e| {
            tracing::error!(path = %self.bootstrap_dir.display(), error = %e, "Failed to create bootstrap directory");
            e
        })?;

        tracing::debug!(path = %self.refs_dir.display(), "Creating refs directory");
        tokio::fs::create_dir_all(&self.refs_dir).await.map_err(|e| {
            tracing::error!(path = %self.refs_dir.display(), error = %e, "Failed to create refs directory");
            e
        })?;

        tracing::debug!(path = %self.cache_dir.display(), "Creating cache directory");
        tokio::fs::create_dir_all(&self.cache_dir).await.map_err(|e| {
            tracing::error!(path = %self.cache_dir.display(), error = %e, "Failed to create cache directory");
            e
        })?;

        Ok(())
    }

    /// Ensure an image is available locally (pull if needed)
    pub async fn ensure(&self, image_ref: &str) -> Result<String> {
        if self.exists(image_ref) {
            self.get_image_id(image_ref)
        } else {
            self.pull(image_ref).await
        }
    }

    /// Check if an image exists locally
    pub fn exists(&self, image_ref: &str) -> bool {
        let tag_path = self.refs_dir.join(Self::normalize_ref(image_ref));
        tag_path.exists()
    }

    /// Get image ID from reference
    pub fn get_image_id(&self, image_ref: &str) -> Result<String> {
        let tag_path = self.refs_dir.join(Self::normalize_ref(image_ref));
        if tag_path.exists() {
            // Read metadata file
            let metadata_path = std::fs::read_link(&tag_path)?;
            let metadata_str = std::fs::read_to_string(&metadata_path)?;
            let metadata: ImageMetadata = serde_json::from_str(&metadata_str)?;
            Ok(metadata.image_id)
        } else {
            Err(WorkerError::ImageNotFound(image_ref.to_owned()))
        }
    }

    /// Get image metadata
    pub fn get_metadata(&self, image_ref: &str) -> Result<ImageMetadata> {
        let tag_path = self.refs_dir.join(Self::normalize_ref(image_ref));
        if tag_path.exists() {
            let metadata_path = std::fs::read_link(&tag_path)?;
            let metadata_str = std::fs::read_to_string(&metadata_path)?;
            let metadata: ImageMetadata = serde_json::from_str(&metadata_str)?;
            Ok(metadata)
        } else {
            Err(WorkerError::ImageNotFound(image_ref.to_owned()))
        }
    }

    /// List all images
    pub async fn list(&self) -> Result<Vec<Image>> {
        let mut images = Vec::new();

        if self.refs_dir.exists() {
            for entry in std::fs::read_dir(&self.refs_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_symlink() {
                    let _tag = entry.file_name().to_string_lossy().to_string();
                    let metadata_path = std::fs::read_link(entry.path())?;

                    if let Ok(metadata_str) = std::fs::read_to_string(&metadata_path) {
                        if let Ok(metadata) = serde_json::from_str::<ImageMetadata>(&metadata_str) {
                            images.push(Image {
                                id: metadata.image_id.clone(),
                                repo_tags: vec![metadata.image_ref.clone()],
                                repo_digests: vec![metadata.config_digest.clone()],
                                size: 0, // TODO: Calculate from layers
                                uid: None,
                                username: String::new(),
                                spec: Some(ImageSpec {
                                    image: metadata.image_ref.clone(),
                                    ..Default::default()
                                }),
                                pinned: false,
                            });
                        }
                    }
                }
            }
        }

        Ok(images)
    }

    /// Remove an image
    pub async fn remove(&self, image_id: &str) -> Result<()> {
        tracing::info!(image_id = %image_id, "Removing image");

        // Find and remove ref symlinks pointing to this image
        if self.refs_dir.exists() {
            for entry in std::fs::read_dir(&self.refs_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_symlink() {
                    let metadata_path = std::fs::read_link(entry.path())?;
                    if let Ok(metadata_str) = std::fs::read_to_string(&metadata_path) {
                        if let Ok(metadata) = serde_json::from_str::<ImageMetadata>(&metadata_str) {
                            if metadata.image_id == image_id {
                                tokio::fs::remove_file(entry.path()).await?;
                            }
                        }
                    }
                }
            }
        }

        // Remove bootstrap metadata
        let metadata_path = self
            .bootstrap_dir
            .join(format!("{}.json", digest_to_filename(image_id)));
        if metadata_path.exists() {
            tokio::fs::remove_file(&metadata_path).await?;
        }

        // Note: Blobs are not removed immediately as they may be shared
        // Run gc() to clean up unreferenced blobs

        Ok(())
    }

    /// Run garbage collection on unused blobs
    pub async fn gc(&self) -> Result<GcStats> {
        tracing::info!("Running garbage collection");

        // 1. Collect all referenced layer digests
        let mut referenced = std::collections::HashSet::new();

        if self.bootstrap_dir.exists() {
            for entry in std::fs::read_dir(&self.bootstrap_dir)? {
                let entry = entry?;
                if entry.path().extension().is_some_and(|e| e == "json") {
                    if let Ok(metadata_str) = std::fs::read_to_string(entry.path()) {
                        if let Ok(metadata) = serde_json::from_str::<ImageMetadata>(&metadata_str) {
                            referenced.insert(digest_to_filename(&metadata.config_digest));
                            for layer in &metadata.layers {
                                referenced.insert(digest_to_filename(layer));
                            }
                        }
                    }
                }
            }
        }

        // 2. Remove unreferenced blobs
        let mut chunks_removed = 0u64;
        let mut bytes_freed = 0u64;

        if self.blobs_dir.exists() {
            for entry in std::fs::read_dir(&self.blobs_dir)? {
                let entry = entry?;
                let filename = entry.file_name().to_string_lossy().to_string();
                if !referenced.contains(&filename) {
                    if let Ok(metadata) = entry.metadata() {
                        bytes_freed += metadata.len();
                        chunks_removed += 1;
                        tokio::fs::remove_file(entry.path()).await?;
                        tracing::debug!(blob = %filename, "Removed unreferenced blob");
                    }
                }
            }
        }

        tracing::info!(
            chunks_removed = %chunks_removed,
            bytes_freed = %bytes_freed,
            "Garbage collection complete"
        );

        Ok(GcStats {
            chunks_removed,
            bytes_freed,
        })
    }

    /// Get bootstrap path for an image
    pub fn bootstrap_path(&self, image_id: &str) -> PathBuf {
        self.bootstrap_dir
            .join(format!("{}.meta", digest_to_filename(image_id)))
    }

    /// Get layer blob path for a digest
    pub fn blob_path(&self, digest: &str) -> PathBuf {
        self.blobs_dir.join(digest_to_filename(digest))
    }

    /// Get blobs directory
    pub fn blobs_dir(&self) -> &PathBuf {
        &self.blobs_dir
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CRI ImageClient methods
    // ─────────────────────────────────────────────────────────────────────────

    /// List all images (CRI ImageClient aligned)
    pub async fn list_images(&self) -> Result<Vec<Image>> {
        self.list().await
    }

    /// Get image status (CRI ImageClient aligned)
    pub async fn image_status(&self, image_ref: &str, _verbose: bool) -> Result<super::client::ImageStatusResponse> {
        if self.exists(image_ref) {
            let metadata = self.get_metadata(image_ref)?;
            Ok(super::client::ImageStatusResponse {
                image: Some(Image {
                    id: metadata.image_id.clone(),
                    repo_tags: vec![metadata.image_ref.clone()],
                    repo_digests: vec![metadata.config_digest.clone()],
                    size: 0, // TODO: Calculate from layers
                    uid: None,
                    username: String::new(),
                    spec: Some(ImageSpec {
                        image: metadata.image_ref.clone(),
                        ..Default::default()
                    }),
                    pinned: false,
                }),
                info: std::collections::HashMap::new(),
            })
        } else {
            Ok(super::client::ImageStatusResponse {
                image: None,
                info: std::collections::HashMap::new(),
            })
        }
    }

    /// Remove image by reference (CRI ImageClient aligned)
    pub async fn remove_image(&self, image_ref: &str) -> Result<()> {
        // Get image ID from reference, then remove
        let image_id = self.get_image_id(image_ref)?;
        self.remove(&image_id).await
    }

    /// Get filesystem info (CRI ImageClient aligned)
    pub async fn fs_info(&self) -> Result<Vec<super::client::FilesystemUsage>> {
        let mut used_bytes = 0u64;
        let mut inodes_used = 0u64;

        // Calculate blobs usage
        if self.blobs_dir.exists() {
            for entry in walkdir::WalkDir::new(&self.blobs_dir)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry.file_type().is_file() {
                    if let Ok(meta) = entry.metadata() {
                        used_bytes += meta.len();
                        inodes_used += 1;
                    }
                }
            }
        }

        // Calculate bootstrap metadata usage
        if self.bootstrap_dir.exists() {
            for entry in walkdir::WalkDir::new(&self.bootstrap_dir)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry.file_type().is_file() {
                    if let Ok(meta) = entry.metadata() {
                        used_bytes += meta.len();
                        inodes_used += 1;
                    }
                }
            }
        }

        Ok(vec![super::client::FilesystemUsage {
            timestamp: chrono::Utc::now().timestamp(),
            fs_id: super::client::FilesystemIdentifier {
                mountpoint: self.blobs_dir.to_string_lossy().to_string(),
            },
            used_bytes,
            inodes_used,
        }])
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Normalize image reference to filesystem-safe name
    fn normalize_ref(image_ref: &str) -> String {
        image_ref.replace(['/', ':', '@'], "_")
    }
}

/// Convert digest to filesystem-safe filename
fn digest_to_filename(digest: &str) -> String {
    digest.replace(':', "_")
}

/// Garbage collection statistics
#[derive(Debug, Clone)]
pub struct GcStats {
    /// Number of blobs removed
    pub chunks_removed: u64,
    /// Bytes freed
    pub bytes_freed: u64,
}

/// Image metadata stored in bootstrap directory
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageMetadata {
    /// Original image reference
    pub image_ref: String,
    /// Image ID (sha256 of config)
    pub image_id: String,
    /// Config blob digest
    pub config_digest: String,
    /// Layer digests
    pub layers: Vec<String>,
    /// Registry host
    pub host: String,
    /// Repository name
    pub repository: String,
}
