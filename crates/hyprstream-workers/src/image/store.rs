//! RafsStore - Dragonfly-native image storage using nydus-storage
//!
//! The on-disk CAS layout (`blobs/`, `bootstrap/`, `cache/`, `refs/`) is
//! driven via raw `std::fs`/path ops for ingest (`pull`/`gc`/`remove`). The
//! read path is additionally exposed as a 9P/VFS `Mount` — see
//! [`super::store_mount::RafsStoreMount`] (#652).
//!
//! Uses Nydus RAFS format for efficient image storage with:
//! - Chunk-level deduplication (across all images)
//! - Lazy loading (on-demand chunk fetch via Dragonfly P2P)
//! - ~80% storage savings vs traditional layers
//!
//! Architecture:
//! ```text
//! ManifestFetcher (HTTP)  →  nydus-storage Registry backend  →  rafs_builder (TarballBuilder)
//!      │                              │                              │
//!      └── OCI manifests only         └── Dragonfly P2P for blobs    └── RAFS bootstrap (.meta)
//! ```
//!
//! `pull()` downloads the OCI layer tarballs into the CAS and then converts them
//! in-process into a RAFS bootstrap (`.meta`) via [`rafs_builder`], so
//! [`RafsStore::bootstrap_path`] resolves to a real RAFS bootstrap consumable by
//! an in-process RAFS `FileSystem` (FS-B, #363) or nydusd (FS-B0, #366).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use nydus_api::RegistryConfig;
use nydus_storage::backend::registry::Registry;
use nydus_storage::backend::{BlobBackend, BlobBufReader};

use crate::config::ImageConfig;
use crate::error::{Result, WorkerError};

use super::{AuthConfig as GenAuthConfig, ImageSpec, ImageInfo, ImageStatusResult, FilesystemUsage, FilesystemIdentifier};
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
        auth: Option<&GenAuthConfig>,
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

        // Convert CRI auth (GenAuthConfig with String fields) to manifest auth (Option<String> fields)
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
            self.download_blob(&img_ref, &manifest.config.digest, &config_path, manifest_auth.as_ref())
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
                self.download_blob(&img_ref, &layer.digest, &layer_path, manifest_auth.as_ref())
                    .await?;
            } else {
                tracing::debug!(digest = %layer.digest, "Layer already cached");
            }
        }

        // 6. Generate image ID (SHA256 of config)
        let config_data = tokio::fs::read(&config_path).await?;
        let image_id = format!("sha256:{}", hex::encode(Sha256::digest(&config_data)));

        // 6b. Build the RAFS bootstrap (.meta) from the pulled OCI layer tarballs.
        //
        // FS-B0 (#366): the layers above were downloaded into the CAS, but RAFS /
        // nydusd consume a RAFS *bootstrap*, not raw OCI tarballs. Convert them
        // in-process via nydus-builder so `bootstrap_path()` resolves to a real
        // bootstrap. This MUST succeed; a missing/JSON bootstrap is a hard error.
        let layer_blob_paths: Vec<PathBuf> = manifest
            .layers
            .iter()
            .map(|l| self.blobs_dir.join(digest_to_filename(&l.digest)))
            .collect();
        let bootstrap_path = self.bootstrap_path(&image_id);
        let blobs_dir = self.blobs_dir.clone();
        let build_layers = layer_blob_paths.clone();
        let build_bootstrap = bootstrap_path.clone();
        tracing::info!(
            image = %image_ref,
            layers = %build_layers.len(),
            bootstrap = %build_bootstrap.display(),
            "Building RAFS bootstrap from OCI layers"
        );
        tokio::task::spawn_blocking(move || {
            super::rafs_builder::build_rafs_bootstrap(&build_layers, &blobs_dir, &build_bootstrap)
        })
        .await
        .map_err(|e| WorkerError::RafsError(format!("RAFS build task panicked: {e}")))?
        .map_err(|e| WorkerError::ImagePullFailed {
            image: image_ref.to_owned(),
            reason: format!("RAFS bootstrap build failed: {e}"),
        })?;

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
    ///
    /// # Runtime-drop safety (#737)
    ///
    /// The nydus-storage `Registry` backend and its blob reader own an
    /// **internal tokio runtime** (for their HTTP transport). tokio forbids
    /// dropping a runtime from within another runtime's context — and that
    /// context is thread-local, so it is present both inside an async task AND
    /// inside a `spawn_blocking` pool thread (the pool thread still carries the
    /// outer runtime handle). Constructing those nydus objects on a
    /// `spawn_blocking` thread and letting them drop there therefore panics with
    /// `Cannot drop a runtime in a context where blocking is not allowed` when
    /// `download_blob` is called from any async caller.
    ///
    /// To stay safe we run the **entire** nydus create + download + drop on a
    /// plain `std::thread` that has NO ambient tokio runtime, and ferry only the
    /// final plain `Result` back over a `oneshot`. No nydus object ever crosses
    /// back into the async context.
    async fn download_blob(
        &self,
        img_ref: &ImageReference,
        digest: &str,
        dest_path: &Path,
        auth: Option<&AuthConfig>,
    ) -> Result<()> {
        // Extract blob ID from digest (remove "sha256:" prefix)
        let blob_id = digest
            .strip_prefix("sha256:")
            .unwrap_or(digest).to_owned(); // Convert to owned String for 'static lifetime

        // Create nydus-storage Registry backend config
        let registry_config = self.create_registry_config(img_ref, &blob_id, auth);
        let dest_path = dest_path.to_path_buf();

        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("rafs-blob-download".to_owned())
            .spawn(move || {
                // ALL nydus objects (backend, reader, buf_reader) are created,
                // used, AND dropped inside this closure — on a thread with no
                // ambient tokio runtime — so their internal runtime drops off
                // any tokio context. Only the plain Result is sent back.
                let result = download_blob_blocking(&registry_config, &blob_id, &dest_path);
                let _ = tx.send(result);
            })
            .map_err(|e| {
                WorkerError::RafsError(format!("failed to spawn blob download thread: {e}"))
            })?;

        rx.await.map_err(|e| {
            WorkerError::RafsError(format!(
                "blob download thread exited without a result: {e}"
            ))
        })?
    }

    /// Create a nydus-storage RegistryConfig for a blob fetch.
    fn create_registry_config(
        &self,
        img_ref: &ImageReference,
        blob_id: &str,
        auth: Option<&AuthConfig>,
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

        // Set authentication if provided (both now use Option<String>)
        if let Some(auth) = auth {
            if let (Some(username), Some(password)) = (&auth.username, &auth.password) {
                if !username.is_empty() && !password.is_empty() {
                    use base64::Engine;
                    let credentials = format!("{}:{}", username, password);
                    config.auth = Some(
                        base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes()),
                    );
                }
            } else if let Some(token) = &auth.token {
                if !token.is_empty() {
                    config.registry_token = Some(token.clone());
                }
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
    pub async fn list(&self) -> Result<Vec<ImageInfo>> {
        let mut images = Vec::new();

        if self.refs_dir.exists() {
            for entry in std::fs::read_dir(&self.refs_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_symlink() {
                    let _tag = entry.file_name().to_string_lossy().to_string();
                    let metadata_path = std::fs::read_link(entry.path())?;

                    if let Ok(metadata_str) = std::fs::read_to_string(&metadata_path) {
                        if let Ok(metadata) = serde_json::from_str::<ImageMetadata>(&metadata_str) {
                            images.push(ImageInfo {
                                id: metadata.image_id.clone(),
                                repo_tags: vec![metadata.image_ref.clone()],
                                repo_digests: vec![metadata.config_digest.clone()],
                                size: self.calculate_image_size(&metadata),
                                uid: -1,
                                username: String::new(),
                                spec: ImageSpec {
                                    image: metadata.image_ref.clone(),
                                    ..Default::default()
                                },
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

        // Remove the RAFS bootstrap (.meta) built by FS-B0.
        let bootstrap_path = self.bootstrap_path(image_id);
        if bootstrap_path.exists() {
            tokio::fs::remove_file(&bootstrap_path).await?;
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
                let path = entry.path();
                match path.extension().and_then(|e| e.to_str()) {
                    Some("json") => {
                        if let Ok(metadata_str) = std::fs::read_to_string(&path) {
                            if let Ok(metadata) = serde_json::from_str::<ImageMetadata>(&metadata_str)
                            {
                                referenced.insert(digest_to_filename(&metadata.config_digest));
                                for layer in &metadata.layers {
                                    referenced.insert(digest_to_filename(layer));
                                }
                            }
                        }
                    }
                    // RAFS bootstraps reference data blobs by their RAFS blob
                    // hash (not OCI layer digests), so mark those as referenced
                    // too — otherwise GC would delete blobs a live bootstrap
                    // depends on.
                    Some("meta") => {
                        if let Ok(blob_ids) = super::rafs_builder::bootstrap_blob_ids(&path) {
                            for blob_id in blob_ids {
                                referenced.insert(blob_id);
                            }
                        }
                    }
                    _ => {}
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

    /// Sum on-disk bytes for each layer blob recorded in `metadata`.
    fn calculate_image_size(&self, metadata: &ImageMetadata) -> u64 {
        metadata
            .layers
            .iter()
            .map(|digest| {
                let path = self.blobs_dir.join(digest_to_filename(digest));
                std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
            })
            .sum()
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

    /// Get bootstrap (RAFS metadata) directory
    pub fn bootstrap_dir(&self) -> &PathBuf {
        &self.bootstrap_dir
    }

    /// Get refs (tag symlink) directory
    pub fn refs_dir(&self) -> &PathBuf {
        &self.refs_dir
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CRI ImageClient methods
    // ─────────────────────────────────────────────────────────────────────────

    /// List all images (CRI ImageClient aligned)
    pub async fn list_images(&self) -> Result<Vec<ImageInfo>> {
        self.list().await
    }

    /// Get image status (CRI ImageClient aligned)
    pub async fn image_status(&self, image_ref: &str, _verbose: bool) -> Result<ImageStatusResult> {
        if self.exists(image_ref) {
            let metadata = self.get_metadata(image_ref)?;
            Ok(ImageStatusResult {
                image: ImageInfo {
                    id: metadata.image_id.clone(),
                    repo_tags: vec![metadata.image_ref.clone()],
                    repo_digests: vec![metadata.config_digest.clone()],
                    size: self.calculate_image_size(&metadata),
                    uid: -1,
                    username: String::new(),
                    spec: ImageSpec {
                        image: metadata.image_ref.clone(),
                        ..Default::default()
                    },
                    pinned: false,
                },
                info: vec![],
            })
        } else {
            // Empty id = not found
            Ok(ImageStatusResult {
                image: ImageInfo::default(),
                info: vec![],
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
    pub async fn fs_info(&self) -> Result<Vec<FilesystemUsage>> {
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

        Ok(vec![FilesystemUsage {
            timestamp: chrono::Utc::now().timestamp(),
            fs_id: FilesystemIdentifier {
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

/// Download a single blob to `dest_path` using nydus-storage's `Registry`
/// backend, entirely synchronously.
///
/// This constructs and drops every nydus object (`Registry` backend, blob
/// reader, `BlobBufReader`) locally. It MUST be called from a thread with no
/// ambient tokio runtime — see [`RafsStore::download_blob`] and #737 — because
/// those objects own an internal tokio runtime that panics if dropped inside a
/// runtime context.
fn download_blob_blocking(
    registry_config: &RegistryConfig,
    blob_id: &str,
    dest_path: &Path,
) -> Result<()> {
    // Create the registry backend.
    let backend = Registry::new(registry_config, Some(blob_id))
        .map_err(|e| WorkerError::RafsError(format!("failed to create registry backend: {e:?}")))?;

    // Get a blob reader.
    let reader = backend
        .get_reader(blob_id)
        .map_err(|e| WorkerError::RafsError(format!("failed to get blob reader: {e:?}")))?;

    // Get blob size (we need to know the size for BlobBufReader).
    let blob_size = reader
        .blob_size()
        .map_err(|e| WorkerError::RafsError(format!("failed to get blob size: {e:?}")))?;

    // Use BlobBufReader for efficient buffered reading.
    let mut buf_reader = BlobBufReader::new(
        1024 * 1024, // 1MB buffer
        reader,
        0,
        blob_size,
    );

    // Create destination file.
    let mut file = std::fs::File::create(dest_path)
        .map_err(|e| WorkerError::IoError(format!("failed to create blob file: {e}")))?;

    // Copy data.
    std::io::copy(&mut buf_reader, &mut file)
        .map_err(|e| WorkerError::IoError(format!("failed to write blob: {e}")))?;

    // Sync to disk.
    file.sync_all()
        .map_err(|e| WorkerError::IoError(format!("failed to sync blob: {e}")))?;

    Ok(())
    // `backend` and `buf_reader` (which owns `reader`) drop here, on the caller's
    // (non-tokio) thread — off any runtime context.
}

/// Convert digest to filesystem-safe filename
///
/// `pub(crate)` so [`super::store_mount::RafsStoreMount`] (#652) can resolve a
/// `sha256:...`-style digest path component to the on-disk blob filename
/// using the same convention as the rest of the store.
pub(crate) fn digest_to_filename(digest: &str) -> String {
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

// ─────────────────────────────────────────────────────────────────────────────
// ImageStore trait impl + inventory registration (#646)
// ─────────────────────────────────────────────────────────────────────────────
//
// `RafsStore` already satisfies the `ImageStore` trait structurally (its public
// methods match the CRI image ops). This impl makes it dispatchable as
// `Arc<dyn ImageStore>` from `WorkerService` WITHOUT a cfg mirror — the trait
// surface is always compiled; only this impl (and the nydus deps it rests on)
// requires `oci-image`. A future non-RAFS image backend adds its own impl +
// `submit!` with zero changes to `WorkerService`.
//
// The inherent methods (e.g. `RafsStore::list_images`) are called via
// fully-qualified syntax because the trait methods share names — this avoids
// the recursion ambiguity without renaming the inherent API.

#[async_trait::async_trait]
impl crate::image::store_trait::ImageStore for RafsStore {
    async fn list_images(&self) -> anyhow::Result<Vec<crate::image::ImageInfo>> {
        RafsStore::list_images(self).await.map_err(Into::into)
    }

    async fn image_status(
        &self,
        image_ref: &str,
        verbose: bool,
    ) -> anyhow::Result<crate::image::ImageStatusResult> {
        RafsStore::image_status(self, image_ref, verbose)
            .await
            .map_err(Into::into)
    }

    async fn pull_with_auth(
        &self,
        image_ref: &str,
        auth: Option<&crate::image::AuthConfig>,
    ) -> anyhow::Result<String> {
        RafsStore::pull_with_auth(self, image_ref, auth)
            .await
            .map_err(Into::into)
    }

    async fn remove_image(&self, image_ref: &str) -> anyhow::Result<()> {
        RafsStore::remove_image(self, image_ref)
            .await
            .map_err(Into::into)
    }

    async fn fs_info(&self) -> anyhow::Result<Vec<crate::image::FilesystemUsage>> {
        RafsStore::fs_info(self).await.map_err(Into::into)
    }
}

inventory::submit! {
    crate::image::store_trait::ImageBackendRegistration {
        name: "rafs",
        construct: |config| {
            let store = RafsStore::new(config.clone()).map_err(anyhow::Error::from)?;
            Ok(std::sync::Arc::new(store) as std::sync::Arc<dyn crate::image::store_trait::ImageStore>)
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression tests (#737)
// ─────────────────────────────────────────────────────────────────────────────
//
// These exercise the runtime-drop-safety of the blob download path. Before the
// #737 fix, the nydus `Registry` backend (which owns an internal tokio runtime)
// was created on a `spawn_blocking` thread and then dropped inside the async
// `download_blob` context, panicking with:
//   "Cannot drop a runtime in a context where blocking is not allowed."
// Any of the tests below driving `download_blob` from a `#[tokio::test]` would
// therefore PANIC (and fail) before the fix, and pass after it — the nydus
// objects are now created and dropped on a plain `std::thread`.
//
// No reachable registry is required: we point the pull at an unreachable local
// address so the download fails fast with a network error *after* the nydus
// backend (the runtime-owning object) has been constructed and must be dropped.

#[cfg(test)]
mod runtime_drop_tests {
    use super::*;
    use crate::config::ImageConfig;

    /// Build a `RafsStore` rooted in a fresh tempdir.
    fn test_store(tmp: &std::path::Path) -> RafsStore {
        let config = ImageConfig {
            blobs_dir: tmp.join("blobs"),
            bootstrap_dir: tmp.join("bootstrap"),
            refs_dir: tmp.join("refs"),
            cache_dir: tmp.join("cache"),
            runtime_dir: tmp.join("run"),
            ..Default::default()
        };
        std::fs::create_dir_all(&config.blobs_dir).unwrap();
        RafsStore::new(config).unwrap()
    }

    /// An `ImageReference` whose host refuses connections immediately, so the
    /// blob download fails fast *after* the nydus backend is constructed.
    fn unreachable_ref() -> ImageReference {
        ImageReference {
            // 127.0.0.1:1 (a privileged port nothing listens on) → connection
            // refused, exercising the create+drop of the nydus backend without
            // needing the network or a real registry.
            host: "127.0.0.1:1".to_owned(),
            repository: "library/does-not-exist".to_owned(),
            reference: "latest".to_owned(),
            is_digest: false,
        }
    }

    /// Drive `download_blob` from a multi-threaded tokio runtime. Must return an
    /// error (unreachable registry) rather than panicking on the nydus runtime
    /// drop. Pre-#737 this panicked.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn download_blob_does_not_panic_multi_thread() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());
        let img_ref = unreachable_ref();
        let dest = store.blobs_dir().join("blob_out");

        let res = store
            .download_blob(&img_ref, "sha256:deadbeef", &dest, None)
            .await;

        // The point of the test is that we got HERE (a returned Result) instead
        // of panicking with "Cannot drop a runtime ...".
        assert!(
            res.is_err(),
            "expected an error pulling from an unreachable registry, got: {res:?}"
        );
    }

    /// Same, but from a current-thread runtime — the flavor most likely to trip
    /// the runtime-in-runtime drop rule.
    #[tokio::test(flavor = "current_thread")]
    async fn download_blob_does_not_panic_current_thread() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());
        let img_ref = unreachable_ref();
        let dest = store.blobs_dir().join("blob_out");

        let res = store
            .download_blob(&img_ref, "sha256:deadbeef", &dest, None)
            .await;

        assert!(
            res.is_err(),
            "expected an error pulling from an unreachable registry, got: {res:?}"
        );
    }
}
