//! Minimal OCI manifest fetcher for image metadata.
//!
//! This module only handles manifest fetching via HTTP. All blob operations
//! go through nydus-storage's backend-registry for Dragonfly P2P compatibility.

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderValue, ACCEPT, AUTHORIZATION, WWW_AUTHENTICATE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// OCI media types for manifest negotiation.
#[allow(dead_code)]
pub mod media_types {
    pub const OCI_MANIFEST: &str = "application/vnd.oci.image.manifest.v1+json";
    pub const OCI_INDEX: &str = "application/vnd.oci.image.index.v1+json";
    pub const DOCKER_MANIFEST: &str = "application/vnd.docker.distribution.manifest.v2+json";
    pub const DOCKER_MANIFEST_LIST: &str =
        "application/vnd.docker.distribution.manifest.list.v2+json";
}

/// Parsed OCI image reference.
#[derive(Debug, Clone)]
pub struct ImageReference {
    /// Registry host (e.g., "registry-1.docker.io")
    pub host: String,
    /// Repository path (e.g., "library/alpine")
    pub repository: String,
    /// Tag or digest reference (e.g., "latest" or "sha256:...")
    pub reference: String,
    /// Whether reference is a digest
    pub is_digest: bool,
}

impl ImageReference {
    /// Parse an image reference string.
    ///
    /// Handles formats like:
    /// - `alpine` -> docker.io/library/alpine:latest
    /// - `alpine:3.18` -> docker.io/library/alpine:3.18
    /// - `myrepo/myimage:tag` -> docker.io/myrepo/myimage:tag
    /// - `ghcr.io/owner/repo:tag` -> ghcr.io/owner/repo:tag
    /// - `registry.example.com/foo/bar@sha256:...` -> registry.example.com/foo/bar@sha256:...
    pub fn parse(image: &str) -> Result<Self> {
        let image = image.trim();
        if image.is_empty() {
            return Err(anyhow!("empty image reference"));
        }

        // Check for digest reference
        let (image_part, reference, is_digest) = if let Some(idx) = image.rfind('@') {
            let (img, digest) = image.split_at(idx);
            (img.to_owned(), digest[1..].to_string(), true)
        } else if let Some(idx) = image.rfind(':') {
            // Check if this colon is part of a port number (host:port/repo)
            let before_colon = &image[..idx];
            if before_colon.contains('/') || !before_colon.contains('.') {
                // It's a tag separator
                let (img, tag) = image.split_at(idx);
                (img.to_owned(), tag[1..].to_string(), false)
            } else {
                // It's a port separator, use default tag
                (image.to_owned(), "latest".to_owned(), false)
            }
        } else {
            (image.to_owned(), "latest".to_owned(), false)
        };

        // Parse host and repository
        let (host, repository) = if let Some(first_slash) = image_part.find('/') {
            let potential_host = &image_part[..first_slash];

            // Check if this is a hostname (contains . or : or is "localhost")
            if potential_host.contains('.')
                || potential_host.contains(':')
                || potential_host == "localhost"
            {
                let host = potential_host.to_owned();
                let repo = image_part[first_slash + 1..].to_string();
                (host, repo)
            } else {
                // No explicit host, use Docker Hub
                ("docker.io".to_owned(), image_part)
            }
        } else {
            // Simple name like "alpine" -> docker.io/library/alpine
            (
                "docker.io".to_owned(),
                format!("library/{image_part}"),
            )
        };

        // Docker Hub's actual registry host
        let host = if host == "docker.io" {
            "registry-1.docker.io".to_owned()
        } else {
            host
        };

        Ok(Self {
            host,
            repository,
            reference,
            is_digest,
        })
    }

    /// Get the manifest URL for this image.
    pub fn manifest_url(&self) -> String {
        format!(
            "https://{}/v2/{}/manifests/{}",
            self.host, self.repository, self.reference
        )
    }

    /// Get the blob URL for a given digest.
    pub fn blob_url(&self, digest: &str) -> String {
        format!(
            "https://{}/v2/{}/blobs/{}",
            self.host, self.repository, digest
        )
    }
}

/// OCI Image Manifest (single-platform).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub media_type: Option<String>,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// OCI Image Index (multi-platform manifest list).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciIndex {
    pub schema_version: u32,
    #[serde(default)]
    pub media_type: Option<String>,
    pub manifests: Vec<ManifestDescriptor>,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Content descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Manifest descriptor with platform info.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestDescriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default)]
    pub platform: Option<Platform>,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Platform specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Platform {
    pub architecture: String,
    pub os: String,
    #[serde(default, rename = "os.version")]
    pub os_version: Option<String>,
    #[serde(default, rename = "os.features")]
    pub os_features: Option<Vec<String>>,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub features: Option<Vec<String>>,
}

/// Result of fetching a manifest - could be single or multi-platform.
#[derive(Debug, Clone)]
pub enum ManifestResult {
    /// Single-platform manifest
    Manifest(OciManifest),
    /// Multi-platform index
    Index(OciIndex),
}

/// Authentication configuration.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
}

/// Minimal manifest fetcher using reqwest.
pub struct ManifestFetcher {
    client: Client,
    /// Cached bearer tokens per realm
    tokens: parking_lot::RwLock<HashMap<String, String>>,
}

impl ManifestFetcher {
    /// Create a new manifest fetcher.
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("failed to create HTTP client")?;

        Ok(Self {
            client,
            tokens: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    /// Fetch manifest for an image reference.
    pub async fn fetch(
        &self,
        image_ref: &ImageReference,
        auth: Option<&AuthConfig>,
    ) -> Result<ManifestResult> {
        let url = image_ref.manifest_url();

        // Try with cached token first
        if let Some(token) = self.get_cached_token(&image_ref.host) {
            match self.fetch_with_token(&url, &token).await {
                Ok(result) => return Ok(result),
                Err(_) => {
                    // Token might be expired, clear it
                    self.tokens.write().remove(&image_ref.host);
                }
            }
        }

        // Try anonymous first
        match self.fetch_anonymous(&url).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                tracing::debug!("anonymous fetch failed: {}, trying auth", e);
            }
        }

        // Need to authenticate
        let token = self.authenticate(image_ref, auth).await?;
        self.tokens
            .write()
            .insert(image_ref.host.clone(), token.clone());
        self.fetch_with_token(&url, &token).await
    }

    /// Fetch a specific platform manifest from an index.
    pub async fn fetch_platform_manifest(
        &self,
        image_ref: &ImageReference,
        digest: &str,
        auth: Option<&AuthConfig>,
    ) -> Result<OciManifest> {
        let url = format!(
            "https://{}/v2/{}/manifests/{}",
            image_ref.host, image_ref.repository, digest
        );

        // Try with cached token
        if let Some(token) = self.get_cached_token(&image_ref.host) {
            if let Ok(result) = self.fetch_manifest_with_token(&url, &token).await {
                return Ok(result);
            }
        }

        // Need to authenticate
        let token = self.authenticate(image_ref, auth).await?;
        self.fetch_manifest_with_token(&url, &token).await
    }

    fn get_cached_token(&self, host: &str) -> Option<String> {
        self.tokens.read().get(host).cloned()
    }

    async fn fetch_anonymous(&self, url: &str) -> Result<ManifestResult> {
        let resp = self
            .client
            .get(url)
            .header(ACCEPT, accept_header())
            .send()
            .await?;

        if resp.status().is_success() {
            self.parse_manifest_response(resp).await
        } else {
            Err(anyhow!(
                "anonymous fetch failed with status {}",
                resp.status()
            ))
        }
    }

    async fn fetch_with_token(&self, url: &str, token: &str) -> Result<ManifestResult> {
        let resp = self
            .client
            .get(url)
            .header(ACCEPT, accept_header())
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .send()
            .await?;

        if resp.status().is_success() {
            self.parse_manifest_response(resp).await
        } else {
            Err(anyhow!("fetch failed with status {}", resp.status()))
        }
    }

    async fn fetch_manifest_with_token(&self, url: &str, token: &str) -> Result<OciManifest> {
        let resp = self
            .client
            .get(url)
            .header(
                ACCEPT,
                format!(
                    "{}, {}",
                    media_types::OCI_MANIFEST,
                    media_types::DOCKER_MANIFEST
                ),
            )
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .send()
            .await?;

        if resp.status().is_success() {
            resp.json()
                .await
                .context("failed to parse manifest JSON")
        } else {
            Err(anyhow!("fetch manifest failed with status {}", resp.status()))
        }
    }

    async fn parse_manifest_response(&self, resp: reqwest::Response) -> Result<ManifestResult> {
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_default();

        let body = resp.text().await.context("failed to read response body")?;

        // Determine if it's an index or manifest based on content type or structure
        if content_type.contains("index")
            || content_type.contains("manifest.list")
            || body.contains("\"manifests\"")
        {
            let index: OciIndex =
                serde_json::from_str(&body).context("failed to parse index JSON")?;
            Ok(ManifestResult::Index(index))
        } else {
            let manifest: OciManifest =
                serde_json::from_str(&body).context("failed to parse manifest JSON")?;
            Ok(ManifestResult::Manifest(manifest))
        }
    }

    /// Authenticate with the registry and get a bearer token.
    async fn authenticate(
        &self,
        image_ref: &ImageReference,
        auth: Option<&AuthConfig>,
    ) -> Result<String> {
        // First, make an unauthenticated request to get the WWW-Authenticate header
        let url = image_ref.manifest_url();
        let resp = self
            .client
            .get(&url)
            .header(ACCEPT, accept_header())
            .send()
            .await?;

        if resp.status() == 401 {
            // Parse WWW-Authenticate header
            let www_auth = resp
                .headers()
                .get(WWW_AUTHENTICATE)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| anyhow!("missing WWW-Authenticate header"))?;

            let auth_params = parse_www_authenticate(www_auth)?;
            let realm = auth_params
                .get("realm")
                .ok_or_else(|| anyhow!("missing realm in WWW-Authenticate"))?;

            // Build token request URL
            let mut token_url = reqwest::Url::parse(realm).context("invalid realm URL")?;
            {
                let mut query = token_url.query_pairs_mut();
                if let Some(service) = auth_params.get("service") {
                    query.append_pair("service", service);
                }
                if let Some(scope) = auth_params.get("scope") {
                    query.append_pair("scope", scope);
                } else {
                    // Default scope for pulling
                    query.append_pair(
                        "scope",
                        &format!("repository:{}:pull", image_ref.repository),
                    );
                }
            }

            // Make token request
            let mut token_req = self.client.get(token_url);
            if let Some(auth_config) = auth {
                if let (Some(user), Some(pass)) = (&auth_config.username, &auth_config.password) {
                    token_req = token_req.basic_auth(user, Some(pass));
                } else if let Some(token) = &auth_config.token {
                    token_req = token_req.bearer_auth(token);
                }
            }

            let token_resp = token_req.send().await?;
            if !token_resp.status().is_success() {
                return Err(anyhow!(
                    "token request failed with status {}",
                    token_resp.status()
                ));
            }

            let token_data: TokenResponse = token_resp.json().await?;
            Ok(token_data.token.or(token_data.access_token).ok_or_else(|| {
                anyhow!("no token in response")
            })?)
        } else if resp.status().is_success() {
            // No auth needed
            Err(anyhow!("no authentication required"))
        } else {
            Err(anyhow!(
                "unexpected status {} during auth check",
                resp.status()
            ))
        }
    }
}


/// Token response from registry auth server.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    token: Option<String>,
    access_token: Option<String>,
}

/// Build the Accept header for manifest requests.
fn accept_header() -> HeaderValue {
    HeaderValue::from_static(
        "application/vnd.oci.image.manifest.v1+json, \
         application/vnd.oci.image.index.v1+json, \
         application/vnd.docker.distribution.manifest.v2+json, \
         application/vnd.docker.distribution.manifest.list.v2+json",
    )
}

/// Parse WWW-Authenticate header.
fn parse_www_authenticate(header: &str) -> Result<HashMap<String, String>> {
    let mut params = HashMap::new();

    // Format: Bearer realm="...",service="...",scope="..."
    let header = header.trim_start_matches("Bearer ");
    for part in header.split(',') {
        let part = part.trim();
        if let Some(eq_idx) = part.find('=') {
            let key = part[..eq_idx].trim().to_owned();
            let value = part[eq_idx + 1..].trim().trim_matches('"').to_owned();
            params.insert(key, value);
        }
    }

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_image() -> Result<()> {
        let ref_ = ImageReference::parse("alpine")?;
        assert_eq!(ref_.host, "registry-1.docker.io");
        assert_eq!(ref_.repository, "library/alpine");
        assert_eq!(ref_.reference, "latest");
        assert!(!ref_.is_digest);
        Ok(())
    }

    #[test]
    fn test_parse_tagged_image() -> Result<()> {
        let ref_ = ImageReference::parse("alpine:3.18")?;
        assert_eq!(ref_.host, "registry-1.docker.io");
        assert_eq!(ref_.repository, "library/alpine");
        assert_eq!(ref_.reference, "3.18");
        Ok(())
    }

    #[test]
    fn test_parse_user_repo() -> Result<()> {
        let ref_ = ImageReference::parse("myuser/myimage:v1")?;
        assert_eq!(ref_.host, "registry-1.docker.io");
        assert_eq!(ref_.repository, "myuser/myimage");
        assert_eq!(ref_.reference, "v1");
        Ok(())
    }

    #[test]
    fn test_parse_custom_registry() -> Result<()> {
        let ref_ = ImageReference::parse("ghcr.io/owner/repo:tag")?;
        assert_eq!(ref_.host, "ghcr.io");
        assert_eq!(ref_.repository, "owner/repo");
        assert_eq!(ref_.reference, "tag");
        Ok(())
    }

    #[test]
    fn test_parse_digest() -> Result<()> {
        let ref_ = ImageReference::parse("alpine@sha256:abc123")?;
        assert_eq!(ref_.host, "registry-1.docker.io");
        assert_eq!(ref_.repository, "library/alpine");
        assert_eq!(ref_.reference, "sha256:abc123");
        assert!(ref_.is_digest);
        Ok(())
    }

    #[test]
    fn test_manifest_url() -> Result<()> {
        let ref_ = ImageReference::parse("alpine:latest")?;
        assert_eq!(
            ref_.manifest_url(),
            "https://registry-1.docker.io/v2/library/alpine/manifests/latest"
        );
        Ok(())
    }

    #[test]
    fn test_parse_www_authenticate() -> Result<()> {
        let header = r#"Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:library/alpine:pull""#;
        let params = parse_www_authenticate(header)?;
        assert_eq!(&params["realm"], "https://auth.docker.io/token");
        assert_eq!(&params["service"], "registry.docker.io");
        assert_eq!(
            &params["scope"],
            "repository:library/alpine:pull"
        );
        Ok(())
    }
}
