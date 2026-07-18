//! Home PDS attachment client (issue #318).
//!
//! A host has exactly one home PDS.  Attachment uses the PDS's OAuth 2.1
//! RFC 8628 device-authorization endpoint and sender-binds the resulting
//! credential to the host's iroh transport identity with RFC 9449 DPoP.
//! The credential is stored in the configured secrets directory, never in the
//! regular configuration file.

use std::time::Duration;

use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signer as _, SigningKey};
use serde::{Deserialize, Serialize};
use tracing::info;
use url::{Host, Url};

use crate::{auth::identity_store, config::HyprConfig};

const HOME_PDS_SECRET: &str = "pds-home.json";
const DEVICE_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const DEFAULT_PDS_SCOPE: &str = "pds:attach";

/// Persisted PDS-scoped OAuth credential for this host.
///
/// This is intentionally a secret-file record: the access and refresh tokens
/// must not be serialized into `[oauth]` or printed by CLI commands.
#[derive(Clone, Serialize, Deserialize)]
pub struct HomePdsAttachment {
    pub pds_url: String,
    pub issuer: String,
    pub did: String,
    pub client_id: String,
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub scope: String,
    pub expires_in: Option<i64>,
}

#[derive(Deserialize)]
struct AuthorizationServerMetadata {
    issuer: String,
    registration_endpoint: String,
    device_authorization_endpoint: String,
    token_endpoint: String,
}

#[derive(Deserialize)]
struct RegistrationResponse {
    client_id: String,
}

#[derive(Deserialize)]
struct DeviceAuthorizationResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    #[serde(default = "default_poll_interval")]
    interval: u64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default = "default_token_type")]
    token_type: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    expires_in: Option<i64>,
}

fn default_poll_interval() -> u64 {
    5
}
fn default_token_type() -> String {
    "DPoP".to_owned()
}

fn attachment_scope(scope: Option<&str>) -> String {
    scope
        .map(str::to_owned)
        .unwrap_or_else(|| DEFAULT_PDS_SCOPE.to_owned())
}

/// Attach this host to `pds_url` using RFC 8628.
///
/// The PDS URL must be an HTTPS origin (except loopback HTTP for local test
/// deployments).  Before any network request, this checks the local secret
/// store so a host can never accidentally acquire credentials from two homes.
pub async fn handle_pds_join(
    config: &HyprConfig,
    pds_url: &str,
    scope: Option<&str>,
) -> Result<()> {
    let pds_url = canonical_pds_url(pds_url)?;
    let secrets_dir = identity_store::credentials_dir_for_config(Some(config))?;
    ensure_no_home_pds(&secrets_dir)?;

    let root_key = if let Some(key) = HyprConfig::node_signing_key_bypass()? {
        key
    } else {
        identity_store::load_or_generate_node_signing_key(&secrets_dir)?
    };
    // This is the same domain-separated key whose verifying key iroh exposes
    // as the node_id.  Its did:key is both the registered host identity and
    // the key used for DPoP possession proofs below.
    let iroh_key = hyprstream_rpc::node_identity::derive_purpose_key(
        &root_key,
        "hyprstream-iroh-transport-v1",
    );
    let did = hyprstream_crypto::did_key::ed25519_to_did_key(iroh_key.verifying_key().as_bytes());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("building PDS OAuth client")?;

    let metadata_url = endpoint_under_pds(&pds_url, ".well-known/oauth-authorization-server")?;
    let metadata = client
        .get(metadata_url)
        .send()
        .await
        .context("fetching PDS OAuth authorization-server metadata")?
        .error_for_status()
        .context("PDS OAuth authorization-server metadata request failed")?
        .json::<AuthorizationServerMetadata>()
        .await
        .context("PDS returned invalid OAuth authorization-server metadata")?;

    let issuer = canonical_pds_url(&metadata.issuer)
        .context("PDS OAuth metadata contains an invalid issuer")?;
    if issuer != pds_url {
        anyhow::bail!(
            "PDS OAuth metadata issuer '{}' does not match requested PDS origin '{}'",
            issuer,
            pds_url
        );
    }
    let registration_endpoint = pds_endpoint(
        &metadata.registration_endpoint,
        "registration_endpoint",
        &pds_url,
    )?;
    let device_endpoint = pds_endpoint(
        &metadata.device_authorization_endpoint,
        "device_authorization_endpoint",
        &pds_url,
    )?;
    let token_endpoint = pds_endpoint(&metadata.token_endpoint, "token_endpoint", &pds_url)?;

    // RFC 7591 registration is necessary because the existing device endpoint
    // correctly rejects unknown clients.  The redirect URI is unused by RFC
    // 8628, but RFC 7591 registration still requires one.
    let registration = client
        .post(registration_endpoint)
        .json(&serde_json::json!({
            "redirect_uris": ["http://127.0.0.1:0/pds-attach"],
            "client_name": "HyprStream host PDS attachment",
            "hyprstream_node_did": did.clone(),
            "grant_types": [DEVICE_CODE_GRANT, "refresh_token"],
            "token_endpoint_auth_method": "none"
        }))
        .send()
        .await
        .context("registering this host as an OAuth device client with the PDS")?
        .error_for_status()
        .context("PDS rejected OAuth device-client registration")?
        .json::<RegistrationResponse>()
        .await
        .context("PDS returned an invalid OAuth client registration")?;

    let requested_scope = attachment_scope(scope);
    let device_form = vec![
        ("client_id", registration.client_id.as_str()),
        ("resource", pds_url.as_str()),
        ("scope", requested_scope.as_str()),
    ];
    let device = client
        .post(device_endpoint)
        .form(&device_form)
        .send()
        .await
        .context("starting PDS device authorization")?
        .error_for_status()
        .context("PDS rejected device authorization")?
        .json::<DeviceAuthorizationResponse>()
        .await
        .context("PDS returned an invalid device authorization response")?;

    #[allow(clippy::print_stdout)]
    {
        println!("Host DID: {did}");
        println!("Open this URL and authorize this host:");
        println!(
            "{}",
            device
                .verification_uri_complete
                .as_deref()
                .unwrap_or(&device.verification_uri)
        );
        println!("User code: {}", device.user_code);
    }
    info!(host_did = %did, "PDS device authorization started");

    let token = poll_device_token(
        &client,
        &token_endpoint,
        &registration.client_id,
        &device.device_code,
        device.interval.max(1),
        &iroh_key,
    )
    .await?;
    if !token.token_type.eq_ignore_ascii_case("DPoP") {
        anyhow::bail!(
            "PDS issued token_type '{}'; home-PDS credentials must be DPoP sender-constrained",
            token.token_type
        );
    }
    let refresh_token = token.refresh_token.context(
        "PDS did not issue a refresh token; refusing to persist a non-renewable home-PDS attachment",
    )?;

    let attachment = HomePdsAttachment {
        pds_url,
        issuer,
        did,
        client_id: registration.client_id,
        access_token: token.access_token,
        refresh_token,
        token_type: token.token_type,
        scope: token.scope,
        expires_in: token.expires_in,
    };
    let encoded = serde_json::to_vec(&attachment).context("serializing PDS attachment")?;
    if !identity_store::write_secret_exclusive(&secrets_dir, HOME_PDS_SECRET, &encoded)? {
        anyhow::bail!(
            "this host was attached to a home PDS while authorization was in progress; refusing to replace it"
        );
    }

    #[allow(clippy::print_stdout)]
    {
        println!(
            "Attached this host ({}) to home PDS {}.",
            attachment.did, attachment.pds_url
        );
    }
    info!(
        host_did = %attachment.did,
        pds_url = %attachment.pds_url,
        "Attached this host to its home PDS"
    );
    Ok(())
}

async fn poll_device_token(
    client: &reqwest::Client,
    token_endpoint: &Url,
    client_id: &str,
    device_code: &str,
    mut interval: u64,
    dpop_key: &SigningKey,
) -> Result<TokenResponse> {
    loop {
        tokio::time::sleep(Duration::from_secs(interval)).await;
        match exchange_device_code(
            client,
            token_endpoint,
            client_id,
            device_code,
            dpop_key,
            None,
        )
        .await?
        {
            DevicePoll::Token(token) => return Ok(token),
            DevicePoll::Pending => continue,
            DevicePoll::SlowDown => {
                interval = interval.saturating_add(5);
                continue;
            }
        }
    }
}

enum DevicePoll {
    Token(TokenResponse),
    Pending,
    SlowDown,
}

async fn exchange_device_code(
    client: &reqwest::Client,
    token_endpoint: &Url,
    client_id: &str,
    device_code: &str,
    dpop_key: &SigningKey,
    nonce: Option<&str>,
) -> Result<DevicePoll> {
    let mut nonce = nonce.map(str::to_owned);
    loop {
        let proof = dpop_proof(dpop_key, "POST", token_endpoint, nonce.as_deref())?;
        let response = client
            .post(token_endpoint.clone())
            .header("DPoP", proof)
            .form(&[
                ("grant_type", DEVICE_CODE_GRANT),
                ("client_id", client_id),
                ("device_code", device_code),
            ])
            .send()
            .await
            .context("polling PDS OAuth token endpoint")?;

        if response.status().is_success() {
            return response
                .json::<TokenResponse>()
                .await
                .context("PDS returned an invalid OAuth token response")
                .map(DevicePoll::Token);
        }

        let dpop_nonce = response
            .headers()
            .get("DPoP-Nonce")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let status = response.status();
        let body: serde_json::Value = response.json().await.unwrap_or_default();
        let error = body
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown_error");
        match error {
            "authorization_pending" => return Ok(DevicePoll::Pending),
            "slow_down" => return Ok(DevicePoll::SlowDown),
            "use_dpop_nonce" => {
                // A nonce retry is not a polling iteration; it has a fresh JTI
                // and does not consume the server's device-code poll interval.
                nonce = Some(
                    dpop_nonce.context("PDS requested DPoP nonce without sending DPoP-Nonce")?,
                );
            }
            _ => anyhow::bail!(
                "PDS device authorization failed ({status}): {}",
                body.get("error_description")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(error)
            ),
        }
    }
}

fn dpop_proof(
    key: &SigningKey,
    method: &str,
    endpoint: &Url,
    nonce: Option<&str>,
) -> Result<String> {
    let htu = endpoint_without_query(endpoint)?;
    let x = URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes());
    let header = serde_json::json!({
        "typ": "dpop+jwt",
        "alg": "EdDSA",
        "jwk": { "kty": "OKP", "crv": "Ed25519", "x": x },
    });
    let mut claims = serde_json::json!({
        "jti": uuid::Uuid::new_v4().to_string(),
        "htm": method,
        "htu": htu,
        "iat": chrono::Utc::now().timestamp(),
    });
    if let Some(nonce) = nonce {
        claims["nonce"] = serde_json::Value::String(nonce.to_owned());
    }
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header)?);
    let claims = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims)?);
    let signed = format!("{header}.{claims}");
    let signature = key.sign(signed.as_bytes());
    Ok(format!(
        "{signed}.{}",
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    ))
}

fn ensure_no_home_pds(secrets_dir: &std::path::Path) -> Result<()> {
    if let Some(bytes) = identity_store::read_secret(secrets_dir, HOME_PDS_SECRET)? {
        let attachment: HomePdsAttachment = serde_json::from_slice(&bytes).context(
            "existing PDS attachment credential is corrupt; refusing to attach another home PDS",
        )?;
        anyhow::bail!(
            "this host is already attached to home PDS '{}'; only one home PDS is allowed",
            attachment.pds_url
        );
    }
    Ok(())
}

fn is_loopback_host(url: &Url) -> bool {
    match url.host() {
        Some(Host::Domain("localhost")) => true,
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        _ => false,
    }
}

fn canonical_pds_url(value: &str) -> Result<String> {
    let mut url = Url::parse(value).context("PDS URL must be an absolute URL")?;
    if url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        anyhow::bail!("PDS URL must not contain credentials, a query, or a fragment");
    }
    let is_loopback = is_loopback_host(&url);
    if url.scheme() != "https" && !(url.scheme() == "http" && is_loopback) {
        anyhow::bail!("PDS URL must use HTTPS (HTTP is allowed only for loopback development)");
    }
    if url.host_str().is_none() {
        anyhow::bail!("PDS URL must include a host");
    }
    url.set_query(None);
    url.set_fragment(None);
    if url.path() != "/" {
        anyhow::bail!("PDS URL must be an origin without a path");
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn endpoint_under_pds(pds_url: &str, path: &str) -> Result<Url> {
    let base = Url::parse(&format!("{}/", pds_url.trim_end_matches('/')))?;
    base.join(path).context("building PDS metadata endpoint")
}

fn pds_endpoint(value: &str, name: &str, pds_url: &str) -> Result<Url> {
    let url = secure_endpoint(value, name)?;
    let origin = canonical_pds_url(url.origin().ascii_serialization().as_str())?;
    if origin != pds_url {
        anyhow::bail!(
            "PDS metadata {name} origin '{}' does not match requested PDS origin '{}'",
            origin,
            pds_url
        );
    }
    Ok(url)
}

fn secure_endpoint(value: &str, name: &str) -> Result<Url> {
    let url =
        Url::parse(value).with_context(|| format!("PDS metadata {name} is not an absolute URL"))?;
    let is_loopback = is_loopback_host(&url);
    if url.scheme() != "https" && !(url.scheme() == "http" && is_loopback) {
        anyhow::bail!("PDS metadata {name} must use HTTPS (or loopback HTTP)");
    }
    if url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
    {
        anyhow::bail!(
            "PDS metadata {name} must be a complete endpoint URL without credentials, query, or fragment"
        );
    }
    Ok(url)
}

fn endpoint_without_query(endpoint: &Url) -> Result<String> {
    let mut endpoint = endpoint.clone();
    endpoint.set_query(None);
    endpoint.set_fragment(None);
    if endpoint.cannot_be_a_base() {
        anyhow::bail!("invalid OAuth endpoint for DPoP proof");
    }
    Ok(endpoint.as_str().trim_end_matches('/').to_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn canonical_pds_requires_secure_origin() {
        assert_eq!(
            canonical_pds_url("https://PDS.example/").unwrap(),
            "https://pds.example"
        );
        assert_eq!(
            canonical_pds_url("http://127.0.0.1:6791/").unwrap(),
            "http://127.0.0.1:6791"
        );
        assert!(canonical_pds_url("http://pds.example").is_err());
        assert!(canonical_pds_url("https://pds.example/path").is_err());
        assert!(canonical_pds_url("https://pds.example/?token=x").is_err());
        assert!(secure_endpoint(
            "https://user:secret@pds.example/oauth/token",
            "token_endpoint",
        )
        .is_err());
        assert_eq!(attachment_scope(None), DEFAULT_PDS_SCOPE);
    }

    #[test]
    fn dpop_proof_is_bound_to_the_endpoint_and_key() {
        let key = SigningKey::generate(&mut rand::rngs::OsRng);
        let endpoint = Url::parse("https://pds.example/oauth/token?ignored=true").unwrap();
        let proof = dpop_proof(&key, "POST", &endpoint, Some("nonce")).unwrap();
        let verified = crate::services::oauth::dpop::verify_dpop_proof(
            &proof,
            "POST",
            "https://pds.example/oauth/token",
            None,
        )
        .unwrap();
        assert_eq!(
            verified.public_key_bytes(),
            Some(*key.verifying_key().as_bytes())
        );
        assert_eq!(verified.nonce.as_deref(), Some("nonce"));
    }

    #[test]
    fn existing_attachment_blocks_second_home() {
        let dir = tempfile::tempdir().unwrap();
        let attachment = HomePdsAttachment {
            pds_url: "https://one.example".to_owned(),
            issuer: "https://one.example".to_owned(),
            did: "did:key:ztest".to_owned(),
            client_id: "client".to_owned(),
            access_token: "access".to_owned(),
            refresh_token: "refresh".to_owned(),
            token_type: "DPoP".to_owned(),
            scope: String::new(),
            expires_in: None,
        };
        identity_store::write_secret(
            dir.path(),
            HOME_PDS_SECRET,
            &serde_json::to_vec(&attachment).unwrap(),
        )
        .unwrap();
        let persisted: HomePdsAttachment = serde_json::from_slice(
            &identity_store::read_secret(dir.path(), HOME_PDS_SECRET)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(persisted.pds_url, attachment.pds_url);
        assert_eq!(persisted.did, attachment.did);
        assert!(ensure_no_home_pds(dir.path()).is_err());
    }
}
