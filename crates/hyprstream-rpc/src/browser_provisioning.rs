//! Accepted-current browser/WebTransport recipient provisioning.
//!
//! The wire document in this module is public discovery material, not an
//! authority token. Production documents are projected by Discovery only from
//! its opaque checkpoint-bound `ResolvedService`. Browser consumers validate
//! the complete context and build per-resolution stores from the result; they
//! never install the material in a process-global trust store.

use std::sync::Arc;

use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};

use crate::crypto::hybrid_kem::{KemTrustStore, KeyedKemTrustStore, RecipientPublic, SuiteId};
use crate::crypto::VerifyingKey;
use crate::envelope::{KeyedPqTrustStore, PqTrustStore};

pub const BROWSER_PROVISIONING_VERSION: &str = "hyprstream.browser-provisioning.v1";
/// WebTransport CONNECT path whose accept boundary requires browser provisioning.
pub const BROWSER_RPC_PATH: &str = "/hyprstream/browser-rpc";
pub const MAX_PROVISIONING_BYTES: usize = 32 * 1024;
pub const MAX_PROVISIONING_LIFETIME_MS: i64 = 60_000;
pub const MAX_BROWSER_BINDING_BYTES: usize = 4096;
pub const MAX_BROWSER_EXTENSION_BYTES: usize = 8 * 1024;
pub const MAX_BROWSER_APPLICATION_BYTES: usize = 4 * 1024 * 1024;
const MAX_BROWSER_CERTIFICATE_HASHES: usize = 16;
const BROWSER_BOUND_PAYLOAD_MAGIC: &[u8; 16] = b"HYPR-BROWSER-V1\0";
const BROWSER_EXTENSION_DOMAIN: &str = "hyprstream.browser-bound-request.v1";
const BROWSER_REQUEST_DOMAIN: &[u8] = b"hyprstream.browser-request-digest.v1\0";
const PROJECTION_SCHEMA_ID: u64 = 0x6879_7072_6270_7631;
const PROJECTION_TYPE_ID: u64 = 0x6272_6f77_7365_7231;

/// Carrier policy selected before any browser dial.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserCarrierProfile {
    /// A Hyprstream-owned endpoint. Transport PQ is required independently of
    /// the mandatory application HyKEM envelope.
    OwnedHybridWebTransport,
    /// A stock public relay. Its WebTransport hop is explicitly classical and
    /// untrusted; only opaque encrypted Objects may traverse it.
    StandardPublicRelay,
}

impl BrowserCarrierProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OwnedHybridWebTransport => "owned-hybrid-webtransport",
            Self::StandardPublicRelay => "standard-public-relay",
        }
    }
}

/// Route role bound into the provisioning context.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserRouteRole {
    Origin,
    Relay,
}

/// Honest transport-security classification. This is kept separate from the
/// application HyKEM requirement so a classical relay cannot raise assurance.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserTransportSecurity {
    OwnedHybridRequired,
    ClassicalUntrusted,
}

/// Exact context requested by a browser consumer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserProvisioningRequest {
    pub service_name: String,
    pub capability: String,
    pub scope: String,
    pub carrier_profile: BrowserCarrierProfile,
}

impl BrowserProvisioningRequest {
    pub fn new(
        service_name: impl Into<String>,
        capability: impl Into<String>,
        scope: impl Into<String>,
        carrier_profile: BrowserCarrierProfile,
    ) -> Result<Self> {
        let request = Self {
            service_name: service_name.into(),
            capability: capability.into(),
            scope: scope.into(),
            carrier_profile,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        validate_service_name(&self.service_name)?;
        validate_bounded_token(&self.capability, "capability", 128)?;
        validate_bounded_token(&self.scope, "scope", 512)?;
        if self.carrier_profile == BrowserCarrierProfile::StandardPublicRelay {
            anyhow::ensure!(
                !self.scope.contains('*') && !self.scope.ends_with('/'),
                "standard-public-relay requires an exact encrypted Object scope"
            );
        }
        Ok(())
    }
}

/// JSON document returned by the production provisioning endpoint.
///
/// Byte strings use canonical unpadded base64url. Deserialization rejects
/// duplicate and unknown fields through Serde's struct visitor.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserProvisioningDocument {
    pub version: String,
    pub service_name: String,
    pub service_did: String,
    pub service_origin: String,
    pub webtransport_url: String,
    pub capability: String,
    pub scope: String,
    pub carrier_profile: BrowserCarrierProfile,
    pub route_role: BrowserRouteRole,
    pub transport_security: BrowserTransportSecurity,
    pub response_key_id: String,
    pub response_ed25519: String,
    pub response_ml_dsa65: String,
    pub request_kem_key_id: String,
    pub request_kem_recipient: String,
    pub accepted_state_digest: String,
    pub accepted_state_epoch: u64,
    pub expires_at_unix_ms: i64,
    pub certificate_hashes: Vec<String>,
    pub application_hybrid_required: bool,
    pub encrypted_objects_required: bool,
    pub projection_signer_ed25519: String,
    pub projection_signer_ml_dsa65: String,
    pub projection_signature: String,
}

/// Raw public values used by the trusted server-side projection.
pub struct BrowserProvisioningMaterial {
    pub service_name: String,
    pub service_did: String,
    pub service_origin: String,
    pub webtransport_url: String,
    pub capability: String,
    pub scope: String,
    pub carrier_profile: BrowserCarrierProfile,
    pub route_role: BrowserRouteRole,
    pub transport_security: BrowserTransportSecurity,
    pub response_key_id: String,
    pub response_ed25519: [u8; 32],
    pub response_ml_dsa65: Vec<u8>,
    pub request_kem_key_id: String,
    pub request_kem_recipient: RecipientPublic,
    pub accepted_state_digest: [u8; 64],
    pub accepted_state_epoch: u64,
    pub expires_at_unix_ms: i64,
    pub certificate_hashes: Vec<[u8; 32]>,
    pub encrypted_objects_required: bool,
}

impl BrowserProvisioningDocument {
    pub fn from_material(material: BrowserProvisioningMaterial) -> Self {
        Self {
            version: BROWSER_PROVISIONING_VERSION.to_owned(),
            service_name: material.service_name,
            service_did: material.service_did,
            service_origin: material.service_origin,
            webtransport_url: material.webtransport_url,
            capability: material.capability,
            scope: material.scope,
            carrier_profile: material.carrier_profile,
            route_role: material.route_role,
            transport_security: material.transport_security,
            response_key_id: material.response_key_id,
            response_ed25519: URL_SAFE_NO_PAD.encode(material.response_ed25519),
            response_ml_dsa65: URL_SAFE_NO_PAD.encode(material.response_ml_dsa65),
            request_kem_key_id: material.request_kem_key_id,
            request_kem_recipient: URL_SAFE_NO_PAD.encode(material.request_kem_recipient.encode()),
            accepted_state_digest: URL_SAFE_NO_PAD.encode(material.accepted_state_digest),
            accepted_state_epoch: material.accepted_state_epoch,
            expires_at_unix_ms: material.expires_at_unix_ms,
            certificate_hashes: {
                let mut hashes = material.certificate_hashes;
                hashes.sort();
                hashes.dedup();
                hashes
                    .into_iter()
                    .map(|hash| URL_SAFE_NO_PAD.encode(hash))
                    .collect()
            },
            application_hybrid_required: true,
            encrypted_objects_required: material.encrypted_objects_required,
            projection_signer_ed25519: String::new(),
            projection_signer_ml_dsa65: String::new(),
            projection_signature: String::new(),
        }
    }

    /// Hybrid-sign the complete projection. This signature protects the HTTP
    /// representation and cache boundary; it is not accepted-state authority.
    pub fn sign_projection(mut self, signing_key: &crate::crypto::SigningKey) -> Result<Self> {
        let pq_signing_key = crate::node_identity::derive_mesh_mldsa_key(signing_key);
        let pq_verifying_key = ml_dsa::Keypair::verifying_key(&pq_signing_key);
        let projection_ed = URL_SAFE_NO_PAD.encode(signing_key.verifying_key().to_bytes());
        let projection_pq =
            URL_SAFE_NO_PAD.encode(crate::crypto::pq::ml_dsa_vk_bytes(&pq_verifying_key));
        anyhow::ensure!(
            projection_ed == self.response_ed25519 && projection_pq == self.response_ml_dsa65,
            "projection signer is not the accepted-current response signing authority"
        );
        self.projection_signer_ed25519 = projection_ed;
        self.projection_signer_ml_dsa65 = projection_pq;
        self.projection_signature.clear();
        let payload = self.projection_payload()?;
        let aad = projection_external_aad();
        let signature = crate::crypto::cose_sign::sign_composite(
            signing_key,
            Some(&pq_signing_key),
            &payload,
            &aad,
        )?;
        self.projection_signature = URL_SAFE_NO_PAD.encode(signature);
        Ok(self)
    }

    fn projection_payload(&self) -> Result<Vec<u8>> {
        let mut unsigned = self.clone();
        unsigned.projection_signature.clear();
        serde_json::to_vec(&unsigned)
            .context("failed to encode browser projection signature payload")
    }
}

/// Accepted-state evidence copied into the signed and HyKEM-sealed request.
/// Dispatch verifies it against the live checkpoint source before handler work.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserRequestBinding {
    pub version: String,
    pub service_name: String,
    pub service_did: String,
    pub service_origin: String,
    pub webtransport_url: String,
    pub certificate_hashes: Vec<String>,
    pub capability: String,
    pub scope: String,
    pub carrier_profile: BrowserCarrierProfile,
    pub response_key_id: String,
    pub response_key_digest: String,
    pub request_kem_key_id: String,
    pub request_kem_digest: String,
    pub accepted_state_digest: String,
    pub accepted_state_epoch: u64,
    pub expires_at_unix_ms: i64,
}

impl BrowserRequestBinding {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() <= MAX_BROWSER_BINDING_BYTES,
            "browser request binding exceeds {MAX_BROWSER_BINDING_BYTES} bytes"
        );
        let binding: Self =
            serde_json::from_slice(bytes).context("invalid browser request binding")?;
        binding.validate_shape()?;
        Ok(binding)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate_shape()?;
        let bytes = serde_json::to_vec(self).context("failed to encode browser request binding")?;
        anyhow::ensure!(
            bytes.len() <= MAX_BROWSER_BINDING_BYTES,
            "browser request binding exceeds {MAX_BROWSER_BINDING_BYTES} bytes"
        );
        Ok(bytes)
    }

    pub fn validate_shape(&self) -> Result<()> {
        anyhow::ensure!(
            self.version == BROWSER_PROVISIONING_VERSION,
            "invalid browser binding version"
        );
        validate_service_name(&self.service_name)?;
        anyhow::ensure!(
            self.service_did.starts_with("did:at9p:"),
            "browser binding requires did:at9p"
        );
        let service_origin = parse_https_url(&self.service_origin, "browser binding service origin")?;
        let webtransport_url =
            parse_https_url(&self.webtransport_url, "browser binding WebTransport route")?;
        anyhow::ensure!(
            self.service_origin == service_origin.as_str()
                && self.webtransport_url == webtransport_url.as_str(),
            "browser binding route evidence is not normalized"
        );
        anyhow::ensure!(
            self.certificate_hashes.len() <= MAX_BROWSER_CERTIFICATE_HASHES,
            "browser binding has too many certificate hashes"
        );
        let mut normalized_hashes = self.certificate_hashes.clone();
        for hash in &normalized_hashes {
            decode_exact::<32>(hash, "certificate hash")?;
        }
        normalized_hashes.sort();
        normalized_hashes.dedup();
        anyhow::ensure!(
            normalized_hashes == self.certificate_hashes,
            "browser binding certificate hashes are not normalized"
        );
        validate_bounded_token(&self.capability, "browser binding capability", 128)?;
        validate_bounded_token(&self.scope, "browser binding scope", 512)?;
        validate_key_id(&self.response_key_id, &self.service_did, "response")?;
        validate_key_id(&self.request_kem_key_id, &self.service_did, "request KEM")?;
        decode_exact::<32>(&self.response_key_digest, "response key digest")?;
        decode_exact::<32>(&self.request_kem_digest, "request KEM digest")?;
        decode_exact::<64>(&self.accepted_state_digest, "accepted-state digest")?;
        anyhow::ensure!(
            self.accepted_state_epoch > 0,
            "browser binding epoch must be non-zero"
        );
        Ok(())
    }

    pub fn accepted_state_digest_bytes(&self) -> Result<[u8; 64]> {
        decode_exact(&self.accepted_state_digest, "accepted-state digest")
    }

    pub fn response_key_digest_bytes(&self) -> Result<[u8; 32]> {
        decode_exact(&self.response_key_digest, "response key digest")
    }

    pub fn request_kem_digest_bytes(&self) -> Result<[u8; 32]> {
        decode_exact(&self.request_kem_digest, "request KEM digest")
    }

    pub fn matches_response_key(&self, key: &[u8]) -> Result<bool> {
        use sha2::{Digest as _, Sha256};
        Ok(self.response_key_digest_bytes()?.as_slice() == Sha256::digest(key).as_slice())
    }

    pub fn matches_request_kem(&self, recipient: &[u8]) -> Result<bool> {
        use sha2::{Digest as _, Sha256};
        Ok(self.request_kem_digest_bytes()?.as_slice() == Sha256::digest(recipient).as_slice())
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BrowserBoundRequestExtension {
    version: String,
    domain: String,
    request_id: u64,
    service_name: String,
    method_discriminator: u16,
    request_digest: String,
    application_len: u32,
    carrier_profile: BrowserCarrierProfile,
    binding: BrowserRequestBinding,
}

/// Explicit dispatch profile for the inner extension. A browser extension is
/// mandatory only on its browser carrier and is rejected as a misclassified
/// payload on every other carrier.
pub enum BrowserTranscriptPolicy<'a> {
    NotBrowserCarrier,
    Required {
        request_id: u64,
        service_name: &'a str,
        carrier_profile: BrowserCarrierProfile,
    },
}

/// Place accepted-current evidence and the application request in one bounded,
/// canonical transcript before the normal hybrid signature and request HyKEM
/// seal are constructed.
pub fn bind_request_payload(
    binding: &BrowserRequestBinding,
    request_id: u64,
    service_name: &str,
    method_discriminator: u16,
    application_payload: &[u8],
) -> Result<Vec<u8>> {
    anyhow::ensure!(request_id != 0, "browser request id must be non-zero");
    anyhow::ensure!(
        !application_payload.is_empty()
            && application_payload.len() <= MAX_BROWSER_APPLICATION_BYTES,
        "browser application request is empty or exceeds {MAX_BROWSER_APPLICATION_BYTES} bytes"
    );
    validate_service_name(service_name)?;
    anyhow::ensure!(
        binding.service_name == service_name,
        "browser binding crosses the request service"
    );
    let extension = BrowserBoundRequestExtension {
        version: BROWSER_PROVISIONING_VERSION.to_owned(),
        domain: BROWSER_EXTENSION_DOMAIN.to_owned(),
        request_id,
        service_name: service_name.to_owned(),
        method_discriminator,
        request_digest: request_digest(request_id, service_name, application_payload),
        application_len: u32::try_from(application_payload.len())
            .context("browser application length overflow")?,
        carrier_profile: binding.carrier_profile,
        binding: binding.clone(),
    };
    let extension = serde_json::to_vec(&extension).context("encode browser request extension")?;
    anyhow::ensure!(
        extension.len() <= MAX_BROWSER_EXTENSION_BYTES,
        "browser request extension exceeds {MAX_BROWSER_EXTENSION_BYTES} bytes"
    );
    let extension_len =
        u32::try_from(extension.len()).context("browser extension length overflow")?;
    let mut payload = Vec::with_capacity(
        BROWSER_BOUND_PAYLOAD_MAGIC.len() + 4 + extension.len() + application_payload.len(),
    );
    payload.extend_from_slice(BROWSER_BOUND_PAYLOAD_MAGIC);
    payload.extend_from_slice(&extension_len.to_be_bytes());
    payload.extend_from_slice(&extension);
    payload.extend_from_slice(application_payload);
    Ok(payload)
}

/// Recover authenticated evidence after signature verification and HyKEM open.
/// Canonical reserialization rejects alternate encodings as well as Serde's
/// unknown/duplicate-field rejection.
pub fn recover_request_payload(
    payload: &[u8],
    policy: BrowserTranscriptPolicy<'_>,
) -> Result<(Option<BrowserRequestTranscript>, Vec<u8>)> {
    let (request_id, service_name, carrier_profile) = match policy {
        BrowserTranscriptPolicy::NotBrowserCarrier => {
            anyhow::ensure!(
                !payload.starts_with(BROWSER_BOUND_PAYLOAD_MAGIC),
                "browser extension arrived on a non-browser carrier"
            );
            return Ok((None, payload.to_vec()));
        }
        BrowserTranscriptPolicy::Required {
            request_id,
            service_name,
            carrier_profile,
        } => (request_id, service_name, carrier_profile),
    };
    let header_len = BROWSER_BOUND_PAYLOAD_MAGIC.len() + 4;
    anyhow::ensure!(
        payload.len() >= header_len && payload.starts_with(BROWSER_BOUND_PAYLOAD_MAGIC),
        "WebTransport request omitted the required browser extension"
    );
    let length_offset = BROWSER_BOUND_PAYLOAD_MAGIC.len();
    let extension_len_bytes: [u8; 4] = payload[length_offset..header_len]
        .try_into()
        .context("invalid browser extension length prefix")?;
    let extension_len = u32::from_be_bytes(extension_len_bytes) as usize;
    anyhow::ensure!(
        extension_len > 0
            && extension_len <= MAX_BROWSER_EXTENSION_BYTES
            && header_len + extension_len <= payload.len(),
        "truncated or oversized browser request extension"
    );
    let extension_bytes = &payload[header_len..header_len + extension_len];
    let extension: BrowserBoundRequestExtension =
        serde_json::from_slice(extension_bytes).context("invalid browser request extension")?;
    anyhow::ensure!(
        serde_json::to_vec(&extension).context("canonicalize browser request extension")?
            == extension_bytes,
        "browser request extension is not canonical"
    );
    let application_start = header_len + extension_len;
    let application_len = extension.application_len as usize;
    anyhow::ensure!(
        application_len > 0
            && application_len <= MAX_BROWSER_APPLICATION_BYTES
            && application_start
                .checked_add(application_len)
                .is_some_and(|total| total == payload.len()),
        "browser application length is empty, oversized, truncated, or has trailing bytes"
    );
    let application_payload = &payload[application_start..];
    anyhow::ensure!(
        extension.version == BROWSER_PROVISIONING_VERSION
            && extension.domain == BROWSER_EXTENSION_DOMAIN
            && extension.request_id == request_id
            && extension.service_name == service_name
            && extension.carrier_profile == carrier_profile
            && extension.binding.service_name == service_name
            && extension.binding.carrier_profile == carrier_profile,
        "browser request extension context mismatch"
    );
    extension.binding.validate_shape()?;
    anyhow::ensure!(
        extension.request_digest == request_digest(request_id, service_name, application_payload),
        "browser request digest commitment mismatch"
    );
    Ok((
        Some(BrowserRequestTranscript {
            binding: extension.binding,
            method_discriminator: extension.method_discriminator,
        }),
        application_payload.to_vec(),
    ))
}

/// Authenticated browser-only transcript metadata exposed to the generated
/// service dispatcher. The generated dispatcher independently decodes its
/// request union and compares that schema discriminator with this commitment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BrowserRequestTranscript {
    pub binding: BrowserRequestBinding,
    pub method_discriminator: u16,
}

/// Independently decode the canonical top-level Cap'n Proto request-union
/// discriminator from a fully decoded request. Generated service dispatch uses
/// this value; the browser extension only carries the client-side commitment.
pub fn canonical_method_discriminator(application_payload: &[u8]) -> Result<u16> {
    let mut slice = application_payload;
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut slice,
        capnp::message::ReaderOptions::default(),
    )
    .context("browser application is not a Cap'n Proto request")?;
    anyhow::ensure!(slice.is_empty(), "Cap'n Proto request has trailing bytes");
    use capnp::message::ReaderSegments as _;
    let bytes = reader
        .get_segments()
        .get_segment(0)
        .context("Cap'n Proto request omitted segment zero")?;
    anyhow::ensure!(bytes.len() >= 24, "Cap'n Proto request root is too small");
    let pointer = u64::from_le_bytes(bytes[..8].try_into().context("root pointer")?);
    anyhow::ensure!(pointer & 3 == 0, "Cap'n Proto request root is not a struct");
    let raw_offset = ((pointer >> 2) & 0x3fff_ffff) as i64;
    let signed_offset = if raw_offset & (1 << 29) != 0 {
        raw_offset - (1 << 30)
    } else {
        raw_offset
    };
    let data_words = ((pointer >> 32) & 0xffff) as usize;
    let target_word = 1i64
        .checked_add(signed_offset)
        .filter(|target| *target >= 1)
        .context("invalid Cap'n Proto root struct offset")? as usize;
    anyhow::ensure!(
        data_words >= 2 && (target_word + data_words) * 8 <= bytes.len(),
        "Cap'n Proto request lacks the canonical top-level method discriminator"
    );
    let discriminator_offset = target_word * 8 + 8;
    Ok(u16::from_le_bytes(
        bytes[discriminator_offset..discriminator_offset + 2]
            .try_into()
            .context("method discriminator")?,
    ))
}

fn request_digest(request_id: u64, service_name: &str, application_payload: &[u8]) -> String {
    use sha2::{Digest as _, Sha256};
    let mut digest = Sha256::new();
    digest.update(BROWSER_REQUEST_DOMAIN);
    digest.update(request_id.to_be_bytes());
    digest.update((service_name.len() as u64).to_be_bytes());
    digest.update(service_name.as_bytes());
    digest.update((application_payload.len() as u64).to_be_bytes());
    digest.update(application_payload);
    URL_SAFE_NO_PAD.encode(digest.finalize())
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait BrowserCurrentnessVerifier: Send + Sync {
    async fn ensure_current(&self, binding: &BrowserRequestBinding) -> Result<()>;
}

/// Opaque, fully validated provisioning snapshot used for dial/seal.
#[derive(Clone)]
pub struct BrowserProvisioning {
    document: BrowserProvisioningDocument,
    server_verifying_key: VerifyingKey,
    response_ml_dsa65: Vec<u8>,
    request_kem_recipient: RecipientPublic,
    accepted_state_digest: [u8; 64],
    certificate_hashes: Vec<[u8; 32]>,
}

impl std::fmt::Debug for BrowserProvisioning {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BrowserProvisioning")
            .field("service_name", &self.document.service_name)
            .field("service_did", &self.document.service_did)
            .field("carrier_profile", &self.document.carrier_profile)
            .field("accepted_state_epoch", &self.document.accepted_state_epoch)
            .field("expires_at_unix_ms", &self.document.expires_at_unix_ms)
            .finish_non_exhaustive()
    }
}

impl BrowserProvisioning {
    pub fn from_json(
        json: &[u8],
        expected: &BrowserProvisioningRequest,
        now_unix_ms: i64,
    ) -> Result<Self> {
        anyhow::ensure!(
            json.len() <= MAX_PROVISIONING_BYTES,
            "browser provisioning document exceeds {MAX_PROVISIONING_BYTES} bytes"
        );
        let document: BrowserProvisioningDocument =
            serde_json::from_slice(json).context("invalid browser provisioning document")?;
        Self::validate(document, expected, now_unix_ms)
    }

    pub fn validate(
        document: BrowserProvisioningDocument,
        expected: &BrowserProvisioningRequest,
        now_unix_ms: i64,
    ) -> Result<Self> {
        expected.validate()?;
        anyhow::ensure!(
            document.version == BROWSER_PROVISIONING_VERSION,
            "unsupported browser provisioning version"
        );
        anyhow::ensure!(
            document.projection_signer_ed25519 == document.response_ed25519
                && document.projection_signer_ml_dsa65 == document.response_ml_dsa65,
            "projection signer is not equality-bound to accepted-current response keys"
        );
        let response_ed25519 = decode_exact::<32>(&document.response_ed25519, "response Ed25519")?;
        let server_verifying_key = VerifyingKey::from_bytes(&response_ed25519)
            .context("invalid browser response Ed25519 key")?;
        let response_ml_dsa65 = decode(&document.response_ml_dsa65, "response ML-DSA-65")?;
        let projection_pq = crate::crypto::pq::ml_dsa_vk_from_bytes(&response_ml_dsa65)
            .context("invalid browser response ML-DSA-65 key")?;
        let projection_signature = decode(&document.projection_signature, "projection signature")?;
        let projection_payload = document.projection_payload()?;
        crate::crypto::cose_sign::verify_composite(
            &projection_signature,
            &server_verifying_key,
            Some(&projection_pq),
            &projection_payload,
            &projection_external_aad(),
            true,
        )
        .context("browser provisioning projection signature verification failed")?;
        validate_service_name(&document.service_name)?;
        anyhow::ensure!(
            document.service_name == expected.service_name
                && document.capability == expected.capability
                && document.scope == expected.scope
                && document.carrier_profile == expected.carrier_profile,
            "browser provisioning context does not match the requested service/capability/scope/profile"
        );
        anyhow::ensure!(
            document.service_did.starts_with("did:at9p:")
                && !document
                    .service_did
                    .bytes()
                    .any(|byte| byte.is_ascii_whitespace()),
            "browser provisioning requires a canonical did:at9p service identity"
        );
        validate_key_id(&document.response_key_id, &document.service_did, "response")?;
        validate_key_id(
            &document.request_kem_key_id,
            &document.service_did,
            "request KEM",
        )?;

        let origin = parse_https_url(&document.service_origin, "service origin")?;
        let route = parse_https_url(&document.webtransport_url, "WebTransport route")?;
        anyhow::ensure!(
            document.service_origin == origin.as_str()
                && document.webtransport_url == route.as_str(),
            "browser provisioning route evidence is not normalized"
        );
        match document.carrier_profile {
            BrowserCarrierProfile::OwnedHybridWebTransport => {
                anyhow::ensure!(
                    document.route_role == BrowserRouteRole::Origin
                        && document.transport_security
                            == BrowserTransportSecurity::OwnedHybridRequired,
                    "owned WebTransport requires the origin role and owned-hybrid transport policy"
                );
                anyhow::ensure!(
                    origin.host_str() == route.host_str()
                        && origin.port_or_known_default() == route.port_or_known_default(),
                    "owned WebTransport route crosses the accepted service origin"
                );
                anyhow::ensure!(
                    !document.encrypted_objects_required,
                    "owned unary provisioning cannot claim the public-relay Object profile"
                );
            }
            BrowserCarrierProfile::StandardPublicRelay => {
                anyhow::ensure!(
                    document.route_role == BrowserRouteRole::Relay
                        && document.transport_security
                            == BrowserTransportSecurity::ClassicalUntrusted,
                    "public relay must remain explicitly classical/untrusted"
                );
                anyhow::ensure!(
                    document.encrypted_objects_required,
                    "public relay requires encrypted application Objects"
                );
            }
        }
        anyhow::ensure!(
            document.application_hybrid_required,
            "application HyKEM is mandatory for every browser carrier profile"
        );
        anyhow::ensure!(
            document.accepted_state_epoch > 0,
            "genesis-only state cannot provision a production browser recipient"
        );
        anyhow::ensure!(
            document.expires_at_unix_ms > now_unix_ms,
            "browser provisioning is expired; re-resolution required"
        );
        anyhow::ensure!(
            document.expires_at_unix_ms - now_unix_ms <= MAX_PROVISIONING_LIFETIME_MS,
            "browser provisioning lifetime exceeds the closed freshness profile"
        );

        let recipient_bytes = decode(&document.request_kem_recipient, "request KEM recipient")?;
        let request_kem_recipient = RecipientPublic::decode(&recipient_bytes)
            .context("invalid browser request KEM recipient")?;
        anyhow::ensure!(
            request_kem_recipient.suite_id == SuiteId::HyKemX25519MlKem768
                && request_kem_recipient.eks.len()
                    == SuiteId::HyKemX25519MlKem768.components().len(),
            "browser request recipient is classical-only or suite-incomplete"
        );
        request_kem_recipient.validate()?;
        let accepted_state_digest =
            decode_exact::<64>(&document.accepted_state_digest, "accepted-state digest")?;
        anyhow::ensure!(
            document.certificate_hashes.len() <= MAX_BROWSER_CERTIFICATE_HASHES,
            "browser provisioning has too many certificate hashes"
        );
        let certificate_hashes = document
            .certificate_hashes
            .iter()
            .map(|value| decode_exact::<32>(value, "certificate hash"))
            .collect::<Result<Vec<_>>>()?;
        let mut normalized_hashes = document.certificate_hashes.clone();
        normalized_hashes.sort();
        normalized_hashes.dedup();
        anyhow::ensure!(
            normalized_hashes == document.certificate_hashes,
            "browser provisioning certificate hashes are not normalized"
        );

        let canonical = serde_json::to_vec(&document)
            .context("failed to canonicalize browser provisioning document")?;
        anyhow::ensure!(
            canonical.len() <= MAX_PROVISIONING_BYTES,
            "browser provisioning document exceeds {MAX_PROVISIONING_BYTES} bytes"
        );
        Ok(Self {
            document,
            server_verifying_key,
            response_ml_dsa65,
            request_kem_recipient,
            accepted_state_digest,
            certificate_hashes,
        })
    }

    pub fn ensure_fresh(&self, now_unix_ms: i64) -> Result<()> {
        anyhow::ensure!(
            now_unix_ms < self.document.expires_at_unix_ms,
            "browser provisioning expired; re-resolution required"
        );
        Ok(())
    }

    pub fn same_snapshot(&self, other: &Self) -> bool {
        let mut left = self.document.clone();
        let mut right = other.document.clone();
        // The HTTP projection TTL and its signature are freshly minted on each
        // no-store fetch. All accepted-state, identity, route, key, capability,
        // and signer bindings must remain byte-identical.
        left.expires_at_unix_ms = 0;
        right.expires_at_unix_ms = 0;
        left.projection_signature.clear();
        right.projection_signature.clear();
        left == right
    }

    pub fn service_name(&self) -> &str {
        &self.document.service_name
    }

    pub fn service_did(&self) -> &str {
        &self.document.service_did
    }

    pub fn service_origin(&self) -> &str {
        &self.document.service_origin
    }

    pub fn webtransport_url(&self) -> &str {
        &self.document.webtransport_url
    }

    pub fn carrier_profile(&self) -> BrowserCarrierProfile {
        self.document.carrier_profile
    }

    pub fn accepted_state_digest(&self) -> &[u8; 64] {
        &self.accepted_state_digest
    }

    pub fn accepted_state_epoch(&self) -> u64 {
        self.document.accepted_state_epoch
    }

    pub fn expires_at_unix_ms(&self) -> i64 {
        self.document.expires_at_unix_ms
    }

    pub fn server_verifying_key(&self) -> VerifyingKey {
        self.server_verifying_key
    }

    pub fn certificate_hashes(&self) -> &[[u8; 32]] {
        &self.certificate_hashes
    }

    pub fn crypto_stores(&self) -> Result<(Arc<dyn KemTrustStore>, Arc<dyn PqTrustStore>)> {
        let ed = self.server_verifying_key.to_bytes();
        let mut kem = KeyedKemTrustStore::new();
        kem.bind(ed, self.request_kem_recipient.clone());
        let pq_key = crate::crypto::pq::ml_dsa_vk_from_bytes(&self.response_ml_dsa65)
            .context("invalid provisioned ML-DSA-65 response key")?;
        let mut pq = KeyedPqTrustStore::new();
        pq.bind(ed, &pq_key);
        Ok((Arc::new(kem), Arc::new(pq)))
    }

    pub fn request_binding(&self) -> Result<BrowserRequestBinding> {
        use sha2::{Digest as _, Sha256};

        let response_key_digest = Sha256::digest(self.server_verifying_key.to_bytes());
        let request_kem_digest = Sha256::digest(self.request_kem_recipient.encode());
        Ok(BrowserRequestBinding {
            version: BROWSER_PROVISIONING_VERSION.to_owned(),
            service_name: self.document.service_name.clone(),
            service_did: self.document.service_did.clone(),
            service_origin: self.document.service_origin.clone(),
            webtransport_url: self.document.webtransport_url.clone(),
            certificate_hashes: self.document.certificate_hashes.clone(),
            capability: self.document.capability.clone(),
            scope: self.document.scope.clone(),
            carrier_profile: self.document.carrier_profile,
            response_key_id: self.document.response_key_id.clone(),
            response_key_digest: URL_SAFE_NO_PAD.encode(response_key_digest),
            request_kem_key_id: self.document.request_kem_key_id.clone(),
            request_kem_digest: URL_SAFE_NO_PAD.encode(request_kem_digest),
            accepted_state_digest: self.document.accepted_state_digest.clone(),
            accepted_state_epoch: self.document.accepted_state_epoch,
            expires_at_unix_ms: self.document.expires_at_unix_ms,
        })
    }
}

fn projection_external_aad() -> Vec<u8> {
    crate::crypto::cose_sign1::build_external_aad(PROJECTION_SCHEMA_ID, PROJECTION_TYPE_ID)
}

fn validate_service_name(value: &str) -> Result<()> {
    anyhow::ensure!(!value.is_empty(), "service name is empty");
    anyhow::ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'),
        "service name is not canonical"
    );
    Ok(())
}

fn validate_bounded_token(value: &str, label: &str, max_len: usize) -> Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= max_len,
        "{label} is empty or too long"
    );
    anyhow::ensure!(
        !value
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace()),
        "{label} contains whitespace or control bytes"
    );
    Ok(())
}

fn validate_key_id(key_id: &str, did: &str, label: &str) -> Result<()> {
    anyhow::ensure!(
        key_id.starts_with(&format!("{did}#")) && key_id.len() > did.len() + 1,
        "{label} key id crosses service authority"
    );
    validate_bounded_token(key_id, &format!("{label} key id"), 512)
}

fn parse_https_url(value: &str, label: &str) -> Result<url::Url> {
    let parsed = url::Url::parse(value).with_context(|| format!("invalid {label}"))?;
    anyhow::ensure!(
        parsed.scheme() == "https"
            && parsed.host_str().is_some()
            && parsed.username().is_empty()
            && parsed.password().is_none()
            && parsed.fragment().is_none(),
        "{label} must be an absolute credential-free HTTPS URL"
    );
    Ok(parsed)
}

fn decode(value: &str, label: &str) -> Result<Vec<u8>> {
    let decoded = URL_SAFE_NO_PAD
        .decode(value)
        .with_context(|| format!("invalid {label} base64url"))?;
    anyhow::ensure!(
        URL_SAFE_NO_PAD.encode(&decoded) == value,
        "non-canonical {label} base64url"
    );
    Ok(decoded)
}

fn decode_exact<const N: usize>(value: &str, label: &str) -> Result<[u8; N]> {
    decode(value, label)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("{label} must be exactly {N} bytes"))
}

#[cfg(any(target_arch = "wasm32", test))]
fn ensure_provisioning_chunk_fits(
    current_len: usize,
    chunk_len: usize,
    declared_len: usize,
) -> Result<()> {
    let next_len = current_len
        .checked_add(chunk_len)
        .context("browser provisioning response length overflow")?;
    anyhow::ensure!(
        next_len <= declared_len && next_len <= MAX_PROVISIONING_BYTES,
        "browser provisioning response exceeded its bounded content-length"
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub async fn fetch_browser_provisioning(
    provisioning_origin: &str,
    expected: &BrowserProvisioningRequest,
) -> Result<BrowserProvisioning> {
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen_futures::JsFuture;

    expected.validate()?;
    let mut url = parse_https_url(provisioning_origin, "provisioning origin")?;
    anyhow::ensure!(
        url.path() == "/" && url.query().is_none(),
        "provisioning origin must not contain a path or query"
    );
    url.set_path(&format!(
        "/.well-known/hyprstream/browser-provisioning/{}",
        expected.service_name
    ));
    url.query_pairs_mut()
        .append_pair("capability", &expected.capability)
        .append_pair("scope", &expected.scope)
        .append_pair("carrier_profile", expected.carrier_profile.as_str());
    let init = web_sys::RequestInit::new();
    init.set_method("GET");
    init.set_cache(web_sys::RequestCache::NoStore);
    init.set_credentials(web_sys::RequestCredentials::SameOrigin);
    init.set_redirect(web_sys::RequestRedirect::Error);
    let fetch_request =
        web_sys::Request::new_with_str_and_init(url.as_str(), &init).map_err(|error| {
            anyhow::anyhow!("browser provisioning request construction failed: {error:?}")
        })?;
    let window = web_sys::window().ok_or_else(|| anyhow::anyhow!("browser window unavailable"))?;
    let response_value = JsFuture::from(window.fetch_with_request(&fetch_request))
        .await
        .map_err(|error| anyhow::anyhow!("browser provisioning fetch failed: {error:?}"))?;
    let response: web_sys::Response = response_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("browser provisioning fetch returned a non-Response"))?;
    anyhow::ensure!(
        response.ok() && !response.redirected(),
        "browser provisioning endpoint returned HTTP {} or redirected",
        response.status()
    );
    let response_url = parse_https_url(&response.url(), "provisioning response URL")?;
    anyhow::ensure!(
        response_url.origin() == url.origin(),
        "browser provisioning response crossed its authenticated origin"
    );
    let headers = response.headers();
    let content_type = headers
        .get("content-type")
        .map_err(|error| {
            anyhow::anyhow!("browser provisioning content-type unavailable: {error:?}")
        })?
        .ok_or_else(|| anyhow::anyhow!("browser provisioning response omitted content-type"))?;
    anyhow::ensure!(
        content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json")),
        "browser provisioning response is not application/json"
    );
    let cache_control = headers
        .get("cache-control")
        .map_err(|error| {
            anyhow::anyhow!("browser provisioning cache policy unavailable: {error:?}")
        })?
        .ok_or_else(|| anyhow::anyhow!("browser provisioning response omitted cache-control"))?;
    anyhow::ensure!(
        cache_control
            .split(',')
            .any(|directive| directive.trim().eq_ignore_ascii_case("no-store")),
        "browser provisioning response is cacheable"
    );
    let content_length = headers
        .get("content-length")
        .map_err(|error| {
            anyhow::anyhow!("browser provisioning content length unavailable: {error:?}")
        })?
        .ok_or_else(|| anyhow::anyhow!("browser provisioning response omitted content-length"))?
        .parse::<usize>()
        .context("invalid browser provisioning content-length")?;
    anyhow::ensure!(
        content_length <= MAX_PROVISIONING_BYTES,
        "browser provisioning response exceeds {MAX_PROVISIONING_BYTES} bytes"
    );
    let body = response
        .body()
        .ok_or_else(|| anyhow::anyhow!("browser provisioning response omitted its body"))?;
    let reader: web_sys::ReadableStreamDefaultReader = body.get_reader().unchecked_into();
    let mut bytes = Vec::with_capacity(content_length);
    loop {
        let result = JsFuture::from(reader.read())
            .await
            .map_err(|error| anyhow::anyhow!("browser provisioning response read failed: {error:?}"))?;
        let done = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("done"))
            .map_err(|_| anyhow::anyhow!("browser provisioning response chunk omitted done"))?
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("browser provisioning response chunk had invalid done"))?;
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("value"))
            .map_err(|_| anyhow::anyhow!("browser provisioning response chunk omitted value"))?;
        let chunk = js_sys::Uint8Array::new(&value);
        let chunk_len = usize::try_from(chunk.length())
            .context("browser provisioning response chunk length overflow")?;
        ensure_provisioning_chunk_fits(bytes.len(), chunk_len, content_length)?;
        let start = bytes.len();
        bytes.resize(start + chunk_len, 0);
        chunk.copy_to(&mut bytes[start..]);
    }
    reader.release_lock();
    anyhow::ensure!(
        bytes.len() == content_length,
        "browser provisioning response size did not match its bounded content-length"
    );
    let now = chrono::Utc::now().timestamp_millis();
    let provisioned = BrowserProvisioning::from_json(&bytes, expected, now)?;
    let service_origin = parse_https_url(provisioned.service_origin(), "service origin")?;
    anyhow::ensure!(
        service_origin.host_str() == url.host_str(),
        "browser provisioning was not fetched from the accepted service domain"
    );
    Ok(provisioned)
}

/// WASM pre-seal guard which re-fetches the accepted-current document before
/// every request and rejects any state/key/service/profile change. The caller
/// must construct a new client from the new snapshot after invalidation.
#[cfg(target_arch = "wasm32")]
pub struct BrowserProvisioningGuard {
    provisioning_origin: String,
    expected: BrowserProvisioningRequest,
    snapshot: BrowserProvisioning,
}

#[cfg(target_arch = "wasm32")]
impl BrowserProvisioningGuard {
    pub fn new(
        provisioning_origin: impl Into<String>,
        expected: BrowserProvisioningRequest,
        snapshot: BrowserProvisioning,
    ) -> Self {
        Self {
            provisioning_origin: provisioning_origin.into(),
            expected,
            snapshot,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
impl crate::rpc_client::PreSealGuard for BrowserProvisioningGuard {
    async fn ensure_current(&self) -> Result<()> {
        self.snapshot
            .ensure_fresh(chrono::Utc::now().timestamp_millis())?;
        let current = fetch_browser_provisioning(&self.provisioning_origin, &self.expected).await?;
        anyhow::ensure!(
            self.snapshot.same_snapshot(&current),
            "accepted browser provisioning changed; re-resolution required before seal"
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn material(now: i64) -> BrowserProvisioningMaterial {
        let response = crate::crypto::SigningKey::from_bytes(&[0x41; 32]);
        let pq = crate::node_identity::derive_mesh_mldsa_key(&response);
        let pq_vk = ml_dsa::Keypair::verifying_key(&pq);
        let recipient = crate::node_identity::derive_mesh_kem_recipient(&response)
            .expect("derive KEM")
            .public();
        BrowserProvisioningMaterial {
            service_name: "model".to_owned(),
            service_did: "did:at9p:test-service".to_owned(),
            service_origin: "https://model.example/".to_owned(),
            webtransport_url: "https://model.example/wt".to_owned(),
            capability: "hyprstream-rpc/1".to_owned(),
            scope: "model".to_owned(),
            carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
            route_role: BrowserRouteRole::Origin,
            transport_security: BrowserTransportSecurity::OwnedHybridRequired,
            response_key_id: "did:at9p:test-service#response".to_owned(),
            response_ed25519: response.verifying_key().to_bytes(),
            response_ml_dsa65: crate::crypto::pq::ml_dsa_vk_bytes(&pq_vk),
            request_kem_key_id: "did:at9p:test-service#mesh-kem".to_owned(),
            request_kem_recipient: recipient,
            accepted_state_digest: [0x33; 64],
            accepted_state_epoch: 9,
            expires_at_unix_ms: now + 30_000,
            certificate_hashes: vec![[0x55; 32]],
            encrypted_objects_required: false,
        }
    }

    fn request() -> BrowserProvisioningRequest {
        BrowserProvisioningRequest::new(
            "model",
            "hyprstream-rpc/1",
            "model",
            BrowserCarrierProfile::OwnedHybridWebTransport,
        )
        .expect("request")
    }

    fn signed(material: BrowserProvisioningMaterial) -> BrowserProvisioningDocument {
        BrowserProvisioningDocument::from_material(material)
            .sign_projection(&crate::crypto::SigningKey::from_bytes(&[0x41; 32]))
            .expect("sign projection")
    }

    fn application_request(method_discriminator: u16) -> Vec<u8> {
        // One three-word segment containing a direct root struct with two data
        // words: the generated service request id followed by its union tag.
        let mut request = Vec::with_capacity(32);
        request.extend_from_slice(&0u32.to_le_bytes());
        request.extend_from_slice(&3u32.to_le_bytes());
        request.extend_from_slice(&(2u64 << 32).to_le_bytes());
        request.extend_from_slice(&91u64.to_le_bytes());
        request.extend_from_slice(&(method_discriminator as u64).to_le_bytes());
        request
    }

    #[test]
    fn accepted_current_hybrid_document_builds_resolution_local_stores() {
        let now = 1_000_000;
        let provisioned = BrowserProvisioning::validate(signed(material(now)), &request(), now)
            .expect("valid provisioning");
        let (kem, pq) = provisioned.crypto_stores().expect("stores");
        let ed = provisioned.server_verifying_key().to_bytes();
        assert!(kem.kem_recipient_for(&ed).is_some());
        assert!(pq.ml_dsa_key_for(&ed).is_some());

        let binding = provisioned.request_binding().expect("binding");
        let application = application_request(7);
        let sealed_transcript =
            bind_request_payload(&binding, 17, "model", 7, &application).expect("bind payload");
        let (opened_binding, opened_payload) = recover_request_payload(
            &sealed_transcript,
            BrowserTranscriptPolicy::Required {
                request_id: 17,
                service_name: "model",
                carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
            },
        )
        .expect("open payload");
        assert_eq!(opened_binding.expect("transcript").binding, binding);
        assert_eq!(opened_payload, application);
    }

    #[test]
    fn stale_cross_service_and_classical_only_documents_fail_closed() {
        let now = 1_000_000;
        let mut stale = material(now);
        stale.expires_at_unix_ms = now;
        assert!(BrowserProvisioning::validate(signed(stale), &request(), now).is_err());

        let mut crossed = material(now);
        crossed.service_name = "policy".to_owned();
        assert!(BrowserProvisioning::validate(signed(crossed), &request(), now).is_err());

        let mut classical = BrowserProvisioningDocument::from_material(material(now));
        classical.request_kem_recipient = URL_SAFE_NO_PAD.encode([0x11; 32]);
        classical = classical
            .sign_projection(&crate::crypto::SigningKey::from_bytes(&[0x41; 32]))
            .expect("sign classical mutation");
        assert!(BrowserProvisioning::validate(classical, &request(), now).is_err());
    }

    #[test]
    fn substituted_projection_signer_or_signature_fails_closed() {
        let now = 1_000_000;
        let mut substituted_key = signed(material(now));
        substituted_key.projection_signer_ed25519 = URL_SAFE_NO_PAD.encode(
            crate::crypto::SigningKey::from_bytes(&[0x72; 32])
                .verifying_key()
                .to_bytes(),
        );
        assert!(BrowserProvisioning::validate(substituted_key, &request(), now).is_err());

        let mut substituted_pq_key = signed(material(now));
        substituted_pq_key.projection_signer_ml_dsa65 = URL_SAFE_NO_PAD.encode(vec![0x73; 1952]);
        assert!(BrowserProvisioning::validate(substituted_pq_key, &request(), now).is_err());

        let mut substituted_signature = signed(material(now));
        let mut signature = decode(
            &substituted_signature.projection_signature,
            "test signature",
        )
        .expect("decode signature");
        let last = signature.len() - 1;
        signature[last] ^= 1;
        substituted_signature.projection_signature = URL_SAFE_NO_PAD.encode(signature);
        assert!(BrowserProvisioning::validate(substituted_signature, &request(), now).is_err());

        assert!(recover_request_payload(
            b"unbound application request",
            BrowserTranscriptPolicy::Required {
                request_id: 17,
                service_name: "model",
                carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
            },
        )
        .is_err());
    }

    #[test]
    fn streaming_body_cap_rejects_lying_length_and_oversize_before_growth() {
        assert!(ensure_provisioning_chunk_fits(0, 9, 8).is_err());
        assert!(ensure_provisioning_chunk_fits(
            0,
            MAX_PROVISIONING_BYTES + 1,
            MAX_PROVISIONING_BYTES,
        )
        .is_err());
        assert!(ensure_provisioning_chunk_fits(7, 1, 8).is_ok());
    }

    #[test]
    fn inner_extension_is_canonical_bounded_and_profile_gated() {
        let now = 1_000_000;
        let provisioned = BrowserProvisioning::validate(signed(material(now)), &request(), now)
            .expect("valid provisioning");
        let binding = provisioned.request_binding().expect("binding");
        let application = application_request(7);
        let framed = bind_request_payload(&binding, 23, "model", 7, &application).expect("frame");
        let required = || BrowserTranscriptPolicy::Required {
            request_id: 23,
            service_name: "model",
            carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
        };

        let (_, recovered) = recover_request_payload(&framed, required()).expect("recover");
        assert_eq!(recovered, application);
        let (none, compatible) =
            recover_request_payload(&application, BrowserTranscriptPolicy::NotBrowserCarrier)
                .expect("explicit non-browser compatibility");
        assert!(none.is_none());
        assert_eq!(compatible, application);
        assert!(
            recover_request_payload(&framed, BrowserTranscriptPolicy::NotBrowserCarrier).is_err()
        );

        assert!(recover_request_payload(&framed[..framed.len() - 1], required()).is_err());
        let mut trailing = framed.clone();
        trailing.push(0);
        assert!(recover_request_payload(&trailing, required()).is_err());
        assert!(recover_request_payload(
            &framed,
            BrowserTranscriptPolicy::Required {
                request_id: 24,
                service_name: "model",
                carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
            }
        )
        .is_err());
        assert!(recover_request_payload(
            &framed,
            BrowserTranscriptPolicy::Required {
                request_id: 23,
                service_name: "policy",
                carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
            }
        )
        .is_err());
        assert!(recover_request_payload(
            &framed,
            BrowserTranscriptPolicy::Required {
                request_id: 23,
                service_name: "model",
                carrier_profile: BrowserCarrierProfile::StandardPublicRelay,
            }
        )
        .is_err());

        let header_len = BROWSER_BOUND_PAYLOAD_MAGIC.len() + 4;
        let extension_len = u32::from_be_bytes(
            framed[BROWSER_BOUND_PAYLOAD_MAGIC.len()..header_len]
                .try_into()
                .expect("length"),
        ) as usize;
        assert!(
            recover_request_payload(&framed[..header_len + extension_len - 1], required(),)
                .is_err()
        );
        let extension = &framed[header_len..header_len + extension_len];
        let application = &framed[header_len + extension_len..];
        let reframe = |replacement: &[u8], replacement_application: &[u8]| {
            let mut value = Vec::new();
            value.extend_from_slice(BROWSER_BOUND_PAYLOAD_MAGIC);
            value.extend_from_slice(&(replacement.len() as u32).to_be_bytes());
            value.extend_from_slice(replacement);
            value.extend_from_slice(replacement_application);
            value
        };

        // Recovery authenticates and returns the explicit client commitment;
        // generated service dispatch independently decodes the request union
        // and rejects a cross-method commitment before invoking the handler.
        let mut mismatched_method: BrowserBoundRequestExtension =
            serde_json::from_slice(extension).expect("extension");
        mismatched_method.method_discriminator += 1;
        let mismatched_method = serde_json::to_vec(&mismatched_method).expect("method extension");
        let (transcript, recovered) =
            recover_request_payload(&reframe(&mismatched_method, application), required())
                .expect("recover authenticated explicit method commitment");
        assert_eq!(recovered, application);
        let transcript = transcript.expect("browser transcript");
        assert_eq!(transcript.method_discriminator, 8);
        let mut ctx = crate::service::EnvelopeContext::from_callback_service(23, "model");
        ctx.browser_method_discriminator = Some(transcript.method_discriminator);
        assert!(ctx.ensure_browser_method(7).is_err());

        let different_method = application_request(8);
        assert_eq!(different_method.len(), application.len());
        assert!(
            recover_request_payload(&reframe(extension, &different_method), required()).is_err()
        );

        let mut unknown_version: BrowserBoundRequestExtension =
            serde_json::from_slice(extension).expect("extension");
        unknown_version.version = "hyprstream.browser-provisioning.v2".to_owned();
        let unknown_version = serde_json::to_vec(&unknown_version).expect("version extension");
        assert!(
            recover_request_payload(&reframe(&unknown_version, application), required()).is_err()
        );

        let mut noncanonical = vec![b' '];
        noncanonical.extend_from_slice(extension);
        assert!(recover_request_payload(&reframe(&noncanonical, application), required()).is_err());

        let mut unknown = extension[..extension.len() - 1].to_vec();
        unknown.extend_from_slice(br#","unknown":true}"#);
        let unknown_error = recover_request_payload(&reframe(&unknown, application), required())
            .expect_err("unknown field must reject");
        assert!(format!("{unknown_error:#}").contains("unknown field"));

        let mut duplicate = extension[..extension.len() - 1].to_vec();
        duplicate.extend_from_slice(br#","version":"hyprstream.browser-provisioning.v1"}"#);
        let duplicate_error =
            recover_request_payload(&reframe(&duplicate, application), required())
                .expect_err("duplicate field must reject");
        assert!(format!("{duplicate_error:#}").contains("duplicate field"));

        let mut oversized = Vec::from(*BROWSER_BOUND_PAYLOAD_MAGIC);
        oversized.extend_from_slice(&((MAX_BROWSER_EXTENSION_BYTES + 1) as u32).to_be_bytes());
        assert!(recover_request_payload(&oversized, required()).is_err());
        assert!(bind_request_payload(&binding, 23, "model", 7, &[]).is_err());
        assert!(bind_request_payload(
            &binding,
            23,
            "model",
            7,
            &vec![0; MAX_BROWSER_APPLICATION_BYTES + 1]
        )
        .is_err());
    }

    #[test]
    fn public_relay_scope_must_be_exact() {
        let exact = BrowserProvisioningRequest::new(
            "model",
            "hyprstream-moq/1",
            "tenant-a/track-a",
            BrowserCarrierProfile::StandardPublicRelay,
        );
        assert!(exact.is_ok());
        for wildcard in ["*", "tenant-a/*", "tenant-*/track-a", "tenant-a/"] {
            assert!(
                BrowserProvisioningRequest::new(
                    "model",
                    "hyprstream-moq/1",
                    wildcard,
                    BrowserCarrierProfile::StandardPublicRelay,
                )
                .is_err(),
                "wildcard relay scope {wildcard:?} must be rejected"
            );
        }
    }

    #[test]
    fn public_relay_is_classical_untrusted_and_requires_encrypted_objects() {
        let now = 1_000_000;
        let mut relay = material(now);
        relay.capability = "hyprstream-moq/1".to_owned();
        relay.scope = "tenant-a/track-a".to_owned();
        relay.carrier_profile = BrowserCarrierProfile::StandardPublicRelay;
        relay.route_role = BrowserRouteRole::Relay;
        relay.transport_security = BrowserTransportSecurity::ClassicalUntrusted;
        relay.webtransport_url = "https://relay.example/moq".to_owned();
        relay.certificate_hashes.clear();
        relay.encrypted_objects_required = true;
        let request = BrowserProvisioningRequest::new(
            "model",
            "hyprstream-moq/1",
            "tenant-a/track-a",
            BrowserCarrierProfile::StandardPublicRelay,
        )
        .expect("request");
        BrowserProvisioning::validate(signed(relay), &request, now).expect("valid relay profile");

        let mut omitted = material(now);
        omitted.capability = "hyprstream-moq/1".to_owned();
        omitted.scope = "tenant-a/track-a".to_owned();
        omitted.carrier_profile = BrowserCarrierProfile::StandardPublicRelay;
        omitted.route_role = BrowserRouteRole::Relay;
        omitted.transport_security = BrowserTransportSecurity::ClassicalUntrusted;
        omitted.webtransport_url = "https://relay.example/moq".to_owned();
        assert!(BrowserProvisioning::validate(signed(omitted), &request, now,).is_err());
    }
}
