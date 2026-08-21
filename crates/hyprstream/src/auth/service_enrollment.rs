//! Service enrollment manifest — the authority-owned declaration of service
//! identity, target clearance, allowed audiences, and workload-session policy
//! (v16 §11).
//!
//! One manifest per deployment lives at `<secrets>/service-enrollment.json`,
//! produced by the wizard/bootstrap path. It is consumed once at startup:
//!
//! - service verifying keys carried here must AGREE with the hybrid
//!   bootstrap-pubkeys file (the enrolled key set is the trust-anchor input —
//!   a disagreement is a startup error, never a silent preference);
//! - the target clearance stamped into service credentials at issuance and
//!   renewal comes from here (renewal re-derives: authority removed from the
//!   manifest cannot survive a renewal, and no issuance can exceed it);
//! - the workload-session policy decides whether a service credential family
//!   may carry a `workload_session_id` (a standalone service credential
//!   omits it — v16 manufactures no session to populate a claim).
//!
//! Absence is legacy-tolerated for this staging wave (a loud warning; service
//! tokens then carry no manifest-backed clearance). Presence with any
//! malformed entry is a hard startup error — fail-closed.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context as _};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::VerifyingKey;
use hyprstream_rpc::auth::mac::SecurityLabel;
use serde::{Deserialize, Serialize};

/// Manifest file name within the resolved secrets directory.
pub const SERVICE_ENROLLMENT_FILE: &str = "service-enrollment.json";

/// Current manifest format version.
pub const SERVICE_ENROLLMENT_VERSION: u32 = 1;

/// One enrolled service: identity key, target clearance, audience ceiling,
/// and workload-session policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEnrollment {
    /// Ed25519 verifying key (base64url, 32 bytes) — must match the hybrid
    /// bootstrap-pubkeys entry for the same service.
    pub ed25519_pubkey: String,
    /// ML-DSA-65 verifying key (base64url) for the hybrid identity. Optional
    /// only for pre-hybrid legacy records; hybrid entries are the norm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ml_dsa_pubkey: Option<String>,
    /// Authority-issued target clearance for this service's credentials.
    pub clearance: SecurityLabel,
    /// Allowed credential audiences. `None` = unrestricted (legacy);
    /// `Some(list)` = issuance for an audience outside the list denies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_audiences: Option<Vec<String>>,
    /// Whether this service runs an enrolled workload credential family that
    /// may carry a `workload_session_id`. Default false: a standalone service
    /// credential carries no session identifier.
    #[serde(default)]
    pub workload_session: bool,
}

/// The parsed, validated manifest. Services are keyed by service name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEnrollmentManifest {
    /// Format version; must equal [`SERVICE_ENROLLMENT_VERSION`].
    pub version: u32,
    /// Enrolled services by name.
    pub services: BTreeMap<String, ServiceEnrollment>,
}

impl ServiceEnrollmentManifest {
    /// Load the manifest from the secrets directory. `Ok(None)` when the file
    /// is absent (legacy deployment — logged by the caller). Any present but
    /// malformed content is a hard error.
    pub fn load(secrets_dir: &Path) -> anyhow::Result<Option<Self>> {
        let path = secrets_dir.join(SERVICE_ENROLLMENT_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let raw =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let manifest: Self =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        manifest
            .validate()
            .with_context(|| format!("validate {}", path.display()))?;
        Ok(Some(manifest))
    }

    /// Structural validation: version, service-name syntax, key decodability.
    fn validate(&self) -> anyhow::Result<()> {
        if self.version != SERVICE_ENROLLMENT_VERSION {
            anyhow::bail!(
                "unsupported service-enrollment version {} (want {SERVICE_ENROLLMENT_VERSION})",
                self.version
            );
        }
        if self.services.is_empty() {
            anyhow::bail!("service-enrollment manifest lists no services");
        }
        for (name, entry) in &self.services {
            crate::auth::identity_store::validate_service_name(name)
                .map_err(|e| anyhow!("invalid service name {name:?}: {e}"))?;
            decode_ed25519(&entry.ed25519_pubkey)
                .with_context(|| format!("service {name:?} ed25519_pubkey"))?;
            if let Some(pq) = &entry.ml_dsa_pubkey {
                let bytes = URL_SAFE_NO_PAD
                    .decode(pq)
                    .with_context(|| format!("service {name:?} ml_dsa_pubkey base64"))?;
                // ML-DSA-65 verifying keys are 1952 bytes (FIPS 204).
                if bytes.len() != 1952 {
                    anyhow::bail!(
                        "service {name:?} ml_dsa_pubkey is {} bytes, want 1952",
                        bytes.len()
                    );
                }
            }
            if let Some(audiences) = &entry.allowed_audiences {
                if audiences.iter().any(|a| a.trim().is_empty()) {
                    anyhow::bail!("service {name:?} has an empty allowed audience");
                }
            }
        }
        Ok(())
    }

    /// The manifest must agree with the bootstrap trust anchors on BOTH key
    /// halves: every bootstrap-registered service appears here with the SAME
    /// Ed25519 key, the ML-DSA-65 half matches where either side carries one
    /// (a present-but-different, missing, or extra PQ half is a mismatch),
    /// and the manifest names no unknown service. Any divergence is a startup
    /// error — the two trust-anchor inputs must never silently disagree.
    pub fn validate_key_agreement(
        &self,
        bootstrap_pubkeys: &HashMap<String, crate::auth::identity_store::BootstrapPubkey>,
    ) -> anyhow::Result<()> {
        for (name, bootstrap) in bootstrap_pubkeys {
            let entry = self
                .services
                .get(name)
                .ok_or_else(|| anyhow!("bootstrap service {name:?} has no enrollment entry"))?;
            let enrolled = decode_ed25519(&entry.ed25519_pubkey)
                .with_context(|| format!("service {name:?} ed25519_pubkey"))?;
            if enrolled != bootstrap.ed25519 {
                anyhow::bail!(
                    "service {name:?}: enrollment Ed25519 key disagrees with bootstrap-pubkeys"
                );
            }
            let enrolled_pq = entry
                .ml_dsa_pubkey
                .as_deref()
                .map(|b64| {
                    URL_SAFE_NO_PAD
                        .decode(b64)
                        .context("base64url decode")
                        .and_then(|bytes| {
                            if bytes.len() == 1952 {
                                Ok(bytes)
                            } else {
                                Err(anyhow!("ML-DSA-65 verifying key must be 1952 bytes"))
                            }
                        })
                })
                .transpose()
                .with_context(|| format!("service {name:?} ml_dsa_pubkey"))?;
            let bootstrap_pq = bootstrap
                .ml_dsa_65
                .as_ref()
                .map(hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes);
            if enrolled_pq != bootstrap_pq {
                anyhow::bail!(
                    "service {name:?}: enrollment ML-DSA-65 key disagrees with bootstrap-pubkeys (present/different/missing halves must match exactly)"
                );
            }
        }
        for name in self.services.keys() {
            if !bootstrap_pubkeys.contains_key(name) {
                anyhow::bail!("enrollment entry {name:?} is not a bootstrap service");
            }
        }
        Ok(())
    }

    /// Load the manifest and validate it against the bootstrap trust anchors
    /// in one step — the startup path's helper. `Ok(None)` = absent (legacy);
    /// any present-but-invalid or disagreeing manifest is a hard error.
    pub fn load_and_validate(secrets_dir: &Path) -> anyhow::Result<Option<Self>> {
        let Some(manifest) = Self::load(secrets_dir)? else {
            return Ok(None);
        };
        let bootstrap = crate::auth::identity_store::load_bootstrap_pubkeys_hybrid(secrets_dir)?;
        manifest.validate_key_agreement(&bootstrap)?;
        Ok(Some(manifest))
    }

    /// Build a manifest from freshly provisioned bootstrap keys. The wizard
    /// default target is `Internal` level with assurance DERIVED FROM THE
    /// BOOTSTRAP KEY MATERIAL: a hybrid (Ed25519 + ML-DSA-65) identity gets
    /// `PqHybrid`, a classical one `Classical` — a hybrid-authenticated
    /// service must be able to dominate the reviewed generated dispatch
    /// target (`internal:pq-hybrid`), which a Classical default would deny at
    /// fresh boot. No session/audience restrictions by default; the operator
    /// reviews every target label at gate 5.
    pub fn from_bootstrap(
        pubkeys: &HashMap<String, crate::auth::identity_store::BootstrapPubkey>,
    ) -> Self {
        let services = pubkeys
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    ServiceEnrollment {
                        ed25519_pubkey: URL_SAFE_NO_PAD.encode(entry.ed25519.to_bytes()),
                        ml_dsa_pubkey: entry.ml_dsa_65.as_ref().map(|vk| {
                            URL_SAFE_NO_PAD.encode(hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(vk))
                        }),
                        clearance: default_clearance_for(entry),
                        allowed_audiences: None,
                        workload_session: false,
                    },
                )
            })
            .collect();
        Self {
            version: SERVICE_ENROLLMENT_VERSION,
            services,
        }
    }

    /// Re-provision validation (wizard re-run). The manifest is authoritative
    /// for key binding, so this NEVER rewrites it: a missing/extra service, a
    /// changed Ed25519 key, or a missing/extra/changed PQ half is a hard
    /// error until the operator deliberately edits the manifest (the
    /// reviewed rotation). Fresh installation derives both artifacts once via
    /// [`Self::from_bootstrap`]; re-provisioning only validates.
    pub fn reconcile_with_bootstrap(
        &self,
        pubkeys: &HashMap<String, crate::auth::identity_store::BootstrapPubkey>,
    ) -> anyhow::Result<()> {
        self.validate()?;
        self.validate_key_agreement(pubkeys)
            .map_err(|e| anyhow::anyhow!(
                "{e} — enrollment is authoritative for key binding; edit {SERVICE_ENROLLMENT_FILE} deliberately to rotate"
            ))
    }

    /// Persist the manifest (public keys + policy only — no secret material).
    pub fn write(&self, secrets_dir: &Path) -> anyhow::Result<()> {
        let path = secrets_dir.join(SERVICE_ENROLLMENT_FILE);
        let data = serde_json::to_vec_pretty(self).context("serialize enrollment manifest")?;
        std::fs::write(&path, data).with_context(|| format!("write {}", path.display()))
    }

    /// Authority-issued target clearance for a service, if enrolled.
    pub fn clearance_for_service(&self, service: &str) -> Option<SecurityLabel> {
        self.services.get(service).map(|e| e.clearance)
    }

    /// Whether issuance for `audience` is permitted for the service.
    /// `None` audience list = unrestricted (legacy); `Some(list)` = the
    /// effective audience must be PRESENT and a member — an enrolled service
    /// with a declared list cannot mint without an exact allowed audience
    /// (fail-closed).
    pub fn allows_audience(&self, service: &str, audience: Option<&str>) -> bool {
        match self.services.get(service) {
            Some(entry) => match &entry.allowed_audiences {
                Some(list) => audience.is_some_and(|aud| list.iter().any(|a| a == aud)),
                None => true,
            },
            None => false,
        }
    }

    /// Whether the service runs an enrolled workload credential family that
    /// may carry a `workload_session_id`.
    pub fn workload_session_policy(&self, service: &str) -> bool {
        self.services
            .get(service)
            .is_some_and(|e| e.workload_session)
    }
}

/// Wizard default target clearance for a service: `Internal` level, with
/// assurance derived from the bootstrap key material — hybrid identities get
/// `PqHybrid` (so the service can dominate the reviewed generated dispatch
/// target at fresh boot), classical identities `Classical`.
fn default_clearance_for(entry: &crate::auth::identity_store::BootstrapPubkey) -> SecurityLabel {
    let assurance = if entry.is_hybrid() {
        hyprstream_rpc::auth::mac::Assurance::PqHybrid
    } else {
        hyprstream_rpc::auth::mac::Assurance::Classical
    };
    SecurityLabel::new(
        hyprstream_rpc::auth::mac::Level::Internal,
        assurance,
        hyprstream_rpc::auth::mac::CompartmentSet::EMPTY,
    )
}

fn decode_ed25519(b64: &str) -> anyhow::Result<VerifyingKey> {
    let bytes = URL_SAFE_NO_PAD.decode(b64).context("base64url decode")?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow!("Ed25519 verifying key must be 32 bytes"))?;
    VerifyingKey::from_bytes(&bytes).context("invalid Ed25519 verifying key")
}

// ── Process-global handle ─────────────────────────────────────────────────

/// Process-global enrollment manifest, installed once at startup. `None`
/// means a legacy deployment without a manifest — service credentials then
/// carry no manifest-backed clearance (the legacy path), never a fabricated
/// one.
static GLOBAL_SERVICE_ENROLLMENT: std::sync::OnceLock<Arc<ServiceEnrollmentManifest>> =
    std::sync::OnceLock::new();

/// Install the process-global manifest. A second install is an error — two
/// manifests in one process is a startup bug, not a race to absorb.
pub fn set_global_service_enrollment(
    manifest: Arc<ServiceEnrollmentManifest>,
) -> Result<(), ServiceEnrollmentAlreadySet> {
    GLOBAL_SERVICE_ENROLLMENT
        .set(manifest)
        .map_err(|_| ServiceEnrollmentAlreadySet)
}

/// Error returned when a second manifest install is attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceEnrollmentAlreadySet;

impl std::fmt::Display for ServiceEnrollmentAlreadySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "service enrollment manifest already installed")
    }
}

impl std::error::Error for ServiceEnrollmentAlreadySet {}

/// The installed manifest, if any.
pub fn global_service_enrollment() -> Option<&'static Arc<ServiceEnrollmentManifest>> {
    GLOBAL_SERVICE_ENROLLMENT.get()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::identity_store::BootstrapPubkey;
    use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};

    fn test_label() -> SecurityLabel {
        SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn test_key_b64() -> (String, VerifyingKey) {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]);
        let vk = sk.verifying_key();
        (URL_SAFE_NO_PAD.encode(vk.to_bytes()), vk)
    }

    fn manifest_with(name: &str, key_b64: &str) -> ServiceEnrollmentManifest {
        let mut services = BTreeMap::new();
        services.insert(
            name.to_owned(),
            ServiceEnrollment {
                ed25519_pubkey: key_b64.to_owned(),
                ml_dsa_pubkey: None,
                clearance: test_label(),
                allowed_audiences: None,
                workload_session: false,
            },
        );
        ServiceEnrollmentManifest {
            version: SERVICE_ENROLLMENT_VERSION,
            services,
        }
    }

    #[test]
    fn valid_manifest_round_trips_and_answers_queries() {
        let (key_b64, vk) = test_key_b64();
        let manifest = manifest_with("oauth", &key_b64);
        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: ServiceEnrollmentManifest = serde_json::from_str(&json).unwrap();
        parsed.validate().unwrap();

        assert_eq!(parsed.clearance_for_service("oauth"), Some(test_label()));
        assert_eq!(parsed.clearance_for_service("unknown"), None);
        assert!(parsed.allows_audience("oauth", Some("anything"))); // None = unrestricted
        assert!(!parsed.allows_audience("unknown", Some("anything")));
        assert!(!parsed.workload_session_policy("oauth"));

        let mut pubkeys = HashMap::new();
        pubkeys.insert("oauth".to_owned(), BootstrapPubkey::classical(vk));
        parsed.validate_key_agreement(&pubkeys).unwrap();
    }

    #[test]
    fn key_disagreement_and_unknown_services_are_hard_errors() {
        let (key_b64, _vk) = test_key_b64();
        let manifest = manifest_with("oauth", &key_b64);

        // Bootstrap knows a service the manifest does not → error.
        let mut pubkeys = HashMap::new();
        pubkeys.insert(
            "registry".to_owned(),
            BootstrapPubkey::classical(
                ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]).verifying_key(),
            ),
        );
        assert!(manifest.validate_key_agreement(&pubkeys).is_err());

        // Same name, different Ed25519 key → error.
        let mut wrong = HashMap::new();
        wrong.insert(
            "oauth".to_owned(),
            BootstrapPubkey::classical(
                ed25519_dalek::SigningKey::from_bytes(&[0x44; 32]).verifying_key(),
            ),
        );
        assert!(manifest.validate_key_agreement(&wrong).is_err());

        // Manifest names a service bootstrap never registered → error.
        let m2 = manifest_with("oauth", &key_b64);
        assert!(m2.validate_key_agreement(&HashMap::new()).is_err());
    }

    #[test]
    fn pq_halves_must_agree_exactly() {
        let (key_b64, vk) = test_key_b64();
        let (pq_sk, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let _ = pq_sk;
        let pq_b64 = URL_SAFE_NO_PAD.encode(hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&pq_vk));

        // Matching hybrid on both sides → OK.
        let mut hybrid_manifest = manifest_with("oauth", &key_b64);
        hybrid_manifest
            .services
            .get_mut("oauth")
            .unwrap()
            .ml_dsa_pubkey = Some(pq_b64.clone());
        let mut hybrid_boot = HashMap::new();
        hybrid_boot.insert(
            "oauth".to_owned(),
            BootstrapPubkey::hybrid(vk, pq_vk.clone()),
        );
        hybrid_manifest
            .validate_key_agreement(&hybrid_boot)
            .unwrap();

        // Manifest carries a PQ half, bootstrap is classical → error.
        let mut classical_boot = HashMap::new();
        classical_boot.insert("oauth".to_owned(), BootstrapPubkey::classical(vk));
        assert!(
            hybrid_manifest
                .validate_key_agreement(&classical_boot)
                .is_err(),
            "extra PQ half in the manifest must fail"
        );

        // Bootstrap is hybrid, manifest lacks the PQ half → error.
        let classical_manifest = manifest_with("oauth", &key_b64);
        assert!(
            classical_manifest
                .validate_key_agreement(&hybrid_boot)
                .is_err(),
            "missing PQ half in the manifest must fail"
        );

        // Both carry a PQ half but the bytes differ → error.
        let (_sk2, pq_vk2) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let mut wrong_pq = manifest_with("oauth", &key_b64);
        wrong_pq.services.get_mut("oauth").unwrap().ml_dsa_pubkey =
            Some(URL_SAFE_NO_PAD.encode(hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&pq_vk2)));
        assert!(
            wrong_pq.validate_key_agreement(&hybrid_boot).is_err(),
            "a different PQ half must fail"
        );
    }

    #[test]
    fn from_bootstrap_manifest_agrees_with_its_source_keys() {
        let (sk, pq_vk) = {
            let sk = ed25519_dalek::SigningKey::from_bytes(&[0x46; 32]);
            let (_pq_sk, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
            (sk, pq_vk)
        };
        let mut pubkeys = HashMap::new();
        pubkeys.insert(
            "oauth".to_owned(),
            BootstrapPubkey::hybrid(sk.verifying_key(), pq_vk.clone()),
        );
        pubkeys.insert(
            "registry".to_owned(),
            BootstrapPubkey::classical(
                ed25519_dalek::SigningKey::from_bytes(&[0x47; 32]).verifying_key(),
            ),
        );
        let manifest = ServiceEnrollmentManifest::from_bootstrap(&pubkeys);
        manifest.validate().unwrap();
        // A freshly generated manifest must agree with its own source keys.
        manifest.validate_key_agreement(&pubkeys).unwrap();

        // Round-trip through the file form.
        let dir = tempfile::tempdir().unwrap();
        manifest.write(dir.path()).unwrap();
        let loaded = ServiceEnrollmentManifest::load(dir.path())
            .unwrap()
            .unwrap();
        loaded.validate_key_agreement(&pubkeys).unwrap();
    }

    #[test]
    fn default_clearance_derives_assurance_from_bootstrap_key() {
        // The reviewed generated dispatch target is internal:pq-hybrid; a
        // hybrid-authenticated service's enrollment clearance must dominate
        // it, or fresh boot denies registerServiceKey.
        let reviewed_target =
            SecurityLabel::new(Level::Internal, Assurance::PqHybrid, CompartmentSet::EMPTY);

        let sk = ed25519_dalek::SigningKey::from_bytes(&[0x48; 32]);
        let (_pq_sk, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let mut pubkeys = HashMap::new();
        pubkeys.insert(
            "oauth".to_owned(),
            BootstrapPubkey::hybrid(sk.verifying_key(), pq_vk.clone()),
        );
        pubkeys.insert(
            "legacy".to_owned(),
            BootstrapPubkey::classical(
                ed25519_dalek::SigningKey::from_bytes(&[0x49; 32]).verifying_key(),
            ),
        );
        let manifest = ServiceEnrollmentManifest::from_bootstrap(&pubkeys);

        let hybrid_clearance = manifest.clearance_for_service("oauth").unwrap();
        assert_eq!(hybrid_clearance.assurance, Assurance::PqHybrid);
        assert!(
            hybrid_clearance.can_access(&reviewed_target),
            "hybrid service clearance must dominate the reviewed internal:pq-hybrid target"
        );

        let classical_clearance = manifest.clearance_for_service("legacy").unwrap();
        assert_eq!(classical_clearance.assurance, Assurance::Classical);
        assert!(
            !classical_clearance.can_access(&reviewed_target),
            "classical clearance must NOT dominate a pq-hybrid target"
        );
    }

    #[test]
    fn reprovision_key_drift_is_never_auto_authorized() {
        // Fresh provision: manifest derives from the bootstrap keys.
        let sk = ed25519_dalek::SigningKey::from_bytes(&[0x50; 32]);
        let (_pq_sk, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let mut pubkeys = HashMap::new();
        pubkeys.insert(
            "oauth".to_owned(),
            BootstrapPubkey::hybrid(sk.verifying_key(), pq_vk.clone()),
        );
        let manifest = ServiceEnrollmentManifest::from_bootstrap(&pubkeys);
        manifest.reconcile_with_bootstrap(&pubkeys).unwrap();

        // Changed Ed25519 key → hard error.
        let mut drifted = HashMap::new();
        drifted.insert(
            "oauth".to_owned(),
            BootstrapPubkey::hybrid(
                ed25519_dalek::SigningKey::from_bytes(&[0x51; 32]).verifying_key(),
                pq_vk,
            ),
        );
        assert!(manifest.reconcile_with_bootstrap(&drifted).is_err());

        // Changed PQ half → hard error.
        let (_s2, pq_vk2) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let mut drifted_pq = HashMap::new();
        drifted_pq.insert(
            "oauth".to_owned(),
            BootstrapPubkey::hybrid(sk.verifying_key(), pq_vk2),
        );
        assert!(manifest.reconcile_with_bootstrap(&drifted_pq).is_err());

        // Hybrid → classical (PQ half dropped) → hard error.
        let mut downgraded = HashMap::new();
        downgraded.insert(
            "oauth".to_owned(),
            BootstrapPubkey::classical(sk.verifying_key()),
        );
        assert!(manifest.reconcile_with_bootstrap(&downgraded).is_err());

        // Extra service → hard error.
        let mut extra = pubkeys.clone();
        extra.insert(
            "registry".to_owned(),
            BootstrapPubkey::classical(
                ed25519_dalek::SigningKey::from_bytes(&[0x52; 32]).verifying_key(),
            ),
        );
        assert!(manifest.reconcile_with_bootstrap(&extra).is_err());

        // Missing service → hard error.
        assert!(manifest.reconcile_with_bootstrap(&HashMap::new()).is_err());
    }

    #[test]
    fn malformed_manifest_content_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SERVICE_ENROLLMENT_FILE);

        // Absent → Ok(None), legacy tolerated.
        assert!(ServiceEnrollmentManifest::load(dir.path())
            .unwrap()
            .is_none());

        // Unparseable → hard error.
        std::fs::write(&path, "{not json").unwrap();
        assert!(ServiceEnrollmentManifest::load(dir.path()).is_err());

        // Wrong version → hard error.
        let (key_b64, _vk) = test_key_b64();
        let mut manifest = manifest_with("oauth", &key_b64);
        manifest.version = 99;
        std::fs::write(&path, serde_json::to_string(&manifest).unwrap()).unwrap();
        assert!(ServiceEnrollmentManifest::load(dir.path()).is_err());

        // Undecodable key → hard error.
        let bad = manifest_with("oauth", "not-base64!!!");
        std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
        assert!(ServiceEnrollmentManifest::load(dir.path()).is_err());

        // Valid structure but unknown to bootstrap → load_and_validate error.
        std::fs::write(
            &path,
            serde_json::to_string(&manifest_with("oauth", &key_b64)).unwrap(),
        )
        .unwrap();
        assert!(ServiceEnrollmentManifest::load(dir.path())
            .unwrap()
            .is_some());
        assert!(
            ServiceEnrollmentManifest::load_and_validate(dir.path()).is_err(),
            "a manifest disagreeing with bootstrap anchors must fail startup validation"
        );
    }

    #[test]
    fn declared_audience_list_is_enforced_fail_closed() {
        let (key_b64, _vk) = test_key_b64();
        let mut manifest = manifest_with("oauth", &key_b64);
        manifest
            .services
            .get_mut("oauth")
            .unwrap()
            .allowed_audiences = Some(vec!["https://aud.example".to_owned()]);
        assert!(manifest.allows_audience("oauth", Some("https://aud.example")));
        assert!(
            !manifest.allows_audience("oauth", Some("https://other.example")),
            "audience outside the declared list must deny"
        );
        assert!(
            !manifest.allows_audience("oauth", None),
            "a declared list denies audience-less minting"
        );
    }
}
