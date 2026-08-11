//! v16 request-proof signer enrollment manifest (design §4.4, §11).
//!
//! Request-proof signer keys are their **own** enrollment, with their own
//! lifecycle. They are deliberately not derived from, and never shared with,
//! the mesh/envelope identity: component-key separation is normative, so a key
//! enrolled for the request-proof WNS hybrid suite MUST NOT simultaneously be
//! enrolled for another protocol or domain separator. Relabelling the mesh
//! Ed25519 + ML-DSA-65 pair as a proof signer would be exactly that violation,
//! and it could not supply an enrollment epoch, validity window, revocation
//! state, approver role, or enrollment-policy identifier either — those are
//! enrollment facts, not transport facts, and inventing them is not enrollment.
//!
//! The manifest is an operator-authored file, established out-of-band like the
//! other admin-anchored trust material. Absent manifest, nothing is enrolled
//! and every authenticated proof denies; there is no derived fallback.
//!
//! Cross-protocol overlap is checked, not merely documented: an entry whose
//! component key is already anchored as a mesh/envelope identity is rejected
//! unless the entry names an explicit operator-approved exception policy.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use hyprstream_rpc::proof::enrollment::{
    ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole, SignerSuiteRecord,
};

/// The enrolled role a manifest entry takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ManifestRole {
    /// A credential `cnf`-bound primary request signer.
    Primary,
    /// An anchored approver in an additional authorization group.
    Approver,
    /// The enrolled response signer for one service domain.
    Service,
}

/// One enrolled signer in the manifest.
///
/// Every lifecycle field is explicit. There is no default epoch, no implied
/// "never expires", and no inferred role: an entry that does not state them is
/// rejected at load rather than enrolled under invented values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// The enrolled principal this signer resolves to.
    pub principal: String,
    /// The role this entry is enrolled for.
    pub role: ManifestRole,
    /// The exact suite ID the signed plan must declare for this signer.
    pub suite_id: String,
    /// The enrollment policy this entry was issued under. Part of the replay
    /// namespace's provenance and the only way an operator may authorize a
    /// cross-protocol key overlap.
    pub enrollment_policy_id: String,
    /// Enrollment epoch. Rotating it changes the replay namespace, so a
    /// re-enrolled signer never inherits a retired namespace's history.
    pub epoch: u64,
    /// Unix seconds after which this enrollment is no longer usable.
    pub not_after: u64,
    /// Whether this enrollment has been revoked.
    #[serde(default)]
    pub revoked: bool,
    /// For `Approver` entries, the approver role a generated rule may name.
    #[serde(default)]
    pub approver_role: Option<String>,
    /// For `Service` entries, the canonical service domain this signer answers
    /// for.
    #[serde(default)]
    pub service_domain: Option<String>,
    /// The pinned component keys, in the suite-declared order.
    pub components: Vec<ManifestComponent>,
}

/// One pinned component key: its key ID and its public key material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestComponent {
    /// Opaque key ID, hex-encoded (1..64 bytes).
    pub kid_hex: String,
    /// Algorithm: `ed25519` or `ml-dsa-65`.
    pub alg: ManifestAlg,
    /// Hex-encoded raw public key (32 bytes Ed25519, 1952 bytes ML-DSA-65).
    pub public_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ManifestAlg {
    Ed25519,
    #[serde(rename = "ml-dsa-65")]
    MlDsa65,
}

/// The manifest file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofEnrollmentManifest {
    #[serde(default)]
    pub entries: Vec<ManifestEntry>,
}

impl ProofEnrollmentManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading proof enrollment manifest {}", path.display()))?;
        toml::from_str(&text)
            .with_context(|| format!("parsing proof enrollment manifest {}", path.display()))
    }
}

/// Build the resolver from a manifest.
///
/// `foreign_protocol_keys` is the set of public keys already enrolled for
/// another protocol — in practice the mesh/envelope identities. An entry
/// reusing one of them is rejected unless it names an exception policy, which
/// is how §4.4's "an exception requires a separate operator-approved analysis
/// and an explicit enrollment-policy identifier" is enforced rather than
/// assumed.
///
/// A rejected entry is skipped with an error log; it is never enrolled under
/// partial or substituted material.
pub fn build_resolver(
    manifest: &ProofEnrollmentManifest,
    foreign_protocol_keys: &HashSet<Vec<u8>>,
    exception_policy_ids: &HashSet<String>,
    now: u64,
) -> InMemoryEnrollmentResolver {
    let mut resolver = InMemoryEnrollmentResolver::new();

    // Component-key separation is not only cross-protocol: within the manifest
    // a component key may appear exactly once, across every entry, suite,
    // role, and logical group. A key present twice would let one holder occupy
    // two logical signers, satisfy a threshold alone, or carry a suite it was
    // not analysed for — so a repeated key disqualifies *every* entry using
    // it, not just the later one. Detecting it up front, before enrolling
    // anything, is what makes that possible.
    let mut seen: std::collections::HashMap<Vec<u8>, usize> = std::collections::HashMap::new();
    for entry in &manifest.entries {
        for component in &entry.components {
            if let Ok(public) = hex::decode(&component.public_hex) {
                *seen.entry(public).or_insert(0) += 1;
            }
        }
    }
    let reused: HashSet<Vec<u8>> = seen
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(key, _)| key)
        .collect();

    for entry in &manifest.entries {
        match enrol_one(
            &mut resolver,
            entry,
            foreign_protocol_keys,
            &reused,
            exception_policy_ids,
            now,
        ) {
            Ok(()) => tracing::info!(
                "proof enrollment: '{}' enrolled as {:?} (epoch {}, suite {})",
                entry.principal,
                entry.role,
                entry.epoch,
                entry.suite_id
            ),
            Err(e) => tracing::error!(
                "proof enrollment: '{}' rejected, not enrolled: {e:#}",
                entry.principal
            ),
        }
    }
    resolver
}

#[allow(clippy::too_many_arguments)]
fn enrol_one(
    resolver: &mut InMemoryEnrollmentResolver,
    entry: &ManifestEntry,
    foreign_protocol_keys: &HashSet<Vec<u8>>,
    reused_within_manifest: &HashSet<Vec<u8>>,
    exception_policy_ids: &HashSet<String>,
    now: u64,
) -> Result<()> {
    if entry.enrollment_policy_id.is_empty() {
        bail!("entry names no enrollment policy");
    }
    if entry.not_after <= now {
        bail!(
            "enrollment validity ({}) is already past",
            entry.not_after
        );
    }
    if entry.components.is_empty() {
        bail!("entry pins no component keys");
    }

    let mut components = Vec::with_capacity(entry.components.len());
    for component in &entry.components {
        let kid = hex::decode(&component.kid_hex).context("kid is not hex")?;
        if kid.is_empty() || kid.len() > hyprstream_rpc::proof::MAX_KID_BYTES {
            bail!("kid must be 1..{} bytes", hyprstream_rpc::proof::MAX_KID_BYTES);
        }
        let public = hex::decode(&component.public_hex).context("public key is not hex")?;

        // Component-key separation (§4.4). A key may not be shared with
        // another protocol, and may not appear twice within this manifest —
        // across entries, suites, roles, or logical groups. Only the
        // cross-protocol case is waivable, and only under an explicit
        // operator-approved exception policy; intra-manifest reuse is never
        // waivable, because no analysis can make one key two logical signers.
        if reused_within_manifest.contains(&public) {
            bail!(
                "component key appears in more than one manifest entry; \
                 a key is exactly one logical signer"
            );
        }
        if foreign_protocol_keys.contains(&public)
            && !exception_policy_ids.contains(&entry.enrollment_policy_id)
        {
            bail!(
                "component key is already enrolled for another protocol; \
                 cross-protocol reuse requires an approved exception policy"
            );
        }

        let key = match component.alg {
            ManifestAlg::Ed25519 => {
                let bytes: [u8; 32] = public
                    .as_slice()
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("Ed25519 key must be 32 bytes"))?;
                ComponentKey::Ed25519(
                    ed25519_dalek::VerifyingKey::from_bytes(&bytes)
                        .map_err(|e| anyhow::anyhow!("invalid Ed25519 key: {e}"))?,
                )
            }
            ManifestAlg::MlDsa65 => ComponentKey::MlDsa65(Box::new(
                hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&public)?,
            )),
        };
        components.push(EnrolledComponent::new(kid, key));
    }

    let role = match entry.role {
        ManifestRole::Primary => SignerRole::Primary,
        ManifestRole::Approver => SignerRole::Approver,
        ManifestRole::Service => SignerRole::Service,
    };
    if role == SignerRole::Approver && entry.approver_role.is_none() {
        bail!("an approver entry must name the approver role it holds");
    }

    let record = SignerSuiteRecord {
        principal: entry.principal.clone(),
        suite_id: entry.suite_id.clone(),
        components,
        epoch: entry.epoch,
        role,
        approver_role: entry.approver_role.clone(),
        enrollment_policy_id: entry.enrollment_policy_id.clone(),
        not_after: entry.not_after,
        revoked: entry.revoked,
    };

    match entry.role {
        ManifestRole::Primary => {
            // The credential `cnf` key is the record's first Ed25519
            // component: the credential binds the proof by pinning exact
            // component keys.
            let cnf = record
                .components
                .iter()
                .find_map(|c| match &c.key {
                    ComponentKey::Ed25519(k) => Some(*k),
                    ComponentKey::MlDsa65(_) => None,
                })
                .ok_or_else(|| anyhow::anyhow!("a primary entry needs an Ed25519 component"))?;
            resolver.enrol_primary(&cnf, record)
        }
        ManifestRole::Approver => resolver.enrol_approver(record),
        ManifestRole::Service => {
            let domain = entry
                .service_domain
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("a service entry must name its service domain"))?;
            resolver.enrol_service(domain, record)
        }
    }
}

/// Every public key this node already holds for another protocol, as raw
/// public-key bytes.
///
/// These are exactly the keys a request-proof enrollment must not silently
/// reuse. The set is deliberately a superset rather than one source: the mesh
/// peer roster (remote identities) *and* this node's own bootstrap identity
/// keys (local identities), because reuse is equally a violation in either
/// direction. Anything unreadable is skipped — a key we cannot decode is not
/// a key we can prove is foreign, and the cross-protocol check is a refusal,
/// so a miss here can only make the manifest more permissive, never less
/// correct about the keys it does know.
pub fn foreign_protocol_keys(
    oauth: Option<&crate::config::OAuthConfig>,
    secrets_dir: Option<&Path>,
) -> HashSet<Vec<u8>> {
    use hyprstream_rpc::did_key::{decode_multikey, MULTICODEC_ED25519_PUB, MULTICODEC_ML_DSA_65_PUB};

    let mut keys = HashSet::new();

    // Remote: the admin-anchored mesh peer roster.
    if let Some(oauth) = oauth {
        for peer in oauth.mesh_peers.values() {
            if let Ok(ed) = decode_multikey(&peer.ed25519_multibase, &MULTICODEC_ED25519_PUB) {
                keys.insert(ed);
            }
            if let Ok(pq) = decode_multikey(&peer.mldsa65_multibase, &MULTICODEC_ML_DSA_65_PUB) {
                keys.insert(pq);
            }
        }
    }

    // Local: this node's own OS-owned bootstrap public keys, which anchor the
    // local services' envelope identities.
    if let Some(dir) = secrets_dir {
        collect_bootstrap_pubkeys(dir, &mut keys);
    }

    keys
}

/// Read raw public keys out of the OS-owned bootstrap-pubkeys directory.
///
/// The file format is the provisioning wizard's; entries that do not decode as
/// hex or base64 public keys are ignored rather than guessed at.
fn collect_bootstrap_pubkeys(secrets_dir: &Path, keys: &mut HashSet<Vec<u8>>) {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    let dir = secrets_dir.join("bootstrap-pubkeys");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for token in text.split_whitespace() {
            if let Ok(bytes) = hex::decode(token) {
                if bytes.len() == 32 || bytes.len() == 1952 {
                    keys.insert(bytes);
                    continue;
                }
            }
            if let Ok(bytes) = STANDARD.decode(token) {
                if bytes.len() == 32 || bytes.len() == 1952 {
                    keys.insert(bytes);
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_rpc::proof::enrollment::EnrollmentResolver;
    use rand::rngs::OsRng;

    const NOW: u64 = 1_000;

    fn ed_component(key: &SigningKey, kid: &str) -> ManifestComponent {
        ManifestComponent {
            kid_hex: hex::encode(kid.as_bytes()),
            alg: ManifestAlg::Ed25519,
            public_hex: hex::encode(key.verifying_key().to_bytes()),
        }
    }

    fn primary_entry(key: &SigningKey) -> ManifestEntry {
        ManifestEntry {
            principal: "client".into(),
            role: ManifestRole::Primary,
            suite_id: hyprstream_rpc::proof::SUITE_CLASSICAL.into(),
            enrollment_policy_id: "proof-signers-v1".into(),
            epoch: 3,
            not_after: 9_000,
            revoked: false,
            approver_role: None,
            service_domain: None,
            components: vec![ed_component(key, "client-1")],
        }
    }

    /// Two entries sharing a component key disqualify BOTH: no analysis can
    /// make one key two logical signers, so the reuse is never waivable.
    #[test]
    fn a_component_key_reused_across_entries_disqualifies_both() {
        let key = SigningKey::generate(&mut OsRng);
        let mut approver = primary_entry(&key);
        approver.principal = "approver".into();
        approver.role = ManifestRole::Approver;
        approver.approver_role = Some("security".into());

        let resolver = build(vec![primary_entry(&key), approver]);
        assert!(
            resolver.resolve_primary(&key.verifying_key()).is_none(),
            "the primary entry must be disqualified"
        );
        assert!(
            resolver.resolve_approver(b"client-1").is_none(),
            "the approver entry must be disqualified too"
        );
    }

    /// Reuse across suites is equally forbidden, and an exception policy does
    /// not waive it — the exception covers cross-protocol reuse only.
    #[test]
    fn intra_manifest_reuse_is_not_waivable_by_an_exception_policy() {
        let key = SigningKey::generate(&mut OsRng);
        let mut hybrid = primary_entry(&key);
        hybrid.principal = "client-hybrid".into();
        hybrid.suite_id = hyprstream_rpc::proof::SUITE_HYBRID.into();
        hybrid.enrollment_policy_id = "approved-overlap".into();
        let mut classical = primary_entry(&key);
        classical.enrollment_policy_id = "approved-overlap".into();

        let mut exceptions = HashSet::new();
        exceptions.insert("approved-overlap".to_owned());
        let resolver = build_resolver(
            &ProofEnrollmentManifest {
                entries: vec![hybrid, classical],
            },
            &HashSet::new(),
            &exceptions,
            NOW,
        );
        assert!(resolver.resolve_primary(&key.verifying_key()).is_none());
    }

    /// A manifest still loads with no mesh configuration: the foreign set is
    /// then whatever local material exists, and enrollment is not skipped.
    #[test]
    fn a_manifest_loads_without_any_mesh_configuration() {
        let key = SigningKey::generate(&mut OsRng);
        let foreign = foreign_protocol_keys(None, None);
        assert!(foreign.is_empty());
        let resolver = build_resolver(
            &ProofEnrollmentManifest {
                entries: vec![primary_entry(&key)],
            },
            &foreign,
            &HashSet::new(),
            NOW,
        );
        assert!(resolver.resolve_primary(&key.verifying_key()).is_some());
    }

    /// The node's own bootstrap identity keys are foreign to proof
    /// enrollment: local reuse is a violation just as remote reuse is.
    #[test]
    fn local_bootstrap_keys_count_as_foreign() {
        let key = SigningKey::generate(&mut OsRng);
        let dir = tempfile::tempdir().expect("tempdir");
        let bootstrap = dir.path().join("bootstrap-pubkeys");
        std::fs::create_dir_all(&bootstrap).expect("mkdir");
        std::fs::write(
            bootstrap.join("registry"),
            hex::encode(key.verifying_key().to_bytes()),
        )
        .expect("write");

        let foreign = foreign_protocol_keys(None, Some(dir.path()));
        assert!(foreign.contains(key.verifying_key().to_bytes().as_slice()));

        let resolver = build_resolver(
            &ProofEnrollmentManifest {
                entries: vec![primary_entry(&key)],
            },
            &foreign,
            &HashSet::new(),
            NOW,
        );
        assert!(
            resolver.resolve_primary(&key.verifying_key()).is_none(),
            "a local envelope identity key must not be enrolled as a proof signer"
        );
    }

    fn build(entries: Vec<ManifestEntry>) -> InMemoryEnrollmentResolver {
        build_resolver(
            &ProofEnrollmentManifest { entries },
            &HashSet::new(),
            &HashSet::new(),
            NOW,
        )
    }

    #[test]
    fn a_manifest_entry_enrols_with_its_own_stated_lifecycle() {
        let key = SigningKey::generate(&mut OsRng);
        let resolver = build(vec![primary_entry(&key)]);
        let record = resolver
            .resolve_primary(&key.verifying_key())
            .expect("the entry must resolve");
        assert_eq!(record.epoch, 3, "the epoch is the manifest's, not invented");
        assert_eq!(record.not_after, 9_000);
        assert_eq!(record.enrollment_policy_id, "proof-signers-v1");
        assert!(!record.revoked);
    }

    /// Component-key separation: a key already enrolled for the mesh/envelope
    /// protocol cannot be relabelled as a request-proof signer.
    #[test]
    fn a_key_enrolled_for_another_protocol_is_rejected() {
        let key = SigningKey::generate(&mut OsRng);
        let mut foreign = HashSet::new();
        foreign.insert(key.verifying_key().to_bytes().to_vec());

        let resolver = build_resolver(
            &ProofEnrollmentManifest {
                entries: vec![primary_entry(&key)],
            },
            &foreign,
            &HashSet::new(),
            NOW,
        );
        assert!(
            resolver.resolve_primary(&key.verifying_key()).is_none(),
            "cross-protocol key reuse must not enrol"
        );
    }

    /// The overlap is permitted only under an explicit operator-approved
    /// exception policy — never inherited silently.
    #[test]
    fn an_approved_exception_policy_permits_a_declared_overlap() {
        let key = SigningKey::generate(&mut OsRng);
        let mut foreign = HashSet::new();
        foreign.insert(key.verifying_key().to_bytes().to_vec());
        let mut entry = primary_entry(&key);
        entry.enrollment_policy_id = "approved-overlap-2026-08".into();
        let mut exceptions = HashSet::new();
        exceptions.insert("approved-overlap-2026-08".to_owned());

        let resolver = build_resolver(
            &ProofEnrollmentManifest {
                entries: vec![entry],
            },
            &foreign,
            &exceptions,
            NOW,
        );
        assert!(resolver.resolve_primary(&key.verifying_key()).is_some());
    }

    #[test]
    fn an_entry_with_no_enrollment_policy_is_rejected() {
        let key = SigningKey::generate(&mut OsRng);
        let mut entry = primary_entry(&key);
        entry.enrollment_policy_id = String::new();
        let resolver = build(vec![entry]);
        assert!(resolver.resolve_primary(&key.verifying_key()).is_none());
    }

    #[test]
    fn an_already_expired_entry_is_not_enrolled() {
        let key = SigningKey::generate(&mut OsRng);
        let mut entry = primary_entry(&key);
        entry.not_after = NOW;
        let resolver = build(vec![entry]);
        assert!(resolver.resolve_primary(&key.verifying_key()).is_none());
    }

    /// A revoked entry loads — revocation is state, not a parse error — and
    /// then denies at use, so the operator sees the record they wrote.
    #[test]
    fn a_revoked_entry_loads_and_denies_at_use() {
        let key = SigningKey::generate(&mut OsRng);
        let mut entry = primary_entry(&key);
        entry.revoked = true;
        let resolver = build(vec![entry]);
        let record = resolver.resolve_primary(&key.verifying_key()).unwrap();
        assert!(record.revoked);
        assert!(record
            .check_usable(NOW, NOW + 10, SignerRole::Primary)
            .is_err());
    }

    #[test]
    fn an_approver_entry_must_name_its_role() {
        let key = SigningKey::generate(&mut OsRng);
        let mut entry = primary_entry(&key);
        entry.role = ManifestRole::Approver;
        entry.principal = "approver".into();
        let without = build(vec![entry.clone()]);
        assert!(without
            .resolve_approver(b"client-1")
            .is_none());

        entry.approver_role = Some("security".into());
        let with = build(vec![entry]);
        let record = with.resolve_approver(b"client-1").expect("must resolve");
        assert_eq!(record.approver_role.as_deref(), Some("security"));
    }

    #[test]
    fn a_service_entry_must_name_its_service_domain() {
        let key = SigningKey::generate(&mut OsRng);
        let mut entry = primary_entry(&key);
        entry.role = ManifestRole::Service;
        let without = build(vec![entry.clone()]);
        assert!(without.resolve_service("registry.svc.hyprstream.test").is_none());

        entry.service_domain = Some("registry.svc.hyprstream.test".into());
        let with = build(vec![entry]);
        assert!(with.resolve_service("registry.svc.hyprstream.test").is_some());
        assert!(with.resolve_service("other.svc.hyprstream.test").is_none());
    }

    #[test]
    fn an_empty_manifest_enrols_nothing() {
        let key = SigningKey::generate(&mut OsRng);
        let resolver = build(vec![]);
        assert!(resolver.resolve_primary(&key.verifying_key()).is_none());
    }

    #[test]
    fn a_manifest_round_trips_through_its_file_format() {
        let key = SigningKey::generate(&mut OsRng);
        let manifest = ProofEnrollmentManifest {
            entries: vec![primary_entry(&key)],
        };
        let text = toml::to_string(&manifest).expect("manifest must serialize");
        let parsed: ProofEnrollmentManifest =
            toml::from_str(&text).expect("manifest must parse back");
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].epoch, 3);
    }

    /// The mesh roster is read only to learn which keys are *already* another
    /// protocol's — never to enrol them.
    #[test]
    fn mesh_keys_are_collected_as_foreign_not_enrolled() {
        use crate::auth::mesh_trust::encode_multikey;
        use crate::config::MeshPeerConfig;
        use hyprstream_rpc::did_key::{MULTICODEC_ED25519_PUB, MULTICODEC_ML_DSA_65_PUB};

        let ed = SigningKey::generate(&mut OsRng);
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ed);
        let mut oauth = crate::config::OAuthConfig::default();
        oauth.mesh_peers.insert(
            "peer-a".to_owned(),
            MeshPeerConfig {
                ed25519_multibase: encode_multikey(
                    &ed.verifying_key().to_bytes(),
                    &MULTICODEC_ED25519_PUB,
                ),
                mldsa65_multibase: encode_multikey(
                    &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk),
                    &MULTICODEC_ML_DSA_65_PUB,
                ),
            },
        );

        let foreign = foreign_protocol_keys(Some(&oauth), None);
        assert!(foreign.contains(ed.verifying_key().to_bytes().as_slice()));

        // And a manifest that tries to reuse the mesh key is refused.
        let resolver = build_resolver(
            &ProofEnrollmentManifest {
                entries: vec![primary_entry(&ed)],
            },
            &foreign,
            &HashSet::new(),
            NOW,
        );
        assert!(resolver.resolve_primary(&ed.verifying_key()).is_none());
    }
}
