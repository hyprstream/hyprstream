//! Offline service identity admission before network services start.

use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Duration, SecondsFormat, Utc};
use ed25519_dalek::SigningKey;
use hyprstream_pds::at9p::{
    CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
};
use hyprstream_pds::at9p_duplicity::AcceptedAt9pState;
use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
use hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use crate::auth::identity_store::{load_existing_service_signing_key, SecretsProfile};
use crate::config::HyprConfig;
use crate::services::discovery::{At9pStateIngest, PdsRecordStore};

/// Machine-readable manifest schema emitted by `--roster-export`.
const VERIFIED_ROSTER_SCHEMA: &str = "hyprstream/verified-service-roster@1";

/// One public roster member projected from the checkpoint-verified accepted
/// state read back from the store after admission. Contains no secret
/// material: only the admitted DID, epoch, bounded validity, and the
/// accepted-state head digest binding the entry to that exact state.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct VerifiedServiceRosterEntry {
    pub service: String,
    pub did: String,
    pub epoch: u64,
    pub expires_at: String,
    pub accepted_head_digest: String,
}

/// Public verified roster manifest. This document is not a trust root: it is a
/// convenience projection for tooling, and the authenticated checkpoint store
/// remains the authoritative source the values were verified against.
#[derive(serde::Serialize)]
struct VerifiedServiceRosterManifest {
    schema: &'static str,
    generated_at: String,
    services: Vec<VerifiedServiceRosterEntry>,
}

/// Provision or renew a complete local service roster. The database writer
/// lock excludes a running registry; callers must order this before services.
///
/// `roster_export` is opt-in: when absent, behavior is unchanged. When given,
/// a JSON manifest of only the verified readback public fields is written
/// atomically (temporary file + rename); any member failure or unsafe output
/// path fails the whole command without leaving partial or new output.
pub fn provision_services(
    config: &HyprConfig,
    services: &[String],
    valid_for_seconds: i64,
    roster_export: Option<&Path>,
) -> Result<()> {
    ensure!(
        (600..=86_400).contains(&valid_for_seconds),
        "service identity lifetime must be 600..86400 seconds"
    );
    validate_service_roster(services)?;
    let verifier = hyprstream_discovery::authenticate_local_deployment_registry()?;
    let secrets = provisioning_secrets_dir(config)?;
    let acceptance =
        load_existing_service_signing_key(&secrets, "registry", SecretsProfile::SharedDirectory)?;
    ensure!(
        verifier.matches(&acceptance.verifying_key()),
        "registry signing key does not match authenticated deployment credential"
    );
    // Load the entire roster before writing any state; missing keys are errors,
    // never a reason to silently generate a replacement service identity.
    let keys = services
        .iter()
        .map(|name| {
            load_existing_service_signing_key(&secrets, name, SecretsProfile::SharedDirectory)
                .map(|key| (name, key))
        })
        .collect::<Result<Vec<_>>>()?;
    let directory = hyprstream_service::deployment_data_dir()?.join("pds-store");
    let probe =
        PdsRecordStore::open_readonly(&directory)?.with_at9p_deployment_verifier(verifier.clone());
    ensure!(
        probe.first_boot_pending()? || !probe.accepted_at9p_states()?.is_empty(),
        "empty unmarked checkpoint store requires explicit initialization or recovery"
    );
    drop(probe);
    let store =
        Arc::new(PdsRecordStore::open(&directory)?.with_at9p_deployment_verifier(verifier.clone()));
    let audit = hyprstream_rpc::node_identity::derive_purpose_key(
        &acceptance,
        "hyprstream-at9p-audit-ed25519-v1",
    );
    let audit_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&audit);
    let ingest = At9pStateIngest::open(
        Arc::clone(&store),
        &directory.join("at9p-duplicity.wal"),
        acceptance,
        audit,
        audit_pq,
    )?;
    let now = Utc::now();
    let mut admitted = Vec::new();
    for (name, key) in &keys {
        admitted.push((
            name.as_str(),
            key,
            provision_one(&store, &ingest, name, key, now, valid_for_seconds)?,
        ));
    }
    // Verify every roster member, not just the first DID that clears the
    // first-boot marker. No service starts while this writer is held. The
    // manifest is built only from these verified readback states, never from
    // the provisioned inputs.
    let now_text = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    let entries = build_verified_roster_entries(&store, &admitted, &now_text)?;
    if let Some(path) = roster_export {
        write_verified_roster_manifest(path, entries)?;
        tracing::info!(path = %path.display(), "verified service roster manifest exported");
    }
    Ok(())
}

/// Read the current accepted roster without acquiring a writer or admitting
/// state. Returns a complete public JSON document only after every member has
/// passed the existing deployment/checkpoint/key-binding/freshness validators.
/// The caller emits these buffered bytes; no output or store is written here.
pub fn inspect_services(config: &HyprConfig, services: &[String]) -> Result<Vec<u8>> {
    validate_service_roster(services)?;
    let verifier = hyprstream_discovery::authenticate_local_deployment_registry()?;
    let secrets = provisioning_secrets_dir(config)?;
    let keys = load_roster_keys(&secrets, services)?;
    let directory = hyprstream_service::deployment_data_dir()?.join("pds-store");
    let store = PdsRecordStore::open_readonly(&directory)?
        .with_at9p_deployment_verifier(verifier);
    inspect_verified_roster(&store, &keys, &Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true))
}

fn validate_service_roster(services: &[String]) -> Result<()> {
    ensure!(!services.is_empty(), "at least one service is required");
    let mut unique = std::collections::HashSet::new();
    for name in services {
        ensure!(unique.insert(name), "duplicate service in bootstrap roster");
        ensure!(
            hyprstream_service::get_factory(name).is_some(),
            "unknown service in bootstrap roster: {name}"
        );
    }
    Ok(())
}

fn load_roster_keys(secrets: &Path, services: &[String]) -> Result<Vec<(String, SigningKey)>> {
    services.iter().map(|name| {
        load_existing_service_signing_key(secrets, name, SecretsProfile::SharedDirectory)
            .map(|key| (name.clone(), key))
    }).collect()
}

fn inspect_verified_roster(
    store: &PdsRecordStore,
    keys: &[(String, SigningKey)],
    now_text: &str,
) -> Result<Vec<u8>> {
    // Enumerate through the authenticated checkpoint path, never deserialize
    // raw DB bytes or use an editable prior export as authority. Missing or
    // ambiguous service IDs cannot silently become first boot or a new DID.
    let states = store.accepted_at9p_states()?;
    let mut admitted = Vec::with_capacity(keys.len());
    for (name, key) in keys {
        let service_id = format!("#{name}");
        let mut matching = states.iter().filter(|state| {
            state.current.services.iter().any(|service| service.id == service_id)
        });
        let state = matching.next().with_context(|| format!("no accepted identity for service {name}"))?;
        ensure!(matching.next().is_none(), "multiple accepted identities match service {name}");
        admitted.push((name.as_str(), key, state.clone()));
    }
    let entries = build_verified_roster_entries(store, &admitted, now_text)?;
    let mut bytes = serialize_verified_roster_manifest(entries)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn serialize_verified_roster_manifest(services: Vec<VerifiedServiceRosterEntry>) -> Result<Vec<u8>> {
    ensure!(!services.is_empty(), "refusing to export an empty verified service roster");
    let manifest = VerifiedServiceRosterManifest {
        schema: VERIFIED_ROSTER_SCHEMA,
        generated_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        services,
    };
    Ok(serde_json::to_vec_pretty(&manifest)?)
}

/// Re-read and re-verify every admitted member from the checkpoint store,
/// projecting only public fields. Any missing, mismatched, expired, or
/// unbound member fails the whole roster.
fn build_verified_roster_entries(
    store: &PdsRecordStore,
    admitted: &[(&str, &SigningKey, AcceptedAt9pState)],
    now_text: &str,
) -> Result<Vec<VerifiedServiceRosterEntry>> {
    let mut entries = Vec::with_capacity(admitted.len());
    for (name, key, state) in admitted {
        let verified = store
            .accepted_at9p_state(&state.did, Some(now_text))?
            .context("provisioned service state disappeared")?;
        hyprstream_service::NativeServiceAnnouncement::from_accepted_state(name, key, &verified)?;
        tracing::info!(service = name, did = %verified.did, epoch = verified.epoch, "checkpoint-accepted service identity ready");
        let expires_at = verified
            .expires_at
            .clone()
            .context("verified accepted state lacks bounded expiry")?;
        entries.push(VerifiedServiceRosterEntry {
            service: (*name).to_owned(),
            did: verified.did,
            epoch: verified.epoch,
            expires_at,
            accepted_head_digest: hex::encode(verified.head_digest),
        });
    }
    Ok(entries)
}

/// Atomically write the verified roster manifest: serialize fully, write to a
/// sibling temporary file (created exclusively), sync, then rename. A failure
/// at any step never creates, truncates, or replaces the target path. Only a
/// temporary file this invocation successfully created is ever removed; if
/// exclusive creation fails because the predictable sibling already exists,
/// that preexisting file belongs to someone else and is left untouched.
fn write_verified_roster_manifest(
    path: &Path,
    services: Vec<VerifiedServiceRosterEntry>,
) -> Result<()> {
    ensure!(
        !services.is_empty(),
        "refusing to export an empty verified service roster"
    );
    ensure!(
        !path.is_dir(),
        "roster export path is a directory: {}",
        path.display()
    );
    let file_name = path
        .file_name()
        .context("roster export path has no file name")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    ensure!(
        parent.is_dir(),
        "roster export parent directory does not exist: {}",
        parent.display()
    );
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    let bytes = serialize_verified_roster_manifest(services)?;
    // Ownership boundary: until the exclusive create succeeds, `temp` is not
    // ours — a collision means a leftover from an interrupted earlier process
    // (possibly with a reused PID) or another entry in the caller's directory,
    // and must survive this failed invocation.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .with_context(|| format!("create roster export temporary {}", temp.display()))?;
    publish_owned_roster_temp(file, &temp, path, &bytes)
}

/// Publish through a temporary file this invocation created exclusively. On
/// any failure before the rename lands, only that owned file is removed;
/// nothing else in the directory is touched.
fn publish_owned_roster_temp(
    mut file: std::fs::File,
    temp: &Path,
    path: &Path,
    bytes: &[u8],
) -> Result<()> {
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(temp, path)
            .with_context(|| format!("publish roster export {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}

/// The command is dispatched after `main` has loaded and validated the
/// operator-selected configuration. Reusing that exact value keeps an explicit
/// `--config` `[secrets].path` authoritative rather than reloading defaults.
fn provisioning_secrets_dir(config: &HyprConfig) -> Result<std::path::PathBuf> {
    HyprConfig::resolve_secrets_dir_for(Some(config))
}

fn provision_one(
    store: &PdsRecordStore,
    ingest: &At9pStateIngest,
    service_name: &str,
    signer: &SigningKey,
    now: DateTime<Utc>,
    valid_for_seconds: i64,
) -> Result<AcceptedAt9pState> {
    let service_id = format!("#{service_name}");
    let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(signer);
    let hybrid = HybridKeyPair::new(
        signer.verifying_key().to_bytes().to_vec(),
        ml_dsa_sk_to_vk_bytes(&pq),
    )?;
    let commitment = hybrid.commitment_digest();
    let carrier =
        hyprstream_rpc::node_identity::derive_purpose_key(signer, "hyprstream-iroh-transport-v1");
    let mut endpoint = ServiceEndpoint::new(
        Transport::Iroh,
        format!("iroh://{}", hex::encode(carrier.verifying_key().to_bytes())),
    )?;
    endpoint.request_kem = Some(
        hyprstream_rpc::node_identity::derive_mesh_kem_recipient(signer)?.public().encode(),
    );
    let service = ServiceEntry::new(&service_id, ServiceType::NinePExport, endpoint)?;
    let states = store.accepted_at9p_states()?;
    let mut matching = states.into_iter().filter(|state| {
        state
            .current
            .services
            .iter()
            .any(|entry| entry.id == service_id)
    });
    let existing = matching.next();
    ensure!(
        matching.next().is_none(),
        "multiple accepted identities match service {service_name}"
    );
    let state = match existing {
        Some(state) => {
            ensure!(
                state.current.subject_keys.contains(&hybrid),
                "accepted service hybrid key does not match the provisioned signer"
            );
            state
        }
        None => {
            let mut body = CapsuleBody::new(vec![hybrid.clone()], vec![service.clone()])?;
            body.next_key_commitments = vec![commitment];
            let genesis = sign_capsule(body, signer, &pq)?;
            let bytes = genesis.to_dag_cbor()?;
            let did = format!("did:at9p:{}", genesis.cid512()?);
            ingest.ingest_genesis(&did, &bytes)?
        }
    };
    let renew_after = now + Duration::seconds(valid_for_seconds / 2);
    if state.epoch > 0
        && state.current.services.iter().any(|entry| entry == &service)
        && state
            .expires_at
            .as_deref()
            .map(DateTime::parse_from_rfc3339)
            .transpose()?
            .is_some_and(|expiry| expiry >= renew_after)
    {
        return Ok(state);
    }
    ensure!(
        !state.terminal,
        "service {service_name} identity has no authorized successor"
    );
    ensure!(
        state.current.next_key_commitments.contains(&commitment),
        "provisioned service key is not committed for the next update"
    );
    // Preserve every other subject key, service entry and future commitment.
    let mut body = state.current.clone();
    let entry = body
        .services
        .iter_mut()
        .find(|entry| entry.id == service_id)
        .context("accepted service entry disappeared")?;
    *entry = service;
    let expires =
        (now + Duration::seconds(valid_for_seconds)).to_rfc3339_opts(SecondsFormat::Secs, true);
    let update = sign_update_record(
        state.subject_cid512.clone(),
        state
            .epoch
            .checked_add(1)
            .context("service identity epoch exhausted")?,
        state.head_digest,
        body,
        expires,
        signer,
        &pq,
    )?;
    ingest.ingest_successor(
        &state.did,
        &update.to_dag_cbor()?,
        &now.to_rfc3339_opts(SecondsFormat::Secs, true),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Inventory contents and names (including directories) without relying on
    // atime, which a read may legitimately update at the filesystem layer.
    fn filesystem_contents(root: &Path) -> Result<std::collections::BTreeMap<std::path::PathBuf, Option<Vec<u8>>>> {
        fn visit(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<std::path::PathBuf, Option<Vec<u8>>>) -> Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                let relative = path.strip_prefix(root)?.to_path_buf();
                if entry.file_type()?.is_dir() {
                    out.insert(relative, None);
                    visit(root, &path, out)?;
                } else {
                    out.insert(relative, Some(std::fs::read(path)?));
                }
            }
            Ok(())
        }
        let mut out = std::collections::BTreeMap::new();
        visit(root, root, &mut out)?;
        Ok(out)
    }

    #[test]
    fn readonly_roster_reads_verified_state_without_changing_store_or_keys() -> Result<()> {
        let (dir, store, ingest, model) = fixture()?;
        let event = SigningKey::from_bytes(&[0x63; 32]);
        let now = Utc::now();
        let accepted_model = provision_one(&store, &ingest, "model", &model, now, 86400)?;
        let accepted_event = provision_one(&store, &ingest, "event", &event, now, 86400)?;
        let credentials = dir.path().join("retained-credentials");
        crate::auth::identity_store::write_secret(&credentials.join("model"), "signing-key", &model.to_bytes())?;
        crate::auth::identity_store::write_secret(&credentials.join("event"), "signing-key", &event.to_bytes())?;
        drop(ingest);
        drop(store);
        let before = filesystem_contents(dir.path())?;
        let keys = load_roster_keys(&credentials, &["model".into(), "event".into()])?;
        let readonly = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x61; 32]).verifying_key());
        let bytes = inspect_verified_roster(&readonly, &keys, &now.to_rfc3339_opts(SecondsFormat::Secs, true))?;
        let document: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(document["schema"], VERIFIED_ROSTER_SCHEMA);
        assert_eq!(document["services"].as_array().context("services")?.len(), 2);
        for (index, state) in [accepted_model, accepted_event].iter().enumerate() {
            assert_eq!(document["services"][index]["did"], state.did);
            assert_eq!(document["services"][index]["epoch"], state.epoch);
            assert_eq!(document["services"][index]["accepted_head_digest"], hex::encode(state.head_digest));
            assert_eq!(document["services"][index]["expires_at"], state.expires_at.as_deref().context("expiry")?);
        }
        assert!(!String::from_utf8(bytes)?.contains(&hex::encode(model.to_bytes())));
        assert_eq!(filesystem_contents(dir.path())?, before);
        Ok(())
    }

    #[test]
    fn readonly_roster_failure_preserves_state_and_never_initializes_missing_input() -> Result<()> {
        let missing_root = tempfile::tempdir()?;
        let before = filesystem_contents(missing_root.path())?;
        assert!(PdsRecordStore::open_readonly(&missing_root.path().join("missing-store")).is_err());
        assert!(load_roster_keys(missing_root.path(), &["model".into()]).is_err());
        assert_eq!(filesystem_contents(missing_root.path())?, before);
        let empty = missing_root.path().join("empty-store");
        std::fs::create_dir(&empty)?;
        let before = filesystem_contents(missing_root.path())?;
        assert!(PdsRecordStore::open_readonly(&empty).is_err());
        assert_eq!(filesystem_contents(missing_root.path())?, before);
        std::fs::write(empty.join("CURRENT"), b"invalid current manifest")?;
        let before = filesystem_contents(missing_root.path())?;
        assert!(PdsRecordStore::open_readonly(&empty).is_err());
        assert_eq!(filesystem_contents(missing_root.path())?, before);

        let (dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        provision_one(&store, &ingest, "model", &key, now, 86400)?;
        drop(ingest);
        drop(store);
        let before = filesystem_contents(dir.path())?;
        let readonly = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x61; 32]).verifying_key());
        // A successful first member cannot produce a partial JSON result when
        // a later requested service is missing or its retained key mismatches.
        assert!(inspect_verified_roster(&readonly, &[("model".into(), key.clone()), ("event".into(), key.clone())], &now.to_rfc3339_opts(SecondsFormat::Secs, true)).is_err());
        assert!(inspect_verified_roster(&readonly, &[("model".into(), SigningKey::from_bytes(&[0x77; 32]))], &now.to_rfc3339_opts(SecondsFormat::Secs, true)).is_err());
        assert!(inspect_verified_roster(&readonly, &[("model".into(), key.clone())], &(now + Duration::hours(25)).to_rfc3339_opts(SecondsFormat::Secs, true)).is_err());
        let wrong_verifier = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x78; 32]).verifying_key());
        assert!(inspect_verified_roster(&wrong_verifier, &[("model".into(), key)], &now.to_rfc3339_opts(SecondsFormat::Secs, true)).is_err());
        assert_eq!(filesystem_contents(dir.path())?, before);
        Ok(())
    }

    #[test]
    fn readonly_roster_rejects_corrupt_checkpoint_and_preserves_failure_evidence() -> Result<()> {
        let (dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let accepted = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        drop(ingest);
        drop(store);
        // Tamper only in fixture setup, then close the writer before capturing
        // the failed reader's complete on-disk before/after evidence.
        let db = rocksdb::DB::open(&rocksdb::Options::default(), dir.path())?;
        db.put(format!("at9p-checkpoint\0{}", accepted.subject_cid512), b"invalid checkpoint")?;
        drop(db);
        let before = filesystem_contents(dir.path())?;
        let readonly = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x61; 32]).verifying_key());
        assert!(inspect_verified_roster(&readonly, &[("model".into(), key)], &now.to_rfc3339_opts(SecondsFormat::Secs, true)).is_err());
        assert_eq!(filesystem_contents(dir.path())?, before);
        Ok(())
    }

    #[test]
    fn readonly_roster_rejects_empty_first_boot_store_without_consuming_marker() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = PdsRecordStore::open(dir.path())?;
        store.mark_first_boot()?;
        drop(store);
        let before = filesystem_contents(dir.path())?;
        let readonly = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x61; 32]).verifying_key());
        assert!(inspect_verified_roster(&readonly, &[("model".into(), SigningKey::from_bytes(&[0x62; 32]))], &Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)).is_err());
        assert!(readonly.first_boot_pending()?);
        assert_eq!(filesystem_contents(dir.path())?, before);
        assert!(validate_service_roster(&[]).is_err());
        assert!(validate_service_roster(&["model".into(), "model".into()]).is_err());
        assert!(validate_service_roster(&["unknown-service".into()]).is_err());
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_rejects_invalid_roster_and_lifetime_before_credentials() {
        let config = HyprConfig::default();
        assert!(provision_services(&config, &["model".to_owned()], 599, None).is_err());
        assert!(provision_services(&config, &["model".to_owned()], 86401, None).is_err());
        assert!(provision_services(&config, &[], 86400, None).is_err());
        assert!(
            provision_services(&config, &["model".to_owned(), "model".to_owned()], 86400, None)
                .is_err()
        );
        assert!(
            provision_services(&config, &["../not-a-service".to_owned()], 86400, None).is_err()
        );
    }

    #[test]
    fn deployment_bootstrap_uses_loaded_config_for_secrets_dir() -> Result<()> {
        let mut config = HyprConfig::default();
        let custom_secrets = tempfile::tempdir()?.path().join("custom-secrets");
        config.secrets.path = Some(custom_secrets.clone());
        assert_eq!(provisioning_secrets_dir(&config)?, custom_secrets);
        Ok(())
    }

    fn fixture() -> Result<(
        tempfile::TempDir,
        Arc<PdsRecordStore>,
        At9pStateIngest,
        SigningKey,
    )> {
        let dir = tempfile::tempdir()?;
        let acceptance = SigningKey::from_bytes(&[0x61; 32]);
        let store = Arc::new(
            PdsRecordStore::open(dir.path())?
                .with_at9p_acceptance_identity(acceptance.verifying_key()),
        );
        let audit = hyprstream_rpc::node_identity::derive_purpose_key(
            &acceptance,
            "hyprstream-at9p-audit-ed25519-v1",
        );
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&audit);
        let ingest = At9pStateIngest::open(
            Arc::clone(&store),
            &dir.path().join("audit.wal"),
            acceptance,
            audit,
            pq,
        )?;
        Ok((dir, store, ingest, SigningKey::from_bytes(&[0x62; 32])))
    }

    #[test]
    fn deployment_bootstrap_admits_bounded_successor_and_reuses_identity() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        assert_eq!(first.epoch, 1);
        assert!(!first.terminal);
        assert_eq!(first.current.services[0].endpoint.request_kem.as_deref(),
            Some(hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&key)?.public().encode().as_slice()));
        hyprstream_service::NativeServiceAnnouncement::from_accepted_state("model", &key, &first)?;
        let retry = provision_one(
            &store,
            &ingest,
            "model",
            &key,
            now + Duration::minutes(1),
            86400,
        )?;
        assert_eq!(retry.did, first.did);
        assert_eq!(retry.head_digest, first.head_digest);
        let renewed = provision_one(
            &store,
            &ingest,
            "model",
            &key,
            now + Duration::hours(13),
            86400,
        )?;
        assert_eq!(renewed.did, first.did);
        assert_eq!(renewed.epoch, 2);
        assert_ne!(renewed.head_digest, first.head_digest);
        assert_eq!(store.accepted_at9p_states()?.len(), 1);
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_resumes_after_genesis_only_partial_progress() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        // Deliberately invalid internal update lifetime simulates interruption
        // after durable genesis; the public command rejects this input earlier.
        assert!(provision_one(&store, &ingest, "model", &key, now, -1).is_err());
        let states = store.accepted_at9p_states()?;
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].epoch, 0);
        let resumed = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        assert_eq!(resumed.did, states[0].did);
        assert_eq!(resumed.epoch, 1);
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_rejects_wrong_key_without_duplicate_genesis() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        let wrong_key = SigningKey::from_bytes(&[0x65; 32]);
        let error = provision_one(&store, &ingest, "model", &wrong_key, now, 86400)
            .err().context("an enrolled service cannot be treated as absent after a key change")?;
        assert!(error.to_string().contains("hybrid key does not match"));
        let states = store.accepted_at9p_states()?;
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].did, first.did);
        assert_eq!(states[0].epoch, first.epoch);
        assert_eq!(states[0].head_digest, first.head_digest);
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_rejects_multiple_service_identities_across_keys() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        // Simulate a pre-existing duplicate, admitted through the real signed
        // genesis/checkpoint path, with the same service ID but a different key.
        let other = SigningKey::from_bytes(&[0x66; 32]);
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&other);
        let pair = HybridKeyPair::new(other.verifying_key().to_bytes().to_vec(), ml_dsa_sk_to_vk_bytes(&pq))?;
        let body = CapsuleBody::new(vec![pair], first.current.services.clone())?;
        let capsule = sign_capsule(body, &other, &pq)?;
        let duplicate = ingest.ingest_genesis(&format!("did:at9p:{}", capsule.cid512()?), &capsule.to_dag_cbor()?)?;
        let error = provision_one(&store, &ingest, "model", &key, now, 86400)
            .err().context("matching a key cannot hide a second service authority")?;
        assert!(error.to_string().contains("multiple accepted identities"));
        let states = store.accepted_at9p_states()?;
        assert_eq!(states.len(), 2);
        assert!(states.iter().any(|state| state.head_digest == first.head_digest));
        assert!(states.iter().any(|state| state.head_digest == duplicate.head_digest));
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_checkpoint_survives_reopen_and_covers_roster() -> Result<()> {
        let (dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        let second_key = SigningKey::from_bytes(&[0x63; 32]);
        let second = provision_one(&store, &ingest, "event", &second_key, now, 86400)?;
        assert_ne!(first.did, second.did);
        drop(ingest);
        drop(store);
        let reopened = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x61; 32]).verifying_key());
        let states = reopened.accepted_at9p_states()?;
        assert_eq!(states.len(), 2);
        assert!(states.iter().all(|state| state.epoch == 1));
        let wrong = PdsRecordStore::open_readonly(dir.path())?
            .with_at9p_acceptance_identity(SigningKey::from_bytes(&[0x64; 32]).verifying_key());
        assert!(wrong.accepted_at9p_states().is_err());
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_never_generates_missing_service_keys() -> Result<()> {
        let dir = tempfile::tempdir()?;
        assert!(load_existing_service_signing_key(
            dir.path(),
            "model",
            SecretsProfile::SharedDirectory
        )
        .is_err());
        assert!(load_existing_service_signing_key(
            dir.path(),
            "policy",
            SecretsProfile::SharedDirectory
        )
        .is_err());
        assert!(!dir.path().join("model").exists());
        assert!(!dir.path().join("signing-key").exists());
        let model_dir = dir.path().join("model");
        crate::auth::identity_store::write_secret(&model_dir, "signing-key", &[0x65; 32])?;
        let loaded = load_existing_service_signing_key(
            dir.path(),
            "model",
            SecretsProfile::SharedDirectory,
        )?;
        assert_eq!(
            loaded.verifying_key(),
            SigningKey::from_bytes(&[0x65; 32]).verifying_key()
        );
        assert!(load_existing_service_signing_key(
            dir.path(),
            "../model",
            SecretsProfile::SharedDirectory
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_rejects_existing_hybrid_mismatch_without_replacing_did() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let other_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(
            &SigningKey::from_bytes(&[0x66; 32]),
        );
        let hybrid = HybridKeyPair::new(
            key.verifying_key().to_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&other_pq),
        )?;
        let endpoint = ServiceEndpoint::new(
            Transport::Iroh,
            format!("iroh://{}", hex::encode([0x67; 32])),
        )?;
        let service = ServiceEntry::new("#model", ServiceType::NinePExport, endpoint)?;
        let body = CapsuleBody::new(vec![hybrid], vec![service])?;
        let genesis = sign_capsule(body, &key, &other_pq)?;
        let did = format!("did:at9p:{}", genesis.cid512()?);
        ingest.ingest_genesis(&did, &genesis.to_dag_cbor()?)?;
        assert!(provision_one(&store, &ingest, "model", &key, Utc::now(), 86400).is_err());
        let states = store.accepted_at9p_states()?;
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].did, did);
        assert_eq!(states[0].epoch, 0);
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_roster_export_matches_verified_readback_not_inputs() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let now_text = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let event_key = SigningKey::from_bytes(&[0x63; 32]);
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        let event = provision_one(&store, &ingest, "event", &event_key, now, 86400)?;
        let admitted = [
            ("model", &key, first.clone()),
            ("event", &event_key, event.clone()),
        ];
        let entries = build_verified_roster_entries(&store, &admitted, &now_text)?;
        assert_eq!(entries.len(), 2);
        // Every exported field equals an independent store readback of the
        // accepted state, not the values handed to the provisioner.
        let model_readback = store
            .accepted_at9p_state(&first.did, Some(&now_text))?
            .context("model readback")?;
        assert_eq!(entries[0].service, "model");
        assert_eq!(entries[0].did, model_readback.did);
        assert_eq!(entries[0].epoch, model_readback.epoch);
        assert_eq!(
            entries[0].expires_at,
            model_readback.expires_at.clone().context("model expiry")?
        );
        assert_eq!(
            entries[0].accepted_head_digest,
            hex::encode(model_readback.head_digest)
        );
        assert_eq!(entries[1].service, "event");
        assert_eq!(entries[1].did, event.did);
        assert_ne!(entries[0].did, entries[1].did);
        // Provenance: after a real signed renewal advances the store, the
        // rebuilt export tracks the new verified state even though the stale
        // in-memory `first` value is still passed in as the admitted input.
        let renewed = provision_one(
            &store,
            &ingest,
            "model",
            &key,
            now + Duration::hours(13),
            86400,
        )?;
        assert_eq!(renewed.epoch, first.epoch + 1);
        let admitted = [
            ("model", &key, first.clone()),
            ("event", &event_key, event),
        ];
        let entries = build_verified_roster_entries(&store, &admitted, &now_text)?;
        assert_eq!(entries[0].did, renewed.did);
        assert_eq!(entries[0].epoch, renewed.epoch);
        assert_eq!(
            entries[0].accepted_head_digest,
            hex::encode(renewed.head_digest)
        );
        assert_eq!(
            entries[0].expires_at,
            renewed.expires_at.clone().context("renewed expiry")?
        );
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_roster_export_is_atomic_and_never_partial() -> Result<()> {
        let (dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let now_text = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        let out_dir = tempfile::tempdir()?;
        let target = out_dir.path().join("roster.json");
        // A member whose key is not the accepted key fails the whole export;
        // the target and any temporary sibling must not exist afterwards.
        let wrong_key = SigningKey::from_bytes(&[0x65; 32]);
        let admitted = [("model", &wrong_key, first.clone())];
        let error = build_verified_roster_entries(&store, &admitted, &now_text)
            .err()
            .context("unaccepted key must fail roster verification")?;
        assert!(error.to_string().contains("accepted current hybrid"));
        let write_result = build_verified_roster_entries(&store, &admitted, &now_text)
            .and_then(|entries| write_verified_roster_manifest(&target, entries));
        assert!(write_result.is_err());
        assert!(!target.exists());
        assert!(std::fs::read_dir(out_dir.path())?.next().is_none());
        // A failed renewal-time export leaves a previously published manifest
        // byte-identical instead of clobbering it with partial content.
        let good = [("model", &key, first.clone())];
        let entries = build_verified_roster_entries(&store, &good, &now_text)?;
        write_verified_roster_manifest(&target, entries)?;
        let published = std::fs::read(&target)?;
        let failed = build_verified_roster_entries(&store, &admitted, &now_text)
            .and_then(|entries| write_verified_roster_manifest(&target, entries));
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&target)?, published);
        // Unsafe output targets fail without touching the store directory.
        let entries = build_verified_roster_entries(&store, &good, &now_text)?;
        assert!(write_verified_roster_manifest(dir.path(), entries.clone()).is_err());
        assert!(
            write_verified_roster_manifest(
                &out_dir.path().join("missing-parent").join("roster.json"),
                entries,
            )
            .is_err()
        );
        assert!(!out_dir.path().join("missing-parent").exists());
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_roster_export_rejects_expired_and_unbounded_state() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        // Readback at a time beyond the bounded validity fails the export.
        let later = (now + Duration::days(2)).to_rfc3339_opts(SecondsFormat::Secs, true);
        let admitted = [("model", &key, first)];
        let error = build_verified_roster_entries(&store, &admitted, &later)
            .err()
            .context("expired accepted state must fail roster export")?;
        assert!(error.to_string().contains("expired"));
        // A genesis-only accepted state has no bounded validity and is refused.
        let genesis = SigningKey::from_bytes(&[0x67; 32]);
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&genesis);
        let pair = HybridKeyPair::new(
            genesis.verifying_key().to_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&pq),
        )?;
        let endpoint = ServiceEndpoint::new(
            Transport::Iroh,
            format!("iroh://{}", hex::encode([0x68; 32])),
        )?;
        let service = ServiceEntry::new("#event", ServiceType::NinePExport, endpoint)?;
        let body = CapsuleBody::new(vec![pair], vec![service])?;
        let capsule = sign_capsule(body, &genesis, &pq)?;
        let genesis_did = format!("did:at9p:{}", capsule.cid512()?);
        let genesis_state =
            ingest.ingest_genesis(&genesis_did, &capsule.to_dag_cbor()?)?;
        assert_eq!(genesis_state.epoch, 0);
        assert!(genesis_state.expires_at.is_none());
        let now_text = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let admitted = [("event", &genesis, genesis_state)];
        let error = build_verified_roster_entries(&store, &admitted, &now_text)
            .err()
            .context("genesis-only state must fail roster export")?;
        assert!(error.to_string().contains("no bounded production expiry"));
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_roster_manifest_is_public_json_only() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let now_text = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        let admitted = [("model", &key, first.clone())];
        let entries = build_verified_roster_entries(&store, &admitted, &now_text)?;
        let out_dir = tempfile::tempdir()?;
        let target = out_dir.path().join("roster.json");
        write_verified_roster_manifest(&target, entries)?;
        let text = std::fs::read_to_string(&target)?;
        let parsed: serde_json::Value = serde_json::from_str(&text)?;
        assert_eq!(parsed["schema"], VERIFIED_ROSTER_SCHEMA);
        let services = parsed["services"].as_array().context("services array")?;
        assert_eq!(services.len(), 1);
        assert_eq!(services[0]["service"], "model");
        assert_eq!(services[0]["did"], first.did);
        assert_eq!(
            services[0]["epoch"],
            serde_json::Value::from(first.epoch)
        );
        assert_eq!(
            services[0]["expires_at"].as_str().context("expires_at")?,
            first.expires_at.as_deref().context("first expiry")?
        );
        assert_eq!(
            services[0]["accepted_head_digest"],
            hex::encode(first.head_digest)
        );
        // The manifest must never leak private key material: neither the raw
        // signing seed nor the derived ML-DSA secret appears in any form.
        assert!(!text.contains(&hex::encode([0x62; 32])));
        assert!(!text.contains("signing-key"));
        Ok(())
    }

    /// The predictable sibling temp path of the current process, exactly as
    /// `write_verified_roster_manifest` computes it.
    fn roster_temp_sibling(target: &Path) -> Result<std::path::PathBuf> {
        let file_name = target
            .file_name()
            .context("roster target has no file name")?;
        Ok(target.with_file_name(format!(
            ".{}.tmp-{}",
            file_name.to_string_lossy(),
            std::process::id()
        )))
    }

    #[test]
    fn deployment_bootstrap_roster_export_collision_preserves_foreign_sibling_and_target(
    ) -> Result<()> {
        let out_dir = tempfile::tempdir()?;
        let target = out_dir.path().join("roster.json");
        // A leftover from an interrupted earlier process (possibly a reused
        // PID) occupies the predictable temp path, and an older published
        // manifest occupies the target. The failed invocation must not delete
        // or modify either: it never owned the sibling.
        let foreign = roster_temp_sibling(&target)?;
        std::fs::write(&foreign, b"interrupted earlier process")?;
        std::fs::write(&target, b"previous good manifest")?;
        let error = write_verified_roster_manifest(
            &target,
            vec![VerifiedServiceRosterEntry {
                service: "model".to_owned(),
                did: "did:at9p:collision".to_owned(),
                epoch: 1,
                expires_at: "2099-01-01T00:00:00Z".to_owned(),
                accepted_head_digest: hex::encode([0x42; 64]),
            }],
        )
        .err()
        .context("temp collision must fail the export")?;
        assert!(error.to_string().contains("create roster export temporary"));
        assert_eq!(std::fs::read(&foreign)?, b"interrupted earlier process");
        assert_eq!(std::fs::read(&target)?, b"previous good manifest");
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_roster_export_failure_after_create_removes_only_owned_temp(
    ) -> Result<()> {
        let out_dir = tempfile::tempdir()?;
        let target = out_dir.path().join("roster.json");
        let temp = roster_temp_sibling(&target)?;
        // Simulate ownership exactly as the writer acquires it, then fail the
        // publication step: the rename destination's parent does not exist.
        let owned = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let missing_target = out_dir.path().join("gone").join("roster.json");
        let result = publish_owned_roster_temp(owned, &temp, &missing_target, b"{}");
        assert!(result.is_err());
        assert!(!temp.exists(), "owned temp must be removed after failure");
        // An unrelated preexisting sibling entry is never cleanup scope.
        let foreign = out_dir.path().join(".roster.json.tmp-foreign");
        std::fs::write(&foreign, b"not ours")?;
        let owned = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        assert!(publish_owned_roster_temp(owned, &temp, &missing_target, b"{}").is_err());
        assert!(!temp.exists());
        assert_eq!(std::fs::read(&foreign)?, b"not ours");
        // Success control: the owned path publishes and removes the temp.
        let owned = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        publish_owned_roster_temp(owned, &temp, &target, b"{\"ok\":true}")?;
        assert!(!temp.exists());
        assert_eq!(std::fs::read(&target)?, b"{\"ok\":true}\n");
        Ok(())
    }

    #[test]
    fn deployment_bootstrap_roster_export_success_control_publishes_atomically() -> Result<()> {
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        let now_text = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let first = provision_one(&store, &ingest, "model", &key, now, 86400)?;
        let admitted = [("model", &key, first.clone())];
        let entries = build_verified_roster_entries(&store, &admitted, &now_text)?;
        let out_dir = tempfile::tempdir()?;
        let target = out_dir.path().join("roster.json");
        write_verified_roster_manifest(&target, entries)?;
        // Successful publication leaves exactly the target: no temp sibling.
        let names = std::fs::read_dir(out_dir.path())?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<std::io::Result<Vec<_>>>()?;
        assert_eq!(names, [std::ffi::OsString::from("roster.json")]);
        let parsed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&target)?)?;
        assert_eq!(parsed["services"][0]["did"], first.did);
        Ok(())
    }
}
