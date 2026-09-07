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
use std::sync::Arc;

use crate::auth::identity_store::{load_existing_service_signing_key, SecretsProfile};
use crate::config::HyprConfig;
use crate::services::discovery::{At9pStateIngest, PdsRecordStore};

/// Provision or renew a complete local service roster. The database writer
/// lock excludes a running registry; callers must order this before services.
pub fn provision_services(
    config: &HyprConfig,
    services: &[String],
    valid_for_seconds: i64,
) -> Result<()> {
    ensure!(
        (600..=86_400).contains(&valid_for_seconds),
        "service identity lifetime must be 600..86400 seconds"
    );
    ensure!(!services.is_empty(), "at least one service is required");
    let mut unique = std::collections::HashSet::new();
    for name in services {
        ensure!(unique.insert(name), "duplicate service in bootstrap roster");
        ensure!(
            hyprstream_service::get_factory(name).is_some(),
            "unknown service in bootstrap roster: {name}"
        );
    }
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
    // first-boot marker. No service starts while this writer is held.
    let now_text = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    for (name, key, state) in admitted {
        let verified = store
            .accepted_at9p_state(&state.did, Some(&now_text))?
            .context("provisioned service state disappeared")?;
        hyprstream_service::NativeServiceAnnouncement::from_accepted_state(name, key, &verified)?;
        tracing::info!(service = name, did = %verified.did, epoch = verified.epoch, "checkpoint-accepted service identity ready");
    }
    Ok(())
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
            && state
                .current
                .subject_keys
                .iter()
                .any(|key| key.ed25519_pub == hybrid.ed25519_pub)
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

    #[test]
    fn deployment_bootstrap_rejects_invalid_roster_and_lifetime_before_credentials() {
        let config = HyprConfig::default();
        assert!(provision_services(&config, &["model".to_owned()], 599).is_err());
        assert!(provision_services(&config, &["model".to_owned()], 86401).is_err());
        assert!(provision_services(&config, &[], 86400).is_err());
        assert!(
            provision_services(&config, &["model".to_owned(), "model".to_owned()], 86400).is_err()
        );
        assert!(provision_services(&config, &["../not-a-service".to_owned()], 86400).is_err());
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
}
