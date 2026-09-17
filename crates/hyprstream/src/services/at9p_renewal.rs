//! In-daemon at9p service-identity renewal timer (design O-Ia).
//!
//! The registry daemon is the deployment's sole read-write record-store
//! owner, so it hosts one periodic task that renews every DEPLOYMENT service
//! identity itself (O-Ia) — the accepted states whose `#service` entries
//! name daemon services (`hyprstream_service::get_factory`), not generic
//! at9p records admitted through the public RPC — reusing the boot
//! provisioner's mint core
//! ([`crate::cli::deployment_bootstrap::provision_one`]) against the daemon's
//! LIVE `PdsRecordStore`/`At9pStateIngest` handles. It must never call
//! `provision_services` as-is — that re-opens the store (self-conflicting on
//! the RocksDB directory LOCK the daemon already holds) and re-runs
//! `authenticate_local_deployment_registry` per invocation, which would drag
//! the ≤1 h credential / 30 d delegation dependency back into every tick.
//! The deployment credential/UCAN is checked exactly once per process start;
//! admission itself needs only the pre-committed on-volume chain keys and the
//! daemon clock, so a running process renews for its lifetime while restarts
//! still gate on the credential (unchanged by this timer).
//!
//! Trust property (review note): signing sibling services' successors
//! routinely from the long-lived registry process differs in process context
//! — not in authority — from the offline exclusive-window mint. The signing
//! material is identical (each service's on-volume key, pre-committed in
//! `next_key_commitments`), and every admission invariant still holds:
//! key mismatch refuses (never substitutes), terminal identity refuses, the
//! same key carries forward, successors are `epoch+1`, expiry is evaluated
//! on the daemon's clock with no skew tolerance, and the commit is the same
//! guarded CAS as the RPC path.
//!
//! Every tick also re-exports the roster projection
//! (`<data>/config/native-network/roster.json` by default) through the same
//! readback-verified atomic writer as the boot command, so host-side
//! monitoring that flags a roster whose `generated_at` ages past a day reads
//! fresh expiries — and a stale roster means this loop died, not that
//! nothing was due. The boot-time `roster.candidate.json` → `config.toml`
//! native-config projection stays boot-scoped host tooling: runtime consumers
//! (native announcements) re-project from the store, never from that file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use ed25519_dalek::SigningKey;
use hyprstream_pds::at9p_duplicity::AcceptedAt9pState;

use crate::auth::identity_store::{load_existing_service_signing_key, SecretsProfile};
use crate::cli::deployment_bootstrap::{export_verified_roster_snapshot, provision_one};
use crate::services::discovery::{At9pStateIngest, PdsPublisher, PdsRecordStore};

/// Default roster projection refreshed every tick: the promoted manifest the
/// host-side renewal monitor reads, under the deployment's shared config
/// tree. The boot unit's `--roster-export` candidate flow is separate.
pub(crate) fn default_roster_export_path() -> Option<PathBuf> {
    hyprstream_rpc::paths::data_dir().map(|data| {
        data.join("config")
            .join("native-network")
            .join("roster.json")
    })
}

/// Spawn the renewal timer on the current tokio runtime (the registry
/// service's process runtime).
///
/// The task holds the publisher by [`Weak`] reference: a strong handle would
/// pin the store's RocksDB directory LOCK after the service stops and block
/// an in-process registry restart (the same lifetime discipline as the ES256
/// promotion hook). When the publisher drops, the next tick observes the
/// dead weak reference and the task exits.
///
/// The first check runs immediately (a daemon starting against states
/// already past their half-TTL window — e.g. after downtime — catches up
/// instead of waiting a full interval); subsequent checks run once per
/// `check_interval` with `MissedTickBehavior::Skip` (the JWT key-rotation
/// task idiom). Tick failures are logged and retried on the next interval.
/// Dropping the returned handle detaches (does not cancel) the task.
pub(crate) fn spawn_at9p_renewal_task(
    publisher: Arc<PdsPublisher>,
    secrets_dir: PathBuf,
    check_interval: std::time::Duration,
    valid_for_seconds: i64,
    roster_export: Option<PathBuf>,
) -> tokio::task::JoinHandle<()> {
    let publisher = Arc::downgrade(&publisher);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(check_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // The first interval tick completes immediately; consume it so the
        // loop below always waits a full interval after each check.
        interval.tick().await;
        loop {
            let Some(publisher) = publisher.upgrade() else {
                tracing::info!(
                    "at9p service-identity renewal: registry publisher dropped; timer exiting"
                );
                break;
            };
            // The tick is fully synchronous RocksDB/Postgres + ML-DSA work;
            // run it on the blocking pool (the `ingestAt9pCandidate` handler
            // uses spawn_blocking for the same ingest operations) instead of
            // stalling an async worker — the immediate startup tick or a due
            // renewal would otherwise block unrelated registry requests.
            let tick_publisher = Arc::clone(&publisher);
            let tick_secrets = secrets_dir.clone();
            let tick_roster = roster_export.clone();
            let tick = tokio::task::spawn_blocking(move || {
                run_renewal_tick(
                    tick_publisher.as_ref(),
                    &tick_secrets,
                    valid_for_seconds,
                    tick_roster.as_deref(),
                    Utc::now(),
                )
            })
            .await;
            match tick {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    tracing::warn!(
                        %error,
                        "at9p service-identity renewal tick failed; retrying next interval"
                    );
                }
                Err(join_error) => {
                    // The blocking task itself failed (panic); retried on the
                    // next interval like any tick error.
                    let error = anyhow::anyhow!(join_error);
                    tracing::warn!(
                        %error,
                        "at9p service-identity renewal tick task failed; retrying next interval"
                    );
                }
            }
            drop(publisher);
            interval.tick().await;
        }
    })
}

/// One renewal tick against the daemon's live publisher handles.
pub(crate) fn run_renewal_tick(
    publisher: &PdsPublisher,
    secrets_dir: &Path,
    valid_for_seconds: i64,
    roster_export: Option<&Path>,
    now: DateTime<Utc>,
) -> Result<()> {
    let store = publisher.at9p_record_store();
    let ingest = publisher
        .at9p_state_ingest()
        .context("at9p renewal requires the daemon-owned state ingest")?;
    renewal_tick(
        store,
        ingest,
        secrets_dir,
        valid_for_seconds,
        roster_export,
        now,
    )
}

/// One renewal tick against explicit handles (the testable core): renew
/// every due identity, then re-export the roster projection. The export is
/// all-or-nothing by construction: a member that fails verification fails
/// the whole snapshot rather than emitting a partial manifest that would
/// hide a sick member from host-side monitoring.
pub(crate) fn renewal_tick(
    store: &PdsRecordStore,
    ingest: &At9pStateIngest,
    secrets_dir: &Path,
    valid_for_seconds: i64,
    roster_export: Option<&Path>,
    now: DateTime<Utc>,
) -> Result<()> {
    let admitted =
        renew_due_service_identities(store, ingest, secrets_dir, valid_for_seconds, now)?;
    if let Some(path) = roster_export {
        export_verified_roster_snapshot(store, &admitted, path)?;
    }
    Ok(())
}

/// Renew every accepted at9p service identity whose remaining life is below
/// the half-TTL window. The renewal set is the intersection of the accepted
/// states with the daemon's own service vocabulary
/// (`hyprstream_service::get_factory` — the same membership test the offline
/// provisioner's `validate_service_roster` applies): the store may also hold
/// generic at9p records admitted through the public `ingestAt9pCandidate`
/// RPC (arbitrary `#id` service entries that are NOT deployment identities),
/// and those must neither be renewed nor abort the tick — a foreign entry
/// with no credentials directory would otherwise fail every tick and starve
/// the real deployment identities past expiry, where renewal is impossible.
/// For the deployment's own services, keys are loaded, never generated: a
/// missing sibling key on the shared credentials volume is an error naming
/// the service, retried on the next tick. Inside the half-TTL window
/// `provision_one` is a verified no-op returning the current state unchanged.
pub(crate) fn renew_due_service_identities(
    store: &PdsRecordStore,
    ingest: &At9pStateIngest,
    secrets_dir: &Path,
    valid_for_seconds: i64,
    now: DateTime<Utc>,
) -> Result<Vec<(String, SigningKey, AcceptedAt9pState)>> {
    let states = store.accepted_at9p_states()?;
    let mut service_epochs: BTreeMap<String, u64> = BTreeMap::new();
    for state in &states {
        for entry in &state.current.services {
            let Some(name) = entry.id.strip_prefix('#') else {
                continue;
            };
            if hyprstream_service::get_factory(name).is_none() {
                tracing::debug!(
                    service = name,
                    "accepted at9p state is not a deployment service identity; not renewed"
                );
                continue;
            }
            service_epochs.insert(entry.id.clone(), state.epoch);
        }
    }
    // BTreeMap iteration is deterministic ('#' sorts before alphanumerics).
    let mut admitted = Vec::with_capacity(service_epochs.len());
    for (service_id, epoch_before) in &service_epochs {
        let name = service_id.strip_prefix('#').unwrap_or(service_id);
        let key =
            load_existing_service_signing_key(secrets_dir, name, SecretsProfile::SharedDirectory)
                .with_context(|| format!("loading existing signing key for service {name}"))?;
        let state = provision_one(store, ingest, name, &key, now, valid_for_seconds)
            .with_context(|| format!("renewing service identity for {name}"))?;
        if state.epoch > *epoch_before {
            tracing::info!(
                service = name,
                did = %state.did,
                epoch = state.epoch,
                "at9p service identity renewed"
            );
        }
        admitted.push((name.to_owned(), key, state));
    }
    Ok(admitted)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use chrono::Duration;

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

    fn write_service_key(dir: &Path, name: &str, key: &SigningKey) -> Result<()> {
        crate::auth::identity_store::write_secret(&dir.join(name), "signing-key", &key.to_bytes())?;
        Ok(())
    }

    fn sorted_states(store: &PdsRecordStore) -> Result<Vec<AcceptedAt9pState>> {
        let mut states = store.accepted_at9p_states()?;
        states.sort_by(|a, b| a.did.cmp(&b.did));
        Ok(states)
    }

    /// A live-shaped roster: two identities provisioned at T0 with `ttl`
    /// seconds, their keys on the shared credentials volume.
    #[allow(clippy::type_complexity)]
    fn provisioned_fixture(
        ttl: i64,
    ) -> Result<(
        tempfile::TempDir,
        Arc<PdsRecordStore>,
        At9pStateIngest,
        tempfile::TempDir,
        SigningKey,
        DateTime<Utc>,
    )> {
        let (dir, store, ingest, key) = fixture()?;
        let event_key = SigningKey::from_bytes(&[0x63; 32]);
        let now = Utc::now();
        provision_one(&store, &ingest, "model", &key, now, ttl)?;
        provision_one(&store, &ingest, "event", &event_key, now, ttl)?;
        let secrets = tempfile::tempdir()?;
        write_service_key(secrets.path(), "model", &key)?;
        write_service_key(secrets.path(), "event", &event_key)?;
        Ok((dir, store, ingest, secrets, key, now))
    }

    #[test]
    fn renewal_tick_is_noop_inside_half_ttl_window() -> Result<()> {
        let (_dir, store, ingest, secrets, _key, now) = provisioned_fixture(600)?;
        let out = tempfile::tempdir()?;
        let roster = out.path().join("roster.json");
        let before = sorted_states(&store)?;
        renewal_tick(&store, &ingest, secrets.path(), 600, Some(&roster), now)?;
        let after = sorted_states(&store)?;
        assert_eq!(after.len(), before.len());
        for (state_before, state_after) in before.iter().zip(after.iter()) {
            assert_eq!(state_before.did, state_after.did);
            assert_eq!(state_before.epoch, state_after.epoch);
            assert_eq!(state_before.head_digest, state_after.head_digest);
        }
        // The roster projection is still exported every tick (loop-liveness
        // signal for host-side monitoring), with both verified members.
        let parsed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&roster)?)?;
        assert_eq!(parsed["schema"], "hyprstream/verified-service-roster@1");
        assert_eq!(parsed["services"].as_array().expect("services").len(), 2);
        Ok(())
    }

    #[test]
    fn renewal_tick_extends_same_did_at_epoch_plus_one_with_later_expiry() -> Result<()> {
        let (_dir, store, ingest, secrets, _key, now) = provisioned_fixture(600)?;
        let out = tempfile::tempdir()?;
        let roster = out.path().join("roster.json");
        let first = sorted_states(&store)?;
        // Past the half-TTL window (600/2 = 300 s) both identities renew.
        let later = now + Duration::seconds(301);
        renewal_tick(&store, &ingest, secrets.path(), 900, Some(&roster), later)?;
        let renewed = sorted_states(&store)?;
        assert_eq!(renewed.len(), first.len());
        for (state_before, state_after) in first.iter().zip(renewed.iter()) {
            assert_eq!(
                state_after.did, state_before.did,
                "renewal must never churn DIDs"
            );
            assert_eq!(state_after.epoch, state_before.epoch + 1);
            assert_ne!(state_after.head_digest, state_before.head_digest);
            let expiry_before = state_before
                .expires_at
                .as_deref()
                .expect("bounded expiry before");
            let expiry_after = state_after
                .expires_at
                .as_deref()
                .expect("bounded expiry after");
            assert!(expiry_after > expiry_before, "expiry must move later");
            // Same key carried forward: the subject key set is unchanged and
            // therefore still pre-committed for the next renewal.
            assert_eq!(
                state_after.current.subject_keys,
                state_before.current.subject_keys
            );
        }
        // The exported roster tracks the renewed readback, not the inputs
        // (keyed by DID: the manifest order is the admitted order).
        let parsed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&roster)?)?;
        let services = parsed["services"].as_array().expect("services");
        assert_eq!(services.len(), 2);
        let by_did: std::collections::HashMap<&str, &serde_json::Value> = services
            .iter()
            .map(|entry| (entry["did"].as_str().expect("did"), entry))
            .collect();
        for state in &renewed {
            let entry = by_did[state.did.as_str()];
            assert_eq!(entry["epoch"], serde_json::Value::from(state.epoch));
            assert_eq!(
                entry["expires_at"],
                state.expires_at.as_deref().expect("expiry")
            );
        }
        // A second tick inside the new half-TTL window is a verified no-op.
        let epoch_snapshot: Vec<u64> = renewed.iter().map(|state| state.epoch).collect();
        renewal_tick(
            &store,
            &ingest,
            secrets.path(),
            900,
            Some(&roster),
            later + Duration::seconds(60),
        )?;
        assert_eq!(
            sorted_states(&store)?
                .into_iter()
                .map(|state| state.epoch)
                .collect::<Vec<_>>(),
            epoch_snapshot
        );
        Ok(())
    }

    #[test]
    fn renewal_tick_refuses_mismatched_key_without_substituting_identity() -> Result<()> {
        let (_dir, store, ingest, secrets, key, now) = provisioned_fixture(600)?;
        // Replace the on-volume key: the tick must refuse (never silently
        // treat the service as absent or mint a new DID for the new key).
        let wrong = SigningKey::from_bytes(&[0x77; 32]);
        write_service_key(secrets.path(), "model", &wrong)?;
        let out = tempfile::tempdir()?;
        let roster = out.path().join("roster.json");
        let later = now + Duration::seconds(301);
        let error = renewal_tick(&store, &ingest, secrets.path(), 900, Some(&roster), later)
            .expect_err("key mismatch must fail the tick");
        assert!(
            format!("{error:#}").contains("hybrid key does not match"),
            "unexpected error: {error:#}"
        );
        // The refused service is untouched: same DID, same epoch, and the
        // accepted key is still the original (the refusal never substitutes).
        let model = sorted_states(&store)?
            .into_iter()
            .find(|state| state.current.services.iter().any(|e| e.id == "#model"))
            .expect("model state");
        assert_eq!(model.epoch, 1);
        assert!(model
            .current
            .subject_keys
            .iter()
            .any(|k| { k.ed25519_pub.as_slice() == key.verifying_key().to_bytes().as_slice() }));
        // All-or-nothing export: the failed tick publishes no roster.
        assert!(!roster.exists());
        Ok(())
    }

    #[test]
    fn renewal_tick_never_generates_missing_service_keys() -> Result<()> {
        let (_dir, store, ingest, secrets, _key, now) = provisioned_fixture(600)?;
        // Remove one sibling's key from the shared credentials volume.
        std::fs::remove_file(secrets.path().join("event").join("signing-key"))?;
        let out = tempfile::tempdir()?;
        let roster = out.path().join("roster.json");
        let later = now + Duration::seconds(301);
        let error = renewal_tick(&store, &ingest, secrets.path(), 900, Some(&roster), later)
            .expect_err("missing key must fail the tick");
        assert!(
            format!("{error:#}").contains("event"),
            "error must name the service: {error:#}"
        );
        // Nothing was generated for the missing service.
        assert!(!secrets.path().join("event").join("signing-key").exists());
        assert_eq!(sorted_states(&store)?.len(), 2);
        assert!(!roster.exists());
        Ok(())
    }

    #[test]
    fn renewal_tick_skips_foreign_at9p_records_without_aborting() -> Result<()> {
        let (_dir, store, ingest, secrets, _key, now) = provisioned_fixture(600)?;
        // Admit a generic at9p capsule (public-RPC shape) whose service id is
        // NOT a deployment service: it has no credentials directory and must
        // neither be renewed nor fail the tick — a hard failure here would
        // starve the deployment identities past expiry on every retry.
        use hyprstream_pds::at9p::{
            CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
        };
        use hyprstream_pds::at9p_sign::sign_capsule;
        use hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes;
        let foreign = SigningKey::from_bytes(&[0x78; 32]);
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&foreign);
        let pair = HybridKeyPair::new(
            foreign.verifying_key().to_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&pq),
        )?;
        let endpoint = ServiceEndpoint::new(
            Transport::Iroh,
            format!("iroh://{}", hex::encode([0x79; 32])),
        )?;
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint)?;
        let body = CapsuleBody::new(vec![pair], vec![service])?;
        let capsule = sign_capsule(body, &foreign, &pq)?;
        let foreign_did = format!("did:at9p:{}", capsule.cid512()?);
        let foreign_state = ingest.ingest_genesis(&foreign_did, &capsule.to_dag_cbor()?)?;
        assert_eq!(foreign_state.epoch, 0);
        // Past the deployment half-TTL: both deployment identities renew and
        // the roster exports exactly the two deployment members.
        let out = tempfile::tempdir()?;
        let roster = out.path().join("roster.json");
        let later = now + Duration::seconds(301);
        renewal_tick(&store, &ingest, secrets.path(), 900, Some(&roster), later)?;
        let states = sorted_states(&store)?;
        assert_eq!(states.len(), 3);
        let foreign_after = states
            .iter()
            .find(|state| state.did == foreign_did)
            .expect("foreign record still accepted");
        assert_eq!(foreign_after.epoch, 0, "foreign record must not be renewed");
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&roster)?)?;
        let services = parsed["services"].as_array().expect("services");
        assert_eq!(services.len(), 2);
        assert!(
            services
                .iter()
                .all(|entry| entry["did"].as_str() != Some(foreign_did.as_str())),
            "foreign record must not appear in the deployment roster"
        );
        Ok(())
    }

    #[test]
    fn renewal_tick_refuses_terminal_identity() -> Result<()> {
        // 24 h base TTL so the sibling #event identity stays fresh across the
        // whole scenario; #model is made terminal below.
        let (_dir, store, ingest, secrets, key, now) = provisioned_fixture(86_400)?;
        // Advance #model to a terminal successor (empty next_key_commitments)
        // through the real ingest path, then require the tick to refuse once
        // the terminal state ages past the half-TTL window.
        use hyprstream_pds::at9p::CapsuleBody;
        use hyprstream_pds::at9p_sign::sign_update_record;
        let state = sorted_states(&store)?
            .into_iter()
            .find(|state| state.current.services.iter().any(|e| e.id == "#model"))
            .expect("model state");
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&key);
        let mut body = CapsuleBody::new(
            state.current.subject_keys.clone(),
            state.current.services.clone(),
        )?;
        body.next_key_commitments = vec![]; // terminal: no authorized successor
        let signed_at = now + Duration::seconds(60);
        let expires = (signed_at + Duration::seconds(7200))
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let update = sign_update_record(
            state.subject_cid512.clone(),
            state.epoch + 1,
            state.head_digest,
            body,
            expires,
            &key,
            &pq,
        )?;
        ingest.ingest_successor(
            &state.did,
            &update.to_dag_cbor()?,
            &signed_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        )?;
        // Remaining life 400 s < half of the tick's 900 s TTL: renewal is
        // due and must refuse on the terminal invariant while still fresh.
        let tick_at = signed_at + Duration::seconds(6800);
        let out = tempfile::tempdir()?;
        let roster = out.path().join("roster.json");
        let error = renewal_tick(&store, &ingest, secrets.path(), 900, Some(&roster), tick_at)
            .expect_err("terminal identity must fail the tick");
        assert!(
            format!("{error:#}").contains("no authorized successor"),
            "unexpected error: {error:#}"
        );
        assert!(!roster.exists());
        Ok(())
    }

    #[tokio::test]
    async fn renewal_task_exits_when_publisher_drops() -> Result<()> {
        // The store directory must outlive the task's final tick.
        let (_dir, store, ingest, key) = fixture()?;
        let now = Utc::now();
        provision_one(&store, &ingest, "model", &key, now, 600)?;
        let secrets = tempfile::tempdir()?;
        write_service_key(secrets.path(), "model", &key)?;
        let p256 = p256::ecdsa::SigningKey::from_slice(&[0x71; 32])?;
        let publisher = Arc::new(
            PdsPublisher::new(Arc::clone(&store), "did:key:test".to_owned(), p256)
                .with_at9p_state_ingest(ingest),
        );
        let roster = tempfile::tempdir()?;
        let handle = spawn_at9p_renewal_task(
            Arc::clone(&publisher),
            secrets.keep(),
            std::time::Duration::from_millis(10),
            900,
            Some(roster.keep().join("roster.json")),
        );
        // Dropping the last strong publisher reference must end the task on a
        // subsequent tick: a surviving task would pin the store's RocksDB
        // directory LOCK and block an in-process registry restart.
        drop(publisher);
        let exited = tokio::time::timeout(std::time::Duration::from_secs(30), handle).await;
        assert!(
            exited.is_ok(),
            "renewal task must exit after the publisher drops"
        );
        Ok(())
    }
}
