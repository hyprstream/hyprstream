//! Concrete reader for daemon-authenticated, monotonic-checkpointed accepted
//! `did:at9p` state. This module is private so production authority cannot be
//! implemented, constructed, or installed by a downstream crate.

use anyhow::{bail, Context as _, Result};
use hyprstream_pds::at9p::h512;
use hyprstream_pds::at9p_duplicity::{AcceptedAt9pState, Watermark};
use hyprstream_pds::at9p_gate::DID_AT9P_PREFIX;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const AT9P_STATE_MAGIC: &[u8; 8] = b"AT9PST02";
const AT9P_STATE_HEADER_LEN: usize = 8 + 1 + 8 + 64 + 1 + 4;
const AT9P_ACCEPTANCE_MAGIC: &[u8; 8] = b"AT9PAC01";
const ED25519_KEY_LEN: usize = 32;
const ED25519_SIGNATURE_LEN: usize = 64;
const AT9P_ACCEPTANCE_PREFIX_LEN: usize =
    AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN + ED25519_SIGNATURE_LEN;
const AT9P_ACCEPTANCE_KEY_AAD: &[u8] = b"hyprstream-at9p-acceptance-key/1";
const AT9P_ACCEPTANCE_STATE_AAD: &[u8] = b"hyprstream-at9p-accepted-state/1";
const AT9P_CHECKPOINT_MAGIC: &[u8; 8] = b"AT9PCK01";
const AT9P_CHECKPOINT_PAYLOAD_LEN: usize = 8 + 8 + 64 + 1 + 64;
const AT9P_CHECKPOINT_LEN: usize = AT9P_CHECKPOINT_PAYLOAD_LEN + ED25519_SIGNATURE_LEN;
const AT9P_CHECKPOINT_AAD: &[u8] = b"hyprstream-at9p-monotonic-checkpoint/1";

pub(super) struct CheckpointedPdsAcceptedStateSource {
    path: PathBuf,
    acceptance_identity: Arc<dyn AcceptanceVerifier>,
}

trait AcceptanceVerifier: Send + Sync {
    fn verify_strict(&self, message: &[u8], signature: &ed25519_dalek::Signature) -> Result<()>;
}

impl AcceptanceVerifier for crate::service::RegistryDeploymentVerifier {
    fn verify_strict(&self, message: &[u8], signature: &ed25519_dalek::Signature) -> Result<()> {
        self.verify_strict(message, signature)
    }
}

#[cfg(test)]
impl AcceptanceVerifier for ed25519_dalek::VerifyingKey {
    fn verify_strict(&self, message: &[u8], signature: &ed25519_dalek::Signature) -> Result<()> {
        self.verify_strict(message, signature).map_err(Into::into)
    }
}

impl CheckpointedPdsAcceptedStateSource {
    pub(super) fn open(
        path: &Path,
        acceptance_identity: crate::service::RegistryDeploymentVerifier,
    ) -> Result<Self> {
        if !path.exists() {
            // Fresh node (first boot — e.g. the #1137 metal stack): the
            // registry service is the sole writer and would create this store
            // on its first start, but the production resolver installs BEFORE
            // any service starts, so a missing store must not deadlock the
            // boot. An empty store IS the genesis state (no accepted at9p
            // state yet). Create it and drop the write handle immediately so
            // the later read-only probe and the writer's own open are
            // uncontended. NOTE: on a node that previously held state this
            // branch means the duplicity history was LOST (deleted volume) —
            // log loudly; the boot proceeds at genesis posture exactly as a
            // first boot would.
            tracing::warn!(
                path = %path.display(),
                "checkpointed PDS store absent — initializing an empty (genesis) store; \
                 if this node previously held at9p state, duplicity history was lost"
            );
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            drop(
                rocksdb::DB::open(&opts, path).with_context(|| {
                    format!("failed to initialize checkpointed PDS store at {path:?}")
                })?,
            );
        }
        let _probe = rocksdb::DB::open_for_read_only(&readonly_opts(), path, false)
            .with_context(|| format!("failed to open checkpointed PDS store at {path:?}"))?;
        Ok(Self {
            path: path.to_path_buf(),
            acceptance_identity: Arc::new(acceptance_identity),
        })
    }

    #[cfg(test)]
    pub(super) fn open_test(
        path: &Path,
        acceptance_identity: ed25519_dalek::VerifyingKey,
    ) -> Result<Self> {
        let _probe = rocksdb::DB::open_for_read_only(&readonly_opts(), path, false)
            .with_context(|| format!("failed to open checkpointed PDS store at {path:?}"))?;
        Ok(Self { path: path.to_path_buf(), acceptance_identity: Arc::new(acceptance_identity) })
    }

    pub(super) fn accepted_state(&self, did: &str) -> Result<Option<AcceptedAt9pState>> {
        let subject = did
            .strip_prefix(DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow::anyhow!("identifier is not did:at9p: {did:?}"))?;
        let db = rocksdb::DB::open_for_read_only(&readonly_opts(), &self.path, false)?;
        let state = load_at9p_state_from_db(&db, subject, self.acceptance_identity.as_ref())?;
        if let Some(state) = &state {
            anyhow::ensure!(state.did == did, "accepted at9p state DID mismatch");
        }
        Ok(state)
    }
}

impl super::service::AcceptedStateSource for CheckpointedPdsAcceptedStateSource {
    fn accepted_state(&self, did: &str) -> Result<Option<AcceptedAt9pState>> {
        self.accepted_state(did)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct At9pCheckpoint {
    watermark: Watermark,
    envelope_digest: [u8; 64],
}

fn readonly_opts() -> rocksdb::Options {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(false);
    opts
}

fn state_key(subject: &str) -> Vec<u8> {
    format!("at9p-state\0{subject}").into_bytes()
}

fn checkpoint_key(subject: &str) -> Vec<u8> {
    format!("at9p-checkpoint\0{subject}").into_bytes()
}

fn acceptance_message(domain: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(domain.len() + payload.len());
    message.extend_from_slice(domain);
    message.extend_from_slice(payload);
    message
}

fn checkpoint_message(subject: &str, payload: &[u8]) -> Result<Vec<u8>> {
    let len = u32::try_from(subject.len()).context("at9p subject exceeds u32")?;
    let mut out = Vec::with_capacity(AT9P_CHECKPOINT_AAD.len() + 4 + subject.len() + payload.len());
    out.extend_from_slice(AT9P_CHECKPOINT_AAD);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(subject.as_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

fn load_at9p_state_from_db(
    db: &rocksdb::DB,
    subject: &str,
    identity: &dyn AcceptanceVerifier,
) -> Result<Option<AcceptedAt9pState>> {
    let snapshot = db.snapshot();
    let envelope = snapshot.get(state_key(subject))?;
    let checkpoint = snapshot.get(checkpoint_key(subject))?;
    match (envelope, checkpoint) {
        (None, None) => Ok(None),
        (Some(_), None) => bail!("accepted at9p state exists without its monotonic checkpoint"),
        (None, Some(_)) => bail!("accepted at9p checkpoint exists without its state envelope"),
        (Some(envelope), Some(checkpoint)) => {
            let state = decode_state(subject, &envelope, identity)?;
            let checkpoint = decode_checkpoint(subject, &checkpoint, identity)?;
            anyhow::ensure!(
                checkpoint.watermark == state.watermark(),
                "accepted at9p checkpoint/body watermark mismatch"
            );
            anyhow::ensure!(
                checkpoint.envelope_digest == h512(&envelope),
                "accepted at9p checkpoint/state envelope mismatch"
            );
            Ok(Some(state))
        }
    }
}

fn decode_checkpoint(
    subject: &str,
    bytes: &[u8],
    identity: &dyn AcceptanceVerifier,
) -> Result<At9pCheckpoint> {
    anyhow::ensure!(
        bytes.len() == AT9P_CHECKPOINT_LEN,
        "accepted at9p checkpoint length mismatch"
    );
    let payload = bytes
        .get(..AT9P_CHECKPOINT_PAYLOAD_LEN)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p checkpoint payload missing"))?;
    anyhow::ensure!(
        payload.starts_with(AT9P_CHECKPOINT_MAGIC),
        "accepted at9p checkpoint has bad version"
    );
    let signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(AT9P_CHECKPOINT_PAYLOAD_LEN..)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p checkpoint signature missing"))?
            .try_into()?,
    );
    identity
        .verify_strict(&checkpoint_message(subject, payload)?, &signature)
        .context("accepted at9p monotonic checkpoint signature rejected")?;
    let epoch = u64::from_be_bytes(
        payload
            .get(8..16)
            .ok_or_else(|| anyhow::anyhow!("checkpoint epoch missing"))?
            .try_into()?,
    );
    let record_digest = payload
        .get(16..80)
        .ok_or_else(|| anyhow::anyhow!("checkpoint digest missing"))?
        .try_into()?;
    let terminal = match payload.get(80) {
        Some(0) => false,
        Some(1) => true,
        _ => bail!("invalid checkpoint terminal flag"),
    };
    Ok(At9pCheckpoint {
        watermark: Watermark {
            epoch,
            record_digest,
            terminal,
        },
        envelope_digest: payload
            .get(81..145)
            .ok_or_else(|| anyhow::anyhow!("checkpoint envelope digest missing"))?
            .try_into()?,
    })
}

fn decode_state(
    subject: &str,
    bytes: &[u8],
    acceptance_identity: &dyn AcceptanceVerifier,
) -> Result<AcceptedAt9pState> {
    anyhow::ensure!(
        bytes.len() >= AT9P_ACCEPTANCE_PREFIX_LEN + AT9P_STATE_HEADER_LEN + ED25519_SIGNATURE_LEN,
        "accepted at9p envelope is truncated"
    );
    anyhow::ensure!(
        bytes.starts_with(AT9P_ACCEPTANCE_MAGIC),
        "accepted at9p envelope has bad version"
    );
    let audit_key_bytes: [u8; ED25519_KEY_LEN] = bytes
        .get(AT9P_ACCEPTANCE_MAGIC.len()..AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing audit key"))?
        .try_into()?;
    let key_signature_start = AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN;
    let key_signature_end = key_signature_start + ED25519_SIGNATURE_LEN;
    let key_signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(key_signature_start..key_signature_end)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing key certificate"))?
            .try_into()?,
    );
    acceptance_identity
        .verify_strict(
            &acceptance_message(AT9P_ACCEPTANCE_KEY_AAD, &audit_key_bytes),
            &key_signature,
        )
        .context("accepted at9p audit-key certificate rejected")?;
    let audit_key = ed25519_dalek::VerifyingKey::from_bytes(&audit_key_bytes)
        .context("accepted at9p envelope has malformed audit key")?;
    let state_signature_start = bytes.len() - ED25519_SIGNATURE_LEN;
    let body = bytes
        .get(AT9P_ACCEPTANCE_PREFIX_LEN..state_signature_start)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing state body"))?;
    let state_signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(state_signature_start..)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing state signature"))?
            .try_into()?,
    );
    audit_key
        .verify_strict(
            &acceptance_message(AT9P_ACCEPTANCE_STATE_AAD, body),
            &state_signature,
        )
        .context("accepted at9p daemon acceptance signature rejected")?;
    decode_state_body(subject, body)
}

fn decode_state_body(subject: &str, bytes: &[u8]) -> Result<AcceptedAt9pState> {
    anyhow::ensure!(
        bytes.len() >= AT9P_STATE_HEADER_LEN,
        "accepted at9p state is truncated"
    );
    anyhow::ensure!(
        bytes.starts_with(AT9P_STATE_MAGIC),
        "accepted at9p state has bad version"
    );
    let kind = *bytes
        .get(8)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing kind"))?;
    let epoch = u64::from_be_bytes(
        bytes
            .get(9..17)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing epoch"))?
            .try_into()?,
    );
    let digest: [u8; 64] = bytes
        .get(17..81)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing digest"))?
        .try_into()?;
    let terminal = match bytes.get(81) {
        Some(0) => false,
        Some(1) => true,
        _ => bail!("accepted at9p state has invalid terminal flag"),
    };
    let head_len = u32::from_be_bytes(
        bytes
            .get(82..86)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing length"))?
            .try_into()?,
    ) as usize;
    anyhow::ensure!(
        bytes.len() == AT9P_STATE_HEADER_LEN + head_len,
        "accepted at9p state length mismatch"
    );
    let head = bytes
        .get(AT9P_STATE_HEADER_LEN..)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing head"))?;
    let state = match kind {
        0 => AcceptedAt9pState::from_persisted_genesis(head)?,
        1 => AcceptedAt9pState::from_persisted_update(head)?,
        _ => bail!("accepted at9p state has unknown head kind {kind}"),
    };
    anyhow::ensure!(
        state.subject_cid512 == subject,
        "accepted at9p state subject mismatch"
    );
    anyhow::ensure!(
        state.epoch == epoch,
        "accepted at9p state epoch/body mismatch"
    );
    anyhow::ensure!(
        state.head_digest == digest,
        "accepted at9p state digest/body mismatch"
    );
    anyhow::ensure!(
        state.terminal == terminal,
        "accepted at9p state terminal/body mismatch"
    );
    Ok(state)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer as _;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType,
        Transport as At9pTransport,
    };
    use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};

    fn accepted_state() -> AcceptedAt9pState {
        let signing = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
        let (pq_signing, pq_verifying) = ml_dsa_generate_keypair();
        let keys = HybridKeyPair::new(
            signing.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_verifying),
        )
        .expect("hybrid keys");
        let endpoint = ServiceEndpoint::new(At9pTransport::Iroh, "iroh://checkpointed-reach")
            .expect("service endpoint");
        let service = ServiceEntry::new("#checkpointed", ServiceType::NinePExport, endpoint)
            .expect("service entry");
        let capsule = sign_capsule(
            CapsuleBody::new(vec![keys], vec![service]).expect("capsule body"),
            &signing,
            &pq_signing,
        )
        .expect("signed capsule");
        AcceptedAt9pState::from_persisted_genesis(&capsule.to_dag_cbor().expect("capsule bytes"))
            .expect("accepted genesis")
    }

    fn encode_body(state: &AcceptedAt9pState) -> Vec<u8> {
        let (kind, head) = match state.head() {
            hyprstream_pds::at9p_duplicity::AcceptedAt9pHead::Genesis(capsule) => {
                (0, capsule.to_dag_cbor().expect("genesis bytes"))
            }
            hyprstream_pds::at9p_duplicity::AcceptedAt9pHead::Update(update) => {
                (1, update.to_dag_cbor().expect("update bytes"))
            }
        };
        let mut out = Vec::new();
        out.extend_from_slice(AT9P_STATE_MAGIC);
        out.push(kind);
        out.extend_from_slice(&state.epoch.to_be_bytes());
        out.extend_from_slice(&state.head_digest);
        out.push(u8::from(state.terminal));
        out.extend_from_slice(
            &u32::try_from(head.len())
                .expect("head length")
                .to_be_bytes(),
        );
        out.extend_from_slice(&head);
        out
    }

    #[test]
    fn update_fixture_encodes_and_decodes_update_kind() {
        let genesis = accepted_state();
        let signing = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
        let (pq_signing, pq_verifying) = ml_dsa_generate_keypair();
        let keys = HybridKeyPair::new(
            signing.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_verifying),
        )
        .expect("hybrid keys");
        let endpoint = ServiceEndpoint::new(At9pTransport::Iroh, "iroh://updated-reach")
            .expect("service endpoint");
        let service = ServiceEntry::new("#updated", ServiceType::NinePExport, endpoint)
            .expect("service entry");
        let update = sign_update_record(
            genesis.subject_cid512.clone(),
            1,
            genesis.head_digest,
            CapsuleBody::new(vec![keys], vec![service]).expect("capsule body"),
            "2099-01-01T00:00:00Z".to_owned(),
            &signing,
            &pq_signing,
        )
        .expect("signed update");
        let state = AcceptedAt9pState::from_persisted_update(
            &update.to_dag_cbor().expect("update bytes"),
        )
        .expect("accepted update");
        assert_eq!(encode_body(&state)[AT9P_STATE_MAGIC.len()], 1);

        let identity = ed25519_dalek::SigningKey::from_bytes(&[0x45; 32]);
        let audit = ed25519_dalek::SigningKey::from_bytes(&[0x46; 32]);
        let (envelope, _) = encode_fixture(&state, &identity, &audit);
        let decoded = decode_state(&state.subject_cid512, &envelope, &identity.verifying_key())
            .expect("decode update fixture");
        assert!(matches!(
            decoded.head(),
            hyprstream_pds::at9p_duplicity::AcceptedAt9pHead::Update(_)
        ));
    }

    fn encode_fixture(
        state: &AcceptedAt9pState,
        identity: &ed25519_dalek::SigningKey,
        audit: &ed25519_dalek::SigningKey,
    ) -> (Vec<u8>, Vec<u8>) {
        let body = encode_body(state);
        let audit_key = audit.verifying_key().to_bytes();
        let key_signature = identity.sign(&acceptance_message(AT9P_ACCEPTANCE_KEY_AAD, &audit_key));
        let state_signature = audit.sign(&acceptance_message(AT9P_ACCEPTANCE_STATE_AAD, &body));
        let mut envelope = Vec::new();
        envelope.extend_from_slice(AT9P_ACCEPTANCE_MAGIC);
        envelope.extend_from_slice(&audit_key);
        envelope.extend_from_slice(&key_signature.to_bytes());
        envelope.extend_from_slice(&body);
        envelope.extend_from_slice(&state_signature.to_bytes());

        let mut payload = Vec::new();
        payload.extend_from_slice(AT9P_CHECKPOINT_MAGIC);
        payload.extend_from_slice(&state.epoch.to_be_bytes());
        payload.extend_from_slice(&state.head_digest);
        payload.push(u8::from(state.terminal));
        payload.extend_from_slice(&h512(&envelope));
        let checkpoint_signature = identity.sign(
            &checkpoint_message(&state.subject_cid512, &payload).expect("checkpoint message"),
        );
        let mut checkpoint = payload;
        checkpoint.extend_from_slice(&checkpoint_signature.to_bytes());
        (envelope, checkpoint)
    }

    #[test]
    fn concrete_checkpointed_pds_reader_accepts_only_matching_daemon_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let identity = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]);
        let audit = ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]);
        let state = accepted_state();
        let (envelope, checkpoint) = encode_fixture(&state, &identity, &audit);
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        let db = rocksdb::DB::open(&options, dir.path()).expect("open fixture store");
        db.put(state_key(&state.subject_cid512), envelope)
            .expect("write accepted state");
        db.put(checkpoint_key(&state.subject_cid512), checkpoint)
            .expect("write checkpoint");
        drop(db);

        let source = CheckpointedPdsAcceptedStateSource::open_test(dir.path(), identity.verifying_key())
            .expect("open checkpointed source");
        let recovered = source
            .accepted_state(&state.did)
            .expect("read checkpointed state")
            .expect("state present");
        assert_eq!(recovered.watermark(), state.watermark());

        let wrong_identity = ed25519_dalek::SigningKey::from_bytes(&[0x44; 32]);
        let rejected =
            CheckpointedPdsAcceptedStateSource::open_test(dir.path(), wrong_identity.verifying_key())
                .expect("open same store")
                .accepted_state(&state.did);
        assert!(
            rejected.is_err(),
            "caller-selected checkpoint identity accepted"
        );
    }
}
