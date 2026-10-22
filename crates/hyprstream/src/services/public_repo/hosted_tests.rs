#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]
use super::*;
use p256::ecdsa::signature::Signer;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const DID: &str = "did:web:alice.accounts.example.com";

pub(crate) struct Reads(pub AtomicUsize, pub AtomicUsize);
impl hyprstream_pds_service::AccountRecordReadAuthorizer for Reads {
    fn check_read(
        &self,
        _: &hyprstream_rpc::Subject,
        _: Option<&str>,
        _: Option<&hyprstream_rpc::auth::mac::SecurityContext>,
        object: &str,
    ) -> hyprstream_rpc::auth::mac::MacDecision {
        self.0.fetch_add(1, Ordering::SeqCst);
        if object.ends_with(hyprstream_pds::ATPROTO_SIGNING_KEY_FILE) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
        hyprstream_rpc::auth::mac::MacDecision::Permit
    }
}

pub(crate) async fn authority_fixture(
) -> (Arc<hyprstream_pds_service::AccountRecordStore>, Arc<Reads>) {
    authority_fixture_with(GenesisFixture::Current).await
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GenesisFixture {
    Current,
    Legacy,
    CorruptNative,
    InvalidNativeSignature,
    ForeignNative,
    ProgressedNative,
    NonemptyNative,
    CorruptPublic,
    WrongKey,
}

pub(crate) async fn authority_fixture_with(
    mode: GenesisFixture,
) -> (Arc<hyprstream_pds_service::AccountRecordStore>, Arc<Reads>) {
    use hyprstream_pds::did_op::*;
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};
    let ed = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    let (pq, vk) = hyprstream_crypto::pq::ml_dsa_generate_keypair();
    let rotations = GenesisRotationKeys::new(
        UserRotationKey::new(
            HybridRotationKey::new(
                ed.verifying_key().to_bytes(),
                hyprstream_crypto::pq::ml_dsa_vk_bytes(&vk),
            )
            .unwrap(),
        ),
        RecoveryKeyEnrollment::Declined,
        HostKeyEnrollment::Absent,
    )
    .unwrap();
    let mint = hyprstream_pds::HostedAccountMint::begin(
        hyprstream_pds::AllocatedAccountName::new("alice", DID).unwrap(),
        rotations,
    )
    .unwrap();
    let doc = mint.seal_did_document("https://pds.example.com").unwrap();
    let (pending, genesis) = mint.prepare_pds_genesis(doc, Tid::from_raw(1)).unwrap();
    let signature = sign_genesis(pending.unsigned_genesis(), &ed, &pq).unwrap();
    let account = pending.seal(signature).unwrap();
    let mut native = genesis.commit_bytes().to_vec();
    let mut unsigned = genesis.commit().unsigned();
    match mode {
        GenesisFixture::CorruptNative => native = b"corrupt native".to_vec(),
        GenesisFixture::InvalidNativeSignature => {
            let mut commit = genesis.commit().clone();
            commit.sig[0] ^= 1;
            native = commit.to_dag_cbor();
        }
        GenesisFixture::ForeignNative => unsigned.did = "did:web:bob.accounts.example.com".into(),
        GenesisFixture::ProgressedNative => unsigned.prev = Some(genesis.commit_cid()),
        GenesisFixture::NonemptyNative => unsigned.data = Cid::from_dag_cbor(b"not empty"),
        _ => {}
    }
    if matches!(
        mode,
        GenesisFixture::ForeignNative
            | GenesisFixture::ProgressedNative
            | GenesisFixture::NonemptyNative
    ) {
        native = Commit::sign(&unsigned, account.atproto_signing_key()).to_dag_cbor();
    }
    let mut repo = SyntheticNode::dir().with_child("commit.cbor", SyntheticNode::file(native));
    if matches!(
        mode,
        GenesisFixture::Current | GenesisFixture::CorruptPublic
    ) {
        let public = if matches!(mode, GenesisFixture::CorruptPublic) {
            b"corrupt public".to_vec()
        } else {
            genesis.public_commit_bytes().to_vec()
        };
        repo = repo.with_child("public-commit.cbor", SyntheticNode::file(public));
    }
    let secret = if matches!(mode, GenesisFixture::WrongKey) {
        p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng)
            .to_bytes()
            .to_vec()
    } else {
        account.atproto_signing_key().to_bytes().to_vec()
    };
    let root = SyntheticNode::dir().with_child(
        "tenant",
        SyntheticNode::dir().with_child(
            "accounts",
            SyntheticNode::dir().with_child(
                "alice",
                SyntheticNode::dir()
                    .with_child(
                        "account-record.cbor",
                        SyntheticNode::file(account.record_bytes().to_vec()),
                    )
                    .with_child(
                        hyprstream_pds::ATPROTO_SIGNING_KEY_FILE,
                        SyntheticNode::file(secret),
                    )
                    .with_child("repo", repo),
            ),
        ),
    );
    let reads = Arc::new(Reads(AtomicUsize::new(0), AtomicUsize::new(0)));
    let store = Arc::new(hyprstream_pds_service::AccountRecordStore::new(
        Arc::new(SyntheticMount::new(root)),
        reads.clone(),
    ));
    store.refresh_hosted_did_index(&authority()).await.unwrap();
    reads.0.store(0, Ordering::SeqCst);
    (store, reads)
}

fn authority() -> hyprstream_rpc::Subject {
    hyprstream_rpc::Subject::new(hyprstream_pds_service::OAUTH_ACCOUNT_RESOLVER_SUBJECT)
}
pub(crate) fn request(id: usize) -> PublicCreateRequest {
    PublicCreateRequest { request_id: format!("hosted-{id}"), principal: DID.into(), did: DID.into(), collection: "app.bsky.feed.post".into(), rkey: Some(AtprotoRecordKey::new(format!("post-{id}")).unwrap()), value: json_to_dag_cbor(&serde_json::json!({"$type":"app.bsky.feed.post","text":"hosted", "createdAt":"2026-09-13T00:00:00Z"})).unwrap(), expected_prev: None }
}
pub(crate) fn hosted(
    store: Arc<PublicRepoStore>,
    accounts: Arc<hyprstream_pds_service::AccountRecordStore>,
) -> HostedAccountPublicRepoWriter {
    HostedAccountPublicRepoWriter::new(
        store,
        accounts,
        authority(),
        Arc::new(HostedAccountSelfAuthorizer),
    )
}
fn external(store: Arc<PublicRepoStore>, key: &p256::ecdsa::SigningKey) -> PublicRepoWriter {
    PublicRepoWriter::new_external(
        store,
        DID,
        *key.verifying_key(),
        Arc::new(HostedAccountSelfAuthorizer),
    )
    .unwrap()
}
fn pending(writer: &PublicRepoWriter, request: PublicCreateRequest) -> Box<PublicPendingCreate> {
    match writer
        .prepare_external_record_with_expected_prev_text(request, None)
        .unwrap()
    {
        PublicCreatePreparation::Pending(p) => p,
        _ => panic!("expected a new preparation"),
    }
}
fn finish(
    writer: &PublicRepoWriter,
    pending: Box<PublicPendingCreate>,
    key: &p256::ecdsa::SigningKey,
) -> Result<PublicCommitResult, PublicRepoWriteError> {
    let sig: p256::ecdsa::Signature = key.sign(&pending.signing_bytes().unwrap());
    writer.finish_external_record(*pending, sig.to_bytes().to_vec())
}

#[tokio::test]
async fn hosted_sequential_writes_retry_reopen_and_authorize_before_authority() {
    let (accounts, reads) = authority_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = hosted(store.clone(), accounts.clone());
    let mut denied = request(1);
    denied.principal = "did:web:bob.accounts.example.com".into();
    assert!(matches!(
        writer.create_record(denied, None).await,
        Err(PublicRepoWriteError::Authorization(_))
    ));
    assert_eq!(reads.0.load(Ordering::SeqCst), 0);
    assert!(store.snapshot(DID).unwrap().is_none());
    let first = writer.create_record(request(1), None).await.unwrap();
    let second = writer.create_record(request(2), None).await.unwrap();
    assert_eq!(
        store.snapshot(DID).unwrap().unwrap().commit.prev,
        Some(first.commit_cid)
    );
    assert_eq!(writer.create_record(request(1), None).await.unwrap(), first);
    drop(writer);
    drop(store);
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = hosted(store.clone(), accounts);
    assert_eq!(
        writer.create_record(request(2), None).await.unwrap(),
        second
    );
    writer.create_record(request(3), None).await.unwrap();
    assert_eq!(store.snapshot(DID).unwrap().unwrap().records.len(), 3);
}

#[test]
fn hosted_external_retry_identity_pending_binding_and_head_cas() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
    let writer = external(store.clone(), &key);
    let mut generated = request(1);
    generated.rkey = None;
    let a = pending(&writer, generated.clone());
    let b = pending(&writer, generated.clone());
    let first = finish(&writer, a, &key).unwrap();
    assert_eq!(finish(&writer, b, &key).unwrap(), first);
    let changed_mode = PublicCreateRequest {
        rkey: Some(AtprotoRecordKey::new(first.uri.rsplit('/').next().unwrap()).unwrap()),
        ..generated
    };
    assert!(writer
        .prepare_external_record_with_expected_prev_text(changed_mode, None)
        .is_err());
    let a = pending(&writer, request(2));
    let stale = pending(&writer, request(3));
    finish(&writer, a, &key).unwrap();
    assert!(matches!(
        finish(&writer, stale, &key),
        Err(PublicRepoWriteError::InvalidSwap)
    ));
    let other_dir = tempfile::tempdir().unwrap();
    let other = external(
        Arc::new(PublicRepoStore::open(other_dir.path()).unwrap()),
        &key,
    );
    assert!(matches!(
        finish(&other, pending(&writer, request(4)), &key),
        Err(PublicRepoWriteError::InvalidRequest(_))
    ));
    let intent_before = store.intent(DID, "hosted-1").unwrap().unwrap();
    let mut changed = intent_before.clone();
    changed.principal = "other".into();
    store
        .db
        .put(
            intent_key(DID, "hosted-1"),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
    let mut retry = request(1);
    retry.rkey = None;
    assert!(matches!(
        writer.prepare_external_record_with_expected_prev_text(retry, None),
        Err(PublicRepoWriteError::InvalidRequest(_))
    ));
}

#[test]
fn hosted_external_prospective_limits_preserve_readable_head() {
    for large in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let writer = external(store.clone(), &key);
        for id in 1..=257 {
            let mut req = request(id);
            if large {
                req.value = json_to_dag_cbor(
                    &serde_json::json!({"$type":"app.bsky.feed.post","text":"x".repeat(60000)}),
                )
                .unwrap();
            }
            let before = store
                .snapshot(DID)
                .unwrap()
                .map(|s| s.commit.cid_atproto().unwrap());
            match writer.prepare_external_record_with_expected_prev_text(req, None) {
                Ok(PublicCreatePreparation::Pending(p)) => {
                    finish(&writer, p, &key).unwrap();
                }
                Err(PublicRepoWriteError::InvalidRequest(_)) => {
                    let snapshot = store.snapshot(DID).unwrap().unwrap();
                    assert_eq!(Some(snapshot.commit.cid_atproto().unwrap()), before);
                    if large {
                        assert!(id < 257);
                    } else {
                        assert_eq!(id, 257);
                    }
                    break;
                }
                _ => panic!("unexpected preparation result"),
            }
            assert!(id < 257, "must reject before exceeding the count budget");
        }
    }
}

#[test]
fn hosted_authority_mode_installation_is_atomic() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
    for id in 0..32 {
        let did = format!("did:web:account{id}.example.com");
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let local_store = store.clone();
        let local_key = key.clone();
        let local_did = did.clone();
        let local_barrier = barrier.clone();
        let local = std::thread::spawn(move || {
            local_barrier.wait();
            PublicRepoWriter::new(
                local_store,
                local_did,
                local_key,
                Arc::new(HostedAccountSelfAuthorizer),
            )
            .is_ok()
        });
        let external_store = store.clone();
        let external_key = *key.verifying_key();
        let external_barrier = barrier.clone();
        let external = std::thread::spawn(move || {
            external_barrier.wait();
            PublicRepoWriter::new_external(
                external_store,
                did,
                external_key,
                Arc::new(HostedAccountSelfAuthorizer),
            )
            .is_ok()
        });
        barrier.wait();
        assert_ne!(local.join().unwrap(), external.join().unwrap());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn hosted_blocking_authorizer_does_not_stall_tokio_worker() {
    struct Slow;
    impl PublicPublicationAuthorizer for Slow {
        fn authorize(&self, _: &str, _: &str, _: &str) -> Result<()> {
            std::thread::sleep(std::time::Duration::from_millis(200));
            anyhow::bail!("deliberate denial")
        }
    }
    let (accounts, reads) = authority_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let writer = HostedAccountPublicRepoWriter::new(
        Arc::new(PublicRepoStore::open(dir.path()).unwrap()),
        accounts,
        authority(),
        Arc::new(Slow),
    );
    let start = std::time::Instant::now();
    let (result, elapsed) = tokio::join!(writer.create_record(request(1), None), async {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        start.elapsed()
    });
    assert!(
        elapsed < std::time::Duration::from_millis(150),
        "worker blocked: {elapsed:?}"
    );
    assert!(matches!(
        result,
        Err(PublicRepoWriteError::Authorization(_))
    ));
    assert_eq!(reads.0.load(Ordering::SeqCst), 0);
}

#[test]
fn hosted_genesis_and_publication_share_account_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
    let writer = Arc::new(external(store.clone(), &key));
    let (root, _) = Node::empty().to_node_data_with_blocks_atproto().unwrap();
    let genesis = Commit::sign_atproto(
        &UnsignedCommit::new(DID, root.cid_atproto().unwrap(), Tid::from_raw(1), None),
        &key,
    )
    .unwrap();
    store
        .seed_public_genesis(DID, genesis.clone(), key.verifying_key())
        .unwrap();
    for id in 1..=16 {
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let seed_store = store.clone();
        let seed_genesis = genesis.clone();
        let seed_key = *key.verifying_key();
        let seed_barrier = barrier.clone();
        let seed = std::thread::spawn(move || {
            seed_barrier.wait();
            seed_store
                .seed_public_genesis(DID, seed_genesis, &seed_key)
                .unwrap();
        });
        let publish_writer = writer.clone();
        let publish_key = key.clone();
        let publish_barrier = barrier.clone();
        let publish = std::thread::spawn(move || {
            publish_barrier.wait();
            finish(
                &publish_writer,
                pending(&publish_writer, request(id)),
                &publish_key,
            )
            .unwrap()
        });
        barrier.wait();
        seed.join().unwrap();
        let result = publish.join().unwrap();
        let snapshot = store.snapshot(DID).unwrap().unwrap();
        assert_eq!(snapshot.records.len(), id);
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), result.commit_cid);
    }
}

#[tokio::test]
async fn hosted_concurrent_unconditional_writes_serialize_and_cancelled_waiter_releases() {
    let (accounts, _) = authority_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = Arc::new(hosted(store.clone(), accounts));
    let (a, b, retry) = tokio::join!(
        writer.create_record(request(1), None),
        writer.create_record(request(2), None),
        writer.create_record(request(1), None),
    );
    let a = a.unwrap();
    b.unwrap();
    assert_eq!(retry.unwrap(), a);
    assert_eq!(store.snapshot(DID).unwrap().unwrap().records.len(), 2);
    let admission = store
        .accounts
        .lock()
        .get(DID)
        .unwrap()
        .external_write
        .clone();
    let held = admission.lock_owned().await;
    let waiting_writer = writer.clone();
    let waiting = tokio::spawn(async move { waiting_writer.create_record(request(3), None).await });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(!waiting.is_finished());
    waiting.abort();
    assert!(waiting.await.unwrap_err().is_cancelled());
    drop(held);
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        writer.create_record(request(4), None),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(store.snapshot(DID).unwrap().unwrap().records.len(), 3);
}

#[tokio::test]
async fn hosted_legacy_genesis_read_create_reopen_reuses_custody_signature() {
    let (accounts, reads) = authority_fixture_with(GenesisFixture::Legacy).await;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = Arc::new(hosted(store.clone(), accounts.clone()));
    let genesis = PublicRepoReader::Hosted {
        writer: writer.clone(),
        did: DID.into(),
    }
    .read(|snapshot| Ok(snapshot.unwrap().0.commit))
    .await
    .unwrap();
    assert_eq!(reads.1.load(Ordering::SeqCst), 1);
    let bytes = store.public_genesis_bytes(DID).unwrap().unwrap();
    assert_eq!(bytes, genesis.to_atproto_dag_cbor().unwrap());
    let swap = genesis.cid_atproto().unwrap().to_string();
    let first = writer.create_record(request(1), Some(&swap)).await.unwrap();
    assert_eq!(reads.1.load(Ordering::SeqCst), 2); // one conversion, one record signature
    assert_eq!(
        writer.create_record(request(1), Some(&swap)).await.unwrap(),
        first
    );
    let read = PublicRepoReader::Hosted {
        writer: writer.clone(),
        did: DID.into(),
    }
    .read(|snapshot| Ok(snapshot.unwrap().0))
    .await
    .unwrap();
    assert_eq!(read.commit.cid_atproto().unwrap(), first.commit_cid);
    assert_eq!(reads.1.load(Ordering::SeqCst), 2); // no signing on read or retry
    drop(writer);
    drop(store);
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = Arc::new(hosted(store.clone(), accounts));
    PublicRepoReader::Hosted {
        writer: writer.clone(),
        did: DID.into(),
    }
    .read(|snapshot| {
        assert_eq!(snapshot.unwrap().0.records.len(), 1);
        Ok(())
    })
    .await
    .unwrap();
    assert_eq!(store.public_genesis_bytes(DID).unwrap().unwrap(), bytes);
    assert_eq!(reads.1.load(Ordering::SeqCst), 2);
    assert_eq!(
        writer.create_record(request(1), Some(&swap)).await.unwrap(),
        first
    );
    store
        .db
        .put(format!("{GENESIS_PREFIX}{DID}"), b"corrupt cached genesis")
        .unwrap();
    assert!(PublicRepoReader::Hosted {
        writer,
        did: DID.into()
    }
    .read(|_| Ok(()))
    .await
    .is_err());
    assert_eq!(
        store
            .snapshot(DID)
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap(),
        first.commit_cid
    );
}

#[tokio::test]
async fn hosted_genesis_rejects_corrupt_foreign_progressed_and_wrong_authority() {
    for mode in [
        GenesisFixture::CorruptNative,
        GenesisFixture::InvalidNativeSignature,
        GenesisFixture::ForeignNative,
        GenesisFixture::ProgressedNative,
        GenesisFixture::NonemptyNative,
        GenesisFixture::CorruptPublic,
        GenesisFixture::WrongKey,
    ] {
        let (accounts, reads) = authority_fixture_with(mode).await;
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let writer = Arc::new(hosted(store.clone(), accounts));
        assert!(
            writer.create_record(request(1), None).await.is_err(),
            "{mode:?}"
        );
        assert!(
            PublicRepoReader::Hosted {
                writer,
                did: DID.into()
            }
            .read(|snapshot| Ok(snapshot.is_some()))
            .await
            .is_err(),
            "{mode:?}"
        );
        assert!(store.snapshot(DID).unwrap().is_none(), "{mode:?}");
        assert!(
            store.public_genesis_bytes(DID).unwrap().is_none(),
            "{mode:?}"
        );
        if !matches!(mode, GenesisFixture::WrongKey) {
            assert_eq!(reads.1.load(Ordering::SeqCst), 0);
        }
    }
    let (accounts, reads) = authority_fixture_with(GenesisFixture::Legacy).await;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = Arc::new(HostedAccountPublicRepoWriter::new(
        store.clone(),
        accounts,
        hyprstream_rpc::Subject::new("service:foreign"),
        Arc::new(HostedAccountSelfAuthorizer),
    ));
    assert!(writer.create_record(request(1), None).await.is_err());
    assert!(PublicRepoReader::Hosted {
        writer,
        did: DID.into()
    }
    .read(|_| Ok(()))
    .await
    .is_err());
    assert_eq!(reads.0.load(Ordering::SeqCst), 0);
    assert!(store.snapshot(DID).unwrap().is_none());
}

#[tokio::test]
async fn hosted_read_initialization_and_create_do_not_rewind_concurrent_head() {
    for mode in [GenesisFixture::Current, GenesisFixture::Legacy] {
        let (accounts, _) = authority_fixture_with(mode).await;
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let writer = Arc::new(hosted(store.clone(), accounts));
        for id in 1..=4 {
            let (read, write) = tokio::join!(
                PublicRepoReader::Hosted {
                    writer: writer.clone(),
                    did: DID.into()
                }
                .read(|snapshot| Ok(snapshot.unwrap().0.records.len())),
                writer.create_record(request(id), None)
            );
            let count = read.unwrap();
            assert!(count == id - 1 || count == id);
            let written = write.unwrap();
            let snapshot = store.snapshot(DID).unwrap().unwrap();
            assert_eq!(snapshot.records.len(), id);
            assert_eq!(snapshot.commit.cid_atproto().unwrap(), written.commit_cid);
        }
    }
}

#[tokio::test]
async fn hosted_legacy_first_create_uses_original_genesis() {
    let (accounts, _) = authority_fixture_with(GenesisFixture::Legacy).await;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = hosted(store.clone(), accounts.clone());
    let result = writer.create_record(request(1), None).await.unwrap();
    let first_pair = accounts
        .hosted_repo_genesis_for_hosted_did(&authority(), DID)
        .await
        .unwrap()
        .unwrap();
    let second_pair = accounts
        .hosted_repo_genesis_for_hosted_did(&authority(), DID)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        first_pair, second_pair,
        "custody conversion must preserve immutable bytes"
    );
    assert_eq!(
        store.public_genesis_bytes(DID).unwrap().unwrap(),
        first_pair.1
    );
    let genesis =
        Commit::from_atproto_dag_cbor(&store.public_genesis_bytes(DID).unwrap().unwrap()).unwrap();
    let snapshot = store.snapshot(DID).unwrap().unwrap();
    assert_eq!(snapshot.commit.prev, Some(genesis.cid_atproto().unwrap()));
    assert_eq!(snapshot.commit.cid_atproto().unwrap(), result.commit_cid);
    assert_eq!(genesis.rev, Tid::from_raw(1));
    assert!(genesis.prev.is_none());
    // Public stores from the prior source have an archive but no locator.
    // Rebuild only that locator, preserving the already-progressed head.
    store.db.delete(format!("{GENESIS_PREFIX}{DID}")).unwrap();
    PublicRepoReader::Hosted {
        writer: Arc::new(writer),
        did: DID.into(),
    }
    .read(|snapshot| {
        assert_eq!(snapshot.unwrap().0.records.len(), 1);
        Ok(())
    })
    .await
    .unwrap();
    assert_eq!(
        store.public_genesis_bytes(DID).unwrap().unwrap(),
        first_pair.1
    );
    assert_eq!(
        store
            .snapshot(DID)
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap(),
        result.commit_cid
    );
}

#[tokio::test]
async fn hosted_current_read_before_write_starts_at_published_genesis() {
    let (accounts, reads) = authority_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
    let writer = Arc::new(hosted(store.clone(), accounts));
    let snapshot = PublicRepoReader::Hosted {
        writer,
        did: DID.into(),
    }
    .read(|snapshot| Ok(snapshot.unwrap().0))
    .await
    .unwrap();
    assert!(snapshot.records.is_empty());
    assert_eq!(snapshot.commit.rev, Tid::from_raw(1));
    assert!(snapshot.commit.prev.is_none());
    assert_eq!(reads.1.load(Ordering::SeqCst), 0);
    assert_eq!(
        store.snapshot(DID).unwrap().unwrap().commit,
        snapshot.commit
    );
}
