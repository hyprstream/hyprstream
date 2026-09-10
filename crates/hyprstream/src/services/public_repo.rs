//! Durable public AT Protocol repository transactions.
//!
//! This module is the storage boundary between native authorization and the
//! public repository codec. It keeps public record bytes and public commit
//! bytes in a separate RocksDB key namespace, and writes the record, signed
//! head and publication intent in one batch. It deliberately does not mint
//! identities, evaluate native policy itself, or expose native signing keys.
//! A caller must supply the native authorization gate for every request.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context as _, Result};
use hyprstream_pds::atproto_cbor::AtprotoRecord;
use hyprstream_pds::commit::{Commit, UnsignedCommit};
use hyprstream_pds::dag_cbor::DagCbor;
use hyprstream_pds::mst::Node;
use hyprstream_pds::repo_authority::accept_repo_authority;
use hyprstream_pds::tid::Tid;
use hyprstream_pds::Cid;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

const RECORD_PREFIX: &str = "public-rk\0";
const COMMIT_PREFIX: &str = "public-commit\0";
const INTENT_PREFIX: &str = "public-intent\0";
const MAX_REQUEST_ID: usize = 128;
const MAX_PRINCIPAL: usize = 512;

fn record_prefix(did: &str) -> Vec<u8> {
    format!("{RECORD_PREFIX}{did}\0").into_bytes()
}

fn record_key(did: &str, collection: &str, rkey: &str) -> Vec<u8> {
    format!("{RECORD_PREFIX}{did}\0{collection}\0{rkey}").into_bytes()
}

fn commit_key(did: &str) -> Vec<u8> {
    format!("{COMMIT_PREFIX}{did}").into_bytes()
}

fn intent_key(did: &str, request_id: &str) -> Vec<u8> {
    format!("{INTENT_PREFIX}{did}\0{request_id}").into_bytes()
}

fn validate_did(did: &str) -> Result<()> {
    let authority = accept_repo_authority(did)?;
    ensure!(
        authority.is_publicly_publishable(),
        "public repo requires did:web or did:plc authority"
    );
    ensure!(!did.contains('\0'), "repo DID contains a NUL byte");
    Ok(())
}

fn validate_request_id(request_id: &str) -> Result<()> {
    ensure!(
        !request_id.is_empty() && request_id.len() <= MAX_REQUEST_ID,
        "invalid publication request id"
    );
    ensure!(
        request_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.'),
        "publication request id contains unsupported characters"
    );
    Ok(())
}

fn validate_principal(principal: &str) -> Result<()> {
    ensure!(
        !principal.is_empty() && principal.len() <= MAX_PRINCIPAL,
        "invalid native principal"
    );
    ensure!(
        !principal.contains('\0'),
        "native principal contains a NUL byte"
    );
    Ok(())
}

/// Native policy gate required by PublicRepoWriter.
pub trait PublicPublicationAuthorizer: Send + Sync {
    /// Authorize one direct publication before any public repository state is
    /// read or written. Implementations should apply native identity,
    /// assurance, account, collection, tenant, expiry and revocation policy.
    fn authorize(&self, principal: &str, account: &str, collection: &str) -> Result<()>;
}

/// A request to create one immutable public record under an owned repo.
#[derive(Clone, Debug)]
pub struct PublicCreateRequest {
    pub request_id: String,
    pub principal: String,
    pub did: String,
    pub collection: String,
    pub rkey: Tid,
    pub value: DagCbor,
    /// Required repo-head CAS value. None means this must create the genesis
    /// record; retries use the request id and never silently fork a head.
    pub expected_prev: Option<Cid>,
}

/// Durable result of a successful public repository transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicCommitResult {
    pub uri: String,
    pub cid: Cid,
    pub commit_cid: Cid,
}

#[derive(Clone, Debug)]
pub struct PublicRepoSnapshot {
    pub did: String,
    pub records: BTreeMap<(String, Tid), AtprotoRecord>,
    pub commit: Commit,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PublicationIntent {
    request_id: String,
    principal: String,
    did: String,
    collection: String,
    rkey: String,
    cid: String,
    commit_cid: String,
}

/// Durable public repository storage. It uses a distinct RocksDB directory
/// and key namespace so native signed artifacts are never re-encoded as public
/// bytes.
pub struct PublicRepoStore {
    db: Arc<rocksdb::DB>,
    write_lock: Mutex<()>,
}

impl std::fmt::Debug for PublicRepoStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublicRepoStore").finish_non_exhaustive()
    }
}

impl PublicRepoStore {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("failed to create public repo store at {path:?}"))?;
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, path)
            .with_context(|| format!("failed to open public repo store at {path:?}"))?;
        Ok(Self {
            db: Arc::new(db),
            write_lock: Mutex::new(()),
        })
    }

    pub fn snapshot(&self, did: &str) -> Result<Option<PublicRepoSnapshot>> {
        validate_did(did)?;
        let prefix = record_prefix(did);
        let mut records = BTreeMap::new();
        for item in self.db.prefix_iterator(&prefix) {
            let (key, bytes) = item.context("public repo record scan failed")?;
            if !key.starts_with(&prefix) {
                break;
            }
            let suffix = std::str::from_utf8(&key[prefix.len()..])
                .context("public repo record key is not UTF-8")?;
            let (collection, rkey) = suffix
                .split_once('\0')
                .ok_or_else(|| anyhow!("public repo record key missing separator"))?;
            let rkey = Tid::parse(rkey)?;
            let record = AtprotoRecord::from_bytes(collection, rkey, &bytes)
                .with_context(|| format!("invalid public record {did}/{collection}/{rkey}"))?;
            records.insert((collection.to_owned(), rkey), record);
        }
        if records.is_empty() {
            return Ok(None);
        }
        let bytes = self
            .db
            .get(commit_key(did))
            .context("public repo commit read failed")?
            .ok_or_else(|| anyhow!("public repo has records but no signed commit"))?;
        let commit = Commit::from_atproto_dag_cbor(&bytes)
            .context("public repo signed commit is invalid")?;
        ensure!(commit.did == did, "public repo commit DID mismatch");
        let keyed = records
            .iter()
            .map(|((collection, rkey), record)| {
                (format!("{collection}/{}", rkey.encode()), record.cid())
            })
            .collect();
        let tree = Node::from_keyed_records(&keyed);
        let (root_data, _) = tree.to_node_data_with_blocks_atproto()?;
        ensure!(
            commit.data == root_data.cid_atproto()?,
            "public repo commit root does not cover stored records"
        );
        Ok(Some(PublicRepoSnapshot {
            did: did.to_owned(),
            records,
            commit,
        }))
    }

    fn intent(&self, did: &str, request_id: &str) -> Result<Option<PublicationIntent>> {
        let Some(bytes) = self.db.get(intent_key(did, request_id))? else {
            return Ok(None);
        };
        Ok(Some(
            serde_json::from_slice(&bytes).context("public publication intent is invalid")?,
        ))
    }

    fn write_transaction(
        &self,
        did: &str,
        record: &AtprotoRecord,
        commit: &Commit,
        intent: &PublicationIntent,
    ) -> Result<()> {
        let commit_bytes = commit.to_atproto_dag_cbor()?;
        let intent_bytes = serde_json::to_vec(intent).context("encode publication intent")?;
        let mut batch = rocksdb::WriteBatch::default();
        batch.put(
            record_key(did, record.collection(), record.rkey().as_str()),
            record.bytes(),
        );
        batch.put(commit_key(did), commit_bytes);
        batch.put(intent_key(did, &intent.request_id), intent_bytes);
        let mut options = rocksdb::WriteOptions::default();
        options.set_sync(true);
        self.db
            .write_opt(batch, &options)
            .context("public repo transaction failed")?;
        Ok(())
    }
}

/// Public repository writer with an explicit native authorization gate.
pub struct PublicRepoWriter {
    store: Arc<PublicRepoStore>,
    did: String,
    signing_key: p256::ecdsa::SigningKey,
    authorizer: Arc<dyn PublicPublicationAuthorizer>,
}

impl std::fmt::Debug for PublicRepoWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublicRepoWriter")
            .field("did", &self.did)
            .finish_non_exhaustive()
    }
}

impl PublicRepoWriter {
    pub fn new(
        store: Arc<PublicRepoStore>,
        did: impl Into<String>,
        signing_key: p256::ecdsa::SigningKey,
        authorizer: Arc<dyn PublicPublicationAuthorizer>,
    ) -> Result<Self> {
        let did = did.into();
        validate_did(&did)?;
        Ok(Self {
            store,
            did,
            signing_key,
            authorizer,
        })
    }

    pub fn create_record(&self, request: PublicCreateRequest) -> Result<PublicCommitResult> {
        ensure!(
            request.did == self.did,
            "public request account does not match writer"
        );
        validate_request_id(&request.request_id)?;
        validate_principal(&request.principal)?;
        self.authorizer
            .authorize(&request.principal, &self.did, &request.collection)?;

        let _guard = self.store.write_lock.lock();
        let record = AtprotoRecord::new(request.collection.clone(), request.rkey, request.value)?;
        if let Some(intent) = self.store.intent(&self.did, &request.request_id)? {
            ensure!(
                intent.did == self.did
                    && intent.collection == record.collection()
                    && intent.rkey == record.rkey().as_str()
                    && intent.cid == record.cid().to_string(),
                "publication request id was reused with different content"
            );
            let snapshot = self
                .store
                .snapshot(&self.did)?
                .ok_or_else(|| anyhow!("publication intent exists without a repository"))?;
            let commit_cid = snapshot.commit.cid_atproto()?;
            ensure!(
                intent.commit_cid == commit_cid.to_string(),
                "publication intent commit does not match repository head"
            );
            return Ok(PublicCommitResult {
                uri: record.uri(&self.did),
                cid: record.cid(),
                commit_cid,
            });
        }

        let existing = self.store.snapshot(&self.did)?;
        let (mut keyed, previous, previous_rev) = match existing {
            Some(snapshot) => {
                let previous = snapshot.commit.cid_atproto()?;
                let previous_rev = snapshot.commit.rev;
                ensure!(
                    request.expected_prev == Some(previous),
                    "public repo head CAS conflict"
                );
                let keyed = snapshot
                    .records
                    .into_iter()
                    .map(|((collection, rkey), record)| {
                        (format!("{collection}/{}", rkey.encode()), record.cid())
                    })
                    .collect();
                (keyed, Some(previous), Some(previous_rev))
            }
            None => {
                ensure!(
                    request.expected_prev.is_none(),
                    "public repo genesis CAS conflict"
                );
                (BTreeMap::new(), None, None)
            }
        };
        let record_key = format!("{}/{}", record.collection(), record.rkey().as_str());
        ensure!(
            !keyed.contains_key(&record_key),
            "record key already exists"
        );
        keyed.insert(record_key, record.cid());

        let tree = Node::from_keyed_records(&keyed);
        let (root_data, _) = tree.to_node_data_with_blocks_atproto()?;
        let unsigned = UnsignedCommit::new(
            self.did.clone(),
            root_data.cid_atproto()?,
            next_revision(previous_rev),
            previous,
        );
        let commit = Commit::sign_atproto(&unsigned, &self.signing_key)?;
        let commit_cid = commit.cid_atproto()?;
        let intent = PublicationIntent {
            request_id: request.request_id,
            principal: request.principal,
            did: self.did.clone(),
            collection: record.collection().to_owned(),
            rkey: record.rkey().as_str().to_owned(),
            cid: record.cid().to_string(),
            commit_cid: commit_cid.to_string(),
        };
        self.store
            .write_transaction(&self.did, &record, &commit, &intent)?;
        Ok(PublicCommitResult {
            uri: record.uri(&self.did),
            cid: record.cid(),
            commit_cid,
        })
    }
}

fn next_revision(previous: Option<Tid>) -> Tid {
    let now = Tid::now();
    match previous {
        Some(previous) if now <= previous => Tid::from_raw(previous.to_raw().saturating_add(1)),
        _ => now,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Gate {
        allow: AtomicBool,
    }

    impl PublicPublicationAuthorizer for Gate {
        fn authorize(&self, _: &str, _: &str, _: &str) -> Result<()> {
            ensure!(self.allow.load(Ordering::Acquire), "publication denied");
            Ok(())
        }
    }

    fn post() -> DagCbor {
        DagCbor::str_map([
            ("$type", DagCbor::Text("app.bsky.feed.post".into())),
            ("text", DagCbor::Text("durable".into())),
        ])
    }

    #[test]
    fn authorized_create_is_atomic_durable_and_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PublicRepoStore::open(dir.path()).expect("store"));
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let writer = PublicRepoWriter::new(store.clone(), "did:web:tormentnexus.social", key, gate)
            .expect("writer");
        let request = PublicCreateRequest {
            request_id: "req-1".into(),
            principal: "did:at9p:agent".into(),
            did: "did:web:tormentnexus.social".into(),
            collection: "app.bsky.feed.post".into(),
            rkey: Tid::from_raw(7),
            value: post(),
            expected_prev: None,
        };
        let first = writer.create_record(request.clone()).expect("create");
        let retry = writer.create_record(request).expect("retry");
        assert_eq!(first, retry);
        let second = writer
            .create_record(PublicCreateRequest {
                request_id: "req-2".into(),
                principal: "did:at9p:agent".into(),
                did: "did:web:tormentnexus.social".into(),
                collection: "app.bsky.feed.post".into(),
                rkey: Tid::from_raw(8),
                value: post(),
                expected_prev: Some(first.commit_cid),
            })
            .expect("second create");
        assert_ne!(first.commit_cid, second.commit_cid);
        let snapshot = store
            .snapshot("did:web:tormentnexus.social")
            .expect("snapshot")
            .expect("repo");
        assert_eq!(snapshot.records.len(), 2);
        snapshot
            .commit
            .verify_atproto(writer.signing_key.verifying_key())
            .expect("commit signature");
        drop(writer);
        drop(store);
        let reopened = PublicRepoStore::open(dir.path()).expect("reopen");
        assert_eq!(
            reopened
                .snapshot("did:web:tormentnexus.social")
                .expect("snapshot")
                .expect("repo")
                .records
                .len(),
            2
        );
    }

    #[test]
    fn authorization_and_head_cas_fail_closed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PublicRepoStore::open(dir.path()).expect("store"));
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(false),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let writer =
            PublicRepoWriter::new(store, "did:web:tormentnexus.social", key, gate).expect("writer");
        let err = writer
            .create_record(PublicCreateRequest {
                request_id: "req-1".into(),
                principal: "agent".into(),
                did: "did:web:tormentnexus.social".into(),
                collection: "app.bsky.feed.post".into(),
                rkey: Tid::from_raw(1),
                value: post(),
                expected_prev: None,
            })
            .expect_err("denied");
        assert!(err.to_string().contains("publication denied"));
    }

    #[test]
    fn at9p_authority_cannot_be_exported_without_bridge() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PublicRepoStore::open(dir.path()).expect("store"));
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        assert!(PublicRepoWriter::new(store, "did:at9p:agent", key, gate).is_err());
    }
}
