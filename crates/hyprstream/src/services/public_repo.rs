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
use base64::{
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
    Engine as _,
};
use hyprstream_pds::atproto_cbor::{AtprotoRecord, AtprotoRecordKey};
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
const COMMIT_BLOCK_PREFIX: &str = "public-commit-block\0";
const INTENT_PREFIX: &str = "public-intent\0";
const MAX_REQUEST_ID: usize = 128;
const MAX_PRINCIPAL: usize = 512;
/// Deployment ceiling on complete canonical DAG-CBOR record bytes (64 KiB).
/// A conservative storage policy, not a universal protocol maximum; large
/// payloads belong in linked blobs. Includes CBOR map/type/length overhead.
/// See https://atproto.com/specs/repository#security-considerations.
pub const MAX_PUBLIC_RECORD_BYTES: usize = 64 * 1024;

fn record_prefix(did: &str) -> Vec<u8> {
    format!("{RECORD_PREFIX}{did}\0").into_bytes()
}

fn record_key(did: &str, collection: &str, rkey: &str) -> Vec<u8> {
    format!("{RECORD_PREFIX}{did}\0{collection}\0{rkey}").into_bytes()
}

fn commit_key(did: &str) -> Vec<u8> {
    format!("{COMMIT_PREFIX}{did}").into_bytes()
}

fn commit_block_key(did: &str, cid: &str) -> Vec<u8> {
    format!("{COMMIT_BLOCK_PREFIX}{did}\0{cid}").into_bytes()
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

    /// Separate account-owner authority: permission to publish a record never
    /// implicitly permits key replacement. Existing authorizers fail closed.
    fn authorize_key_promotion(&self, _principal: &str, _account: &str) -> Result<()> {
        Err(anyhow!(
            "repository signing key promotion is not authorized"
        ))
    }
}

/// A request to create one immutable public record under an owned repo.
#[derive(Clone, Debug)]
pub struct PublicCreateRequest {
    pub request_id: String,
    pub principal: String,
    pub did: String,
    pub collection: String,
    /// A validated AT record key, including singleton keys such as `self`.
    pub rkey: AtprotoRecordKey,
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
    pub records: BTreeMap<(String, AtprotoRecordKey), AtprotoRecord>,
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

#[derive(Default)]
struct AccountSigningState {
    // Held through snapshot/rebuild/sign/persist and through promotion. All
    // handles for exactly one DID share this guard and its active authority.
    active_key: Mutex<Option<p256::ecdsa::SigningKey>>,
}

/// Native callers retain genesis-or-exact CAS; XRPC may omit its condition.
enum PublicHeadCondition {
    Unconditional,
    Exact(Option<Cid>),
}

impl PublicHeadCondition {
    fn matches(&self, actual: Option<Cid>) -> bool {
        match self {
            Self::Unconditional => true,
            Self::Exact(expected) => *expected == actual,
        }
    }
}

/// Durable public repository storage. It uses a distinct RocksDB directory
/// and key namespace so native signed artifacts are never re-encoded as public
/// bytes.
pub struct PublicRepoStore {
    db: Arc<rocksdb::DB>,
    // Held only to find/insert a shared account state, never during account
    // locking, cryptography or database access. Unrelated DIDs can progress.
    accounts: Mutex<BTreeMap<String, Arc<AccountSigningState>>>,
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
            accounts: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn snapshot(&self, did: &str) -> Result<Option<PublicRepoSnapshot>> {
        validate_did(did)?;
        // Iteration and the head lookup must observe the same database
        // sequence, even when a writer commits a batch during the scan.
        let snapshot = self.db.snapshot();
        let prefix = record_prefix(did);
        let mut records = BTreeMap::new();
        for item in snapshot.iterator(rocksdb::IteratorMode::From(
            &prefix,
            rocksdb::Direction::Forward,
        )) {
            let (key, bytes) = item.context("public repo record scan failed")?;
            if !key.starts_with(&prefix) {
                break;
            }
            let suffix = std::str::from_utf8(&key[prefix.len()..])
                .context("public repo record key is not UTF-8")?;
            let (collection, rkey) = suffix
                .split_once('\0')
                .ok_or_else(|| anyhow!("public repo record key missing separator"))?;
            let rkey = AtprotoRecordKey::new(rkey)?;
            let record =
                AtprotoRecord::from_bytes(collection, &rkey, &bytes).with_context(|| {
                    format!("invalid public record {did}/{collection}/{}", rkey.as_str())
                })?;
            records.insert((collection.to_owned(), rkey), record);
        }
        if records.is_empty() {
            return Ok(None);
        }
        let bytes = snapshot
            .get(commit_key(did))
            .context("public repo commit read failed")?
            .ok_or_else(|| anyhow!("public repo has records but no signed commit"))?;
        let commit = Commit::from_atproto_dag_cbor(&bytes)
            .context("public repo signed commit is invalid")?;
        ensure!(commit.did == did, "public repo commit DID mismatch");
        let keyed = records
            .iter()
            .map(|((collection, rkey), record)| {
                (format!("{collection}/{}", rkey.as_str()), record.cid())
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
        // The mutable head may advance before a caller retries. Retain the
        // exact commit block that backs each durable publication receipt.
        batch.put(
            commit_block_key(did, &commit.cid_atproto()?.to_string()),
            &commit_bytes,
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
    account: Arc<AccountSigningState>,
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
    pub fn did(&self) -> &str {
        &self.did
    }

    /// Bind a trusted account key to this store. All writer handles for the
    /// same DID share one active authority; a different key requires promotion.
    pub fn new(
        store: Arc<PublicRepoStore>,
        did: impl Into<String>,
        signing_key: p256::ecdsa::SigningKey,
        authorizer: Arc<dyn PublicPublicationAuthorizer>,
    ) -> Result<Self> {
        let did = did.into();
        validate_did(&did)?;
        let account = {
            let mut accounts = store.accounts.lock();
            Arc::clone(accounts.entry(did.clone()).or_default())
        };
        {
            let mut state = account.active_key.lock();
            if let Some(active) = state.as_ref() {
                ensure!(
                    active.verifying_key() == signing_key.verifying_key(),
                    "repo signing key differs from active authority; use explicit promotion"
                );
            } else {
                // On reopen, never silently adopt a key that disagrees with
                // the durable head. Recover/promote through the account owner.
                if let Some(repo) = store.snapshot(&did)? {
                    repo.commit
                        .verify_atproto(signing_key.verifying_key())
                        .context("repo head does not match supplied active signing authority")?;
                }
                *state = Some(signing_key);
            }
        }
        Ok(Self {
            store,
            did,
            account,
            authorizer,
        })
    }

    /// Resolve this account's active public key from shared transaction state.
    /// DID-document publication must use this authority after promotion returns.
    pub fn active_verifying_key(&self) -> Result<p256::ecdsa::VerifyingKey> {
        self.account
            .active_key
            .lock()
            .as_ref()
            .map(|key| *key.verifying_key())
            .ok_or_else(|| anyhow!("no active signing authority for repository"))
    }

    /// Explicit account-owner promotion boundary. The trusted caller must
    /// validate/secure the candidate and publish it in the account DID document
    /// only after this succeeds. This is not the node-global OAuth key store.
    ///
    /// Serializes with all writers for this store/account. Re-signs an idle
    /// head over identical canonical unsigned bytes, persists the new head and
    /// immutable block synchronously, then exposes the candidate to surviving
    /// writers and public-key readers. Errors leave the active key unchanged.
    /// Old blocks and publication intents remain immutable for original retries.
    /// This in-process boundary does not subscribe to external key rotations;
    /// account lifecycle wiring must route promotion through it.
    pub fn promote_signing_key(
        &self,
        principal: &str,
        expected_active: &p256::ecdsa::VerifyingKey,
        candidate: p256::ecdsa::SigningKey,
    ) -> Result<Option<Cid>> {
        validate_principal(principal)?;
        self.authorizer
            .authorize_key_promotion(principal, &self.did)?;
        let mut state = self.account.active_key.lock();
        let active = state
            .as_mut()
            .ok_or_else(|| anyhow!("no active signing authority for repository"))?;
        ensure!(
            active.verifying_key() == expected_active,
            "repo signing authority CAS conflict"
        );
        let head = if let Some(repo) = self.store.snapshot(&self.did)? {
            repo.commit
                .verify_atproto(active.verifying_key())
                .context("current repo head does not verify under active authority")?;
            let resigned = Commit::sign_atproto(&repo.commit.unsigned(), &candidate)?;
            let bytes = resigned.to_atproto_dag_cbor()?;
            let cid = resigned.cid_atproto()?;
            let mut batch = rocksdb::WriteBatch::default();
            batch.put(commit_block_key(&self.did, &cid.to_string()), &bytes);
            batch.put(commit_key(&self.did), bytes);
            let mut options = rocksdb::WriteOptions::default();
            options.set_sync(true);
            self.store
                .db
                .write_opt(batch, &options)
                .context("public repo signing authority reconciliation failed")?;
            Some(cid)
        } else {
            None
        };
        *active = candidate;
        Ok(head)
    }

    pub fn create_record(&self, request: PublicCreateRequest) -> Result<PublicCommitResult> {
        let condition = PublicHeadCondition::Exact(request.expected_prev);
        self.create_record_with_condition(request, condition)
    }

    fn create_record_with_condition(
        &self,
        request: PublicCreateRequest,
        condition: PublicHeadCondition,
    ) -> Result<PublicCommitResult> {
        ensure!(
            request.did == self.did,
            "public request account does not match writer"
        );
        validate_request_id(&request.request_id)?;
        validate_principal(&request.principal)?;
        self.authorizer
            .authorize(&request.principal, &self.did, &request.collection)?;

        let state = self.account.active_key.lock();
        let signing_key = state
            .as_ref()
            .ok_or_else(|| anyhow!("no active signing authority for repository"))?;
        let record = AtprotoRecord::new(request.collection.clone(), &request.rkey, request.value)?;
        ensure!(
            record.bytes().len() <= MAX_PUBLIC_RECORD_BYTES,
            "public record exceeds {MAX_PUBLIC_RECORD_BYTES}-byte canonical encoded limit"
        );
        if let Some(intent) = self.store.intent(&self.did, &request.request_id)? {
            ensure!(
                intent.request_id == request.request_id
                    && intent.principal == request.principal
                    && intent.did == self.did
                    && intent.collection == record.collection()
                    && intent.rkey == record.rkey().as_str()
                    && intent.cid == record.cid().to_string(),
                "publication request id was reused with different content"
            );
            let snapshot = self
                .store
                .snapshot(&self.did)?
                .ok_or_else(|| anyhow!("publication intent exists without a repository"))?;
            ensure!(
                snapshot
                    .records
                    .get(&(record.collection().to_owned(), request.rkey))
                    .is_some_and(|stored| stored.bytes() == record.bytes()),
                "publication intent record does not match repository"
            );
            let commit_bytes = self
                .store
                .db
                .get(commit_block_key(&self.did, &intent.commit_cid))?
                .ok_or_else(|| anyhow!("publication intent commit block is missing"))?;
            let commit = Commit::from_atproto_dag_cbor(&commit_bytes)
                .context("publication intent commit is invalid")?;
            let commit_cid = commit.cid_atproto()?;
            ensure!(
                commit.did == self.did && intent.commit_cid == commit_cid.to_string(),
                "publication intent commit does not match stored block"
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
                    condition.matches(Some(previous)),
                    "public repo head CAS conflict"
                );
                let keyed = snapshot
                    .records
                    .into_iter()
                    .map(|((collection, rkey), record)| {
                        (format!("{collection}/{}", rkey.as_str()), record.cid())
                    })
                    .collect();
                (keyed, Some(previous), Some(previous_rev))
            }
            None => {
                ensure!(condition.matches(None), "public repo genesis CAS conflict");
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
        let commit = Commit::sign_atproto(&unsigned, signing_key)?;
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

    /// XRPC creation with an optional base32 CID head condition. Unlike the
    /// native typed API, omission permits a new write on any current head.
    /// Authorization, retry lookup, head comparison and the write all use the
    /// shared transaction path; no repository state is read by this adapter.
    /// This argument supersedes `request.expected_prev`.
    pub fn create_record_with_expected_prev_text(
        &self,
        request: PublicCreateRequest,
        expected_prev: Option<&str>,
    ) -> Result<PublicCommitResult> {
        let condition = match expected_prev {
            None => PublicHeadCondition::Unconditional,
            Some(expected) => PublicHeadCondition::Exact(Some(
                parse_atproto_json_cid(expected).context("invalid swapCommit CID")?,
            )),
        };
        self.create_record_with_condition(request, condition)
    }
}

/// Decode a canonical AT JSON CID link without accepting other codecs,
/// hashes, overlong varints, or noncanonical base32 spellings.
fn parse_atproto_json_cid(text: &str) -> Result<Cid> {
    // CIDv1, one-byte raw/DAG-CBOR codec, sha2-256, and 32 digest bytes.
    // These 36 bytes occupy 58 base32 digits plus the multibase prefix.
    ensure!(
        text.len() == 59 && text.starts_with('b'),
        "invalid AT CID text"
    );
    let mut raw = Vec::with_capacity(36);
    let mut pending = 0u16;
    let mut bits = 0u32;
    for digit in text.bytes().skip(1) {
        let digit = match digit {
            b'a'..=b'z' => digit - b'a',
            b'2'..=b'7' => digit - b'2' + 26,
            _ => return Err(anyhow!("AT CID must use lowercase base32")),
        };
        pending = (pending << 5) | u16::from(digit);
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            raw.push((pending >> bits) as u8);
            pending &= (1 << bits) - 1;
        }
    }
    ensure!(
        pending == 0
            && raw.len() == 36
            && matches!(raw.as_slice(), [1, 0x55 | 0x71, 0x12, 0x20, ..]),
        "AT CID requires canonical raw or DAG-CBOR SHA-256 bytes"
    );
    Cid::from_bytes(&raw)
}

/// Convert AT JSON into the value type used by the public codec. Reserved
/// single-key `$link` and `$bytes` objects become links and byte strings,
/// never ordinary maps. Invalid wrappers, floats and integers outside signed
/// 64-bit range are rejected before serialization.
pub fn json_to_dag_cbor(value: &serde_json::Value) -> Result<DagCbor> {
    Ok(match value {
        serde_json::Value::Null => DagCbor::Null,
        serde_json::Value::Bool(value) => DagCbor::Bool(*value),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                if value >= 0 {
                    DagCbor::Unsigned(value as u64)
                } else {
                    DagCbor::Negative(value as i128)
                }
            } else {
                return Err(anyhow!("AT record numbers must be signed integers"));
            }
        }
        serde_json::Value::String(value) => DagCbor::Text(value.clone()),
        serde_json::Value::Array(values) => DagCbor::List(
            values
                .iter()
                .map(json_to_dag_cbor)
                .collect::<Result<Vec<_>>>()?,
        ),
        serde_json::Value::Object(values) if values.contains_key("$link") => {
            ensure!(values.len() == 1, "AT JSON link must contain only $link");
            let text = values
                .get("$link")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("AT JSON $link must be a CID string"))?;
            DagCbor::Link(parse_atproto_json_cid(text)?)
        }
        serde_json::Value::Object(values) if values.contains_key("$bytes") => {
            ensure!(values.len() == 1, "AT JSON bytes must contain only $bytes");
            let text = values
                .get("$bytes")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("AT JSON $bytes must be a base64 string"))?;
            let bytes = STANDARD
                .decode(text)
                .or_else(|_| STANDARD_NO_PAD.decode(text))
                .context("invalid AT JSON base64 bytes")?;
            DagCbor::Bytes(bytes)
        }
        serde_json::Value::Object(values) => DagCbor::Map(
            values
                .iter()
                .map(|(key, value)| Ok((DagCbor::Text(key.clone()), json_to_dag_cbor(value)?)))
                .collect::<Result<Vec<_>>>()?,
        ),
    })
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

        fn authorize_key_promotion(&self, _: &str, _: &str) -> Result<()> {
            ensure!(self.allow.load(Ordering::Acquire), "promotion denied");
            Ok(())
        }
    }

    fn post() -> DagCbor {
        DagCbor::str_map([
            ("$type", DagCbor::Text("app.bsky.feed.post".into())),
            ("text", DagCbor::Text("durable".into())),
        ])
    }

    fn create_request(rkey: u64, expected_prev: Option<Cid>) -> PublicCreateRequest {
        PublicCreateRequest {
            request_id: format!("req-{rkey}"),
            principal: "did:at9p:agent".into(),
            did: "did:web:tormentnexus.social".into(),
            collection: "app.bsky.feed.post".into(),
            rkey: Tid::from_raw(rkey).into(),
            value: post(),
            expected_prev,
        }
    }

    #[test]
    fn public_repo_account_locks_allow_independent_dids_and_serialize_same_did() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let key_a = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let key_b = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let next_b = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let next_b_vk = *next_b.verifying_key();
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let request_a = create_request(1, None);
        let writer_a =
            PublicRepoWriter::new(store.clone(), &request_a.did, key_a.clone(), gate.clone())
                .unwrap();
        let same_did =
            PublicRepoWriter::new(store.clone(), &request_a.did, key_a.clone(), gate.clone())
                .unwrap();
        assert!(Arc::ptr_eq(&writer_a.account, &same_did.account));
        let mut request_b = create_request(1, None);
        request_b.did = "did:web:independent.example.com".into();
        let did_b = request_b.did.clone();

        // Model a long account-A transaction/promotion without sleeping or
        // filling a large repo. Another A handle must wait; B must finish its
        // constructor, create and key promotion while A is still locked.
        let held = writer_a.account.active_key.lock();
        let (started_tx, started_rx) = mpsc::channel();
        let (same_tx, same_rx) = mpsc::channel();
        let same_thread = std::thread::spawn(move || {
            let _ = started_tx.send(());
            let _ = same_tx.send(same_did.create_record(request_a));
        });
        let other_store = store.clone();
        let (other_tx, other_rx) = mpsc::channel();
        let other_thread = std::thread::spawn(move || {
            let result = (|| -> Result<()> {
                let other = PublicRepoWriter::new(
                    other_store.clone(),
                    &request_b.did,
                    key_b.clone(),
                    gate,
                )?;
                let first = other.create_record(request_b.clone())?;
                let head =
                    other.promote_signing_key("did:at9p:agent", key_b.verifying_key(), next_b)?;
                ensure!(
                    head.is_some() && head != Some(first.commit_cid),
                    "B promotion must reconcile its head"
                );
                ensure!(
                    other.create_record(request_b)? == first,
                    "B retry must retain original receipt"
                );
                Ok(())
            })();
            let _ = other_tx.send(result);
        });
        let started = started_rx.recv_timeout(Duration::from_secs(10));
        let independent = other_rx.recv_timeout(Duration::from_secs(10));
        let same_while_locked = same_rx.try_recv();
        // Always release before assertions/joins: a regression must fail the
        // bounded timeout rather than leave worker threads deadlocked.
        drop(held);
        same_thread.join().unwrap();
        other_thread.join().unwrap();
        started.unwrap();
        independent
            .expect("unrelated DID blocked behind account A")
            .unwrap();
        assert!(
            matches!(same_while_locked, Err(mpsc::TryRecvError::Empty)),
            "same-DID write bypassed its transaction lock"
        );
        let result_a = same_rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let repo_a = store
            .snapshot("did:web:tormentnexus.social")
            .unwrap()
            .unwrap();
        assert_eq!(repo_a.commit.cid_atproto().unwrap(), result_a.commit_cid);
        repo_a.commit.verify_atproto(key_a.verifying_key()).unwrap();
        assert_eq!(
            writer_a.active_verifying_key().unwrap(),
            *key_a.verifying_key()
        );
        store
            .snapshot(&did_b)
            .unwrap()
            .unwrap()
            .commit
            .verify_atproto(&next_b_vk)
            .unwrap();
    }

    #[test]
    fn public_repo_rotation_reconciles_idle_head_and_surviving_writers() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let old = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let candidate = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let request = create_request(1, None);
        let writer =
            PublicRepoWriter::new(store.clone(), &request.did, old.clone(), gate.clone()).unwrap();
        let survivor =
            PublicRepoWriter::new(store.clone(), &request.did, old.clone(), gate.clone()).unwrap();
        let first = writer.create_record(request.clone()).unwrap();
        let original = store.snapshot(&request.did).unwrap().unwrap().commit;
        let original_block = store
            .db
            .get(commit_block_key(
                &request.did,
                &first.commit_cid.to_string(),
            ))
            .unwrap()
            .unwrap();
        let reconciled = writer
            .promote_signing_key("did:at9p:agent", old.verifying_key(), candidate.clone())
            .unwrap()
            .unwrap();
        assert_ne!(reconciled, first.commit_cid);
        assert_eq!(
            survivor.active_verifying_key().unwrap(),
            *candidate.verifying_key()
        );
        let idle = store.snapshot(&request.did).unwrap().unwrap().commit;
        assert_eq!(idle.unsigned(), original.unsigned());
        idle.verify_atproto(candidate.verifying_key()).unwrap();
        assert!(idle.verify_atproto(old.verifying_key()).is_err());
        assert_eq!(survivor.create_record(request.clone()).unwrap(), first);
        assert_eq!(
            store
                .db
                .get(commit_block_key(
                    &request.did,
                    &first.commit_cid.to_string()
                ))
                .unwrap()
                .unwrap(),
            original_block
        );
        assert!(
            PublicRepoWriter::new(store.clone(), &request.did, old.clone(), gate.clone()).is_err()
        );
        let second = survivor
            .create_record(create_request(2, Some(reconciled)))
            .unwrap();
        let head = store.snapshot(&request.did).unwrap().unwrap().commit;
        head.verify_atproto(candidate.verifying_key()).unwrap();
        assert!(head.verify_atproto(old.verifying_key()).is_err());
        assert_eq!(head.prev, Some(reconciled));
        assert_eq!(head.cid_atproto().unwrap(), second.commit_cid);
        drop(survivor);
        drop(writer);
        drop(store);
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        assert!(PublicRepoWriter::new(store.clone(), &request.did, old, gate.clone()).is_err());
        let reopened = PublicRepoWriter::new(store, &request.did, candidate, gate).unwrap();
        assert_eq!(reopened.create_record(request).unwrap(), first);
    }

    #[test]
    fn public_repo_rotation_failure_keeps_active_authority_and_durable_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let old = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let candidate = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let request = create_request(1, None);
        let writer = PublicRepoWriter::new(store.clone(), &request.did, old.clone(), gate).unwrap();
        writer.create_record(request.clone()).unwrap();
        let all_bytes = || {
            store
                .db
                .iterator(rocksdb::IteratorMode::Start)
                .map(|entry| entry.unwrap())
                .collect::<Vec<_>>()
        };
        let before = all_bytes();
        struct RecordOnly;
        impl PublicPublicationAuthorizer for RecordOnly {
            fn authorize(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
        }
        let unprivileged = PublicRepoWriter::new(
            store.clone(),
            &request.did,
            old.clone(),
            Arc::new(RecordOnly),
        )
        .unwrap();
        assert!(unprivileged
            .promote_signing_key("did:at9p:agent", old.verifying_key(), candidate.clone())
            .is_err());
        assert_eq!(writer.active_verifying_key().unwrap(), *old.verifying_key());
        assert_eq!(all_bytes(), before);
        assert!(writer
            .promote_signing_key(
                "did:at9p:agent",
                candidate.verifying_key(),
                candidate.clone()
            )
            .is_err());
        assert_eq!(writer.active_verifying_key().unwrap(), *old.verifying_key());
        assert_eq!(all_bytes(), before);
        // A validly encoded head signed by the wrong key must not be blessed
        // by rotation. Reconciliation fails without exposing the candidate.
        let head = store.snapshot(&request.did).unwrap().unwrap().commit;
        let substituted = Commit::sign_atproto(&head.unsigned(), &candidate).unwrap();
        store
            .db
            .put(
                commit_key(&request.did),
                substituted.to_atproto_dag_cbor().unwrap(),
            )
            .unwrap();
        let before = all_bytes();
        assert!(writer
            .promote_signing_key("did:at9p:agent", old.verifying_key(), candidate)
            .is_err());
        assert_eq!(writer.active_verifying_key().unwrap(), *old.verifying_key());
        assert_eq!(all_bytes(), before);
    }

    #[test]
    fn public_repo_oauth_port_authority_survives_commit_reopen_and_retry() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let mut request = create_request(1, None);
        request.did = "did:web:pds.example.test%3A8443".into();
        let writer =
            PublicRepoWriter::new(store.clone(), &request.did, key.clone(), gate.clone()).unwrap();
        let result = writer.create_record(request.clone()).unwrap();
        assert!(result
            .uri
            .starts_with("at://did:web:pds.example.test%3A8443/"));
        drop(writer);
        drop(store);
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let snapshot = store.snapshot(&request.did).unwrap().unwrap();
        assert_eq!(snapshot.commit.did, request.did);
        snapshot.commit.verify_atproto(key.verifying_key()).unwrap();
        let writer = PublicRepoWriter::new(store, &request.did, key, gate).unwrap();
        assert_eq!(writer.create_record(request).unwrap(), result);
    }

    #[test]
    fn public_repo_encoded_size_boundary_rejects_without_writes() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let writer = PublicRepoWriter::new(
            store.clone(),
            "did:web:tormentnexus.social",
            p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng),
            gate,
        )
        .unwrap();
        let sized_request = |rkey, prev, size| {
            let mut request = create_request(rkey, prev);
            let value = |payload| {
                DagCbor::str_map([
                    ("$type", DagCbor::Text(request.collection.clone())),
                    ("text", DagCbor::Text("x".repeat(payload))),
                ])
            };
            let overhead = AtprotoRecord::new(&request.collection, &request.rkey, value(1024))
                .unwrap()
                .bytes()
                .len()
                - 1024;
            request.value = value(size - overhead);
            assert_eq!(
                AtprotoRecord::new(&request.collection, &request.rkey, request.value.clone())
                    .unwrap()
                    .bytes()
                    .len(),
                size
            );
            request
        };
        let oversized = sized_request(1, None, MAX_PUBLIC_RECORD_BYTES + 1);
        assert!(writer
            .create_record(oversized)
            .unwrap_err()
            .to_string()
            .contains("canonical encoded limit"));
        assert!(store
            .db
            .iterator(rocksdb::IteratorMode::Start)
            .next()
            .is_none());
        let at_limit = sized_request(1, None, MAX_PUBLIC_RECORD_BYTES);
        let first = writer.create_record(at_limit.clone()).unwrap();
        assert_eq!(writer.create_record(at_limit).unwrap(), first);
        let all_bytes = || {
            store
                .db
                .iterator(rocksdb::IteratorMode::Start)
                .map(|item| item.unwrap())
                .collect::<Vec<_>>()
        };
        let before = all_bytes();
        assert!(writer
            .create_record(sized_request(
                2,
                Some(first.commit_cid),
                MAX_PUBLIC_RECORD_BYTES + 1
            ))
            .unwrap_err()
            .to_string()
            .contains("canonical encoded limit"));
        assert_eq!(
            all_bytes(),
            before,
            "oversize rejection must not modify record/head/intent/block state"
        );
        let snapshot = store
            .snapshot("did:web:tormentnexus.social")
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), first.commit_cid);
        assert_eq!(snapshot.records.len(), 1);
    }

    #[test]
    fn public_repo_rejects_invalid_authorities_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        for did in [
            "did:plc:abc",
            "did:plc:ewvi7nxzyoun6zhxrhs64oiz#key",
            "did:web:example.com#fragment",
            "did:web:example.com?query",
            "did:web:example.com/path",
            "did:web:example..com",
            "did:web:example.com%2Fpath",
            "did:web:example.arpa",
            "did:web:example.onion",
            "did:web:example.123",
            "did:web:example.1com",
            "did:web:127.0.0.1",
        ] {
            assert!(PublicRepoWriter::new(store.clone(), did, key.clone(), gate.clone()).is_err());
            assert!(store.snapshot(did).is_err());
        }
        assert!(store
            .db
            .iterator(rocksdb::IteratorMode::Start)
            .next()
            .is_none());
        let did = "did:plc:ewvi7nxzyoun6zhxrhs64oiz";
        let writer = PublicRepoWriter::new(store.clone(), did, key, gate).unwrap();
        let mut request = create_request(1, None);
        request.did = did.to_owned();
        writer.create_record(request).unwrap();
        assert_eq!(store.snapshot(did).unwrap().unwrap().commit.did, did);
    }

    #[test]
    fn public_repo_general_keys_preserve_profile_self_through_reopen_and_retry() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let mut profile = create_request(1, None);
        profile.collection = "app.bsky.actor.profile".into();
        profile.rkey = AtprotoRecordKey::new("self").unwrap();
        profile.value = DagCbor::str_map([
            ("$type", DagCbor::Text(profile.collection.clone())),
            ("displayName", DagCbor::Text("Profile".into())),
        ]);
        let writer =
            PublicRepoWriter::new(store.clone(), &profile.did, key.clone(), gate.clone()).unwrap();
        let first = writer.create_record(profile.clone()).unwrap();
        assert_eq!(
            first.uri,
            format!("at://{}/app.bsky.actor.profile/self", profile.did)
        );
        let mut post = create_request(2, Some(first.commit_cid));
        post.rkey = AtprotoRecordKey::new("custom-key:~").unwrap();
        let second = writer.create_record(post.clone()).unwrap();
        assert_eq!(writer.create_record(profile.clone()).unwrap(), first);
        let mut duplicate = profile.clone();
        duplicate.request_id = "duplicate-profile".into();
        duplicate.expected_prev = Some(second.commit_cid);
        assert!(writer
            .create_record(duplicate)
            .unwrap_err()
            .to_string()
            .contains("already exists"));
        drop(writer);
        drop(store);

        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let snapshot = store.snapshot(&profile.did).unwrap().unwrap();
        assert_eq!(snapshot.records.len(), 2);
        let stored = &snapshot.records[&(profile.collection.clone(), profile.rkey.clone())];
        assert_eq!(stored.rkey().as_str(), "self");
        assert_eq!(stored.cid(), first.cid);
        assert_eq!(stored.value(), &profile.value);
        assert!(snapshot.records.contains_key(&(post.collection, post.rkey)));
        snapshot.commit.verify_atproto(key.verifying_key()).unwrap();
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), second.commit_cid);
        let writer = PublicRepoWriter::new(store.clone(), &profile.did, key, gate).unwrap();
        assert_eq!(writer.create_record(profile).unwrap(), first);

        // Storage reconstruction validates keys instead of treating an
        // arbitrary path or separator as an ordinary record key.
        store
            .db
            .put(
                record_key(writer.did.as_str(), "app.bsky.actor.profile", "bad/key"),
                stored.bytes(),
            )
            .unwrap();
        assert!(store.snapshot(&writer.did).is_err());
    }

    #[test]
    fn authorized_create_is_atomic_durable_and_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PublicRepoStore::open(dir.path()).expect("store"));
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let writer = PublicRepoWriter::new(
            store.clone(),
            "did:web:tormentnexus.social",
            key.clone(),
            gate.clone(),
        )
        .expect("writer");
        let request = PublicCreateRequest {
            request_id: "req-1".into(),
            principal: "did:at9p:agent".into(),
            did: "did:web:tormentnexus.social".into(),
            collection: "app.bsky.feed.post".into(),
            rkey: Tid::from_raw(7).into(),
            value: post(),
            expected_prev: None,
        };
        let first = writer.create_record(request.clone()).expect("create");
        let retry = writer.create_record(request.clone()).expect("retry");
        assert_eq!(first, retry);
        let second = writer
            .create_record(PublicCreateRequest {
                request_id: "req-2".into(),
                principal: "did:at9p:agent".into(),
                did: "did:web:tormentnexus.social".into(),
                collection: "app.bsky.feed.post".into(),
                rkey: Tid::from_raw(8).into(),
                value: post(),
                expected_prev: Some(first.commit_cid),
            })
            .expect("second create");
        assert_ne!(first.commit_cid, second.commit_cid);
        assert_eq!(
            writer
                .create_record(request.clone())
                .expect("retry after head advance"),
            first
        );
        let snapshot = store
            .snapshot("did:web:tormentnexus.social")
            .expect("snapshot")
            .expect("repo");
        assert_eq!(snapshot.records.len(), 2);
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), second.commit_cid);
        snapshot
            .commit
            .verify_atproto(&writer.active_verifying_key().unwrap())
            .expect("commit signature");
        drop(writer);
        drop(store);
        let reopened = Arc::new(PublicRepoStore::open(dir.path()).expect("reopen"));
        assert_eq!(
            reopened
                .snapshot("did:web:tormentnexus.social")
                .expect("snapshot")
                .expect("repo")
                .records
                .len(),
            2
        );
        let writer = PublicRepoWriter::new(
            reopened.clone(),
            "did:web:tormentnexus.social",
            key,
            gate.clone(),
        )
        .expect("reopened writer");
        assert_eq!(
            writer
                .create_record(request.clone())
                .expect("durable retry"),
            first
        );
        assert_eq!(
            reopened
                .snapshot(&request.did)
                .unwrap()
                .unwrap()
                .commit
                .cid_atproto()
                .unwrap(),
            second.commit_cid
        );
        gate.allow.store(false, Ordering::Release);
        assert!(writer
            .create_record(request)
            .unwrap_err()
            .to_string()
            .contains("publication denied"));
    }

    #[test]
    fn retry_checks_intent_record_and_commit_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let request = create_request(1, None);
        let writer = PublicRepoWriter::new(store.clone(), &request.did, key, gate).unwrap();
        let first = writer.create_record(request.clone()).unwrap();
        let mut changed = request.clone();
        changed.value = DagCbor::str_map([
            ("$type", DagCbor::Text(request.collection.clone())),
            ("text", DagCbor::Text("changed".into())),
        ]);
        assert!(writer
            .create_record(changed)
            .unwrap_err()
            .to_string()
            .contains("different content"));
        let mut changed = request.clone();
        changed.principal = "another-agent".into();
        assert!(writer
            .create_record(changed)
            .unwrap_err()
            .to_string()
            .contains("different content"));

        let block_key = commit_block_key(&request.did, &first.commit_cid.to_string());
        let original = store.db.get(&block_key).unwrap().unwrap();
        let mut tampered = Commit::from_atproto_dag_cbor(&original).unwrap();
        tampered.rev = Tid::from_raw(tampered.rev.to_raw() + 1);
        store
            .db
            .put(&block_key, tampered.to_atproto_dag_cbor().unwrap())
            .unwrap();
        assert!(writer
            .create_record(request.clone())
            .unwrap_err()
            .to_string()
            .contains("does not match stored block"));
        store.db.delete(&block_key).unwrap();
        assert!(writer
            .create_record(request.clone())
            .unwrap_err()
            .to_string()
            .contains("commit block is missing"));
        store.db.put(&block_key, original).unwrap();

        // A valid repository containing another record must not satisfy an
        // intent whose record path has disappeared.
        let other_request = create_request(2, Some(first.commit_cid));
        writer.create_record(other_request).unwrap();
        let mut intent = store
            .intent(&request.did, &request.request_id)
            .unwrap()
            .unwrap();
        let missing = create_request(3, None);
        intent.rkey = missing.rkey.as_str().to_owned();
        store
            .db
            .put(
                intent_key(&request.did, &request.request_id),
                serde_json::to_vec(&intent).unwrap(),
            )
            .unwrap();
        let mut missing = missing;
        missing.request_id = request.request_id;
        assert!(writer
            .create_record(missing)
            .unwrap_err()
            .to_string()
            .contains("record does not match repository"));
    }

    #[test]
    fn snapshots_remain_consistent_during_concurrent_publications() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = *key.verifying_key();
        let writer =
            PublicRepoWriter::new(store.clone(), "did:web:tormentnexus.social", key, gate).unwrap();
        let first = writer.create_record(create_request(1, None)).unwrap();
        let start = std::sync::Barrier::new(4);
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| {
                    start.wait();
                    let mut reads = 0;
                    while !done.load(Ordering::Acquire) || reads == 0 {
                        let snapshot = store
                            .snapshot("did:web:tormentnexus.social")
                            .unwrap()
                            .unwrap();
                        assert!(!snapshot.records.is_empty());
                        snapshot.commit.verify_atproto(&verifying_key).unwrap();
                        reads += 1;
                    }
                });
            }
            start.wait();
            let mut head = first.commit_cid;
            for rkey in 2..=64 {
                head = writer
                    .create_record(create_request(rkey, Some(head)))
                    .unwrap()
                    .commit_cid;
            }
            done.store(true, Ordering::Release);
        });
        assert_eq!(
            store
                .snapshot("did:web:tormentnexus.social")
                .unwrap()
                .unwrap()
                .records
                .len(),
            64
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
                rkey: Tid::from_raw(1).into(),
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

/// Native callers retain genesis-or-exact CAS; XRPC may omit its condition.
enum PublicHeadCondition {
    Unconditional,
    Exact(Option<Cid>),
}

impl PublicHeadCondition {
    fn matches(&self, actual: Option<Cid>) -> bool {
        match self {
            Self::Unconditional => true,
            Self::Exact(expected) => *expected == actual,
        }
    }
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublicRepoStore").finish_non_exhaustive()
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
    fn create_record_with_condition(
        &self,
        request: PublicCreateRequest,
        condition: PublicHeadCondition,
    ) -> Result<PublicCommitResult> {
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
                intent.request_id == request.request_id
                    && intent.principal == request.principal
                    && intent.did == self.did
                    && intent.collection == record.collection()
                    && intent.rkey == record.rkey().as_str()
                    && intent.cid == record.cid().to_string(),
                "publication request id was reused with different content"
            );
            let snapshot = self
                .store
                .snapshot(&self.did)?
                .ok_or_else(|| anyhow!("publication intent exists without a repository"))?;
            ensure!(
                snapshot
                    .records
                    .get(&(request.collection.clone(), request.rkey))
                    .is_some_and(|stored| stored.cid() == record.cid()),
                "publication intent does not match stored record"
            );
            // The intent was persisted atomically with this immutable record.
            // Return its original result even after later writes advance the
            // head; a retry is not a new conditional write.
            let commit_cid = parse_atproto_json_cid(&intent.commit_cid)
                .context("publication intent has invalid commit CID")?;
            ensure!(
                commit_cid.as_bytes().get(1) == Some(&0x71),
                "publication intent commit must use DAG-CBOR"
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
                    condition.matches(Some(previous)),
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
                ensure!(condition.matches(None), "public repo genesis CAS conflict");
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
    fn transaction_fixture() -> (
        tempfile::TempDir,
        Arc<PublicRepoStore>,
        Arc<Gate>,
        PublicRepoWriter,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PublicRepoStore::open(dir.path()).expect("store"));
        let gate = Arc::new(Gate {
            allow: AtomicBool::new(true),
        });
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let writer = PublicRepoWriter::new(
            store.clone(),
            "did:web:tormentnexus.social",
            key,
            gate.clone(),
        )
        .expect("writer");
        (dir, store, gate, writer)
    }
    fn transaction_request(id: u64) -> PublicCreateRequest {
        PublicCreateRequest {
            request_id: format!("req-{id}"),
            principal: "did:at9p:agent".into(),
            did: "did:web:tormentnexus.social".into(),
            collection: "app.bsky.feed.post".into(),
            rkey: Tid::from_raw(id),
            value: post(),
            expected_prev: None,
        }
    }
    #[test]
    fn xrpc_omitted_swap_allows_subsequent_creates_without_weakening_native_cas() {
        let (_dir, store, _gate, writer) = transaction_fixture();
        let first = writer
            .create_record_with_expected_prev_text(transaction_request(7), None)
            .expect("genesis");
        let native_none = writer
            .create_record(transaction_request(8))
            .expect_err("native genesis only");
        assert!(native_none.to_string().contains("CAS conflict"));
        let second = writer
            .create_record_with_expected_prev_text(transaction_request(8), None)
            .expect("unconditional second write");
        let mut third = transaction_request(9);
        third.expected_prev = Some(first.commit_cid);
        assert!(
            writer.create_record(third.clone()).is_err(),
            "native stale CAS"
        );
        third.expected_prev = Some(second.commit_cid);
        writer.create_record(third).expect("native exact CAS");
        assert_eq!(
            store.snapshot(writer.did()).unwrap().unwrap().records.len(),
            3
        );
    }
    #[test]
    fn xrpc_retries_return_original_result_after_later_commit_and_reopen() {
        let (dir, store, gate, writer) = transaction_fixture();
        let first = writer
            .create_record_with_expected_prev_text(transaction_request(7), None)
            .expect("genesis");
        let previous = first.commit_cid.to_string();
        let second = writer
            .create_record_with_expected_prev_text(transaction_request(8), Some(&previous))
            .expect("conditional second write");
        assert_eq!(
            writer
                .create_record_with_expected_prev_text(transaction_request(8), Some(&previous))
                .unwrap(),
            second,
            "immediate retry must not compare against its own new head"
        );
        let third = writer
            .create_record_with_expected_prev_text(transaction_request(9), None)
            .expect("head advances again");
        let key = writer.signing_key.clone();
        drop(writer);
        drop(store);
        let store = Arc::new(PublicRepoStore::open(dir.path()).expect("reopen"));
        let writer = PublicRepoWriter::new(
            store.clone(),
            "did:web:tormentnexus.social",
            key,
            gate.clone(),
        )
        .unwrap();
        assert_eq!(
            writer
                .create_record_with_expected_prev_text(transaction_request(7), None)
                .unwrap(),
            first
        );
        assert_eq!(
            writer
                .create_record_with_expected_prev_text(transaction_request(8), Some(&previous))
                .unwrap(),
            second,
            "durable retry returns original commit, not the latest head"
        );
        let mut changed = transaction_request(8);
        changed.value = DagCbor::str_map([
            ("$type", DagCbor::Text("app.bsky.feed.post".into())),
            ("text", DagCbor::Text("changed".into())),
        ]);
        assert!(writer
            .create_record_with_expected_prev_text(changed, Some(&previous))
            .is_err());
        let mut different_principal = transaction_request(8);
        different_principal.principal = "did:at9p:other".into();
        assert!(writer
            .create_record_with_expected_prev_text(different_principal, Some(&previous))
            .is_err());
        gate.allow.store(false, Ordering::Release);
        let denied = writer
            .create_record_with_expected_prev_text(transaction_request(8), Some(&previous))
            .unwrap_err();
        assert!(
            denied.to_string().contains("publication denied"),
            "retries reauthorize"
        );
        let snapshot = store.snapshot(writer.did()).unwrap().unwrap();
        assert_eq!(snapshot.records.len(), 3);
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), third.commit_cid);
    }
    #[test]
    fn xrpc_cas_and_unconditional_writes_share_one_atomic_head_selection() {
        for conditional in [true, false] {
            let (_dir, store, _gate, writer) = transaction_fixture();
            let first = writer.create_record(transaction_request(7)).unwrap();
            let writer = Arc::new(writer);
            let start = Arc::new(std::sync::Barrier::new(2));
            let results = std::thread::scope(|scope| {
                let mut workers = Vec::new();
                for id in [8, 9] {
                    let writer = writer.clone();
                    let start = start.clone();
                    let expected = conditional.then(|| first.commit_cid.to_string());
                    workers.push(scope.spawn(move || {
                        start.wait();
                        writer.create_record_with_expected_prev_text(
                            transaction_request(id),
                            expected.as_deref(),
                        )
                    }));
                }
                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("writer thread"))
                    .collect::<Vec<_>>()
            });
            let expected_successes = if conditional { 1 } else { 2 };
            assert_eq!(
                results.iter().filter(|result| result.is_ok()).count(),
                expected_successes
            );
            for error in results.iter().filter_map(|result| result.as_ref().err()) {
                assert!(error.to_string().contains("CAS conflict"));
            }
            let snapshot = store.snapshot(writer.did()).unwrap().unwrap();
            assert_eq!(snapshot.records.len(), 1 + expected_successes);
            snapshot
                .commit
                .verify_atproto(writer.signing_key.verifying_key())
                .unwrap();
        }
    }
    #[test]
    fn xrpc_swap_authorizes_before_reading_repository_state() {
        let (_dir, store, gate, writer) = transaction_fixture();
        // A read would fail decoding this record before reaching the gate in
        // the old text adapter. A denied request must not inspect it at all.
        let key = record_key(
            writer.did(),
            "app.bsky.feed.post",
            Tid::from_raw(7).encode().as_str(),
        );
        store.db.put(&key, b"malformed record").unwrap();
        gate.allow.store(false, Ordering::Release);
        let expected = Cid::from_dag_cbor(b"any head").to_string();
        let error = writer
            .create_record_with_expected_prev_text(transaction_request(8), Some(&expected))
            .unwrap_err();
        assert!(error.to_string().contains("publication denied"));
        assert_eq!(store.db.get(key).unwrap().unwrap(), b"malformed record");
        assert!(store.db.get(commit_key(writer.did())).unwrap().is_none());
        assert!(store.intent(writer.did(), "req-8").unwrap().is_none());
    }
    #[test]
    fn json_links_and_bytes_match_independent_public_record_fixture() {
        // Independently encoded using Python hashlib/base64 and a minimal
        // RFC 8949 encoder with public map ordering and tag-42 links.
        let json = serde_json::json!({
            "$type": "app.bsky.feed.post",
            "text": "json",
            "payload": {"$bytes": "AQI="},
            "links": [{"$link": "bafyreifqwkmiw256ojf2zws6tzjeonw6bpd5vza4i22ccpcq4hjv2ts7cm"}],
            "blob": {
                "$type": "blob",
                "ref": {"$link": "bafkreifbfby75yqq7odbski6v2qziwa4xustdzfsg5m5ejpwqbush5rsei"},
                "mimeType": "image/png",
                "size": 2
            }
        });
        let value = json_to_dag_cbor(&json).expect("AT JSON");
        let record = AtprotoRecord::new("app.bsky.feed.post", Tid::from_raw(7), value)
            .expect("public record");
        let fixture = hex::decode(concat!(
            "a564626c6f62a463726566d82a58250001551220a12871fee210fb8619291eaea194581cbd2531e4b23759d225f6806923f63222",
            "6473697a650265247479706564626c6f62686d696d655479706569696d6167652f706e676474657874646a736f6e",
            "652474797065726170702e62736b792e666565642e706f7374656c696e6b7381d82a58250001711220b0b2988b6bbe724bacda5e9e524736de0bc7dae41c46b4213c50e1d35d4e5f13",
            "677061796c6f6164420102"
        )).expect("fixture");
        assert_eq!(record.bytes(), fixture);
        assert_eq!(
            record.cid().to_string(),
            "bafyreie3irn4tnywq3hxjm3knxuq7ovhni7cjqxa7weo7dncxsw6rj3s4m"
        );
        assert!(
            AtprotoRecord::from_bytes("app.bsky.feed.post", Tid::from_raw(7), &fixture).is_ok()
        );
    }
    #[test]
    fn json_bytes_accept_standard_base64_with_optional_padding() {
        for text in ["", "AQI", "AQI=", "+/8=", "+/8"] {
            let value = json_to_dag_cbor(&serde_json::json!({"$bytes": text})).expect("bytes");
            let expected = match text {
                "" => vec![],
                "AQI" | "AQI=" => vec![1, 2],
                _ => vec![251, 255],
            };
            assert_eq!(value, DagCbor::Bytes(expected));
        }
    }
    #[test]
    fn json_rejects_malformed_reserved_wrappers_at_any_depth() {
        let cid = Cid::from_raw(b"blob").to_string();
        let malformed = [
            serde_json::json!({"$link": false}),
            serde_json::json!({"$link": "not-a-cid"}),
            serde_json::json!({"$link": cid, "extra": 1}),
            serde_json::json!({"$link": cid, "$bytes": "AQI="}),
            serde_json::json!({"$bytes": 12}),
            serde_json::json!({"$bytes": "AQI=", "extra": 1}),
            serde_json::json!({"$bytes": "-_8="}),
            serde_json::json!({"$bytes": "AQJ="}),
            serde_json::json!({"$bytes": "AQI=="}),
        ];
        for value in malformed {
            assert!(json_to_dag_cbor(&value).is_err(), "{value}");
            assert!(json_to_dag_cbor(&serde_json::json!({"nested": [value]})).is_err());
        }
    }
    #[test]
    fn json_links_reject_noncanonical_and_unsupported_cids() {
        let cid = Cid::from_raw(b"blob");
        let text = cid.to_string();
        assert_eq!(parse_atproto_json_cid(&text).expect("canonical"), cid);
        assert!(parse_atproto_json_cid(&text.to_uppercase()).is_err());
        assert!(parse_atproto_json_cid(&format!("{text}=")).is_err());
        let mut noncanonical = text.into_bytes();
        let last = noncanonical.last_mut().expect("last digit");
        // The last base32 digit has two zero padding bits. Setting one leaves
        // the decoded CID bytes unchanged but must not be accepted.
        let alphabet = b"abcdefghijklmnopqrstuvwxyz234567";
        let position = alphabet.iter().position(|c| c == last).expect("base32");
        *last = alphabet[position + 1];
        assert!(parse_atproto_json_cid(&String::from_utf8(noncanonical).unwrap()).is_err());
        for (position, replacement) in [(1, 0x70), (2, 0x13), (3, 0x1f)] {
            let mut raw = cid.as_bytes().to_vec();
            raw[position] = replacement;
            // Encode invalid bytes independently of the production parser.
            let encoded = raw
                .iter()
                .flat_map(|byte| (0..8).rev().map(move |bit| (byte >> bit) & 1))
                .collect::<Vec<_>>();
            let mut text = String::from("b");
            for chunk in encoded.chunks(5) {
                let n = chunk.iter().fold(0u8, |n, bit| (n << 1) | bit) << (5 - chunk.len());
                text.push(alphabet[n as usize] as char);
            }
            assert!(parse_atproto_json_cid(&text).is_err());
        }
    }
}
