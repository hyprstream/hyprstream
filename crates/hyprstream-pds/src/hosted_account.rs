//! Hosted-account record and mint state machine.
//!
//! The mint path is intentionally split around the user signature:
//!
//! 1. [`HostedAccountMint::begin`] consumes an already-allocated account name
//!    and generates a fresh per-account ES256 `#atproto` key.
//! 2. The caller uses [`HostedAccountMint::atproto_verifying_key`] to build and
//!    seal the canonical DID document (B2/#1164), then supplies that document's
//!    CID to [`HostedAccountMint::prepare_genesis`].
//! 3. The user signs [`PendingHostedAccountMint::signing_bytes`] with the
//!    required priority-zero Hybrid rotation key.
//! 4. [`PendingHostedAccountMint::seal`] verifies that signature before it can
//!    produce an [`AccountRecord`].
//!
//! A host never receives the user's rotation private key. The generated ES256
//! private key is returned separately from the public account record so a
//! caller can place it in the deployment's secret store; it is never serialized
//! into the record or genesis operation.
//!
//! A2 (#1160) owns label allocation and never-reuse. A3 (#1162) owns deriving
//! the host from the configured deployment zone. This module consumes the
//! narrow seam between them as [`AllocatedAccountName`] and independently
//! revalidates the permanent host-form DID before signing anything.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use p256::ecdsa::{SigningKey as AtprotoSigningKey, VerifyingKey as AtprotoVerifyingKey};
use rand::rngs::OsRng;

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;
use crate::did_op::{
    validate_host_form_did_web, DidOpSignature, GenesisDidOp, GenesisRepoHead, GenesisRotationKeys,
    UnsignedGenesisDidOp,
};

/// Version of the durable hosted-account record.
pub const ACCOUNT_RECORD_VERSION: u16 = 1;
const COMPRESSED_P256_PUBLIC_KEY_LEN: usize = 33;
const ACCOUNT_RECORD_FILE: &str = "account-record.cbor";
const GENESIS_DID_OP_FILE: &str = "genesis.didop.cbor";
const ATPROTO_SIGNING_KEY_FILE: &str = "atproto-signing-key";

/// The output of the A2 allocation + A3 configured-zone seam.
///
/// Construction is strict but cannot itself prove never-reuse; the A2 allocator
/// must only produce this value after atomically reserving the label forever.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocatedAccountName {
    label: String,
    did: String,
}

impl AllocatedAccountName {
    pub fn new(label: impl Into<String>, did: impl Into<String>) -> Result<Self> {
        let name = Self {
            label: label.into(),
            did: did.into(),
        };
        name.validate()?;
        Ok(name)
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn did(&self) -> &str {
        &self.did
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.label.is_empty(), "allocated account label is empty");
        ensure!(
            self.label.len() <= 63,
            "allocated account label exceeds 63 octets"
        );
        ensure!(
            self.label == self.label.to_ascii_lowercase(),
            "allocated account label must already be lowercase"
        );
        ensure!(
            self.label
                .bytes()
                .all(|byte| { byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' }),
            "allocated account label must be a single LDH label"
        );
        ensure!(
            !self.label.starts_with('-') && !self.label.ends_with('-'),
            "allocated account label must not start or end with a hyphen"
        );
        validate_host_form_did_web(&self.did)?;
        let host = self
            .did
            .strip_prefix("did:web:")
            .ok_or_else(|| anyhow::anyhow!("hosted account DID must use did:web"))?;
        ensure!(
            host.split('.').next() == Some(self.label.as_str()),
            "allocated account label does not match the first label of its did:web host"
        );
        Ok(())
    }
}

/// The public durable record locating a hosted account's permanent genesis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccountRecord {
    name: AllocatedAccountName,
    atproto_key: Vec<u8>,
    genesis_op: Cid,
    current_op: Cid,
    doc_cid: Cid,
}

impl AccountRecord {
    fn new(
        name: AllocatedAccountName,
        atproto_key: Vec<u8>,
        genesis_op: Cid,
        doc_cid: Cid,
    ) -> Result<Self> {
        let record = Self {
            name,
            atproto_key,
            genesis_op,
            current_op: genesis_op,
            doc_cid,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn version(&self) -> u16 {
        ACCOUNT_RECORD_VERSION
    }

    pub fn name(&self) -> &AllocatedAccountName {
        &self.name
    }

    pub fn atproto_key_bytes(&self) -> &[u8] {
        &self.atproto_key
    }

    pub fn genesis_op(&self) -> Cid {
        self.genesis_op
    }

    pub fn current_op(&self) -> Cid {
        self.current_op
    }

    pub fn doc_cid(&self) -> Cid {
        self.doc_cid
    }

    pub fn atproto_verifying_key(&self) -> Result<AtprotoVerifyingKey> {
        AtprotoVerifyingKey::from_sec1_bytes(&self.atproto_key)
            .context("account record contains an invalid #atproto P-256 key")
    }

    pub fn to_dag_cbor(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(self.to_value().encode())
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        reject_unknown(
            &value,
            &[
                "version",
                "label",
                "did",
                "atproto_key",
                "genesis_op",
                "current_op",
                "doc_cid",
            ],
            "account record",
        )?;
        let version = required(&value, "version", "account record")?.as_unsigned()?;
        ensure!(
            version == u64::from(ACCOUNT_RECORD_VERSION),
            "unsupported account record version {version}"
        );
        let record = Self {
            name: AllocatedAccountName::new(
                required(&value, "label", "account record")?
                    .as_str()?
                    .to_owned(),
                required(&value, "did", "account record")?
                    .as_str()?
                    .to_owned(),
            )?,
            atproto_key: required(&value, "atproto_key", "account record")?
                .as_bytes()?
                .to_vec(),
            genesis_op: *required(&value, "genesis_op", "account record")?.as_link()?,
            current_op: *required(&value, "current_op", "account record")?.as_link()?,
            doc_cid: *required(&value, "doc_cid", "account record")?.as_link()?,
        };
        record.validate()?;
        ensure!(
            record.to_dag_cbor()? == bytes,
            "account record is not canonical DAG-CBOR"
        );
        Ok(record)
    }

    fn validate(&self) -> Result<()> {
        self.name.validate()?;
        ensure!(
            self.atproto_key.len() == COMPRESSED_P256_PUBLIC_KEY_LEN,
            "account #atproto key must be a {COMPRESSED_P256_PUBLIC_KEY_LEN}-byte compressed P-256 point"
        );
        self.atproto_verifying_key()?;
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("atproto_key", DagCbor::Bytes(self.atproto_key.clone())),
            ("current_op", DagCbor::Link(self.current_op)),
            ("did", DagCbor::Text(self.name.did.clone())),
            ("doc_cid", DagCbor::Link(self.doc_cid)),
            ("genesis_op", DagCbor::Link(self.genesis_op)),
            ("label", DagCbor::Text(self.name.label.clone())),
            (
                "version",
                DagCbor::Unsigned(u64::from(ACCOUNT_RECORD_VERSION)),
            ),
        ])
    }
}

/// Mint state after label allocation and per-account ES256 generation.
///
/// This type is deliberately neither `Clone` nor `Debug`: it owns secret key
/// material.
pub struct HostedAccountMint {
    name: AllocatedAccountName,
    rotations: GenesisRotationKeys,
    atproto_signing_key: AtprotoSigningKey,
}

impl HostedAccountMint {
    /// Begin minting. A user-held priority-zero key is mandatory in
    /// `rotations`; there is no overload that omits it.
    pub fn begin(name: AllocatedAccountName, rotations: GenesisRotationKeys) -> Result<Self> {
        name.validate()?;
        // Reconstructing from canonical bytes re-applies every slot invariant
        // even if this crate later gains internal mutation helpers.
        let rotations = GenesisRotationKeys::new(
            rotations.user().clone(),
            rotations.recovery().clone(),
            rotations.host().clone(),
        )?;
        Ok(Self {
            name,
            rotations,
            atproto_signing_key: AtprotoSigningKey::random(&mut OsRng),
        })
    }

    /// The generated account-specific `#atproto` key used by B2 to build the
    /// canonical DID document before genesis is signed.
    pub fn atproto_verifying_key(&self) -> AtprotoVerifyingKey {
        *self.atproto_signing_key.verifying_key()
    }

    /// Bind the sealed DID document and deliberate genesis repo-head state.
    pub fn prepare_genesis(
        self,
        doc_cid: Cid,
        head_at_op: GenesisRepoHead,
    ) -> Result<PendingHostedAccountMint> {
        let unsigned =
            UnsignedGenesisDidOp::new(self.name.did.clone(), doc_cid, self.rotations, head_at_op)?;
        Ok(PendingHostedAccountMint {
            name: self.name,
            atproto_signing_key: self.atproto_signing_key,
            unsigned,
        })
    }
}

/// Mint state waiting for the user-held priority-zero signature.
///
/// No [`AccountRecord`] exists yet.
pub struct PendingHostedAccountMint {
    name: AllocatedAccountName,
    atproto_signing_key: AtprotoSigningKey,
    unsigned: UnsignedGenesisDidOp,
}

impl PendingHostedAccountMint {
    pub fn unsigned_genesis(&self) -> &UnsignedGenesisDidOp {
        &self.unsigned
    }

    /// Exact canonical bytes the user's Hybrid signer must sign.
    pub fn signing_bytes(&self) -> Result<Vec<u8>> {
        self.unsigned.to_dag_cbor()
    }

    /// Verify the user signature and produce the sealed account artifacts.
    ///
    /// Invalid, host-signed, classical-only, or tampered signatures fail before
    /// an account record is constructed.
    pub fn seal(self, signature: DidOpSignature) -> Result<SealedHostedAccount> {
        let genesis = GenesisDidOp::seal(self.unsigned, signature)?;
        let genesis_bytes = genesis.to_dag_cbor()?;
        let genesis_cid = Cid::from_dag_cbor(&genesis_bytes);
        let atproto_key = self
            .atproto_signing_key
            .verifying_key()
            .to_encoded_point(true)
            .as_bytes()
            .to_vec();
        let record = AccountRecord::new(
            self.name,
            atproto_key,
            genesis_cid,
            genesis.unsigned().doc_cid(),
        )?;
        let record_bytes = record.to_dag_cbor()?;
        Ok(SealedHostedAccount {
            record,
            record_bytes,
            genesis,
            genesis_bytes,
            atproto_signing_key: self.atproto_signing_key,
        })
    }
}

/// Fully sealed account artifacts ready for an atomic persistence boundary.
///
/// The secret ES256 key is separate from both byte vectors. Persistence is a
/// caller-owned seam because the deployment decides its secret store and B2
/// decides its CAS; this type ensures those writers receive one coherent,
/// already-verified generation.
pub struct SealedHostedAccount {
    record: AccountRecord,
    record_bytes: Vec<u8>,
    genesis: GenesisDidOp,
    genesis_bytes: Vec<u8>,
    atproto_signing_key: AtprotoSigningKey,
}

impl SealedHostedAccount {
    pub fn record(&self) -> &AccountRecord {
        &self.record
    }

    pub fn record_bytes(&self) -> &[u8] {
        &self.record_bytes
    }

    pub fn genesis(&self) -> &GenesisDidOp {
        &self.genesis
    }

    pub fn genesis_bytes(&self) -> &[u8] {
        &self.genesis_bytes
    }

    pub fn atproto_signing_key(&self) -> &AtprotoSigningKey {
        &self.atproto_signing_key
    }

    /// Consume the bundle for a persistence implementation.
    pub fn into_parts(
        self,
    ) -> (
        AccountRecord,
        Vec<u8>,
        GenesisDidOp,
        Vec<u8>,
        AtprotoSigningKey,
    ) {
        (
            self.record,
            self.record_bytes,
            self.genesis,
            self.genesis_bytes,
            self.atproto_signing_key,
        )
    }

    /// Persist this mint generation and return its public account record.
    ///
    /// Retaining the bundle permits a retry after a crash or I/O error. The
    /// sealed record fixes its label, so it cannot be redirected to another
    /// account location.
    pub fn write_to(&self, store: &DirectoryHostedAccountStore) -> Result<AccountRecord> {
        store.write_new(self)?;
        Ok(self.record.clone())
    }
}

/// Fail-closed filesystem persistence for newly minted hosted accounts.
///
/// Each allocated label gets one directory beneath `root`. Files are first
/// written and synced in an unpublished private staging directory, then that
/// complete directory is atomically published without replacement. Files are
/// written in this order:
///
/// 1. the generated `#atproto` private key;
/// 2. the sealed genesis operation;
/// 3. the public account record, **last**, as the publication marker.
///
/// A crash before publication leaves no label directory. A crash after
/// publication leaves the complete account. A retry removes only an invalid or
/// incomplete legacy residue; it never replaces a complete account.
/// C2 (#1168) owns later crash-atomic *rotation*; this writer only establishes
/// the immutable first generation.
#[derive(Clone, Debug)]
pub struct DirectoryHostedAccountStore {
    root: PathBuf,
    #[cfg(test)]
    fault: Option<WriteFault>,
}

impl DirectoryHostedAccountStore {
    /// Configure the caller-selected durable account root.
    ///
    /// There is intentionally no default path.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            #[cfg(test)]
            fault: None,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Location of an account record after a successful mint.
    pub fn account_record_path(&self, label: &str) -> PathBuf {
        self.root.join(label).join(ACCOUNT_RECORD_FILE)
    }

    fn write_new(&self, account: &SealedHostedAccount) -> Result<()> {
        validate_sealed_bundle(account)?;
        ensure_private_directory(&self.root)?;
        let account_dir = self.root.join(account.record.name().label());
        if complete_account_matches(&account_dir, account)? {
            return Ok(());
        }
        remove_incomplete_account_directory(&account_dir)?;

        let staging_dir = self.new_staging_directory(account.record.name().label())?;
        self.inject_fault(WriteFault::StagingDirectoryCreate)?;

        self.write_new_file(
            &staging_dir.join(ATPROTO_SIGNING_KEY_FILE),
            account.atproto_signing_key.to_bytes().as_slice(),
            WriteFault::AtprotoFileCreate,
            WriteFault::AtprotoFileWrite,
            WriteFault::AtprotoFileSync,
        )?;
        self.write_new_file(
            &staging_dir.join(GENESIS_DID_OP_FILE),
            &account.genesis_bytes,
            WriteFault::GenesisFileCreate,
            WriteFault::GenesisFileWrite,
            WriteFault::GenesisFileSync,
        )?;
        self.write_new_file(
            &staging_dir.join(ACCOUNT_RECORD_FILE),
            &account.record_bytes,
            WriteFault::RecordFileCreate,
            WriteFault::RecordFileWrite,
            WriteFault::RecordFileSync,
        )?;
        sync_directory(&staging_dir)?;
        self.inject_fault(WriteFault::StagingDirectorySync)?;

        match rename_no_replace(&staging_dir, &account_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if complete_account_matches(&account_dir, account)? {
                    return Ok(());
                }
                bail!(
                    "hosted-account label directory {:?} was concurrently claimed by an incomplete account",
                    account_dir
                );
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to atomically publish hosted-account label directory {:?}",
                        account_dir
                    )
                });
            }
        }
        self.inject_fault(WriteFault::Publish)?;
        sync_directory(&self.root)?;
        self.inject_fault(WriteFault::RootDirectorySync)?;
        Ok(())
    }

    fn new_staging_directory(&self, label: &str) -> Result<PathBuf> {
        for attempt in 0_u32..128 {
            let staging = self.root.join(format!(
                ".{label}.mint-{}-{attempt}",
                rand::random::<u128>()
            ));
            match std::fs::create_dir(&staging) {
                Ok(()) => {
                    set_private_directory_permissions(&staging)?;
                    return Ok(staging);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to create hosted-account staging directory {staging:?}")
                    });
                }
            }
        }
        bail!("could not allocate a unique hosted-account staging directory")
    }

    fn write_new_file(
        &self,
        path: &Path,
        bytes: &[u8],
        create_fault: WriteFault,
        write_fault: WriteFault,
        sync_fault: WriteFault,
    ) -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options
            .open(path)
            .with_context(|| format!("failed to create hosted-account file {path:?}"))?;
        self.inject_fault(create_fault)?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write hosted-account file {path:?}"))?;
        self.inject_fault(write_fault)?;
        file.sync_all()
            .with_context(|| format!("failed to sync hosted-account file {path:?}"))?;
        self.inject_fault(sync_fault)
    }

    #[cfg(test)]
    fn with_fault(root: impl Into<PathBuf>, fault: WriteFault) -> Self {
        Self {
            root: root.into(),
            fault: Some(fault),
        }
    }

    #[cfg(test)]
    fn inject_fault(&self, point: WriteFault) -> Result<()> {
        if self.fault == Some(point) {
            bail!("injected crash after {point:?}")
        }
        Ok(())
    }

    #[cfg(not(test))]
    fn inject_fault(&self, _point: WriteFault) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WriteFault {
    StagingDirectoryCreate,
    AtprotoFileCreate,
    AtprotoFileWrite,
    AtprotoFileSync,
    GenesisFileCreate,
    GenesisFileWrite,
    GenesisFileSync,
    RecordFileCreate,
    RecordFileWrite,
    RecordFileSync,
    StagingDirectorySync,
    Publish,
    RootDirectorySync,
}

#[cfg(test)]
const WRITE_FAULTS: [WriteFault; 13] = [
    WriteFault::StagingDirectoryCreate,
    WriteFault::AtprotoFileCreate,
    WriteFault::AtprotoFileWrite,
    WriteFault::AtprotoFileSync,
    WriteFault::GenesisFileCreate,
    WriteFault::GenesisFileWrite,
    WriteFault::GenesisFileSync,
    WriteFault::RecordFileCreate,
    WriteFault::RecordFileWrite,
    WriteFault::RecordFileSync,
    WriteFault::StagingDirectorySync,
    WriteFault::Publish,
    WriteFault::RootDirectorySync,
];

/// Return `true` only for a fully valid, byte-identical already-published
/// account. Invalid or partial legacy directories are deliberately reported as
/// incomplete so a mint retry can replace them through the no-replace publish
/// transition; a valid different account is never removed or overwritten.
fn complete_account_matches(path: &Path, expected: &SealedHostedAccount) -> Result<bool> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect hosted-account directory {path:?}"));
        }
    };
    ensure!(
        metadata.file_type().is_dir(),
        "hosted-account label path {path:?} is not a directory"
    );

    let read = |name: &str| -> Result<Option<Vec<u8>>> {
        let file = path.join(name);
        match std::fs::read(&file) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => {
                Err(error).with_context(|| format!("failed to read hosted-account file {file:?}"))
            }
        }
    };
    let (Some(record_bytes), Some(genesis_bytes), Some(secret_bytes)) = (
        read(ACCOUNT_RECORD_FILE)?,
        read(GENESIS_DID_OP_FILE)?,
        read(ATPROTO_SIGNING_KEY_FILE)?,
    ) else {
        return Ok(false);
    };

    let Ok(record) = AccountRecord::from_dag_cbor(&record_bytes) else {
        return Ok(false);
    };
    let Ok(genesis) = GenesisDidOp::from_dag_cbor(&genesis_bytes) else {
        return Ok(false);
    };
    let Ok(secret) = AtprotoSigningKey::from_slice(&secret_bytes) else {
        return Ok(false);
    };
    if record.genesis_op() != genesis.cid()?
        || record.current_op() != record.genesis_op()
        || record.doc_cid() != genesis.unsigned().doc_cid()
        || record.atproto_verifying_key()?.to_encoded_point(true)
            != secret.verifying_key().to_encoded_point(true)
    {
        return Ok(false);
    }

    ensure!(
        record_bytes == expected.record_bytes
            && genesis_bytes == expected.genesis_bytes
            && secret_bytes == expected.atproto_signing_key.to_bytes().as_slice(),
        "hosted-account label directory {path:?} already contains a different complete account"
    );
    Ok(true)
}

fn remove_incomplete_account_directory(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_dir(),
                "hosted-account label path {path:?} is not a directory"
            );
            std::fs::remove_dir_all(path).with_context(|| {
                format!("failed to remove incomplete hosted-account directory {path:?}")
            })?;
            sync_directory(
                path.parent()
                    .ok_or_else(|| anyhow::anyhow!("hosted-account directory has no parent"))?,
            )?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect hosted-account directory {path:?}")),
    }
}

#[cfg(target_os = "linux")]
fn rename_no_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        from,
        rustix::fs::CWD,
        to,
        rustix::fs::RenameFlags::NOREPLACE,
    )?;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn rename_no_replace(_from: &Path, _to: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic no-replace directory publication requires Linux renameat2",
    ))
}

fn validate_sealed_bundle(account: &SealedHostedAccount) -> Result<()> {
    ensure!(
        AccountRecord::from_dag_cbor(&account.record_bytes)? == account.record,
        "sealed account record bytes do not match the in-memory record"
    );
    ensure!(
        GenesisDidOp::from_dag_cbor(&account.genesis_bytes)? == account.genesis,
        "sealed genesis bytes do not match the in-memory operation"
    );
    ensure!(
        account.record.genesis_op() == account.genesis.cid()?,
        "account record does not name its sealed genesis operation"
    );
    ensure!(
        account.record.doc_cid() == account.genesis.unsigned().doc_cid(),
        "account record and genesis operation disagree on the DID document CID"
    );
    ensure!(
        account
            .record
            .atproto_verifying_key()?
            .to_encoded_point(true)
            == account
                .atproto_signing_key
                .verifying_key()
                .to_encoded_point(true),
        "account record #atproto public key does not match the generated private key"
    );
    Ok(())
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("failed to create hosted-account root {path:?}"))?;
    ensure!(
        std::fs::symlink_metadata(path)?.file_type().is_dir(),
        "hosted-account root is not a directory"
    );
    set_private_directory_permissions(path)
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed to secure hosted-account directory {path:?}"))
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("failed to open hosted-account directory {path:?} for sync"))?
        .sync_all()
        .with_context(|| format!("failed to sync hosted-account directory {path:?}"))
}

fn required<'a>(value: &'a DagCbor, key: &str, what: &str) -> Result<&'a DagCbor> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{what} missing required field {key:?}"))
}

fn reject_unknown(value: &DagCbor, allowed: &[&str], what: &str) -> Result<()> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    for (key, _) in value.as_map()? {
        let key = key.as_str()?;
        if !allowed.contains(key) {
            bail!("{what} contains unknown field {key:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::did_op::{
        sign_genesis, HostKeyEnrollment, HybridRotationKey, RecoveryKeyEnrollment, UserRotationKey,
    };
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};

    struct UserSigner {
        ed: SigningKey,
        pq: MlDsaSigningKey,
        public: HybridRotationKey,
    }

    fn user_signer() -> UserSigner {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let public =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&pq_vk)).unwrap();
        UserSigner { ed, pq, public }
    }

    fn begin(user: &UserSigner) -> HostedAccountMint {
        HostedAccountMint::begin(
            AllocatedAccountName::new("alice", "did:web:alice.acct.example.com").unwrap(),
            GenesisRotationKeys::new(
                UserRotationKey::new(user.public.clone()),
                RecoveryKeyEnrollment::Declined,
                HostKeyEnrollment::Absent,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn mint_generates_account_specific_atproto_key_and_never_serializes_secret() {
        let user = user_signer();
        let first = begin(&user);
        let first_public = first.atproto_verifying_key();
        let second = begin(&user);
        assert_ne!(
            first_public.to_encoded_point(true),
            second.atproto_verifying_key().to_encoded_point(true)
        );

        let pending = first
            .prepare_genesis(Cid::from_raw(b"did document"), GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &user.ed, &user.pq).unwrap();
        let sealed = pending.seal(signature).unwrap();
        assert_eq!(
            sealed
                .record()
                .atproto_verifying_key()
                .unwrap()
                .to_encoded_point(true),
            first_public.to_encoded_point(true)
        );
        let secret = sealed.atproto_signing_key().to_bytes();
        assert!(
            !sealed
                .record_bytes()
                .windows(secret.len())
                .any(|window| window == secret.as_slice()),
            "account record must never contain the #atproto private key"
        );
        assert!(
            !sealed
                .genesis_bytes()
                .windows(secret.len())
                .any(|window| window == secret.as_slice()),
            "genesis operation must never contain the #atproto private key"
        );
    }

    #[test]
    fn account_record_is_not_produced_before_valid_user_signature() {
        let user = user_signer();
        let attacker = user_signer();
        let pending = begin(&user)
            .prepare_genesis(Cid::from_raw(b"did document"), GenesisRepoHead::EmptyRepo)
            .unwrap();
        assert!(sign_genesis(pending.unsigned_genesis(), &attacker.ed, &attacker.pq).is_err());
    }

    #[test]
    fn account_and_genesis_roundtrip_byte_exactly() {
        let user = user_signer();
        let pending = begin(&user)
            .prepare_genesis(Cid::from_raw(b"did document"), GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &user.ed, &user.pq).unwrap();
        let sealed = pending.seal(signature).unwrap();
        let account = AccountRecord::from_dag_cbor(sealed.record_bytes()).unwrap();
        assert_eq!(account.to_dag_cbor().unwrap(), sealed.record_bytes());
        let genesis = GenesisDidOp::from_dag_cbor(sealed.genesis_bytes()).unwrap();
        assert_eq!(genesis.to_dag_cbor().unwrap(), sealed.genesis_bytes());
        assert_eq!(account.genesis_op(), genesis.cid().unwrap());
        assert_eq!(account.current_op(), account.genesis_op());
        assert_eq!(account.doc_cid(), genesis.unsigned().doc_cid());
    }

    #[test]
    fn allocated_name_rejects_path_form_and_label_mismatch() {
        assert!(AllocatedAccountName::new("alice", "did:web:example.com:users:alice").is_err());
        assert!(AllocatedAccountName::new("alice", "did:web:bob.acct.example.com").is_err());
        assert!(AllocatedAccountName::new(
            "alice.example",
            "did:web:alice.example.acct.example.com"
        )
        .is_err());
    }

    #[test]
    fn operation_one_version_is_sealed_before_account_record() {
        let user = user_signer();
        let pending = begin(&user)
            .prepare_genesis(Cid::from_raw(b"did document"), GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signing_value = DagCbor::decode(&pending.signing_bytes().unwrap()).unwrap();
        assert_eq!(
            signing_value.get("version").unwrap().as_unsigned().unwrap(),
            1
        );
        let signature = sign_genesis(pending.unsigned_genesis(), &user.ed, &user.pq).unwrap();
        assert!(pending.seal(signature).is_ok());
    }

    #[test]
    fn durable_writer_publishes_record_last_and_never_overwrites() {
        let user = user_signer();
        let pending = begin(&user)
            .prepare_genesis(Cid::from_raw(b"did document"), GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &user.ed, &user.pq).unwrap();
        let sealed = pending.seal(signature).unwrap();
        let expected_record = sealed.record().clone();
        let expected_record_bytes = sealed.record_bytes().to_vec();
        let expected_genesis_bytes = sealed.genesis_bytes().to_vec();
        let expected_secret = sealed.atproto_signing_key().to_bytes();

        let temporary = tempfile::tempdir().unwrap();
        let store = DirectoryHostedAccountStore::new(temporary.path().join("accounts"));
        let stored_record = sealed.write_to(&store).unwrap();
        assert_eq!(stored_record, expected_record);

        let account_dir = store.root().join("alice");
        assert_eq!(
            std::fs::read(account_dir.join(ACCOUNT_RECORD_FILE)).unwrap(),
            expected_record_bytes
        );
        assert_eq!(
            std::fs::read(account_dir.join(GENESIS_DID_OP_FILE)).unwrap(),
            expected_genesis_bytes
        );
        assert_eq!(
            std::fs::read(account_dir.join(ATPROTO_SIGNING_KEY_FILE)).unwrap(),
            expected_secret.as_slice()
        );

        // The label directory is the exclusive never-overwrite boundary.
        let second = begin(&user)
            .prepare_genesis(
                Cid::from_raw(b"other did document"),
                GenesisRepoHead::EmptyRepo,
            )
            .unwrap();
        let second_signature = sign_genesis(second.unsigned_genesis(), &user.ed, &user.pq).unwrap();
        assert!(second
            .seal(second_signature)
            .unwrap()
            .write_to(&store)
            .is_err());
        assert_eq!(
            std::fs::read(account_dir.join(ACCOUNT_RECORD_FILE)).unwrap(),
            expected_record_bytes
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(account_dir.join(ATPROTO_SIGNING_KEY_FILE))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn durable_writer_recovers_from_every_interrupted_boundary() {
        let user = user_signer();
        let pending = begin(&user)
            .prepare_genesis(Cid::from_raw(b"did document"), GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &user.ed, &user.pq).unwrap();
        let sealed = pending.seal(signature).unwrap();

        for fault in WRITE_FAULTS {
            let temporary = tempfile::tempdir().unwrap();
            let root = temporary.path().join("accounts");
            let interrupted = DirectoryHostedAccountStore::with_fault(&root, fault);
            assert!(
                interrupted.write_new(&sealed).is_err(),
                "fault injection at {fault:?} must interrupt mint publication"
            );

            let final_directory = root.join(sealed.record().name().label());
            if final_directory.exists() {
                assert!(
                    complete_account_matches(&final_directory, &sealed).unwrap(),
                    "a crash at {fault:?} may publish only a complete account"
                );
            }

            let recovery = DirectoryHostedAccountStore::new(&root);
            recovery.write_new(&sealed).unwrap();
            assert!(complete_account_matches(&final_directory, &sealed).unwrap());
        }
    }
}
