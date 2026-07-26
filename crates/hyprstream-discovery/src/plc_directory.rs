//! Operable, fail-closed directory for Hyprstream's hybrid `did:plc` log.
//!
//! This is deliberately separate from `hyprstream_rpc::did_plc`: that module
//! remains the read-only federation resolver for classical PLC directories.
//! This module owns only locally registered hybrid identities and never stores
//! or returns a tenant.
//!
//! Directory records contain canonical operation bytes and a PDS endpoint.
//! They do not contain live transport reach. Every identity read decodes and
//! verifies the complete operation log with [`crate::did_op::verify_did_op_log`].
//! Every connect-time read then asks [`LivePlcDiscovery`] for current reach, so
//! certificate rotation cannot leave stale pins on an inventory/directory row.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use ed25519_dalek::SigningKey;
use hyprstream_crypto::cose_sign::{
    assemble_composite_nested, sign_composite, split_composite, verify_composite,
};
use hyprstream_crypto::pq::{ml_dsa_vk_bytes, MlDsaSigningKey};
use hyprstream_pds::dag_cbor::DagCbor;
use hyprstream_rpc::identity::UNAUTHENTICATED_DID_SENTINEL;
use hyprstream_rpc::service_entry::BrowserQuicReach;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::did_op::{verify_did_op_log, DidOp, HybridRotationKey, VerifiedDidOpLog};
use crate::RegistryDeploymentVerifier;

const DIRECTORY_GENESIS_TYPE: &str = "plc_directory_genesis";
const DIRECTORY_GENESIS_SIGNATURE_CONTEXT: &[u8] = b"hyprstream.plc-directory-genesis/1";

/// Hybrid-signed binding between operation zero and its PDS resolution pointer.
///
/// The merged operation-log format intentionally contains only authorization
/// state. This companion artifact prevents the directory store from changing a
/// PDS endpoint without a signature by one of genesis's declared Hybrid keys.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedPlcDirectoryGenesis {
    did: String,
    genesis_cid: String,
    pds_endpoint: String,
    signature: crate::did_op::HybridDidOpSignature,
}

impl SignedPlcDirectoryGenesis {
    /// Create the self-service artifact on the identity holder's signer.
    pub fn sign(
        genesis: &DidOp,
        pds_endpoint: &str,
        ed25519: &SigningKey,
        mldsa65: &MlDsaSigningKey,
    ) -> Result<Self> {
        validate_https_endpoint(pds_endpoint, "PDS endpoint")?;
        let did = genesis.genesis_did()?;
        ensure!(
            genesis
                .rotation_keys
                .iter()
                .any(|key| key.matches_signers(ed25519, mldsa65)),
            "directory genesis signer is not one of the declared Hybrid rotation keys"
        );
        let mut binding = Self {
            did,
            genesis_cid: genesis.cid().encode(),
            pds_endpoint: pds_endpoint.to_owned(),
            signature: crate::did_op::HybridDidOpSignature {
                ed25519: Vec::new(),
                mldsa65: Vec::new(),
            },
        };
        let composite = sign_composite(
            ed25519,
            Some(mldsa65),
            &binding.signable_bytes(),
            DIRECTORY_GENESIS_SIGNATURE_CONTEXT,
        )
        .context("hybrid-signing PLC directory genesis binding")?;
        let (ed25519, mldsa65) =
            split_composite(&composite).context("splitting directory genesis signature")?;
        binding.signature = crate::did_op::HybridDidOpSignature {
            ed25519,
            mldsa65: mldsa65.ok_or_else(|| {
                anyhow::anyhow!("directory genesis signing produced no ML-DSA-65 component")
            })?,
        };
        binding.signature.validate_shape()?;
        Ok(binding)
    }

    #[must_use]
    pub fn did(&self) -> &str {
        &self.did
    }

    #[must_use]
    pub fn pds_endpoint(&self) -> &str {
        &self.pds_endpoint
    }

    /// Canonical signed storage form.
    #[must_use]
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value(true).encode()
    }

    /// Strictly decode the canonical signed storage form.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes).context("decoding directory genesis binding")?;
        require_exact_fields(
            &value,
            &["did", "genesisCid", "pdsEndpoint", "sig", "type"],
            "directory genesis binding",
        )?;
        ensure!(
            required(&value, "type")?.as_str()? == DIRECTORY_GENESIS_TYPE,
            "unsupported directory genesis binding type"
        );
        let signature = required(&value, "sig")?;
        require_exact_fields(
            signature,
            &["ed25519", "mldsa65"],
            "directory genesis signature",
        )?;
        let binding = Self {
            did: required(&value, "did")?.as_str()?.to_owned(),
            genesis_cid: required(&value, "genesisCid")?.as_str()?.to_owned(),
            pds_endpoint: required(&value, "pdsEndpoint")?.as_str()?.to_owned(),
            signature: crate::did_op::HybridDidOpSignature {
                ed25519: required(signature, "ed25519")?.as_bytes()?.to_vec(),
                mldsa65: required(signature, "mldsa65")?.as_bytes()?.to_vec(),
            },
        };
        binding.signature.validate_shape()?;
        ensure!(
            binding.to_dag_cbor() == bytes,
            "directory genesis binding is not canonical DAG-CBOR"
        );
        Ok(binding)
    }

    fn verify(&self, expected_did: &str, genesis: &DidOp) -> Result<()> {
        ensure!(
            self.did == expected_did,
            "directory genesis binding DID does not match requested DID"
        );
        ensure!(
            self.genesis_cid == genesis.cid().encode(),
            "directory genesis binding does not match operation zero"
        );
        validate_https_endpoint(&self.pds_endpoint, "PDS endpoint")?;
        self.signature.validate_shape()?;
        let payload = self.signable_bytes();
        let mut errors = Vec::new();
        for key in &genesis.rotation_keys {
            let result = (|| {
                let (ed25519, mldsa65) = key.verifying_keys()?;
                let composite = assemble_composite_nested(
                    (ed25519.to_bytes().to_vec(), self.signature.ed25519.clone()),
                    Some((ml_dsa_vk_bytes(&mldsa65), self.signature.mldsa65.clone())),
                )
                .context("assembling directory genesis signature")?;
                let verified = verify_composite(
                    &composite,
                    &ed25519,
                    Some(&mldsa65),
                    &payload,
                    DIRECTORY_GENESIS_SIGNATURE_CONTEXT,
                    true,
                )
                .context("verifying directory genesis signature")?;
                ensure!(
                    verified.eddsa && verified.ml_dsa,
                    "both directory genesis signature components were not verified"
                );
                Ok::<(), anyhow::Error>(())
            })();
            match result {
                Ok(()) => return Ok(()),
                Err(error) => errors.push(error.to_string()),
            }
        }
        bail!(
            "directory genesis binding did not verify under a declared Hybrid key: {}",
            errors.join("; ")
        )
    }

    fn signable_bytes(&self) -> Vec<u8> {
        self.to_value(false).encode()
    }

    fn to_value(&self, include_signature: bool) -> DagCbor {
        let mut fields = vec![
            ("did", DagCbor::Text(self.did.clone())),
            ("genesisCid", DagCbor::Text(self.genesis_cid.clone())),
            ("pdsEndpoint", DagCbor::Text(self.pds_endpoint.clone())),
            ("type", DagCbor::Text(DIRECTORY_GENESIS_TYPE.to_owned())),
        ];
        if include_signature {
            fields.push((
                "sig",
                DagCbor::str_map([
                    ("ed25519", DagCbor::Bytes(self.signature.ed25519.clone())),
                    ("mldsa65", DagCbor::Bytes(self.signature.mldsa65.clone())),
                ]),
            ));
        }
        DagCbor::str_map(fields)
    }
}

/// Persistence shape for one directory identity.
///
/// Operation bytes remain the storage boundary so resolution cannot inherit a
/// stale "already verified" bit from a database. Implementations may persist
/// this value in any authority-owned store; callers must treat loaded records
/// as untrusted until [`PlcDirectory::resolve_identity`] returns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlcDirectoryRecord {
    operation_log: Vec<Vec<u8>>,
    genesis_binding: Vec<u8>,
}

impl PlcDirectoryRecord {
    /// Rehydrate a stored record. Validation intentionally happens at resolve.
    #[must_use]
    pub fn from_stored_parts(operation_log: Vec<Vec<u8>>, genesis_binding: Vec<u8>) -> Self {
        Self {
            operation_log,
            genesis_binding,
        }
    }

    /// Canonical signed operations, in log order.
    #[must_use]
    pub fn operation_log(&self) -> &[Vec<u8>] {
        &self.operation_log
    }

    /// Hybrid-signed genesis binding in canonical DAG-CBOR.
    #[must_use]
    pub fn genesis_binding(&self) -> &[u8] {
        &self.genesis_binding
    }
}

/// Atomic genesis storage used by the operable directory.
///
/// There is intentionally no update method in the demo cut. Rotation and
/// witness policy are deferred to #1168–1171.
pub trait PlcDirectoryStore: Send + Sync {
    /// Insert operation zero exactly once for `did`.
    fn create_genesis(&self, did: &str, record: PlcDirectoryRecord) -> Result<()>;

    /// Load one record for verification, or `None` when it is absent.
    fn get(&self, did: &str) -> Result<Option<PlcDirectoryRecord>>;
}

/// In-memory store for embedded/demo deployments.
#[derive(Default)]
pub struct InMemoryPlcDirectoryStore {
    records: RwLock<BTreeMap<String, PlcDirectoryRecord>>,
}

impl PlcDirectoryStore for InMemoryPlcDirectoryStore {
    fn create_genesis(&self, did: &str, record: PlcDirectoryRecord) -> Result<()> {
        let mut records = self.records.write();
        ensure!(
            !records.contains_key(did),
            "did:plc genesis already exists for {did}"
        );
        records.insert(did.to_owned(), record);
        Ok(())
    }

    fn get(&self, did: &str) -> Result<Option<PlcDirectoryRecord>> {
        Ok(self.records.read().get(did).cloned())
    }
}

/// Resolver for live PDS transport metadata.
///
/// Implementations are trusted service capabilities, not client callbacks.
/// Any implementation performing network I/O must authenticate its caller and
/// constrain egress before resolving the authority-stored PDS endpoint.
pub trait LivePlcDiscovery: Send + Sync {
    /// Resolve current reach for this verified directory identity.
    fn current(&self, did: &str, pds_endpoint: &str) -> Result<BrowserQuicReach>;
}

/// Current verified identity state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedPlcIdentity {
    did: String,
    rotation_keys: Vec<HybridRotationKey>,
    pds_endpoint: String,
}

impl ResolvedPlcIdentity {
    #[must_use]
    pub fn did(&self) -> &str {
        &self.did
    }

    #[must_use]
    pub fn rotation_keys(&self) -> &[HybridRotationKey] {
        &self.rotation_keys
    }

    #[must_use]
    pub fn pds_endpoint(&self) -> &str {
        &self.pds_endpoint
    }
}

/// Frontend connect-time discovery contract.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectTimeDiscovery {
    quic_url: String,
    cert_hash: String,
}

impl ConnectTimeDiscovery {
    #[must_use]
    pub fn quic_url(&self) -> &str {
        &self.quic_url
    }

    #[must_use]
    pub fn cert_hash(&self) -> &str {
        &self.cert_hash
    }
}

/// Future authorization seam for a post-genesis successor.
///
/// #1168–1171 will supply the custody, recovery-window, fork, and witness
/// implementation. The demo directory intentionally exposes no append method,
/// so implementing this trait cannot mutate a log until that policy lands.
pub trait DidOpSuccessorWitness: Send + Sync {
    fn authorize_successor(&self, current: &VerifiedDidOpLog, successor: &DidOp) -> Result<()>;
}

/// Operable directory over canonical Hybrid DID operation logs.
pub struct PlcDirectory {
    store: Arc<dyn PlcDirectoryStore>,
    live_discovery: Arc<dyn LivePlcDiscovery>,
    deployment_authority: RegistryDeploymentVerifier,
}

impl PlcDirectory {
    #[must_use]
    pub fn new(
        store: Arc<dyn PlcDirectoryStore>,
        live_discovery: Arc<dyn LivePlcDiscovery>,
        deployment_authority: RegistryDeploymentVerifier,
    ) -> Self {
        Self {
            store,
            live_discovery,
            deployment_authority,
        }
    }

    /// Register a self-service genesis signed by one declared Hybrid key.
    ///
    /// `claimed_did` is checked against the self-certifying genesis hash. A
    /// client therefore cannot choose another DID or register the
    /// unauthenticated sentinel.
    pub fn create_self_service(
        &self,
        claimed_did: &str,
        genesis: DidOp,
        binding: SignedPlcDirectoryGenesis,
    ) -> Result<ResolvedPlcIdentity> {
        self.create_verified_genesis(claimed_did, genesis, binding)
    }

    /// Mint and register an operator-manual genesis.
    ///
    /// Both signing keys must match the authenticated deployment trust root
    /// retained by [`RegistryDeploymentVerifier`]. The root must also appear
    /// in `rotation_keys`; [`DidOp::signed_genesis`] enforces that binding.
    pub fn create_operator_manual(
        &self,
        rotation_keys: Vec<HybridRotationKey>,
        pds_endpoint: &str,
        ed25519: &SigningKey,
        mldsa65: &MlDsaSigningKey,
    ) -> Result<ResolvedPlcIdentity> {
        let pinned_root = self
            .deployment_authority
            .deployment_did_op_key()
            .context("authenticated deployment root is not a valid DID operation key")?;
        let supplied = HybridRotationKey::from_signing_keys(ed25519, mldsa65);
        ensure!(
            supplied == pinned_root,
            "operator-manual genesis signer does not match the authenticated Hybrid deployment root"
        );
        let genesis = DidOp::signed_genesis(rotation_keys, ed25519, mldsa65)
            .context("deployment-authority genesis signing failed")?;
        let did = genesis.genesis_did()?;
        let binding = SignedPlcDirectoryGenesis::sign(&genesis, pds_endpoint, ed25519, mldsa65)?;
        self.create_verified_genesis(&did, genesis, binding)
    }

    fn create_verified_genesis(
        &self,
        claimed_did: &str,
        genesis: DidOp,
        binding: SignedPlcDirectoryGenesis,
    ) -> Result<ResolvedPlcIdentity> {
        ensure_registerable_did(claimed_did)?;
        let verified = verify_did_op_log(claimed_did, std::slice::from_ref(&genesis))
            .context("genesis operation failed Hybrid log verification")?;
        ensure!(
            verified.sequence == 0,
            "genesis registration unexpectedly resolved a successor"
        );
        binding
            .verify(claimed_did, &genesis)
            .context("directory genesis binding failed Hybrid verification")?;
        self.store.create_genesis(
            claimed_did,
            PlcDirectoryRecord {
                operation_log: vec![genesis.to_dag_cbor()],
                genesis_binding: binding.to_dag_cbor(),
            },
        )?;
        self.resolve_identity(claimed_did)
    }

    /// Resolve current keys and the PDS pointer after full log verification.
    pub fn resolve_identity(&self, did: &str) -> Result<ResolvedPlcIdentity> {
        ensure_registerable_did(did)?;
        let record = self
            .store
            .get(did)?
            .ok_or_else(|| anyhow::anyhow!("did:plc identity not found: {did}"))?;
        let log = record
            .operation_log()
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                DidOp::from_dag_cbor(bytes)
                    .with_context(|| format!("decoding DID operation {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let verified =
            verify_did_op_log(did, &log).context("stored DID operation log failed verification")?;
        let genesis = log
            .first()
            .ok_or_else(|| anyhow::anyhow!("stored DID operation log is empty"))?;
        let binding = SignedPlcDirectoryGenesis::from_dag_cbor(record.genesis_binding())
            .context("stored directory genesis binding is invalid")?;
        binding
            .verify(did, genesis)
            .context("stored directory genesis binding failed Hybrid verification")?;
        Ok(ResolvedPlcIdentity {
            did: verified.did,
            rotation_keys: verified.rotation_keys,
            pds_endpoint: binding.pds_endpoint,
        })
    }

    /// Resolve live `{quicUrl, certHash}` for one connection attempt.
    ///
    /// No live value is read from or written to [`PlcDirectoryStore`].
    pub fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery> {
        let identity = self.resolve_identity(did)?;
        let reach = self
            .live_discovery
            .current(identity.did(), identity.pds_endpoint())
            .context("live PLC discovery failed")?;
        validate_https_endpoint(&reach.quic_url, "QUIC URL")?;
        validate_cert_hash(&reach.cert_hash)?;
        Ok(ConnectTimeDiscovery {
            quic_url: reach.quic_url,
            cert_hash: reach.cert_hash,
        })
    }
}

fn ensure_registerable_did(did: &str) -> Result<()> {
    ensure!(
        did != UNAUTHENTICATED_DID_SENTINEL,
        "{UNAUTHENTICATED_DID_SENTINEL} is reserved for credential absence and is non-registerable"
    );
    ensure!(
        hyprstream_rpc::did_plc::is_did_plc(did),
        "operable PLC directory accepts only did:plc identities"
    );
    Ok(())
}

fn validate_https_endpoint(endpoint: &str, what: &str) -> Result<()> {
    let url = reqwest::Url::parse(endpoint).with_context(|| format!("{what} is not a URL"))?;
    ensure!(url.scheme() == "https", "{what} must use https");
    ensure!(url.host().is_some(), "{what} must include a host");
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "{what} must not contain credentials"
    );
    ensure!(
        url.fragment().is_none(),
        "{what} must not contain a fragment"
    );
    Ok(())
}

fn validate_cert_hash(cert_hash: &str) -> Result<()> {
    let bytes = STANDARD
        .decode(cert_hash)
        .context("certificate hash is not canonical base64")?;
    ensure!(
        bytes.len() == 32,
        "certificate hash must contain one SHA-256 digest"
    );
    ensure!(
        STANDARD.encode(&bytes) == cert_hash,
        "certificate hash is not canonical base64"
    );
    Ok(())
}

fn required<'a>(value: &'a DagCbor, field: &str) -> Result<&'a DagCbor> {
    value
        .get(field)
        .ok_or_else(|| anyhow::anyhow!("missing required field {field:?}"))
}

fn require_exact_fields(value: &DagCbor, expected: &[&str], what: &str) -> Result<()> {
    let map = value
        .as_map()
        .with_context(|| format!("{what} must be a map"))?;
    ensure!(
        map.len() == expected.len(),
        "{what} must contain exactly fields {expected:?}"
    );
    for field in expected {
        ensure!(
            value.get(field).is_some(),
            "{what} is missing required field {field:?}"
        );
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicU8, Ordering};

    use hyprstream_crypto::pq::{ml_dsa_sk_from_seed, MlDsaSigningKey};

    use super::*;

    struct Keys {
        ed: SigningKey,
        pq: MlDsaSigningKey,
        public: HybridRotationKey,
    }

    fn keys(seed: u8) -> Keys {
        let ed = SigningKey::from_bytes(&[seed; 32]);
        let pq = ml_dsa_sk_from_seed(&[seed; 32]);
        let public = HybridRotationKey::from_signing_keys(&ed, &pq);
        Keys { ed, pq, public }
    }

    struct ChangingDiscovery {
        generation: AtomicU8,
    }

    impl LivePlcDiscovery for ChangingDiscovery {
        fn current(&self, _did: &str, pds_endpoint: &str) -> Result<BrowserQuicReach> {
            let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(BrowserQuicReach {
                quic_url: format!("{pds_endpoint}:4433/wt"),
                cert_hash: STANDARD.encode([generation; 32]),
            })
        }
    }

    fn fixture() -> (Arc<InMemoryPlcDirectoryStore>, PlcDirectory, Keys) {
        let authority = keys(41);
        let verifier =
            RegistryDeploymentVerifier::for_test_deployment_root(&authority.ed, &authority.pq)
                .unwrap();
        let store = Arc::new(InMemoryPlcDirectoryStore::default());
        let directory = PlcDirectory::new(
            store.clone(),
            Arc::new(ChangingDiscovery {
                generation: AtomicU8::new(0),
            }),
            verifier,
        );
        (store, directory, authority)
    }

    fn self_service_artifacts(
        user: &Keys,
        pds_endpoint: &str,
    ) -> (String, DidOp, SignedPlcDirectoryGenesis) {
        let genesis = DidOp::signed_genesis(vec![user.public.clone()], &user.ed, &user.pq).unwrap();
        let did = genesis.genesis_did().unwrap();
        let binding =
            SignedPlcDirectoryGenesis::sign(&genesis, pds_endpoint, &user.ed, &user.pq).unwrap();
        (did, genesis, binding)
    }

    #[test]
    fn self_service_genesis_resolves_verified_keys_and_pds() {
        let (_store, directory, _authority) = fixture();
        let user = keys(7);
        let (did, genesis, binding) = self_service_artifacts(&user, "https://pds.example");

        let resolved = directory
            .create_self_service(&did, genesis, binding)
            .unwrap();

        assert_eq!(resolved.did(), did);
        assert_eq!(resolved.rotation_keys(), &[user.public]);
        assert_eq!(resolved.pds_endpoint(), "https://pds.example");
    }

    #[test]
    fn did_unknown_genesis_fails_closed_without_storage() {
        let (store, directory, _authority) = fixture();
        let user = keys(8);
        let (_did, genesis, binding) = self_service_artifacts(&user, "https://pds.example");

        let error = directory
            .create_self_service(UNAUTHENTICATED_DID_SENTINEL, genesis, binding)
            .unwrap_err();

        assert!(error.to_string().contains("non-registerable"));
        assert!(store.get(UNAUTHENTICATED_DID_SENTINEL).unwrap().is_none());
    }

    #[test]
    fn operator_manual_genesis_requires_authenticated_hybrid_root() {
        let (_store, directory, authority) = fixture();
        let resolved = directory
            .create_operator_manual(
                vec![authority.public],
                "https://operator-pds.example",
                &authority.ed,
                &authority.pq,
            )
            .unwrap();
        assert!(resolved.did().starts_with("did:plc:"));

        let wrong = keys(42);
        let error = directory
            .create_operator_manual(
                vec![wrong.public],
                "https://operator-pds.example",
                &wrong.ed,
                &wrong.pq,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("authenticated Hybrid deployment root"));
    }

    #[test]
    fn classical_only_directory_binding_fails_closed() {
        let (_store, directory, _authority) = fixture();
        let user = keys(43);
        let (did, genesis, mut binding) = self_service_artifacts(&user, "https://pds.example");
        binding.signature.mldsa65.clear();

        let error = directory
            .create_self_service(&did, genesis, binding)
            .unwrap_err();
        assert!(format!("{error:#}").contains("ML-DSA-65 signature must be"));
    }

    #[test]
    fn duplicate_genesis_is_rejected() {
        let (_store, directory, _authority) = fixture();
        let user = keys(9);
        let (did, genesis, binding) = self_service_artifacts(&user, "https://pds.example");
        directory
            .create_self_service(&did, genesis.clone(), binding.clone())
            .unwrap();

        let error = directory
            .create_self_service(&did, genesis, binding)
            .unwrap_err();
        assert!(error.to_string().contains("already exists"));
    }

    #[test]
    fn stored_log_is_reverified_on_every_identity_resolution() {
        let (store, directory, _authority) = fixture();
        let user = keys(10);
        let (did, genesis, binding) = self_service_artifacts(&user, "https://pds.example");
        directory
            .create_self_service(&did, genesis, binding)
            .unwrap();

        let mut records = store.records.write();
        let record = records.get_mut(&did).unwrap();
        let last = record.operation_log[0].len() - 1;
        record.operation_log[0][last] ^= 1;
        drop(records);

        assert!(directory.resolve_identity(&did).is_err());
    }

    #[test]
    fn stored_pds_binding_is_reverified_on_every_identity_resolution() {
        let (store, directory, _authority) = fixture();
        let user = keys(12);
        let (did, genesis, binding) = self_service_artifacts(&user, "https://pds.example");
        directory
            .create_self_service(&did, genesis, binding)
            .unwrap();

        let mut records = store.records.write();
        let record = records.get_mut(&did).unwrap();
        let last = record.genesis_binding.len() - 1;
        record.genesis_binding[last] ^= 1;
        drop(records);

        assert!(directory.resolve_identity(&did).is_err());
    }

    #[test]
    fn connect_time_reach_is_live_and_never_stored_on_directory_row() {
        let (store, directory, _authority) = fixture();
        let user = keys(11);
        let (did, genesis, binding) = self_service_artifacts(&user, "https://pds.example");
        directory
            .create_self_service(&did, genesis, binding)
            .unwrap();

        let first = directory.resolve(&did).unwrap();
        let second = directory.resolve(&did).unwrap();
        assert_ne!(first.cert_hash(), second.cert_hash());
        assert_eq!(first.quic_url(), second.quic_url());

        let stored = store.get(&did).unwrap().unwrap();
        assert_eq!(stored.operation_log().len(), 1);
        let binding = SignedPlcDirectoryGenesis::from_dag_cbor(stored.genesis_binding()).unwrap();
        assert_eq!(binding.pds_endpoint(), "https://pds.example");
        assert!(!stored
            .genesis_binding()
            .windows(b"certHash".len())
            .any(|window| window == b"certHash"));
    }
}
