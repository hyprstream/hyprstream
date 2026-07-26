//! Authority-bound hosted PDS account minting.
//!
//! The client request carries only a requested handle. The deployment
//! authority allocates both the permanent host-form DID and the local tenant;
//! neither value can be asserted in the registration payload. A successful
//! mint publishes one coherent bundle containing the account record, sealed
//! DID document, genesis operation-log entry, empty MST, signed repo commit,
//! and account-specific `#atproto` signing key.
//!
//! This module performs no federation intake or resolver network I/O. A later
//! registration/RPC adapter that invokes did:web intake must authenticate and
//! rate-limit the caller and constrain the resolvable host set before the
//! first resolver call. Unauthenticated intake is forbidden.

use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use hyprstream_pds::{
    AllocatedAccountName, DidOpSignature, GenesisRepoHead, GenesisRotationKeys, HostedAccountMint,
    HostedRepoGenesis, SealedHostedAccount, UnsignedGenesisDidOp, DID_DOCUMENT_PATH,
    DID_OPERATION_LOG_PATH,
};
use hyprstream_rpc::service_entry::BrowserQuicReach;

use crate::validate_tenant_component;

/// Client-controlled portion of hosted-account registration.
///
/// The registration transport must route any supplied `tenant` member through
/// [`Self::from_client_fields`], which rejects it before allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedAccountRegistrationRequest {
    handle: String,
}

impl HostedAccountRegistrationRequest {
    fn new(handle: impl Into<String>) -> Self {
        Self {
            handle: handle.into(),
        }
    }

    /// Construct from transport fields while rejecting client tenant input.
    pub fn from_client_fields(
        handle: impl Into<String>,
        client_asserted_tenant: Option<&str>,
    ) -> Result<Self> {
        ensure!(
            client_asserted_tenant.is_none(),
            "client-asserted tenant is forbidden for hosted-account registration"
        );
        Ok(Self::new(handle))
    }

    #[must_use]
    pub fn handle(&self) -> &str {
        &self.handle
    }
}

/// Exact registration result consumed by the browser-facing API.
///
/// The wire adapter maps the final three Rust fields to `pdsEndpoint`,
/// `quicUrl`, and `certHash`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedAccountRegistrationResult {
    pub handle: String,
    pub did: String,
    pub pds_endpoint: String,
    pub quic_url: String,
    pub cert_hash: String,
}

/// One authority-owned allocation of a local tenant and permanent DID.
///
/// The registration client never constructs or supplies this value. It is
/// returned by the [`HostedAccountAuthority`] installed by the server.
#[derive(Clone)]
pub struct AuthorityHostedAccountBinding {
    tenant: String,
    name: AllocatedAccountName,
    rotations: GenesisRotationKeys,
}

impl AuthorityHostedAccountBinding {
    /// Construct an allocation in an authority implementation.
    pub fn new(
        tenant: impl Into<String>,
        name: AllocatedAccountName,
        rotations: GenesisRotationKeys,
    ) -> Result<Self> {
        let tenant = tenant.into();
        validate_tenant_component(&tenant)?;
        Ok(Self {
            tenant,
            name,
            rotations,
        })
    }

    #[must_use]
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    #[must_use]
    pub fn name(&self) -> &AllocatedAccountName {
        &self.name
    }
}

/// Deployment authority that allocates local tenant bindings.
pub trait HostedAccountAuthority: Send + Sync {
    fn allocate(&self, requested_handle: &str) -> Result<AuthorityHostedAccountBinding>;
}

/// Request-scoped signer for the user-held priority-zero Hybrid rotation key.
pub trait HostedAccountGenesisSigner {
    fn sign(&self, unsigned: &UnsignedGenesisDidOp) -> Result<DidOpSignature>;
}

/// One live discovery read used atomically for both mint input and response.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveHostedPdsDiscovery {
    pub pds_endpoint: String,
    pub browser_quic_reach: BrowserQuicReach,
}

/// Resolver for the currently served PDS and QUIC discovery metadata.
///
/// Implementations must derive `browser_quic_reach` from the current
/// `QuicTransport` service entry with
/// [`hyprstream_rpc::service_entry::decode_browser_quic_reach`]. They must not
/// accept a manually supplied transport URL or certificate pin.
pub trait HostedPdsDiscovery: Send + Sync {
    fn current(&self) -> Result<LiveHostedPdsDiscovery>;
}

/// Borrowed, already-verified publication generation.
///
/// The publisher must durably commit the whole generation before returning.
/// The public account record is the publication marker; storage
/// implementations should therefore make it visible last.
pub struct HostedPdsAccountPublication<'a> {
    tenant: &'a str,
    account: &'a SealedHostedAccount,
    repo: &'a HostedRepoGenesis,
}

impl HostedPdsAccountPublication<'_> {
    #[must_use]
    pub fn tenant(&self) -> &str {
        self.tenant
    }

    #[must_use]
    pub fn label(&self) -> &str {
        self.account.record().name().label()
    }

    #[must_use]
    pub fn account(&self) -> &SealedHostedAccount {
        self.account
    }

    #[must_use]
    pub fn repo(&self) -> &HostedRepoGenesis {
        self.repo
    }

    /// Exact bytes served at [`DID_DOCUMENT_PATH`].
    #[must_use]
    pub fn did_document(&self) -> (&'static str, &[u8]) {
        (DID_DOCUMENT_PATH, self.account.did_document().as_bytes())
    }

    /// Genesis entry establishing the operation log served at
    /// [`DID_OPERATION_LOG_PATH`].
    #[must_use]
    pub fn did_operation_log_genesis(&self) -> (&'static str, &[u8]) {
        (DID_OPERATION_LOG_PATH, self.account.genesis_bytes())
    }
}

/// Durable publication boundary for hosted PDS account generations.
pub trait HostedPdsAccountPublisher: Send + Sync {
    fn publish(&self, publication: HostedPdsAccountPublication<'_>) -> Result<()>;
}

/// Hosted-account mint service used by the later registration API.
pub struct HostedPdsAccountMinter {
    authority: Arc<dyn HostedAccountAuthority>,
    discovery: Arc<dyn HostedPdsDiscovery>,
    publisher: Arc<dyn HostedPdsAccountPublisher>,
}

impl HostedPdsAccountMinter {
    #[must_use]
    pub fn new(
        authority: Arc<dyn HostedAccountAuthority>,
        discovery: Arc<dyn HostedPdsDiscovery>,
        publisher: Arc<dyn HostedPdsAccountPublisher>,
    ) -> Self {
        Self {
            authority,
            discovery,
            publisher,
        }
    }

    /// Mint, sign, and durably publish one local hosted PDS account.
    pub fn mint(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
    ) -> Result<HostedAccountRegistrationResult> {
        ensure!(
            !request.handle.is_empty(),
            "hosted account handle must not be empty"
        );

        // Read discovery before allocating a permanent name. A deployment with
        // no live PDS/QUIC reach must not consume a never-reusable label.
        let live = self
            .discovery
            .current()
            .context("live hosted-PDS discovery is unavailable")?;
        ensure!(
            !live.browser_quic_reach.quic_url.is_empty()
                && !live.browser_quic_reach.cert_hash.is_empty(),
            "live hosted-PDS QUIC discovery is incomplete"
        );

        let binding = self
            .authority
            .allocate(&request.handle)
            .context("hosted-account authority allocation failed")?;
        ensure!(
            binding.name.label() == request.handle,
            "authority allocation does not match the requested handle"
        );

        let mint = HostedAccountMint::begin(binding.name.clone(), binding.rotations.clone())
            .context("hosted-account mint initialization failed")?;
        let document = mint
            .seal_did_document(&live.pds_endpoint)
            .context("hosted DID document sealing failed")?;
        let (pending, repo) = mint
            .prepare_pds_genesis(document, hyprstream_pds::tid::Tid::now())
            .context("hosted PDS repo genesis failed")?;
        let signature = signer
            .sign(pending.unsigned_genesis())
            .context("hosted DID genesis signing failed")?;
        let account = pending
            .seal(signature)
            .context("hosted DID genesis signature is invalid")?;

        ensure!(
            account.genesis().unsigned().head_at_op()
                == GenesisRepoHead::Existing(repo.commit_cid()),
            "hosted DID genesis is not bound to the signed PDS repo"
        );
        repo.verify(&account.record().atproto_verifying_key()?)
            .context("hosted PDS repo verification failed")?;

        self.publisher
            .publish(HostedPdsAccountPublication {
                tenant: binding.tenant(),
                account: &account,
                repo: &repo,
            })
            .context("hosted PDS account publication failed")?;

        Ok(registration_result(account, live.browser_quic_reach))
    }
}

fn registration_result(
    account: SealedHostedAccount,
    browser_reach: BrowserQuicReach,
) -> HostedAccountRegistrationResult {
    let document = account.did_document();
    HostedAccountRegistrationResult {
        handle: document.handle().to_owned(),
        did: document.did().to_owned(),
        pds_endpoint: document.pds_endpoint().to_owned(),
        quic_url: browser_reach.quic_url,
        cert_hash: browser_reach.cert_hash,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_pds::{
        sign_genesis, AccountRecord, HostKeyEnrollment, HybridRotationKey, RecoveryKeyEnrollment,
        UserRotationKey,
    };
    use hyprstream_rpc::auth::mac::{MacDecision, SecurityContext};
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};
    use rand::rngs::OsRng;

    use super::*;
    use crate::{
        AccountRecordReadAuthorizer, AccountRecordStore, PDS_ACCOUNTS_DIRECTORY,
        PDS_ACCOUNT_RECORD_FILE,
    };

    struct TestAuthority {
        calls: AtomicUsize,
        rotations: GenesisRotationKeys,
    }

    impl HostedAccountAuthority for TestAuthority {
        fn allocate(&self, requested_handle: &str) -> Result<AuthorityHostedAccountBinding> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            AuthorityHostedAccountBinding::new(
                "tenant-authority",
                AllocatedAccountName::new(
                    requested_handle,
                    format!("did:web:{requested_handle}.accounts.example"),
                )?,
                self.rotations.clone(),
            )
        }
    }

    struct TestSigner {
        ed: SigningKey,
        pq: MlDsaSigningKey,
    }

    impl HostedAccountGenesisSigner for TestSigner {
        fn sign(&self, unsigned: &UnsignedGenesisDidOp) -> Result<DidOpSignature> {
            sign_genesis(unsigned, &self.ed, &self.pq)
        }
    }

    #[derive(Clone)]
    struct StaticDiscovery;

    impl HostedPdsDiscovery for StaticDiscovery {
        fn current(&self) -> Result<LiveHostedPdsDiscovery> {
            Ok(LiveHostedPdsDiscovery {
                pds_endpoint: "https://pds.example".to_owned(),
                browser_quic_reach: BrowserQuicReach {
                    quic_url: "https://pds.example:4433/wt".to_owned(),
                    cert_hash: "paWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaU=".to_owned(),
                },
            })
        }
    }

    #[derive(Clone)]
    struct Published {
        tenant: String,
        label: String,
        account_record: Vec<u8>,
        did_document_path: String,
        did_document: Vec<u8>,
        operation_log_path: String,
        operation_log_genesis: Vec<u8>,
        mst_root: hyprstream_pds::Cid,
        mst_blocks: Vec<(hyprstream_pds::Cid, Vec<u8>)>,
        commit: Vec<u8>,
    }

    #[derive(Default)]
    struct CapturingPublisher {
        published: OnceLock<Published>,
    }

    impl HostedPdsAccountPublisher for CapturingPublisher {
        fn publish(&self, publication: HostedPdsAccountPublication<'_>) -> Result<()> {
            let (did_document_path, did_document) = publication.did_document();
            let (operation_log_path, operation_log_genesis) =
                publication.did_operation_log_genesis();
            let published = Published {
                tenant: publication.tenant().to_owned(),
                label: publication.label().to_owned(),
                account_record: publication.account().record_bytes().to_vec(),
                did_document_path: did_document_path.to_owned(),
                did_document: did_document.to_vec(),
                operation_log_path: operation_log_path.to_owned(),
                operation_log_genesis: operation_log_genesis.to_vec(),
                mst_root: publication.repo().mst_root(),
                mst_blocks: publication.repo().mst_blocks().to_vec(),
                commit: publication.repo().commit_bytes().to_vec(),
            };
            self.published
                .set(published)
                .map_err(|_| anyhow::anyhow!("account was published more than once"))?;
            Ok(())
        }
    }

    fn fixture() -> (
        Arc<TestAuthority>,
        Arc<CapturingPublisher>,
        TestSigner,
        HostedPdsAccountMinter,
    ) {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let rotation =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&pq_vk)).unwrap();
        let authority = Arc::new(TestAuthority {
            calls: AtomicUsize::new(0),
            rotations: GenesisRotationKeys::new(
                UserRotationKey::new(rotation),
                RecoveryKeyEnrollment::Declined,
                HostKeyEnrollment::Absent,
            )
            .unwrap(),
        });
        let publisher = Arc::new(CapturingPublisher::default());
        let minter = HostedPdsAccountMinter::new(
            authority.clone(),
            Arc::new(StaticDiscovery),
            publisher.clone(),
        );
        (authority, publisher, TestSigner { ed, pq }, minter)
    }

    struct PermitReads;

    impl AccountRecordReadAuthorizer for PermitReads {
        fn check_read(
            &self,
            _subject: &Subject,
            _verified_tenant: Option<&str>,
            _security_context: Option<&SecurityContext>,
            _object_id: &str,
        ) -> MacDecision {
            MacDecision::Permit
        }
    }

    #[test]
    fn mint_publishes_did_log_and_signed_repo_and_returns_live_discovery() {
        let (_authority, publisher, signer, minter) = fixture();
        let result = minter
            .mint(&HostedAccountRegistrationRequest::new("alice"), &signer)
            .unwrap();
        let published = publisher
            .published
            .get()
            .cloned()
            .expect("mint must publish");

        assert_eq!(published.did_document_path, DID_DOCUMENT_PATH);
        assert!(!published.did_document.is_empty());
        assert_eq!(published.operation_log_path, DID_OPERATION_LOG_PATH);
        assert!(!published.operation_log_genesis.is_empty());
        let document =
            hyprstream_pds::SealedHostedDidDocument::from_canonical_json(&published.did_document)
                .unwrap();
        let genesis =
            hyprstream_pds::GenesisDidOp::from_dag_cbor(&published.operation_log_genesis).unwrap();
        assert!(published
            .mst_blocks
            .iter()
            .any(|(cid, bytes)| *cid == published.mst_root
                && hyprstream_pds::Cid::from_dag_cbor(bytes) == *cid));
        let commit = hyprstream_pds::commit::Commit::from_dag_cbor(&published.commit).unwrap();
        let record = AccountRecord::from_dag_cbor(&published.account_record).unwrap();
        commit
            .verify(&record.atproto_verifying_key().unwrap())
            .unwrap();
        assert_eq!(document.did(), record.name().did());
        assert_eq!(document.cid(), record.doc_cid());
        assert_eq!(genesis.cid().unwrap(), record.genesis_op());
        assert_eq!(
            genesis.unsigned().head_at_op(),
            GenesisRepoHead::Existing(commit.cid())
        );
        assert_eq!(commit.data, published.mst_root);

        assert_eq!(result.handle, "at://alice.accounts.example");
        assert_eq!(result.did, "did:web:alice.accounts.example");
        assert_eq!(result.pds_endpoint, "https://pds.example");
        assert_eq!(result.quic_url, "https://pds.example:4433/wt");
        assert_eq!(
            result.cert_hash,
            "paWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaU="
        );
    }

    #[tokio::test]
    async fn minted_did_resolves_to_the_authority_owned_tenant() {
        let (_authority, publisher, signer, minter) = fixture();
        let result = minter
            .mint(&HostedAccountRegistrationRequest::new("alice"), &signer)
            .unwrap();
        let published = publisher
            .published
            .get()
            .cloned()
            .expect("mint must publish");
        let root = SyntheticNode::dir().with_child(
            published.tenant.clone(),
            SyntheticNode::dir().with_child(
                PDS_ACCOUNTS_DIRECTORY,
                SyntheticNode::dir().with_child(
                    published.label,
                    SyntheticNode::dir().with_child(
                        PDS_ACCOUNT_RECORD_FILE,
                        SyntheticNode::file(published.account_record),
                    ),
                ),
            ),
        );
        let store =
            AccountRecordStore::new(Arc::new(SyntheticMount::new(root)), Arc::new(PermitReads));

        assert_eq!(
            store
                .resolve_tenant_for_hosted_did(
                    &Subject::new(crate::OAUTH_ACCOUNT_RESOLVER_SUBJECT),
                    &result.did,
                )
                .await
                .unwrap(),
            Some("tenant-authority".to_owned())
        );
    }

    #[test]
    fn client_asserted_tenant_is_rejected_before_authority_or_publication() {
        let (authority, publisher, _signer, _minter) = fixture();
        let error =
            HostedAccountRegistrationRequest::from_client_fields("alice", Some("tenant-client"))
                .unwrap_err();

        assert!(
            error.to_string().contains("client-asserted tenant"),
            "{error}"
        );
        assert_eq!(authority.calls.load(Ordering::SeqCst), 0);
        assert!(publisher.published.get().is_none());
    }
}
