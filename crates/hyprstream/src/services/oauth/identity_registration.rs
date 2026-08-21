//! Identity registration, federation-intake, and public resolution HTTP face.
//!
//! Registration has two distinct routes with the same handle-only body:
//! browser-session self service and bearer-authenticated operator manual
//! registration. The route, never a client field, selects the authority path.
//!
//! Federation intake is deliberately exposed only beside the authenticated
//! self-service route. Its `did:web` arm checks an exact, deployment-owned
//! HTTPS-origin allowlist before [`FederationIntake::intake`] can perform any
//! resolver network I/O.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context as _, Result};
use async_trait::async_trait;
use axum::extract::{Extension, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use hyprstream_discovery::{
    ConnectTimeDiscovery, InMemoryPlcDirectoryStore, LivePlcDiscovery, PlcDirectory,
};
use hyprstream_pds::{
    sign_genesis, AccountRecord, AllocatedAccountName, DidOpSignature, DirectoryHostedAccountStore,
    GenesisDidOp, GenesisRepoHead, GenesisRotationKeys, HostKeyEnrollment, HybridRotationKey,
    RecoveryKeyEnrollment, SealedHostedDidDocument, UnsignedGenesisDidOp, UserRotationKey,
};
use hyprstream_pds_service::federation_intake::{
    FederatedDidDocumentResolver, FederatedDidResolver, FederationIntake,
    InMemoryIdentityInventory, InventoryEntry,
};
use hyprstream_pds_service::hosted_account_mint::{
    AuthorityHostedAccountBinding, HostedAccountAuthority, HostedAccountGenesisSigner,
    HostedAccountRegistrationRequest, HostedAccountRegistrationResult, HostedPdsAccountMinter,
    HostedPdsAccountPublication, HostedPdsAccountPublisher, HostedPdsDiscovery,
    LiveHostedPdsDiscovery,
};
use hyprstream_rpc::identity::UNAUTHENTICATED_DID_SENTINEL;
use serde::{Deserialize, Serialize};
use tracing::warn;
use url::Url;
use zeroize::Zeroize as _;

use super::auth::AuthenticatedUser;
use super::session;
use super::state::OAuthState;
use crate::account::{AccountZone, AccountZoneConfig};
use crate::config::{OAuthConfig, QuicConfig};
use crate::server::middleware::RateLimiter;

const REGISTRATION_RATE_LIMIT_REQUESTS: u32 = 10;
const REGISTRATION_RATE_LIMIT_WINDOW_SECS: i64 = 60;
const PUBLIC_RESOLVE_RATE_LIMIT_BUCKET: &str = "identity-resolve-unauthenticated-floor";
const COLD_SIGNUP_GLOBAL_RATE_LIMIT_BUCKET: &str = "oauth-authorize-signup-global-floor";
const MAX_AUTHORITY_ARTIFACT_BYTES: u64 = 64 * 1024;
const ACCOUNT_GENESIS_ED25519_PURPOSE: &str = "hyprstream-hosted-account-genesis-ed25519-v1";
const ACCOUNT_GENESIS_MLDSA65_PURPOSE: &str = "hyprstream-hosted-account-genesis-mldsa65-v1";
const SIGNUP_TRANSACTION_FILE: &str = "signup-transaction.json";

#[derive(Debug, Deserialize, Serialize)]
struct HostedAccountReservation {
    version: u8,
    did: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    transaction_id: Option<String>,
}

/// Assemble the production registration face installed by the OAuth factory.
///
/// The account zone may be unconfigured: the API remains installed so
/// federation intake still works, while minting fails closed at the authority
/// boundary with the account-zone configuration error.
pub(crate) fn production_identity_registration_api(
    oauth: &OAuthConfig,
    account: &AccountZoneConfig,
    quic: &QuicConfig,
    oauth_signing_key: ed25519_dalek::SigningKey,
    pds_root: PathBuf,
) -> Result<Arc<IdentityRegistrationApi>> {
    let ttl = Duration::from_secs(oauth.atproto_did_cache_ttl_secs);
    let plc_config = hyprstream_rpc::did_plc::PlcResolverConfig::new(
        Url::parse(&oauth.atproto_plc_directory_url)
            .context("identity intake PLC directory URL is invalid")?,
        ttl,
    )
    .context("identity intake PLC resolver configuration is invalid")?;
    let plc = Arc::new(
        hyprstream_rpc::did_plc::DidPlcResolver::new(plc_config)
            .context("identity intake PLC resolver construction failed")?,
    );
    let web = Arc::new(hyprstream_rpc::did_web::DidWebResolver::new(
        hyprstream_rpc::did_web::HttpDidDocFetcher::new(ttl)
            .context("identity intake did:web HTTPS client construction failed")?,
    ));
    let resolver: Arc<dyn FederatedDidDocumentResolver> =
        Arc::new(FederatedDidResolver::new(plc, web));
    let live_discovery = Arc::new(ConfiguredHostedPdsDiscovery {
        pds_endpoint: oauth.issuer_url(),
        quic: quic.clone(),
    });
    let directory = PlcDirectory::new(
        Arc::new(InMemoryPlcDirectoryStore::default()),
        live_discovery.clone(),
        hyprstream_discovery::deployment_registry_verifier()
            .context("identity PLC directory authority is not installed")?,
    );
    let identity_resolver = Arc::new(AuthorityConnectTimeResolver {
        plc: directory,
        hosted: account
            .resolve_zone()
            .ok()
            .map(|zone| HostedConnectTimeResolver {
                pds_root: pds_root.clone(),
                zone,
                live_discovery,
            }),
    });
    compose_identity_registration_api(
        oauth,
        account,
        quic,
        oauth_signing_key,
        pds_root,
        resolver,
        identity_resolver,
    )
}

pub(super) fn compose_identity_registration_api(
    oauth: &OAuthConfig,
    account: &AccountZoneConfig,
    quic: &QuicConfig,
    oauth_signing_key: ed25519_dalek::SigningKey,
    pds_root: PathBuf,
    resolver: Arc<dyn FederatedDidDocumentResolver>,
    identity_resolver: Arc<dyn IdentityConnectTimeResolver>,
) -> Result<Arc<IdentityRegistrationApi>> {
    let authority = Arc::new(ProductionHostedAccountAuthority {
        account: account.clone(),
        pds_root: pds_root.clone(),
        signing_root: oauth_signing_key.clone(),
    });
    let discovery = Arc::new(ConfiguredHostedPdsDiscovery {
        pds_endpoint: oauth.issuer_url(),
        quic: quic.clone(),
    });
    let publisher = Arc::new(ProductionHostedAccountPublisher { pds_root });
    let minter = Arc::new(HostedPdsAccountMinter::new(authority, discovery, publisher));
    let signer = Arc::new(ProductionRegistrationGenesisSigner {
        signing_root: oauth_signing_key,
    });
    let intake = Arc::new(FederationIntake::new(
        resolver,
        Arc::new(InMemoryIdentityInventory::default()),
        vec![oauth.issuer_url()],
    ));
    let allowlist = DidWebOriginAllowlist::new(&oauth.identity_registration_did_web_origins)
        .context("identity registration did:web origin allowlist is invalid")?;
    Ok(Arc::new(IdentityRegistrationApi::new(
        minter,
        signer,
        intake,
        identity_resolver,
        allowlist,
        Arc::new(RateLimiter::new(
            REGISTRATION_RATE_LIMIT_REQUESTS,
            REGISTRATION_RATE_LIMIT_WINDOW_SECS,
        )),
    )))
}

struct ProductionHostedAccountAuthority {
    account: AccountZoneConfig,
    pds_root: PathBuf,
    signing_root: ed25519_dalek::SigningKey,
}

impl ProductionHostedAccountAuthority {
    fn allocate_inner(
        &self,
        requested_handle: &str,
        transaction_id: Option<&str>,
    ) -> Result<AuthorityHostedAccountBinding> {
        let zone = self
            .account
            .resolve_zone()
            .context("hosted-account minting is not configured")?;
        let host = zone.host_for_label(requested_handle)?;
        let name = AllocatedAccountName::new(requested_handle, format!("did:web:{host}"))?;
        let tenant = zone.apex().to_owned();

        // Allocation is a permanent, authority-owned reservation. OAuth
        // signup binds it to the already-committed AS credential transaction,
        // allowing only that exact verified pair to resume.
        let reservation_dir = self.pds_root.join(&tenant).join(".allocated");
        create_private_dir_all(&reservation_dir)?;
        let reservation = reservation_dir.join(requested_handle);
        let reservation_value = HostedAccountReservation {
            version: 1,
            did: name.did().to_owned(),
            transaction_id: transaction_id.map(str::to_owned),
        };
        let mut staged_reservation = tempfile::Builder::new()
            .prefix(".allocation-")
            .tempfile_in(&reservation_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(
                staged_reservation.path(),
                std::fs::Permissions::from_mode(0o600),
            )?;
        }
        serde_json::to_writer(staged_reservation.as_file_mut(), &reservation_value)?;
        staged_reservation.as_file_mut().write_all(b"\n")?;
        staged_reservation.as_file_mut().sync_all()?;
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        match nix::fcntl::renameat2(
            None,
            staged_reservation.path(),
            None,
            &reservation,
            nix::fcntl::RenameFlags::RENAME_NOREPLACE,
        ) {
            Ok(()) => {
                sync_directory(&reservation_dir)?;
                sync_directory(
                    reservation_dir
                        .parent()
                        .context("hosted-account reservation directory has no tenant parent")?,
                )?;
            }
            Err(nix::errno::Errno::EEXIST) => {
                let existing: HostedAccountReservation = serde_json::from_slice(
                    &read_authority_artifact(&reservation)
                        .context("reading hosted-account reservation")?,
                )
                .context("verifying hosted-account reservation")?;
                ensure!(
                    existing.version == 1
                        && existing.did == name.did()
                        && transaction_id.is_some()
                        && existing.transaction_id.as_deref() == transaction_id,
                    std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        format!("hosted-account label {requested_handle:?} is unavailable"),
                    )
                );
                sync_directory(&reservation_dir)?;
                sync_directory(
                    reservation_dir
                        .parent()
                        .context("hosted-account reservation directory has no tenant parent")?,
                )?;
            }
            Err(error) => return Err(error.into()),
        }
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        anyhow::bail!("atomic hosted-account allocation requires Linux renameat2");

        AuthorityHostedAccountBinding::new(
            tenant,
            name,
            account_genesis_rotation_keys(&self.signing_root, requested_handle)?,
        )
    }
}

impl HostedAccountAuthority for ProductionHostedAccountAuthority {
    fn allocate(&self, requested_handle: &str) -> Result<AuthorityHostedAccountBinding> {
        self.allocate_inner(requested_handle, None)
    }

    fn allocate_for_transaction(
        &self,
        requested_handle: &str,
        transaction_id: &str,
    ) -> Result<AuthorityHostedAccountBinding> {
        ensure!(
            !transaction_id.is_empty(),
            "hosted-account reservation transaction id is empty"
        );
        self.allocate_inner(requested_handle, Some(transaction_id))
    }
}

struct ProductionRegistrationGenesisSigner {
    signing_root: ed25519_dalek::SigningKey,
}

impl RegistrationGenesisSigner for ProductionRegistrationGenesisSigner {
    fn sign_genesis(
        &self,
        caller: &str,
        operator_manual: bool,
        unsigned: &UnsignedGenesisDidOp,
    ) -> Result<DidOpSignature> {
        ensure!(
            !caller.is_empty() && caller != UNAUTHENTICATED_DID_SENTINEL,
            "genesis signing requires an authenticated caller"
        );
        ensure!(
            !operator_manual || caller.starts_with("service:"),
            "manual genesis signing requires a service authority"
        );
        let label = hosted_account_label(unsigned.did())?;
        let (ed, pq) = account_genesis_signing_keys(&self.signing_root, label);
        sign_genesis(unsigned, &ed, &pq)
    }
}

fn account_genesis_signing_keys(
    root: &ed25519_dalek::SigningKey,
    label: &str,
) -> (
    ed25519_dalek::SigningKey,
    hyprstream_crypto::pq::MlDsaSigningKey,
) {
    let ed = hyprstream_rpc::node_identity::derive_purpose_key(
        root,
        &format!("{ACCOUNT_GENESIS_ED25519_PURPOSE}/{label}"),
    );
    let pq_seed_key = hyprstream_rpc::node_identity::derive_purpose_key(
        root,
        &format!("{ACCOUNT_GENESIS_MLDSA65_PURPOSE}/{label}"),
    );
    let mut pq_seed = pq_seed_key.to_bytes();
    let pq = hyprstream_crypto::pq::ml_dsa_sk_from_seed(&pq_seed);
    pq_seed.zeroize();
    (ed, pq)
}

fn account_genesis_rotation_keys(
    root: &ed25519_dalek::SigningKey,
    label: &str,
) -> Result<GenesisRotationKeys> {
    let (ed, pq) = account_genesis_signing_keys(root, label);
    let hybrid = HybridRotationKey::new(
        ed.verifying_key().to_bytes(),
        hyprstream_crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
    )?;
    GenesisRotationKeys::new(
        UserRotationKey::new(hybrid),
        RecoveryKeyEnrollment::Declined,
        HostKeyEnrollment::Absent,
    )
}

fn hosted_account_label(did: &str) -> Result<&str> {
    let host = did
        .strip_prefix("did:web:")
        .context("hosted-account genesis DID must use did:web")?;
    let label = host
        .split('.')
        .next()
        .context("hosted-account genesis DID has no account label")?;
    ensure!(
        !label.is_empty() && !label.contains([':', '/', '%']),
        "hosted-account genesis DID has an invalid account label"
    );
    Ok(label)
}

struct ConfiguredHostedPdsDiscovery {
    pds_endpoint: String,
    quic: QuicConfig,
}

impl HostedPdsDiscovery for ConfiguredHostedPdsDiscovery {
    fn current(&self) -> Result<LiveHostedPdsDiscovery> {
        ensure!(
            self.quic.enabled,
            "hosted-account minting requires live QUIC"
        );
        let endpoint = Url::parse(&self.pds_endpoint).context("hosted PDS endpoint is invalid")?;
        ensure!(
            endpoint.scheme() == "https",
            "hosted PDS endpoint must use HTTPS"
        );
        let pds_endpoint = endpoint.origin().ascii_serialization();
        let host = endpoint
            .host_str()
            .context("hosted PDS endpoint has no host")?;
        let port = self.quic.socket_addr()?.port();
        let quic_url = format!("https://{host}:{port}");
        let (cert_chain, _) = self.quic.load_tls_materials()?;
        let leaf = cert_chain
            .first()
            .context("hosted PDS QUIC certificate chain is empty")?;
        let digest = ring::digest::digest(&ring::digest::SHA256, leaf);
        let mut cert_hash = [0_u8; 32];
        cert_hash.copy_from_slice(digest.as_ref());
        let auth = hyprstream_rpc::transport::QuicServerAuth::pinned(vec![cert_hash])?;
        let entry = serde_json::json!({
            "id": format!("{pds_endpoint}#quic"),
            "type": "QuicTransport",
            "serviceEndpoint": hyprstream_rpc::service_entry::encode_quic(
                &quic_url,
                &auth,
                &["hyprstream-rpc/1"],
            ),
        });
        Ok(LiveHostedPdsDiscovery {
            pds_endpoint,
            browser_quic_reach: hyprstream_rpc::service_entry::decode_browser_quic_reach(&entry)?,
        })
    }
}

impl LivePlcDiscovery for ConfiguredHostedPdsDiscovery {
    fn current(
        &self,
        _did: &str,
        pds_endpoint: &str,
    ) -> Result<hyprstream_rpc::service_entry::BrowserQuicReach> {
        let current = <Self as HostedPdsDiscovery>::current(self)?;
        let bound_endpoint = Url::parse(pds_endpoint)
            .context("PLC directory PDS endpoint is invalid")?
            .origin()
            .ascii_serialization();
        ensure!(
            current.pds_endpoint == bound_endpoint,
            "PLC directory identity is not bound to this hosted PDS"
        );
        Ok(current.browser_quic_reach)
    }
}

struct HostedConnectTimeResolver {
    pds_root: PathBuf,
    zone: AccountZone,
    live_discovery: Arc<ConfiguredHostedPdsDiscovery>,
}

impl HostedConnectTimeResolver {
    fn local_label<'a>(&self, did: &'a str) -> Option<&'a str> {
        let host = did.strip_prefix("did:web:")?;
        let suffix = format!(".{}", self.zone.apex());
        let label = host.strip_suffix(&suffix)?;
        let name = AllocatedAccountName::new(label, did).ok()?;
        (self.zone.host_for_label(name.label()).ok()?.as_str() == host).then_some(label)
    }

    fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery> {
        let label = self
            .local_label(did)
            .context("DID is not a local hosted account")?;
        let account_dir = self
            .pds_root
            .join(self.zone.apex())
            .join("accounts")
            .join(label);
        let record = AccountRecord::from_dag_cbor(
            &read_authority_artifact(&account_dir.join("account-record.cbor"))
                .context("reading hosted account record")?,
        )
        .context("verifying hosted account record")?;
        let genesis = GenesisDidOp::from_dag_cbor(
            &read_authority_artifact(&account_dir.join("genesis.didop.cbor"))
                .context("reading hosted account genesis")?,
        )
        .context("verifying hosted account genesis")?;
        let document = SealedHostedDidDocument::from_canonical_json(
            &read_authority_artifact(&account_dir.join("did-document.json"))
                .context("reading hosted account DID document")?,
        )
        .context("verifying hosted account DID document")?;
        ensure!(
            record.name().label() == label
                && record.name().did() == did
                && document.did() == did
                && record.genesis_op() == genesis.cid()?
                && record.current_op() == record.genesis_op()
                && record.doc_cid() == genesis.unsigned().doc_cid()
                && record.doc_cid() == document.cid()
                && record.atproto_verifying_key()?.to_encoded_point(true)
                    == document.atproto_verifying_key()?.to_encoded_point(true),
            "hosted account authority artifacts are inconsistent"
        );
        let reach = <ConfiguredHostedPdsDiscovery as LivePlcDiscovery>::current(
            self.live_discovery.as_ref(),
            did,
            document.pds_endpoint(),
        )?;
        serde_json::from_value(serde_json::json!({
            "quicUrl": reach.quic_url,
            "certHash": reach.cert_hash,
        }))
        .context("encoding hosted connect-time discovery")
    }
}

impl IdentityConnectTimeResolver for HostedConnectTimeResolver {
    fn recognizes(&self, did: &str) -> bool {
        self.local_label(did).is_some()
    }

    fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery> {
        HostedConnectTimeResolver::resolve(self, did)
    }

    fn hosted_tenant(&self, did: &str) -> Result<Option<String>> {
        // Verify the authority artifacts before returning the deployment-owned
        // tenant. Classification by DID suffix alone is not sufficient.
        HostedConnectTimeResolver::resolve(self, did)?;
        Ok(Some(self.zone.apex().to_owned()))
    }
}

fn read_authority_artifact(path: &Path) -> Result<Vec<u8>> {
    let mut limited = File::open(path)?.take(MAX_AUTHORITY_ARTIFACT_BYTES + 1);
    let mut bytes = Vec::new();
    limited.read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_AUTHORITY_ARTIFACT_BYTES as usize,
        "authority artifact exceeds the {MAX_AUTHORITY_ARTIFACT_BYTES}-byte read limit"
    );
    Ok(bytes)
}

struct AuthorityConnectTimeResolver {
    plc: PlcDirectory,
    hosted: Option<HostedConnectTimeResolver>,
}

impl IdentityConnectTimeResolver for AuthorityConnectTimeResolver {
    fn recognizes(&self, did: &str) -> bool {
        hyprstream_rpc::did_plc::is_did_plc(did)
            || self
                .hosted
                .as_ref()
                .and_then(|resolver| resolver.local_label(did))
                .is_some()
    }

    fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery> {
        if hyprstream_rpc::did_plc::is_did_plc(did) {
            return self.plc.resolve(did);
        }
        self.hosted
            .as_ref()
            .context("hosted account resolution is not configured")?
            .resolve(did)
    }

    fn hosted_tenant(&self, did: &str) -> Result<Option<String>> {
        if hyprstream_rpc::did_plc::is_did_plc(did) {
            return Ok(None);
        }
        match &self.hosted {
            Some(hosted) => hosted.hosted_tenant(did),
            None => Ok(None),
        }
    }
}

struct ProductionHostedAccountPublisher {
    pds_root: PathBuf,
}

impl HostedPdsAccountPublisher for ProductionHostedAccountPublisher {
    fn publish(&self, publication: HostedPdsAccountPublication<'_>) -> Result<()> {
        let tenant_root = self.pds_root.join(publication.tenant());
        let accounts_root = tenant_root.join("accounts");
        create_private_dir_all(&accounts_root)?;
        sync_directory(&tenant_root)?;
        let staging = tempfile::Builder::new()
            .prefix(".account-mint-")
            .tempdir_in(&tenant_root)?;
        let staging_store = DirectoryHostedAccountStore::new(staging.path());
        publication.account().write_to(&staging_store)?;
        let staged_account = staging.path().join(publication.label());
        write_repo_genesis(&staged_account, publication.repo())?;
        if let Some(transaction_id) = publication.transaction_id() {
            let transaction = serde_json::to_vec(&serde_json::json!({
                "version": 1,
                "transaction_id": transaction_id,
            }))?;
            write_private_file(&staged_account.join(SIGNUP_TRANSACTION_FILE), &transaction)?;
            sync_directory(&staged_account)?;
        }

        let final_account = accounts_root.join(publication.label());
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        match nix::fcntl::renameat2(
            None,
            &staged_account,
            None,
            &final_account,
            nix::fcntl::RenameFlags::RENAME_NOREPLACE,
        ) {
            Ok(()) => {}
            Err(nix::errno::Errno::EEXIST) if publication.transaction_id().is_some() => {
                verify_resumable_signup_publication(&final_account, &publication)?;
                sync_directory(&accounts_root)?;
                return Ok(());
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to atomically publish hosted account {:?}",
                        publication.label()
                    )
                });
            }
        }
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        anyhow::bail!("atomic hosted-account publication requires Linux renameat2");
        sync_directory(&accounts_root)
    }
}

fn verify_resumable_signup_publication(
    account_dir: &Path,
    expected: &HostedPdsAccountPublication<'_>,
) -> Result<()> {
    let transaction: serde_json::Value = serde_json::from_slice(
        &read_authority_artifact(&account_dir.join(SIGNUP_TRANSACTION_FILE))
            .context("reading hosted signup transaction marker")?,
    )
    .context("verifying hosted signup transaction marker")?;
    ensure!(
        transaction
            .get("version")
            .and_then(serde_json::Value::as_u64)
            == Some(1)
            && transaction
                .get("transaction_id")
                .and_then(serde_json::Value::as_str)
                == expected.transaction_id(),
        "published hosted account belongs to a different signup transaction"
    );

    let record_bytes = read_authority_artifact(&account_dir.join("account-record.cbor"))
        .context("reading published hosted account record")?;
    let record =
        AccountRecord::from_dag_cbor(&record_bytes).context("verifying hosted account record")?;
    let genesis = GenesisDidOp::from_dag_cbor(
        &read_authority_artifact(&account_dir.join("genesis.didop.cbor"))
            .context("reading published hosted account genesis")?,
    )
    .context("verifying hosted account genesis")?;
    let document = SealedHostedDidDocument::from_canonical_json(
        &read_authority_artifact(&account_dir.join("did-document.json"))
            .context("reading published hosted account DID document")?,
    )
    .context("verifying hosted account DID document")?;
    let expected_name = expected.account().record().name();
    ensure!(
        record.name() == expected_name
            && document.did() == expected_name.did()
            && document.handle() == expected.account().did_document().handle()
            && document.pds_endpoint() == expected.account().did_document().pds_endpoint()
            && record.genesis_op() == genesis.cid()?
            && record.current_op() == record.genesis_op()
            && record.doc_cid() == genesis.unsigned().doc_cid()
            && record.doc_cid() == document.cid()
            && record.atproto_verifying_key()?.to_encoded_point(true)
                == document.atproto_verifying_key()?.to_encoded_point(true),
        "published hosted signup authority artifacts are inconsistent"
    );

    let commit_bytes = read_authority_artifact(&account_dir.join("repo/commit.cbor"))
        .context("reading published hosted repo commit")?;
    let commit = hyprstream_pds::commit::Commit::from_dag_cbor(&commit_bytes)
        .context("verifying published hosted repo commit")?;
    commit
        .verify(&record.atproto_verifying_key()?)
        .context("published hosted repo signature is invalid")?;
    let commit_cid = commit.cid();
    ensure!(
        commit.did == expected_name.did()
            && commit.prev.is_none()
            && genesis.unsigned().head_at_op() == GenesisRepoHead::Existing(commit_cid)
            && read_authority_artifact(
                &account_dir.join(format!("repo/blocks/{}.cbor", commit.data))
            )
            .is_ok_and(|bytes| hyprstream_pds::Cid::from_dag_cbor(&bytes) == commit.data)
            && std::fs::read_to_string(account_dir.join("repo/head"))
                .is_ok_and(|head| head == commit_cid.to_string()),
        "published hosted signup repo genesis is incomplete or inconsistent"
    );
    Ok(())
}

fn write_repo_genesis(account_dir: &Path, repo: &hyprstream_pds::HostedRepoGenesis) -> Result<()> {
    let repo_dir = account_dir.join("repo");
    let blocks_dir = repo_dir.join("blocks");
    create_private_dir_all(&blocks_dir)?;
    for (cid, bytes) in repo.mst_blocks() {
        write_private_file(&blocks_dir.join(format!("{cid}.cbor")), bytes)?;
    }
    write_private_file(&repo_dir.join("commit.cbor"), repo.commit_bytes())?;
    write_private_file(
        &repo_dir.join("head"),
        repo.commit_cid().to_string().as_bytes(),
    )?;
    sync_directory(&blocks_dir)?;
    sync_directory(&repo_dir)?;
    sync_directory(account_dir)
}

fn create_private_dir_all(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn private_create_new(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options.open(path)
}

fn write_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = private_create_new(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

/// Handle-only registration payload.
///
/// `deny_unknown_fields` makes `tenant`, `did`, mode, and transport metadata
/// invalid at the JSON boundary instead of silently ignoring them.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegisterHostedAccountRequest {
    pub handle: String,
}

/// Browser registration result.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RegisterHostedAccountResponse {
    pub handle: String,
    pub did: String,
    pub pds_endpoint: String,
    pub quic_url: String,
    pub cert_hash: String,
}

impl From<HostedAccountRegistrationResult> for RegisterHostedAccountResponse {
    fn from(result: HostedAccountRegistrationResult) -> Self {
        Self {
            handle: result.handle,
            did: result.did,
            pds_endpoint: result.pds_endpoint,
            quic_url: result.quic_url,
            cert_hash: result.cert_hash,
        }
    }
}

/// Federation-add payload.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FederationIntakeRequest {
    pub did: String,
}

/// Public connect-time resolution query.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResolveIdentityQuery {
    pub did: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegistrationPath {
    SelfService,
    OperatorManual,
}

/// An identity created only after the corresponding HTTP authenticator passes.
#[derive(Clone, Debug)]
pub(crate) struct AuthenticatedIdentityCaller {
    subject: String,
}

impl AuthenticatedIdentityCaller {
    fn new(subject: impl Into<String>) -> Result<Self, IdentityApiError> {
        let subject = subject.into();
        if subject.is_empty() || subject == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::Unauthenticated);
        }
        Ok(Self { subject })
    }

    fn subject(&self) -> &str {
        &self.subject
    }
}

/// Select and use the hybrid genesis signer for one authenticated request.
///
/// Implementations may map self-service callers to user-held key material and
/// operator-manual callers to the deployment-authority workflow. The client
/// cannot select this path in its JSON body.
pub trait RegistrationGenesisSigner: Send + Sync {
    fn sign_genesis(
        &self,
        caller: &str,
        operator_manual: bool,
        unsigned: &UnsignedGenesisDidOp,
    ) -> Result<DidOpSignature>;
}

/// Injectable mint boundary used by the HTTP adapter and causal tests.
pub trait HostedRegistrationMint: Send + Sync {
    fn mint(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
    ) -> Result<HostedAccountRegistrationResult>;

    fn mint_for_transaction(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
        _transaction_id: &str,
    ) -> Result<HostedAccountRegistrationResult> {
        self.mint(request, signer)
    }
}

impl HostedRegistrationMint for HostedPdsAccountMinter {
    fn mint(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
    ) -> Result<HostedAccountRegistrationResult> {
        HostedPdsAccountMinter::mint(self, request, signer)
    }

    fn mint_for_transaction(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
        transaction_id: &str,
    ) -> Result<HostedAccountRegistrationResult> {
        HostedPdsAccountMinter::mint_for_transaction(self, request, signer, transaction_id)
    }
}

/// Injectable federation-intake boundary.
#[async_trait]
pub trait FederatedIdentityIntake: Send + Sync {
    async fn intake(&self, did: &str) -> Result<InventoryEntry>;
}

#[async_trait]
impl FederatedIdentityIntake for FederationIntake {
    async fn intake(&self, did: &str) -> Result<InventoryEntry> {
        FederationIntake::intake(self, did).await
    }
}

/// Injectable connect-time boundary over authority-known local identities.
pub trait IdentityConnectTimeResolver: Send + Sync {
    /// Pure classification that must not perform resolver or network I/O.
    fn recognizes(&self, did: &str) -> bool;

    fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery>;

    /// Return the deployment-authority tenant for a verified local hosted DID.
    ///
    /// Federation resolvers and test fakes default to no local binding.
    fn hosted_tenant(&self, _did: &str) -> Result<Option<String>> {
        Ok(None)
    }
}

/// Exact HTTPS origins that may be resolved through the `did:web` intake arm.
///
/// Wildcards and suffix matching are intentionally unsupported. An empty set
/// means `did:web` intake is disabled, while fixed-directory `did:plc` intake
/// remains available.
#[derive(Clone, Debug)]
pub struct DidWebOriginAllowlist {
    origins: BTreeSet<String>,
}

impl DidWebOriginAllowlist {
    pub fn new<I, S>(origins: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut normalized = BTreeSet::new();
        for origin in origins {
            let origin = origin.as_ref();
            let parsed = Url::parse(origin)?;
            ensure!(
                parsed.scheme() == "https"
                    && parsed.host_str().is_some_and(|host| !host.contains('*'))
                    && parsed.username().is_empty()
                    && parsed.password().is_none()
                    && parsed.path() == "/"
                    && parsed.query().is_none()
                    && parsed.fragment().is_none(),
                "did:web allowlist entries must be exact HTTPS origins"
            );
            normalized.insert(parsed.origin().ascii_serialization());
        }
        Ok(Self {
            origins: normalized,
        })
    }

    fn require_allowed(&self, did: &str) -> Result<(), IdentityApiError> {
        let document_url = hyprstream_rpc::did_web::did_web_to_url(did)
            .map_err(|_| IdentityApiError::InvalidRequest)?;
        let parsed = Url::parse(&document_url).map_err(|_| IdentityApiError::InvalidRequest)?;
        let origin = parsed.origin().ascii_serialization();
        if !self.origins.contains(&origin) {
            return Err(IdentityApiError::ResolvableHostDenied);
        }
        Ok(())
    }
}

/// Rate-limited wire adapter over hosted mint, intake, and public local resolution.
pub struct IdentityRegistrationApi {
    minter: Arc<dyn HostedRegistrationMint>,
    signer: Arc<dyn RegistrationGenesisSigner>,
    intake: Arc<dyn FederatedIdentityIntake>,
    identity_resolver: Arc<dyn IdentityConnectTimeResolver>,
    did_web_origins: DidWebOriginAllowlist,
    rate_limiter: Arc<RateLimiter>,
}

impl IdentityRegistrationApi {
    #[must_use]
    pub fn new(
        minter: Arc<dyn HostedRegistrationMint>,
        signer: Arc<dyn RegistrationGenesisSigner>,
        intake: Arc<dyn FederatedIdentityIntake>,
        identity_resolver: Arc<dyn IdentityConnectTimeResolver>,
        did_web_origins: DidWebOriginAllowlist,
        rate_limiter: Arc<RateLimiter>,
    ) -> Self {
        Self {
            minter,
            signer,
            intake,
            identity_resolver,
            did_web_origins,
            rate_limiter,
        }
    }

    fn register(
        &self,
        caller: &AuthenticatedIdentityCaller,
        path: RegistrationPath,
        request: RegisterHostedAccountRequest,
    ) -> Result<RegisterHostedAccountResponse, IdentityApiError> {
        if path == RegistrationPath::OperatorManual && !caller.subject().starts_with("service:") {
            return Err(IdentityApiError::Forbidden);
        }
        self.check_rate(caller)?;
        if request.handle.is_empty() || request.handle == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::InvalidRequest);
        }

        let mint_request =
            HostedAccountRegistrationRequest::from_client_fields(request.handle, None)
                .map_err(|_| IdentityApiError::InvalidRequest)?;
        let signer = CallerGenesisSigner {
            provider: self.signer.as_ref(),
            caller: caller.subject(),
            operator_manual: path == RegistrationPath::OperatorManual,
        };
        let result = self
            .minter
            .mint(&mint_request, &signer)
            .map_err(IdentityApiError::Backend)?;
        if result.did == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::Backend(anyhow::anyhow!(
                "hosted-account mint returned the reserved unauthenticated DID"
            )));
        }
        Ok(RegisterHostedAccountResponse::from(result))
    }

    /// Mint a hosted account from a cryptographically authenticated OAuth
    /// authorize transaction.
    ///
    /// This is deliberately an internal call, not an HTTP registration route.
    /// `dpop_jkt` comes from the verified PAR snapshot and `fingerprint` from
    /// the verified vault proof. Both are used only as rate-limit identities;
    /// the deployment authority still allocates the DID and tenant.
    pub(super) fn mint_for_oauth_signup(
        &self,
        handle: &str,
        dpop_jkt: &str,
        fingerprint: &str,
    ) -> Result<RegisterHostedAccountResponse, IdentityApiError> {
        if handle.is_empty()
            || handle == UNAUTHENTICATED_DID_SENTINEL
            || dpop_jkt.is_empty()
            || fingerprint.is_empty()
        {
            return Err(IdentityApiError::InvalidRequest);
        }
        if self
            .rate_limiter
            .check_and_increment(&format!("oauth-authorize-signup-key:{fingerprint}"))
        {
            return Err(IdentityApiError::RateLimited);
        }

        let mint_request =
            HostedAccountRegistrationRequest::from_client_fields(handle.to_owned(), None)
                .map_err(|_| IdentityApiError::InvalidRequest)?;
        let caller = format!("oauth-signup:{fingerprint}");
        let signer = CallerGenesisSigner {
            provider: self.signer.as_ref(),
            caller: &caller,
            operator_manual: false,
        };
        let result = self
            .minter
            .mint_for_transaction(&mint_request, &signer, fingerprint)
            .map_err(|error| {
                if error.chain().any(|cause| {
                    cause
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|io| io.kind() == std::io::ErrorKind::AlreadyExists)
                }) {
                    IdentityApiError::HandleUnavailable
                } else {
                    IdentityApiError::Backend(error)
                }
            })?;
        if result.did == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::Backend(anyhow::anyhow!(
                "hosted-account mint returned the reserved unauthenticated DID"
            )));
        }
        Ok(result.into())
    }

    /// Charge the public/global and verified PAR-DPoP signup buckets before
    /// performing even an Ed25519 proof verification. A single-use authorize
    /// nonce limits replay; these bounded buckets limit fresh-PAR abuse.
    pub(super) fn check_oauth_signup_rate(&self, dpop_jkt: &str) -> Result<(), IdentityApiError> {
        if dpop_jkt.is_empty() {
            return Err(IdentityApiError::InvalidRequest);
        }
        for bucket in [
            COLD_SIGNUP_GLOBAL_RATE_LIMIT_BUCKET.to_owned(),
            format!("oauth-authorize-signup-dpop:{dpop_jkt}"),
        ] {
            if self.rate_limiter.check_and_increment(&bucket) {
                return Err(IdentityApiError::RateLimited);
            }
        }
        Ok(())
    }

    async fn intake(
        &self,
        caller: &AuthenticatedIdentityCaller,
        request: FederationIntakeRequest,
    ) -> Result<InventoryEntry, IdentityApiError> {
        self.check_rate(caller)?;
        if request.did == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::InvalidRequest);
        }
        if request.did.starts_with("did:web:") {
            self.did_web_origins.require_allowed(&request.did)?;
        } else if !hyprstream_rpc::did_plc::is_did_plc(&request.did) {
            return Err(IdentityApiError::InvalidRequest);
        }
        self.intake
            .intake(&request.did)
            .await
            .map_err(IdentityApiError::Backend)
    }

    fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery, IdentityApiError> {
        if self
            .rate_limiter
            .check_and_increment(PUBLIC_RESOLVE_RATE_LIMIT_BUCKET)
        {
            return Err(IdentityApiError::ResolveRateLimited);
        }
        if did == UNAUTHENTICATED_DID_SENTINEL || !self.identity_resolver.recognizes(did) {
            return Err(IdentityApiError::NotFound(anyhow::anyhow!(
                "identity is outside the local authority directory"
            )));
        }
        self.identity_resolver
            .resolve(did)
            .map_err(IdentityApiError::NotFound)
    }

    pub(super) fn hosted_tenant(&self, did: &str) -> Result<Option<String>> {
        self.identity_resolver.hosted_tenant(did)
    }

    fn check_rate(&self, caller: &AuthenticatedIdentityCaller) -> Result<(), IdentityApiError> {
        if self.rate_limiter.check_and_increment(caller.subject()) {
            return Err(IdentityApiError::RateLimited);
        }
        Ok(())
    }
}

struct CallerGenesisSigner<'a> {
    provider: &'a dyn RegistrationGenesisSigner,
    caller: &'a str,
    operator_manual: bool,
}

impl HostedAccountGenesisSigner for CallerGenesisSigner<'_> {
    fn sign(&self, unsigned: &UnsignedGenesisDidOp) -> Result<DidOpSignature> {
        self.provider
            .sign_genesis(self.caller, self.operator_manual, unsigned)
    }
}

#[derive(Debug)]
pub(super) enum IdentityApiError {
    Unauthenticated,
    Forbidden,
    RateLimited,
    ResolveRateLimited,
    ResolvableHostDenied,
    InvalidRequest,
    HandleUnavailable,
    NotFound(anyhow::Error),
    Backend(anyhow::Error),
}

impl IntoResponse for IdentityApiError {
    fn into_response(self) -> Response {
        let (status, code, description) = match &self {
            Self::Unauthenticated => (
                StatusCode::UNAUTHORIZED,
                "authentication_required",
                "Authenticated caller required",
            ),
            Self::Forbidden => (
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                "Operator authority required",
            ),
            Self::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "Identity mutation rate limit exceeded",
            ),
            Self::ResolveRateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "Identity resolution rate limit exceeded",
            ),
            Self::ResolvableHostDenied => (
                StatusCode::FORBIDDEN,
                "origin_not_allowed",
                "DID web origin is not permitted",
            ),
            Self::InvalidRequest => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Registration or intake request is invalid",
            ),
            Self::HandleUnavailable => (
                StatusCode::CONFLICT,
                "handle_unavailable",
                "Requested hosted-account handle is unavailable",
            ),
            Self::NotFound(_) => (
                StatusCode::NOT_FOUND,
                "not_found",
                "Identity resolution not found",
            ),
            Self::Backend(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Identity operation failed",
            ),
        };
        if let Self::Backend(error) = &self {
            warn!(%error, "identity registration/intake backend failed");
        }
        if let Self::NotFound(error) = &self {
            warn!(%error, "identity resolution failed closed");
        }
        (
            status,
            Json(serde_json::json!({
                "error": code,
                "error_description": description,
            })),
        )
            .into_response()
    }
}

fn unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "temporarily_unavailable",
            "error_description": "Identity registration is not configured",
        })),
    )
        .into_response()
}

pub(super) async fn require_registration_session(
    State(state): State<Arc<OAuthState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let session = match session::extract_session_id(request.headers()) {
        Some(session_id) => state.sessions.get(&session_id).await,
        None => None,
    };
    let Some(session) = session else {
        return IdentityApiError::Unauthenticated.into_response();
    };
    let caller = match AuthenticatedIdentityCaller::new(session.username) {
        Ok(caller) => caller,
        Err(error) => return error.into_response(),
    };
    request.extensions_mut().insert(caller);
    next.run(request).await
}

pub(super) async fn register_self_service(
    State(state): State<Arc<OAuthState>>,
    Extension(caller): Extension<AuthenticatedIdentityCaller>,
    Json(request): Json<RegisterHostedAccountRequest>,
) -> Response {
    match state.identity_registration_api.as_deref() {
        Some(api) => api
            .register(&caller, RegistrationPath::SelfService, request)
            .map(Json)
            .into_response(),
        None => unavailable_response(),
    }
}

pub(super) async fn register_operator_manual(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(request): Json<RegisterHostedAccountRequest>,
) -> Response {
    let caller = match AuthenticatedIdentityCaller::new(user.user) {
        Ok(caller) => caller,
        Err(error) => return error.into_response(),
    };
    match state.identity_registration_api.as_deref() {
        Some(api) => api
            .register(&caller, RegistrationPath::OperatorManual, request)
            .map(Json)
            .into_response(),
        None => unavailable_response(),
    }
}

pub(super) async fn intake_federated_identity(
    State(state): State<Arc<OAuthState>>,
    Extension(caller): Extension<AuthenticatedIdentityCaller>,
    Json(request): Json<FederationIntakeRequest>,
) -> Response {
    match state.identity_registration_api.as_deref() {
        Some(api) => api.intake(&caller, request).await.map(Json).into_response(),
        None => unavailable_response(),
    }
}

pub(super) async fn resolve_identity(
    State(state): State<Arc<OAuthState>>,
    Query(query): Query<ResolveIdentityQuery>,
) -> Response {
    match state.identity_registration_api.as_deref() {
        Some(api) => api.resolve(&query.did).map(Json).into_response(),
        None => unavailable_response(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::bail;

    use super::*;

    const PUBLIC_PLC_DID: &str = "did:plc:ewvi7nxzyoun6zhxrhs64oiz";
    const LOCAL_HOSTED_DID: &str = "did:web:alice.accounts.example.com";

    struct FakeMint {
        calls: AtomicUsize,
    }

    impl HostedRegistrationMint for FakeMint {
        fn mint(
            &self,
            request: &HostedAccountRegistrationRequest,
            _signer: &dyn HostedAccountGenesisSigner,
        ) -> Result<HostedAccountRegistrationResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let did = if request.handle() == "reserved-result" {
                UNAUTHENTICATED_DID_SENTINEL.to_owned()
            } else {
                format!("did:web:{}.accounts.example", request.handle())
            };
            Ok(HostedAccountRegistrationResult {
                handle: format!("at://{}.accounts.example", request.handle()),
                did,
                pds_endpoint: "https://pds.example".to_owned(),
                quic_url: "https://pds.example:4433/wt".to_owned(),
                cert_hash: "zQmPin".to_owned(),
            })
        }
    }

    struct FakeSigner;

    impl RegistrationGenesisSigner for FakeSigner {
        fn sign_genesis(
            &self,
            _caller: &str,
            _operator_manual: bool,
            _unsigned: &UnsignedGenesisDidOp,
        ) -> Result<DidOpSignature> {
            bail!("fake mint must not invoke the signer")
        }
    }

    struct CountingIntake {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl FederatedIdentityIntake for CountingIntake {
        async fn intake(&self, _did: &str) -> Result<InventoryEntry> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            bail!("counting intake reached")
        }
    }

    struct CountingIdentityResolver {
        calls: AtomicUsize,
    }

    impl IdentityConnectTimeResolver for CountingIdentityResolver {
        fn recognizes(&self, did: &str) -> bool {
            matches!(did, PUBLIC_PLC_DID | LOCAL_HOSTED_DID)
                || did == "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"
        }

        fn resolve(&self, did: &str) -> Result<ConnectTimeDiscovery> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !matches!(did, PUBLIC_PLC_DID | LOCAL_HOSTED_DID) {
                bail!("identity not found");
            }
            Ok(serde_json::from_value(serde_json::json!({
                "quicUrl": "https://pds.example:4433",
                "certHash": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
            }))?)
        }
    }

    fn fixture(
        max_requests: u32,
    ) -> (
        Arc<FakeMint>,
        Arc<CountingIntake>,
        Arc<CountingIdentityResolver>,
        IdentityRegistrationApi,
    ) {
        let mint = Arc::new(FakeMint {
            calls: AtomicUsize::new(0),
        });
        let intake = Arc::new(CountingIntake {
            calls: AtomicUsize::new(0),
        });
        let identity_resolver = Arc::new(CountingIdentityResolver {
            calls: AtomicUsize::new(0),
        });
        let api = IdentityRegistrationApi::new(
            mint.clone(),
            Arc::new(FakeSigner),
            intake.clone(),
            identity_resolver.clone(),
            DidWebOriginAllowlist::new(["https://federated.example"]).unwrap(),
            Arc::new(RateLimiter::new(max_requests, 60)),
        );
        (mint, intake, identity_resolver, api)
    }

    #[test]
    fn handle_only_contract_rejects_client_authority_fields() {
        let error = serde_json::from_value::<RegisterHostedAccountRequest>(
            serde_json::json!({"handle": "alice", "tenant": "attacker"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));

        let error = serde_json::from_value::<RegisterHostedAccountRequest>(
            serde_json::json!({"handle": "alice", "mode": "operator"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn self_service_and_manual_paths_return_exact_registration_contract() {
        let (mint, _intake, _plc_resolver, api) = fixture(10);
        let self_caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        let response = api
            .register(
                &self_caller,
                RegistrationPath::SelfService,
                RegisterHostedAccountRequest {
                    handle: "alice".to_owned(),
                },
            )
            .unwrap();
        assert_eq!(response.handle, "at://alice.accounts.example");
        assert_eq!(response.did, "did:web:alice.accounts.example");
        assert_eq!(response.pds_endpoint, "https://pds.example");
        assert_eq!(response.quic_url, "https://pds.example:4433/wt");
        assert_eq!(response.cert_hash, "zQmPin");
        assert_eq!(
            serde_json::to_value(&response).unwrap(),
            serde_json::json!({
                "handle": "at://alice.accounts.example",
                "did": "did:web:alice.accounts.example",
                "pdsEndpoint": "https://pds.example",
                "quicUrl": "https://pds.example:4433/wt",
                "certHash": "zQmPin",
            })
        );

        let operator = AuthenticatedIdentityCaller::new("service:identity-operator").unwrap();
        api.register(
            &operator,
            RegistrationPath::OperatorManual,
            RegisterHostedAccountRequest {
                handle: "bob".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(mint.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn manual_path_rejects_non_service_and_unknown_before_mint() {
        let (mint, _intake, _plc_resolver, api) = fixture(10);
        let user = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        assert!(matches!(
            api.register(
                &user,
                RegistrationPath::OperatorManual,
                RegisterHostedAccountRequest {
                    handle: "alice".to_owned(),
                },
            ),
            Err(IdentityApiError::Forbidden)
        ));
        assert!(matches!(
            api.register(
                &user,
                RegistrationPath::SelfService,
                RegisterHostedAccountRequest {
                    handle: UNAUTHENTICATED_DID_SENTINEL.to_owned(),
                },
            ),
            Err(IdentityApiError::InvalidRequest)
        ));
        assert_eq!(mint.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn reserved_did_is_rejected_again_at_the_api_output_boundary() {
        let (mint, _intake, _plc_resolver, api) = fixture(10);
        let caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        assert!(matches!(
            api.register(
                &caller,
                RegistrationPath::SelfService,
                RegisterHostedAccountRequest {
                    handle: "reserved-result".to_owned(),
                },
            ),
            Err(IdentityApiError::Backend(_))
        ));
        assert_eq!(mint.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn rate_limit_runs_before_mint() {
        let (mint, _intake, _plc_resolver, api) = fixture(1);
        let caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        let request = || RegisterHostedAccountRequest {
            handle: "alice".to_owned(),
        };
        api.register(&caller, RegistrationPath::SelfService, request())
            .unwrap();
        assert!(matches!(
            api.register(&caller, RegistrationPath::SelfService, request()),
            Err(IdentityApiError::RateLimited)
        ));
        assert_eq!(mint.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn intake_constrains_did_web_before_resolver_io() {
        let (_mint, intake, _plc_resolver, api) = fixture(10);
        let caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();

        assert!(matches!(
            api.intake(
                &caller,
                FederationIntakeRequest {
                    did: "did:web:127.0.0.1%3A8443".to_owned(),
                },
            )
            .await,
            Err(IdentityApiError::ResolvableHostDenied)
        ));
        assert!(matches!(
            api.intake(
                &caller,
                FederationIntakeRequest {
                    did: UNAUTHENTICATED_DID_SENTINEL.to_owned(),
                },
            )
            .await,
            Err(IdentityApiError::InvalidRequest)
        ));
        assert_eq!(intake.calls.load(Ordering::SeqCst), 0);

        let error = api
            .intake(
                &caller,
                FederationIntakeRequest {
                    did: "did:web:federated.example".to_owned(),
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IdentityApiError::Backend(_)));
        assert_eq!(intake.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn public_resolution_returns_exact_connect_time_contract() {
        let (_mint, _intake, identity_resolver, api) = fixture(10);

        let response = api.resolve(PUBLIC_PLC_DID).unwrap();

        assert_eq!(response.quic_url(), "https://pds.example:4433");
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            serde_json::json!({
                "quicUrl": "https://pds.example:4433",
                "certHash": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
            })
        );
        assert_eq!(identity_resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn public_resolution_covers_local_hosted_inventory_entries() {
        let (_mint, _intake, identity_resolver, api) = fixture(10);

        let response = api.resolve(LOCAL_HOSTED_DID).unwrap();

        assert_eq!(response.quic_url(), "https://pds.example:4433");
        assert_eq!(identity_resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn public_resolution_rejects_sentinel_and_external_dids_before_resolver_io() {
        let (_mint, _intake, identity_resolver, api) = fixture(10);

        for did in [
            UNAUTHENTICATED_DID_SENTINEL,
            "did:web:foreign.example",
            "did:key:zAttacker",
        ] {
            assert!(matches!(
                api.resolve(did),
                Err(IdentityApiError::NotFound(_))
            ));
        }
        assert_eq!(identity_resolver.calls.load(Ordering::SeqCst), 0);

        assert!(matches!(
            api.resolve("did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"),
            Err(IdentityApiError::NotFound(_))
        ));
        assert_eq!(identity_resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn public_resolution_rate_limit_runs_before_resolver_io() {
        let (_mint, _intake, identity_resolver, api) = fixture(1);

        api.resolve(PUBLIC_PLC_DID).unwrap();
        assert!(matches!(
            api.resolve(PUBLIC_PLC_DID),
            Err(IdentityApiError::ResolveRateLimited)
        ));
        assert_eq!(identity_resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn allowlist_accepts_only_exact_https_origins() {
        assert!(DidWebOriginAllowlist::new(["http://example.com"]).is_err());
        assert!(DidWebOriginAllowlist::new(["https://example.com/path"]).is_err());
        assert!(DidWebOriginAllowlist::new(["https://*.example.com"]).is_err());

        let allowlist =
            DidWebOriginAllowlist::new(["https://example.com", "https://example.com:8443"])
                .unwrap();
        assert!(allowlist.require_allowed("did:web:example.com").is_ok());
        assert!(allowlist
            .require_allowed("did:web:example.com%3A8443")
            .is_ok());
        assert!(matches!(
            allowlist.require_allowed("did:web:sub.example.com"),
            Err(IdentityApiError::ResolvableHostDenied)
        ));
    }

    fn oauth_state() -> (Arc<OAuthState>, crate::config::CorsConfig) {
        use hyprstream_rpc::crypto::CryptoPolicy;
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x51; 32]);
        let remote_key = ed25519_dalek::SigningKey::from_bytes(&[0x52; 32]).verifying_key();
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(signing_key.clone()),
                    LazyUdsTransport::new("/dev/null/identity-registration-test.sock".into()),
                    Some(remote_key),
                )
                .with_response_verify_policy(CryptoPolicy::Classical),
            )
        };
        let mut config = crate::config::OAuthConfig::default();
        config.external_url = Some("https://pds.example.test".to_owned());
        let cors = config.cors.clone();
        (
            Arc::new(OAuthState::new(
                &config,
                crate::services::PolicyClient::new(make_client()),
                crate::services::DiscoveryClient::new(make_client()),
                signing_key.verifying_key().to_bytes(),
            )),
            cors,
        )
    }

    struct FixtureFederatedResolver;

    #[async_trait]
    impl FederatedDidDocumentResolver for FixtureFederatedResolver {
        async fn resolve_federated_document(&self, did: &str) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "id": did,
                "alsoKnownAs": ["at://foreign.example"],
                "service": [{
                    "id": format!("{did}#atproto_pds"),
                    "type": "AtprotoPersonalDataServer",
                    "serviceEndpoint": "https://foreign.example",
                }],
            }))
        }
    }

    struct ProductionRouterFixture {
        state: Arc<OAuthState>,
        cors: crate::config::CorsConfig,
        quic: QuicConfig,
        signing_key: ed25519_dalek::SigningKey,
        storage: tempfile::TempDir,
    }

    fn production_router_fixture() -> ProductionRouterFixture {
        use hyprstream_rpc::crypto::CryptoPolicy;
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let storage = tempfile::TempDir::new().unwrap();
        let certified =
            rcgen::generate_simple_self_signed(vec!["pds.example.test".to_owned()]).unwrap();
        let cert_path = storage.path().join("quic-cert.pem");
        let key_path = storage.path().join("quic-key.pem");
        std::fs::write(&cert_path, certified.cert.pem()).unwrap();
        std::fs::write(&key_path, certified.key_pair.serialize_pem()).unwrap();

        let mut oauth = crate::config::OAuthConfig::default();
        oauth.external_url = Some("https://pds.example.test".to_owned());
        oauth.identity_registration_did_web_origins = vec!["https://foreign.example".to_owned()];
        let account = AccountZoneConfig {
            zone: Some("accounts.example.com".to_owned()),
            ..AccountZoneConfig::default()
        };
        let quic = QuicConfig {
            enabled: true,
            bind_addr: "127.0.0.1:4433".to_owned(),
            server_name: "pds.example.test".to_owned(),
            cert_path: cert_path.to_string_lossy().into_owned(),
            key_path: key_path.to_string_lossy().into_owned(),
            iroh: false,
            relay: String::new(),
        };
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x53; 32]);
        let api = compose_identity_registration_api(
            &oauth,
            &account,
            &quic,
            signing_key.clone(),
            storage.path().join("pds"),
            Arc::new(FixtureFederatedResolver),
            Arc::new(CountingIdentityResolver {
                calls: AtomicUsize::new(0),
            }),
        )
        .unwrap();

        let remote_key = ed25519_dalek::SigningKey::from_bytes(&[0x54; 32]).verifying_key();
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(signing_key.clone()),
                    LazyUdsTransport::new(
                        "/dev/null/identity-registration-production-test.sock".into(),
                    ),
                    Some(remote_key),
                )
                .with_response_verify_policy(CryptoPolicy::Classical),
            )
        };
        let cors = oauth.cors.clone();
        let state = Arc::new(
            OAuthState::new(
                &oauth,
                crate::services::PolicyClient::new(make_client()),
                crate::services::DiscoveryClient::new(make_client()),
                signing_key.verifying_key().to_bytes(),
            )
            .with_identity_registration_api(api),
        );
        ProductionRouterFixture {
            state,
            cors,
            quic,
            signing_key,
            storage,
        }
    }

    fn post(
        path: &str,
        body: &'static str,
        cookie: Option<String>,
        bearer: Option<&str>,
    ) -> axum::http::Request<axum::body::Body> {
        let mut request = axum::http::Request::post(path)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();
        if let Some(cookie) = cookie {
            request
                .headers_mut()
                .insert(axum::http::header::COOKIE, cookie.parse().unwrap());
        }
        if let Some(bearer) = bearer {
            request.headers_mut().insert(
                axum::http::header::AUTHORIZATION,
                format!("Bearer {bearer}").parse().unwrap(),
            );
        }
        request
    }

    fn get(path: &str) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::get(path)
            .body(axum::body::Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn live_routes_never_expose_unauthenticated_registration_or_intake() {
        use tower::ServiceExt;

        let (state, cors) = oauth_state();
        let app = super::super::create_app(Arc::clone(&state), &cors);

        for path in [
            "/api/identity/register",
            "/api/identity/intake",
            "/api/identity/register/manual",
        ] {
            let response = app
                .clone()
                .oneshot(post(path, r#"{"handle":"alice"}"#, None, None))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{path} must authenticate before its handler"
            );
        }

        let session_id = state
            .sessions
            .create("did:web:alice.example".to_owned(), "local".to_owned())
            .await;
        let response = app
            .oneshot(post(
                "/api/identity/register",
                r#"{"handle":"alice"}"#,
                Some(format!("{}={session_id}", session::SESSION_COOKIE_NAME)),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "an authenticated route with no injected minter must fail closed"
        );
    }

    #[tokio::test]
    async fn public_resolve_route_is_floor_readable_for_plc_and_hosted_local_dids() {
        use tower::ServiceExt;

        for did in [PUBLIC_PLC_DID, LOCAL_HOSTED_DID] {
            let fixture = production_router_fixture();
            let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
                .oneshot(get(&format!("/api/identity/resolve?did={did}")))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "{did} must resolve without authentication"
            );
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                value,
                serde_json::json!({
                    "quicUrl": "https://pds.example:4433",
                    "certHash": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
                })
            );
            assert_eq!(value.as_object().unwrap().len(), 2);
        }
    }

    #[tokio::test]
    async fn public_resolve_route_collapses_sentinel_external_and_missing_to_not_found() {
        use tower::ServiceExt;

        for did in [
            UNAUTHENTICATED_DID_SENTINEL,
            "did:web:foreign.example",
            "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa",
        ] {
            let fixture = production_router_fixture();
            let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
                .oneshot(get(&format!("/api/identity/resolve?did={did}")))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
                serde_json::json!({
                    "error": "not_found",
                    "error_description": "Identity resolution not found",
                })
            );
        }
    }

    #[tokio::test]
    async fn production_composition_live_session_register_mints_and_publishes() {
        use tower::ServiceExt;

        let fixture = production_router_fixture();
        let session_id = fixture
            .state
            .sessions
            .create("did:web:member.example".to_owned(), "local".to_owned())
            .await;
        let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(post(
                "/api/identity/register",
                r#"{"handle":"alice"}"#,
                Some(format!("{}={session_id}", session::SESSION_COOKIE_NAME)),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 5);
        assert_eq!(value["handle"], "at://alice.accounts.example.com");
        assert_eq!(value["did"], "did:web:alice.accounts.example.com");
        assert_eq!(value["pdsEndpoint"], "https://pds.example.test");
        assert_eq!(value["quicUrl"], "https://pds.example.test:4433");
        assert!(value["certHash"]
            .as_str()
            .is_some_and(|hash| !hash.is_empty()));
        let published = fixture
            .storage
            .path()
            .join("pds/accounts.example.com/accounts/alice");
        assert!(published.join("account-record.cbor").is_file());
        assert!(published.join("genesis.didop.cbor").is_file());
        assert!(published.join("did-document.json").is_file());
        assert!(published.join("repo/commit.cbor").is_file());
    }

    #[test]
    fn production_cold_signup_mint_resumes_published_genesis_for_exact_transaction() {
        let fixture = production_router_fixture();
        let api = fixture.state.identity_registration_api.as_ref().unwrap();

        let first = api
            .mint_for_oauth_signup("alice", "verified-dpop-jkt", "SHA256:vault-key")
            .unwrap();
        let second = api
            .mint_for_oauth_signup("alice", "verified-dpop-jkt", "SHA256:vault-key")
            .unwrap();

        assert_eq!(second, first);
        assert!(matches!(
            api.mint_for_oauth_signup("alice", "verified-dpop-jkt", "SHA256:different-key"),
            Err(IdentityApiError::HandleUnavailable)
        ));
        let account = fixture
            .storage
            .path()
            .join("pds/accounts.example.com/accounts/alice");
        assert!(account.join("account-record.cbor").is_file());
        assert!(account.join("repo/commit.cbor").is_file());
        let transaction: serde_json::Value =
            serde_json::from_slice(&std::fs::read(account.join(SIGNUP_TRANSACTION_FILE)).unwrap())
                .unwrap();
        assert_eq!(transaction["transaction_id"], "SHA256:vault-key");
    }

    #[tokio::test]
    async fn minted_hosted_account_is_browser_resolvable_from_authority_storage() {
        use tower::ServiceExt;

        let mut fixture = production_router_fixture();
        let session_id = fixture
            .state
            .sessions
            .create("did:web:member.example".to_owned(), "local".to_owned())
            .await;
        let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(post(
                "/api/identity/register",
                r#"{"handle":"alice"}"#,
                Some(format!("{}={session_id}", session::SESSION_COOKIE_NAME)),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let identity_resolver = Arc::new(HostedConnectTimeResolver {
            pds_root: fixture.storage.path().join("pds"),
            zone: AccountZone::new("accounts.example.com").unwrap(),
            live_discovery: Arc::new(ConfiguredHostedPdsDiscovery {
                pds_endpoint: "https://pds.example.test".to_owned(),
                quic: fixture.quic.clone(),
            }),
        });
        assert!(!identity_resolver.recognizes("did:web:foreign.example"));
        let api = IdentityRegistrationApi::new(
            Arc::new(FakeMint {
                calls: AtomicUsize::new(0),
            }),
            Arc::new(FakeSigner),
            Arc::new(CountingIntake {
                calls: AtomicUsize::new(0),
            }),
            identity_resolver,
            DidWebOriginAllowlist::new(["https://foreign.example"]).unwrap(),
            Arc::new(RateLimiter::new(10, 60)),
        );
        Arc::get_mut(&mut fixture.state)
            .unwrap()
            .identity_registration_api = Some(Arc::new(api));

        let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(get(
                "/api/identity/resolve?did=did:web:alice.accounts.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let resolved: ConnectTimeDiscovery = serde_json::from_slice(&body).unwrap();
        assert_eq!(resolved.quic_url(), "https://pds.example.test:4433");
        assert!(!resolved.cert_hash().is_empty());
    }

    #[tokio::test]
    async fn production_composition_live_global_manual_register_mints() {
        use tower::ServiceExt;

        let fixture = production_router_fixture();
        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new(
            "service:identity-operator".to_owned(),
            now,
            now + 60,
        )
        .with_issuer("https://pds.example.test".to_owned())
        .with_audience(Some("https://pds.example.test".to_owned()))
        .with_client_id("hyprstream-oauth-client-1");
        let token = hyprstream_rpc::auth::jwt::encode(&claims, &fixture.signing_key);
        let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(post(
                "/api/identity/register/manual",
                r#"{"handle":"operator"}"#,
                None,
                Some(&token),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["did"], "did:web:operator.accounts.example.com");
        assert!(fixture
            .storage
            .path()
            .join("pds/accounts.example.com/accounts/operator/account-record.cbor")
            .is_file());
    }

    #[tokio::test]
    async fn production_composition_live_allowlisted_intake_indexes_identity() {
        use tower::ServiceExt;

        let fixture = production_router_fixture();
        let session_id = fixture
            .state
            .sessions
            .create("did:web:member.example".to_owned(), "local".to_owned())
            .await;
        let response = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(post(
                "/api/identity/intake",
                r#"{"did":"did:web:foreign.example"}"#,
                Some(format!("{}={session_id}", session::SESSION_COOKIE_NAME)),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["did"], "did:web:foreign.example");
        assert_eq!(value["handle"], "foreign.example");
        assert_eq!(value["kind"], "federated");
        assert!(value["tenant"].is_null());
        assert_eq!(value["pdsEndpoint"], "https://foreign.example");
    }
}
