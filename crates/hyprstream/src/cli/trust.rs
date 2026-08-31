//! Deployment trust-authority minting.
//!
//! The root authority is an out-of-band, age-encrypted operator asset. Hourly
//! registry credentials should be minted by a narrowly scoped delegated signer;
//! direct root signing exists only for bootstrap and recovery.

#![allow(clippy::print_stdout)]

use crate::auth::age_seal::{AgeIdentities, AgeIdentitySource, AgeRecipients};
use crate::cli::commands::{
    DelegateRegistrySignerArgs, InstallDeploymentTrustArgs, MintAnchorCapsuleArgs,
    MintDeploymentCaArgs, MintRegistryJwtArgs, RotateAuthorityArgs, TrustCommand,
    VerifyDeploymentArgs,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine as _,
};
use ed25519_dalek::{pkcs8::EncodePrivateKey as _, Signer as _, SigningKey, VerifyingKey};
use hyprstream_discovery::did_op::{
    verify_did_op_log, DidOp, HybridDidOpSignature, HybridRotationKey, DID_OP_SIGNATURE_CONTEXT,
};
use hyprstream_discovery::{
    DeploymentAuthorityCheckpoint as AuthorityCheckpointFile,
    DeploymentAuthorityLog as AuthorityLogFile, RegistryDelegationArtifact as DelegationArtifact,
};
use hyprstream_pds::at9p::{
    CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
};
use hyprstream_pds::at9p_sign::{sign_capsule_detached, CapsuleEd25519Signer};
use hyprstream_rpc::transport::QuicServerAuth;
use hyprstream_rpc::{
    auth::ucan::{
        validate as validate_ucan, Ability, Capability, CaveatValue, Caveats, Did, Resource, Ucan,
        UcanError, UcanPayload, UcanVerifier,
    },
    crypto::{
        cose_sign::{assemble_composite_nested, inner_tbs, outer_tbs},
        pq::{
            ml_dsa_generate_keypair, ml_dsa_sign, ml_dsa_sk_from_seed, ml_dsa_sk_to_seed,
            ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey,
        },
    },
};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::OpenOptions,
    io::{Read as _, Write as _},
    os::fd::{FromRawFd as _, RawFd},
    os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use zeroize::{Zeroize as _, Zeroizing};

const PUBLIC_CA_BYTES: usize = 32 + 1_952;
const HYBRID_SIGNATURE_BYTES: usize = 3_309 + 64;
const REGISTRY_AUDIENCE: &str = "urn:hyprstream:service:registry";
const REGISTRY_PROFILE: &str = "hyprstream.registry-deployment.v1";
const AUTHORITY_BUNDLE_SCHEMA: &str = "hyprstream.deployment-authority.v1";
const AUTHORITY_LOG_SCHEMA: &str = "hyprstream.deployment-authority-log.v1";
const AUTHORITY_CHECKPOINT_SCHEMA: &str = "hyprstream.deployment-authority-checkpoint.v1";
const DELEGATION_SCHEMA: &str = "hyprstream.registry-delegation.v1";
const PUBLISHER_MANIFEST_SCHEMA: &str = "hyprstream.deployment-trust-publisher-manifest.v1";
/// Capsule service id the DID-anchored resolver requires for deployment reach.
const ANCHOR_REACH_SERVICE: &str = "#ns";
const DELEGATION_RESOURCE_PREFIX: &str = "hyprstream://deployment";
const DELEGATION_ABILITY: &str = "mint-registry-jwt";
const MAX_AUTHORITY_LOG_OPERATIONS: usize = 128;
const MAX_DELEGATION_BYTES: usize = 256 * 1024;
const MAX_CLOUD_SECRET_BYTES: usize = 64 * 1024;
const MAX_AGE_CIPHERTEXT_BYTES: usize = 256 * 1024;
const MAX_AGE_IDENTITY_BYTES: usize = 4 * 1024;
const PUBLIC_CA_INSTALL_PATH: &str = "/etc/hyprstream/trust/deployment-ca.hybrid";
const AUTHORITY_LOG_INSTALL_PATH: &str = "/etc/hyprstream/trust/deployment-authority.log.json";
const AUTHORITY_CHECKPOINT_INSTALL_PATH: &str =
    "/etc/hyprstream/trust/deployment-authority.head.json";
const REGISTRY_JWT_INSTALL_PATH: &str = "/run/hyprstream/credentials/registry-service.jwt";
const DEPLOYMENT_TRUST_DIR: &str = "/etc/hyprstream/trust";
const DELEGATED_DIR: &str = "/etc/hyprstream/trust/delegated";
const DELEGATED_SIGNER_INSTALL_PATH: &str =
    "/etc/hyprstream/trust/delegated/registry-delegated-signer.age";
const DELEGATION_INSTALL_PATH: &str =
    "/etc/hyprstream/trust/delegated/registry-signer.delegation.json";
const REGISTRY_PUBLIC_KEY_INSTALL_PATH: &str =
    "/etc/hyprstream/trust/delegated/registry-public-key";
const REFRESH_IDENTITY_INSTALL_PATH: &str = "/etc/hyprstream/trust/delegated/refresh-identity";
const TRUST_REFRESH_SERVICE_UNIT_PATH: &str =
    "/etc/systemd/system/hyprstream-trust-refresh.service";
const TRUST_REFRESH_TIMER_UNIT_PATH: &str = "/etc/systemd/system/hyprstream-trust-refresh.timer";
const CREDENTIALS_RUN_DIR: &str = "/run/hyprstream/credentials";

#[cfg(test)]
static SECRET_BUNDLE_DROPS: [std::sync::atomic::AtomicUsize; 3] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum AuthorityPurpose {
    Root,
    RotatedAuthority,
    RegistryDelegatedSigner,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Ed25519Secret {
    Software {
        seed_b64: String,
        public_b64: String,
    },
    YubikeyPiv {
        slot: String,
        public_b64: String,
        recovery_seed_b64: String,
    },
}

impl Drop for Ed25519Secret {
    fn drop(&mut self) {
        match self {
            Self::Software { seed_b64, .. } => seed_b64.zeroize(),
            Self::YubikeyPiv {
                recovery_seed_b64, ..
            } => recovery_seed_b64.zeroize(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityBundle {
    schema: String,
    purpose: AuthorityPurpose,
    deployment_domain: String,
    public_key_sha256: String,
    ed25519: Ed25519Secret,
    ml_dsa_65_seed_b64: String,
    recipient_count: usize,
}

impl Drop for AuthorityBundle {
    fn drop(&mut self) {
        self.ml_dsa_65_seed_b64.zeroize();
        #[cfg(test)]
        SECRET_BUNDLE_DROPS[match &self.purpose {
            AuthorityPurpose::Root => 0,
            AuthorityPurpose::RotatedAuthority => 1,
            AuthorityPurpose::RegistryDelegatedSigner => 2,
        }]
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Debug, Serialize)]
struct CaMintOutput {
    schema: &'static str,
    deployment_domain: String,
    authority_log_did: String,
    authority_log_sequence: u64,
    authority_log_head_cid: String,
    authority_log_path: String,
    authority_log_install_path: &'static str,
    authority_checkpoint_path: String,
    authority_checkpoint_install_path: &'static str,
    public_ca_path: String,
    public_ca_install_path: &'static str,
    public_ca_bytes: usize,
    public_ca_base64: String,
    public_ca_sha256: String,
    authority_key_path: String,
    authority_key_export_allowed: bool,
    recipient_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublisherArtifact {
    local_path: String,
    install_path: String,
    encoding: String,
    base64: String,
    sha256: String,
    size_bytes: usize,
    cloud_secret_store: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublisherManifest {
    schema: String,
    deployment_domain: String,
    profile: String,
    audience: String,
    expires_at: i64,
    public_ca: PublisherArtifact,
    authority_log: PublisherArtifact,
    authority_checkpoint: PublisherArtifact,
    registry_jwt: PublisherArtifact,
    private_authority_exported: bool,
    terraform_state_may_contain_private_authority: bool,
}

enum LoadedEdSigner {
    Software(SigningKey),
    YubikeyPiv { slot: String, public: VerifyingKey },
}

impl LoadedEdSigner {
    fn verifying_key(&self) -> VerifyingKey {
        match self {
            Self::Software(key) => key.verifying_key(),
            Self::YubikeyPiv { public, .. } => *public,
        }
    }

    fn sign(&self, message: &[u8]) -> Result<[u8; 64]> {
        match self {
            Self::Software(key) => Ok(key.sign(message).to_bytes()),
            Self::YubikeyPiv { slot, public } => piv_sign(slot, public, message),
        }
    }
}

impl CapsuleEd25519Signer for LoadedEdSigner {
    fn verifying_key(&self) -> VerifyingKey {
        Self::verifying_key(self)
    }

    fn sign_detached(&self, tbs: &[u8]) -> Result<[u8; 64]> {
        self.sign(tbs)
    }
}

struct LoadedAuthority {
    bundle: AuthorityBundle,
    ed: LoadedEdSigner,
    pq: MlDsaSigningKey,
}

impl LoadedAuthority {
    fn public_bytes(&self) -> Vec<u8> {
        public_pair_bytes(&self.ed.verifying_key(), &self.pq)
    }
}

/// Dispatch one early-startup trust command.
pub fn handle_trust_command(command: TrustCommand) -> Result<()> {
    match command {
        TrustCommand::MintDeploymentCa(args) => mint_deployment_ca(&args),
        TrustCommand::DelegateRegistrySigner(args) => delegate_registry_signer(&args),
        TrustCommand::MintRegistryJwt(args) => mint_registry_jwt(&args),
        TrustCommand::VerifyDeployment(args) => verify_deployment(&args),
        TrustCommand::RotateAuthority(args) => rotate_authority(&args),
        TrustCommand::Install(args) => install_deployment_trust(&args),
        TrustCommand::MintAnchorCapsule(args) => mint_anchor_capsule(&args),
    }
}

fn mint_deployment_ca(args: &MintDeploymentCaArgs) -> Result<()> {
    preflight_outputs(
        [
            &args.public_ca,
            &args.authority_key,
            &args.authority_log,
            &args.authority_checkpoint,
        ],
        args.force,
    )?;
    let recipients = root_recipient_ring(args)?;

    let (ed_secret, ed_signer) = match args.piv_slot.as_deref() {
        Some(slot) => {
            // Destructive overwrite guard: ykman piv keys import silently
            // replaces whatever is in the slot. Check first and refuse unless
            // --force was given, so a daily-driver YubiKey's existing key
            // (SSH, PIV cert, age identity) is not destroyed behind a
            // routine-looking touch prompt.
            if !args.force && piv_slot_occupied(slot)? {
                anyhow::bail!(
                    "PIV slot {slot} already contains a key or certificate. \
                     Re-running will IRREVERSIBLY overwrite it. Pass --force to \
                     confirm the overwrite."
                );
            }
            if args.force {
                println!(
                    "  WARNING: --force bypasses the PIV slot occupancy check. \
                     Any existing key in slot {slot} will be IRREVERSIBLY destroyed."
                );
            }
            let recovery_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let (slot, public) = piv_import_ed25519(slot, &recovery_key)?;
            (
                Ed25519Secret::YubikeyPiv {
                    slot: slot.clone(),
                    public_b64: STANDARD.encode(public.as_bytes()),
                    recovery_seed_b64: encode_ed25519_seed_b64(&recovery_key),
                },
                LoadedEdSigner::YubikeyPiv { slot, public },
            )
        }
        None => {
            let key = SigningKey::generate(&mut rand::rngs::OsRng);
            (
                Ed25519Secret::Software {
                    seed_b64: encode_ed25519_seed_b64(&key),
                    public_b64: STANDARD.encode(key.verifying_key().as_bytes()),
                },
                LoadedEdSigner::Software(key),
            )
        }
    };
    let (pq, _) = ml_dsa_generate_keypair();
    let public_ca = public_pair_bytes(&ed_signer.verifying_key(), &pq);
    ensure!(
        public_ca.len() == PUBLIC_CA_BYTES,
        "internal public-root layout error"
    );
    let deployment_domain = hyprstream_discovery::verify_deployment_public_ca(&public_ca)
        .context("production HybridDeploymentCa rejected generated public root")?;

    let root_key = HybridRotationKey::new(
        ed_signer.verifying_key().to_bytes(),
        ml_dsa_sk_to_vk_bytes(&pq),
    )?;
    let genesis = sign_did_op(
        DidOp {
            sequence: 0,
            prev: None,
            rotation_keys: vec![root_key],
            signature: placeholder_did_signature(),
        },
        &ed_signer,
        &pq,
    )?;
    let authority_log = authority_log_from_ops(&deployment_domain, vec![genesis])?;
    let verified_log = validate_authority_log_root(&public_ca, &authority_log)?;
    let checkpoint = authority_checkpoint(&deployment_domain, &verified_log);
    validate_authority_log(&public_ca, &authority_log, &checkpoint)?;

    let bundle = AuthorityBundle {
        schema: AUTHORITY_BUNDLE_SCHEMA.to_owned(),
        purpose: AuthorityPurpose::Root,
        deployment_domain: deployment_domain.clone(),
        public_key_sha256: sha256_hex(&public_ca),
        ed25519: ed_secret,
        ml_dsa_65_seed_b64: encode_ml_dsa_seed_b64(&pq),
        recipient_count: recipients.len(),
    };
    let plaintext = serialize_secret_json(&bundle).context("encode authority bundle")?;
    let encrypted = encrypt_age(&plaintext, &recipients)?;

    commit_outputs(vec![
        PendingOutput::new(&args.public_ca, public_ca.clone(), 0o644),
        PendingOutput::new(&args.authority_key, encrypted, 0o600),
        PendingOutput::new(
            &args.authority_log,
            pretty_json_bytes(&authority_log)?,
            0o644,
        ),
        PendingOutput::new(
            &args.authority_checkpoint,
            pretty_json_bytes(&checkpoint)?,
            0o644,
        ),
    ])?;

    let output = CaMintOutput {
        schema: "hyprstream.deployment-ca-mint-output.v1",
        deployment_domain,
        authority_log_did: authority_log.did,
        authority_log_sequence: checkpoint.sequence,
        authority_log_head_cid: checkpoint.head_cid,
        authority_log_path: display_path(&args.authority_log),
        authority_log_install_path: AUTHORITY_LOG_INSTALL_PATH,
        authority_checkpoint_path: display_path(&args.authority_checkpoint),
        authority_checkpoint_install_path: AUTHORITY_CHECKPOINT_INSTALL_PATH,
        public_ca_path: display_path(&args.public_ca),
        public_ca_install_path: PUBLIC_CA_INSTALL_PATH,
        public_ca_bytes: public_ca.len(),
        public_ca_base64: STANDARD.encode(&public_ca),
        public_ca_sha256: sha256_hex(&public_ca),
        authority_key_path: display_path(&args.authority_key),
        authority_key_export_allowed: false,
        recipient_count: recipients.len(),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn delegate_registry_signer(args: &DelegateRegistrySignerArgs) -> Result<()> {
    preflight_outputs([&args.delegated_key, &args.delegation], args.force)?;
    ensure!(
        !args.signer_recipients.is_empty(),
        "at least one --signer-recipient is required for the rotatable online signer"
    );
    let signer_recipients = distinct_recipients(args.signer_recipients.clone())?;
    let public_ca = read_limited(&args.public_ca, PUBLIC_CA_BYTES)?;
    let log: AuthorityLogFile = read_json_limited(&args.authority_log, MAX_CLOUD_SECRET_BYTES)?;
    let checkpoint: AuthorityCheckpointFile =
        read_json_limited(&args.authority_checkpoint, MAX_CLOUD_SECRET_BYTES)?;
    let active = validate_authority_log(&public_ca, &log, &checkpoint)?;

    let identities = combined_identities(&args.identities, &args.yubikey_identities)?;
    let authority = decrypt_authority(&args.authority_key, &identities, args.software_recovery)?;
    ensure!(
        authority.bundle.purpose != AuthorityPurpose::RegistryDelegatedSigner,
        "a delegated signer cannot delegate another registry signer"
    );
    ensure_active_authority(&authority, &active)?;

    let delegated_ed = SigningKey::generate(&mut rand::rngs::OsRng);
    let (delegated_pq, _) = ml_dsa_generate_keypair();
    let delegated_public = public_pair_bytes(&delegated_ed.verifying_key(), &delegated_pq);

    let now = now_unix_u64()?;
    let expiration = now
        .checked_add(args.delegation_ttl_seconds)
        .ok_or_else(|| anyhow!("delegation expiration overflow"))?;
    let capability =
        registry_mint_capability(&authority.bundle.deployment_domain, &delegated_public);
    let payload = UcanPayload {
        issuer: Did::from_ed25519(&authority.ed.verifying_key().to_bytes()),
        audience: Did::from_ed25519(&delegated_ed.verifying_key().to_bytes()),
        capabilities: vec![capability],
        not_before: Some(now),
        expiration: Some(expiration),
        nonce: random_bytes(16),
    };
    let ucan = sign_ucan(payload, &authority.ed, &authority.pq)?;
    validate_registry_delegation_ucan(
        &ucan,
        &authority.bundle.deployment_domain,
        &delegated_public,
        &active.rotation_keys,
        now,
    )?;

    let delegated_bundle = AuthorityBundle {
        schema: AUTHORITY_BUNDLE_SCHEMA.to_owned(),
        purpose: AuthorityPurpose::RegistryDelegatedSigner,
        deployment_domain: authority.bundle.deployment_domain.clone(),
        public_key_sha256: sha256_hex(&delegated_public),
        ed25519: Ed25519Secret::Software {
            seed_b64: encode_ed25519_seed_b64(&delegated_ed),
            public_b64: STANDARD.encode(delegated_ed.verifying_key().as_bytes()),
        },
        ml_dsa_65_seed_b64: encode_ml_dsa_seed_b64(&delegated_pq),
        recipient_count: signer_recipients.len(),
    };
    let plaintext = serialize_secret_json(&delegated_bundle)?;
    let encrypted = encrypt_age(&plaintext, &signer_recipients)?;

    let artifact = DelegationArtifact {
        schema: DELEGATION_SCHEMA.to_owned(),
        deployment_domain: authority.bundle.deployment_domain.clone(),
        authority_log_did: log.did.clone(),
        delegated_public_key_b64: STANDARD.encode(&delegated_public),
        ucan_b64: STANDARD.encode(ucan.to_cbor()?),
    };
    validate_delegation_artifact(&public_ca, &log, &checkpoint, &artifact, now)?;
    commit_outputs(vec![
        PendingOutput::new(&args.delegated_key, encrypted, 0o600),
        PendingOutput::new(&args.delegation, pretty_json_bytes(&artifact)?, 0o644),
    ])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hyprstream.registry-delegation-output.v1",
            "deployment_domain": artifact.deployment_domain,
            "delegated_key_path": display_path(&args.delegated_key),
            "delegation_path": display_path(&args.delegation),
            "delegation_expires_at": expiration,
            "scope": {
                "sub": "service:registry",
                "aud": REGISTRY_AUDIENCE,
                "profile": REGISTRY_PROFILE,
                "max_ttl_seconds": 3600
            }
        }))?
    );
    Ok(())
}

fn mint_registry_jwt(args: &MintRegistryJwtArgs) -> Result<()> {
    preflight_outputs([&args.jwt, &args.contract], args.force)?;
    let public_ca = read_limited(&args.public_ca, PUBLIC_CA_BYTES)?;
    let root_domain = hyprstream_discovery::verify_deployment_public_ca(&public_ca)?;
    let registry_key: [u8; 32] = read_limited(&args.registry_public_key, 32)?
        .try_into()
        .map_err(|_| anyhow!("registry public key must be exactly 32 bytes"))?;
    VerifyingKey::from_bytes(&registry_key).context("invalid registry Ed25519 public key")?;
    let identities = mint_identities(args)?;
    let authority_log_bytes = read_limited(&args.authority_log, MAX_CLOUD_SECRET_BYTES)?;
    let installed_log: AuthorityLogFile =
        serde_json::from_slice(&authority_log_bytes).context("decode installed authority log")?;
    let authority_checkpoint_bytes =
        read_limited(&args.authority_checkpoint, MAX_CLOUD_SECRET_BYTES)?;
    let installed_checkpoint: AuthorityCheckpointFile =
        serde_json::from_slice(&authority_checkpoint_bytes)
            .context("decode installed authority checkpoint")?;
    let active = validate_authority_log(&public_ca, &installed_log, &installed_checkpoint)?;

    let (signer, delegation_b64) = if args.root {
        let authority =
            decrypt_authority(&args.authority_key, &identities, args.software_recovery)?;
        ensure!(
            authority.bundle.purpose == AuthorityPurpose::Root,
            "--root accepts only the original root authority bundle"
        );
        ensure!(
            authority.public_bytes() == public_ca,
            "root authority does not match the pinned public CA"
        );
        ensure_active_authority(&authority, &active)?;
        (authority, None)
    } else {
        let artifact_path = args
            .delegation
            .as_ref()
            .ok_or_else(|| anyhow!("--delegation is required"))?;
        let artifact: DelegationArtifact = read_json_limited(artifact_path, MAX_DELEGATION_BYTES)?;
        validate_delegation_artifact(
            &public_ca,
            &installed_log,
            &installed_checkpoint,
            &artifact,
            now_unix_u64()?,
        )?;
        let delegated = match (
            args.via_delegated_signer.as_deref(),
            args.via_delegated_signer_fd,
        ) {
            (Some(delegated_path), None) => {
                decrypt_authority(delegated_path, &identities, args.software_recovery)?
            }
            (None, Some(fd)) => {
                let ciphertext = read_fd_limited(fd, MAX_AGE_CIPHERTEXT_BYTES, "delegated signer")?;
                decrypt_authority_ciphertext(&ciphertext, &identities, args.software_recovery)?
            }
            // Clap requires exactly one signer source unless --root; this arm
            // only fires if that invariant is bypassed programmatically.
            _ => bail!("--via-delegated-signer or --via-delegated-signer-fd is required"),
        };
        ensure!(
            delegated.bundle.purpose == AuthorityPurpose::RegistryDelegatedSigner,
            "selected key is not a registry delegated signer"
        );
        let declared = STANDARD
            .decode(&artifact.delegated_public_key_b64)
            .context("decode delegated public key")?;
        ensure!(
            delegated.public_bytes() == declared,
            "delegated private key does not match the root-authorized delegation"
        );
        let artifact_json = serde_json::to_vec(&artifact)?;
        (delegated, Some(URL_SAFE_NO_PAD.encode(artifact_json)))
    };
    ensure!(
        signer.bundle.deployment_domain == root_domain,
        "signer is bound to a different deployment domain"
    );

    let now = chrono::Utc::now().timestamp();
    let exp = now
        .checked_add(i64::from(args.ttl_seconds))
        .ok_or_else(|| anyhow!("credential expiration overflow"))?;
    let token = encode_registry_jwt(&signer, &registry_key, now, exp, delegation_b64.as_deref())?;
    ensure!(
        token.len() <= MAX_CLOUD_SECRET_BYTES,
        "registry JWT exceeds the 64 KiB cloud-secret contract"
    );
    let verified = hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
        &public_ca,
        &authority_log_bytes,
        &authority_checkpoint_bytes,
        &token,
    )
    .context("production deployment verifier rejected minted credential")?;
    ensure!(
        verified.registry_public_key == registry_key && verified.deployment_domain == root_domain,
        "production verification result does not match minted inputs"
    );

    let jwt_bytes = token.as_bytes();
    let contract = PublisherManifest {
        schema: PUBLISHER_MANIFEST_SCHEMA.to_owned(),
        deployment_domain: root_domain,
        profile: REGISTRY_PROFILE.to_owned(),
        audience: REGISTRY_AUDIENCE.to_owned(),
        expires_at: exp,
        public_ca: PublisherArtifact {
            local_path: display_path(&args.public_ca),
            install_path: PUBLIC_CA_INSTALL_PATH.to_owned(),
            encoding: "raw".to_owned(),
            base64: STANDARD.encode(&public_ca),
            sha256: sha256_hex(&public_ca),
            size_bytes: public_ca.len(),
            cloud_secret_store: true,
        },
        authority_log: PublisherArtifact {
            local_path: display_path(&args.authority_log),
            install_path: AUTHORITY_LOG_INSTALL_PATH.to_owned(),
            encoding: "json".to_owned(),
            base64: STANDARD.encode(&authority_log_bytes),
            sha256: sha256_hex(&authority_log_bytes),
            size_bytes: authority_log_bytes.len(),
            cloud_secret_store: true,
        },
        authority_checkpoint: PublisherArtifact {
            local_path: display_path(&args.authority_checkpoint),
            install_path: AUTHORITY_CHECKPOINT_INSTALL_PATH.to_owned(),
            encoding: "json".to_owned(),
            base64: STANDARD.encode(&authority_checkpoint_bytes),
            sha256: sha256_hex(&authority_checkpoint_bytes),
            size_bytes: authority_checkpoint_bytes.len(),
            cloud_secret_store: true,
        },
        registry_jwt: PublisherArtifact {
            local_path: display_path(&args.jwt),
            install_path: REGISTRY_JWT_INSTALL_PATH.to_owned(),
            encoding: "utf8".to_owned(),
            base64: STANDARD.encode(jwt_bytes),
            sha256: sha256_hex(jwt_bytes),
            size_bytes: jwt_bytes.len(),
            cloud_secret_store: true,
        },
        private_authority_exported: false,
        terraform_state_may_contain_private_authority: false,
    };
    commit_outputs(vec![
        PendingOutput::new(&args.jwt, token.into_bytes(), 0o600),
        PendingOutput::new(&args.contract, pretty_json_bytes(&contract)?, 0o600),
    ])?;
    println!("{}", serde_json::to_string_pretty(&contract)?);
    Ok(())
}

fn verify_deployment(args: &VerifyDeploymentArgs) -> Result<()> {
    let public_ca = read_limited(&args.public_ca, PUBLIC_CA_BYTES)?;
    let token_bytes = read_limited(&args.jwt, MAX_CLOUD_SECRET_BYTES)?;
    let token = String::from_utf8(token_bytes.clone()).context("registry JWT is not UTF-8")?;
    ensure!(
        !token.is_empty() && !token.bytes().any(|byte| byte.is_ascii_whitespace()),
        "registry JWT must be compact with no surrounding or embedded whitespace"
    );
    let contract = args
        .contract
        .as_ref()
        .map(|path| read_json_limited::<PublisherManifest>(path, 1024 * 1024))
        .transpose()?;
    let authority_log = read_limited(&args.authority_log, MAX_CLOUD_SECRET_BYTES)?;
    let authority_checkpoint = read_limited(&args.authority_checkpoint, MAX_CLOUD_SECRET_BYTES)?;
    let verified = hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
        &public_ca,
        &authority_log,
        &authority_checkpoint,
        &token,
    )?;
    if let Some(contract) = contract {
        ensure!(
            contract.schema == PUBLISHER_MANIFEST_SCHEMA,
            "unsupported contract schema"
        );
        ensure!(
            contract.deployment_domain == verified.deployment_domain,
            "contract deployment domain does not match authenticated JWT/root"
        );
        ensure!(
            contract.profile == REGISTRY_PROFILE,
            "contract profile does not match the fixed deployment profile"
        );
        ensure!(
            contract.audience == REGISTRY_AUDIENCE,
            "contract audience does not match the fixed registry audience"
        );
        ensure!(
            contract.expires_at == verified.expires_at,
            "contract expiry does not equal the authenticated JWT exp"
        );
        verify_publisher_artifact(
            &contract.public_ca,
            &public_ca,
            &display_path(&args.public_ca),
            PUBLIC_CA_INSTALL_PATH,
            "raw",
        )?;
        verify_publisher_artifact(
            &contract.authority_log,
            &authority_log,
            &display_path(&args.authority_log),
            AUTHORITY_LOG_INSTALL_PATH,
            "json",
        )?;
        verify_publisher_artifact(
            &contract.authority_checkpoint,
            &authority_checkpoint,
            &display_path(&args.authority_checkpoint),
            AUTHORITY_CHECKPOINT_INSTALL_PATH,
            "json",
        )?;
        verify_publisher_artifact(
            &contract.registry_jwt,
            &token_bytes,
            &display_path(&args.jwt),
            REGISTRY_JWT_INSTALL_PATH,
            "utf8",
        )?;
        ensure!(
            !contract.private_authority_exported
                && !contract.terraform_state_may_contain_private_authority,
            "contract permits private authority export"
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "valid": true,
            "deployment_domain": verified.deployment_domain,
            "registry_public_key_base64": STANDARD.encode(verified.registry_public_key),
            "public_ca_bytes": public_ca.len(),
            "profile": REGISTRY_PROFILE,
            "audience": REGISTRY_AUDIENCE
        }))?
    );
    Ok(())
}

fn verify_publisher_artifact(
    artifact: &PublisherArtifact,
    expected_bytes: &[u8],
    expected_local_path: &str,
    expected_install_path: &str,
    expected_encoding: &str,
) -> Result<()> {
    ensure!(
        artifact.local_path == expected_local_path,
        "contract artifact local path does not match the verified input"
    );
    ensure!(
        artifact.install_path == expected_install_path,
        "contract artifact install path is not the fixed deployment path"
    );
    ensure!(
        artifact.encoding == expected_encoding,
        "contract artifact encoding is not the fixed deployment encoding"
    );
    ensure!(
        artifact.cloud_secret_store,
        "contract artifact unexpectedly forbids cloud secret-store publication"
    );
    let decoded = STANDARD
        .decode(&artifact.base64)
        .context("decode contract artifact base64")?;
    ensure!(
        STANDARD.encode(&decoded) == artifact.base64,
        "contract artifact base64 is not canonical"
    );
    ensure!(
        decoded == expected_bytes,
        "contract artifact bytes do not match the verified input"
    );
    ensure!(
        artifact.size_bytes == decoded.len(),
        "contract artifact size does not match authenticated bytes"
    );
    ensure!(
        artifact.sha256 == sha256_hex(&decoded),
        "contract artifact SHA-256 does not match authenticated bytes"
    );
    Ok(())
}

/// Nothing else in the repository installs the OS-owned trust directory or
/// keeps the 1-hour registry credential refreshed — every
/// prior `trust mint-*` command only produces local artifacts under the
/// operator's ceremony directory. This command projects those artifacts onto
/// the fixed paths `hyprstream-discovery::service::read_trusted_artifact`
/// actually reads, and, when delegated-signer material is supplied,
/// additionally installs and enables a systemd timer that keeps the registry
/// credential minted within its TTL.
fn install_deployment_trust(args: &InstallDeploymentTrustArgs) -> Result<()> {
    ensure!(
        nix::unistd::Uid::effective().is_root(),
        "trust install must run as root: the fixed paths under {DEPLOYMENT_TRUST_DIR} must be \
         root-owned (see hyprstream-discovery's read_trusted_artifact), and this command does not \
         attempt to chown files it does not itself own"
    );
    validate_refresh_interval(&args.refresh_interval)?;

    let public_ca = read_limited(&args.public_ca, PUBLIC_CA_BYTES)?;
    ensure!(
        public_ca.len() == PUBLIC_CA_BYTES,
        "public_ca must be exactly {PUBLIC_CA_BYTES} bytes"
    );
    hyprstream_discovery::verify_deployment_public_ca(&public_ca)
        .context("public_ca is not a valid production deployment CA")?;
    let authority_log: AuthorityLogFile =
        read_json_limited(&args.authority_log, MAX_CLOUD_SECRET_BYTES)?;
    let authority_checkpoint: AuthorityCheckpointFile =
        read_json_limited(&args.authority_checkpoint, MAX_CLOUD_SECRET_BYTES)?;
    validate_authority_log(&public_ca, &authority_log, &authority_checkpoint)
        .context("authority log/checkpoint do not verify against public_ca")?;

    ensure_root_owned_dir(Path::new(DEPLOYMENT_TRUST_DIR), 0o755)?;

    let public_ca_dest = PathBuf::from(PUBLIC_CA_INSTALL_PATH);
    let authority_log_dest = PathBuf::from(AUTHORITY_LOG_INSTALL_PATH);
    let authority_checkpoint_dest = PathBuf::from(AUTHORITY_CHECKPOINT_INSTALL_PATH);
    preflight_outputs(
        [
            &public_ca_dest,
            &authority_log_dest,
            &authority_checkpoint_dest,
        ],
        args.force,
    )?;
    commit_outputs(vec![
        PendingOutput::new(&public_ca_dest, public_ca.clone(), 0o644),
        PendingOutput::new(
            &authority_log_dest,
            pretty_json_bytes(&authority_log)?,
            0o644,
        ),
        PendingOutput::new(
            &authority_checkpoint_dest,
            pretty_json_bytes(&authority_checkpoint)?,
            0o644,
        ),
    ])?;

    let mut refresher_enabled = false;
    if let Some(delegated_key) = &args.delegated_key {
        let delegation = args
            .delegation
            .as_ref()
            .ok_or_else(|| anyhow!("--delegation is required with --delegated-key"))?;
        let registry_public_key = args
            .registry_public_key
            .as_ref()
            .ok_or_else(|| anyhow!("--registry-public-key is required with --delegated-key"))?;
        let refresh_identity = args
            .refresh_identity
            .as_ref()
            .ok_or_else(|| anyhow!("--refresh-identity is required with --delegated-key"))?;
        install_trust_refresher(
            args,
            &public_ca,
            &authority_log,
            &authority_checkpoint,
            delegated_key,
            delegation,
            registry_public_key,
            refresh_identity,
        )?;
        refresher_enabled = true;
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hyprstream.deployment-trust-install-output.v1",
            "installed": [
                PUBLIC_CA_INSTALL_PATH,
                AUTHORITY_LOG_INSTALL_PATH,
                AUTHORITY_CHECKPOINT_INSTALL_PATH,
            ],
            "refresher_enabled": refresher_enabled,
        }))?
    );
    Ok(())
}

/// `mkdir -p` with the given mode on the leaf. Every component of the path —
/// not just the leaf — is inspected with `symlink_metadata` and must be a
/// non-symlink directory owned by root (or by the given owner) and writable by
/// nobody else, mirroring the fail-closed posture `read_trusted_artifact`
/// requires of every ancestor of a trust path it reads. Guarantee: at the
/// moment each component was inspected it was a root/owner-owned real
/// directory no other user could write into. Not guaranteed: the
/// checks are lstat-based, not O_NOFOLLOW-handle-based, so a component
/// swapped between inspection and use is not detected (TOCTOU).
fn ensure_root_owned_dir(dir: &Path, mode: u32) -> Result<()> {
    ensure_owned_dir(dir, mode, 0)
}

fn ensure_owned_dir(dir: &Path, mode: u32, owner_uid: u32) -> Result<()> {
    let mut ancestors: Vec<&Path> = dir.ancestors().collect();
    ancestors.reverse();
    for component in ancestors {
        if component.as_os_str().is_empty() {
            continue;
        }
        let is_leaf = component == dir;
        match std::fs::symlink_metadata(component) {
            Ok(metadata) => {
                ensure!(
                    !metadata.file_type().is_symlink(),
                    "refusing to install through a symlinked path component: {}",
                    component.display()
                );
                ensure!(
                    metadata.is_dir(),
                    "install path component exists and is not a directory: {}",
                    component.display()
                );
                let uid = metadata.uid();
                ensure!(
                    uid == 0 || uid == owner_uid,
                    "install path component {} is owned by uid {uid}, not root; refusing to \
                     install trust material under a directory another user controls",
                    component.display()
                );
                let mode = metadata.mode();
                ensure!(
                    mode & 0o022 == 0,
                    "install path component {} is group/world writable (mode {:04o}); refusing \
                     to install trust material under a directory other users can modify",
                    component.display(),
                    mode & 0o7777
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(component)
                    .with_context(|| format!("create directory {}", component.display()))?;
                if !is_leaf {
                    std::fs::set_permissions(component, std::fs::Permissions::from_mode(0o755))
                        .with_context(|| format!("set permissions on {}", component.display()))?;
                }
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect directory {}", component.display()))
            }
        }
        if is_leaf {
            std::fs::set_permissions(component, std::fs::Permissions::from_mode(mode))
                .with_context(|| format!("set permissions on {}", component.display()))?;
        }
    }
    Ok(())
}

/// Fail closed if the `age` binary is not on `PATH`, the refresh identity
/// cannot decrypt the delegated signer ciphertext, the decrypted bundle is not
/// a `RegistryDelegatedSigner`, or its public bytes do not match the
/// root-authorized delegation artifact. Every condition is checked before the
/// installer writes any output or enables the timer, so a misconfigured
/// deployment is reported immediately rather than discovered by the first
/// unattended timer firing.
fn trial_decrypt_delegated_signer(
    delegated_key: &Path,
    refresh_identity: &Path,
    artifact: &DelegationArtifact,
) -> Result<()> {
    // The refresher shells out to the `age` binary; a host without it cannot
    // honor the timer, so fail closed early with a clear message.
    let age_probe = std::process::Command::new("age")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context(
            "the `age` binary is not installed or not executable; the trust refresher requires it",
        )?;
    ensure!(
        age_probe.success(),
        "the `age` binary is not installed or not executable; the trust refresher requires it"
    );

    // Reproduce the exact decrypt path the refresher's ExecStart follows:
    // `age --decrypt --identity <refresh_identity>` on the delegated signer.
    // decrypt_authority reconstructs the full LoadedAuthority and asserts the
    // bundle's own public-key fingerprint, so the public-bytes comparison uses
    // the identical derivation path the refresher will exercise on every timer
    // firing. software_recovery is false because the refresher ExecStart never
    // passes --software-recovery.
    let identities = AgeIdentities::new(vec![refresh_identity.to_path_buf()])?;
    let trial = decrypt_authority(delegated_key, &identities, false)
        .context("refresh identity cannot decrypt the delegated signer ciphertext")?;
    ensure!(
        trial.bundle.purpose == AuthorityPurpose::RegistryDelegatedSigner,
        "decrypted delegated signer is not a RegistryDelegatedSigner; the \
         refresher ExecStart rejects any other purpose"
    );
    let trial_public = trial.public_bytes();
    let declared = STANDARD
        .decode(&artifact.delegated_public_key_b64)
        .context("decode delegated public key from delegation artifact")?;
    ensure!(
        trial_public == declared,
        "decrypted delegated signer public key does not match the root-authorized delegation"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn install_trust_refresher(
    args: &InstallDeploymentTrustArgs,
    public_ca: &[u8],
    authority_log: &AuthorityLogFile,
    authority_checkpoint: &AuthorityCheckpointFile,
    delegated_key: &Path,
    delegation: &Path,
    registry_public_key: &Path,
    refresh_identity: &Path,
) -> Result<()> {
    let delegated_key_bytes = read_limited(delegated_key, MAX_AGE_CIPHERTEXT_BYTES)?;
    let delegation_artifact: DelegationArtifact =
        read_json_limited(delegation, MAX_DELEGATION_BYTES)?;
    // Full cryptographic validation against the just-verified authority log:
    // an expired, tampered, or wrong-deployment delegation must fail install
    // rather than be discovered by the first unattended refresh.
    validate_delegation_artifact(
        public_ca,
        authority_log,
        authority_checkpoint,
        &delegation_artifact,
        now_unix_u64()?,
    )
    .context("delegation artifact does not verify against the installed authority log")?;
    let registry_public_key_bytes = read_limited(registry_public_key, 32)?;
    let registry_public_key_array: [u8; 32] = registry_public_key_bytes
        .clone()
        .try_into()
        .map_err(|_| anyhow!("registry_public_key must be exactly 32 bytes"))?;
    VerifyingKey::from_bytes(&registry_public_key_array)
        .context("invalid registry Ed25519 public key")?;
    let refresh_identity_bytes = read_limited(refresh_identity, MAX_AGE_IDENTITY_BYTES)?;
    validate_age_identity_contents(&refresh_identity_bytes)
        .with_context(|| format!("refresh identity {}", refresh_identity.display()))?;

    // Trial-decrypt the delegated signer with the refresh identity before
    // writing any outputs or enabling the timer. If the identity cannot open
    // the ciphertext, or the decrypted public key does not match the root-
    // authorized delegation, the install must fail closed here rather than
    // silently succeed and then fail on every timer firing.
    trial_decrypt_delegated_signer(delegated_key, refresh_identity, &delegation_artifact)
        .context("trial decrypt of the delegated signer with the refresh identity failed")?;

    ensure_root_owned_dir(Path::new(DELEGATED_DIR), 0o750)?;
    ensure_root_owned_dir(Path::new(CREDENTIALS_RUN_DIR), 0o750)?;

    let delegated_key_dest = PathBuf::from(DELEGATED_SIGNER_INSTALL_PATH);
    let delegation_dest = PathBuf::from(DELEGATION_INSTALL_PATH);
    let registry_public_key_dest = PathBuf::from(REGISTRY_PUBLIC_KEY_INSTALL_PATH);
    let refresh_identity_dest = PathBuf::from(REFRESH_IDENTITY_INSTALL_PATH);
    preflight_outputs(
        [
            &delegated_key_dest,
            &delegation_dest,
            &registry_public_key_dest,
            &refresh_identity_dest,
        ],
        args.force,
    )?;
    commit_outputs(vec![
        PendingOutput::new(&delegated_key_dest, delegated_key_bytes, 0o600),
        PendingOutput::new(
            &delegation_dest,
            pretty_json_bytes(&delegation_artifact)?,
            0o644,
        ),
        PendingOutput::new(&registry_public_key_dest, registry_public_key_bytes, 0o644),
        PendingOutput::new(&refresh_identity_dest, refresh_identity_bytes, 0o600),
    ])?;

    let service_dest = PathBuf::from(TRUST_REFRESH_SERVICE_UNIT_PATH);
    let timer_dest = PathBuf::from(TRUST_REFRESH_TIMER_UNIT_PATH);
    preflight_outputs([&service_dest, &timer_dest], true)?;
    commit_outputs(vec![
        PendingOutput::new(
            &service_dest,
            trust_refresh_service_unit().into_bytes(),
            0o644,
        ),
        PendingOutput::new(
            &timer_dest,
            trust_refresh_timer_unit(&args.refresh_interval).into_bytes(),
            0o644,
        ),
    ])?;

    if !args.no_enable {
        run_systemctl(&["daemon-reload"])?;
        run_systemctl(&["enable", "--now", "hyprstream-trust-refresh.timer"])?;
    }
    Ok(())
}

fn run_systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .args(args)
        .status()
        .with_context(|| format!("run systemctl {}", args.join(" ")))?;
    ensure!(
        status.success(),
        "systemctl {} failed: {status}",
        args.join(" ")
    );
    Ok(())
}

/// Refuse a refresh interval that is not a plain systemd time span of the
/// form `<positive integer><s|min|h>`. The value is interpolated verbatim
/// into the generated timer unit, so anything containing whitespace or
/// control characters could inject unit directives, and a typo would only
/// surface as a late systemd parse failure instead of failing the install.
fn validate_refresh_interval(interval: &str) -> Result<()> {
    let digits_ok = |digits: &str| {
        !digits.is_empty()
            && !digits.starts_with('0')
            && digits.bytes().all(|byte| byte.is_ascii_digit())
    };
    let valid = interval
        .strip_suffix("min")
        .or_else(|| interval.strip_suffix('s'))
        .or_else(|| interval.strip_suffix('h'))
        .is_some_and(digits_ok);
    ensure!(
        valid,
        "invalid --refresh-interval {interval:?}: use a positive integer with an s, min, or h \
         suffix and no whitespace (for example 30min)"
    );
    Ok(())
}

/// The refresh identity is passed by path to `age --decrypt --identity`, which
/// expects a plaintext identity file: optional `#` comment or blank lines plus
/// at least one native X25519 identity line. Encrypted or plugin identities
/// would make the unattended refresher fail at its first timer firing, so
/// they are rejected at install time.
fn validate_age_identity_contents(bytes: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(bytes).context("age identity file is not UTF-8 text")?;
    let mut has_identity = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        ensure!(
            line.starts_with("AGE-SECRET-KEY-1"),
            "age identity file must contain only comment lines and plaintext \
             AGE-SECRET-KEY-1... identity lines"
        );
        has_identity = true;
    }
    ensure!(
        has_identity,
        "age identity file contains no AGE-SECRET-KEY-1... identity line"
    );
    Ok(())
}

/// The refresher only ever needs the delegated (non-root) signer: it must
/// never be able to sign directly with the root authority, matching the
/// least-privilege split the ceremony (docs/deployment-trust-ceremony.md)
/// establishes between minting and day-to-day operation. The installed
/// refresh identity enforces the same split at the encryption layer: it is a
/// `--signer-recipient` of the delegated signer only, so even with the unit's
/// read access to the trust directory it can never open the operator-held
/// root authority bundle, which is sealed to a disjoint recipient ring.
fn trust_refresh_service_unit() -> String {
    // The unit assumes the packaged binary location; the installer does not
    // encode its own (possibly temporary) executable path into the unit.
    format!(
        r#"[Unit]
Description=Hyprstream registry deployment credential refresh
After=network-online.target
Wants=network-online.target
ConditionPathExists={DEPLOYMENT_TRUST_DIR}/deployment-ca.hybrid

[Service]
Type=oneshot
ExecStart=/usr/bin/hyprstream trust mint-registry-jwt \
  --public-ca {PUBLIC_CA_INSTALL_PATH} \
  --authority-log {AUTHORITY_LOG_INSTALL_PATH} \
  --authority-checkpoint {AUTHORITY_CHECKPOINT_INSTALL_PATH} \
  --via-delegated-signer {DELEGATED_SIGNER_INSTALL_PATH} \
  --delegation {DELEGATION_INSTALL_PATH} \
  --registry-public-key {REGISTRY_PUBLIC_KEY_INSTALL_PATH} \
  --identity {REFRESH_IDENTITY_INSTALL_PATH} \
  --jwt {REGISTRY_JWT_INSTALL_PATH} \
  --contract {CREDENTIALS_RUN_DIR}/deployment-trust.contract.json \
  --force
# mint-registry-jwt writes {REGISTRY_JWT_INSTALL_PATH} through the same
# staged-file-then-rename commit path every other trust artifact uses, so a
# concurrent reader never observes a partially-written credential.
# RuntimeDirectory recreates the tmpfs-backed credentials directory after a
# reboot (ReadWritePaths sandbox setup fails if it is missing); Preserve keeps
# the minted credential alive after this oneshot unit exits.
RuntimeDirectory=hyprstream/credentials
RuntimeDirectoryMode=0750
RuntimeDirectoryPreserve=yes
ProtectSystem=strict
ReadWritePaths={CREDENTIALS_RUN_DIR}
ReadOnlyPaths={DEPLOYMENT_TRUST_DIR}
NoNewPrivileges=yes
"#
    )
}

fn trust_refresh_timer_unit(refresh_interval: &str) -> String {
    format!(
        r#"[Unit]
Description=Periodic refresh of the Hyprstream registry deployment credential

[Timer]
OnBootSec=30s
OnUnitActiveSec={refresh_interval}
AccuracySec=1min
Persistent=true

[Install]
WantedBy=timers.target
"#
    )
}

fn rotate_authority(args: &RotateAuthorityArgs) -> Result<()> {
    ensure!(
        args.add ^ args.replace,
        "select exactly one rotation mode: --add or --replace"
    );
    preflight_outputs(
        [
            &args.new_authority_key,
            &args.new_public_key,
            &args.authority_log_out,
            &args.authority_checkpoint_out,
        ],
        args.force,
    )?;
    let recipients = distinct_recipients(args.recipients.clone())?;
    ensure!(
        recipients.len() >= 2,
        "authority rotation requires at least two distinct age recipients"
    );
    let public_ca = read_limited(&args.public_ca, PUBLIC_CA_BYTES)?;
    let log: AuthorityLogFile = read_json_limited(&args.authority_log, MAX_CLOUD_SECRET_BYTES)?;
    let checkpoint: AuthorityCheckpointFile =
        read_json_limited(&args.authority_checkpoint, MAX_CLOUD_SECRET_BYTES)?;
    let active = validate_authority_log(&public_ca, &log, &checkpoint)?;
    let identities = combined_identities(&args.identities, &args.yubikey_identities)?;
    let current = decrypt_authority(&args.authority_key, &identities, args.software_recovery)?;
    ensure_active_authority(&current, &active)?;

    let new_ed = SigningKey::generate(&mut rand::rngs::OsRng);
    let (new_pq, _) = ml_dsa_generate_keypair();
    let new_public = public_pair_bytes(&new_ed.verifying_key(), &new_pq);
    let new_rotation_key = HybridRotationKey::new(
        new_ed.verifying_key().to_bytes(),
        ml_dsa_sk_to_vk_bytes(&new_pq),
    )?;
    let next_keys = if args.add {
        let mut keys = active.rotation_keys.clone();
        ensure!(
            keys.len() < hyprstream_discovery::did_op::MAX_ROTATION_KEYS,
            "authority set is already at its maximum size"
        );
        keys.push(new_rotation_key);
        keys
    } else {
        vec![new_rotation_key]
    };
    let operations = decode_authority_operations(&log)?;
    let previous = operations
        .last()
        .ok_or_else(|| anyhow!("authority log is empty"))?;
    let next = sign_did_op(
        DidOp {
            sequence: previous
                .sequence
                .checked_add(1)
                .ok_or_else(|| anyhow!("authority sequence overflow"))?,
            prev: Some(previous.cid().encode()),
            rotation_keys: next_keys,
            signature: placeholder_did_signature(),
        },
        &current.ed,
        &current.pq,
    )?;
    let mut operations = operations;
    operations.push(next);
    let next_log = authority_log_from_ops(&log.deployment_domain, operations)?;
    let next_verified = validate_authority_log_root(&public_ca, &next_log)?;
    let expected_sequence = checkpoint
        .sequence
        .checked_add(1)
        .ok_or_else(|| anyhow!("authority checkpoint sequence overflow"))?;
    ensure!(
        next_verified.did == checkpoint.did && next_verified.sequence == expected_sequence,
        "authority rotation did not advance the trusted checkpoint exactly once"
    );
    let next_checkpoint = authority_checkpoint(&next_log.deployment_domain, &next_verified);
    validate_authority_log(&public_ca, &next_log, &next_checkpoint)?;

    let new_bundle = AuthorityBundle {
        schema: AUTHORITY_BUNDLE_SCHEMA.to_owned(),
        purpose: AuthorityPurpose::RotatedAuthority,
        deployment_domain: log.deployment_domain.clone(),
        public_key_sha256: sha256_hex(&new_public),
        ed25519: Ed25519Secret::Software {
            seed_b64: encode_ed25519_seed_b64(&new_ed),
            public_b64: STANDARD.encode(new_ed.verifying_key().as_bytes()),
        },
        ml_dsa_65_seed_b64: encode_ml_dsa_seed_b64(&new_pq),
        recipient_count: recipients.len(),
    };
    let plaintext = serialize_secret_json(&new_bundle)?;
    let encrypted = encrypt_age(&plaintext, &recipients)?;

    commit_outputs(vec![
        PendingOutput::new(&args.new_public_key, new_public, 0o644),
        PendingOutput::new(&args.new_authority_key, encrypted, 0o600),
        PendingOutput::new(
            &args.authority_log_out,
            pretty_json_bytes(&next_log)?,
            0o644,
        ),
        PendingOutput::new(
            &args.authority_checkpoint_out,
            pretty_json_bytes(&next_checkpoint)?,
            0o644,
        ),
    ])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hyprstream.authority-rotation-output.v1",
            "deployment_domain": next_log.deployment_domain,
            "authority_log_did": next_log.did,
            "sequence": next_checkpoint.sequence,
            "head_cid": next_checkpoint.head_cid,
            "mode": if args.add { "add" } else { "replace" },
            "new_public_key_path": display_path(&args.new_public_key),
            "new_authority_key_path": display_path(&args.new_authority_key),
            "authority_log_path": display_path(&args.authority_log_out),
            "authority_checkpoint_path": display_path(&args.authority_checkpoint_out)
        }))?
    );
    Ok(())
}

/// Mint the deployment anchor capsule plus the `did:web` document that vouches
/// for it.
///
/// The two outputs are inseparable: the `did:at9p` identifier IS the BLAKE3-512
/// CID of the capsule, and the document must name that exact identifier back.
/// Both are re-verified through the production DID-anchored resolver before
/// anything is written.
fn mint_anchor_capsule(args: &MintAnchorCapsuleArgs) -> Result<()> {
    preflight_outputs([&args.capsule_out, &args.did_json_out], args.force)?;
    ensure!(
        args.did_web.starts_with("did:web:"),
        "--did-web must be a did:web identifier (for example did:web:staging.example.com)"
    );
    let reach = anchor_reach_endpoint(args)?;

    let public_ca = read_limited(&args.public_ca, PUBLIC_CA_BYTES)?;
    let log: AuthorityLogFile = read_json_limited(&args.authority_log, MAX_CLOUD_SECRET_BYTES)?;
    let checkpoint: AuthorityCheckpointFile =
        read_json_limited(&args.authority_checkpoint, MAX_CLOUD_SECRET_BYTES)?;
    let active = validate_authority_log(&public_ca, &log, &checkpoint)?;

    let identities = combined_identities(&args.identities, &args.yubikey_identities)?;
    let authority = decrypt_authority(&args.authority_key, &identities, args.software_recovery)?;
    ensure_anchor_authority(&authority, &public_ca, &active)?;

    let minted = build_anchor_material(args, &reach, &authority)?;
    commit_outputs(vec![
        PendingOutput::new(&args.capsule_out, minted.capsule_bytes.clone(), 0o644),
        PendingOutput::new(
            &args.did_json_out,
            pretty_json_bytes(&minted.document)?,
            0o644,
        ),
    ])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hyprstream.deployment-anchor-capsule-output.v1",
            "deployment_domain": authority.bundle.deployment_domain,
            "cluster_at9p_did": minted.at9p_did,
            "cluster_did_web": args.did_web,
            "capsule_path": display_path(&args.capsule_out),
            "capsule_bytes": minted.capsule_bytes.len(),
            "capsule_sha256": sha256_hex(&minted.capsule_bytes),
            "did_json_path": display_path(&args.did_json_out),
            "publish": {
                "capsule": format!(".well-known/at9p/{}.cbor", minted.cid512),
                "document": ".well-known/did.json",
            },
            "reach": reach.summary,
            "mesh_material_published": args.mesh_pq.is_some(),
        }))?
    );
    Ok(())
}

/// A minted, self-verified anchor pair, before it touches the filesystem.
struct MintedAnchor {
    capsule_bytes: Vec<u8>,
    cid512: String,
    at9p_did: String,
    document: serde_json::Value,
}

/// Build and self-verify the anchor capsule and its document.
///
/// The capsule publishes the deployment authority as its primary subject key —
/// that key IS the deployment CA the resolver installs — and signs itself with
/// that same authority under pinned Hybrid.
fn build_anchor_material(
    args: &MintAnchorCapsuleArgs,
    reach: &AnchorReach,
    authority: &LoadedAuthority,
) -> Result<MintedAnchor> {
    let subject_key = HybridKeyPair::new(
        authority.ed.verifying_key().to_bytes(),
        ml_dsa_sk_to_vk_bytes(&authority.pq),
    )
    .context("deployment authority is not a valid at9p hybrid subject key")?;
    let service = ServiceEntry::new(
        ANCHOR_REACH_SERVICE,
        ServiceType::NinePExport,
        reach.endpoint.clone(),
    )
    .context("build the deployment-reach service entry")?;
    let mut body = CapsuleBody::new(vec![subject_key], vec![service])
        .context("build the anchor capsule body")?;
    body.also_known_as = Some(vec![args.did_web.clone()]);
    let capsule = sign_capsule_detached(body, &authority.ed, &authority.pq)
        .context("hybrid-sign the anchor capsule")?;
    let capsule_bytes = capsule.to_dag_cbor()?;
    let cid512 = capsule.cid512()?;
    let at9p_did = format!("did:at9p:{cid512}");
    let document = anchor_did_document(args, &at9p_did, &authority.ed.verifying_key(), reach)?;

    // Publishing material the production resolver would reject is the one
    // failure this command exists to prevent: check before anything is written.
    let verified =
        verify_anchor_material_offline(&at9p_did, &args.did_web, &document, &capsule_bytes)
            .context(
                "the minted anchor pair was rejected by the production DID-anchored verifier; \
                 nothing was written",
            )?;
    ensure!(
        verified.deployment_ca_public == authority.public_bytes(),
        "verified anchor capsule yields a different deployment CA than the signing authority"
    );

    Ok(MintedAnchor {
        capsule_bytes,
        cid512,
        at9p_did,
        document,
    })
}

/// The capsule's deployment-reach entry plus the document-side mechanics that
/// must describe the same endpoint.
struct AnchorReach {
    endpoint: ServiceEndpoint,
    /// Optional `did:web` transport service entry contributing channel
    /// mechanics (SNI, WebPKI policy, cert pins) for the capsule-bound socket.
    document_service: Option<serde_json::Value>,
    summary: serde_json::Value,
}

/// Build the `#ns` endpoint from the operator-supplied anchor-node reach.
fn anchor_reach_endpoint(args: &MintAnchorCapsuleArgs) -> Result<AnchorReach> {
    match (args.iroh_node_id.as_deref(), args.quic_endpoint) {
        (Some(node_id), None) => {
            let node_id = node_id.strip_prefix("did:key:").unwrap_or(node_id);
            hyprstream_rpc::did_key::decode_ed25519_multikey(node_id)
                .context("--iroh-node-id must be an Ed25519 Multikey (z6Mk...)")?;
            let mut endpoint = ServiceEndpoint::new(Transport::Iroh, format!("iroh://{node_id}"))
                .context("build the iroh deployment-reach endpoint")?;
            endpoint.node_id = Some(node_id.to_owned());
            endpoint.relay = args.iroh_relay.clone();
            Ok(AnchorReach {
                endpoint,
                document_service: None,
                summary: serde_json::json!({
                    "transport": "iroh",
                    "node_id": node_id,
                    "relay": args.iroh_relay,
                }),
            })
        }
        (None, Some(address)) => {
            let endpoint = ServiceEndpoint::new(Transport::Quic, format!("quic://{address}"))
                .context("build the QUIC deployment-reach endpoint")?;
            let pins = args
                .quic_cert_sha256
                .iter()
                .map(|pin| {
                    let raw = hex::decode(pin.trim())
                        .with_context(|| format!("--quic-cert-sha256 {pin} is not hex"))?;
                    let raw: [u8; 32] = raw.try_into().map_err(|_| {
                        anyhow!("--quic-cert-sha256 {pin} is not a 32-byte SHA-256 digest")
                    })?;
                    Ok(raw)
                })
                .collect::<Result<Vec<_>>>()?;
            let sni = args
                .quic_sni
                .clone()
                .unwrap_or_else(|| address.ip().to_string());
            let auth = if pins.is_empty() {
                ensure!(
                    args.quic_web_pki,
                    "a QUIC anchor reach needs channel mechanics: pass --quic-cert-sha256 \
                     for a pinned leaf certificate, or --quic-web-pki with --quic-sni for \
                     a WebPKI-terminated endpoint"
                );
                QuicServerAuth::web_pki()
            } else if args.quic_web_pki {
                QuicServerAuth::web_pki_pinned(pins)?
            } else {
                QuicServerAuth::pinned(pins)?
            };
            let document_service = Some(serde_json::json!({
                "id": format!("{}#quic", args.did_web),
                "type": "QuicTransport",
                "serviceEndpoint": hyprstream_rpc::service_entry::encode_quic(
                    &format!("https://{sni}:{}", address.port()),
                    &auth,
                    &["hyprstream-rpc/1"],
                ),
            }));
            Ok(AnchorReach {
                endpoint,
                document_service,
                summary: serde_json::json!({
                    "transport": "quic",
                    "address": address.to_string(),
                    "sni": sni,
                    "cert_pins": args.quic_cert_sha256.len(),
                    "web_pki": args.quic_web_pki,
                }),
            })
        }
        _ => bail!(
            "exactly one anchor reach is required: pass --iroh-node-id <MULTIKEY> or \
             --quic-endpoint <IP:PORT>"
        ),
    }
}

/// Render the deployment `did:web` document that reciprocally vouches for the
/// anchor capsule.
///
/// The reciprocal `alsoKnownAs` is what the resolver checks; the `#mesh-kem`
/// keyAgreement and the single `#mesh-pq` verification method are what a
/// remote-node bootstrap additionally requires, and they belong to the
/// Discovery service the capsule's reach points at — not to the CA.
fn anchor_did_document(
    args: &MintAnchorCapsuleArgs,
    at9p_did: &str,
    ca_ed: &VerifyingKey,
    reach: &AnchorReach,
) -> Result<serde_json::Value> {
    let did_web = &args.did_web;
    let mut verification_method = vec![serde_json::json!({
        "id": format!("{did_web}#deployment-ca"),
        "type": "Multikey",
        "controller": did_web,
        "publicKeyMultibase": hyprstream_rpc::did_key::ed25519_to_did_key(ca_ed.as_bytes())
            .strip_prefix("did:key:")
            .unwrap_or_default()
            .to_owned(),
    })];
    let mut key_agreement = Vec::new();
    if let (Some(pq), Some(x25519), Some(mlkem)) = (
        args.mesh_pq.as_deref(),
        args.mesh_kem_x25519.as_deref(),
        args.mesh_kem_mlkem768.as_deref(),
    ) {
        verification_method.push(serde_json::json!({
            "id": format!("{did_web}#mesh-pq"),
            "type": "Multikey",
            "controller": did_web,
            "publicKeyMultibase": pq,
        }));
        key_agreement.push(serde_json::json!({
            "id": format!("{did_web}#mesh-kem-x25519"),
            "type": "Multikey",
            "controller": did_web,
            "publicKeyMultibase": x25519,
        }));
        key_agreement.push(serde_json::json!({
            "id": format!("{did_web}#mesh-kem-mlkem768"),
            "type": "Multikey",
            "controller": did_web,
            "publicKeyMultibase": mlkem,
        }));
    }
    let service: Vec<serde_json::Value> = reach.document_service.clone().into_iter().collect();
    let document = serde_json::json!({
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1",
        ],
        "id": did_web,
        "alsoKnownAs": [at9p_did],
        "verificationMethod": verification_method,
        "keyAgreement": key_agreement,
        "service": service,
    });

    // Re-parse the rendered document with the production extractors: material
    // that decodes to nothing here would leave a remote-node bootstrap without
    // an encryption recipient or a response-authentication anchor.
    if args.mesh_pq.is_some() {
        ensure!(
            hyprstream_rpc::did_web::mesh_kem_recipient(&document).is_some(),
            "--mesh-kem-x25519 / --mesh-kem-mlkem768 did not decode to an x25519 + ML-KEM-768 \
             hybrid recipient; pass the Discovery service's multibase #mesh-kem keys"
        );
        let pq_keys = hyprstream_rpc::did_web::verification_method_ml_dsa_65_keys(&document);
        ensure!(
            pq_keys.len() == 1,
            "--mesh-pq must decode to exactly one ML-DSA-65 Multikey (decoded {})",
            pq_keys.len()
        );
    }
    Ok(document)
}

/// Round-trip a freshly minted capsule/document pair through the production
/// DID-anchored verifier, serving the capsule from memory instead of the
/// deployment's well-known endpoint.
fn verify_anchor_material_offline(
    at9p_did: &str,
    did_web: &str,
    document: &serde_json::Value,
    capsule_bytes: &[u8],
) -> Result<hyprstream_discovery::VerifiedAnchorMaterial> {
    struct MintedCapsule(Vec<u8>);

    #[async_trait::async_trait]
    impl hyprstream_discovery::at9p_resolver::CapsuleSource for MintedCapsule {
        async fn fetch_capsule(&self, _did: &str) -> Result<Vec<u8>> {
            Ok(self.0.clone())
        }
    }

    let anchors = match hyprstream_discovery::DeploymentTrustSource::from_anchors(
        Some(at9p_did),
        Some(did_web),
    )? {
        hyprstream_discovery::DeploymentTrustSource::DidAnchored(anchors) => anchors,
        hyprstream_discovery::DeploymentTrustSource::OsOwnedFiles => {
            bail!("minted anchors did not select the DID-anchored trust source")
        }
    };
    let source = std::sync::Arc::new(MintedCapsule(capsule_bytes.to_vec()));
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build the verification runtime")?
        .block_on(hyprstream_discovery::verify_anchor_material(
            &anchors, document, source,
        ))
}

fn root_recipient_ring(args: &MintDeploymentCaArgs) -> Result<Vec<String>> {
    for recipient in &args.yubikey_recipients {
        ensure!(
            recipient.starts_with("age1yubikey1"),
            "--yubikey value is not an age-plugin-yubikey recipient"
        );
    }
    let all = args
        .recipients
        .iter()
        .chain(&args.yubikey_recipients)
        .chain(&args.kms_plugin_recipients)
        .cloned()
        .collect();
    let recipients = distinct_recipients(all)?;
    ensure!(
        recipients.len() >= 2,
        "root authority requires at least two distinct age recipients \
         (primary + backup/break-glass); a sole YubiKey is forbidden"
    );
    Ok(recipients)
}

fn distinct_recipients(recipients: Vec<String>) -> Result<Vec<String>> {
    let mut unique = BTreeSet::new();
    for recipient in recipients {
        let recipient = recipient.trim();
        ensure!(!recipient.is_empty(), "age recipient is empty");
        ensure!(
            recipient.is_ascii() && !recipient.contains(['\n', '\r', '\0']),
            "age recipient contains invalid characters"
        );
        unique.insert(recipient.to_owned());
    }
    Ok(unique.into_iter().collect())
}

fn combined_identities(generic: &[PathBuf], yubikey: &[PathBuf]) -> Result<AgeIdentities> {
    let identities: Vec<_> = generic.iter().chain(yubikey).cloned().collect();
    ensure!(
        !identities.is_empty(),
        "at least one --identity or --yubikey-identity is required"
    );
    AgeIdentities::new(identities)
}

/// Resolve the mint identity set: either the on-disk path forms (existing
/// ceremony tooling) or the inherited-FD form used by the staging stack
/// (systemd `LoadCredentialEncrypted` + podman `--preserve-fds`), which clap
/// makes mutually exclusive with `--identity`. FD identity bytes are validated
/// and size-capped here, then handed to the `age` child through anonymous
/// memfds so plaintext never touches a filesystem path.
fn mint_identities(args: &MintRegistryJwtArgs) -> Result<AgeIdentities> {
    if args.identity_fds.is_empty() {
        return combined_identities(&args.identities, &args.yubikey_identities);
    }
    let mut bytes = Vec::with_capacity(args.identity_fds.len());
    for &fd in &args.identity_fds {
        bytes.push(Zeroizing::new(read_fd_limited(
            fd,
            MAX_AGE_IDENTITY_BYTES,
            "age identity",
        )?));
    }
    if args.yubikey_identities.is_empty() {
        return AgeIdentities::new_in_memory(bytes);
    }
    // YubiKey path identities may be mixed in, exactly like the path forms.
    let mut sources: Vec<_> = args
        .yubikey_identities
        .iter()
        .cloned()
        .map(AgeIdentitySource::Path)
        .collect();
    sources.extend(bytes.into_iter().map(AgeIdentitySource::InMemory));
    AgeIdentities::from_sources(sources)
}

fn encrypt_age(plaintext: &[u8], recipients: &[String]) -> Result<Vec<u8>> {
    AgeRecipients::new(recipients.to_vec())?
        .seal(plaintext, MAX_AGE_CIPHERTEXT_BYTES)
        .context("encrypt authority through deployment age seam")
}

const MAX_AGE_PLAINTEXT_BYTES: usize = 128 * 1024;

fn decrypt_age(path: &Path, identities: &AgeIdentities) -> Result<Zeroizing<Vec<u8>>> {
    identities
        .open_file(path, MAX_AGE_PLAINTEXT_BYTES)
        .context("decrypt authority through deployment age seam")
}

fn decrypt_age_bytes(ciphertext: &[u8], identities: &AgeIdentities) -> Result<Zeroizing<Vec<u8>>> {
    identities
        .open(ciphertext, MAX_AGE_PLAINTEXT_BYTES)
        .context("decrypt authority through deployment age seam")
}

fn decrypt_authority(
    path: &Path,
    identities: &AgeIdentities,
    software_recovery: bool,
) -> Result<LoadedAuthority> {
    let plaintext = decrypt_age(path, identities)?;
    decode_authority(&plaintext, software_recovery)
}

/// Decrypt an age ciphertext already held in memory (inherited-FD form).
fn decrypt_authority_ciphertext(
    ciphertext: &[u8],
    identities: &AgeIdentities,
    software_recovery: bool,
) -> Result<LoadedAuthority> {
    let plaintext = decrypt_age_bytes(ciphertext, identities)?;
    decode_authority(&plaintext, software_recovery)
}

fn decode_authority(plaintext: &[u8], software_recovery: bool) -> Result<LoadedAuthority> {
    let bundle: AuthorityBundle =
        serde_json::from_slice(plaintext).context("decode authority bundle")?;
    ensure!(
        bundle.schema == AUTHORITY_BUNDLE_SCHEMA,
        "unsupported authority bundle schema"
    );
    let pq_seed = decode_fixed_b64::<32>(&bundle.ml_dsa_65_seed_b64, "ML-DSA-65 seed")?;
    let pq = ml_dsa_sk_from_seed(&pq_seed);
    let ed = match &bundle.ed25519 {
        Ed25519Secret::Software {
            seed_b64,
            public_b64,
        } => {
            ensure!(
                !software_recovery,
                "--software-recovery applies only to a PIV-backed authority"
            );
            let seed = decode_fixed_b64::<32>(seed_b64, "Ed25519 seed")?;
            let key = SigningKey::from_bytes(&seed);
            ensure!(
                STANDARD.decode(public_b64)? == key.verifying_key().as_bytes(),
                "authority Ed25519 public key does not match seed"
            );
            LoadedEdSigner::Software(key)
        }
        Ed25519Secret::YubikeyPiv {
            slot,
            public_b64,
            recovery_seed_b64,
        } => {
            validate_piv_slot(slot)?;
            let public_bytes = decode_fixed_b64::<32>(public_b64, "PIV Ed25519 public key")?;
            let public =
                VerifyingKey::from_bytes(&public_bytes).context("invalid PIV public key")?;
            if software_recovery {
                let seed = decode_fixed_b64::<32>(recovery_seed_b64, "PIV recovery Ed25519 seed")?;
                let key = SigningKey::from_bytes(&seed);
                ensure!(
                    key.verifying_key() == public,
                    "PIV recovery seed does not match the recorded public key"
                );
                LoadedEdSigner::Software(key)
            } else {
                LoadedEdSigner::YubikeyPiv {
                    slot: slot.clone(),
                    public,
                }
            }
        }
    };
    let loaded = LoadedAuthority { bundle, ed, pq };
    let public = loaded.public_bytes();
    ensure!(
        sha256_hex(&public) == loaded.bundle.public_key_sha256,
        "authority public-key fingerprint mismatch"
    );
    Ok(loaded)
}

fn public_pair_bytes(ed: &VerifyingKey, pq: &MlDsaSigningKey) -> Vec<u8> {
    let mut output = Vec::with_capacity(PUBLIC_CA_BYTES);
    output.extend_from_slice(ed.as_bytes());
    output.extend_from_slice(&ml_dsa_sk_to_vk_bytes(pq));
    output
}

fn parse_public_pair(
    bytes: &[u8],
) -> Result<(VerifyingKey, hyprstream_rpc::crypto::pq::MlDsaVerifyingKey)> {
    ensure!(
        bytes.len() == PUBLIC_CA_BYTES,
        "hybrid public key must be exactly {PUBLIC_CA_BYTES} bytes"
    );
    let ed_bytes: [u8; 32] = bytes[..32]
        .try_into()
        .map_err(|_| anyhow!("invalid Ed25519 public-key length"))?;
    let ed = VerifyingKey::from_bytes(&ed_bytes).context("invalid Ed25519 public key")?;
    let pq = ml_dsa_vk_from_bytes(&bytes[32..])?;
    Ok((ed, pq))
}

fn encode_registry_jwt(
    signer: &LoadedAuthority,
    registry_key: &[u8; 32],
    now: i64,
    exp: i64,
    delegation_b64: Option<&str>,
) -> Result<String> {
    let signer_ed = signer.ed.verifying_key();
    let signer_pq = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&signer.pq))?;
    let kid = hyprstream_rpc::auth::composite_kid(&signer_pq, &signer_ed);
    let protected = serde_json::json!({
        "alg": "ML-DSA-65-Ed25519",
        "typ": "wit+jwt",
        "kid": kid
    });
    let mut claims = serde_json::Map::new();
    claims.insert(
        "iss".to_owned(),
        serde_json::Value::String(format!(
            "urn:hyprstream:deployment:{}",
            signer.bundle.deployment_domain
        )),
    );
    claims.insert(
        "sub".to_owned(),
        serde_json::Value::String("service:registry".to_owned()),
    );
    claims.insert(
        "aud".to_owned(),
        serde_json::Value::String(REGISTRY_AUDIENCE.to_owned()),
    );
    claims.insert("exp".to_owned(), serde_json::Value::Number(exp.into()));
    claims.insert("nbf".to_owned(), serde_json::Value::Number(now.into()));
    claims.insert("iat".to_owned(), serde_json::Value::Number(now.into()));
    claims.insert(
        "deployment_domain".to_owned(),
        serde_json::Value::String(signer.bundle.deployment_domain.clone()),
    );
    claims.insert(
        "profile".to_owned(),
        serde_json::Value::String(REGISTRY_PROFILE.to_owned()),
    );
    claims.insert(
        "cnf".to_owned(),
        serde_json::json!({
            "jwk": {
                "kty": "OKP",
                "crv": "Ed25519",
                "x": URL_SAFE_NO_PAD.encode(registry_key)
            }
        }),
    );
    if let Some(delegation) = delegation_b64 {
        claims.insert(
            "delegation".to_owned(),
            serde_json::Value::String(delegation.to_owned()),
        );
    }
    let protected_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&protected)?);
    let claims_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims)?);
    let signing_input = format!("{protected_b64}.{claims_b64}");
    let pq_signature = ml_dsa_sign(&signer.pq, signing_input.as_bytes());
    let ed_signature = signer.ed.sign(signing_input.as_bytes())?;
    let mut signature = Vec::with_capacity(HYBRID_SIGNATURE_BYTES);
    signature.extend_from_slice(&pq_signature);
    signature.extend_from_slice(&ed_signature);
    ensure!(
        signature.len() == HYBRID_SIGNATURE_BYTES,
        "internal hybrid JWT signature length error"
    );
    Ok(format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature)
    ))
}

fn registry_mint_capability(deployment_domain: &str, delegated_public: &[u8]) -> Capability {
    let mut caveats = BTreeMap::new();
    caveats.insert(
        "audience".to_owned(),
        CaveatValue::Text(REGISTRY_AUDIENCE.to_owned()),
    );
    caveats.insert(
        "deployment_domain".to_owned(),
        CaveatValue::Text(deployment_domain.to_owned()),
    );
    caveats.insert(
        "delegated_public_key_b64".to_owned(),
        CaveatValue::Text(STANDARD.encode(delegated_public)),
    );
    caveats.insert("max_ttl_seconds".to_owned(), CaveatValue::Int(3_600));
    caveats.insert(
        "profile".to_owned(),
        CaveatValue::Text(REGISTRY_PROFILE.to_owned()),
    );
    Capability::with_caveats(
        Resource::new(format!(
            "{DELEGATION_RESOURCE_PREFIX}/{deployment_domain}/service/registry"
        )),
        Ability::new(DELEGATION_ABILITY),
        Caveats(caveats),
    )
}

fn sign_ucan(payload: UcanPayload, ed: &LoadedEdSigner, pq: &MlDsaSigningKey) -> Result<Ucan> {
    let payload_bytes = payload.signing_bytes()?;
    let signature = sign_nested(
        &payload_bytes,
        hyprstream_rpc::auth::ucan::token::UCAN_AAD,
        ed,
        pq,
    )?;
    Ok(Ucan {
        payload,
        proofs: Vec::new(),
        signature,
    })
}

fn sign_nested(
    payload: &[u8],
    aad: &[u8],
    ed: &LoadedEdSigner,
    pq: &MlDsaSigningKey,
) -> Result<Vec<u8>> {
    let ed_public = ed.verifying_key();
    let pq_public = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(pq))?;
    let ed_tbs = inner_tbs(ed_public.to_bytes().to_vec(), payload, aad, true);
    let ed_signature = ed.sign(&ed_tbs)?;
    let pq_tbs = outer_tbs(ml_dsa_vk_bytes(&pq_public), payload, &ed_signature, aad);
    let pq_signature = ml_dsa_sign(pq, &pq_tbs);
    assemble_composite_nested(
        (ed_public.to_bytes().to_vec(), ed_signature.to_vec()),
        Some((ml_dsa_vk_bytes(&pq_public), pq_signature)),
    )
}

fn sign_did_op(mut op: DidOp, ed: &LoadedEdSigner, pq: &MlDsaSigningKey) -> Result<DidOp> {
    let composite = sign_nested(&op.signable_bytes(), DID_OP_SIGNATURE_CONTEXT, ed, pq)?;
    let (ed_signature, pq_signature) =
        hyprstream_rpc::crypto::cose_sign::split_composite(&composite)?;
    op.signature = HybridDidOpSignature {
        ed25519: ed_signature,
        mldsa65: pq_signature.ok_or_else(|| anyhow!("missing ML-DSA signature"))?,
    };
    DidOp::from_dag_cbor(&op.to_dag_cbor()).context("self-verify signed DID operation")
}

fn placeholder_did_signature() -> HybridDidOpSignature {
    HybridDidOpSignature {
        ed25519: vec![0; 64],
        mldsa65: vec![0; 3_309],
    }
}

fn authority_log_from_ops(
    deployment_domain: &str,
    operations: Vec<DidOp>,
) -> Result<AuthorityLogFile> {
    ensure!(!operations.is_empty(), "authority log cannot be empty");
    ensure!(
        operations.len() <= MAX_AUTHORITY_LOG_OPERATIONS,
        "authority log has too many operations"
    );
    let did = operations[0].genesis_did()?;
    verify_did_op_log(&did, &operations)?;
    let log = AuthorityLogFile {
        schema: AUTHORITY_LOG_SCHEMA.to_owned(),
        deployment_domain: deployment_domain.to_owned(),
        did,
        operations_b64: operations
            .iter()
            .map(|op| STANDARD.encode(op.to_dag_cbor()))
            .collect(),
    };
    ensure!(
        pretty_json_bytes(&log)?.len() <= MAX_CLOUD_SECRET_BYTES,
        "authority log exceeds the 64 KiB cloud-secret contract"
    );
    Ok(log)
}

fn decode_authority_operations(log: &AuthorityLogFile) -> Result<Vec<DidOp>> {
    ensure!(
        log.schema == AUTHORITY_LOG_SCHEMA,
        "unsupported authority-log schema"
    );
    ensure!(
        !log.operations_b64.is_empty() && log.operations_b64.len() <= MAX_AUTHORITY_LOG_OPERATIONS,
        "authority log operation count is invalid"
    );
    log.operations_b64
        .iter()
        .map(|encoded| {
            let bytes = STANDARD
                .decode(encoded)
                .context("decode authority operation")?;
            DidOp::from_dag_cbor(&bytes)
        })
        .collect()
}

fn validate_authority_log_root(
    public_ca: &[u8],
    log: &AuthorityLogFile,
) -> Result<hyprstream_discovery::did_op::VerifiedDidOpLog> {
    let root_domain = hyprstream_discovery::verify_deployment_public_ca(public_ca)?;
    ensure!(
        log.deployment_domain == root_domain,
        "authority log deployment domain does not match pinned root"
    );
    let operations = decode_authority_operations(log)?;
    let verified = verify_did_op_log(&log.did, &operations)?;
    let (root_ed, root_pq) = parse_public_pair(public_ca)?;
    let genesis = operations
        .first()
        .ok_or_else(|| anyhow!("authority log is empty"))?;
    let genesis_composite = assemble_composite_nested(
        (
            root_ed.to_bytes().to_vec(),
            genesis.signature.ed25519.clone(),
        ),
        Some((ml_dsa_vk_bytes(&root_pq), genesis.signature.mldsa65.clone())),
    )?;
    hyprstream_rpc::crypto::cose_sign::verify_composite(
        &genesis_composite,
        &root_ed,
        Some(&root_pq),
        &genesis.signable_bytes(),
        DID_OP_SIGNATURE_CONTEXT,
        true,
    )
    .context("authority-log genesis was not signed by the pinned root")?;
    ensure!(
        genesis.rotation_keys.iter().any(|key| {
            key.ed25519_pub == root_ed.to_bytes() && key.mldsa65_pub == ml_dsa_vk_bytes(&root_pq)
        }),
        "authority log genesis is not anchored by the pinned public root"
    );
    Ok(verified)
}

fn authority_checkpoint(
    deployment_domain: &str,
    verified: &hyprstream_discovery::did_op::VerifiedDidOpLog,
) -> AuthorityCheckpointFile {
    AuthorityCheckpointFile {
        schema: AUTHORITY_CHECKPOINT_SCHEMA.to_owned(),
        deployment_domain: deployment_domain.to_owned(),
        did: verified.did.clone(),
        sequence: verified.sequence,
        head_cid: verified.head_cid.clone(),
    }
}

fn validate_authority_log(
    public_ca: &[u8],
    log: &AuthorityLogFile,
    checkpoint: &AuthorityCheckpointFile,
) -> Result<hyprstream_discovery::did_op::VerifiedDidOpLog> {
    let verified = validate_authority_log_root(public_ca, log)?;
    ensure!(
        checkpoint.schema == AUTHORITY_CHECKPOINT_SCHEMA,
        "unsupported authority-checkpoint schema"
    );
    ensure!(
        checkpoint.deployment_domain == log.deployment_domain,
        "authority checkpoint deployment domain does not match log"
    );
    ensure!(
        checkpoint.did == log.did
            && checkpoint.did == verified.did
            && checkpoint.sequence == verified.sequence
            && checkpoint.head_cid == verified.head_cid,
        "authority-log head does not match the independently trusted checkpoint"
    );
    Ok(verified)
}

fn ensure_active_authority(
    authority: &LoadedAuthority,
    active: &hyprstream_discovery::did_op::VerifiedDidOpLog,
) -> Result<()> {
    let public = authority.public_bytes();
    ensure!(
        active
            .rotation_keys
            .iter()
            .any(|key| { public[..32] == key.ed25519_pub && public[32..] == key.mldsa65_pub }),
        "authority key is not active at the rotation-log head"
    );
    Ok(())
}

/// Verify that `authority` is the deployment root that the resolver pins and is
/// therefore permitted to mint an anchor capsule.
///
/// This bundles the three guard checks `mint_anchor_capsule` runs before it
/// builds the capsule body: purpose (not a delegated signer), CA-binding (key
/// matches the pinned root), and rotation-log activeness. It is the seam the
/// minter's self-check tests exercise so removing any single guard turns the
/// test red.
fn ensure_anchor_authority(
    authority: &LoadedAuthority,
    public_ca: &[u8],
    active: &hyprstream_discovery::did_op::VerifiedDidOpLog,
) -> Result<()> {
    ensure!(
        authority.bundle.purpose != AuthorityPurpose::RegistryDelegatedSigner,
        "the anchor capsule must be signed by the deployment authority itself; \
         a registry-scoped delegated signer cannot anchor a deployment"
    );
    ensure!(
        authority.public_bytes() == public_ca,
        "anchor capsule signing authority does not match the pinned public CA; \
         only the deployment root that the resolver pins may anchor a deployment"
    );
    ensure_active_authority(authority, active)?;
    Ok(())
}

struct AuthorityUcanVerifier<'a> {
    keys: &'a [HybridRotationKey],
}

impl UcanVerifier for AuthorityUcanVerifier<'_> {
    fn verify(
        &self,
        _issuer: &Did,
        ed_key: &[u8; 32],
        payload: &[u8],
        signature: &[u8],
    ) -> std::result::Result<(), UcanError> {
        let key = self
            .keys
            .iter()
            .find(|key| &key.ed25519_pub == ed_key)
            .ok_or_else(|| {
                UcanError::BadSignature("issuer is not an active authority".to_owned())
            })?;
        let ed = VerifyingKey::from_bytes(ed_key)
            .map_err(|error| UcanError::BadSignature(error.to_string()))?;
        let pq = ml_dsa_vk_from_bytes(&key.mldsa65_pub)
            .map_err(|error| UcanError::BadSignature(error.to_string()))?;
        hyprstream_rpc::crypto::cose_sign::verify_composite(
            signature,
            &ed,
            Some(&pq),
            payload,
            hyprstream_rpc::auth::ucan::token::UCAN_AAD,
            true,
        )
        .map(|_| ())
        .map_err(|error| UcanError::BadSignature(error.to_string()))
    }
}

fn validate_registry_delegation_ucan(
    ucan: &Ucan,
    deployment_domain: &str,
    delegated_public: &[u8],
    active_keys: &[HybridRotationKey],
    now: u64,
) -> Result<()> {
    ensure!(
        ucan.proofs.is_empty(),
        "registry delegation must be one root-authorized link"
    );
    let verifier = AuthorityUcanVerifier { keys: active_keys };
    validate_ucan(ucan, &verifier, now).context("validate registry UCAN delegation")?;
    ensure!(
        active_keys
            .iter()
            .any(|key| ucan.issuer().to_ed25519().ok() == Some(key.ed25519_pub)),
        "delegation issuer is not active"
    );
    let delegated_ed: [u8; 32] = delegated_public
        .get(..32)
        .ok_or_else(|| anyhow!("delegated public key is truncated"))?
        .try_into()
        .map_err(|_| anyhow!("delegated Ed25519 key is malformed"))?;
    ensure!(
        ucan.audience().to_ed25519()? == delegated_ed,
        "delegation audience does not match delegated signer"
    );
    ensure!(
        ucan.capabilities()
            == [registry_mint_capability(
                deployment_domain,
                delegated_public
            )],
        "delegation capability is not the exact registry-only scope"
    );
    ensure!(
        ucan.payload.expiration.is_some(),
        "registry delegation must expire"
    );
    Ok(())
}

fn validate_delegation_artifact(
    public_ca: &[u8],
    authority_log: &AuthorityLogFile,
    authority_checkpoint: &AuthorityCheckpointFile,
    artifact: &DelegationArtifact,
    now: u64,
) -> Result<()> {
    ensure!(
        artifact.schema == DELEGATION_SCHEMA,
        "unsupported delegation schema"
    );
    let active = validate_authority_log(public_ca, authority_log, authority_checkpoint)?;
    ensure!(
        artifact.deployment_domain == authority_log.deployment_domain,
        "delegation domain does not match authority log"
    );
    ensure!(
        artifact.authority_log_did == authority_log.did,
        "delegation names a different authority log"
    );
    let delegated_public = STANDARD
        .decode(&artifact.delegated_public_key_b64)
        .context("decode delegated public key")?;
    parse_public_pair(&delegated_public)?;
    let ucan_bytes = STANDARD
        .decode(&artifact.ucan_b64)
        .context("decode delegation UCAN")?;
    ensure!(
        ucan_bytes.len() <= MAX_DELEGATION_BYTES,
        "delegation UCAN is too large"
    );
    let ucan = Ucan::from_cbor(&ucan_bytes)?;
    validate_registry_delegation_ucan(
        &ucan,
        &artifact.deployment_domain,
        &delegated_public,
        &active.rotation_keys,
        now,
    )
}

/// `ykman` argv (after the program name) that asks whether a PIV slot holds
/// a private key, plus the stderr marker it prints when the slot has none.
///
/// `{slot}` is substituted with the validated slot id.
const PIV_KEY_PROBE: PivProbe = PivProbe {
    argv: &["piv", "keys", "info", "{slot}"],
    absent_marker: "No key stored in slot",
};

/// `ykman` argv that asks whether a PIV slot holds a certificate object,
/// plus the stderr marker it prints when the slot has none. `-` sends any
/// certificate found to stdout, which the probe discards.
const PIV_CERT_PROBE: PivProbe = PivProbe {
    argv: &["piv", "certificates", "export", "{slot}", "-"],
    absent_marker: "No certificate found",
};

/// A single `ykman` presence probe: what to run, and how to recognise a
/// definitive "nothing is here" answer.
#[derive(Clone, Copy, Debug)]
struct PivProbe {
    argv: &'static [&'static str],
    absent_marker: &'static str,
}

/// Decide whether a probe definitively reported an empty slot.
///
/// `ykman` signals absence as exit 1 with a specific message on stderr.
/// Every other outcome is indeterminate and must NOT read as absent:
/// exit 0 (the object exists), a different exit-1 message (PCSC failure,
/// no token present, locked slot), exit 2 (argument or subcommand error —
/// including a subcommand that does not exist), or no exit code at all
/// (killed by signal). Refusing a destructive import on an indeterminate
/// answer is recoverable; permitting one is not.
fn probe_reports_absent(probe: PivProbe, exit_code: Option<i32>, stderr: &str) -> bool {
    exit_code == Some(1) && stderr.contains(probe.absent_marker)
}

/// Combine both probe results into the occupancy decision.
///
/// The slot is occupied unless BOTH probes definitively reported absence.
/// Split out from [`piv_slot_occupied`] so the decision — including the
/// fail-closed handling of every indeterminate outcome — is unit-testable
/// against the exact exit codes and stderr text the real binary emits.
fn slot_occupied_from_probes(key: (Option<i32>, &str), cert: (Option<i32>, &str)) -> bool {
    !probe_reports_absent(PIV_KEY_PROBE, key.0, key.1)
        || !probe_reports_absent(PIV_CERT_PROBE, cert.0, cert.1)
}

/// Run one `ykman` presence probe and return its exit code and stderr.
///
/// A failure to launch `ykman` at all is returned as an error rather than
/// as a probe result: the caller propagates it, which aborts the mint
/// before anything destructive runs.
fn run_piv_probe(probe: PivProbe, slot: &str) -> Result<(Option<i32>, String)> {
    let mut command = Command::new("ykman");
    for arg in probe.argv {
        command.arg(if *arg == "{slot}" { slot } else { arg });
    }
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("launch ykman {}", probe.argv.join(" ")))?;
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Ok((output.status.code(), stderr))
}

/// Check whether a PIV slot already holds a key or certificate.
///
/// `ykman piv keys import` silently overwrites slot contents; this check
/// gives the caller a chance to warn or refuse before the import runs.
///
/// Probes BOTH objects, because either alone misses a real state:
/// - `ykman piv keys info <slot>` reports key metadata regardless of
///   certificate state, which is what catches the key-but-no-certificate
///   slot [`piv_import_ed25519`] leaves behind.
/// - `ykman piv certificates export <slot> -` catches a slot holding a
///   certificate whose private key lives elsewhere or has been deleted.
///
/// Any outcome that is not a definitive "absent" — PCSC failure, token
/// removed, locked slot, unexpected exit status — reads as occupied. If
/// `ykman` cannot be launched at all this returns an error, which also
/// stops the import.
fn piv_slot_occupied(slot: &str) -> Result<bool> {
    let slot = validate_piv_slot(slot)?;
    let key = run_piv_probe(PIV_KEY_PROBE, &slot)?;
    // Short-circuit: a key is the object the guard exists to protect, so
    // there is no reason to touch the card a second time once one is
    // known (or suspected) to be present.
    if !probe_reports_absent(PIV_KEY_PROBE, key.0, &key.1) {
        return Ok(true);
    }
    let cert = run_piv_probe(PIV_CERT_PROBE, &slot)?;
    Ok(slot_occupied_from_probes(
        (key.0, &key.1),
        (cert.0, &cert.1),
    ))
}

fn piv_import_ed25519(slot: &str, key: &SigningKey) -> Result<(String, VerifyingKey)> {
    let slot = validate_piv_slot(slot)?;
    let private_der = key
        .to_pkcs8_der()
        .context("encode Ed25519 key for PIV import")?;
    let mut child = Command::new("ykman")
        .args([
            "piv",
            "keys",
            "import",
            "--pin-policy",
            "ALWAYS",
            "--touch-policy",
            "ALWAYS",
        ])
        .arg(&slot)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("launch ykman PIV import (YubiKey 5.7.4+ is required)")?;
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("ykman stdin unavailable"))?
        .write_all(private_der.as_bytes())
        .context("write Ed25519 PKCS#8 key to ykman")?;
    let status = child.wait().context("wait for ykman PIV import")?;
    ensure!(status.success(), "ykman PIV Ed25519 import failed");
    let public = key.verifying_key();
    piv_sign(
        &slot,
        &public,
        b"hyprstream deployment authority PIV import self-check",
    )
    .context("PIV import self-check")?;
    Ok((slot, public))
}

fn piv_sign(slot: &str, public: &VerifyingKey, message: &[u8]) -> Result<[u8; 64]> {
    validate_piv_slot(slot)?;
    let input = tempfile::NamedTempFile::new().context("create PIV signing input")?;
    let output = tempfile::NamedTempFile::new().context("create PIV signing output")?;
    std::fs::write(input.path(), message).context("write PIV signing input")?;
    let status = Command::new("yubico-piv-tool")
        .args(["-a", "verify-pin", "--sign", "-s"])
        .arg(slot)
        .args(["-A", "ED25519", "-i"])
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("launch yubico-piv-tool")?;
    ensure!(status.success(), "YubiKey PIV signing failed");
    let signature: [u8; 64] = std::fs::read(output.path())?
        .try_into()
        .map_err(|_| anyhow!("YubiKey returned a non-64-byte Ed25519 signature"))?;
    public
        .verify_strict(message, &ed25519_dalek::Signature::from_bytes(&signature))
        .context("YubiKey signature self-check failed")?;
    Ok(signature)
}

fn validate_piv_slot(slot: &str) -> Result<String> {
    let slot = slot.trim().to_ascii_lowercase();
    let allowed = matches!(
        slot.as_str(),
        "9a" | "9c"
            | "9d"
            | "9e"
            | "82"
            | "83"
            | "84"
            | "85"
            | "86"
            | "87"
            | "88"
            | "89"
            | "8a"
            | "8b"
            | "8c"
            | "8d"
            | "8e"
            | "8f"
            | "90"
            | "91"
            | "92"
            | "93"
            | "94"
            | "95"
    );
    ensure!(allowed, "unsupported or unsafe PIV slot {slot:?}");
    Ok(slot)
}

fn preflight_outputs<'a>(paths: impl IntoIterator<Item = &'a PathBuf>, force: bool) -> Result<()> {
    let mut seen = BTreeSet::new();
    for path in paths {
        ensure!(
            seen.insert(path.clone()),
            "duplicate output path {}",
            path.display()
        );
        match std::fs::symlink_metadata(path) {
            Ok(metadata) => {
                ensure!(
                    force,
                    "refusing to replace existing output {} without --force",
                    path.display()
                );
                ensure!(
                    metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                    "refusing non-regular or symlink output target {}",
                    path.display()
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect output target {}", path.display()));
            }
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let parent_metadata = std::fs::metadata(parent)
            .with_context(|| format!("inspect output parent {}", parent.display()))?;
        ensure!(
            parent_metadata.is_dir(),
            "output parent is not a directory: {}",
            parent.display()
        );
    }
    Ok(())
}

fn pretty_json_bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn serialize_secret_json(value: &impl Serialize) -> Result<Zeroizing<Vec<u8>>> {
    let mut plaintext = Zeroizing::new(Vec::new());
    serde_json::to_writer(&mut *plaintext, value)?;
    Ok(plaintext)
}

struct PendingOutput<'a> {
    path: &'a Path,
    bytes: Vec<u8>,
    mode: u32,
}

impl<'a> PendingOutput<'a> {
    fn new(path: &'a Path, bytes: Vec<u8>, mode: u32) -> Self {
        Self { path, bytes, mode }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommitOperation {
    StageCreate(usize),
    StagePermissions(usize),
    StageWrite(usize),
    StageSync(usize),
    StageKeep(usize),
    BackupMetadata(usize),
    BackupCreate(usize),
    BackupPlaceholderKeep(usize),
    BackupPlaceholderRemove(usize),
    BackupRename(usize),
    BackupParentsSync,
    CommitRename(usize),
    CommitParentsSync,
    BackupDelete(usize),
    CleanupParentsSync,
    RollbackRemove(usize),
    RollbackRestore(usize),
    RollbackParentsSync,
    StageCleanup(usize),
}

#[derive(Default)]
struct CommitFaultInjector {
    #[cfg(test)]
    faults: std::collections::VecDeque<CommitOperation>,
}

impl CommitFaultInjector {
    fn check(&mut self, operation: CommitOperation) -> Result<()> {
        #[cfg(test)]
        if self.faults.front() == Some(&operation) {
            self.faults.pop_front();
            bail!("injected transaction fault at {operation:?}");
        }
        let _ = operation;
        Ok(())
    }

    #[cfg(test)]
    fn with_faults(faults: impl IntoIterator<Item = CommitOperation>) -> Self {
        Self {
            faults: faults.into_iter().collect(),
        }
    }
}

/// Stage every output before moving any destination, then roll back the whole
/// set if any subsequent operation fails. Each rename is filesystem-atomic;
/// every fallible step after the first retained staging path is routed through
/// the same rollback/cleanup path.
fn commit_outputs(outputs: Vec<PendingOutput<'_>>) -> Result<()> {
    commit_outputs_with_faults(outputs, &mut CommitFaultInjector::default())
}

fn destination_exists_for_backup(
    output: &PendingOutput<'_>,
    index: usize,
    faults: &mut CommitFaultInjector,
) -> Result<bool> {
    faults.check(CommitOperation::BackupMetadata(index))?;
    match std::fs::symlink_metadata(output.path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "refusing non-regular or symlink output target {}",
                output.path.display()
            );
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("inspect output target {}", output.path.display()))
        }
    }
}

fn commit_outputs_with_faults(
    outputs: Vec<PendingOutput<'_>>,
    faults: &mut CommitFaultInjector,
) -> Result<()> {
    let mut staged = Vec::with_capacity(outputs.len());
    for (index, output) in outputs.iter().enumerate() {
        let parent = output.path.parent().unwrap_or_else(|| Path::new("."));
        let staged_result = (|| -> Result<PathBuf> {
            faults.check(CommitOperation::StageCreate(index))?;
            let mut temp = tempfile::NamedTempFile::new_in(parent)
                .with_context(|| format!("create staged output for {}", output.path.display()))?;
            faults.check(CommitOperation::StagePermissions(index))?;
            temp.as_file_mut()
                .set_permissions(std::fs::Permissions::from_mode(output.mode))?;
            faults.check(CommitOperation::StageWrite(index))?;
            temp.write_all(&output.bytes)?;
            faults.check(CommitOperation::StageSync(index))?;
            temp.as_file_mut().sync_all()?;
            faults.check(CommitOperation::StageKeep(index))?;
            let (_file, path) = temp.keep().context("retain staged output")?;
            Ok(path)
        })();
        match staged_result {
            Ok(path) => staged.push(path),
            Err(error) => {
                return Err(transaction_failure(
                    error,
                    format!("stage {}", output.path.display()),
                    Ok(()),
                    cleanup_paths(&staged, faults),
                ));
            }
        }
    }

    let mut backups: Vec<Option<PathBuf>> = vec![None; outputs.len()];
    for (index, output) in outputs.iter().enumerate() {
        let destination_exists = match destination_exists_for_backup(output, index, faults) {
            Ok(exists) => exists,
            Err(error) => {
                return Err(transaction_failure(
                    error,
                    format!("inspect existing {}", output.path.display()),
                    restore_outputs(&outputs, &backups, 0, faults),
                    cleanup_paths(&staged, faults),
                ));
            }
        };
        if destination_exists {
            let parent = output.path.parent().unwrap_or_else(|| Path::new("."));
            let backup = (|| -> Result<PathBuf> {
                faults.check(CommitOperation::BackupCreate(index))?;
                let placeholder = tempfile::Builder::new()
                    .prefix(".hyprstream-backup-")
                    .tempfile_in(parent)
                    .with_context(|| format!("stage backup for {}", output.path.display()))?;
                faults.check(CommitOperation::BackupPlaceholderKeep(index))?;
                let (_file, backup) = placeholder.keep().with_context(|| {
                    format!("retain backup placeholder for {}", output.path.display())
                })?;
                let removal = faults
                    .check(CommitOperation::BackupPlaceholderRemove(index))
                    .and_then(|()| {
                        std::fs::remove_file(&backup).with_context(|| {
                            format!("remove backup placeholder {}", backup.display())
                        })
                    });
                if let Err(error) = removal {
                    bail!(
                        "backup placeholder removal failed; preserved empty recovery marker at {}: {error:#}",
                        backup.display()
                    );
                }
                Ok(backup)
            })();
            let backup = match backup {
                Ok(backup) => backup,
                Err(error) => {
                    return Err(transaction_failure(
                        error,
                        format!("reserve backup for {}", output.path.display()),
                        restore_outputs(&outputs, &backups, 0, faults),
                        cleanup_paths(&staged, faults),
                    ));
                }
            };
            let rename = faults
                .check(CommitOperation::BackupRename(index))
                .and_then(|()| {
                    std::fs::rename(output.path, &backup)
                        .with_context(|| format!("back up existing {}", output.path.display()))
                });
            if let Err(error) = rename {
                return Err(transaction_failure(
                    error,
                    format!("back up existing {}", output.path.display()),
                    restore_outputs(&outputs, &backups, 0, faults),
                    cleanup_paths(&staged, faults),
                ));
            }
            backups[index] = Some(backup);
        }
    }
    if let Err(error) = sync_output_parents(&outputs, faults, CommitOperation::BackupParentsSync) {
        return Err(transaction_failure(
            error,
            "durably record output backups".to_owned(),
            restore_outputs(&outputs, &backups, 0, faults),
            cleanup_paths(&staged, faults),
        ));
    }

    for (index, output) in outputs.iter().enumerate() {
        let rename = faults
            .check(CommitOperation::CommitRename(index))
            .and_then(|()| {
                std::fs::rename(&staged[index], output.path)
                    .with_context(|| format!("commit {}", output.path.display()))
            });
        if let Err(error) = rename {
            return Err(transaction_failure(
                error,
                format!("commit {}", output.path.display()),
                restore_outputs(&outputs, &backups, index, faults),
                cleanup_paths(&staged[index..], faults),
            ));
        }
    }
    if let Err(error) = sync_output_parents(&outputs, faults, CommitOperation::CommitParentsSync) {
        return Err(transaction_failure(
            error,
            "durably record committed outputs".to_owned(),
            restore_outputs(&outputs, &backups, outputs.len(), faults),
            Ok(()),
        ));
    }

    let mut cleanup_errors = Vec::new();
    for (index, backup) in backups.into_iter().enumerate() {
        if let Some(backup) = backup {
            let removal = faults
                .check(CommitOperation::BackupDelete(index))
                .and_then(|()| {
                    std::fs::remove_file(&backup)
                        .with_context(|| format!("remove backup {}", backup.display()))
                });
            if let Err(error) = removal {
                cleanup_errors.push(format!("{error:#}"));
            }
        }
    }
    if let Err(error) = sync_output_parents(&outputs, faults, CommitOperation::CleanupParentsSync) {
        cleanup_errors.push(format!("durably record backup deletion: {error:#}"));
    }
    ensure!(
        cleanup_errors.is_empty(),
        "outputs committed, but transaction cleanup was incomplete; preserved backup state where possible: {}",
        cleanup_errors.join("; ")
    );
    Ok(())
}

fn transaction_failure(
    primary: anyhow::Error,
    context: String,
    rollback: Result<()>,
    staged_cleanup: Result<()>,
) -> anyhow::Error {
    let mut message = format!("{context}: {primary:#}");
    if let Err(error) = rollback {
        message.push_str(&format!(
            "; ROLLBACK INCOMPLETE (backup state preserved where possible): {error:#}"
        ));
    }
    if let Err(error) = staged_cleanup {
        message.push_str(&format!("; staged-output cleanup incomplete: {error:#}"));
    }
    anyhow!(message)
}

fn restore_outputs(
    outputs: &[PendingOutput<'_>],
    backups: &[Option<PathBuf>],
    committed: usize,
    faults: &mut CommitFaultInjector,
) -> Result<()> {
    let mut errors = Vec::new();
    for (index, output) in outputs.iter().take(committed).enumerate() {
        let removal = faults
            .check(CommitOperation::RollbackRemove(index))
            .and_then(|()| {
                std::fs::remove_file(output.path)
                    .with_context(|| format!("remove newly committed {}", output.path.display()))
            });
        if let Err(error) = removal {
            errors.push(format!(
                "remove newly committed {}: {error:#}",
                output.path.display(),
            ));
        }
    }
    for (index, (output, backup)) in outputs.iter().zip(backups).enumerate().rev() {
        if let Some(backup) = backup {
            let restore = faults
                .check(CommitOperation::RollbackRestore(index))
                .and_then(|()| {
                    std::fs::rename(backup, output.path).with_context(|| {
                        format!(
                            "restore {} from {}",
                            output.path.display(),
                            backup.display()
                        )
                    })
                });
            if let Err(error) = restore {
                errors.push(format!(
                    "restore {} from {}: {error:#}",
                    output.path.display(),
                    backup.display(),
                ));
            }
        }
    }
    if let Err(error) = sync_output_parents(outputs, faults, CommitOperation::RollbackParentsSync) {
        errors.push(format!("durably record rollback: {error:#}"));
    }
    ensure!(errors.is_empty(), "{}", errors.join("; "));
    Ok(())
}

fn cleanup_paths(paths: &[PathBuf], faults: &mut CommitFaultInjector) -> Result<()> {
    let mut errors = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        if let Err(error) = faults.check(CommitOperation::StageCleanup(index)) {
            errors.push(format!("remove staged {}: {error:#}", path.display()));
            continue;
        }
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                errors.push(format!("remove staged {}: {error}", path.display()));
            }
        }
    }
    ensure!(errors.is_empty(), "{}", errors.join("; "));
    Ok(())
}

fn sync_output_parents(
    outputs: &[PendingOutput<'_>],
    faults: &mut CommitFaultInjector,
    operation: CommitOperation,
) -> Result<()> {
    faults.check(operation)?;
    let parents: BTreeSet<_> = outputs
        .iter()
        .map(|output| {
            output
                .path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        })
        .collect();
    for parent in parents {
        let directory = std::fs::File::open(&parent)
            .with_context(|| format!("open output directory {}", parent.display()))?;
        directory
            .sync_all()
            .with_context(|| format!("fsync output directory {}", parent.display()))?;
    }
    Ok(())
}

fn read_limited(path: &Path, max: usize) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "{} is not a regular file",
        path.display()
    );
    ensure!(
        metadata.len() <= u64::try_from(max).unwrap_or(u64::MAX),
        "{} exceeds {max} bytes",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(u64::try_from(max + 1).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= max, "{} exceeds {max} bytes", path.display());
    Ok(bytes)
}

fn read_json_limited<T: for<'de> Deserialize<'de>>(path: &Path, max: usize) -> Result<T> {
    serde_json::from_slice(&read_limited(path, max)?)
        .with_context(|| format!("decode JSON {}", path.display()))
}

/// Read an inherited credential file descriptor to EOF under a hard size cap.
///
/// The descriptor is duplicated first so the caller's fd stays open; the read
/// then consumes the exact byte stream (pipe-friendly, no seek assumptions).
/// Any I/O error, short read, EOF error, or over-cap stream fails closed.
/// Used by the systemd `LoadCredentialEncrypted` / podman `--preserve-fds`
/// interface so plaintext credentials never touch a filesystem path.
fn read_fd_limited(fd: RawFd, max: usize, description: &str) -> Result<Vec<u8>> {
    let duped = unsafe { libc::dup(fd) };
    if duped < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("dup inherited {description} fd {fd}"));
    }
    let file = unsafe { std::fs::File::from_raw_fd(duped) };
    let mut bytes = Vec::new();
    file.take(u64::try_from(max + 1).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read inherited {description} fd {fd}"))?;
    ensure!(
        bytes.len() <= max,
        "inherited {description} fd {fd} exceeds {max} bytes"
    );
    Ok(bytes)
}

fn decode_fixed_b64<const N: usize>(value: &str, description: &str) -> Result<Zeroizing<[u8; N]>> {
    let decoded = Zeroizing::new(
        STANDARD
            .decode(value)
            .with_context(|| format!("decode {description}"))?,
    );
    ensure!(
        decoded.len() == N,
        "{description} must decode to exactly {N} bytes"
    );
    let mut fixed = Zeroizing::new([0; N]);
    fixed.copy_from_slice(&decoded);
    Ok(fixed)
}

fn encode_secret_b64<const N: usize>(secret: Zeroizing<[u8; N]>) -> String {
    STANDARD.encode(secret.as_ref())
}

fn encode_ed25519_seed_b64(key: &SigningKey) -> String {
    encode_secret_b64(Zeroizing::new(key.to_bytes()))
}

fn encode_ml_dsa_seed_b64(key: &MlDsaSigningKey) -> String {
    encode_secret_b64(Zeroizing::new(ml_dsa_sk_to_seed(key)))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn display_path(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .canonicalize()
                .ok()
                .and_then(|parent| path.file_name().map(|name| parent.join(name)))
                .unwrap_or_else(|| path.to_path_buf())
        })
        .display()
        .to_string()
}

fn random_bytes(len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    bytes
}

fn now_unix_u64() -> Result<u64> {
    chrono::Utc::now()
        .timestamp()
        .try_into()
        .map_err(|_| anyhow!("system clock precedes Unix epoch"))
}

#[cfg(test)]
// `print_stderr` is exempted alongside unwrap/expect so a test that has to
// skip (external binary absent) can say so instead of passing silently.
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd as _;

    fn test_authority(
        purpose: AuthorityPurpose,
        deployment_domain: Option<String>,
    ) -> LoadedAuthority {
        let ed = SigningKey::generate(&mut rand::rngs::OsRng);
        let (pq, _) = ml_dsa_generate_keypair();
        let public = public_pair_bytes(&ed.verifying_key(), &pq);
        let domain = deployment_domain
            .unwrap_or_else(|| hyprstream_discovery::verify_deployment_public_ca(&public).unwrap());
        LoadedAuthority {
            bundle: AuthorityBundle {
                schema: AUTHORITY_BUNDLE_SCHEMA.to_owned(),
                purpose,
                deployment_domain: domain,
                public_key_sha256: sha256_hex(&public),
                ed25519: Ed25519Secret::Software {
                    seed_b64: encode_ed25519_seed_b64(&ed),
                    public_b64: STANDARD.encode(ed.verifying_key().as_bytes()),
                },
                ml_dsa_65_seed_b64: encode_ml_dsa_seed_b64(&pq),
                recipient_count: 2,
            },
            ed: LoadedEdSigner::Software(ed),
            pq,
        }
    }

    fn checkpoint_for(log: &AuthorityLogFile) -> AuthorityCheckpointFile {
        let operations = decode_authority_operations(log).unwrap();
        let verified = verify_did_op_log(&log.did, &operations).unwrap();
        authority_checkpoint(&log.deployment_domain, &verified)
    }

    fn verify_with_log(
        public_ca: &[u8],
        log: &AuthorityLogFile,
        checkpoint: &AuthorityCheckpointFile,
        token: &str,
    ) -> Result<hyprstream_discovery::VerifiedDeploymentArtifacts> {
        hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            public_ca,
            &serde_json::to_vec(log)?,
            &serde_json::to_vec(checkpoint)?,
            token,
        )
    }

    fn transaction_fault_case(
        faults: impl IntoIterator<Item = CommitOperation>,
    ) -> (tempfile::TempDir, PathBuf, PathBuf, Result<()>) {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first");
        let second = directory.path().join("second");
        std::fs::write(&first, b"old-first").unwrap();
        std::fs::write(&second, b"old-second").unwrap();
        let result = commit_outputs_with_faults(
            vec![
                PendingOutput::new(&first, b"new-first".to_vec(), 0o600),
                PendingOutput::new(&second, b"new-second".to_vec(), 0o600),
            ],
            &mut CommitFaultInjector::with_faults(faults),
        );
        (directory, first, second, result)
    }

    fn directory_entries(directory: &Path) -> Vec<String> {
        let mut entries: Vec<_> = std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .into_string()
                    .expect("test path is UTF-8")
            })
            .collect();
        entries.sort();
        entries
    }

    #[test]
    fn generated_secret_bundles_zeroize_on_error_for_every_authority_purpose() {
        let purposes = [
            (AuthorityPurpose::Root, 0),
            (AuthorityPurpose::RotatedAuthority, 1),
            (AuthorityPurpose::RegistryDelegatedSigner, 2),
        ];
        for (purpose, counter) in purposes {
            let before = SECRET_BUNDLE_DROPS[counter].load(std::sync::atomic::Ordering::SeqCst);
            let result = (|| -> Result<()> {
                let _authority = test_authority(purpose, None);
                bail!("injected error after guarded seed encoding");
            })();
            assert!(result.is_err());
            assert!(
                SECRET_BUNDLE_DROPS[counter].load(std::sync::atomic::Ordering::SeqCst) > before,
                "secret bundle did not run its wiping drop on the error path"
            );
        }

        let guarded = Zeroizing::new([0xA5; 32]);
        assert_eq!(
            STANDARD.decode(encode_secret_b64(guarded)).unwrap(),
            vec![0xA5; 32]
        );
    }

    #[test]
    fn transaction_faults_before_durable_commit_restore_the_complete_old_set() {
        let faults = [
            CommitOperation::StageCreate(1),
            CommitOperation::StagePermissions(1),
            CommitOperation::StageWrite(1),
            CommitOperation::StageSync(1),
            CommitOperation::StageKeep(1),
            CommitOperation::BackupMetadata(1),
            CommitOperation::BackupCreate(1),
            CommitOperation::BackupPlaceholderKeep(1),
            CommitOperation::BackupRename(0),
            CommitOperation::BackupRename(1),
            CommitOperation::BackupParentsSync,
            CommitOperation::CommitRename(0),
            CommitOperation::CommitRename(1),
            CommitOperation::CommitParentsSync,
        ];
        for fault in faults {
            let (directory, first, second, result) = transaction_fault_case([fault]);
            assert!(result.is_err(), "{fault:?} unexpectedly succeeded");
            assert_eq!(
                std::fs::read(&first).unwrap(),
                b"old-first",
                "{fault:?} did not restore the first output"
            );
            assert_eq!(
                std::fs::read(&second).unwrap(),
                b"old-second",
                "{fault:?} did not restore the second output"
            );
            assert_eq!(
                directory_entries(directory.path()),
                vec!["first", "second"],
                "{fault:?} left unreported transaction state"
            );
        }
    }

    #[test]
    fn transaction_cleanup_faults_return_error_with_recovery_state() {
        let (directory, first, second, result) =
            transaction_fault_case([CommitOperation::BackupDelete(0)]);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("transaction cleanup was incomplete"));
        assert_eq!(std::fs::read(&first).unwrap(), b"new-first");
        assert_eq!(std::fs::read(&second).unwrap(), b"new-second");
        assert!(
            directory_entries(directory.path())
                .iter()
                .any(|name| name.starts_with(".hyprstream-backup-")),
            "failed backup deletion did not preserve the recovery file"
        );

        let (directory, first, second, result) =
            transaction_fault_case([CommitOperation::CleanupParentsSync]);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("durably record backup deletion"));
        assert_eq!(std::fs::read(&first).unwrap(), b"new-first");
        assert_eq!(std::fs::read(&second).unwrap(), b"new-second");
        assert_eq!(directory_entries(directory.path()), vec!["first", "second"]);
    }

    #[test]
    fn transaction_reports_and_preserves_incomplete_rollback_or_staging_cleanup() {
        let (directory, first, second, result) = transaction_fault_case([
            CommitOperation::CommitRename(1),
            CommitOperation::RollbackRemove(0),
        ]);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("ROLLBACK INCOMPLETE"));
        assert_eq!(std::fs::read(&first).unwrap(), b"old-first");
        assert_eq!(std::fs::read(&second).unwrap(), b"old-second");
        assert_eq!(directory_entries(directory.path()), vec!["first", "second"]);

        let (directory, first, second, result) = transaction_fault_case([
            CommitOperation::CommitRename(1),
            CommitOperation::RollbackRestore(1),
        ]);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("ROLLBACK INCOMPLETE"));
        assert_eq!(std::fs::read(&first).unwrap(), b"old-first");
        assert!(!second.exists());
        assert!(
            directory_entries(directory.path())
                .iter()
                .any(|name| name.starts_with(".hyprstream-backup-")),
            "failed restoration did not preserve its backup"
        );

        let (_directory, first, second, result) = transaction_fault_case([
            CommitOperation::CommitRename(1),
            CommitOperation::RollbackParentsSync,
        ]);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("ROLLBACK INCOMPLETE"));
        assert_eq!(std::fs::read(&first).unwrap(), b"old-first");
        assert_eq!(std::fs::read(&second).unwrap(), b"old-second");

        let (directory, first, second, result) = transaction_fault_case([
            CommitOperation::StageCreate(1),
            CommitOperation::StageCleanup(0),
        ]);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("staged-output cleanup incomplete"));
        assert_eq!(std::fs::read(&first).unwrap(), b"old-first");
        assert_eq!(std::fs::read(&second).unwrap(), b"old-second");
        assert_eq!(directory_entries(directory.path()).len(), 3);

        let (directory, first, second, result) =
            transaction_fault_case([CommitOperation::BackupPlaceholderRemove(1)]);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("preserved empty recovery marker"));
        assert_eq!(std::fs::read(&first).unwrap(), b"old-first");
        assert_eq!(std::fs::read(&second).unwrap(), b"old-second");
        let entries = directory_entries(directory.path());
        assert_eq!(entries.len(), 3);
        assert!(
            entries
                .iter()
                .any(|name| name.starts_with(".hyprstream-backup-")),
            "failed placeholder unlink did not preserve and expose its path"
        );
    }

    #[test]
    fn recipient_ring_rejects_duplicates_and_singletons() {
        let one = distinct_recipients(vec!["age1one".to_owned(), "age1one".to_owned()]).unwrap();
        assert_eq!(one.len(), 1);
        let args = MintDeploymentCaArgs {
            public_ca: "ca".into(),
            authority_key: "key".into(),
            authority_log: "log".into(),
            authority_checkpoint: "head".into(),
            recipients: vec!["age1one".to_owned(), "age1one".to_owned()],
            yubikey_recipients: vec![],
            kms_plugin_recipients: vec![],
            piv_slot: None,
            force: false,
        };
        assert!(root_recipient_ring(&args).is_err());
    }

    #[test]
    fn public_pair_layout_is_ed_then_ml_dsa() {
        let ed = SigningKey::generate(&mut rand::rngs::OsRng);
        let (pq, _) = ml_dsa_generate_keypair();
        let bytes = public_pair_bytes(&ed.verifying_key(), &pq);
        assert_eq!(bytes.len(), PUBLIC_CA_BYTES);
        assert_eq!(&bytes[..32], ed.verifying_key().as_bytes());
        assert_eq!(&bytes[32..], ml_dsa_sk_to_vk_bytes(&pq));
        assert!(hyprstream_discovery::verify_deployment_public_ca(&bytes).is_ok());
    }

    #[test]
    fn exact_registry_capability_cannot_widen() {
        let (pq, _) = ml_dsa_generate_keypair();
        let ed = SigningKey::generate(&mut rand::rngs::OsRng);
        let public = public_pair_bytes(&ed.verifying_key(), &pq);
        let capability = registry_mint_capability("domain", &public);
        assert_eq!(capability.ability.as_str(), DELEGATION_ABILITY);
        assert!(!capability.resource.as_str().contains('*'));
        assert_eq!(
            capability.caveats.0["max_ttl_seconds"],
            CaveatValue::Int(3600)
        );
    }

    #[test]
    fn publisher_manifest_has_no_authority_key_field() {
        let value = serde_json::to_value(PublisherManifest {
            schema: PUBLISHER_MANIFEST_SCHEMA.to_owned(),
            deployment_domain: "d".to_owned(),
            profile: REGISTRY_PROFILE.to_owned(),
            audience: REGISTRY_AUDIENCE.to_owned(),
            expires_at: 1,
            public_ca: PublisherArtifact {
                local_path: "ca".to_owned(),
                install_path: "ca".to_owned(),
                encoding: "raw".to_owned(),
                base64: "x".to_owned(),
                sha256: "x".to_owned(),
                size_bytes: 1,
                cloud_secret_store: true,
            },
            authority_log: PublisherArtifact {
                local_path: "log".to_owned(),
                install_path: AUTHORITY_LOG_INSTALL_PATH.to_owned(),
                encoding: "json".to_owned(),
                base64: "eA==".to_owned(),
                sha256: sha256_hex(b"x"),
                size_bytes: 1,
                cloud_secret_store: true,
            },
            authority_checkpoint: PublisherArtifact {
                local_path: "head".to_owned(),
                install_path: AUTHORITY_CHECKPOINT_INSTALL_PATH.to_owned(),
                encoding: "json".to_owned(),
                base64: "eA==".to_owned(),
                sha256: sha256_hex(b"x"),
                size_bytes: 1,
                cloud_secret_store: true,
            },
            registry_jwt: PublisherArtifact {
                local_path: "jwt".to_owned(),
                install_path: "jwt".to_owned(),
                encoding: "utf8".to_owned(),
                base64: "x".to_owned(),
                sha256: "x".to_owned(),
                size_bytes: 1,
                cloud_secret_store: true,
            },
            private_authority_exported: false,
            terraform_state_may_contain_private_authority: false,
        })
        .unwrap();
        let object = value.as_object().unwrap();
        assert!(!object.contains_key("authority_key"));
        assert_eq!(object["private_authority_exported"], false);
    }

    #[test]
    fn publisher_artifact_verification_covers_every_contract_field() {
        let bytes = b"authenticated artifact bytes";
        let valid = PublisherArtifact {
            local_path: "/operator/artifact".to_owned(),
            install_path: PUBLIC_CA_INSTALL_PATH.to_owned(),
            encoding: "raw".to_owned(),
            base64: STANDARD.encode(bytes),
            sha256: sha256_hex(bytes),
            size_bytes: bytes.len(),
            cloud_secret_store: true,
        };
        verify_publisher_artifact(
            &valid,
            bytes,
            "/operator/artifact",
            PUBLIC_CA_INSTALL_PATH,
            "raw",
        )
        .unwrap();

        let mutations: [(&str, fn(&mut PublisherArtifact)); 7] = [
            ("local_path", |artifact| {
                artifact.local_path = "/attacker".to_owned();
            }),
            ("install_path", |artifact| {
                artifact.install_path = "/tmp/attacker".to_owned();
            }),
            ("encoding", |artifact| {
                artifact.encoding = "utf8".to_owned();
            }),
            ("base64", |artifact| {
                artifact.base64 = STANDARD.encode(b"attacker");
            }),
            ("sha256", |artifact| {
                artifact.sha256 = "00".repeat(32);
            }),
            ("size_bytes", |artifact| {
                artifact.size_bytes += 1;
            }),
            ("cloud_secret_store", |artifact| {
                artifact.cloud_secret_store = false;
            }),
        ];
        for (field, mutate) in mutations {
            let mut changed = valid.clone();
            mutate(&mut changed);
            assert!(
                verify_publisher_artifact(
                    &changed,
                    bytes,
                    "/operator/artifact",
                    PUBLIC_CA_INSTALL_PATH,
                    "raw",
                )
                .is_err(),
                "modified contract field {field} was accepted"
            );
        }
    }

    #[test]
    fn delegated_credential_follows_add_then_fails_after_replace() {
        let root = test_authority(AuthorityPurpose::Root, None);
        let public_ca = root.public_bytes();
        let root_rotation_key = HybridRotationKey::new(
            root.ed.verifying_key().to_bytes(),
            ml_dsa_sk_to_vk_bytes(&root.pq),
        )
        .unwrap();
        let genesis = sign_did_op(
            DidOp {
                sequence: 0,
                prev: None,
                rotation_keys: vec![root_rotation_key.clone()],
                signature: placeholder_did_signature(),
            },
            &root.ed,
            &root.pq,
        )
        .unwrap();
        let genesis_log =
            authority_log_from_ops(&root.bundle.deployment_domain, vec![genesis.clone()]).unwrap();
        let genesis_checkpoint = checkpoint_for(&genesis_log);

        let delegated = test_authority(
            AuthorityPurpose::RegistryDelegatedSigner,
            Some(root.bundle.deployment_domain.clone()),
        );
        let delegated_public = delegated.public_bytes();
        let now = now_unix_u64().unwrap();
        let ucan = sign_ucan(
            UcanPayload {
                issuer: Did::from_ed25519(&root.ed.verifying_key().to_bytes()),
                audience: Did::from_ed25519(&delegated.ed.verifying_key().to_bytes()),
                capabilities: vec![registry_mint_capability(
                    &root.bundle.deployment_domain,
                    &delegated_public,
                )],
                not_before: Some(now),
                expiration: Some(now + 3_600),
                nonce: random_bytes(16),
            },
            &root.ed,
            &root.pq,
        )
        .unwrap();
        let artifact = DelegationArtifact {
            schema: DELEGATION_SCHEMA.to_owned(),
            deployment_domain: root.bundle.deployment_domain.clone(),
            authority_log_did: genesis_log.did.clone(),
            delegated_public_key_b64: STANDARD.encode(&delegated_public),
            ucan_b64: STANDARD.encode(ucan.to_cbor().unwrap()),
        };
        let registry = SigningKey::generate(&mut rand::rngs::OsRng);
        let direct_token = encode_registry_jwt(
            &root,
            registry.verifying_key().as_bytes(),
            i64::try_from(now).unwrap(),
            i64::try_from(now + 60).unwrap(),
            None,
        )
        .unwrap();
        let token = encode_registry_jwt(
            &delegated,
            registry.verifying_key().as_bytes(),
            i64::try_from(now).unwrap(),
            i64::try_from(now + 60).unwrap(),
            Some(&URL_SAFE_NO_PAD.encode(serde_json::to_vec(&artifact).unwrap())),
        )
        .unwrap();
        verify_with_log(&public_ca, &genesis_log, &genesis_checkpoint, &direct_token).unwrap();
        verify_with_log(&public_ca, &genesis_log, &genesis_checkpoint, &token).unwrap();

        let added = test_authority(
            AuthorityPurpose::RotatedAuthority,
            Some(root.bundle.deployment_domain.clone()),
        );
        let added_rotation_key = HybridRotationKey::new(
            added.ed.verifying_key().to_bytes(),
            ml_dsa_sk_to_vk_bytes(&added.pq),
        )
        .unwrap();
        let add = sign_did_op(
            DidOp {
                sequence: 1,
                prev: Some(genesis.cid().encode()),
                rotation_keys: vec![root_rotation_key, added_rotation_key.clone()],
                signature: placeholder_did_signature(),
            },
            &root.ed,
            &root.pq,
        )
        .unwrap();
        let add_log = authority_log_from_ops(
            &root.bundle.deployment_domain,
            vec![genesis.clone(), add.clone()],
        )
        .unwrap();
        let add_checkpoint = checkpoint_for(&add_log);
        verify_with_log(&public_ca, &add_log, &add_checkpoint, &token).unwrap();

        let replace = sign_did_op(
            DidOp {
                sequence: 2,
                prev: Some(add.cid().encode()),
                rotation_keys: vec![added_rotation_key],
                signature: placeholder_did_signature(),
            },
            &root.ed,
            &root.pq,
        )
        .unwrap();
        let replace_log =
            authority_log_from_ops(&root.bundle.deployment_domain, vec![genesis, add, replace])
                .unwrap();
        let replace_checkpoint = checkpoint_for(&replace_log);
        assert!(
            verify_with_log(&public_ca, &genesis_log, &replace_checkpoint, &token,).is_err(),
            "historical authority-log prefix matched against a newer checkpoint"
        );
        assert!(verify_with_log(&public_ca, &replace_log, &replace_checkpoint, &token,).is_err());
        assert!(
            verify_with_log(&public_ca, &replace_log, &replace_checkpoint, &direct_token,).is_err()
        );
    }

    #[test]
    fn compact_registry_jwt_rejects_any_whitespace() {
        let root = test_authority(AuthorityPurpose::Root, None);
        let registry = SigningKey::generate(&mut rand::rngs::OsRng);
        let now = chrono::Utc::now().timestamp();
        let token = encode_registry_jwt(
            &root,
            registry.verifying_key().as_bytes(),
            now,
            now + 60,
            None,
        )
        .unwrap();
        assert!(hyprstream_discovery::verify_genesis_deployment_artifacts(
            &root.public_bytes(),
            &token
        )
        .is_ok());
        assert!(
            hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
                &root.public_bytes(),
                b"",
                b"",
                &token,
            )
            .is_err(),
            "enrolled verification inferred direct-root genesis mode from omitted trust files"
        );
        assert!(hyprstream_discovery::verify_genesis_deployment_artifacts(
            &root.public_bytes(),
            &format!("{token}\n")
        )
        .is_err());
    }

    #[test]
    fn trust_refresh_service_unit_only_uses_the_delegated_signer() {
        let unit = trust_refresh_service_unit();
        assert!(unit.contains(
            "--via-delegated-signer /etc/hyprstream/trust/delegated/registry-delegated-signer.age"
        ));
        assert!(unit.contains(
            "--delegation /etc/hyprstream/trust/delegated/registry-signer.delegation.json"
        ));
        assert!(unit.contains("--identity /etc/hyprstream/trust/delegated/refresh-identity"));
        assert!(unit.contains("--jwt /run/hyprstream/credentials/registry-service.jwt"));
        assert!(unit.contains("--force"));
        assert!(
            !unit.contains("--root") && !unit.contains("deployment-ca.age"),
            "refresher unit must never reference the root authority — only the delegated signer"
        );
    }

    #[test]
    fn trust_refresh_service_unit_recreates_the_runtime_credentials_directory() {
        let unit = trust_refresh_service_unit();
        assert!(unit.contains("RuntimeDirectory=hyprstream/credentials"));
        assert!(unit.contains("RuntimeDirectoryMode=0750"));
        assert!(unit.contains("RuntimeDirectoryPreserve=yes"));
        assert!(unit.contains("ReadWritePaths=/run/hyprstream/credentials"));
    }

    #[test]
    fn trust_refresh_timer_unit_uses_the_requested_interval() {
        let unit = trust_refresh_timer_unit("30min");
        assert!(unit.contains("OnUnitActiveSec=30min"));
        assert!(unit.contains("WantedBy=timers.target"));
    }

    #[test]
    fn refresh_interval_accepts_only_plain_time_spans() {
        for valid in ["30min", "45s", "1h", "90s"] {
            validate_refresh_interval(valid).unwrap();
        }
        for invalid in [
            "",
            "30",
            "30 min",
            "0s",
            "015min",
            "30m",
            "5hr",
            "-5min",
            "30min\nOnCalendar=*-*-* *:*:*",
            "30min\n[Install]\nWantedBy=multi-user.target",
        ] {
            assert!(
                validate_refresh_interval(invalid).is_err(),
                "interval {invalid:?} was accepted"
            );
        }
    }

    #[test]
    fn refresh_identity_must_be_a_plaintext_age_identity() {
        validate_age_identity_contents(
            b"# created: 2026-01-01\n# public key: age1example\nAGE-SECRET-KEY-1EXAMPLE\n",
        )
        .unwrap();
        for invalid in [
            &b""[..],
            b"# only comments\n",
            b"age-encryption.org/v1\n-> X25519 ciphertext",
            b"AGE-PLUGIN-YUBIKEY-1EXAMPLE\n",
            b"\xff\xfe not utf-8 \xff",
        ] {
            assert!(
                validate_age_identity_contents(invalid).is_err(),
                "identity contents {invalid:?} were accepted"
            );
        }
    }

    fn current_uid() -> u32 {
        nix::unistd::Uid::effective().as_raw()
    }

    /// A tempdir under the crate directory rather than the system temp dir:
    /// ensure_owned_dir inspects every ancestor of the install path, and the
    /// sticky world-writable system temp dir would be rejected as an ancestor.
    fn owned_ancestry_tempdir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("trust-install-test-")
            .tempdir_in(env!("CARGO_MANIFEST_DIR"))
            .unwrap()
    }

    #[test]
    fn ensure_owned_dir_creates_missing_directory_with_mode() {
        let base = owned_ancestry_tempdir();
        let target = base.path().join("a/b/c");
        ensure_owned_dir(&target, 0o750, current_uid()).unwrap();
        let metadata = std::fs::metadata(&target).unwrap();
        assert!(metadata.is_dir());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o750);
        // Idempotent: a second call against the same real directory succeeds.
        ensure_owned_dir(&target, 0o750, current_uid()).unwrap();
    }

    #[test]
    fn ensure_owned_dir_rejects_symlinked_target() {
        let base = owned_ancestry_tempdir();
        let real = base.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = base.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(ensure_owned_dir(&link, 0o755, current_uid()).is_err());
    }

    #[test]
    fn ensure_owned_dir_rejects_group_writable_existing_component() {
        let base = owned_ancestry_tempdir();
        let mid = base.path().join("mid");
        std::fs::create_dir(&mid).unwrap();
        std::fs::set_permissions(&mid, std::fs::Permissions::from_mode(0o775)).unwrap();
        let error = ensure_owned_dir(&mid.join("leaf"), 0o755, current_uid()).unwrap_err();
        assert!(
            error.to_string().contains("group/world writable"),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn ensure_owned_dir_rejects_symlinked_intermediate_component() {
        let base = owned_ancestry_tempdir();
        let real = base.path().join("real");
        std::fs::create_dir_all(real.join("leaf")).unwrap();
        let mid = base.path().join("mid");
        std::os::unix::fs::symlink(&real, &mid).unwrap();
        // The full path exists and its leaf is a real directory, but it is
        // reached through a symlinked intermediate component.
        let through_link = mid.join("leaf");
        assert!(std::fs::metadata(&through_link).unwrap().is_dir());
        assert!(ensure_owned_dir(&through_link, 0o755, current_uid()).is_err());
    }

    /// Generate a fresh age identity/recipient pair via `age-keygen`, returning
    /// (identity_lines, recipient_string). Returns None when the `age-keygen`
    /// binary is not available (CI without age installed).
    fn age_keypair() -> Option<(String, String)> {
        let output = std::process::Command::new("age-keygen")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        let recipient = text
            .lines()
            .find_map(|line| line.strip_prefix("# public key: ").map(str::to_owned))?;
        let identity = text
            .lines()
            .find(|line| line.starts_with("AGE-SECRET-KEY-1"))?
            .to_owned();
        Some((identity, recipient))
    }

    /// Return true when the `age` binary is available on PATH.
    fn age_available() -> bool {
        std::process::Command::new("age")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    /// Build a temp-dir fixture: encrypts a delegated authority bundle to the
    /// given recipient, writes the ciphertext and identity to temp files, and
    /// returns (delegated_key_path, identity_path, public_bytes).
    fn trial_fixture(
        dir: &Path,
        authority: &LoadedAuthority,
        identity: &str,
        recipient: &str,
    ) -> (PathBuf, PathBuf, Vec<u8>) {
        let public_bytes = authority.public_bytes();
        let plaintext = serialize_secret_json(&authority.bundle).unwrap();
        let encrypted = encrypt_age(&plaintext, &[recipient.to_owned()]).unwrap();
        let delegated_key_path = dir.join("delegated.age");
        std::fs::write(&delegated_key_path, &encrypted).unwrap();
        let identity_path = dir.join("identity");
        std::fs::write(&identity_path, identity).unwrap();
        (delegated_key_path, identity_path, public_bytes)
    }

    fn placeholder_artifact(domain: &str, public_bytes: &[u8]) -> DelegationArtifact {
        DelegationArtifact {
            schema: DELEGATION_SCHEMA.to_owned(),
            deployment_domain: domain.to_owned(),
            authority_log_did: "did:web:placeholder".to_owned(),
            delegated_public_key_b64: STANDARD.encode(public_bytes),
            ucan_b64: STANDARD.encode(b"placeholder"),
        }
    }

    #[test]
    fn trial_decrypt_succeeds_when_identity_matches() {
        if !age_available() {
            eprintln!("skipping: age binary not on PATH");
            return;
        }
        let Some((identity, recipient)) = age_keypair() else {
            eprintln!("skipping: age-keygen binary not on PATH");
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        let authority = test_authority(AuthorityPurpose::RegistryDelegatedSigner, None);
        let (delegated_key_path, identity_path, public_bytes) =
            trial_fixture(dir.path(), &authority, &identity, &recipient);
        let artifact = placeholder_artifact(&authority.bundle.deployment_domain, &public_bytes);

        trial_decrypt_delegated_signer(&delegated_key_path, &identity_path, &artifact)
            .expect("matching identity and artifact must pass trial decrypt");
    }

    #[test]
    fn trial_decrypt_fails_when_identity_does_not_match() {
        if !age_available() {
            eprintln!("skipping: age binary not on PATH");
            return;
        }
        let Some((identity, recipient)) = age_keypair() else {
            eprintln!("skipping: age-keygen binary not on PATH");
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        let authority = test_authority(AuthorityPurpose::RegistryDelegatedSigner, None);
        let (delegated_key_path, _identity_path, public_bytes) =
            trial_fixture(dir.path(), &authority, &identity, &recipient);

        // A different identity that cannot decrypt the ciphertext.
        let wrong_identity = age_keypair().expect("age-keygen available").0;
        let wrong_identity_path = dir.path().join("wrong-identity");
        std::fs::write(&wrong_identity_path, &wrong_identity).unwrap();

        let artifact = placeholder_artifact(&authority.bundle.deployment_domain, &public_bytes);

        let err =
            trial_decrypt_delegated_signer(&delegated_key_path, &wrong_identity_path, &artifact)
                .unwrap_err();
        assert!(
            err.to_string().contains("cannot decrypt"),
            "expected decrypt failure, got: {err}"
        );
    }

    #[test]
    fn trial_decrypt_fails_when_public_key_mismatch() {
        if !age_available() {
            eprintln!("skipping: age binary not on PATH");
            return;
        }
        let Some((identity, recipient)) = age_keypair() else {
            eprintln!("skipping: age-keygen binary not on PATH");
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        let authority = test_authority(AuthorityPurpose::RegistryDelegatedSigner, None);
        let (delegated_key_path, identity_path, _public_bytes) =
            trial_fixture(dir.path(), &authority, &identity, &recipient);

        // Artifact claims a different public key than the one encrypted in the
        // delegated signer ciphertext.
        let other = test_authority(AuthorityPurpose::RegistryDelegatedSigner, None);
        let artifact =
            placeholder_artifact(&authority.bundle.deployment_domain, &other.public_bytes());

        let err = trial_decrypt_delegated_signer(&delegated_key_path, &identity_path, &artifact)
            .unwrap_err();
        assert!(
            err.to_string().contains("does not match"),
            "expected public-key mismatch, got: {err}"
        );
    }

    #[test]
    fn trial_decrypt_fails_when_bundle_is_not_a_delegated_signer() {
        if !age_available() {
            eprintln!("skipping: age binary not on PATH");
            return;
        }
        let Some((identity, recipient)) = age_keypair() else {
            eprintln!("skipping: age-keygen binary not on PATH");
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        // Root authority accidentally installed as the delegated signer.
        let authority = test_authority(AuthorityPurpose::Root, None);
        let (delegated_key_path, identity_path, public_bytes) =
            trial_fixture(dir.path(), &authority, &identity, &recipient);
        let artifact = placeholder_artifact(&authority.bundle.deployment_domain, &public_bytes);

        let err = trial_decrypt_delegated_signer(&delegated_key_path, &identity_path, &artifact)
            .unwrap_err();
        assert!(
            err.to_string().contains("not a RegistryDelegatedSigner"),
            "expected purpose mismatch, got: {err}"
        );
    }

   fn multibase(codec: [u8; 2], key: &[u8]) -> String {
        let mut payload = Vec::with_capacity(2 + key.len());
        payload.extend_from_slice(&codec);
        payload.extend_from_slice(key);
        format!("z{}", bs58::encode(payload).into_string())
    }

    /// Anchor-minting arguments for an iroh-reachable deployment whose
    /// Discovery service is `discovery`, with paths that are never written
    /// (these tests exercise the material, not the commit).
    fn anchor_args(did_web: &str, node_id: &str, discovery: &SigningKey) -> MintAnchorCapsuleArgs {
        let kem = hyprstream_rpc::node_identity::derive_mesh_kem_recipient(discovery)
            .unwrap()
            .public();
        let mesh_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(discovery);
        MintAnchorCapsuleArgs {
            public_ca: PathBuf::from("deployment-ca.hybrid"),
            authority_key: PathBuf::from("deployment-ca.age"),
            authority_log: PathBuf::from("deployment-authority.log.json"),
            authority_checkpoint: PathBuf::from("deployment-authority.head.json"),
            identities: Vec::new(),
            yubikey_identities: Vec::new(),
            software_recovery: false,
            did_web: did_web.to_owned(),
            iroh_node_id: Some(node_id.to_owned()),
            iroh_relay: None,
            quic_endpoint: None,
            quic_sni: None,
            quic_cert_sha256: Vec::new(),
            quic_web_pki: false,
            mesh_kem_x25519: Some(multibase([0xec, 0x01], &kem.eks[0])),
            mesh_kem_mlkem768: Some(multibase([0x8c, 0x24], &kem.eks[1])),
            mesh_pq: Some(multibase([0x91, 0x24], &ml_dsa_sk_to_vk_bytes(&mesh_pq))),
            capsule_out: PathBuf::from("anchor-capsule.cbor"),
            did_json_out: PathBuf::from("did.json"),
            force: false,
        }
    }

    /// The whole point of the minter: what it emits must survive the same
    /// resolution the DID-anchored bootstrap performs, and must hand back the
    /// deployment CA that signed it.
    #[test]
    fn minted_anchor_pair_verifies_through_the_production_resolver() {
        let authority = test_authority(AuthorityPurpose::Root, None);
        let carrier = SigningKey::generate(&mut rand::rngs::OsRng);
        let discovery = SigningKey::generate(&mut rand::rngs::OsRng);
        let node_id =
            hyprstream_rpc::did_key::ed25519_to_did_key(carrier.verifying_key().as_bytes())
                .strip_prefix("did:key:")
                .unwrap()
                .to_owned();
        let args = anchor_args("did:web:staging.example.com", &node_id, &discovery);
        let reach = anchor_reach_endpoint(&args).unwrap();

        let minted = build_anchor_material(&args, &reach, &authority).unwrap();

        // Independently re-run the production verifier over the emitted bytes.
        let verified = verify_anchor_material_offline(
            &minted.at9p_did,
            &args.did_web,
            &minted.document,
            &minted.capsule_bytes,
        )
        .expect("minted anchor capsule must verify through the production resolver");
        assert_eq!(verified.at9p_did, minted.at9p_did);
        assert_eq!(verified.deployment_ca_public, authority.public_bytes());
        assert_eq!(
            verified.discovery_transport.endpoint,
            hyprstream_rpc::transport::EndpointType::Iroh {
                node_id: carrier.verifying_key().to_bytes(),
                direct_addrs: Vec::new(),
                relay_url: None,
            }
        );

        // The document half carries what a remote-node bootstrap additionally
        // demands: a hybrid KEM recipient and exactly one ML-DSA-65 anchor.
        assert_eq!(
            hyprstream_rpc::did_web::mesh_kem_recipient(&minted.document),
            Some(
                hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&discovery)
                    .unwrap()
                    .public()
            )
        );
        assert_eq!(
            hyprstream_rpc::did_web::verification_method_ml_dsa_65_keys(&minted.document).len(),
            1
        );
    }

    /// A node identity capsule (no deployment-reach entry) must be refused, and
    /// the refusal must name the command that produces a usable one.
    #[test]
    fn capsule_without_deployment_reach_is_refused_with_minting_guidance() {
        let did_web = "did:web:staging.example.com";
        let ed = SigningKey::generate(&mut rand::rngs::OsRng);
        let (pq, _) = ml_dsa_generate_keypair();
        let subject =
            HybridKeyPair::new(ed.verifying_key().to_bytes(), ml_dsa_sk_to_vk_bytes(&pq)).unwrap();
        let endpoint =
            ServiceEndpoint::new(Transport::Https, "https://staging.example.com").unwrap();
        let service = ServiceEntry::new("#pds", ServiceType::AtprotoPds, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![subject], vec![service]).unwrap();
        body.also_known_as = Some(vec![did_web.to_owned()]);
        let capsule = sign_capsule_detached(body, &ed, &pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let at9p_did = format!("did:at9p:{}", capsule.cid512().unwrap());
        let document = serde_json::json!({
            "id": did_web,
            "alsoKnownAs": [at9p_did],
        });

        let error = verify_anchor_material_offline(&at9p_did, did_web, &document, &bytes)
            .expect_err("a capsule with no deployment reach must not verify");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("closed deployment-anchor profile violation"),
            "a node capsule must be rejected by the closed anchor profile: {rendered}"
        );
        assert!(
            rendered.contains("mint-anchor-capsule"),
            "rejection must tell the operator how to mint a usable anchor: {rendered}"
        );
    }

    /// A rotated authority whose key differs from the pinned root CA must not
    /// be able to mint an anchor capsule — the capsule subject key would not
    /// be the key the resolver pins from `deployment-ca.hybrid`.
    #[test]
    fn rotated_authority_not_matching_pinned_ca_is_rejected_for_anchor_mint() {
        let root = test_authority(AuthorityPurpose::Root, None);
        let public_ca = root.public_bytes();

        // A rotated authority with a different key.
        let rotated = test_authority(
            AuthorityPurpose::RotatedAuthority,
            Some(root.bundle.deployment_domain.clone()),
        );
        assert_ne!(
            rotated.public_bytes(),
            public_ca,
            "test setup: rotated key must differ from root"
        );

        // Build the authority log so that the rotated key is active.
        let root_rotation_key = HybridRotationKey::new(
            root.ed.verifying_key().to_bytes(),
            ml_dsa_sk_to_vk_bytes(&root.pq),
        )
        .unwrap();
        let rotated_rotation_key = HybridRotationKey::new(
            rotated.ed.verifying_key().to_bytes(),
            ml_dsa_sk_to_vk_bytes(&rotated.pq),
        )
        .unwrap();
        let genesis = sign_did_op(
            DidOp {
                sequence: 0,
                prev: None,
                rotation_keys: vec![root_rotation_key],
                signature: placeholder_did_signature(),
            },
            &root.ed,
            &root.pq,
        )
        .unwrap();
        let add_rotation = sign_did_op(
            DidOp {
                sequence: 1,
                prev: Some(genesis.cid().encode()),
                rotation_keys: vec![rotated_rotation_key],
                signature: placeholder_did_signature(),
            },
            &root.ed,
            &root.pq,
        )
        .unwrap();
        let log =
            authority_log_from_ops(&root.bundle.deployment_domain, vec![genesis, add_rotation])
                .unwrap();
        let verified = validate_authority_log(&public_ca, &log, &checkpoint_for(&log)).unwrap();

        // The rotated key IS active in the rotation log ...
        ensure_active_authority(&rotated, &verified).unwrap();

        // ... but it does NOT match the pinned public CA, so the anchor mint
        // guard (the same one `mint_anchor_capsule` runs before it builds the
        // capsule body) must refuse it. Routing through `ensure_anchor_authority`
        // makes the test causal: remove the CA-binding `ensure!` inside it and
        // this assertion fails.
        assert!(
            rotated.bundle.purpose != AuthorityPurpose::RegistryDelegatedSigner,
            "test setup: rotated must pass the purpose guard"
        );
        assert_ne!(
            rotated.public_bytes(),
            public_ca,
            "the CA-binding check would not fire if the keys matched"
        );
        let error = ensure_anchor_authority(&rotated, &public_ca, &verified)
            .expect_err("a rotated authority not matching the pinned CA must be refused");
        assert!(
            format!("{error:#}").contains("does not match the pinned public CA"),
            "rejection must name the CA-binding mismatch: {error:#}"
        );
    }

    /// A QUIC anchor reach with --quic-web-pki and no pins must emit a
    /// QuicTransport service entry (WebPKI without cert pinning).
    #[test]
    fn quic_web_pki_without_pins_emits_transport_entry() {
        let discovery = SigningKey::generate(&mut rand::rngs::OsRng);
        let node_id =
            hyprstream_rpc::did_key::ed25519_to_did_key(discovery.verifying_key().as_bytes())
                .strip_prefix("did:key:")
                .unwrap()
                .to_owned();
        let mut args = anchor_args("did:web:staging.example.com", &node_id, &discovery);
        args.iroh_node_id = None;
        args.quic_endpoint = Some("203.0.113.5:443".parse().unwrap());
        args.quic_web_pki = true;
        args.quic_sni = Some("staging.example.com".to_owned());
        // No quic_cert_sha256 — pure WebPKI mode.

        let reach = anchor_reach_endpoint(&args).unwrap();
        let doc_service = reach
            .document_service
            .as_ref()
            .expect("WebPKI-without-pins must still emit a QuicTransport entry");
        let service_endpoint = doc_service
            .get("serviceEndpoint")
            .expect("entry must have a serviceEndpoint");
        let uri = service_endpoint
            .get("uri")
            .and_then(|v| v.as_str())
            .expect("entry must have a uri");
        assert!(
            uri.contains("staging.example.com"),
            "uri must carry the SNI hostname: {uri}"
        );
        let webpki = service_endpoint
            .get("webpki")
            .and_then(serde_json::Value::as_bool)
            .expect("entry must have a webpki flag");
        assert!(webpki, "webpki must be true for a --quic-web-pki reach");
        let cert_hashes = service_endpoint
            .get("certHashes")
            .and_then(|v| v.as_array())
            .expect("entry must have a certHashes array");
        assert!(
            cert_hashes.is_empty(),
            "certHashes must be empty when no pins were supplied"
        );
    }

    /// A QUIC anchor reach with hostname SNI and pins must produce a document
    /// service entry whose URI uses the hostname, so that the resolver can
    /// match it against the capsule's QUIC socket address by port.
    #[test]
    fn quic_hostname_sni_with_pins_produces_matchable_entry() {
        let discovery = SigningKey::generate(&mut rand::rngs::OsRng);
        let node_id =
            hyprstream_rpc::did_key::ed25519_to_did_key(discovery.verifying_key().as_bytes())
                .strip_prefix("did:key:")
                .unwrap()
                .to_owned();
        let mut args = anchor_args("did:web:staging.example.com", &node_id, &discovery);
        args.iroh_node_id = None;
        args.quic_endpoint = Some("203.0.113.5:443".parse().unwrap());
        args.quic_sni = Some("staging.example.com".to_owned());
        args.quic_cert_sha256 = vec!["aa".repeat(32)];
        args.quic_web_pki = true;

        let reach = anchor_reach_endpoint(&args).unwrap();
        let doc_service = reach
            .document_service
            .as_ref()
            .expect("hostname-SNI with pins must emit a QuicTransport entry");
        let uri = doc_service
            .get("serviceEndpoint")
            .and_then(|v| v.get("uri"))
            .and_then(|v| v.as_str())
            .expect("entry must have a uri");
        assert!(
            uri.contains("staging.example.com"),
            "uri must use the SNI hostname, not the IP: {uri}"

        );
    }

    // ---- PIV slot occupancy guard ------------------------------------

    /// Verbatim stderr from `ykman piv keys info 9c` against a YubiKey
    /// whose slot holds no key.
    const YKMAN_NO_KEY: &str = "ERROR: No key stored in slot 9C (SIGNATURE).\n";
    /// Verbatim stderr from `ykman piv certificates export 9c -` against a
    /// YubiKey whose slot holds no certificate.
    const YKMAN_NO_CERT: &str = "ERROR: No certificate found.\n";

    /// The only state that permits a destructive import: both objects
    /// definitively reported absent.
    #[test]
    fn piv_slot_empty_only_when_both_probes_report_absent() {
        assert!(!slot_occupied_from_probes(
            (Some(1), YKMAN_NO_KEY),
            (Some(1), YKMAN_NO_CERT),
        ));
    }

    /// The primary scenario the guard exists for: `piv_import_ed25519`
    /// imports a key without minting a certificate, so a certificate-only
    /// probe reads that slot as empty. The key probe must catch it.
    #[test]
    fn piv_slot_with_key_but_no_certificate_is_occupied() {
        assert!(slot_occupied_from_probes(
            (Some(0), ""),
            (Some(1), YKMAN_NO_CERT),
        ));
    }

    /// The mirror case: a certificate whose private key lives elsewhere.
    #[test]
    fn piv_slot_with_certificate_but_no_key_is_occupied() {
        assert!(slot_occupied_from_probes(
            (Some(1), YKMAN_NO_KEY),
            (Some(0), "")
        ));
    }

    /// Exit 1 carrying a message OTHER than the absence marker is a
    /// transient failure, not an empty slot. Either probe failing this
    /// way must refuse the import.
    #[test]
    fn piv_slot_transient_failure_is_not_absence() {
        let transient = "ERROR: Failed to connect to YubiKey.\n";
        assert!(slot_occupied_from_probes(
            (Some(1), transient),
            (Some(1), YKMAN_NO_CERT),
        ));
        assert!(slot_occupied_from_probes(
            (Some(1), YKMAN_NO_KEY),
            (Some(1), transient),
        ));
    }

    /// Exit 2 is how `ykman` reports a subcommand or argument it does not
    /// recognise. It must never read as "slot is empty" — that inversion
    /// is exactly how a wrong command name turns the guard into a
    /// rubber stamp.
    #[test]
    fn piv_slot_argument_error_is_not_absence() {
        assert!(slot_occupied_from_probes(
            (Some(2), "Error: No such command 'list'.\n"),
            (Some(1), YKMAN_NO_CERT),
        ));
    }

    /// Absence requires BOTH the exit code and the marker. Matching the
    /// marker alone would let any failure that happens to quote the
    /// message — a usage error echoing it, a wrapper relaying it —
    /// clear the guard.
    #[test]
    fn piv_slot_absence_marker_alone_is_not_absence() {
        assert!(slot_occupied_from_probes(
            (Some(2), "Usage error near: No key stored in slot 9C\n"),
            (Some(1), YKMAN_NO_CERT),
        ));
        assert!(slot_occupied_from_probes(
            (Some(1), YKMAN_NO_KEY),
            (Some(2), "Usage error near: No certificate found\n"),
        ));
    }

    /// A probe killed by a signal reports no exit code at all.
    #[test]
    fn piv_slot_signal_kill_is_not_absence() {
        assert!(slot_occupied_from_probes(
            (None, ""),
            (Some(1), YKMAN_NO_CERT)
        ));
    }

    /// The classification tests above all feed the probes synthetic
    /// output, so none of them can catch the failure that actually
    /// shipped: an argv naming a subcommand `ykman` does not have. This
    /// test runs the real binary and asserts each probe's subcommand is
    /// recognised — `ykman` exits 0 on `--help` for a command it has and
    /// 2 for one it does not, without touching a YubiKey.
    ///
    /// Skipped when `ykman` is not installed, so the suite stays
    /// hermetic; on any host that has it, a renamed or invented
    /// subcommand fails here instead of silently inverting the guard.
    #[test]
    fn ykman_probe_subcommands_exist_in_the_real_binary() {
        for probe in [PIV_KEY_PROBE, PIV_CERT_PROBE] {
            // Everything up to the slot placeholder is the subcommand path.
            let subcommand: Vec<&str> = probe
                .argv
                .iter()
                .copied()
                .take_while(|arg| *arg != "{slot}")
                .collect();
            let mut command = Command::new("ykman");
            let status = command
                .args(&subcommand)
                .arg("--help")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let Ok(status) = status else {
                eprintln!("ykman not installed; skipping subcommand existence check");
                return;
            };
            assert_eq!(
                status.code(),
                Some(0),
                "ykman does not recognise the subcommand `{}` used by the PIV \
                 slot probe",
                subcommand.join(" ")
            );
        }
    }

    // ---- inherited-FD credential interface (#1561) ---------------------

    /// Write `bytes` into a pipe and return the read end, emulating an
    /// inherited, non-seekable credential fd (systemd LoadCredentialEncrypted
    /// + podman --preserve-fds).
    fn fd_pipe(bytes: &[u8]) -> std::fs::File {
        let mut fds = [0_i32; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "pipe failed");
        let mut writer = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        writer.write_all(bytes).expect("write pipe");
        drop(writer); // closing the write end gives the reader a clean EOF
        unsafe { std::fs::File::from_raw_fd(fds[0]) }
    }

    fn age_keygen_to_file(path: &Path) -> String {
        let output = Command::new("age-keygen")
            .output()
            .expect("launch age-keygen");
        assert!(output.status.success(), "age-keygen failed");
        std::fs::write(path, &output.stdout).unwrap();
        let text = String::from_utf8(output.stdout).unwrap();
        for line in text.lines() {
            let line = line.trim();
            for prefix in ["# public key: ", "Public key: "] {
                if let Some(recipient) = line.strip_prefix(prefix) {
                    return recipient.to_owned();
                }
            }
        }
        panic!("age-keygen printed no recipient");
    }

    struct MintFdFixture {
        // Holds the tempdir open for the fixture's lifetime.
        _dir: tempfile::TempDir,
        public_ca: PathBuf,
        authority_key: PathBuf,
        authority_log: PathBuf,
        authority_checkpoint: PathBuf,
        delegation: PathBuf,
        registry_public_key: PathBuf,
        delegated_key_bytes: Vec<u8>,
        signer_identity_bytes: Vec<u8>,
        registry_key_bytes: [u8; 32],
    }

    /// Run the real path-form ceremony (mint-deployment-ca +
    /// delegate-registry-signer) into a tempdir, returning everything the
    /// FD-form mint needs.
    fn mint_fd_fixture() -> MintFdFixture {
        let dir = tempfile::tempdir().unwrap();
        let root_identity = dir.path().join("root.identity");
        let root_recipient = age_keygen_to_file(&root_identity);
        let backup_identity = dir.path().join("backup.identity");
        let backup_recipient = age_keygen_to_file(&backup_identity);
        let signer_identity = dir.path().join("signer.identity");
        let signer_recipient = age_keygen_to_file(&signer_identity);

        let public_ca = dir.path().join("deployment-ca.hybrid");
        let authority_key = dir.path().join("deployment-ca.age");
        let authority_log = dir.path().join("deployment-authority.log.json");
        let authority_checkpoint = dir.path().join("deployment-authority.head.json");
        mint_deployment_ca(&MintDeploymentCaArgs {
            public_ca: public_ca.clone(),
            authority_key: authority_key.clone(),
            authority_log: authority_log.clone(),
            authority_checkpoint: authority_checkpoint.clone(),
            recipients: vec![root_recipient, backup_recipient],
            yubikey_recipients: vec![],
            kms_plugin_recipients: vec![],
            piv_slot: None,
            force: false,
        })
        .unwrap();

        let delegated_key = dir.path().join("registry-delegated-signer.age");
        let delegation = dir.path().join("registry-signer.delegation.json");
        delegate_registry_signer(&DelegateRegistrySignerArgs {
            public_ca: public_ca.clone(),
            authority_log: authority_log.clone(),
            authority_checkpoint: authority_checkpoint.clone(),
            authority_key: authority_key.clone(),
            identities: vec![root_identity],
            yubikey_identities: vec![],
            software_recovery: false,
            signer_recipients: vec![signer_recipient],
            delegated_key: delegated_key.clone(),
            delegation: delegation.clone(),
            delegation_ttl_seconds: 2_592_000,
            force: false,
        })
        .unwrap();

        let registry_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let registry_public_key = dir.path().join("registry-public-key");
        std::fs::write(
            &registry_public_key,
            registry_key.verifying_key().as_bytes(),
        )
        .unwrap();

        MintFdFixture {
            delegated_key_bytes: std::fs::read(&delegated_key).unwrap(),
            signer_identity_bytes: std::fs::read(&signer_identity).unwrap(),
            registry_key_bytes: registry_key.verifying_key().to_bytes(),
            _dir: dir,
            public_ca,
            authority_key,
            authority_log,
            authority_checkpoint,
            delegation,
            registry_public_key,
        }
    }

    fn mint_fd_args(
        fixture: &MintFdFixture,
        signer_fd: RawFd,
        identity_fd: RawFd,
        out: &Path,
    ) -> MintRegistryJwtArgs {
        MintRegistryJwtArgs {
            public_ca: fixture.public_ca.clone(),
            authority_key: fixture.authority_key.clone(),
            identities: vec![],
            identity_fds: vec![identity_fd],
            yubikey_identities: vec![],
            software_recovery: false,
            via_delegated_signer: None,
            via_delegated_signer_fd: Some(signer_fd),
            delegation: Some(fixture.delegation.clone()),
            authority_log: fixture.authority_log.clone(),
            authority_checkpoint: fixture.authority_checkpoint.clone(),
            root: false,
            registry_public_key: fixture.registry_public_key.clone(),
            ttl_seconds: 3600,
            jwt: out.join("registry-service.jwt"),
            contract: out.join("deployment-trust.contract.json"),
            force: false,
        }
    }

    #[test]
    fn mint_registry_jwt_via_inherited_fds_round_trips_through_production_verifier() {
        let fixture = mint_fd_fixture();
        let out = tempfile::tempdir().unwrap();
        let signer = fd_pipe(&fixture.delegated_key_bytes);
        let identity = fd_pipe(&fixture.signer_identity_bytes);
        let args = mint_fd_args(
            &fixture,
            signer.as_raw_fd(),
            identity.as_raw_fd(),
            out.path(),
        );
        mint_registry_jwt(&args).expect("FD-form mint must succeed");

        // The same production verifier the registry runs at boot must accept
        // the credential minted purely from inherited fds.
        let token = std::fs::read_to_string(out.path().join("registry-service.jwt")).unwrap();
        let verified = hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            &std::fs::read(&fixture.public_ca).unwrap(),
            &std::fs::read(&fixture.authority_log).unwrap(),
            &std::fs::read(&fixture.authority_checkpoint).unwrap(),
            &token,
        )
        .expect("production verifier must accept the FD-minted credential");
        assert_eq!(verified.registry_public_key, fixture.registry_key_bytes);
    }

    #[test]
    fn mint_registry_jwt_fd_flags_conflict_with_path_forms() {
        use clap::Subcommand as _;
        let parse = |extra: &[&str]| {
            TrustCommand::augment_subcommands(clap::Command::new("hyprstream"))
                .try_get_matches_from(
                    ["hyprstream", "mint-registry-jwt"]
                        .into_iter()
                        .chain(extra.iter().copied()),
                )
        };
        // Signer path + signer fd conflict.
        assert!(parse(&[
            "--via-delegated-signer",
            "k.age",
            "--via-delegated-signer-fd",
            "3",
            "--identity",
            "id",
        ])
        .is_err());
        // Identity path + identity fd conflict.
        assert!(parse(&[
            "--via-delegated-signer",
            "k.age",
            "--identity",
            "id",
            "--identity-fd",
            "4",
        ])
        .is_err());
        // --root + signer fd conflict.
        assert!(parse(&[
            "--root",
            "--via-delegated-signer-fd",
            "3",
            "--identity",
            "id"
        ])
        .is_err());
        // Exactly one signer source is required unless --root.
        assert!(parse(&["--identity", "id"]).is_err());
        // The pure FD form parses.
        if let Err(error) = parse(&[
            "--via-delegated-signer-fd",
            "3",
            "--identity-fd",
            "4",
            "--delegation",
            "d.json",
            "--registry-public-key",
            "r",
        ]) {
            panic!("FD form must parse: {error}");
        }
        // The path forms are unchanged.
        assert!(parse(&[
            "--via-delegated-signer",
            "k.age",
            "--identity",
            "id",
            "--delegation",
            "d.json",
            "--registry-public-key",
            "r",
        ])
        .is_ok());
    }

    #[test]
    fn mint_registry_jwt_fd_read_failures_fail_closed() {
        let fixture = mint_fd_fixture();
        let run = |signer_fd: RawFd, identity_fd: RawFd| {
            let out = tempfile::tempdir().unwrap();
            let args = mint_fd_args(&fixture, signer_fd, identity_fd, out.path());
            let result = mint_registry_jwt(&args);
            (out, result)
        };

        // An invalid descriptor must fail the mint, not fall back anywhere.
        let identity = fd_pipe(&fixture.signer_identity_bytes);
        let (out, result) = run(-1, identity.as_raw_fd());
        assert!(result.is_err(), "invalid signer fd must fail closed");
        assert!(!out.path().join("registry-service.jwt").exists());

        // A write-only descriptor yields a read error, not an empty secret.
        let write_only = OpenOptions::new().write(true).open("/dev/null").unwrap();
        let identity = fd_pipe(&fixture.signer_identity_bytes);
        let (out, result) = run(write_only.as_raw_fd(), identity.as_raw_fd());
        assert!(result.is_err(), "unreadable signer fd must fail closed");
        assert!(!out.path().join("registry-service.jwt").exists());

        // An over-cap identity stream is rejected before decryption.
        let signer = fd_pipe(&fixture.delegated_key_bytes);
        let oversize = fd_pipe(vec![b'x'; MAX_AGE_IDENTITY_BYTES + 1].as_slice());
        let (out, result) = run(signer.as_raw_fd(), oversize.as_raw_fd());
        assert!(result.is_err(), "oversize identity fd must fail closed");
        assert!(!out.path().join("registry-service.jwt").exists());

        // An empty identity stream is rejected before decryption.
        let signer = fd_pipe(&fixture.delegated_key_bytes);
        let empty = fd_pipe(b"");
        let (out, result) = run(signer.as_raw_fd(), empty.as_raw_fd());
        assert!(result.is_err(), "empty identity fd must fail closed");
        assert!(!out.path().join("registry-service.jwt").exists());

        // A truncated signer ciphertext (short read / EOF mid-stream) must not
        // mint a partial credential.
        let truncated =
            fd_pipe(&fixture.delegated_key_bytes[..fixture.delegated_key_bytes.len() / 2]);
        let identity = fd_pipe(&fixture.signer_identity_bytes);
        let (out, result) = run(truncated.as_raw_fd(), identity.as_raw_fd());
        assert!(result.is_err(), "truncated signer fd must fail closed");
        assert!(!out.path().join("registry-service.jwt").exists());

        // Identity bytes that cannot open the ciphertext fail in age.
        let signer = fd_pipe(&fixture.delegated_key_bytes);
        let wrong = fd_pipe(b"# not an identity\n");
        let (out, result) = run(signer.as_raw_fd(), wrong.as_raw_fd());
        assert!(
            result.is_err(),
            "non-decrypting identity fd must fail closed"
        );
        assert!(!out.path().join("registry-service.jwt").exists());
    }
}
