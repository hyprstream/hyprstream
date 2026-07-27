//! Deployment trust-authority minting.
//!
//! The root authority is an out-of-band, age-encrypted operator asset. Hourly
//! registry credentials should be minted by a narrowly scoped delegated signer;
//! direct root signing exists only for bootstrap and recovery.

#![allow(clippy::print_stdout)]

use crate::cli::commands::{
    DelegateRegistrySignerArgs, MintDeploymentCaArgs, MintRegistryJwtArgs, RotateAuthorityArgs,
    TrustCommand, VerifyDeploymentArgs,
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
    DeploymentAuthorityLog as AuthorityLogFile, RegistryDelegationArtifact as DelegationArtifact,
};
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
    os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use zeroize::Zeroize as _;

const PUBLIC_CA_BYTES: usize = 32 + 1_952;
const HYBRID_SIGNATURE_BYTES: usize = 3_309 + 64;
const REGISTRY_AUDIENCE: &str = "urn:hyprstream:service:registry";
const REGISTRY_PROFILE: &str = "hyprstream.registry-deployment.v1";
const AUTHORITY_BUNDLE_SCHEMA: &str = "hyprstream.deployment-authority.v1";
const AUTHORITY_LOG_SCHEMA: &str = "hyprstream.deployment-authority-log.v1";
const DELEGATION_SCHEMA: &str = "hyprstream.registry-delegation.v1";
const PUBLISHER_MANIFEST_SCHEMA: &str = "hyprstream.deployment-trust-publisher-manifest.v1";
const DELEGATION_RESOURCE_PREFIX: &str = "hyprstream://deployment";
const DELEGATION_ABILITY: &str = "mint-registry-jwt";
const MAX_AUTHORITY_LOG_OPERATIONS: usize = 128;
const MAX_DELEGATION_BYTES: usize = 256 * 1024;
const MAX_CLOUD_SECRET_BYTES: usize = 64 * 1024;

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
    }
}

#[derive(Debug, Serialize)]
struct CaMintOutput {
    schema: &'static str,
    deployment_domain: String,
    authority_log_did: String,
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
    authority_log: Option<PublisherArtifact>,
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
    }
}

fn mint_deployment_ca(args: &MintDeploymentCaArgs) -> Result<()> {
    preflight_outputs(
        [&args.public_ca, &args.authority_key, &args.authority_log],
        args.force,
    )?;
    let recipients = root_recipient_ring(args)?;

    let (ed_secret, ed_signer) = match args.piv_slot.as_deref() {
        Some(slot) => {
            let recovery_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let (slot, public) = piv_import_ed25519(slot, &recovery_key)?;
            (
                Ed25519Secret::YubikeyPiv {
                    slot: slot.clone(),
                    public_b64: STANDARD.encode(public.as_bytes()),
                    recovery_seed_b64: STANDARD.encode(recovery_key.to_bytes()),
                },
                LoadedEdSigner::YubikeyPiv { slot, public },
            )
        }
        None => {
            let key = SigningKey::generate(&mut rand::rngs::OsRng);
            (
                Ed25519Secret::Software {
                    seed_b64: STANDARD.encode(key.to_bytes()),
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
    validate_authority_log(&public_ca, &authority_log)?;

    let bundle = AuthorityBundle {
        schema: AUTHORITY_BUNDLE_SCHEMA.to_owned(),
        purpose: AuthorityPurpose::Root,
        deployment_domain: deployment_domain.clone(),
        public_key_sha256: sha256_hex(&public_ca),
        ed25519: ed_secret,
        ml_dsa_65_seed_b64: STANDARD.encode(ml_dsa_sk_to_seed(&pq)),
        recipient_count: recipients.len(),
    };
    let mut plaintext = serde_json::to_vec(&bundle).context("encode authority bundle")?;
    let encrypted = encrypt_age(&plaintext, &recipients)?;
    plaintext.zeroize();

    commit_outputs(vec![
        PendingOutput::new(&args.public_ca, public_ca.clone(), 0o644),
        PendingOutput::new(&args.authority_key, encrypted, 0o600),
        PendingOutput::new(
            &args.authority_log,
            pretty_json_bytes(&authority_log)?,
            0o644,
        ),
    ])?;

    let output = CaMintOutput {
        schema: "hyprstream.deployment-ca-mint-output.v1",
        deployment_domain,
        authority_log_did: authority_log.did,
        public_ca_path: display_path(&args.public_ca),
        public_ca_install_path: "/etc/hyprstream/trust/deployment-ca.hybrid",
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
    let active = validate_authority_log(&public_ca, &log)?;

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
            seed_b64: STANDARD.encode(delegated_ed.to_bytes()),
            public_b64: STANDARD.encode(delegated_ed.verifying_key().as_bytes()),
        },
        ml_dsa_65_seed_b64: STANDARD.encode(ml_dsa_sk_to_seed(&delegated_pq)),
        recipient_count: signer_recipients.len(),
    };
    let mut plaintext = serde_json::to_vec(&delegated_bundle)?;
    let encrypted = encrypt_age(&plaintext, &signer_recipients)?;
    plaintext.zeroize();

    let artifact = DelegationArtifact {
        schema: DELEGATION_SCHEMA.to_owned(),
        deployment_domain: authority.bundle.deployment_domain.clone(),
        authority_log_did: log.did.clone(),
        delegated_public_key_b64: STANDARD.encode(&delegated_public),
        ucan_b64: STANDARD.encode(ucan.to_cbor()?),
    };
    validate_delegation_artifact(&public_ca, &log, &artifact, now)?;
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
    let identities = combined_identities(&args.identities, &args.yubikey_identities)?;

    let (signer, delegation_b64, authority_log) = if args.root {
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
        let authority_log = args
            .authority_log
            .as_ref()
            .map(|path| read_json_limited::<AuthorityLogFile>(path, MAX_CLOUD_SECRET_BYTES))
            .transpose()?;
        if let Some(log) = authority_log.as_ref() {
            let active = validate_authority_log(&public_ca, log)?;
            ensure_active_authority(&authority, &active)?;
        }
        (authority, None, authority_log)
    } else {
        let delegated_path = args
            .via_delegated_signer
            .as_ref()
            .ok_or_else(|| anyhow!("--via-delegated-signer is required"))?;
        let artifact_path = args
            .delegation
            .as_ref()
            .ok_or_else(|| anyhow!("--delegation is required"))?;
        let artifact: DelegationArtifact = read_json_limited(artifact_path, MAX_DELEGATION_BYTES)?;
        let authority_log_path = args
            .authority_log
            .as_ref()
            .ok_or_else(|| anyhow!("--authority-log is required with delegated signing"))?;
        let installed_log: AuthorityLogFile =
            read_json_limited(authority_log_path, MAX_CLOUD_SECRET_BYTES)?;
        validate_authority_log(&public_ca, &installed_log)?;
        validate_delegation_artifact(&public_ca, &installed_log, &artifact, now_unix_u64()?)?;
        let delegated = decrypt_authority(delegated_path, &identities, args.software_recovery)?;
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
        (
            delegated,
            Some(URL_SAFE_NO_PAD.encode(artifact_json)),
            Some(installed_log),
        )
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
    let verified = match authority_log.as_ref() {
        Some(log) => hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            &public_ca,
            &serde_json::to_vec(log)?,
            &token,
        ),
        None => hyprstream_discovery::verify_deployment_artifacts(&public_ca, &token),
    }
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
            install_path: "/etc/hyprstream/trust/deployment-ca.hybrid".to_owned(),
            encoding: "raw".to_owned(),
            base64: STANDARD.encode(&public_ca),
            sha256: sha256_hex(&public_ca),
            size_bytes: public_ca.len(),
            cloud_secret_store: true,
        },
        authority_log: authority_log
            .as_ref()
            .map(|log| {
                let path = args
                    .authority_log
                    .as_ref()
                    .ok_or_else(|| anyhow!("authority log path disappeared"))?;
                let bytes = read_limited(path, MAX_CLOUD_SECRET_BYTES)?;
                let persisted: AuthorityLogFile = serde_json::from_slice(&bytes)?;
                ensure!(
                    &persisted == log,
                    "authority log changed while building publisher manifest"
                );
                Ok::<PublisherArtifact, anyhow::Error>(PublisherArtifact {
                    local_path: display_path(path),
                    install_path: "/etc/hyprstream/trust/deployment-authority.log.json".to_owned(),
                    encoding: "json".to_owned(),
                    base64: STANDARD.encode(&bytes),
                    sha256: sha256_hex(&bytes),
                    size_bytes: bytes.len(),
                    cloud_secret_store: true,
                })
            })
            .transpose()?,
        registry_jwt: PublisherArtifact {
            local_path: display_path(&args.jwt),
            install_path: "/run/hyprstream/credentials/registry-service.jwt".to_owned(),
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
    let token = String::from_utf8(read_limited(&args.jwt, MAX_CLOUD_SECRET_BYTES)?)
        .context("registry JWT is not UTF-8")?;
    ensure!(
        !token.is_empty() && !token.bytes().any(|byte| byte.is_ascii_whitespace()),
        "registry JWT must be compact with no surrounding or embedded whitespace"
    );
    let contract = args
        .contract
        .as_ref()
        .map(|path| read_json_limited::<PublisherManifest>(path, 1024 * 1024))
        .transpose()?;
    let authority_log = match (&args.authority_log, contract.as_ref()) {
        (Some(path), _) => Some(read_limited(path, MAX_CLOUD_SECRET_BYTES)?),
        (None, Some(contract)) => contract
            .authority_log
            .as_ref()
            .map(|artifact| STANDARD.decode(&artifact.base64))
            .transpose()?,
        (None, None) => None,
    };
    let verified = match authority_log.as_deref() {
        Some(log) => hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            &public_ca, log, &token,
        ),
        None => hyprstream_discovery::verify_deployment_artifacts(&public_ca, &token),
    }?;
    if let Some(contract) = contract {
        ensure!(
            contract.schema == PUBLISHER_MANIFEST_SCHEMA,
            "unsupported contract schema"
        );
        ensure!(
            STANDARD.decode(&contract.public_ca.base64)? == public_ca,
            "contract public CA does not match file"
        );
        ensure!(
            STANDARD.decode(&contract.registry_jwt.base64)? == token.as_bytes(),
            "contract JWT does not match file"
        );
        if let Some(artifact) = contract.authority_log.as_ref() {
            let log = authority_log
                .as_ref()
                .ok_or_else(|| anyhow!("contract authority log was not verified"))?;
            ensure!(
                STANDARD.decode(&artifact.base64)? == *log,
                "contract authority log does not match verified bytes"
            );
        }
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
    let active = validate_authority_log(&public_ca, &log)?;
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
    validate_authority_log(&public_ca, &next_log)?;

    let new_bundle = AuthorityBundle {
        schema: AUTHORITY_BUNDLE_SCHEMA.to_owned(),
        purpose: AuthorityPurpose::RotatedAuthority,
        deployment_domain: log.deployment_domain.clone(),
        public_key_sha256: sha256_hex(&new_public),
        ed25519: Ed25519Secret::Software {
            seed_b64: STANDARD.encode(new_ed.to_bytes()),
            public_b64: STANDARD.encode(new_ed.verifying_key().as_bytes()),
        },
        ml_dsa_65_seed_b64: STANDARD.encode(ml_dsa_sk_to_seed(&new_pq)),
        recipient_count: recipients.len(),
    };
    let mut plaintext = serde_json::to_vec(&new_bundle)?;
    let encrypted = encrypt_age(&plaintext, &recipients)?;
    plaintext.zeroize();

    commit_outputs(vec![
        PendingOutput::new(&args.new_public_key, new_public, 0o644),
        PendingOutput::new(&args.new_authority_key, encrypted, 0o600),
        PendingOutput::new(
            &args.authority_log_out,
            pretty_json_bytes(&next_log)?,
            0o644,
        ),
    ])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hyprstream.authority-rotation-output.v1",
            "deployment_domain": next_log.deployment_domain,
            "authority_log_did": next_log.did,
            "sequence": next_log.operations_b64.len() - 1,
            "mode": if args.add { "add" } else { "replace" },
            "new_public_key_path": display_path(&args.new_public_key),
            "new_authority_key_path": display_path(&args.new_authority_key),
            "authority_log_path": display_path(&args.authority_log_out)
        }))?
    );
    Ok(())
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

fn combined_identities(generic: &[PathBuf], yubikey: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let identities: Vec<_> = generic.iter().chain(yubikey).cloned().collect();
    ensure!(
        !identities.is_empty(),
        "at least one --identity or --yubikey-identity is required"
    );
    for identity in &identities {
        ensure!(
            identity.is_file(),
            "age identity file does not exist: {}",
            identity.display()
        );
    }
    Ok(identities)
}

fn encrypt_age(plaintext: &[u8], recipients: &[String]) -> Result<Vec<u8>> {
    ensure!(!recipients.is_empty(), "no age recipients supplied");
    let mut command = Command::new("age");
    command
        .arg("--encrypt")
        .arg("--output")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    for recipient in recipients {
        command.arg("--recipient").arg(recipient);
    }
    command.arg("-");
    let mut child = command.spawn().context("launch age encryption")?;
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("age stdin unavailable"))?
        .write_all(plaintext)
        .context("write authority plaintext to age")?;
    let output = child
        .wait_with_output()
        .context("wait for age encryption")?;
    ensure!(output.status.success(), "age encryption failed");
    ensure!(!output.stdout.is_empty(), "age produced empty ciphertext");
    Ok(output.stdout)
}

fn decrypt_age(path: &Path, identities: &[PathBuf]) -> Result<Vec<u8>> {
    let mut command = Command::new("age");
    command
        .arg("--decrypt")
        .arg("--output")
        .arg("-")
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    for identity in identities {
        command.arg("--identity").arg(identity);
    }
    command.arg(path);
    let output = command.output().context("launch age decryption")?;
    ensure!(output.status.success(), "age decryption failed");
    ensure!(
        output.stdout.len() <= 128 * 1024,
        "decrypted authority bundle is too large"
    );
    Ok(output.stdout)
}

fn decrypt_authority(
    path: &Path,
    identities: &[PathBuf],
    software_recovery: bool,
) -> Result<LoadedAuthority> {
    let mut plaintext = decrypt_age(path, identities)?;
    let bundle: AuthorityBundle =
        serde_json::from_slice(&plaintext).context("decode authority bundle")?;
    plaintext.zeroize();
    ensure!(
        bundle.schema == AUTHORITY_BUNDLE_SCHEMA,
        "unsupported authority bundle schema"
    );
    let mut pq_seed = decode_fixed_b64::<32>(&bundle.ml_dsa_65_seed_b64, "ML-DSA-65 seed")?;
    let pq = ml_dsa_sk_from_seed(&pq_seed);
    pq_seed.zeroize();
    let ed = match &bundle.ed25519 {
        Ed25519Secret::Software {
            seed_b64,
            public_b64,
        } => {
            ensure!(
                !software_recovery,
                "--software-recovery applies only to a PIV-backed authority"
            );
            let mut seed = decode_fixed_b64::<32>(seed_b64, "Ed25519 seed")?;
            let key = SigningKey::from_bytes(&seed);
            seed.zeroize();
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
                let mut seed =
                    decode_fixed_b64::<32>(recovery_seed_b64, "PIV recovery Ed25519 seed")?;
                let key = SigningKey::from_bytes(&seed);
                seed.zeroize();
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

fn validate_authority_log(
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
    artifact: &DelegationArtifact,
    now: u64,
) -> Result<()> {
    ensure!(
        artifact.schema == DELEGATION_SCHEMA,
        "unsupported delegation schema"
    );
    let active = validate_authority_log(public_ca, authority_log)?;
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
        if path.exists() && !force {
            bail!(
                "refusing to replace existing output {} without --force",
                path.display()
            );
        }
        if let Ok(metadata) = std::fs::symlink_metadata(path) {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "refusing non-regular or symlink output target {}",
                path.display()
            );
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "output parent does not exist: {}",
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

/// Stage every output before moving any destination, then roll back the whole
/// set if a rename fails. Each rename is filesystem-atomic; the rollback closes
/// the ordinary error path across the multi-file mint commands.
fn commit_outputs(outputs: Vec<PendingOutput<'_>>) -> Result<()> {
    let mut staged = Vec::with_capacity(outputs.len());
    for output in &outputs {
        let parent = output.path.parent().unwrap_or_else(|| Path::new("."));
        let mut temp = tempfile::NamedTempFile::new_in(parent)
            .with_context(|| format!("create staged output for {}", output.path.display()))?;
        temp.as_file_mut()
            .set_permissions(std::fs::Permissions::from_mode(output.mode))?;
        temp.write_all(&output.bytes)?;
        temp.as_file_mut().sync_all()?;
        let (_file, path) = temp.keep().context("retain staged output")?;
        staged.push(path);
    }

    let mut backups: Vec<Option<PathBuf>> = vec![None; outputs.len()];
    for (index, output) in outputs.iter().enumerate() {
        if output.path.exists() {
            let parent = output.path.parent().unwrap_or_else(|| Path::new("."));
            let placeholder = tempfile::Builder::new()
                .prefix(".hyprstream-backup-")
                .tempfile_in(parent)
                .with_context(|| format!("stage backup for {}", output.path.display()))?;
            let backup = placeholder.path().to_path_buf();
            drop(placeholder);
            if let Err(error) = std::fs::rename(output.path, &backup) {
                restore_outputs(&outputs, &backups, 0);
                cleanup_paths(&staged);
                return Err(error)
                    .with_context(|| format!("back up existing {}", output.path.display()));
            }
            backups[index] = Some(backup);
        }
    }

    for (index, output) in outputs.iter().enumerate() {
        if let Err(error) = std::fs::rename(&staged[index], output.path) {
            restore_outputs(&outputs, &backups, index);
            cleanup_paths(&staged[index..]);
            return Err(error).with_context(|| format!("commit {}", output.path.display()));
        }
    }

    for backup in backups.into_iter().flatten() {
        let _ = std::fs::remove_file(backup);
    }
    sync_output_parents(&outputs);
    Ok(())
}

fn restore_outputs(outputs: &[PendingOutput<'_>], backups: &[Option<PathBuf>], committed: usize) {
    for output in outputs.iter().take(committed) {
        let _ = std::fs::remove_file(output.path);
    }
    for (output, backup) in outputs.iter().zip(backups).rev() {
        if let Some(backup) = backup {
            let _ = std::fs::rename(backup, output.path);
        }
    }
    sync_output_parents(outputs);
}

fn cleanup_paths(paths: &[PathBuf]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

fn sync_output_parents(outputs: &[PendingOutput<'_>]) {
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
        if let Ok(directory) = std::fs::File::open(parent) {
            let _ = directory.sync_all();
        }
    }
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

fn decode_fixed_b64<const N: usize>(value: &str, description: &str) -> Result<[u8; N]> {
    STANDARD
        .decode(value)
        .with_context(|| format!("decode {description}"))?
        .try_into()
        .map_err(|_| anyhow!("{description} must decode to exactly {N} bytes"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn display_path(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
                    seed_b64: STANDARD.encode(ed.to_bytes()),
                    public_b64: STANDARD.encode(ed.verifying_key().as_bytes()),
                },
                ml_dsa_65_seed_b64: STANDARD.encode(ml_dsa_sk_to_seed(&pq)),
                recipient_count: 2,
            },
            ed: LoadedEdSigner::Software(ed),
            pq,
        }
    }

    #[test]
    fn recipient_ring_rejects_duplicates_and_singletons() {
        let one = distinct_recipients(vec!["age1one".to_owned(), "age1one".to_owned()]).unwrap();
        assert_eq!(one.len(), 1);
        let args = MintDeploymentCaArgs {
            public_ca: "ca".into(),
            authority_key: "key".into(),
            authority_log: "log".into(),
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
            authority_log: None,
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
        hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            &public_ca,
            &serde_json::to_vec(&genesis_log).unwrap(),
            &direct_token,
        )
        .unwrap();
        hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            &public_ca,
            &serde_json::to_vec(&genesis_log).unwrap(),
            &token,
        )
        .unwrap();

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
        hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
            &public_ca,
            &serde_json::to_vec(&add_log).unwrap(),
            &token,
        )
        .unwrap();

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
        assert!(
            hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
                &public_ca,
                &serde_json::to_vec(&replace_log).unwrap(),
                &token,
            )
            .is_err()
        );
        assert!(
            hyprstream_discovery::verify_deployment_artifacts_with_authority_log(
                &public_ca,
                &serde_json::to_vec(&replace_log).unwrap(),
                &direct_token,
            )
            .is_err()
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
        assert!(
            hyprstream_discovery::verify_deployment_artifacts(&root.public_bytes(), &token).is_ok()
        );
        assert!(hyprstream_discovery::verify_deployment_artifacts(
            &root.public_bytes(),
            &format!("{token}\n")
        )
        .is_err());
    }
}
