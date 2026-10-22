//! S4 policy bootload (#570): the missing daemon-boot step that installs a
//! **verified** [`CompiledPolicy`] so the MAC PDP inputs are real.
//!
//! Everything else in the S4 stack was already built (the
//! [`LatticeTeEvaluator`](crate::mac::te::LatticeTeEvaluator) PDP, the
//! [`CachingAvc`](crate::mac::avc::CachingAvc), the [`PolicyLoader`], the
//! [`SignedPolicy`](crate::mac::compiled::SignedPolicy) sign/verify path), but
//! **nothing ever ran the compile → sign → verify-load → install chain at
//! startup**. So [`crate::mac::compiled_policy`] (the process-global `OnceLock`)
//! stayed `None`, and every downstream seam fell back to deny-all — in
//! particular [`crate::mac::exchange_enrollment_resolver`] returned the
//! [`DenyUnlabeledResolver`](crate::services::oauth::token_exchange::DenyUnlabeledResolver),
//! which is what kept #698's already-merged
//! [`EnrollmentSubjectContextResolver`](crate::mac::EnrollmentSubjectContextResolver)
//! inert. This module is the one missing wire.
//!
//! Installing a compiled policy makes the PDP inputs real: it flips
//! [`crate::mac::compiled_policy`] to `Some`, so the enrollment resolver
//! resolves genuine clearances instead of denying every DID. The active 9P PEP
//! records the loaded generation and policy hash on every audited decision.
//!
//! ## Fail-closed
//!
//! The signer is hybrid (EdDSA + ML-DSA-65 COSE composite) and the loader
//! verifies with `require_pq = true` (design §14 — never Ed25519-only).
//! Configured artifacts require only the enrolled authority's public keys;
//! absent configuration, a node without an active ML-DSA signing key installs
//! no baseline (the resolver keeps denying), never a classical-only baseline.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context};
use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::ucan::approval::{ApprovalBinding, SignedApproval};
use hyprstream_rpc::auth::ucan::token::{Did, Ucan, UcanPayload};
use hyprstream_rpc::crypto::pq::MlDsaSigningKey;

use crate::mac::compiled::cose::{HybridPolicySigner, HybridPolicyVerifier};
use crate::mac::compiled::{
    sign_policy, CompiledPolicy, PolicyApproval, PolicyDistError, PolicyLoader, SignedPolicy,
};
use crate::mac::compiler::{compile, PermissionMap};
use crate::mac::install_compiled_policy;
use crate::mac::lattice::{Lattice, LatticeVersion, SecurityLabel};
use crate::mac::permission_map::ScopePermissionMap;
use crate::mac::te::SubjectType;

/// Errors from the boot path. All fail-closed: on any error nothing is installed
/// and [`crate::mac::compiled_policy`] stays `None` (the resolver keeps denying).
#[derive(Debug, thiserror::Error)]
pub enum BootPolicyError {
    /// Signing, hashing, or the verify-once-at-load step failed.
    #[error("policy bootload failed: {0}")]
    Policy(#[from] PolicyDistError),
}

/// Load explicitly configured artifacts against the OS-owned deployment's
/// chain-enrolled Policy authority, then install before the baseline OnceLock.
/// No signing keys, runtime approval, unanchored roster, or fallback are used.
/// DID-anchored deployments are intentionally unsupported by this bounded path.
pub fn install_configured_boot_policy(
    artifacts: &crate::config::MacBootPolicyConfig,
    secrets_dir: &Path,
    is_os_owned_bootstrap: bool,
) -> anyhow::Result<Arc<CompiledPolicy>> {
    ensure!(
        is_os_owned_bootstrap,
        "configured MAC boot policy requires OS-owned deployment trust"
    );
    let entries = crate::auth::identity_store::load_bootstrap_pubkeys_hybrid(secrets_dir)?;
    crate::auth::identity_store::ensure_bootstrap_pubkeys_hybrid(&entries)?;
    crate::auth::identity_store::ensure_bootstrap_pubkeys_enrolled(secrets_dir, &entries)?;
    let authority = entries
        .get("policy")
        .context("no enrolled Policy authority in bootstrap roster")?;
    let policy = Arc::new(load_boot_policy_files(artifacts, authority)?);
    ensure!(
        install_compiled_policy(policy.clone()),
        "MAC policy already installed; configured policy cannot replace write-once seam"
    );
    Ok(policy)
}

fn load_boot_policy_files(
    artifacts: &crate::config::MacBootPolicyConfig,
    authority: &crate::auth::identity_store::BootstrapPubkey,
) -> anyhow::Result<CompiledPolicy> {
    let signed: SignedPolicy = serde_json::from_slice(
        &std::fs::read(&artifacts.policy_path).context("cannot read signed MAC policy")?,
    )
    .context("invalid signed MAC policy JSON")?;
    let approval: SignedApproval = serde_json::from_slice(
        &std::fs::read(&artifacts.approval_path).context("cannot read signed MAC approval")?,
    )
    .context("invalid signed MAC approval JSON")?;
    let policy = verify_boot_policy(&signed, &approval, authority)?;
    let expected_hash: [u8; 32] = hex::decode(&artifacts.expected_policy_hash)
        .context("expected MAC boot-policy hash must be hexadecimal")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected MAC boot-policy hash must be exactly 32 bytes"))?;
    check_reviewed_policy(&policy, artifacts.expected_generation, &expected_hash)
        .context("configured MAC boot-policy rollback pin rejected")?;
    Ok(policy)
}

fn verify_boot_policy(
    signed: &SignedPolicy,
    approval: &SignedApproval,
    authority: &crate::auth::identity_store::BootstrapPubkey,
) -> anyhow::Result<CompiledPolicy> {
    let pq_vk = authority
        .ml_dsa_65
        .as_ref()
        .context("Policy authority lacks ML-DSA-65 key")?;
    let binding = approval
        .verify(&authority.ed25519, pq_vk)
        .context("MAC policy approval signature rejected")?;
    let verifier = HybridPolicyVerifier {
        ed_vk: &authority.ed25519,
        pq_vk: Some(pq_vk),
        require_pq: true,
    };
    let policy = PolicyLoader::new(verifier)
        .with_approval(PolicyApproval::from_verified(binding))
        .load(signed)?;
    let (allow, escalate) = policy.matrix.sorted_rules();
    ensure!(
        allow.is_empty() && escalate.is_empty(),
        "configured MAC boot policy is enrollment-only: TE allow and escalation rules must be empty"
    );
    Ok(policy)
}

/// One-shot offline authority action, never called by daemon boot. The operator
/// explicitly reviews an enrollment-only policy and empty UCAN; this
/// does not infer approval or validate a delegated UCAN chain on their behalf.
/// Both objects are signed by the existing enrolled Policy authority with
/// distinct signature domains. No new keys or trust ceremony are created.
pub fn sign_reviewed_boot_policy(
    args: &crate::cli::commands::trust::SignMacBootPolicyArgs,
) -> anyhow::Result<()> {
    let policy: CompiledPolicy = serde_json::from_slice(
        &std::fs::read(&args.policy_input).context("cannot read reviewed policy")?,
    )
    .context("invalid reviewed CompiledPolicy JSON")?;
    let grant: Ucan = serde_json::from_slice(
        &std::fs::read(&args.reviewed_ucan).context("cannot read reviewed UCAN")?,
    )
    .context("invalid reviewed UCAN JSON")?;
    let expected_hash: [u8; 32] = hex::decode(&args.expected_hash)
        .context("expected hash must be hexadecimal")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected hash must be exactly 32 bytes"))?;
    // Review mismatch must fail before even loading the private signer.
    check_reviewed_policy(&policy, args.expected_generation, &expected_hash)?;
    check_enrollment_only(&policy, &grant)?;
    let entries = crate::auth::identity_store::load_bootstrap_pubkeys_hybrid(&args.secrets_dir)?;
    crate::auth::identity_store::ensure_bootstrap_pubkeys_hybrid(&entries)?;
    crate::auth::identity_store::ensure_bootstrap_pubkeys_enrolled(&args.secrets_dir, &entries)?;
    let authority = entries
        .get("policy")
        .context("no enrolled Policy authority in bootstrap roster")?;
    let ed_sk = crate::auth::identity_store::load_existing_service_signing_key(
        &args.secrets_dir,
        "policy",
        crate::auth::identity_store::SecretsProfile::SharedDirectory,
    )?;
    let (signed, approval) = sign_reviewed_policy(
        &policy,
        &grant,
        args.expected_generation,
        &expected_hash,
        &ed_sk,
        authority,
    )?;
    publish_boot_artifacts(&args.output_dir, &signed, &approval)
}

fn check_reviewed_policy(
    policy: &CompiledPolicy,
    generation: u64,
    hash: &[u8; 32],
) -> anyhow::Result<()> {
    ensure!(
        policy.generation == generation,
        "reviewed policy generation mismatch"
    );
    ensure!(
        policy.lattice()?.version().generation() == generation,
        "reviewed lattice generation mismatch"
    );
    ensure!(
        policy.policy_hash()? == *hash,
        "reviewed policy hash mismatch (including enrollment)"
    );
    Ok(())
}

fn check_enrollment_only(policy: &CompiledPolicy, grant: &Ucan) -> anyhow::Result<()> {
    let (allow, escalate) = policy.matrix.sorted_rules();
    ensure!(
        allow.is_empty() && escalate.is_empty() && grant.payload.capabilities.is_empty(),
        "boot artifact authoring is enrollment-only: TE rules and UCAN capabilities must be empty"
    );
    Ok(())
}

fn sign_reviewed_policy(
    policy: &CompiledPolicy,
    grant: &Ucan,
    generation: u64,
    hash: &[u8; 32],
    ed_sk: &SigningKey,
    authority: &crate::auth::identity_store::BootstrapPubkey,
) -> anyhow::Result<(SignedPolicy, SignedApproval)> {
    check_reviewed_policy(policy, generation, hash)?;
    check_enrollment_only(policy, grant)?;
    let signer_public = crate::auth::identity_store::BootstrapPubkey::for_service_key(ed_sk)?;
    ensure!(
        authority.is_hybrid() && signer_public.to_key_bytes() == authority.to_key_bytes(),
        "existing Policy signer does not match the enrolled hybrid authority"
    );
    let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(ed_sk);
    let signed = sign_policy(
        policy,
        &HybridPolicySigner {
            ed_sk,
            pq_sk: Some(&pq_sk),
        },
    )?;
    let approval = SignedApproval::sign(
        ApprovalBinding::new(grant, generation, *hash)?,
        ed_sk,
        &pq_sk,
    )?;
    // Exercise exactly the production verification path before publishing.
    verify_boot_policy(&signed, &approval, authority)?;
    Ok((signed, approval))
}

fn publish_boot_artifacts(
    output_dir: &Path,
    signed: &SignedPolicy,
    approval: &SignedApproval,
) -> anyhow::Result<()> {
    use std::io::Write as _;
    let parent = output_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staged = tempfile::tempdir_in(parent).context("cannot stage MAC boot artifacts")?;
    for (name, bytes) in [
        ("signed-policy.json", serde_json::to_vec(signed)?),
        ("signed-approval.json", serde_json::to_vec(approval)?),
    ] {
        let mut file = std::fs::File::create(staged.path().join(name))?;
        file.write_all(&bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            file.set_permissions(std::fs::Permissions::from_mode(0o644))?;
        }
        file.sync_all()?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(staged.path(), std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::File::open(staged.path())?.sync_all()?;
    // Exclusively reserve a NEW empty destination. Renaming the staged directory
    // over this reservation publishes both files together; no existing output
    // (even an empty directory) is overwritten. A failure may leave only the
    // empty reservation, which cannot be loaded as a policy.
    std::fs::create_dir(output_dir).context("MAC boot output directory must not exist")?;
    std::fs::rename(staged.path(), output_dir).context("cannot publish MAC boot artifacts")?;
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}

/// Compile a validated grant into a TE policy, hybrid-PQC-sign it, verify it
/// **once at load**, and install it process-globally. Returns the installed
/// [`CompiledPolicy`].
///
/// The signer and verifier use the supplied policy-authority keypair and run
/// the artifact through the full [`sign_policy`] → [`PolicyLoader::load`] path
/// (rather than short-circuiting the crypto). `require_pq = true` — the ML-DSA
/// outer layer is mandatory; a classical-only signature is rejected.
///
/// The caller MUST have already validated `grant`'s UCAN chain — `compile`
/// reads only its capability set. Enrollment is the authority-owned
/// DID→clearance table (#698), embedded in and covered by the signature.
/// `approval` MUST come from an upstream-verified S5 approval binding (normally
/// [`PolicyApproval::from_verified`]); this generic path never derives an
/// approval from the policy it is loading. The loader independently recomputes
/// the compiled policy hash and rejects a mismatched approval.
///
/// ## Return: `(policy, installed)` — the write-once truth
///
/// Returns the policy **this** call compiled+verified together with an
/// `installed` flag from [`install_compiled_policy`]: `true` iff this call's
/// policy actually became the process-global seam, `false` if one was already
/// installed and the global slot is unchanged. The returned `Arc` is always this
/// call's own artifact regardless — but when `installed == false`, that artifact
/// is NOT what [`crate::mac::compiled_policy`] returns.
///
/// ## Write-once limitation (do not build a real-policy swap on this yet)
///
/// The seam is a `OnceLock`: the **first** installed policy pins it for the
/// process lifetime. So this API installs a real policy ONLY on a node that has
/// not already installed the boot baseline. Configured boot artifacts are
/// loaded before the baseline; they cannot replace it afterward. Swapping a
/// newer verified generation in atomically (e.g. `ArcSwap<CompiledPolicy>` keyed
/// by `generation`) is the follow-up; until then, treat `installed == false` as
/// "the earlier policy still governs the seam", never as a successful swap.
///
/// ## Key provenance
///
/// `ed_sk` and `pq_sk` are the policy authority's hybrid signing keys. The
/// approval is a separate reviewed binding supplied by the caller and checked
/// independently from the artifact signature.
pub fn compile_sign_load_install(
    grant: &Ucan,
    lattice: &Lattice,
    permissions: &impl PermissionMap,
    enrollment: BTreeMap<String, SecurityLabel>,
    approval: PolicyApproval,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<(Arc<CompiledPolicy>, bool), BootPolicyError> {
    // Control plane: lower the grant → TE matrix (S5 compiler) and attach the
    // authority-owned enrollment. Both travel inside the signature.
    let policy = compile(grant, lattice, permissions).with_enrollment(enrollment);

    sign_load_install(policy, approval, ed_sk, pq_sk)
}

/// Sign, verify-load, and install an already compiled policy. The approval is
/// always explicit here: arbitrary policy compilation has no self-approval
/// escape hatch.
fn sign_load_install(
    policy: CompiledPolicy,
    approval: PolicyApproval,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<(Arc<CompiledPolicy>, bool), BootPolicyError> {
    // Sign hybrid (EdDSA + ML-DSA-65 COSE composite; cose_sign::sign_composite
    // under the HybridPolicySigner adapter).
    let signer = HybridPolicySigner {
        ed_sk,
        pq_sk: Some(pq_sk),
    };
    let signed = sign_policy(&policy, &signer)?;

    // Verify ONCE at load (require_pq = true, fail-closed) against the policy
    // authority keys, then independently enforce the supplied approval hash.
    let ed_vk = ed_sk.verifying_key();
    // `verifying_key` is provided by the `ml_dsa::Keypair` trait; call it
    // fully-qualified (matching `MlDsaKeySlot::verifying_key`) so this module
    // needs no extra trait import.
    let pq_vk = ml_dsa::Keypair::verifying_key(pq_sk).clone();
    let verifier = HybridPolicyVerifier {
        ed_vk: &ed_vk,
        pq_vk: Some(&pq_vk),
        require_pq: true,
    };
    let loaded = PolicyLoader::new(verifier)
        .with_approval(approval)
        .load(&signed)?;

    let policy = Arc::new(loaded);
    let installed = install_compiled_policy(policy.clone());
    Ok((policy, installed))
}

/// Construct and install the one artifact allowed to derive its approval
/// locally: the dormant, empty baseline. This helper accepts no grant,
/// enrollment, lattice, permission map, or approval from its caller, so the
/// self-approval exception cannot be reused for a non-empty policy.
fn install_self_approved_empty_baseline(
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<(Arc<CompiledPolicy>, bool), BootPolicyError> {
    let did = Did::from_ed25519(&ed_sk.verifying_key().to_bytes());
    let grant = Ucan {
        payload: UcanPayload {
            issuer: did.clone(),
            audience: did,
            capabilities: vec![],
            not_before: None,
            expiration: None,
            nonce: vec![],
        },
        proofs: vec![],
        signature: vec![],
    };

    let lattice = Lattice::new(LatticeVersion(1), []);
    let permissions = ScopePermissionMap::new(SubjectType(1), Vec::<(String, String)>::new());
    let policy = compile(&grant, &lattice, &permissions).with_enrollment(BTreeMap::new());
    let approval = PolicyApproval {
        generation: policy.generation,
        approved_hash: policy.policy_hash()?,
    };

    sign_load_install(policy, approval, ed_sk, pq_sk)
}

/// Install the node's baseline compiled policy at daemon boot.
///
/// The baseline is intentionally minimal: a self-issued empty grant over an
/// empty lattice generation and an empty object registry, so the compiled TE
/// matrix is empty (grants nothing) and no subject is enrolled. Its only effect
/// is to make [`crate::mac::compiled_policy`] return `Some`, which switches the
/// grant path's subject resolver from deny-all to the real
/// [`EnrollmentSubjectContextResolver`](crate::mac::EnrollmentSubjectContextResolver).
/// The active 9P PEP uses this policy as decision provenance.
///
/// Unlike [`compile_sign_load_install`], this path locally derives an approval;
/// that exception is confined to a private helper which constructs the empty
/// policy internally and accepts no caller-controlled policy inputs. At daemon
/// boot this is the first (and, given the `OnceLock` write-once seam, only)
/// install, so `installed` is normally `true`; a `false` means a policy was
/// already installed this process.
///
/// A future policy-authoring step (config-driven UCAN grant + enrollment) will
/// need a **swap-capable** seam to replace this baseline at runtime — it cannot
/// do so through the current write-once [`install_compiled_policy`] (see that
/// function's and [`compile_sign_load_install`]'s docs). The empty baseline is
/// the fail-closed default until that seam lands.
pub fn install_baseline_boot_policy(
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<(Arc<CompiledPolicy>, bool), BootPolicyError> {
    install_self_approved_empty_baseline(ed_sk, pq_sk)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::mac::compiled_policy;
    use crate::mac::lattice::{Assurance, CompartmentSet, Level};
    use crate::mac::EnrollmentSubjectContextResolver;
    use crate::services::oauth::token_exchange::SubjectContextResolver as _;
    use hyprstream_rpc::auth::ucan::approval::{ApprovalBinding, SignedApproval};
    use hyprstream_rpc::auth::ucan::capability::{Ability, Capability, Resource};
    use hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair;
    use rand::rngs::OsRng;

    fn cap(res: &str, ab: &str) -> Capability {
        Capability::new(Resource::new(res), Ability::new(ab))
    }

    fn grant_with(caps: Vec<Capability>) -> Ucan {
        let did = Did::from_ed25519(&[0u8; 32]);
        Ucan {
            payload: UcanPayload {
                issuer: did.clone(),
                audience: did,
                capabilities: caps,
                not_before: None,
                expiration: None,
                nonce: vec![],
            },
            proofs: vec![],
            signature: vec![],
        }
    }

    fn boot_artifacts() -> (
        SignedPolicy,
        SignedApproval,
        crate::auth::identity_store::BootstrapPubkey,
        SigningKey,
        MlDsaSigningKey,
    ) {
        boot_artifacts_at_generation(7)
    }

    fn boot_artifacts_at_generation(
        generation: u32,
    ) -> (
        SignedPolicy,
        SignedApproval,
        crate::auth::identity_store::BootstrapPubkey,
        SigningKey,
        MlDsaSigningKey,
    ) {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let policy = CompiledPolicy::new(
            crate::mac::te::TeMatrix::default(),
            &Lattice::new(LatticeVersion(generation), []),
        )
        .with_enrollment(BTreeMap::from([(
            "did:key:zApprovedBootFixture".to_owned(),
            SecurityLabel::new(
                Level::Internal,
                Assurance::Unverified,
                CompartmentSet::EMPTY,
            ),
        )]));
        let signed = sign_policy(
            &policy,
            &HybridPolicySigner {
                ed_sk: &ed,
                pq_sk: Some(&pq),
            },
        )
        .unwrap();
        let binding = ApprovalBinding::new(
            &grant_with(vec![]),
            policy.generation,
            policy.policy_hash().unwrap(),
        )
        .unwrap();
        let approval = SignedApproval::sign(binding, &ed, &pq).unwrap();
        let authority =
            crate::auth::identity_store::BootstrapPubkey::hybrid(ed.verifying_key(), pq_vk);
        (signed, approval, authority, ed, pq)
    }

    #[test]
    fn configured_boot_files_load_reviewed_enrollment_without_te_authority() {
        let (signed, approval, authority, _, _) = boot_artifacts();
        let expected = verify_boot_policy(&signed, &approval, &authority).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config = crate::config::MacBootPolicyConfig {
            policy_path: dir.path().join("policy.json"),
            approval_path: dir.path().join("approval.json"),
            expected_generation: expected.generation,
            expected_policy_hash: hex::encode(expected.policy_hash().unwrap()),
        };
        std::fs::write(&config.policy_path, serde_json::to_vec(&signed).unwrap()).unwrap();
        std::fs::write(
            &config.approval_path,
            serde_json::to_vec(&approval).unwrap(),
        )
        .unwrap();
        let policy = load_boot_policy_files(&config, &authority).unwrap();
        assert_eq!(policy.matrix.allow_len(), 0);
        assert_eq!(
            policy.clearance_for("did:key:zApprovedBootFixture"),
            Some(SecurityLabel::new(
                Level::Internal,
                Assurance::Unverified,
                CompartmentSet::EMPTY
            ))
        );
        let resolver = EnrollmentSubjectContextResolver::new(Arc::new(policy));
        assert!(resolver.resolve("did:key:zApprovedBootFixture").is_some());
        assert!(resolver.resolve("did:key:zNotEnrolled").is_none());

        std::fs::remove_file(&config.approval_path).unwrap();
        assert!(load_boot_policy_files(&config, &authority).is_err());
        std::fs::write(&config.approval_path, b"{}").unwrap();
        assert!(load_boot_policy_files(&config, &authority).is_err());
        std::fs::remove_file(&config.policy_path).unwrap();
        assert!(load_boot_policy_files(&config, &authority).is_err());
    }

    #[test]
    fn configured_boot_pin_rejects_older_valid_pair_after_restart_reload() {
        let (current_signed, current_approval, authority, ed, pq) = boot_artifacts_at_generation(7);
        let current = verify_boot_policy(&current_signed, &current_approval, &authority).unwrap();
        let older = CompiledPolicy::new(
            crate::mac::te::TeMatrix::default(),
            &Lattice::new(LatticeVersion(6), []),
        )
        .with_enrollment(current.enrollment.clone());
        let older_signed = sign_policy(
            &older,
            &HybridPolicySigner {
                ed_sk: &ed,
                pq_sk: Some(&pq),
            },
        )
        .unwrap();
        let older_binding = ApprovalBinding::new(
            &grant_with(vec![]),
            older.generation,
            older.policy_hash().unwrap(),
        )
        .unwrap();
        let older_approval = SignedApproval::sign(older_binding, &ed, &pq).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config = crate::config::MacBootPolicyConfig {
            policy_path: dir.path().join("policy.json"),
            approval_path: dir.path().join("approval.json"),
            expected_generation: current.generation,
            expected_policy_hash: hex::encode(current.policy_hash().unwrap()),
        };

        // A process restart rereads the same OS-owned generation/hash pin.
        // The current pair therefore remains loadable after restart.
        std::fs::write(
            &config.policy_path,
            serde_json::to_vec(&current_signed).unwrap(),
        )
        .unwrap();
        std::fs::write(
            &config.approval_path,
            serde_json::to_vec(&current_approval).unwrap(),
        )
        .unwrap();
        assert_eq!(
            load_boot_policy_files(&config, &authority)
                .unwrap()
                .generation,
            7
        );
        assert_eq!(
            load_boot_policy_files(&config, &authority)
                .unwrap()
                .generation,
            7
        );

        // Both older artifacts are genuinely signed by the trusted authority
        // and approve each other. Replacing them together still cannot roll
        // back the independently provisioned OS-owned pin.
        std::fs::write(
            &config.policy_path,
            serde_json::to_vec(&older_signed).unwrap(),
        )
        .unwrap();
        std::fs::write(
            &config.approval_path,
            serde_json::to_vec(&older_approval).unwrap(),
        )
        .unwrap();
        let error = load_boot_policy_files(&config, &authority).unwrap_err();
        assert!(
            format!("{error:#}").contains("generation mismatch"),
            "{error:#}"
        );

        // Recovery is restoring the matching backed-up release, not accepting
        // whichever valid signed pair happens to be on disk.
        std::fs::write(
            &config.policy_path,
            serde_json::to_vec(&current_signed).unwrap(),
        )
        .unwrap();
        std::fs::write(
            &config.approval_path,
            serde_json::to_vec(&current_approval).unwrap(),
        )
        .unwrap();
        assert_eq!(
            load_boot_policy_files(&config, &authority)
                .unwrap()
                .generation,
            7
        );
    }

    #[test]
    fn configured_boot_rejects_validly_signed_allow_and_escalation_rules() {
        use crate::mac::te::{Action, ObjectType, TeMatrix, TeRule};
        let (signed, approval, authority, ed, pq) = boot_artifacts();
        let enrolled = verify_boot_policy(&signed, &approval, &authority).unwrap();
        let rule = TeRule {
            subject_type: SubjectType(1),
            object_type: ObjectType(1),
            action: Action(1),
        };
        let dir = tempfile::tempdir().unwrap();
        let policy_path = dir.path().join("policy.json");
        let approval_path = dir.path().join("approval.json");
        for matrix in [
            TeMatrix::from_allow([rule]),
            TeMatrix::new(Default::default(), [rule].into_iter().collect()),
        ] {
            let mut policy = enrolled.clone();
            policy.matrix = matrix;
            let signed = sign_policy(
                &policy,
                &HybridPolicySigner {
                    ed_sk: &ed,
                    pq_sk: Some(&pq),
                },
            )
            .unwrap();
            let config = crate::config::MacBootPolicyConfig {
                policy_path: policy_path.clone(),
                approval_path: approval_path.clone(),
                expected_generation: policy.generation,
                expected_policy_hash: hex::encode(policy.policy_hash().unwrap()),
            };
            let approval = SignedApproval::sign(
                ApprovalBinding::new(
                    &grant_with(vec![]),
                    policy.generation,
                    policy.policy_hash().unwrap(),
                )
                .unwrap(),
                &ed,
                &pq,
            )
            .unwrap();
            // Both signatures and the exact hash/generation are genuine.
            // The configured file-loader boundary must still reject TE authority.
            std::fs::write(&config.policy_path, serde_json::to_vec(&signed).unwrap()).unwrap();
            std::fs::write(
                &config.approval_path,
                serde_json::to_vec(&approval).unwrap(),
            )
            .unwrap();
            let error = load_boot_policy_files(&config, &authority).unwrap_err();
            assert!(error.to_string().contains("enrollment-only"), "{error:#}");
        }
    }

    #[test]
    fn configured_boot_rejects_tampering_wrong_keys_and_unapproved_hash() {
        let (signed, approval, authority, ed, pq) = boot_artifacts();
        let (other_signed, other_approval, other_authority, _, _) = boot_artifacts();
        assert!(verify_boot_policy(&signed, &approval, &other_authority).is_err());
        assert!(verify_boot_policy(&other_signed, &approval, &authority).is_err());
        assert!(verify_boot_policy(&signed, &other_approval, &authority).is_err());
        let mut bad = signed.clone();
        bad.signature[0] ^= 1;
        assert!(verify_boot_policy(&bad, &approval, &authority).is_err());
        let mut bad_approval = approval.clone();
        bad_approval.binding.bundle_hash[0] ^= 1;
        assert!(verify_boot_policy(&signed, &bad_approval, &authority).is_err());
        // A valid authority signature cannot approve the wrong compiled hash.
        let wrong_hash = SignedApproval::sign(bad_approval.binding, &ed, &pq).unwrap();
        assert!(verify_boot_policy(&signed, &wrong_hash, &authority).is_err());
        let classical = crate::auth::identity_store::BootstrapPubkey::classical(authority.ed25519);
        assert!(verify_boot_policy(&signed, &approval, &classical).is_err());
        // Changing either half of the pinned hybrid authority denies.
        let mixed = crate::auth::identity_store::BootstrapPubkey::hybrid(
            authority.ed25519,
            other_authority.ml_dsa_65.unwrap(),
        );
        assert!(verify_boot_policy(&signed, &approval, &mixed).is_err());
    }

    #[test]
    fn configured_boot_refuses_unanchored_roster_and_unsupported_trust_source() {
        let (_, _, authority, _, _) = boot_artifacts();
        let dir = tempfile::tempdir().unwrap();
        let config = crate::config::MacBootPolicyConfig {
            policy_path: dir.path().join("policy.json"),
            approval_path: dir.path().join("approval.json"),
            expected_generation: 7,
            expected_policy_hash: hex::encode([0u8; 32]),
        };
        let err = install_configured_boot_policy(&config, dir.path(), false).unwrap_err();
        assert!(err.to_string().contains("OS-owned"));
        crate::auth::identity_store::write_bootstrap_pubkeys_hybrid(
            dir.path(),
            &std::collections::HashMap::from([("policy".to_owned(), authority)]),
        )
        .unwrap();
        let err = install_configured_boot_policy(&config, dir.path(), true).unwrap_err();
        assert!(err.to_string().contains("enrollment attestation"));
    }

    #[test]
    fn configured_boot_requires_both_paths_when_present() {
        assert!(serde_json::from_str::<crate::config::PolicyServiceConfig>(
            r#"{"boot_policy":{"policy_path":"policy.json"}}"#,
        )
        .is_err());
        assert!(
            serde_json::from_str::<crate::config::PolicyServiceConfig>("{}")
                .unwrap()
                .boot_policy
                .is_none()
        );
        assert!(serde_json::from_str::<crate::config::PolicyServiceConfig>(
            r#"{"boot_policy":{"policy_path":"policy.json","approval_path":"approval.json","expected_generation":7,"expected_policy_hash":"0000000000000000000000000000000000000000000000000000000000000000"}}"#,
        )
        .is_ok());
    }

    #[test]
    fn reviewed_boot_authoring_round_trips_and_never_replaces_outputs() {
        let (signed, approval, authority, ed, _) = boot_artifacts();
        let policy = verify_boot_policy(&signed, &approval, &authority).unwrap();
        let derived_authority =
            crate::auth::identity_store::BootstrapPubkey::for_service_key(&ed).unwrap();
        let hash = policy.policy_hash().unwrap();
        let (signed, approval) = sign_reviewed_policy(
            &policy,
            &grant_with(vec![]),
            7,
            &hash,
            &ed,
            &derived_authority,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("reviewed");
        publish_boot_artifacts(&output, &signed, &approval).unwrap();
        let config = crate::config::MacBootPolicyConfig {
            policy_path: output.join("signed-policy.json"),
            approval_path: output.join("signed-approval.json"),
            expected_generation: policy.generation,
            expected_policy_hash: hex::encode(hash),
        };
        let loaded = load_boot_policy_files(&config, &derived_authority).unwrap();
        assert_eq!(loaded.policy_hash().unwrap(), hash);
        assert_eq!(loaded.enrollment, policy.enrollment);
        assert!(publish_boot_artifacts(&output, &signed, &approval).is_err());
        assert_eq!(
            load_boot_policy_files(&config, &derived_authority)
                .unwrap()
                .policy_hash()
                .unwrap(),
            hash
        );
    }

    #[test]
    fn reviewed_boot_authoring_rejects_mismatch_wrong_signer_and_capabilities() {
        let (signed, approval, authority, ed, _) = boot_artifacts();
        let mut policy = verify_boot_policy(&signed, &approval, &authority).unwrap();
        let pinned = crate::auth::identity_store::BootstrapPubkey::for_service_key(&ed).unwrap();
        let hash = policy.policy_hash().unwrap();
        let grant = grant_with(vec![]);
        assert!(sign_reviewed_policy(&policy, &grant, 8, &hash, &ed, &pinned).is_err());
        assert!(sign_reviewed_policy(&policy, &grant, 7, &[0; 32], &ed, &pinned).is_err());
        let wrong_ed = SigningKey::generate(&mut OsRng);
        assert!(sign_reviewed_policy(&policy, &grant, 7, &hash, &wrong_ed, &pinned).is_err());
        assert!(sign_reviewed_policy(
            &policy,
            &grant_with(vec![cap("mac://model/llama", "query")]),
            7,
            &hash,
            &ed,
            &pinned
        )
        .is_err());
        policy.enrollment.clear();
        assert!(sign_reviewed_policy(&policy, &grant, 7, &hash, &ed, &pinned).is_err());
        let nonempty = compile(
            &grant_with(vec![cap("mac://model/llama", "query")]),
            &Lattice::new(LatticeVersion(7), []),
            &ScopePermissionMap::new(SubjectType(1), [("model", "llama")]),
        );
        assert!(sign_reviewed_policy(
            &nonempty,
            &grant,
            7,
            &nonempty.policy_hash().unwrap(),
            &ed,
            &pinned
        )
        .is_err());
    }

    #[test]
    fn reviewed_boot_command_checks_hash_before_loading_any_credentials() {
        let (signed, approval, authority, _, _) = boot_artifacts();
        let policy = verify_boot_policy(&signed, &approval, &authority).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let args = crate::cli::commands::trust::SignMacBootPolicyArgs {
            policy_input: dir.path().join("input.json"),
            reviewed_ucan: dir.path().join("ucan.json"),
            expected_generation: 7,
            expected_hash: "00".repeat(32),
            secrets_dir: dir.path().join("no-credentials"),
            output_dir: dir.path().join("output"),
        };
        std::fs::write(&args.policy_input, serde_json::to_vec(&policy).unwrap()).unwrap();
        std::fs::write(
            &args.reviewed_ucan,
            serde_json::to_vec(&grant_with(vec![])).unwrap(),
        )
        .unwrap();
        assert!(sign_reviewed_boot_policy(&args)
            .unwrap_err()
            .to_string()
            .contains("hash mismatch"));
        assert!(!args.secrets_dir.exists());
        assert!(!args.output_dir.exists());
    }

    /// The whole missing boot chain, end to end: compile a real grant, sign it
    /// hybrid, verify-once-at-load, install it — then the enrollment resolver
    /// resolves a real clearance for an enrolled DID (instead of the deny-all
    /// `DenyUnlabeledResolver` that a `None` compiled policy forces), and the
    /// process-global `compiled_policy()` is `Some`.
    #[test]
    fn boot_installs_policy_and_resolver_resolves_real_clearance() {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();

        // A registry with one model, a grant to read it, and an enrolled subject.
        let lattice = Lattice::new(LatticeVersion(1), []);
        let permissions = ScopePermissionMap::new(SubjectType(1), [("model", "llama")]);
        let grant = grant_with(vec![cap("mac://model/llama", "query")]);

        let did = "did:key:zBootActor";
        let clearance =
            SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY);
        let enrollment = BTreeMap::from([(did.to_owned(), clearance)]);

        // The generic path receives an independently signed+verified S5 binding;
        // it never derives approval from the policy it is about to load.
        let expected = compile(&grant, &lattice, &permissions).with_enrollment(enrollment.clone());
        let binding =
            ApprovalBinding::new(&grant, expected.generation, expected.policy_hash().unwrap())
                .unwrap();
        let signed_approval = SignedApproval::sign(binding, &ed_sk, &pq_sk).unwrap();
        signed_approval
            .verify_binds(
                &ed_sk.verifying_key(),
                &pq_vk,
                &grant,
                expected.generation,
                &expected.policy_hash().unwrap(),
            )
            .unwrap();
        let verified = signed_approval
            .verify(&ed_sk.verifying_key(), &pq_vk)
            .unwrap();
        let approval = PolicyApproval::from_verified(verified);

        let (policy, _installed) = compile_sign_load_install(
            &grant,
            &lattice,
            &permissions,
            enrollment,
            approval,
            &ed_sk,
            &pq_sk,
        )
        .expect("hybrid sign + verify-once-at-load must succeed");

        // The grant lowered to exactly one allow rule (compile ran, and the
        // signature verified — a bad signature would have failed `load`).
        assert_eq!(policy.matrix.allow_len(), 1, "one model × one verb");

        // The process-global OnceLock is populated (monotonic across the test
        // binary; asserting `Some` never races).
        assert!(
            compiled_policy().is_some(),
            "boot must install the process-global compiled policy"
        );

        // The #698 resolver over THIS verified policy resolves the enrolled DID
        // to a real clearance — the dormant activation the boot step unblocks.
        let resolver = EnrollmentSubjectContextResolver::new(policy);
        let ctx = resolver
            .resolve(did)
            .expect("an enrolled DID must resolve to a real clearance, not deny-all");
        assert_eq!(ctx.clearance().level, Level::Secret);
        // Decision D: assurance floored at Classical regardless of the table.
        assert_eq!(ctx.clearance().assurance, Assurance::Classical);

        // An unenrolled DID still denies (fail-closed contract unchanged).
        assert!(resolver.resolve("did:key:zStranger").is_none());
    }

    #[test]
    fn generic_bootload_rejects_mismatched_approval() {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let lattice = Lattice::new(LatticeVersion(7), []);
        let permissions = ScopePermissionMap::new(SubjectType(1), [("model", "llama")]);
        let grant = grant_with(vec![cap("mac://model/llama", "query")]);
        let mismatched = PolicyApproval {
            generation: 7,
            approved_hash: [0u8; 32],
        };

        let result = compile_sign_load_install(
            &grant,
            &lattice,
            &permissions,
            BTreeMap::new(),
            mismatched,
            &ed_sk,
            &pq_sk,
        );

        assert!(matches!(
            result,
            Err(BootPolicyError::Policy(
                PolicyDistError::NoMatchingApproval { generation: 7 }
            ))
        ));
    }

    /// The production baseline: a self-issued empty grant compiles to an empty
    /// matrix and installs cleanly (dormant — grants nothing, enrolls nobody),
    /// still exercising the full hybrid sign + verify-once-at-load path.
    #[test]
    fn baseline_boot_policy_installs_empty_dormant_matrix() {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();

        let (policy, _installed) = install_baseline_boot_policy(&ed_sk, &pq_sk)
            .expect("empty baseline must sign + verify-load");

        assert_eq!(policy.generation, 1);
        assert_eq!(policy.matrix.allow_len(), 0, "empty grant → empty matrix");
        assert!(
            policy.clearance_for("did:key:zAnyone").is_none(),
            "the baseline enrolls nobody (fail-closed default)"
        );
        assert!(compiled_policy().is_some());
    }

    /// Seam-level "un-inert but dormant" assertion: after the baseline boot the
    /// process-global seam is populated, so [`crate::mac::exchange_enrollment_resolver`]
    /// returns the real [`EnrollmentSubjectContextResolver`] (no longer the
    /// deny-all `DenyUnlabeledResolver`) — yet, because the baseline enrolls
    /// nobody, that resolver still denies **every** DID. This is the truest
    /// statement of what this PR does: it flips the seam on without granting
    /// anything.
    ///
    /// The asserted DID is one no other test in this binary enrolls, so the
    /// `is_none()` holds regardless of which test's install won the write-once
    /// global slot.
    #[test]
    fn baseline_boot_flips_seam_on_but_still_denies_every_did() {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();

        let _ = install_baseline_boot_policy(&ed_sk, &pq_sk)
            .expect("empty baseline must sign + verify-load");

        // The seam flipped on (a compiled policy is installed → the grant path
        // resolves through EnrollmentSubjectContextResolver, not deny-all).
        assert!(
            compiled_policy().is_some(),
            "the seam must be populated after boot"
        );

        // …and it still denies every DID (dormant): the enrollment resolver
        // returns None for an unenrolled subject, exactly like the deny-all
        // fallback would — un-inert, but grants nothing.
        let resolver = crate::mac::exchange_enrollment_resolver();
        assert!(
            resolver
                .resolve("did:key:zNobodyEverEnrollsThisSeamTest")
                .is_none(),
            "the flipped-on seam must still deny an unenrolled DID"
        );
    }
}
