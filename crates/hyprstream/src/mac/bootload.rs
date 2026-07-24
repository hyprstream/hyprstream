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
//! verifies with `require_pq = true` (design §14 — never Ed25519-only). A node
//! with no active ML-DSA signing key installs **nothing** (the caller keeps the
//! resolver denying), never a classical-only baseline.

use std::collections::BTreeMap;
use std::sync::Arc;

use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::ucan::token::{Did, Ucan, UcanPayload};
use hyprstream_rpc::crypto::pq::MlDsaSigningKey;

use crate::mac::compiled::cose::{HybridPolicySigner, HybridPolicyVerifier};
use crate::mac::compiled::{
    sign_policy, CompiledPolicy, PolicyApproval, PolicyDistError, PolicyLoader,
};
use crate::mac::compiler::{compile, PermissionMap};
use crate::mac::lattice::{Lattice, LatticeVersion, SecurityLabel};
use crate::mac::permission_map::ScopePermissionMap;
use crate::mac::te::SubjectType;
use crate::mac::install_compiled_policy;

/// Errors from the boot path. All fail-closed: on any error nothing is installed
/// and [`crate::mac::compiled_policy`] stays `None` (the resolver keeps denying).
#[derive(Debug, thiserror::Error)]
pub enum BootPolicyError {
    /// Signing, hashing, or the verify-once-at-load step failed.
    #[error("policy bootload failed: {0}")]
    Policy(#[from] PolicyDistError),
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
/// not already installed the boot baseline — which, at daemon startup, it always
/// has. A config-driven real policy therefore CANNOT replace the baseline through
/// this function today: it will compile+verify fine, get `installed == false`,
/// and silently leave the empty baseline as the live seam. Swapping a
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
