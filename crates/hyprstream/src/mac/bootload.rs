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
//! ## DORMANT — this does NOT enable enforcement
//!
//! Installing a compiled policy only makes the PDP *inputs* real: it flips
//! [`crate::mac::compiled_policy`] to `Some`, so the enrollment resolver
//! resolves genuine clearances instead of denying every DID. The per-op
//! deciders remain AllowAll — no PEP consults this PDP yet (see the "Current
//! status" note in `CLAUDE.md` / the epic). Turning enforcement *on* is a
//! separate, deliberate step; this keeps the framework fail-closed and correct
//! while wiring the last dormant input.
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
use crate::mac::compiled::{sign_policy, CompiledPolicy, PolicyDistError, PolicyLoader};
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
/// The signer and verifier use the **same** node keys: at boot the node is its
/// own baseline-policy authority, so it self-signs and self-verifies the
/// artifact through the full [`sign_policy`] → [`PolicyLoader::load`] path
/// (rather than short-circuiting the crypto). `require_pq = true` — the ML-DSA
/// outer layer is mandatory; a classical-only signature is rejected.
///
/// The caller MUST have already validated `grant`'s UCAN chain — `compile`
/// reads only its capability set. Enrollment is the authority-owned
/// DID→clearance table (#698), embedded in and covered by the signature.
///
/// Installation is once-per-process (the underlying `OnceLock`): a second call
/// leaves the first-installed policy in place, but always returns the policy
/// **this** call compiled+verified (so callers observe their own artifact even
/// when another already won the global slot).
pub fn compile_sign_load_install(
    grant: &Ucan,
    lattice: &Lattice,
    permissions: &impl PermissionMap,
    enrollment: BTreeMap<String, SecurityLabel>,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<Arc<CompiledPolicy>, BootPolicyError> {
    // Control plane: lower the grant → TE matrix (S5 compiler) and attach the
    // authority-owned enrollment. Both travel inside the signature.
    let policy = compile(grant, lattice, permissions).with_enrollment(enrollment);

    // Sign hybrid (EdDSA + ML-DSA-65 COSE composite; cose_sign::sign_composite
    // under the HybridPolicySigner adapter).
    let signer = HybridPolicySigner {
        ed_sk,
        pq_sk: Some(pq_sk),
    };
    let signed = sign_policy(&policy, &signer)?;

    // Verify ONCE at load (require_pq = true, fail-closed). Same key material —
    // the node self-signs its baseline; a production distribution flow would
    // anchor the authority's key instead, but the load/verify contract is
    // identical.
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
    let loaded = PolicyLoader::new(verifier).load(&signed)?;

    let policy = Arc::new(loaded);
    install_compiled_policy(policy.clone());
    Ok(policy)
}

/// Install the node's **dormant baseline** compiled policy at daemon boot.
///
/// The baseline is intentionally minimal: a self-issued empty grant over an
/// empty lattice generation and an empty object registry, so the compiled TE
/// matrix is empty (grants nothing) and no subject is enrolled. Its only effect
/// is to make [`crate::mac::compiled_policy`] return `Some`, which switches the
/// grant path's subject resolver from deny-all to the real
/// [`EnrollmentSubjectContextResolver`](crate::mac::EnrollmentSubjectContextResolver).
/// Enforcement stays AllowAll — this is the dormant activation prerequisite, not
/// the enforcement flip.
///
/// A future policy-authoring step (config-driven UCAN grant + enrollment) feeds
/// a real matrix through the same [`compile_sign_load_install`] path; the empty
/// baseline is the fail-closed default until then.
pub fn install_baseline_boot_policy(
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<Arc<CompiledPolicy>, BootPolicyError> {
    // Self-issued empty grant: the node is its own baseline authority and grants
    // itself nothing. `compile` reads only the (empty) capability set.
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

    // Empty lattice (generation 1) and empty object registry → empty matrix.
    let lattice = Lattice::new(LatticeVersion(1), []);
    let permissions = ScopePermissionMap::new(SubjectType(1), Vec::<(String, String)>::new());

    compile_sign_load_install(
        &grant,
        &lattice,
        &permissions,
        BTreeMap::new(),
        ed_sk,
        pq_sk,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::mac::compiled_policy;
    use crate::mac::lattice::{Assurance, CompartmentSet, Level};
    use crate::mac::EnrollmentSubjectContextResolver;
    use crate::services::oauth::token_exchange::SubjectContextResolver as _;
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
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();

        // A registry with one model, a grant to read it, and an enrolled subject.
        let lattice = Lattice::new(LatticeVersion(1), []);
        let permissions = ScopePermissionMap::new(SubjectType(1), [("model", "llama")]);
        let grant = grant_with(vec![cap("mac://model/llama", "query")]);

        let did = "did:key:zBootActor";
        let clearance =
            SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY);
        let enrollment = BTreeMap::from([(did.to_owned(), clearance)]);

        let policy = compile_sign_load_install(
            &grant,
            &lattice,
            &permissions,
            enrollment,
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

    /// The production baseline: a self-issued empty grant compiles to an empty
    /// matrix and installs cleanly (dormant — grants nothing, enrolls nobody),
    /// still exercising the full hybrid sign + verify-once-at-load path.
    #[test]
    fn baseline_boot_policy_installs_empty_dormant_matrix() {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();

        let policy = install_baseline_boot_policy(&ed_sk, &pq_sk)
            .expect("empty baseline must sign + verify-load");

        assert_eq!(policy.generation, 1);
        assert_eq!(policy.matrix.allow_len(), 0, "empty grant → empty matrix");
        assert!(
            policy.clearance_for("did:key:zAnyone").is_none(),
            "the baseline enrolls nobody (fail-closed default)"
        );
        assert!(compiled_policy().is_some());
    }
}
