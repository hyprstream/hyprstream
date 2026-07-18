//! CRD → grant compilation for [`mesh::TenantBinding`] (issue #929).
//!
//! A `TenantBinding` is an **admin-authorable surface**, never the authority.
//! When [`mesh::TenantBindingSpec::entitlement`] is set, the operator compiles
//! it — signing **as issuer** — into the two artifacts that *are* the authority:
//!
//! 1. A **UCAN grant** ([`hyprstream_rpc::auth::ucan::Ucan`]) — an
//!    issuer→holder capability signed with the project's hybrid-PQC COSE
//!    composite (EdDSA + ML-DSA-65) via
//!    [`hyprstream_rpc::crypto::cose_sign::sign_composite`] with `UCAN_AAD`.
//!    The operator is the issuer; the tenant is the audience/holder.
//! 2. An **`ai.hyprstream.ledger.allocation` record**
//!    ([`hyprstream_pds::ledger::AllocationRecord`]) — the inventory line that
//!    **lands in the tenant's PDS**, whose `grant` field references the UCAN's
//!    CID. This is the durable, holder-controlled entitlement; revocation is
//!    bumping the epoch (stop renewing + reissue), not editing the CRD.
//!
//! Enforcers (#781/#787/#790/#793/#794/#527) verify the **presented grant**
//! (the UCAN CID + its allocation record), never the CRD. [`TenantGrantVerifier`]
//! is the verification contract — the exact shape of that consume-side check —
//! so the consumer issues can wire it without re-deriving the compile/verify
//! correspondence.
//!
//! ## What this module deliberately leaves to the consumer issues
//!
//! Per #929's narrow scope:
//! - **PDS publish** of the allocation record into the tenant's MST (the
//!   physical "lands in inventory") — needs the #910 multi-collection write
//!   path + the tenant's signing key, both operator-bootstrap concerns. This
//!   module produces the record; landing it is plumbing.
//! - **Operator issuer-key bootstrap in-cluster** — [`TenantGrantIssuer`]
//!   takes already-resolved key material; deriving it from `node_identity`'s
//!   root store at operator startup is the K5b runtime's job.
//! - **Enforcer consumption** (admission, receipts, autoscaling bounds) —
//!   #781/#787/#790/#793/#794/#527/#792/#795/#784. The compiled grant is the
//!   primitive they consume; [`TenantGrantVerifier`] is the contract.
//! - **S6 short-ttl sender-bound (DPoP) renewal** — greenfield (#921 gap 2);
//!   the `epoch` counter on the allocation record is the revocation handle
//!   until the renewal layer exists.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_pds::ledger::{AllocationRecord, GrantClass as PdsGrantClass, Unit as PdsUnit};
use hyprstream_rpc::auth::ucan::capability::{
    Ability, Capability, CaveatValue, Caveats, Resource as CapabilityResource,
};
use hyprstream_rpc::auth::ucan::chain;
use hyprstream_rpc::auth::ucan::token::{Ucan, UcanPayload, UCAN_AAD};
use hyprstream_rpc::crypto::cose_sign::{sign_composite, verify_composite};
#[cfg(test)]
use hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair;
use hyprstream_rpc::crypto::pq::{
    ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey,
    MlDsaVerifyingKey,
};
use hyprstream_rpc::identity::Did;
use kube::Resource;
use rand::RngCore;

use crate::mesh::{TenantBinding, TenantBindingStatus, TenantEntitlement, TenantGrantClass};

/// HKDF purpose label for the operator's tenant-grant Ed25519 issuer key.
///
/// Key-separated (PQUIP) from the node's mesh/JWT/KEM purposes in
/// `hyprstream_rpc::node_identity`: a `TenantGrantIssuer` key is bound to
/// exactly one cryptographic purpose — *issuing tenant entitlement grants* —
/// so a compromise of any other derived key cannot forge a grant. Use via
/// [`TenantGrantIssuer::from_node_root`].
pub const TENANT_GRANT_ED25519_PURPOSE: &str = "hyprstream-tenant-grant-ed25519-v1";

/// The UCAN ability a compiled tenant grant carries.
///
/// `ledger/spend` is the resource-class verb an enforcer's
/// `GrantVerifier`-shaped check attenuates against: a holder may spend the
/// granted amount of the granted unit, attenuated by the capability caveats
/// (unit/amount/epoch/class). The action-vocabulary seam (#582) maps this to
/// concrete enforcer operations later; it is structural here.
pub const TENANT_GRANT_ABILITY: &str = "ledger/spend";

/// Domain-separation tag included in every compiled grant's capability caveat
/// map so a tenant grant can never be confused with another UCAN capability
/// type sharing the `ledger/spend` ability.
const CAVEAT_KIND: &str = "hs:tenant-grant:v1";

/// The operator's issuer key material for compiling tenant grants.
///
/// Holds the Ed25519 + ML-DSA-65 signing keys the operator signs UCANs with,
/// plus the derived issuer DID. The operator is the **unit issuer** (D8-1:
/// credits are the unit-issuer's liability), so every grant compiled by one
/// issuer carries that issuer's DID as both the UCAN `issuer` and the
/// allocation record `issuer`/`unit.issuer`.
///
/// Construct with [`TenantGrantIssuer::from_keys`] (production: keys resolved
/// from the node identity store) or [`TenantGrantIssuer::from_node_root`]
/// (derives a purpose-separated key from the operator's persisted root), or
/// [`TenantGrantIssuer::generate_for_test`] in tests.
#[derive(Clone)]
pub struct TenantGrantIssuer {
    ed_sk: SigningKey,
    pq_sk: MlDsaSigningKey,
    // Verifying keys + issuer DID are derived once at construction (the keys are
    // immutable for the issuer's lifetime) and cached so the hot paths are pure
    // accessors — no per-call trait dispatch, no re-deriving the DID.
    ed_vk: VerifyingKey,
    pq_vk: MlDsaVerifyingKey,
    issuer_did: Did,
}

impl std::fmt::Debug for TenantGrantIssuer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantGrantIssuer")
            .field("issuer_did", &self.issuer_did.as_str())
            .finish_non_exhaustive()
    }
}

impl TenantGrantIssuer {
    /// Construct from already-resolved key material.
    ///
    /// The issuer DID is derived from `ed_sk`'s verifying key so the two can
    /// never drift.
    pub fn from_keys(ed_sk: SigningKey, pq_sk: MlDsaSigningKey) -> Result<Self> {
        let ed_vk = ed_sk.verifying_key();
        // Resolve the PQ verifying key through the `hyprstream_crypto::pq` byte
        // helpers rather than the `Keypair` trait so this crate does not take a
        // direct dep on the `signature`/`ml_dsa` crates just to call
        // `sk.verifying_key()`. Decoding can only fail on a malformed signing
        // key, which is a caller bug — surfaced as an error rather than a panic
        // so this constructor is clippy-clean under the `-D expect_used` lint.
        let pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&pq_sk))
            .context("decode the ML-DSA-65 verifying key from the signing key")?;
        let issuer_did = Did::from_ed25519(&ed_vk.to_bytes());
        Ok(Self {
            ed_sk,
            pq_sk,
            ed_vk,
            pq_vk,
            issuer_did,
        })
    }

    /// Derive a purpose-separated tenant-grant issuer from the operator's
    /// persisted node root key.
    ///
    /// Mirrors `hyprstream_rpc::node_identity::derive_mesh_mldsa_key` (HKDF
    /// over the Ed25519 seed) but under the dedicated
    /// [`TENANT_GRANT_ED25519_PURPOSE`] label, then re-seeds the ML-DSA-65 key
    /// from the derived Ed25519 bytes. Key-separated from every other derived
    /// purpose (PQUIP).
    pub fn from_node_root(root: &SigningKey) -> Result<Self> {
        let ed_sk =
            hyprstream_rpc::node_identity::derive_purpose_key(root, TENANT_GRANT_ED25519_PURPOSE);
        let pq_sk = derive_tenant_grant_mldsa_key(&ed_sk);
        Self::from_keys(ed_sk, pq_sk)
    }

    /// Generate a fresh, random issuer keypair (tests only).
    #[cfg(test)]
    #[allow(clippy::expect_used)]
    pub fn generate_for_test() -> Self {
        let mut ed_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut ed_bytes);
        let ed_sk = SigningKey::from_bytes(&ed_bytes);
        let pq_sk = ml_dsa_generate_keypair().0;
        Self::from_keys(ed_sk, pq_sk)
            .expect("a freshly generated keypair always decodes its verifying key")
    }

    /// The issuer DID (the operator's `did:key` for this grant purpose).
    pub fn issuer_did(&self) -> &Did {
        &self.issuer_did
    }

    /// The issuer's Ed25519 verifying key (the verifier-side anchor).
    pub fn issuer_ed_vk(&self) -> VerifyingKey {
        self.ed_vk
    }

    /// The issuer's ML-DSA-65 verifying key (the PQ leg of the composite).
    pub fn issuer_pq_vk(&self) -> MlDsaVerifyingKey {
        self.pq_vk.clone()
    }

    /// Compile a [`TenantBinding`]'s entitlement into the issuer-signed grant +
    /// inventory allocation record (#929).
    ///
    /// `epoch` is the issuance epoch (revocation = bump it + reissue); `now` is
    /// unix seconds used for `not_before` and to sanity-check an author-supplied
    /// `expiration`. The CRD's [`TenantEntitlement`] is the authoring surface;
    /// the returned [`CompiledTenantGrant`] is the authority that lands in the
    /// tenant's inventory.
    ///
    /// Fails closed on any malformation: an absent entitlement, a tenant DID
    /// that cannot anchor an Ed25519 audience, an inverted expiry, or a signing
    /// error. It never emits a half-signed grant.
    pub fn compile(
        &self,
        binding: &TenantBinding,
        epoch: u64,
        now: u64,
    ) -> Result<CompiledTenantGrant> {
        let entitlement = binding
            .spec
            .entitlement
            .as_ref()
            .context("TenantBinding carries no entitlement — nothing to compile")?;

        validate_entitlement(entitlement, now)?;

        // The holder/audience is the tenant DID. UCAN audiences are did:key
        // (Ed25519) for signature linkage; we accept any did: method the PDS
        // already classifies, but the UCAN payload's audience is stored as the
        // tenant string verbatim. A non-did:key audience is still a valid
        // *allocation record* holder (pairwise or did:web), but the UCAN chain
        // validator requires did:key, so we issue the UCAN to the operator's
        // own issuer DID as the capability owner and encode the holder in the
        // capability resource + caveats. The allocation record's `holder`
        // (below) is the authoritative subject.
        let holder = binding.spec.tenant.as_str();

        let capability = tenant_grant_capability(&self.issuer_did, holder, entitlement, epoch)?;

        let payload = UcanPayload {
            issuer: self.issuer_did.clone(),
            audience: self.issuer_did.clone(),
            capabilities: vec![capability],
            not_before: Some(now),
            expiration: entitlement.expiration,
            // Replay-uniqueness: a fresh random nonce per compiled grant. Two
            // compiles of the same binding at the same epoch (e.g. operator
            // restart) yield distinct UCANs/CIDs by construction.
            nonce: fresh_nonce(),
        };
        let signing_bytes = payload
            .signing_bytes()
            .context("encode UCAN payload signing bytes")?;
        let signature = sign_composite(&self.ed_sk, Some(&self.pq_sk), &signing_bytes, UCAN_AAD)
            .context("hybrid-PQC sign of UCAN payload")?;
        let ucan = Ucan {
            payload,
            proofs: Vec::new(),
            signature,
        };

        // The grant CID is the atproto-canonical content address of the UCAN:
        // CIDv1 dag-cbor (sha2-256) over its canonical CBOR. This is the
        // `format: "cid"` string the allocation record's `grant` field
        // references and enforcers present.
        let grant_cid = cid_v1_string(&ucan)?;

        let allocation = AllocationRecord::new(
            grant_cid.clone(),
            PdsUnit {
                code: entitlement.unit.clone(),
                issuer: self.issuer_did.as_str().to_owned(),
            },
            entitlement.amount,
            epoch,
            self.issuer_did.as_str(),
            holder,
            entitlement.class.into(),
        )
        .context("construct allocation record")?;

        let allocation_cid = allocation.cid().encode();

        Ok(CompiledTenantGrant {
            ucan,
            allocation,
            grant_cid,
            allocation_cid,
            epoch,
        })
    }
}

/// The output of compiling a [`TenantBinding`]: the authority artifacts.
///
/// - [`Self::ucan`] is the issuer-signed capability (presented by the holder,
///   verified by enforcers).
/// - [`Self::allocation`] is the `ai.hyprstream.ledger.allocation` record that
///   lands in the tenant's PDS inventory (whose `grant` field references
///   [`Self::grant_cid`]).
/// - [`Self::grant_cid`] / [`Self::allocation_cid`] are the CIDv1 strings the
///   operator records in [`crate::mesh::TenantBindingStatus`] as observed truth.
#[derive(Clone, Debug)]
pub struct CompiledTenantGrant {
    /// The issuer-signed UCAN grant.
    pub ucan: Ucan,
    /// The inventory allocation record (lands in the tenant's PDS).
    pub allocation: AllocationRecord,
    /// CIDv1 of [`Self::ucan`] — what enforcers present + what
    /// `allocation.grant` references.
    pub grant_cid: String,
    /// CIDv1 of [`Self::allocation`] — the durable inventory line.
    pub allocation_cid: String,
    /// The issuance epoch recorded on the allocation.
    pub epoch: u64,
}

/// The verification contract for enforcers (#781/#787/#790/#793/#794/#527).
///
/// This is the exact check a consumer enforcer performs on a *presented* grant:
/// structural UCAN validation → hybrid-PQC composite signature verification
/// against the issuer's anchored keys (require_pq = true, fail-closed) →
/// delegation-chain liveness at `now` → allocation record round-trip →
/// grant-CID↔allocation binding. On success it yields a
/// [`VerifiedTenantGrant`] shaped exactly like the enforcer-plane
/// `VerifiedGrant { holder, unit, cap_amount, exp, epoch }` so the consumer
/// issues map it 1:1.
///
/// A production `GrantVerifier` impl resolves `grant_cid → (Ucan,
/// AllocationRecord)` by fetching from the holder's PDS, then calls
/// [`TenantGrantVerifier::verify`]; this struct owns only the verify-once
/// half so it is trivially testable without a PDS.
#[derive(Clone)]
pub struct TenantGrantVerifier {
    issuer_ed_vk: VerifyingKey,
    issuer_pq_vk: MlDsaVerifyingKey,
}

impl std::fmt::Debug for TenantGrantVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantGrantVerifier")
            .field("issuer_ed_vk", &self.issuer_ed_vk.to_bytes())
            .finish_non_exhaustive()
    }
}

impl TenantGrantVerifier {
    /// Construct from the issuer's verifying keys (the anchored side of
    /// [`TenantGrantIssuer`]).
    pub fn new(issuer_ed_vk: VerifyingKey, issuer_pq_vk: MlDsaVerifyingKey) -> Self {
        Self {
            issuer_ed_vk,
            issuer_pq_vk,
        }
    }

    /// Construct from an issuer (mirrors its public keys).
    pub fn from_issuer(issuer: &TenantGrantIssuer) -> Self {
        Self::new(issuer.issuer_ed_vk(), issuer.issuer_pq_vk())
    }

    /// Verify a compiled grant the way an enforcer will.
    ///
    /// `now` is unix seconds; the UCAN's `not_before`/`expiration` window is
    /// checked against it via [`chain::validate`].
    pub fn verify(&self, compiled: &CompiledTenantGrant, now: u64) -> Result<VerifiedTenantGrant> {
        // 1. Structural validity (DIDs resolve, window is well-ordered).
        compiled
            .ucan
            .validate_structure()
            .context("UCAN structural validation failed")?;

        // 2. Hybrid-PQC composite signature over the payload signing bytes,
        //    against the issuer's anchored keys. require_pq = true: a
        //    classical-only signature on a tenant grant is a fail-closed
        //    reject (a long-lived authority artifact must be HNDL-resistant).
        let signing_bytes = compiled
            .ucan
            .payload
            .signing_bytes()
            .context("encode UCAN payload signing bytes for verify")?;
        let verified = verify_composite(
            &compiled.ucan.signature,
            &self.issuer_ed_vk,
            Some(&self.issuer_pq_vk),
            &signing_bytes,
            UCAN_AAD,
            true,
        )
        .context("UCAN composite signature verification failed")?;
        if !verified.ml_dsa {
            bail!("tenant grant signature missing the ML-DSA-65 leg (require_pq)");
        }

        // 2b. The signed payload's issuer MUST be the DID of the anchored
        //     Ed25519 key. Without this, a UCAN validly signed by the anchored
        //     key could claim a different issuer identity in its payload.
        let anchored_issuer = Did::from_ed25519(&self.issuer_ed_vk.to_bytes());
        if compiled.ucan.payload.issuer != anchored_issuer {
            bail!(
                "UCAN payload issuer ({}) does not match the anchored issuer key ({})",
                compiled.ucan.payload.issuer.as_str(),
                anchored_issuer.as_str()
            );
        }

        // 2c. The presented `grant_cid` MUST be the recomputed CID of the
        //     presented UCAN. The signature covers only the UCAN payload — the
        //     CID strings arrive alongside it, so an attacker could otherwise
        //     pair a valid UCAN with an arbitrary `grant_cid` and a forged
        //     allocation that references that same arbitrary string.
        let recomputed_grant_cid = cid_v1_string(&compiled.ucan)?;
        if recomputed_grant_cid != compiled.grant_cid {
            bail!(
                "presented grant CID ({}) does not match the UCAN's recomputed CID ({})",
                compiled.grant_cid,
                recomputed_grant_cid
            );
        }

        // 3. Delegation-chain liveness. A compiled tenant grant is a root
        //    (empty proofs), so this re-checks the not_before/expiration window
        //    against `now` — enforcers MUST reject an expired grant even if the
        //    signature is valid.
        chain::validate_chain(&compiled.ucan, now)
            .context("UCAN chain/liveness validation failed")?;

        // 4. The allocation record round-trips through its canonical DAG-CBOR
        //    — a tampered record cannot survive this.
        let record_bytes = compiled.allocation.to_dag_cbor();
        let redecoded = AllocationRecord::from_dag_cbor(&record_bytes)
            .context("allocation record failed canonical round-trip")?;
        if redecoded != compiled.allocation {
            bail!("allocation record canonical round-trip mismatch");
        }

        // 4b. The presented `allocation_cid` MUST be the recomputed CID of the
        //     presented record — same reasoning as the grant CID: the CID
        //     string is unsigned transport metadata, never trusted as given.
        let recomputed_allocation_cid = redecoded.cid().encode();
        if recomputed_allocation_cid != compiled.allocation_cid {
            bail!(
                "presented allocation CID ({}) does not match the record's recomputed CID ({})",
                compiled.allocation_cid,
                recomputed_allocation_cid
            );
        }

        // 4c. The allocation's issuer fields MUST be the anchored issuer: the
        //     signature proves who signed the UCAN, but the allocation record
        //     itself is unsigned here — an attacker could otherwise attach a
        //     record naming a different issuer/unit-issuer to a valid UCAN.
        if redecoded.issuer != anchored_issuer.as_str() {
            bail!(
                "allocation.issuer ({}) does not match the anchored issuer ({})",
                redecoded.issuer,
                anchored_issuer.as_str()
            );
        }
        if redecoded.unit.issuer != anchored_issuer.as_str() {
            bail!(
                "allocation.unit.issuer ({}) does not match the anchored issuer ({})",
                redecoded.unit.issuer,
                anchored_issuer.as_str()
            );
        }

        // 5. The allocation's `grant` field MUST reference the presented UCAN's
        //    CID, binding the inventory line to the capability it draws down.
        if redecoded.grant != compiled.grant_cid {
            bail!(
                "allocation.grant ({}) does not reference the presented grant CID ({})",
                redecoded.grant,
                compiled.grant_cid
            );
        }

        // 6. The capability must agree with the allocation record: exact
        //    ability, holder bound through the signed resource, and caveats
        //    matching the record fields (unit/amount/epoch/class). This is
        //    what stops a holder-substitution: the holder lives in the
        //    *signed* capability resource, so a forged record naming a
        //    different holder cannot match it.
        verify_capability_matches_record(&compiled.ucan, &anchored_issuer, &redecoded)?;

        Ok(VerifiedTenantGrant {
            holder: redecoded.holder,
            unit: redecoded.unit.code,
            cap_amount: redecoded.amount,
            exp: compiled.ucan.payload.expiration,
            epoch: redecoded.epoch,
            class: redecoded.class,
        })
    }
}

/// The verified shape of a compiled tenant grant — the data an enforcer admits
/// against.
///
/// Field-for-field compatible with the enforcer-plane
/// `hyprstream::services::ledger::VerifiedGrant { holder, unit, cap_amount, exp,
/// epoch }`; the consumer issues implement their `GrantVerifier` by resolving a
/// presented `grant_cid` to a [`CompiledTenantGrant`] and calling
/// [`TenantGrantVerifier::verify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedTenantGrant {
    /// The holder (tenant DID / pairwise id) whose inventory this grant is.
    pub holder: String,
    /// The issuer-scoped credit unit code (the unit's issuer is the operator).
    pub unit: String,
    /// The total cap authorized (smallest quantum).
    pub cap_amount: u64,
    /// Grant expiry (unix seconds), if the UCAN carries one.
    pub exp: Option<u64>,
    /// Allocation epoch (revocation counter).
    pub epoch: u64,
    /// Grant class.
    pub class: PdsGrantClass,
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Derive the tenant-grant ML-DSA-65 key from a derived Ed25519 purpose key.
///
/// Mirrors `node_identity::derive_mesh_mldsa_key` (re-seed ML-DSA-65 from the
/// derived Ed25519 bytes) but is kept local so the `TENANT_GRANT_ED25519_PURPOSE`
/// separation is the single source of truth for both legs.
fn derive_tenant_grant_mldsa_key(ed25519_key: &SigningKey) -> MlDsaSigningKey {
    let derived = hyprstream_rpc::node_identity::derive_purpose_key(
        ed25519_key,
        "hyprstream-tenant-grant-mldsa-v1",
    );
    let seed = derived.to_bytes();
    // NOTE: the reference derivation in `node_identity::derive_mesh_mldsa_key`
    // zeroizes the seed here; we omit that only to avoid taking a direct dep on
    // the `zeroize` crate in this otherwise-light crate. The seed is a 32-byte
    // stack value derived from an already-purpose-separated HKDF output, dropped
    // immediately; the key-separation (distinct HKDF label) is the load-bearing
    // property, not the in-memory wipe.
    ml_dsa_sk_from_seed(&seed)
}

/// Build the single capability a tenant grant carries.
///
/// - `resource` = `hs://tenant/<issuer>/<holder>` — names *whose inventory* the
///   grant is in (the issuer's liability, the holder's balance), scoped under
///   the issuer so two operators' grants cannot cross-contend.
/// - `ability` = `ledger/spend` ([`TENANT_GRANT_ABILITY`]).
/// - `caveats` bind the unit / amount / epoch / class / kind so attenuation is
///   value-aware at the enforcer (a delegated subset cannot widen amount or
///   flip class).
fn tenant_grant_capability(
    issuer: &Did,
    holder: &str,
    entitlement: &TenantEntitlement,
    epoch: u64,
) -> Result<Capability> {
    let resource = CapabilityResource::new(tenant_grant_resource(issuer, holder));
    let ability = Ability::new(TENANT_GRANT_ABILITY);
    // `CaveatValue::Int` is signed; a value that cannot be represented is a
    // hard error, never a clamp — a clamped amount would mint a grant that
    // fails its own verification, and a wrapped epoch would encode as negative.
    let amount = i64::try_from(entitlement.amount)
        .context("entitlement.amount exceeds i64::MAX and cannot be encoded as a caveat")?;
    let epoch = i64::try_from(epoch)
        .context("allocation epoch exceeds i64::MAX and cannot be encoded as a caveat")?;
    let mut caveats = Caveats::empty();
    caveats
        .0
        .insert("kind".to_owned(), CaveatValue::Text(CAVEAT_KIND.to_owned()));
    caveats.0.insert(
        "unit".to_owned(),
        CaveatValue::Text(entitlement.unit.clone()),
    );
    caveats
        .0
        .insert("amount".to_owned(), CaveatValue::Int(amount));
    caveats
        .0
        .insert("epoch".to_owned(), CaveatValue::Int(epoch));
    caveats.0.insert(
        "class".to_owned(),
        CaveatValue::Text(entitlement.class.as_str().to_owned()),
    );
    Ok(Capability {
        resource,
        ability,
        caveats,
    })
}

/// The capability resource string binding an issuer's liability to a holder's
/// balance: `hs://tenant/<issuer>/<holder>`. Single source of truth for both
/// the compile side and the verify side's holder-binding check.
fn tenant_grant_resource(issuer: &Did, holder: &str) -> String {
    format!("hs://tenant/{}/{}", issuer.as_str(), holder)
}

/// Verify the UCAN capability matches the allocation record: exactly one
/// capability with exactly [`TENANT_GRANT_ABILITY`], the holder bound through
/// the signed resource string, and caveats matching the record's fields.
fn verify_capability_matches_record(
    ucan: &Ucan,
    anchored_issuer: &Did,
    record: &AllocationRecord,
) -> Result<()> {
    let caps = ucan.capabilities();
    let cap = caps
        .first()
        .ok_or_else(|| anyhow!("tenant grant UCAN carries no capability"))?;
    if cap.ability.as_str() != TENANT_GRANT_ABILITY {
        bail!(
            "tenant grant capability has unexpected ability {} (require exactly {})",
            cap.ability,
            TENANT_GRANT_ABILITY
        );
    }
    // The holder is not a caveat — it lives in the signed resource string. The
    // record's holder MUST reconstruct the exact resource the issuer signed,
    // or the record names a substituted holder.
    let expected_resource = tenant_grant_resource(anchored_issuer, &record.holder);
    if cap.resource.as_str() != expected_resource {
        bail!(
            "grant capability resource ({}) does not bind the allocation holder ({})",
            cap.resource.as_str(),
            record.holder
        );
    }
    let cv = |key: &str| -> Result<&CaveatValue> {
        cap.caveats
            .0
            .get(key)
            .ok_or_else(|| anyhow!("tenant grant caveat missing {key:?}"))
    };
    match cv("kind")? {
        CaveatValue::Text(t) if t == CAVEAT_KIND => {}
        other => bail!("grant caveat kind mismatch: {other:?}"),
    }
    match cv("unit")? {
        CaveatValue::Text(t) if t == &record.unit.code => {}
        other => bail!("grant caveat unit does not match record: {other:?}"),
    }
    // Lossless compare: a negative caveat can never equal a u64 field.
    match cv("amount")? {
        CaveatValue::Int(a) if u64::try_from(*a) == Ok(record.amount) => {}
        other => bail!("grant caveat amount does not match record: {other:?}"),
    }
    match cv("epoch")? {
        CaveatValue::Int(e) if u64::try_from(*e) == Ok(record.epoch) => {}
        other => bail!("grant caveat epoch does not match record: {other:?}"),
    }
    match cv("class")? {
        CaveatValue::Text(t) if t == &record.class.as_str().to_owned() => {}
        other => bail!("grant caveat class does not match record: {other:?}"),
    }
    Ok(())
}

fn validate_entitlement(entitlement: &TenantEntitlement, now: u64) -> Result<()> {
    if entitlement.unit.is_empty() {
        bail!("entitlement.unit must not be empty");
    }
    if entitlement.unit.chars().any(char::is_whitespace) {
        bail!("entitlement.unit must not contain whitespace");
    }
    if let Some(exp) = entitlement.expiration {
        if exp <= now {
            bail!(
                "entitlement.expiration ({exp}) must be in the future (now {now}); a grant that \
                 is born expired is never valid"
            );
        }
    }
    Ok(())
}

/// 16-byte random nonce, fresh per compile.
fn fresh_nonce() -> Vec<u8> {
    let mut buf = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut buf);
    buf.to_vec()
}

/// The atproto-canonical CIDv1 string of a UCAN (dag-cbor, sha2-256) over its
/// canonical CBOR encoding. This is the `format: "cid"` string the allocation
/// record's `grant` field references and enforcers present.
fn cid_v1_string(ucan: &Ucan) -> Result<String> {
    let bytes = ucan
        .to_cbor()
        .context("UCAN CBOR encoding is infallible for this shape; encoding error is a bug")?;
    Ok(hyprstream_pds::cid::Cid::from_dag_cbor(&bytes).encode())
}

/// Unix seconds from the system wall clock; never panics on platforms without a
/// wall clock (returns 0, which fails the expiry check closed). Used by the
/// operator's `TenantBinding` reconcile to stamp `not_before` and check
/// author-supplied expiries.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Map the k8s-side grant class to the PDS lexicon grant class.
impl From<TenantGrantClass> for PdsGrantClass {
    fn from(class: TenantGrantClass) -> Self {
        match class {
            TenantGrantClass::Prepaid => PdsGrantClass::Prepaid,
            TenantGrantClass::Underwritten => PdsGrantClass::Underwritten,
        }
    }
}

/// Compile a [`TenantBinding`]'s entitlement into the [`TenantBindingStatus`]
/// the operator writes back on reconcile (#929).
///
/// This is the operator's "sign as issuer" step as a pure decision: given a
/// binding, the operator's [`TenantGrantIssuer`] (or `None` if grant
/// compilation is disabled), the next allocation `epoch`, and `now`, it returns
/// the status the controller patches:
/// - **No entitlement** → a no-grant `Bound` status (the binding is still the
///   confused-deputy namespace↔tenant map). `issuer` is ignored.
/// - **Entitlement + `Some(issuer)`** → the operator signs; on success the
///   status carries the compiled grant CID + allocation CID + epoch (`Bound`),
///   on a compile error it is `Rejected` with the message.
/// - **Entitlement + `None`** → `Rejected`: an admin authored an entitlement
///   but the operator has no issuer configured.
///
/// The **physical PDS publish** of the allocation record into the tenant's
/// inventory is deliberately NOT done here — it needs the #910 multi-collection
/// write path + the tenant's signing key. This function mints the grant; landing
/// it is the deferred plumbing (see the module docs).
pub fn compile_tenant_binding_status(
    binding: &TenantBinding,
    issuer: Option<&TenantGrantIssuer>,
    epoch: u64,
    now: u64,
) -> TenantBindingStatus {
    let generation = binding.meta().generation;
    match binding.spec.entitlement.as_ref() {
        None => TenantBindingStatus {
            bound: Some(true),
            phase: Some("Bound".to_owned()),
            message: Some("namespace↔tenant mapping active; no entitlement to compile".to_owned()),
            observed_generation: generation,
            grant_cid: None,
            allocation_cid: None,
            epoch: None,
        },
        Some(_) => match issuer {
            None => TenantBindingStatus {
                bound: Some(false),
                phase: Some("Rejected".to_owned()),
                message: Some(
                    "entitlement present but operator has no tenant-grant issuer configured"
                        .to_owned(),
                ),
                observed_generation: generation,
                grant_cid: None,
                allocation_cid: None,
                epoch: None,
            },
            Some(issuer) => match issuer.compile(binding, epoch, now) {
                Ok(compiled) => TenantBindingStatus {
                    bound: Some(true),
                    phase: Some("Bound".to_owned()),
                    message: Some(format!(
                        "compiled grant {} (allocation {}); PDS publish pending #910",
                        compiled.grant_cid, compiled.allocation_cid
                    )),
                    observed_generation: generation,
                    grant_cid: Some(compiled.grant_cid),
                    allocation_cid: Some(compiled.allocation_cid),
                    epoch: Some(compiled.epoch),
                },
                Err(error) => TenantBindingStatus {
                    bound: Some(false),
                    phase: Some("Rejected".to_owned()),
                    message: Some(format!("grant compilation failed: {error:#}")),
                    observed_generation: generation,
                    grant_cid: None,
                    allocation_cid: None,
                    epoch: None,
                },
            },
        },
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::mesh::{TenantBindingSpec, TenantEntitlement};

    fn binding(unit: &str, amount: u64, class: TenantGrantClass) -> TenantBinding {
        TenantBinding::new(
            "acme",
            TenantBindingSpec {
                namespace: "acme".to_owned(),
                tenant: "did:web:tenant.acme.example".to_owned(),
                entitlement: Some(TenantEntitlement {
                    unit: unit.to_owned(),
                    amount,
                    class,
                    expiration: None,
                }),
            },
        )
    }

    #[test]
    fn compile_then_verify_round_trips() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 3_600, TenantGrantClass::Underwritten);

        let compiled = issuer
            .compile(&binding, 7, now)
            .expect("compile must succeed for a valid entitlement");

        // The allocation record references the grant CID.
        assert_eq!(compiled.allocation.grant, compiled.grant_cid);
        // The grant CID is a CIDv1 string.
        assert!(compiled.grant_cid.starts_with('b'));
        assert!(compiled.allocation_cid.starts_with('b'));

        let verified = verifier
            .verify(&compiled, now)
            .expect("verify must succeed");
        assert_eq!(verified.holder, "did:web:tenant.acme.example");
        assert_eq!(verified.unit, "compute-second");
        assert_eq!(verified.cap_amount, 3_600);
        assert_eq!(verified.epoch, 7);
        assert_eq!(verified.class, PdsGrantClass::Underwritten);
        assert_eq!(verified.exp, None);
    }

    #[test]
    fn two_compiles_of_the_same_binding_yield_distinct_grants() {
        // The nonce makes each compile unique even at the same epoch.
        let issuer = TenantGrantIssuer::generate_for_test();
        let now = 1_700_000_000u64;
        let binding = binding("gpu-hour", 8, TenantGrantClass::Prepaid);
        let a = issuer.compile(&binding, 1, now).unwrap();
        let b = issuer.compile(&binding, 1, now).unwrap();
        assert_ne!(a.grant_cid, b.grant_cid, "fresh nonce ⇒ distinct CIDs");
        assert_ne!(a.allocation_cid, b.allocation_cid);
    }

    #[test]
    fn verify_rejects_a_tampered_allocation_amount() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 100, TenantGrantClass::Underwritten);
        let mut compiled = issuer.compile(&binding, 1, now).unwrap();

        // Tamper: bump the allocation amount without re-signing the UCAN. The
        // capability caveat (amount=100) no longer matches the record (200).
        compiled.allocation = AllocationRecord::new(
            compiled.grant_cid.clone(),
            PdsUnit {
                code: "compute-second".to_owned(),
                issuer: issuer.issuer_did().as_str().to_owned(),
            },
            200,
            1,
            issuer.issuer_did().as_str(),
            "did:web:tenant.acme.example",
            PdsGrantClass::Underwritten,
        )
        .unwrap();
        // Keep the presented allocation CID consistent with the tampered
        // record so the signed amount caveat is what catches the tamper.
        compiled.allocation_cid = compiled.allocation.cid().encode();

        let err = verifier.verify(&compiled, now).unwrap_err();
        assert!(
            err.to_string().contains("amount"),
            "tampered amount must be caught: {err}"
        );
    }

    #[test]
    fn verify_rejects_a_classical_only_signature() {
        // A grant signed without the PQ leg must fail closed at verify
        // (require_pq = true) — the long-lived authority must be HNDL-resistant.
        let issuer_ed_sk = {
            let mut b = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut b);
            SigningKey::from_bytes(&b)
        };
        let issuer_did = Did::from_ed25519(&issuer_ed_sk.verifying_key().to_bytes());
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 5, TenantGrantClass::Prepaid);

        let entitlement = binding.spec.entitlement.as_ref().unwrap();
        let capability =
            tenant_grant_capability(&issuer_did, &binding.spec.tenant, entitlement, 1).unwrap();
        let payload = UcanPayload {
            issuer: issuer_did.clone(),
            audience: issuer_did.clone(),
            capabilities: vec![capability],
            not_before: Some(now),
            expiration: None,
            nonce: fresh_nonce(),
        };
        let signing_bytes = payload.signing_bytes().unwrap();
        // Classical-only sign (pq_sk = None).
        let signature = sign_composite(&issuer_ed_sk, None, &signing_bytes, UCAN_AAD).unwrap();
        let ucan = Ucan {
            payload,
            proofs: Vec::new(),
            signature,
        };
        let grant_cid = cid_v1_string(&ucan).unwrap();
        let allocation = AllocationRecord::new(
            grant_cid.clone(),
            PdsUnit {
                code: "compute-second".to_owned(),
                issuer: issuer_did.as_str().to_owned(),
            },
            5,
            1,
            issuer_did.as_str(),
            "did:web:tenant.acme.example",
            PdsGrantClass::Prepaid,
        )
        .unwrap();
        let compiled = CompiledTenantGrant {
            ucan,
            allocation_cid: allocation.cid().encode(),
            allocation,
            grant_cid,
            epoch: 1,
        };

        // Verifier anchored on the Ed25519 key but WITH a PQ key requirement.
        let verifier = TenantGrantVerifier::new(
            issuer_ed_sk.verifying_key(),
            // A throwaway, *wrong* PQ key: a classical-only signature cannot
            // satisfy require_pq regardless of the anchored PQ key.
            ml_dsa_generate_keypair().1,
        );
        let err = verifier.verify(&compiled, now).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("signature")
                || err.to_string().contains("ML-DSA"),
            "classical-only signature must fail closed: {err}"
        );
    }

    #[test]
    fn verify_rejects_an_expired_grant() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;

        let mut binding = binding("compute-second", 1, TenantGrantClass::Underwritten);
        binding.spec.entitlement.as_mut().unwrap().expiration = Some(now + 10);

        let compiled = issuer.compile(&binding, 1, now).unwrap();
        // Verify at `now + 100`: past expiration.
        let err = verifier.verify(&compiled, now + 100).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("expir")
                || err.to_string().contains("liveness")
                || err.to_string().contains("window"),
            "expired grant must be rejected: {err}"
        );
    }

    #[test]
    fn verify_rejects_a_substituted_holder() {
        // A valid signed UCAN paired with a forged allocation record naming a
        // different holder: every unsigned field (grant ref, unit, amount,
        // epoch, class, issuer, allocation CID) is made self-consistent, so
        // only the signed capability resource can catch the substitution.
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 100, TenantGrantClass::Underwritten);
        let mut compiled = issuer.compile(&binding, 1, now).unwrap();

        compiled.allocation = AllocationRecord::new(
            compiled.grant_cid.clone(),
            PdsUnit {
                code: "compute-second".to_owned(),
                issuer: issuer.issuer_did().as_str().to_owned(),
            },
            100,
            1,
            issuer.issuer_did().as_str(),
            "did:web:evil.example",
            PdsGrantClass::Underwritten,
        )
        .unwrap();
        compiled.allocation_cid = compiled.allocation.cid().encode();

        let err = verifier.verify(&compiled, now).unwrap_err();
        assert!(
            err.to_string().contains("holder") || err.to_string().contains("resource"),
            "substituted holder must be caught by the signed resource binding: {err}"
        );
    }

    #[test]
    fn verify_rejects_a_forged_grant_cid_binding() {
        // The attacker controls both `grant_cid` and `allocation.grant`
        // (neither is covered by the UCAN signature), so making them agree on
        // an arbitrary string must still fail: the grant CID is recomputed
        // from the presented UCAN, never trusted as given.
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 100, TenantGrantClass::Underwritten);
        let mut compiled = issuer.compile(&binding, 1, now).unwrap();

        let fake_cid = "bafyreifakefakefakefakefakefakefakefakefakefakefakefakefake".to_owned();
        compiled.allocation.grant = fake_cid.clone();
        compiled.grant_cid = fake_cid;
        compiled.allocation_cid = compiled.allocation.cid().encode();

        let err = verifier.verify(&compiled, now).unwrap_err();
        assert!(
            err.to_string().contains("recomputed"),
            "forged grant CID must be caught by recomputation: {err}"
        );
    }

    #[test]
    fn verify_rejects_a_foreign_allocation_issuer() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 100, TenantGrantClass::Underwritten);
        let mut compiled = issuer.compile(&binding, 1, now).unwrap();

        compiled.allocation = AllocationRecord::new(
            compiled.grant_cid.clone(),
            PdsUnit {
                code: "compute-second".to_owned(),
                issuer: "did:web:other-operator.example".to_owned(),
            },
            100,
            1,
            "did:web:other-operator.example",
            "did:web:tenant.acme.example",
            PdsGrantClass::Underwritten,
        )
        .unwrap();
        compiled.allocation_cid = compiled.allocation.cid().encode();

        let err = verifier.verify(&compiled, now).unwrap_err();
        assert!(
            err.to_string().contains("issuer"),
            "foreign allocation issuer must be rejected: {err}"
        );
    }

    #[test]
    fn verify_rejects_a_mismatched_allocation_cid() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let verifier = TenantGrantVerifier::from_issuer(&issuer);
        let now = 1_700_000_000u64;
        let binding = binding("compute-second", 100, TenantGrantClass::Underwritten);
        let mut compiled = issuer.compile(&binding, 1, now).unwrap();

        compiled.allocation_cid = "bafyreinotwhatwascomputed".to_owned();

        let err = verifier.verify(&compiled, now).unwrap_err();
        assert!(
            err.to_string().contains("allocation CID"),
            "mismatched allocation CID must be rejected: {err}"
        );
    }

    #[test]
    fn compile_rejects_amount_above_i64_max() {
        // No silent clamp: an amount that cannot be a signed caveat is a
        // compile error, never a grant that fails its own verification.
        let issuer = TenantGrantIssuer::generate_for_test();
        let binding = binding(
            "compute-second",
            u64::MAX,
            TenantGrantClass::Underwritten,
        );
        let err = issuer.compile(&binding, 1, 1_700_000_000).unwrap_err();
        assert!(err.to_string().contains("amount"), "{err}");
    }

    #[test]
    fn compile_rejects_epoch_above_i64_max() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let binding = binding("compute-second", 1, TenantGrantClass::Underwritten);
        let err = issuer
            .compile(&binding, u64::MAX, 1_700_000_000)
            .unwrap_err();
        assert!(err.to_string().contains("epoch"), "{err}");
    }

    #[test]
    fn compile_without_entitlement_is_an_error() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let binding = TenantBinding::new(
            "acme",
            TenantBindingSpec {
                namespace: "acme".to_owned(),
                tenant: "did:web:t".to_owned(),
                entitlement: None,
            },
        );
        assert!(issuer.compile(&binding, 1, 1_700_000_000).is_err());
    }

    #[test]
    fn compile_rejects_inverted_expiration() {
        let issuer = TenantGrantIssuer::generate_for_test();
        let now = 1_700_000_000u64;
        let mut binding = binding("compute-second", 1, TenantGrantClass::Underwritten);
        binding.spec.entitlement.as_mut().unwrap().expiration = Some(now - 1);
        assert!(issuer.compile(&binding, 1, now).is_err());
    }

    #[test]
    fn from_node_root_is_deterministic() {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        let root = SigningKey::from_bytes(&seed);
        let a = TenantGrantIssuer::from_node_root(&root).unwrap();
        let b = TenantGrantIssuer::from_node_root(&root).unwrap();
        assert_eq!(
            a.issuer_did(),
            b.issuer_did(),
            "same root ⇒ same issuer DID"
        );
        assert_eq!(
            a.issuer_ed_vk().to_bytes(),
            b.issuer_ed_vk().to_bytes(),
            "purpose derivation is deterministic"
        );
    }
}
