//! Dependency-light semantic contracts for dual-attested resources (#1067).
//!
//! This crate deliberately exposes **no Serde implementation and no wire
//! codec**. #1065 exclusively owns canonical DAG-CBOR/CDDL. Values entering a
//! trust boundary must pass the validating constructors in this crate; public
//! struct layout is not a persistence or interoperability format.

#![forbid(unsafe_code)]

use core::fmt;

/// Current semantic-contract version. Wire format versions remain #1065-owned.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u16)]
pub enum ContractVersion {
    V1 = 1,
}

/// Canonical intent format understood by a configured #1065 codec.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u16)]
pub enum IntentFormatVersion {
    V1 = 1,
}

/// Digest suite for `ResourceIntent` canonical bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u16)]
pub enum DigestSuite {
    Blake3_256 = 1,
}

/// Signature profile reference; this is semantic metadata, not wire encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u16)]
pub enum SignatureSuite {
    /// `hyprstream-crypto` `id-MLDSA65-Ed25519`, composite profile v1.
    CompositeEd25519MlDsa65V1 = 1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KeyPurpose {
    MacResourceAttestation,
    LedgerResourceAttestation,
    RegistrarFinalization,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceId([u8; 16]);

impl ResourceId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OperationId([u8; 16]);

impl OperationId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransferId([u8; 16]);

impl TransferId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IntentDigest([u8; 32]);

impl IntentDigest {
    const DOMAIN: &'static [u8] = b"hs-resource-intent-v1\0";

    #[must_use]
    pub fn compute(suite: DigestSuite, canonical_bytes: &[u8]) -> Self {
        match suite {
            DigestSuite::Blake3_256 => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(Self::DOMAIN);
                hasher.update(canonical_bytes);
                Self(*hasher.finalize().as_bytes())
            }
        }
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Cid(Vec<u8>);

impl Cid {
    /// #1065's codec must validate canonical CID bytes before calling this.
    pub fn from_validated_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Did(String);

impl Did {
    /// DID syntax alone is not trust; accepted-current resolution is mandatory.
    pub fn from_validated_string(value: String) -> Self {
        Self(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A stable, private ownership commitment. It is deliberately not assignable
/// to anonymous capability, entitlement, label, or public-evidence fields.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OwnerCommitment(Vec<u8>);

impl OwnerCommitment {
    /// #1065 validates the selected ownership-commitment construction.
    pub fn from_verified_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// A blinded anonymous capability value bound to exactly one intent digest.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OperationCapabilityCommitment {
    intent_digest: IntentDigest,
    bytes: Vec<u8>,
}

impl OperationCapabilityCommitment {
    pub fn from_verified_bytes(intent_digest: IntentDigest, bytes: Vec<u8>) -> Self {
        Self {
            intent_digest,
            bytes,
        }
    }

    pub const fn intent_digest(&self) -> IntentDigest {
        self.intent_digest
    }
}

/// A one-use anonymous entitlement/nullifier bound to exactly one intent.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OperationEntitlementCommitment {
    intent_digest: IntentDigest,
    bytes: Vec<u8>,
}

impl OperationEntitlementCommitment {
    pub fn from_verified_bytes(intent_digest: IntentDigest, bytes: Vec<u8>) -> Self {
        Self {
            intent_digest,
            bytes,
        }
    }

    pub const fn intent_digest(&self) -> IntentDigest {
        self.intent_digest
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ContentLabelCommitment(Vec<u8>);

impl ContentLabelCommitment {
    pub fn from_verified_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PrivateEvidenceCommitment(Vec<u8>);

impl PrivateEvidenceCommitment {
    pub fn from_verified_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// The only evidence commitment permitted in a public projection. Its digest
/// binding prevents copying a holder/entitlement value across redemptions.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PublicEvidenceCommitment {
    intent_digest: IntentDigest,
    bytes: Vec<u8>,
}

impl PublicEvidenceCommitment {
    pub fn from_verified_bytes(intent_digest: IntentDigest, bytes: Vec<u8>) -> Self {
        Self {
            intent_digest,
            bytes,
        }
    }

    pub const fn intent_digest(&self) -> IntentDigest {
        self.intent_digest
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Signature(Vec<u8>);

impl Signature {
    pub fn from_envelope_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// Mirrors the canonical `hyprstream_rpc::auth::mac::Assurance` discriminants.
/// It has no serialization contract. #1065 MUST use checked conversion and
/// byte-compatibility vectors against the canonical owner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Assurance {
    Unverified = 0,
    Classical = 1,
    PqHybrid = 2,
}

impl Assurance {
    #[must_use]
    pub fn minimum_all(values: impl IntoIterator<Item = Self>) -> Self {
        values.into_iter().min().unwrap_or(Self::Unverified)
    }
}

impl TryFrom<u8> for Assurance {
    type Error = ContractError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unverified),
            1 => Ok(Self::Classical),
            2 => Ok(Self::PqHybrid),
            _ => Err(ContractError::UnsupportedAssurance),
        }
    }
}

impl From<Assurance> for u8 {
    fn from(value: Assurance) -> Self {
        value as u8
    }
}

/// The minimum assurance a canonical intent is permitted to require. Unlike
/// [`Assurance`], this type intentionally has no `Unverified` variant: an
/// intent cannot make unverified evidence sufficient by selecting a zero floor.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum RequiredAssurance {
    Classical = 1,
    PqHybrid = 2,
}

impl RequiredAssurance {
    const fn is_met_by(self, effective: Assurance) -> bool {
        (effective as u8) >= (self as u8)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OwnershipRef {
    Identified(Did),
    Pairwise(Did),
    Committed(OwnerCommitment),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControllerRef {
    Identified(Did),
    Pairwise(Did),
    AnonymousCapability(OperationCapabilityCommitment),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PayerRef {
    Identified(Did),
    Pairwise(Did),
    AnonymousEntitlement(OperationEntitlementCommitment),
}

/// Closed operation vocabulary. Namespace link/rename/bind operations use
/// #1071's separate child-owned vocabulary and will never be added here
/// without a semantic-contract version increase.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceOperation {
    Create,
    Mutate,
    TransferTitle,
    Delete,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ResourceKind {
    ImmutableBlob,
    MutableFile,
    Directory,
}

/// Content-CID presence is derived from the closed resource-kind/operation
/// matrix, never selected by an untrusted canonical claim.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ContentCidRequirement {
    Required,
    Forbidden,
}

const fn content_cid_requirement(
    resource_kind: &ResourceKind,
    operation: ResourceOperation,
) -> Result<ContentCidRequirement, ContractError> {
    match (resource_kind, operation) {
        (ResourceKind::ImmutableBlob, ResourceOperation::Create)
        | (ResourceKind::MutableFile, ResourceOperation::Create | ResourceOperation::Mutate) => {
            Ok(ContentCidRequirement::Required)
        }
        (ResourceKind::Directory, ResourceOperation::Create | ResourceOperation::Mutate)
        | (_, ResourceOperation::TransferTitle | ResourceOperation::Delete) => {
            Ok(ContentCidRequirement::Forbidden)
        }
        (ResourceKind::ImmutableBlob, ResourceOperation::Mutate) => {
            Err(ContractError::InvalidResourceOperation)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredecessorRef {
    pub manifest_cid: Cid,
    pub version: u64,
}

/// Semantic claims returned by the trusted #1065 canonical decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntentClaims {
    pub resource_id: ResourceId,
    pub operation_id: OperationId,
    pub operation: ResourceOperation,
    pub resource_kind: ResourceKind,
    pub expected_predecessor: Option<PredecessorRef>,
    pub owner: OwnershipRef,
    pub controller: ControllerRef,
    pub payer: PayerRef,
    pub required_assurance: RequiredAssurance,
    pub unit: String,
    pub max_charge: u128,
}

/// Untrusted wire/persistence candidate presented to the validating boundary.
pub struct IntentCandidate {
    pub contract_version: ContractVersion,
    pub format_version: IntentFormatVersion,
    pub digest_suite: DigestSuite,
    pub claimed_digest: IntentDigest,
    pub canonical_bytes: Vec<u8>,
}

/// Trusted codec boundary implemented by #1065.
pub trait IntentValidator {
    fn validate_and_decode(
        &self,
        version: IntentFormatVersion,
        canonical_bytes: &[u8],
    ) -> Result<IntentClaims, ContractError>;
}

/// A canonical intent can only be created after canonical decoding and digest
/// verification. Its fields are intentionally private.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalResourceIntent {
    contract_version: ContractVersion,
    format_version: IntentFormatVersion,
    digest_suite: DigestSuite,
    digest: IntentDigest,
    canonical_bytes: Vec<u8>,
    claims: IntentClaims,
}

impl CanonicalResourceIntent {
    pub fn validate(
        candidate: IntentCandidate,
        validator: &dyn IntentValidator,
    ) -> Result<Self, ContractError> {
        if candidate.canonical_bytes.is_empty() || candidate.canonical_bytes.len() > 1_048_576 {
            return Err(ContractError::IntentBounds);
        }
        let claims =
            validator.validate_and_decode(candidate.format_version, &candidate.canonical_bytes)?;
        let predecessor_is_valid = matches!(claims.operation, ResourceOperation::Create)
            == claims.expected_predecessor.is_none();
        if !predecessor_is_valid {
            return Err(ContractError::InvalidPredecessor);
        }
        content_cid_requirement(&claims.resource_kind, claims.operation)?;
        let digest = IntentDigest::compute(candidate.digest_suite, &candidate.canonical_bytes);
        if digest != candidate.claimed_digest {
            return Err(ContractError::DigestMismatch);
        }
        Ok(Self {
            contract_version: candidate.contract_version,
            format_version: candidate.format_version,
            digest_suite: candidate.digest_suite,
            digest,
            canonical_bytes: candidate.canonical_bytes,
            claims,
        })
    }

    pub const fn digest(&self) -> IntentDigest {
        self.digest
    }

    pub fn claims(&self) -> &IntentClaims {
        &self.claims
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub const fn contract_version(&self) -> ContractVersion {
        self.contract_version
    }

    pub const fn format_version(&self) -> IntentFormatVersion {
        self.format_version
    }

    pub const fn digest_suite(&self) -> DigestSuite {
        self.digest_suite
    }
}

/// Signer selection and historical-verification coordinates authenticated by
/// the envelope. `(signer, key_id, key_epoch)` resolves through retained
/// accepted-current evidence; `state_epoch` is not a substitute for key epoch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignatureEnvelope {
    pub contract_version: ContractVersion,
    pub suite: SignatureSuite,
    pub signer: Did,
    pub key_id: String,
    pub key_epoch: u64,
    pub purpose: KeyPurpose,
    pub signed_domain: String,
    pub signature: Signature,
}

/// Anonymous profiles use a blinded, operation-scoped value that is not
/// holder-stable or cross-operation-linkable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapabilityBinding {
    IdentifiedRoot(Cid),
    AnonymousOperationCommitment(OperationCapabilityCommitment),
}

/// Digest-only MAC statement; richer claim data is re-derived from canonical bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MacAttestationCandidate {
    pub contract_version: ContractVersion,
    pub intent_digest: IntentDigest,
    pub policy_epoch: u64,
    pub state_epoch: u64,
    pub controller: ControllerRef,
    pub capability_binding: CapabilityBinding,
    pub content_label_commitment: ContentLabelCommitment,
    /// Unix seconds on the registrar's monotonic-clamped time basis.
    pub expires_at: u64,
    pub envelope: SignatureEnvelope,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LedgerPhase {
    Reserved,
    Posted,
    ZeroCostPosted,
    Voided,
    Compensated,
}

/// Digest-only economic statement. This is private registrar/evidence material,
/// never the ordinary namespace/public-PDS projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerAttestationCandidate {
    pub contract_version: ContractVersion,
    pub intent_digest: IntentDigest,
    pub ledger_id: Did,
    pub payer: PayerRef,
    pub issuer: Did,
    pub unit: String,
    pub amount: u128,
    pub transfer_id: TransferId,
    pub phase: LedgerPhase,
    pub envelope: SignatureEnvelope,
}

/// An accepted-current authority root supplied by the authority-specific
/// verifier. A DID or a `KeyPurpose` label is not an authority root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedAuthority {
    pub root: Cid,
    pub signer: Did,
    pub key_id: String,
    pub key_epoch: u64,
}

/// Result material returned by a trusted attestation verifier. The verifier is
/// the cryptographic/accepted-current boundary; this crate then seals it into
/// a non-public verified evidence value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedEvidence {
    pub attestation_cid: Cid,
    pub authority: AcceptedAuthority,
    /// Assurance derived while verifying this leg's accepted-current evidence,
    /// rather than copied from the untrusted candidate.
    pub assurance: Assurance,
}

pub trait MacAttestationVerifier {
    /// Verify the envelope and resolve it under the MAC accepted-current root.
    fn verify_accepted_current(
        &self,
        intent: &CanonicalResourceIntent,
        candidate: &MacAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError>;
}

pub trait LedgerAttestationVerifier {
    /// Verify the envelope and resolve it under the ledger accepted-current root.
    fn verify_accepted_current(
        &self,
        intent: &CanonicalResourceIntent,
        candidate: &LedgerAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError>;
}

/// Deployment policy for the separation of the two signing authorities.
/// Default deployments reject a shared accepted root or signer. Service
/// or host co-location is not itself evidence of independence: if it lets one
/// compromise control both roots/keys, this policy must reject the deployment
/// or the deployment has explicitly accepted that residual risk outside this
/// contract.
pub trait AttestationIndependencePolicy {
    fn require_independence(
        &self,
        mac: &AcceptedAuthority,
        ledger: &AcceptedAuthority,
    ) -> Result<(), ContractError>;
}

/// The baseline policy used by the canonical profile rejects a shared
/// authority root or signer DID. It deliberately does not compare `key_id`:
/// that value can be a local fragment (for example, `#resource-v1`) scoped by
/// its authority namespace, and equal fragments do not prove shared key
/// material. A deployment that needs cross-namespace key-material detection
/// must supply an independence policy with a globally scoped verifier-derived
/// key identity or fingerprint.
pub struct RejectSharedAttester;

impl AttestationIndependencePolicy for RejectSharedAttester {
    fn require_independence(
        &self,
        mac: &AcceptedAuthority,
        ledger: &AcceptedAuthority,
    ) -> Result<(), ContractError> {
        if mac.root == ledger.root || mac.signer == ledger.signer {
            return Err(ContractError::SharedAttester);
        }
        Ok(())
    }
}

/// Verified evidence is minted only by `verify_and_join`; raw candidates do
/// not satisfy the registrar interface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedMacAttestation {
    attestation: MacAttestationCandidate,
    evidence: VerifiedEvidence,
}

impl VerifiedMacAttestation {
    pub fn attestation(&self) -> &MacAttestationCandidate {
        &self.attestation
    }

    pub fn attestation_cid(&self) -> &Cid {
        &self.evidence.attestation_cid
    }

    pub fn authority(&self) -> &AcceptedAuthority {
        &self.evidence.authority
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedLedgerAttestation {
    attestation: LedgerAttestationCandidate,
    evidence: VerifiedEvidence,
}

impl VerifiedLedgerAttestation {
    pub fn attestation(&self) -> &LedgerAttestationCandidate {
        &self.attestation
    }

    pub fn attestation_cid(&self) -> &Cid {
        &self.evidence.attestation_cid
    }

    pub fn authority(&self) -> &AcceptedAuthority {
        &self.evidence.authority
    }
}

/// Joined attestations can only be constructed after both signatures have been
/// verified under their own accepted-current roots, the roots pass deployment
/// separation policy, and all cross-claim/accounting checks pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualAttestation {
    mac: VerifiedMacAttestation,
    ledger: VerifiedLedgerAttestation,
    effective_assurance: Assurance,
}

impl DualAttestation {
    pub fn verify_and_join(
        intent: &CanonicalResourceIntent,
        mac: MacAttestationCandidate,
        ledger: LedgerAttestationCandidate,
        mac_verifier: &dyn MacAttestationVerifier,
        ledger_verifier: &dyn LedgerAttestationVerifier,
        independence: &dyn AttestationIndependencePolicy,
    ) -> Result<Self, ContractError> {
        if mac.intent_digest != intent.digest()
            || ledger.intent_digest != intent.digest()
            || mac.intent_digest != ledger.intent_digest
            || mac.controller != intent.claims().controller
            || ledger.payer != intent.claims().payer
        {
            return Err(ContractError::IntentMismatch);
        }
        if !matches!(
            ledger.phase,
            LedgerPhase::Posted | LedgerPhase::ZeroCostPosted
        ) {
            return Err(ContractError::LedgerNotPosted);
        }
        if ledger.unit != intent.claims().unit || ledger.amount > intent.claims().max_charge {
            return Err(ContractError::ChargeMismatch);
        }
        if mac.envelope.purpose != KeyPurpose::MacResourceAttestation
            || ledger.envelope.purpose != KeyPurpose::LedgerResourceAttestation
        {
            return Err(ContractError::WrongKeyPurpose);
        }
        if let ControllerRef::AnonymousCapability(binding) = &mac.controller {
            if binding.intent_digest() != intent.digest() {
                return Err(ContractError::IntentMismatch);
            }
        }
        if let CapabilityBinding::AnonymousOperationCommitment(binding) = &mac.capability_binding {
            if binding.intent_digest() != intent.digest() {
                return Err(ContractError::IntentMismatch);
            }
        }
        if let PayerRef::AnonymousEntitlement(binding) = &ledger.payer {
            if binding.intent_digest() != intent.digest() {
                return Err(ContractError::IntentMismatch);
            }
        }
        let mac_evidence = mac_verifier.verify_accepted_current(intent, &mac)?;
        let ledger_evidence = ledger_verifier.verify_accepted_current(intent, &ledger)?;
        if mac_evidence.authority.signer != mac.envelope.signer
            || mac_evidence.authority.key_id != mac.envelope.key_id
            || mac_evidence.authority.key_epoch != mac.envelope.key_epoch
            || ledger_evidence.authority.signer != ledger.envelope.signer
            || ledger_evidence.authority.key_id != ledger.envelope.key_id
            || ledger_evidence.authority.key_epoch != ledger.envelope.key_epoch
        {
            return Err(ContractError::InvalidAttestationEvidence);
        }
        independence.require_independence(&mac_evidence.authority, &ledger_evidence.authority)?;
        let effective_assurance =
            Assurance::minimum_all([mac_evidence.assurance, ledger_evidence.assurance]);
        if !intent
            .claims()
            .required_assurance
            .is_met_by(effective_assurance)
        {
            return Err(ContractError::AssuranceRequirementNotMet);
        }
        Ok(Self {
            mac: VerifiedMacAttestation {
                attestation: mac,
                evidence: mac_evidence,
            },
            ledger: VerifiedLedgerAttestation {
                attestation: ledger,
                evidence: ledger_evidence,
            },
            effective_assurance,
        })
    }

    pub const fn effective_assurance(&self) -> Assurance {
        self.effective_assurance
    }

    pub fn mac(&self) -> &VerifiedMacAttestation {
        &self.mac
    }

    pub fn ledger(&self) -> &VerifiedLedgerAttestation {
        &self.ledger
    }
}

/// Registrar term is monotonically increasing across leadership/authority
/// epochs (the #1069-owned backend term-allocation rule); generation is
/// monotonically increasing per resource inside a term. The compare-and-apply
/// contract orders numeric `(registrar_term, generation, resource_version)`
/// tuples.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FencingToken {
    pub resource_id: ResourceId,
    pub registrar_term: u64,
    pub generation: u64,
}

/// Signed registrar statement. It is untrusted until accepted by
/// `FinalizedResource::verify` and a configured accepted-current verifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalizationStatement {
    pub contract_version: ContractVersion,
    pub operation_id: OperationId,
    pub resource_id: ResourceId,
    pub resource_version: u64,
    pub intent_digest: IntentDigest,
    pub fence: FencingToken,
    pub manifest_cid: Cid,
    pub content_cid: Option<Cid>,
    pub mac_attestation_cid: Cid,
    pub ledger_attestation_cid: Cid,
    pub private_evidence_commitment: PrivateEvidenceCommitment,
    pub envelope: SignatureEnvelope,
}

pub trait FinalizationProofVerifier {
    /// Verify the registrar statement under accepted-current registrar
    /// authority, with the verified attestation provenance it names.
    fn verify(
        &self,
        statement: &FinalizationStatement,
        dual: &DualAttestation,
    ) -> Result<(), ContractError>;
}

/// Verified registrar output. Private fields prevent fabrication by namespace,
/// storage, PDS, or recovery callers.
///
/// ```compile_fail
/// use hyprstream_resource::FinalizedResource;
/// // No public fields or unchecked constructor exist.
/// let forged = FinalizedResource {};
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalizedResource {
    operation_id: OperationId,
    resource_id: ResourceId,
    resource_version: u64,
    fence: FencingToken,
    manifest_cid: Cid,
    content_cid: Option<Cid>,
    intent_digest: IntentDigest,
    mac_attestation_cid: Cid,
    ledger_attestation_cid: Cid,
    private_evidence_commitment: PrivateEvidenceCommitment,
}

impl FinalizedResource {
    pub fn verify(
        intent: &CanonicalResourceIntent,
        dual: &DualAttestation,
        statement: &FinalizationStatement,
        verifier: &dyn FinalizationProofVerifier,
    ) -> Result<Self, ContractError> {
        verifier.verify(statement, dual)?;
        if statement.envelope.purpose != KeyPurpose::RegistrarFinalization {
            return Err(ContractError::WrongKeyPurpose);
        }
        if statement.intent_digest != intent.digest()
            // Ties the JOINED attestation to the intent being finalized. Without this the
            // statement can name the right intent while the dual evidence attests another
            // one — the statement's own digest proves nothing about what was attested.
            // Deleted in 26b0bce5f while closing the authority-floor findings; restored.
            || dual.mac().attestation().intent_digest != statement.intent_digest
            || statement.operation_id != intent.claims().operation_id
            || statement.resource_id != intent.claims().resource_id
            || statement.fence.resource_id != statement.resource_id
        {
            return Err(ContractError::IntentMismatch);
        }
        if statement.mac_attestation_cid != *dual.mac().attestation_cid()
            || statement.ledger_attestation_cid != *dual.ledger().attestation_cid()
        {
            return Err(ContractError::AttestationProvenanceMismatch);
        }
        match content_cid_requirement(&intent.claims().resource_kind, intent.claims().operation)? {
            ContentCidRequirement::Required if statement.content_cid.is_none() => {
                return Err(ContractError::ContentRequired);
            }
            ContentCidRequirement::Forbidden if statement.content_cid.is_some() => {
                return Err(ContractError::ContentForbidden);
            }
            ContentCidRequirement::Required | ContentCidRequirement::Forbidden => {}
        }
        Ok(Self {
            operation_id: statement.operation_id,
            resource_id: statement.resource_id,
            resource_version: statement.resource_version,
            fence: statement.fence,
            manifest_cid: statement.manifest_cid.clone(),
            content_cid: statement.content_cid.clone(),
            intent_digest: statement.intent_digest,
            mac_attestation_cid: statement.mac_attestation_cid.clone(),
            ledger_attestation_cid: statement.ledger_attestation_cid.clone(),
            private_evidence_commitment: statement.private_evidence_commitment.clone(),
        })
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn resource_id(&self) -> ResourceId {
        self.resource_id
    }

    pub const fn resource_version(&self) -> u64 {
        self.resource_version
    }

    pub const fn fence(&self) -> FencingToken {
        self.fence
    }

    pub fn manifest_cid(&self) -> &Cid {
        &self.manifest_cid
    }

    pub fn content_cid(&self) -> Option<&Cid> {
        self.content_cid.as_ref()
    }

    pub fn mac_attestation_cid(&self) -> &Cid {
        &self.mac_attestation_cid
    }

    pub fn ledger_attestation_cid(&self) -> &Cid {
        &self.ledger_attestation_cid
    }

    pub fn private_evidence_commitment(&self) -> &PrivateEvidenceCommitment {
        &self.private_evidence_commitment
    }

    /// Registrar-minted publication material. Callers cannot assemble a
    /// projection directly, and the public evidence must belong to this exact
    /// finalized intent.
    pub fn public_projection(
        &self,
        public_evidence_commitment: PublicEvidenceCommitment,
    ) -> Result<PublicResourceProjection, ContractError> {
        if public_evidence_commitment.intent_digest() != self.intent_digest {
            return Err(ContractError::IntentMismatch);
        }
        Ok(PublicResourceProjection {
            resource_id: self.resource_id,
            resource_version: self.resource_version,
            manifest_cid: self.manifest_cid.clone(),
            content_cid: self.content_cid.clone(),
            public_evidence_commitment,
        })
    }

    pub fn publish_effect(
        &self,
        entry: Option<ProjectionEntryId>,
        projection: PublicResourceProjection,
    ) -> Result<ProjectionEffect, ContractError> {
        if projection.resource_id != self.resource_id
            || projection.resource_version != self.resource_version
        {
            return Err(ContractError::InvalidProjection);
        }
        Ok(ProjectionEffect {
            operation_id: self.operation_id,
            resource_id: self.resource_id,
            resource_version: self.resource_version,
            fence: self.fence,
            entry,
            action: ProjectionAction::Publish(projection),
        })
    }

    pub fn withdraw_effect(
        &self,
        entry: Option<ProjectionEntryId>,
        target: WithdrawalTarget,
    ) -> Result<ProjectionEffect, ContractError> {
        if target.resource_id != self.resource_id || target.fence.resource_id != self.resource_id {
            return Err(ContractError::InvalidProjection);
        }
        Ok(ProjectionEffect {
            operation_id: self.operation_id,
            resource_id: self.resource_id,
            resource_version: self.resource_version,
            fence: self.fence,
            entry,
            action: ProjectionAction::Withdraw(target),
        })
    }
}

/// Privacy-minimized namespace/public-PDS projection. It contains no detailed
/// attestation CID, payer, issuer, unit, amount, transfer ID, capability root,
/// or signing-key coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicResourceProjection {
    resource_id: ResourceId,
    resource_version: u64,
    manifest_cid: Cid,
    content_cid: Option<Cid>,
    public_evidence_commitment: PublicEvidenceCommitment,
}

impl PublicResourceProjection {
    pub const fn resource_id(&self) -> ResourceId {
        self.resource_id
    }

    pub const fn resource_version(&self) -> u64 {
        self.resource_version
    }

    pub fn manifest_cid(&self) -> &Cid {
        &self.manifest_cid
    }

    pub fn content_cid(&self) -> Option<&Cid> {
        self.content_cid.as_ref()
    }

    pub fn public_evidence_commitment(&self) -> &PublicEvidenceCommitment {
        &self.public_evidence_commitment
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProjectionEntryId(Vec<u8>);

impl ProjectionEntryId {
    pub fn from_validated_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WithdrawalTarget {
    pub resource_id: ResourceId,
    pub resource_version: u64,
    pub fence: FencingToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectionAction {
    Publish(PublicResourceProjection),
    Withdraw(WithdrawalTarget),
}

/// Ordered desired-state effect. Publishers compare `(registrar_term,
/// generation, resource_version)` and reject stale or conflicting effects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionEffect {
    operation_id: OperationId,
    resource_id: ResourceId,
    resource_version: u64,
    fence: FencingToken,
    entry: Option<ProjectionEntryId>,
    action: ProjectionAction,
}

impl ProjectionEffect {
    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }
    pub const fn resource_id(&self) -> ResourceId {
        self.resource_id
    }
    pub const fn resource_version(&self) -> u64 {
        self.resource_version
    }
    pub const fn fence(&self) -> FencingToken {
        self.fence
    }
    pub fn entry(&self) -> Option<&ProjectionEntryId> {
        self.entry.as_ref()
    }
    pub const fn action(&self) -> &ProjectionAction {
        &self.action
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegistrarState {
    Requested,
    MacAuthorized,
    LedgerReserved,
    Materialized,
    LedgerPosted,
    Quarantined,
    ManualReview,
    Finalized,
    Voided,
    Compensated,
    RejectedConflict,
}

impl RegistrarState {
    /// Terminal states require exactly one matching `Resolution` record.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Finalized | Self::Voided | Self::Compensated | Self::RejectedConflict
        )
    }

    fn matches_resolution(self, resolution: &Resolution) -> bool {
        matches!(
            (self, resolution),
            (Self::Finalized, Resolution::Finalized { .. })
                | (Self::Voided, Resolution::Voided { .. })
                | (Self::Compensated, Resolution::Compensated { .. })
                | (Self::RejectedConflict, Resolution::RejectedConflict { .. })
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QuarantineReason {
    LedgerOutcomeUnknown,
    PostedBeforeFinalization,
    FinalizationConflict,
    CompensationFailed,
    DependencyStateUnverifiable,
    InvariantViolation,
}

/// The losing side's accounting disposition in a `Resolution::RejectedConflict`
/// record. The referenced transfer MUST be durably confirmed by the economic
/// authority before the terminal resolution is committed, so an identical
/// retry or recovery query discovers the exact economic outcome.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConflictAccountingResolution {
    /// Losing reservation released without posting.
    Voided { void_transfer_id: TransferId },
    /// Losing posted charge corrected by an immutable compensating transfer.
    Compensated {
        compensation_transfer_id: TransferId,
    },
}

/// Immutable terminal resolution. `ManualReview` and `Quarantined` are never
/// terminal until one of these records is durably committed. A record that
/// references an accounting effect (`Voided`, `Compensated`, or the losing
/// side's disposition in `RejectedConflict`) is committed only after the
/// referenced transfer is durably confirmed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Resolution {
    Finalized {
        target: ResolutionTarget,
        resource_version: u64,
        manifest_cid: Cid,
    },
    Voided {
        target: ResolutionTarget,
        void_transfer_id: TransferId,
    },
    Compensated {
        target: ResolutionTarget,
        compensation_transfer_id: TransferId,
    },
    RejectedConflict {
        target: ResolutionTarget,
        winning_operation_id: OperationId,
        winning_version: u64,
        losing_accounting: ConflictAccountingResolution,
    },
}

/// Immutable identity of the operation whose terminal outcome is recorded.
/// Every resolution variant carries it; no terminal accounting outcome can be
/// attached to a different operation/resource/fence.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResolutionTarget {
    pub operation_id: OperationId,
    pub resource_id: ResourceId,
    pub fence: FencingToken,
}

impl Resolution {
    pub const fn target(&self) -> ResolutionTarget {
        match self {
            Self::Finalized { target, .. }
            | Self::Voided { target, .. }
            | Self::Compensated { target, .. }
            | Self::RejectedConflict { target, .. } => *target,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionRef {
    pub record_cid: Cid,
    pub outcome: Resolution,
}

/// Registrar snapshot of one operation. Fields are private: construction goes
/// through [`RegistrationSnapshot::new`], which rejects terminal states
/// without exactly one matching resolution and nonterminal states carrying
/// one, so an inconsistent state/resolution pair is unrepresentable.
///
/// ```compile_fail
/// use hyprstream_resource::RegistrationSnapshot;
/// // No public fields or unchecked constructor exist.
/// let forged = RegistrationSnapshot {};
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistrationSnapshot {
    operation_id: OperationId,
    intent_digest: IntentDigest,
    state: RegistrarState,
    fencing_token: FencingToken,
    quarantine_reason: Option<QuarantineReason>,
    resolution: Option<ResolutionRef>,
}

impl RegistrationSnapshot {
    pub fn new(
        operation_id: OperationId,
        intent_digest: IntentDigest,
        state: RegistrarState,
        fencing_token: FencingToken,
        quarantine_reason: Option<QuarantineReason>,
        resolution: Option<ResolutionRef>,
    ) -> Result<Self, ContractError> {
        let consistent = match &resolution {
            Some(reference) => {
                state.is_terminal()
                    && state.matches_resolution(&reference.outcome)
                    && reference.outcome.target().operation_id == operation_id
                    && reference.outcome.target().resource_id == fencing_token.resource_id
                    && reference.outcome.target().fence == fencing_token
            }
            None => !state.is_terminal(),
        };
        let quarantine_is_consistent = matches!(
            state,
            RegistrarState::Quarantined | RegistrarState::ManualReview
        ) == quarantine_reason.is_some();
        if !consistent || !quarantine_is_consistent {
            return Err(ContractError::InconsistentResolution);
        }
        Ok(Self {
            operation_id,
            intent_digest,
            state,
            fencing_token,
            quarantine_reason,
            resolution,
        })
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn intent_digest(&self) -> IntentDigest {
        self.intent_digest
    }

    pub const fn state(&self) -> RegistrarState {
        self.state
    }

    pub const fn fencing_token(&self) -> FencingToken {
        self.fencing_token
    }

    pub const fn quarantine_reason(&self) -> Option<QuarantineReason> {
        self.quarantine_reason
    }

    pub const fn resolution(&self) -> Option<&ResolutionRef> {
        self.resolution.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthorityError {
    Denied,
    /// Retry classification only. For publication/visibility this is exactly a
    /// deny and MUST NOT trigger weaker authority or profile fallback.
    Unavailable,
    Stale,
    InvalidAttestation,
    IntentMismatch,
}

impl fmt::Display for AuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AuthorityError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractError {
    UnsupportedAssurance,
    AssuranceRequirementNotMet,
    IntentBounds,
    NonCanonicalIntent,
    DigestMismatch,
    IntentMismatch,
    LedgerNotPosted,
    ChargeMismatch,
    WrongKeyPurpose,
    InvalidAttestationEvidence,
    SharedAttester,
    AttestationProvenanceMismatch,
    InvalidPredecessor,
    InvalidResourceOperation,
    ContentRequired,
    ContentForbidden,
    InvalidProjection,
    InvalidFinalizationProof,
    StaleProjection,
    InconsistentResolution,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ContractError {}

pub trait MacAuthority {
    fn attest(
        &self,
        intent: &CanonicalResourceIntent,
    ) -> Result<MacAttestationCandidate, AuthorityError>;
    fn revalidate(
        &self,
        intent: &CanonicalResourceIntent,
        attestation: &MacAttestationCandidate,
    ) -> Result<(), AuthorityError>;
}

pub trait EconomicAuthority {
    fn reserve(
        &self,
        intent: &CanonicalResourceIntent,
    ) -> Result<LedgerAttestationCandidate, AuthorityError>;

    /// MUST require a `Reserved` input, enforce `actual_amount <= reserved`,
    /// and preserve the deterministic transfer lineage from operation ID.
    fn post(
        &self,
        intent: &CanonicalResourceIntent,
        reservation: &LedgerAttestationCandidate,
        actual_amount: u128,
    ) -> Result<LedgerAttestationCandidate, AuthorityError>;

    fn void(
        &self,
        intent: &CanonicalResourceIntent,
        reservation: &LedgerAttestationCandidate,
    ) -> Result<LedgerAttestationCandidate, AuthorityError>;

    /// MUST require a `Posted` input and create a new immutable correction.
    fn compensate(
        &self,
        intent: &CanonicalResourceIntent,
        posted: &LedgerAttestationCandidate,
    ) -> Result<LedgerAttestationCandidate, AuthorityError>;
}

pub trait ResourceMaterializer {
    type Staged;
    type Error;

    fn stage(
        &self,
        operation_id: OperationId,
        fence: FencingToken,
    ) -> Result<Self::Staged, Self::Error>;
    fn seal(&self, staged: &Self::Staged, fence: FencingToken) -> Result<Cid, Self::Error>;
    fn discard(&self, staged: &Self::Staged, fence: FencingToken) -> Result<(), Self::Error>;
}

pub trait ResourceRegistrar {
    type Error;

    fn begin(&self, intent: CanonicalResourceIntent) -> Result<RegistrationSnapshot, Self::Error>;
    fn get(&self, operation_id: OperationId) -> Result<RegistrationSnapshot, Self::Error>;

    /// Implementations revalidate MAC immediately before their linearizable CAS,
    /// sign a finalization statement, verify it into `FinalizedResource`, and
    /// durably record the exact `Resolution` returned to identical retries.
    fn finalize(
        &self,
        operation_id: OperationId,
        fence: FencingToken,
        intent: &CanonicalResourceIntent,
        attestations: &DualAttestation,
        content_cid: Option<&Cid>,
    ) -> Result<FinalizedResource, Self::Error>;

    /// Commits the immutable terminal `Resolution`. When the resolution
    /// references an accounting effect (`Voided`, `Compensated`, or the losing
    /// side's disposition in `RejectedConflict`), the implementation MUST
    /// durably confirm the referenced transfer with the economic authority
    /// before committing the resolution; until then the operation remains
    /// nonterminal.
    fn resolve(
        &self,
        operation_id: OperationId,
        fence: FencingToken,
        resolution: Resolution,
    ) -> Result<RegistrationSnapshot, Self::Error>;
}

pub trait ResourcePublisher {
    type Error;

    /// Atomically compare-and-apply desired projection state. Duplicate effects
    /// return the original outcome; older term/generation/version rejects.
    fn compare_and_apply(&self, effect: &ProjectionEffect) -> Result<(), Self::Error>;
}
