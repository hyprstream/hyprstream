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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Commitment(Vec<u8>);

impl Commitment {
    /// The owning profile validates construction and bounds before conversion.
    pub fn from_verified_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OwnershipRef {
    Identified(Did),
    Pairwise(Did),
    Committed(Commitment),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControllerRef {
    Identified(Did),
    Pairwise(Did),
    AnonymousCapability(Commitment),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PayerRef {
    Identified(Did),
    Pairwise(Did),
    AnonymousEntitlement(Commitment),
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

/// Anonymous profiles MUST use `AnonymousOperationCommitment`: a blinded,
/// operation-scoped value that is not holder-stable or cross-operation-linkable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapabilityBinding {
    IdentifiedRoot(Cid),
    AnonymousOperationCommitment(Commitment),
}

/// Digest-only MAC statement; richer claim data is re-derived from canonical bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MacAttestation {
    pub contract_version: ContractVersion,
    pub intent_digest: IntentDigest,
    pub policy_epoch: u64,
    pub state_epoch: u64,
    pub controller: ControllerRef,
    pub capability_binding: CapabilityBinding,
    pub content_label_commitment: Commitment,
    pub assurance: Assurance,
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
pub struct LedgerAttestation {
    pub contract_version: ContractVersion,
    pub intent_digest: IntentDigest,
    pub ledger_id: Did,
    pub payer: PayerRef,
    pub issuer: Did,
    pub unit: String,
    pub amount: u128,
    pub transfer_id: TransferId,
    pub phase: LedgerPhase,
    pub assurance: Assurance,
    pub envelope: SignatureEnvelope,
}

/// Joined attestations can only be constructed after all cross-authority claim
/// and accounting-bound checks pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualAttestation {
    mac: MacAttestation,
    ledger: LedgerAttestation,
    effective_assurance: Assurance,
}

impl DualAttestation {
    pub fn join(
        intent: &CanonicalResourceIntent,
        mac: MacAttestation,
        ledger: LedgerAttestation,
    ) -> Result<Self, ContractError> {
        if mac.intent_digest != intent.digest()
            || ledger.intent_digest != intent.digest()
            || mac.intent_digest != ledger.intent_digest
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
        let effective_assurance = Assurance::minimum_all([mac.assurance, ledger.assurance]);
        Ok(Self {
            mac,
            ledger,
            effective_assurance,
        })
    }

    pub const fn effective_assurance(&self) -> Assurance {
        self.effective_assurance
    }

    pub fn mac(&self) -> &MacAttestation {
        &self.mac
    }

    pub fn ledger(&self) -> &LedgerAttestation {
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
    pub private_evidence_commitment: Commitment,
    pub envelope: SignatureEnvelope,
}

pub trait FinalizationProofVerifier {
    fn verify(&self, statement: &FinalizationStatement) -> Result<(), ContractError>;
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
    private_evidence_commitment: Commitment,
}

impl FinalizedResource {
    pub fn verify(
        intent: &CanonicalResourceIntent,
        dual: &DualAttestation,
        statement: &FinalizationStatement,
        verifier: &dyn FinalizationProofVerifier,
    ) -> Result<Self, ContractError> {
        verifier.verify(statement)?;
        if statement.envelope.purpose != KeyPurpose::RegistrarFinalization {
            return Err(ContractError::WrongKeyPurpose);
        }
        if statement.intent_digest != intent.digest()
            || dual.mac().intent_digest != statement.intent_digest
            || statement.operation_id != intent.claims().operation_id
            || statement.resource_id != intent.claims().resource_id
            || statement.fence.resource_id != statement.resource_id
        {
            return Err(ContractError::IntentMismatch);
        }
        Ok(Self {
            operation_id: statement.operation_id,
            resource_id: statement.resource_id,
            resource_version: statement.resource_version,
            fence: statement.fence,
            manifest_cid: statement.manifest_cid.clone(),
            content_cid: statement.content_cid.clone(),
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

    pub fn private_evidence_commitment(&self) -> &Commitment {
        &self.private_evidence_commitment
    }
}

/// Privacy-minimized namespace/public-PDS projection. It contains no detailed
/// attestation CID, payer, issuer, unit, amount, transfer ID, capability root,
/// or signing-key coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicResourceProjection {
    pub resource_id: ResourceId,
    pub resource_version: u64,
    pub manifest_cid: Cid,
    pub content_cid: Option<Cid>,
    pub public_evidence_commitment: Commitment,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectionAction {
    Publish(PublicResourceProjection),
    Withdraw,
}

/// Ordered desired-state effect. Publishers compare `(registrar_term,
/// generation, resource_version)` and reject stale or conflicting effects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionEffect {
    pub operation_id: OperationId,
    pub resource_id: ResourceId,
    pub resource_version: u64,
    pub fence: FencingToken,
    pub action: ProjectionAction,
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
        operation_id: OperationId,
        resource_version: u64,
        manifest_cid: Cid,
    },
    Voided {
        void_transfer_id: TransferId,
    },
    Compensated {
        compensation_transfer_id: TransferId,
    },
    RejectedConflict {
        winning_operation_id: OperationId,
        winning_version: u64,
        losing_accounting: ConflictAccountingResolution,
    },
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
            Some(reference) => state.is_terminal() && state.matches_resolution(&reference.outcome),
            None => !state.is_terminal(),
        };
        if !consistent {
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
    IntentBounds,
    NonCanonicalIntent,
    DigestMismatch,
    IntentMismatch,
    LedgerNotPosted,
    ChargeMismatch,
    WrongKeyPurpose,
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
    fn attest(&self, intent: &CanonicalResourceIntent) -> Result<MacAttestation, AuthorityError>;
    fn revalidate(
        &self,
        intent: &CanonicalResourceIntent,
        attestation: &MacAttestation,
    ) -> Result<(), AuthorityError>;
}

pub trait EconomicAuthority {
    fn reserve(
        &self,
        intent: &CanonicalResourceIntent,
    ) -> Result<LedgerAttestation, AuthorityError>;

    /// MUST require a `Reserved` input, enforce `actual_amount <= reserved`,
    /// and preserve the deterministic transfer lineage from operation ID.
    fn post(
        &self,
        intent: &CanonicalResourceIntent,
        reservation: &LedgerAttestation,
        actual_amount: u128,
    ) -> Result<LedgerAttestation, AuthorityError>;

    fn void(
        &self,
        intent: &CanonicalResourceIntent,
        reservation: &LedgerAttestation,
    ) -> Result<LedgerAttestation, AuthorityError>;

    /// MUST require a `Posted` input and create a new immutable correction.
    fn compensate(
        &self,
        intent: &CanonicalResourceIntent,
        posted: &LedgerAttestation,
    ) -> Result<LedgerAttestation, AuthorityError>;
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
