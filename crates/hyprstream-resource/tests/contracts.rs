//! Contract tests: validating constructors and negative controls.
//! Panics are idiomatic in test bodies.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_resource::{
    Assurance, CanonicalResourceIntent, Cid, Commitment, ContractError, ContractVersion,
    ControllerRef, Did, DigestSuite, DualAttestation, FinalizationProofVerifier,
    FinalizationStatement, IntentCandidate, IntentClaims, IntentDigest, IntentFormatVersion,
    IntentValidator, KeyPurpose, LedgerAttestation, LedgerPhase, MacAttestation, OperationId,
    PayerRef, ResourceId, ResourceKind, ResourceOperation, Signature, SignatureEnvelope,
    SignatureSuite, TransferId,
};

struct TestCodec;

impl IntentValidator for TestCodec {
    fn validate_and_decode(
        &self,
        _version: IntentFormatVersion,
        canonical_bytes: &[u8],
    ) -> Result<IntentClaims, ContractError> {
        if canonical_bytes != b"canonical-intent-v1" {
            return Err(ContractError::NonCanonicalIntent);
        }
        Ok(IntentClaims {
            resource_id: ResourceId::from_bytes([1; 16]),
            operation_id: OperationId::from_bytes([2; 16]),
            operation: ResourceOperation::Create,
            resource_kind: ResourceKind::ImmutableBlob,
            expected_predecessor: None,
            unit: "storage.bytes".into(),
            max_charge: 100,
        })
    }
}

fn intent() -> CanonicalResourceIntent {
    let bytes = b"canonical-intent-v1".to_vec();
    CanonicalResourceIntent::validate(
        IntentCandidate {
            contract_version: ContractVersion::V1,
            format_version: IntentFormatVersion::V1,
            digest_suite: DigestSuite::Blake3_256,
            claimed_digest: IntentDigest::compute(DigestSuite::Blake3_256, &bytes),
            canonical_bytes: bytes,
        },
        &TestCodec,
    )
    .unwrap()
}

fn did(value: &str) -> Did {
    Did::from_validated_string(value.into())
}

fn envelope(purpose: KeyPurpose) -> SignatureEnvelope {
    SignatureEnvelope {
        contract_version: ContractVersion::V1,
        suite: SignatureSuite::CompositeEd25519MlDsa65V1,
        signer: did("did:key:signer"),
        key_id: "#resource-v1".into(),
        key_epoch: 3,
        purpose,
        signed_domain: "hs-test-v1".into(),
        signature: Signature::from_envelope_bytes(vec![1]),
    }
}

fn attestations(digest: IntentDigest) -> (MacAttestation, LedgerAttestation) {
    (
        MacAttestation {
            contract_version: ContractVersion::V1,
            intent_digest: digest,
            policy_epoch: 7,
            state_epoch: 9,
            controller: ControllerRef::Identified(did("did:key:controller")),
            capability_binding: hyprstream_resource::CapabilityBinding::IdentifiedRoot(
                Cid::from_validated_bytes(vec![4]),
            ),
            content_label_commitment: Commitment::from_verified_bytes(vec![5]),
            assurance: Assurance::PqHybrid,
            expires_at: 100,
            envelope: envelope(KeyPurpose::MacResourceAttestation),
        },
        LedgerAttestation {
            contract_version: ContractVersion::V1,
            intent_digest: digest,
            ledger_id: did("did:key:ledger"),
            payer: PayerRef::Identified(did("did:key:payer")),
            issuer: did("did:key:issuer"),
            unit: "storage.bytes".into(),
            amount: 42,
            transfer_id: TransferId::from_bytes([6; 16]),
            phase: LedgerPhase::Posted,
            assurance: Assurance::Classical,
            envelope: envelope(KeyPurpose::LedgerResourceAttestation),
        },
    )
}

#[test]
fn assurance_is_clamped_to_the_weakest_leg() {
    assert_eq!(
        Assurance::minimum_all([
            Assurance::PqHybrid,
            Assurance::Classical,
            Assurance::PqHybrid,
        ]),
        Assurance::Classical
    );
    assert_eq!(Assurance::minimum_all([]), Assurance::Unverified);
}

#[test]
fn mismatched_digest_and_bytes_are_rejected() {
    let err = CanonicalResourceIntent::validate(
        IntentCandidate {
            contract_version: ContractVersion::V1,
            format_version: IntentFormatVersion::V1,
            digest_suite: DigestSuite::Blake3_256,
            claimed_digest: IntentDigest::from_bytes([0; 32]),
            canonical_bytes: b"canonical-intent-v1".to_vec(),
        },
        &TestCodec,
    )
    .unwrap_err();
    assert_eq!(err, ContractError::DigestMismatch);
}

#[test]
fn dual_attestation_rejects_crossed_claim_and_unposted_ledger() {
    let intent = intent();
    let (mac, mut ledger) = attestations(intent.digest());
    ledger.intent_digest = IntentDigest::from_bytes([9; 32]);
    assert_eq!(
        DualAttestation::join(&intent, mac, ledger).unwrap_err(),
        ContractError::IntentMismatch
    );

    let (mac, mut ledger) = attestations(intent.digest());
    ledger.phase = LedgerPhase::Reserved;
    assert_eq!(
        DualAttestation::join(&intent, mac, ledger).unwrap_err(),
        ContractError::LedgerNotPosted
    );
}

struct RejectProof;

impl FinalizationProofVerifier for RejectProof {
    fn verify(&self, _statement: &FinalizationStatement) -> Result<(), ContractError> {
        Err(ContractError::InvalidFinalizationProof)
    }
}

#[test]
fn forged_finalization_proof_is_rejected() {
    let intent = intent();
    let (mac, ledger) = attestations(intent.digest());
    let dual = DualAttestation::join(&intent, mac, ledger).unwrap();
    let statement = FinalizationStatement {
        contract_version: ContractVersion::V1,
        operation_id: intent.claims().operation_id,
        resource_id: intent.claims().resource_id,
        resource_version: 1,
        intent_digest: intent.digest(),
        fence: hyprstream_resource::FencingToken {
            resource_id: intent.claims().resource_id,
            registrar_term: 1,
            generation: 1,
        },
        manifest_cid: Cid::from_validated_bytes(vec![1]),
        content_cid: Some(Cid::from_validated_bytes(vec![2])),
        mac_attestation_cid: Cid::from_validated_bytes(vec![3]),
        ledger_attestation_cid: Cid::from_validated_bytes(vec![4]),
        private_evidence_commitment: Commitment::from_verified_bytes(vec![5]),
        envelope: envelope(KeyPurpose::RegistrarFinalization),
    };

    assert_eq!(
        hyprstream_resource::FinalizedResource::verify(&intent, &dual, &statement, &RejectProof)
            .unwrap_err(),
        ContractError::InvalidFinalizationProof
    );
}
