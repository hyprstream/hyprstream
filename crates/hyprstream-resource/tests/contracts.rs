//! Contract tests: every load-bearing validating constructor has a negative
//! control. These fixtures deliberately use distinct MAC and ledger roots.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_resource::{
    AcceptedAuthority, Assurance, CanonicalResourceIntent, Cid, ContentLabelCommitment,
    ContractError, ContractVersion, ControllerRef, Did, DigestSuite, DualAttestation,
    FinalizationProofVerifier, FinalizationStatement, IntentCandidate, IntentClaims, IntentDigest,
    IntentFormatVersion, IntentValidator, KeyPurpose, LedgerAttestationCandidate,
    LedgerAttestationVerifier, LedgerPhase, MacAttestationCandidate, MacAttestationVerifier,
    OperationId, OwnerCommitment, PayerRef, PrivateEvidenceCommitment, PublicEvidenceCommitment,
    RegistrarState, RegistrationSnapshot, RejectSharedAttester, RequiredAssurance, Resolution,
    ResolutionRef, ResolutionTarget, ResourceId, ResourceKind, ResourceOperation, Signature,
    SignatureEnvelope, SignatureSuite, TransferId, VerifiedEvidence,
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
            owner: hyprstream_resource::OwnershipRef::Committed(
                OwnerCommitment::from_verified_bytes(vec![1]),
            ),
            controller: ControllerRef::Identified(did("did:key:controller")),
            payer: PayerRef::Identified(did("did:key:payer")),
            required_assurance: RequiredAssurance::Classical,
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

fn envelope(purpose: KeyPurpose, signer: &str, key_id: &str) -> SignatureEnvelope {
    SignatureEnvelope {
        contract_version: ContractVersion::V1,
        suite: SignatureSuite::CompositeEd25519MlDsa65V1,
        signer: did(signer),
        key_id: key_id.into(),
        key_epoch: 3,
        purpose,
        signed_domain: "hs-test-v1".into(),
        signature: Signature::from_envelope_bytes(vec![1]),
    }
}

fn attestations(digest: IntentDigest) -> (MacAttestationCandidate, LedgerAttestationCandidate) {
    (
        MacAttestationCandidate {
            contract_version: ContractVersion::V1,
            intent_digest: digest,
            policy_epoch: 7,
            state_epoch: 9,
            controller: ControllerRef::Identified(did("did:key:controller")),
            capability_binding: hyprstream_resource::CapabilityBinding::IdentifiedRoot(
                Cid::from_validated_bytes(vec![4]),
            ),
            content_label_commitment: ContentLabelCommitment::from_verified_bytes(vec![5]),
            expires_at: 100,
            envelope: envelope(KeyPurpose::MacResourceAttestation, "did:key:mac", "#mac-v1"),
        },
        LedgerAttestationCandidate {
            contract_version: ContractVersion::V1,
            intent_digest: digest,
            ledger_id: did("did:key:ledger"),
            payer: PayerRef::Identified(did("did:key:payer")),
            issuer: did("did:key:issuer"),
            unit: "storage.bytes".into(),
            amount: 42,
            transfer_id: TransferId::from_bytes([6; 16]),
            phase: LedgerPhase::Posted,
            envelope: envelope(
                KeyPurpose::LedgerResourceAttestation,
                "did:key:ledger-signer",
                "#ledger-v1",
            ),
        },
    )
}

struct TestMacVerifier;
struct TestLedgerVerifier;
struct MismatchedMacVerifier;
struct UnverifiedMacVerifier;
struct UnverifiedLedgerVerifier;

fn verified(
    candidate: &SignatureEnvelope,
    cid: u8,
    root: u8,
    assurance: Assurance,
) -> VerifiedEvidence {
    VerifiedEvidence {
        attestation_cid: Cid::from_validated_bytes(vec![cid]),
        authority: AcceptedAuthority {
            root: Cid::from_validated_bytes(vec![root]),
            signer: candidate.signer.clone(),
            key_id: candidate.key_id.clone(),
            key_epoch: candidate.key_epoch,
        },
        assurance,
    }
}

impl MacAttestationVerifier for TestMacVerifier {
    fn verify_accepted_current(
        &self,
        _intent: &CanonicalResourceIntent,
        candidate: &MacAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError> {
        Ok(verified(&candidate.envelope, 30, 31, Assurance::PqHybrid))
    }
}

impl MacAttestationVerifier for MismatchedMacVerifier {
    fn verify_accepted_current(
        &self,
        _intent: &CanonicalResourceIntent,
        candidate: &MacAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError> {
        let mut evidence = verified(&candidate.envelope, 30, 31, Assurance::PqHybrid);
        evidence.authority.signer = did("did:key:not-the-envelope-signer");
        Ok(evidence)
    }
}

impl LedgerAttestationVerifier for TestLedgerVerifier {
    fn verify_accepted_current(
        &self,
        _intent: &CanonicalResourceIntent,
        candidate: &LedgerAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError> {
        Ok(verified(&candidate.envelope, 40, 41, Assurance::Classical))
    }
}

impl MacAttestationVerifier for UnverifiedMacVerifier {
    fn verify_accepted_current(
        &self,
        _intent: &CanonicalResourceIntent,
        candidate: &MacAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError> {
        Ok(verified(&candidate.envelope, 30, 31, Assurance::Unverified))
    }
}

impl LedgerAttestationVerifier for UnverifiedLedgerVerifier {
    fn verify_accepted_current(
        &self,
        _intent: &CanonicalResourceIntent,
        candidate: &LedgerAttestationCandidate,
    ) -> Result<VerifiedEvidence, ContractError> {
        Ok(verified(&candidate.envelope, 40, 41, Assurance::Unverified))
    }
}

fn dual(intent: &CanonicalResourceIntent) -> DualAttestation {
    let (mac, ledger) = attestations(intent.digest());
    DualAttestation::verify_and_join(
        intent,
        mac,
        ledger,
        &TestMacVerifier,
        &TestLedgerVerifier,
        &RejectSharedAttester,
    )
    .unwrap()
}

#[test]
fn assurance_is_clamped_to_the_weakest_leg() {
    assert_eq!(
        Assurance::minimum_all([Assurance::PqHybrid, Assurance::Classical]),
        Assurance::Classical
    );
    assert_eq!(Assurance::minimum_all([]), Assurance::Unverified);
}

#[test]
fn canonical_intent_rejects_mismatched_digest_and_missing_predecessor() {
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

    struct MissingPredecessor;
    impl IntentValidator for MissingPredecessor {
        fn validate_and_decode(
            &self,
            _: IntentFormatVersion,
            _: &[u8],
        ) -> Result<IntentClaims, ContractError> {
            let mut claims =
                TestCodec.validate_and_decode(IntentFormatVersion::V1, b"canonical-intent-v1")?;
            claims.operation = ResourceOperation::Mutate;
            Ok(claims)
        }
    }
    let bytes = b"canonical-intent-v1".to_vec();
    assert_eq!(
        CanonicalResourceIntent::validate(
            IntentCandidate {
                contract_version: ContractVersion::V1,
                format_version: IntentFormatVersion::V1,
                digest_suite: DigestSuite::Blake3_256,
                claimed_digest: IntentDigest::compute(DigestSuite::Blake3_256, &bytes),
                canonical_bytes: bytes
            },
            &MissingPredecessor,
        )
        .unwrap_err(),
        ContractError::InvalidPredecessor,
    );

    struct ImmutableMutation;
    impl IntentValidator for ImmutableMutation {
        fn validate_and_decode(
            &self,
            _: IntentFormatVersion,
            _: &[u8],
        ) -> Result<IntentClaims, ContractError> {
            let mut claims =
                TestCodec.validate_and_decode(IntentFormatVersion::V1, b"canonical-intent-v1")?;
            claims.operation = ResourceOperation::Mutate;
            claims.expected_predecessor = Some(hyprstream_resource::PredecessorRef {
                manifest_cid: Cid::from_validated_bytes(vec![1]),
                version: 1,
            });
            Ok(claims)
        }
    }
    let bytes = b"immutable-mutation-v1".to_vec();
    assert_eq!(
        CanonicalResourceIntent::validate(
            IntentCandidate {
                contract_version: ContractVersion::V1,
                format_version: IntentFormatVersion::V1,
                digest_suite: DigestSuite::Blake3_256,
                claimed_digest: IntentDigest::compute(DigestSuite::Blake3_256, &bytes),
                canonical_bytes: bytes,
            },
            &ImmutableMutation,
        )
        .unwrap_err(),
        ContractError::InvalidResourceOperation,
    );
}

#[test]
fn verified_dual_rejects_crossed_claims_unverified_evidence_and_shared_authority() {
    let intent = intent();
    let (mac, mut ledger) = attestations(intent.digest());
    ledger.phase = LedgerPhase::Reserved;
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester
        )
        .unwrap_err(),
        ContractError::LedgerNotPosted
    );

    let (mac, mut ledger) = attestations(intent.digest());
    ledger.amount = 101;
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester
        )
        .unwrap_err(),
        ContractError::ChargeMismatch
    );

    let (mut mac, ledger) = attestations(intent.digest());
    mac.controller = ControllerRef::Identified(did("did:key:substituted-controller"));
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester,
        )
        .unwrap_err(),
        ContractError::IntentMismatch
    );

    let (mac, mut ledger) = attestations(intent.digest());
    ledger.payer = PayerRef::Identified(did("did:key:substituted-payer"));
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester,
        )
        .unwrap_err(),
        ContractError::IntentMismatch,
    );

    let (mac, mut ledger) = attestations(intent.digest());
    ledger.unit = "network.egress".into();
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester,
        )
        .unwrap_err(),
        ContractError::ChargeMismatch,
    );

    let (mut mac, ledger) = attestations(intent.digest());
    mac.capability_binding = hyprstream_resource::CapabilityBinding::AnonymousOperationCommitment(
        hyprstream_resource::OperationCapabilityCommitment::from_verified_bytes(
            IntentDigest::from_bytes([99; 32]),
            vec![4],
        ),
    );
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester,
        )
        .unwrap_err(),
        ContractError::IntentMismatch,
    );

    let (mut mac, ledger) = attestations(intent.digest());
    mac.envelope.purpose = KeyPurpose::RegistrarFinalization;
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester
        )
        .unwrap_err(),
        ContractError::WrongKeyPurpose
    );

    let (mac, mut ledger) = attestations(intent.digest());
    ledger.envelope.signer = mac.envelope.signer.clone();
    ledger.envelope.key_id = mac.envelope.key_id.clone();
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester
        )
        .unwrap_err(),
        ContractError::SharedAttester
    );

    let (mac, ledger) = attestations(intent.digest());
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &MismatchedMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester,
        )
        .unwrap_err(),
        ContractError::InvalidAttestationEvidence
    );

    let (mac, ledger) = attestations(intent.digest());
    assert_eq!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &UnverifiedMacVerifier,
            &UnverifiedLedgerVerifier,
            &RejectSharedAttester,
        )
        .unwrap_err(),
        ContractError::AssuranceRequirementNotMet,
    );

    let (mac, mut ledger) = attestations(intent.digest());
    ledger.envelope.key_id = mac.envelope.key_id.clone();
    assert!(
        DualAttestation::verify_and_join(
            &intent,
            mac,
            ledger,
            &TestMacVerifier,
            &TestLedgerVerifier,
            &RejectSharedAttester,
        )
        .is_ok(),
        "distinct accepted roots and signer DIDs remain independent when local key fragments match",
    );
}

struct AcceptProof;
impl FinalizationProofVerifier for AcceptProof {
    fn verify(&self, _: &FinalizationStatement, _: &DualAttestation) -> Result<(), ContractError> {
        Ok(())
    }
}

fn statement(intent: &CanonicalResourceIntent, dual: &DualAttestation) -> FinalizationStatement {
    FinalizationStatement {
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
        mac_attestation_cid: dual.mac().attestation_cid().clone(),
        ledger_attestation_cid: dual.ledger().attestation_cid().clone(),
        private_evidence_commitment: PrivateEvidenceCommitment::from_verified_bytes(vec![5]),
        envelope: envelope(
            KeyPurpose::RegistrarFinalization,
            "did:key:registrar",
            "#registrar-v1",
        ),
    }
}

fn wrong_statement_digest(statement: &mut FinalizationStatement) {
    statement.intent_digest = IntentDigest::from_bytes([99; 32]);
}

fn wrong_statement_operation(statement: &mut FinalizationStatement) {
    statement.operation_id = OperationId::from_bytes([99; 16]);
}

fn wrong_statement_resource(statement: &mut FinalizationStatement) {
    statement.resource_id = ResourceId::from_bytes([99; 16]);
}

fn wrong_statement_fence_resource(statement: &mut FinalizationStatement) {
    statement.fence.resource_id = ResourceId::from_bytes([99; 16]);
}

#[test]
fn finalization_rejects_each_cross_claim_mutation_and_missing_required_content() {
    let intent = intent();
    let dual = dual(&intent);
    let statement = statement(&intent, &dual);
    let finalized =
        hyprstream_resource::FinalizedResource::verify(&intent, &dual, &statement, &AcceptProof)
            .unwrap();
    assert_eq!(
        finalized.mac_attestation_cid(),
        dual.mac().attestation_cid()
    );

    let mutations: [fn(&mut FinalizationStatement); 4] = [
        wrong_statement_digest,
        wrong_statement_operation,
        wrong_statement_resource,
        wrong_statement_fence_resource,
    ];
    for mutate in mutations {
        let mut crossed = statement.clone();
        mutate(&mut crossed);
        assert_eq!(
            hyprstream_resource::FinalizedResource::verify(&intent, &dual, &crossed, &AcceptProof)
                .unwrap_err(),
            ContractError::IntentMismatch,
        );
    }

    let mut provenance = statement.clone();
    provenance.mac_attestation_cid = Cid::from_validated_bytes(vec![99]);
    assert_eq!(
        hyprstream_resource::FinalizedResource::verify(&intent, &dual, &provenance, &AcceptProof)
            .unwrap_err(),
        ContractError::AttestationProvenanceMismatch
    );

    let mut wrong_purpose = statement.clone();
    wrong_purpose.envelope.purpose = KeyPurpose::MacResourceAttestation;
    assert_eq!(
        hyprstream_resource::FinalizedResource::verify(
            &intent,
            &dual,
            &wrong_purpose,
            &AcceptProof
        )
        .unwrap_err(),
        ContractError::WrongKeyPurpose
    );

    let mut no_content = statement;
    no_content.content_cid = None;
    assert_eq!(
        hyprstream_resource::FinalizedResource::verify(&intent, &dual, &no_content, &AcceptProof)
            .unwrap_err(),
        ContractError::ContentRequired
    );
}

struct DirectoryCodec;

impl IntentValidator for DirectoryCodec {
    fn validate_and_decode(
        &self,
        _: IntentFormatVersion,
        _: &[u8],
    ) -> Result<IntentClaims, ContractError> {
        let mut claims =
            TestCodec.validate_and_decode(IntentFormatVersion::V1, b"canonical-intent-v1")?;
        claims.resource_kind = ResourceKind::Directory;
        Ok(claims)
    }
}

#[test]
fn finalization_rejects_content_forbidden_by_the_kind_operation_matrix() {
    let bytes = b"directory-create-v1".to_vec();
    let intent = CanonicalResourceIntent::validate(
        IntentCandidate {
            contract_version: ContractVersion::V1,
            format_version: IntentFormatVersion::V1,
            digest_suite: DigestSuite::Blake3_256,
            claimed_digest: IntentDigest::compute(DigestSuite::Blake3_256, &bytes),
            canonical_bytes: bytes,
        },
        &DirectoryCodec,
    )
    .unwrap();
    let dual = dual(&intent);
    let mut with_content = statement(&intent, &dual);
    assert_eq!(
        hyprstream_resource::FinalizedResource::verify(
            &intent,
            &dual,
            &with_content,
            &AcceptProof,
        )
        .unwrap_err(),
        ContractError::ContentForbidden,
    );
    with_content.content_cid = None;
    assert!(hyprstream_resource::FinalizedResource::verify(
        &intent,
        &dual,
        &with_content,
        &AcceptProof,
    )
    .is_ok(),);
}

#[test]
fn public_projection_is_minted_from_finalization_and_intent_scoped() {
    let intent = intent();
    let dual = dual(&intent);
    let finalized = hyprstream_resource::FinalizedResource::verify(
        &intent,
        &dual,
        &statement(&intent, &dual),
        &AcceptProof,
    )
    .unwrap();
    let projection = finalized
        .public_projection(PublicEvidenceCommitment::from_verified_bytes(
            intent.digest(),
            vec![7],
        ))
        .unwrap();
    finalized.publish_effect(None, projection).unwrap();
    assert_eq!(
        finalized
            .public_projection(PublicEvidenceCommitment::from_verified_bytes(
                IntentDigest::from_bytes([8; 32]),
                vec![7]
            ))
            .unwrap_err(),
        ContractError::IntentMismatch
    );
}

#[test]
fn registration_snapshot_binds_every_resolution_to_its_operation_and_requires_quarantine_reason() {
    let operation_id = OperationId::from_bytes([2; 16]);
    let fence = hyprstream_resource::FencingToken {
        resource_id: ResourceId::from_bytes([1; 16]),
        registrar_term: 1,
        generation: 1,
    };
    let target = ResolutionTarget {
        operation_id,
        resource_id: fence.resource_id,
        fence,
    };
    let resolution = Some(ResolutionRef {
        record_cid: Cid::from_validated_bytes(vec![8]),
        outcome: Resolution::Voided {
            target,
            void_transfer_id: TransferId::from_bytes([9; 16]),
        },
    });
    RegistrationSnapshot::new(
        operation_id,
        IntentDigest::from_bytes([7; 32]),
        RegistrarState::Voided,
        fence,
        None,
        resolution,
    )
    .unwrap();

    let wrong = ResolutionTarget {
        operation_id: OperationId::from_bytes([3; 16]),
        ..target
    };
    assert_eq!(
        RegistrationSnapshot::new(
            operation_id,
            IntentDigest::from_bytes([7; 32]),
            RegistrarState::Voided,
            fence,
            None,
            Some(ResolutionRef {
                record_cid: Cid::from_validated_bytes(vec![8]),
                outcome: Resolution::Voided {
                    target: wrong,
                    void_transfer_id: TransferId::from_bytes([9; 16])
                }
            })
        )
        .unwrap_err(),
        ContractError::InconsistentResolution
    );

    let wrong_resource = ResolutionTarget {
        resource_id: ResourceId::from_bytes([3; 16]),
        ..target
    };
    assert_eq!(
        RegistrationSnapshot::new(
            operation_id,
            IntentDigest::from_bytes([7; 32]),
            RegistrarState::Voided,
            fence,
            None,
            Some(ResolutionRef {
                record_cid: Cid::from_validated_bytes(vec![8]),
                outcome: Resolution::Voided {
                    target: wrong_resource,
                    void_transfer_id: TransferId::from_bytes([9; 16]),
                },
            }),
        )
        .unwrap_err(),
        ContractError::InconsistentResolution,
    );

    let wrong_fence = ResolutionTarget {
        fence: hyprstream_resource::FencingToken {
            generation: 2,
            ..fence
        },
        ..target
    };
    assert_eq!(
        RegistrationSnapshot::new(
            operation_id,
            IntentDigest::from_bytes([7; 32]),
            RegistrarState::Voided,
            fence,
            None,
            Some(ResolutionRef {
                record_cid: Cid::from_validated_bytes(vec![8]),
                outcome: Resolution::Voided {
                    target: wrong_fence,
                    void_transfer_id: TransferId::from_bytes([9; 16]),
                },
            }),
        )
        .unwrap_err(),
        ContractError::InconsistentResolution,
    );

    assert_eq!(
        RegistrationSnapshot::new(
            operation_id,
            IntentDigest::from_bytes([7; 32]),
            RegistrarState::Quarantined,
            fence,
            None,
            None
        )
        .unwrap_err(),
        ContractError::InconsistentResolution
    );
}
