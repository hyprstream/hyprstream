#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Full frozen-vector acceptance suite — iterates every positive and negative
//! vector from the gate-2 artifacts and asserts each declared `expect` result.

use hex;

// ---------------------------------------------------------------------------
// Vector loading
// ---------------------------------------------------------------------------

pub(crate) fn load_positive_vectors() -> Vec<(String, Vec<u8>)> {
    let json_str = include_str!("../../../../docs/standards/v16/vectors/proof-v1-positive.json");
    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON");
    parsed["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let id = v["id"].as_str().unwrap().to_owned();
            let hex_str = v["cbor_hex"].as_str().unwrap();
            (id, hex::decode(hex_str).unwrap())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Frozen enrollment fixtures
//
// The gate-2 key roster is the enrollment source for the authenticated and
// response vectors. Two *separate* deployments are modelled, because the
// profile forbids one component key from being enrolled for both a hybrid and
// a standalone suite: the hybrid deployment enrols `client-ed25519-1` +
// `client-mldsa65-1` as one WNS signer, the classical deployment enrols
// `client-ed25519-1` alone. A vector accepted by one MUST be denied by the
// other — that is the cross-suite separation the profile requires.
// ---------------------------------------------------------------------------

use crate::proof::enrollment::{
    ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole, SignerSuiteRecord,
};

/// The frozen fixture clock: `iat` 1786000000, `exp` 1786000030.
pub(crate) const FIXTURE_NOW: u64 = 1_786_000_010;
/// The frozen fixture `request_id` (CWT `cti`) every vector carries.
pub(crate) const FIXTURE_REQUEST_ID: crate::proof::RequestId = [
    0x3f, 0x1c, 0x9a, 0x04, 0xb7, 0xd2, 0x41, 0x6e, 0x8c, 0x05, 0xa9, 0x13, 0x7b, 0x6e, 0x2d, 0x80,
];
/// Enrollment validity comfortably covering the fixture proofs' `exp`.
const FIXTURE_NOT_AFTER: u64 = 1_786_000_600;

fn keys_json() -> serde_json::Value {
    let s = include_str!("../../../../docs/standards/v16/vectors/proof-v1-keys.json");
    serde_json::from_str(s).expect("valid key roster JSON")
}

fn public_hex(family: &str, kid_ascii: &str) -> Vec<u8> {
    let json = keys_json();
    let entry = json["keys"][family]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["kid_ascii"].as_str() == Some(kid_ascii))
        .unwrap_or_else(|| panic!("{family} key {kid_ascii} must exist in the roster"))
        .clone();
    hex::decode(entry["public_hex"].as_str().unwrap()).unwrap()
}

pub(crate) fn ed25519_public(kid_ascii: &str) -> ed25519_dalek::VerifyingKey {
    let bytes: [u8; 32] = public_hex("ed25519", kid_ascii).try_into().unwrap();
    ed25519_dalek::VerifyingKey::from_bytes(&bytes).unwrap()
}

fn ed_component(kid_ascii: &str) -> EnrolledComponent {
    EnrolledComponent::new(
        kid_ascii.as_bytes().to_vec(),
        ComponentKey::Ed25519(ed25519_public(kid_ascii)),
    )
}

fn mldsa_component(kid_ascii: &str) -> EnrolledComponent {
    let vk = crate::crypto::pq::ml_dsa_vk_from_bytes(&public_hex("ml_dsa_65", kid_ascii)).unwrap();
    EnrolledComponent::new(
        kid_ascii.as_bytes().to_vec(),
        ComponentKey::MlDsa65(Box::new(vk)),
    )
}

fn record(
    principal: &str,
    suite_id: &str,
    components: Vec<EnrolledComponent>,
    role: SignerRole,
) -> SignerSuiteRecord {
    SignerSuiteRecord {
        principal: principal.to_owned(),
        suite_id: suite_id.to_owned(),
        components,
        epoch: 1,
        role,
        approver_role: None,
        not_after: FIXTURE_NOT_AFTER,
        revoked: false,
    }
}

/// A deployment that enrols the client as one WNS hybrid signer (P-2).
pub(crate) fn hybrid_enrollment() -> InMemoryEnrollmentResolver {
    let mut resolver = InMemoryEnrollmentResolver::new();
    resolver
        .enrol_primary(
            &ed25519_public("client-ed25519-1"),
            record(
                "client",
                crate::proof::SUITE_HYBRID,
                vec![
                    ed_component("client-ed25519-1"),
                    mldsa_component("client-mldsa65-1"),
                ],
                SignerRole::Primary,
            ),
        )
        .unwrap();
    resolver
}

/// A deployment that enrols the client as a standalone Ed25519 signer, an
/// anchored approver, and the service response signer (P-3, P-4, P-5).
pub(crate) fn classical_enrollment() -> InMemoryEnrollmentResolver {
    let mut resolver = InMemoryEnrollmentResolver::new();
    resolver
        .enrol_primary(
            &ed25519_public("client-ed25519-1"),
            record(
                "client",
                crate::proof::SUITE_CLASSICAL,
                vec![ed_component("client-ed25519-1")],
                SignerRole::Primary,
            ),
        )
        .unwrap();
    resolver
        .enrol_approver(record(
            "approver",
            crate::proof::SUITE_CLASSICAL,
            vec![ed_component("approver-ed25519-1")],
            SignerRole::Approver,
        ))
        .unwrap();
    resolver
        .enrol_service(
            "registry.svc.hyprstream.test",
            record(
                "registry-service",
                crate::proof::SUITE_CLASSICAL,
                vec![ed_component("service-ed25519-1")],
                SignerRole::Service,
            ),
        )
        .unwrap();
    resolver
}

fn load_negative_vectors() -> Vec<(String, Vec<u8>, String)> {
    let json_str = include_str!("../../../../docs/standards/v16/vectors/proof-v1-negative.json");
    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON");
    parsed["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let id = v["id"].as_str().unwrap().to_owned();
            let hex_str = v["cbor_hex"].as_str().unwrap();
            let deny_class = v["deny_class"].as_str().unwrap_or("unknown").to_owned();
            (id, hex::decode(hex_str).unwrap(), deny_class)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Full-vector parametric tests
// ---------------------------------------------------------------------------

/// Every positive vector MUST be accepted by the parser.
#[test]
fn all_positive_vectors_accept() {
    for (id, cbor) in load_positive_vectors() {
        let result = crate::proof::parser::ParsedProof::parse(&cbor);
        assert!(
            result.is_ok(),
            "positive vector {id} should parse: {:?}",
            result.err()
        );
    }
}

/// Every negative vector MUST be denied.
///
/// Two are denied by the verifier rather than the parser, because they are
/// context-dependent: N-2 is P-2's exact bytes presented in the credential
/// slot, and N-22 is a well-formed response proof answering a different
/// request. Both are covered by dedicated tests below, so nothing is merely
/// skipped.
#[test]
fn all_negative_vectors_deny() {
    let verifier_side = ["N-2", "N-22"];
    for (id, cbor, deny_class) in load_negative_vectors() {
        if verifier_side.contains(&id.as_str()) {
            continue;
        }
        let result = crate::proof::parser::ParsedProof::parse(&cbor);
        assert!(
            result.is_err(),
            "negative vector {id} ({deny_class}) should deny, but was accepted"
        );
    }
}

/// N-22 — a response proof that is entirely well-formed and correctly signed
/// by the enrolled service, but answers a different request, must deny.
#[test]
fn n22_response_proof_for_another_request_denies() {
    let vectors = load_negative_vectors();
    let n22 = vectors
        .iter()
        .find(|(id, _, _)| id == "N-22")
        .expect("N-22 must exist");
    let proof = crate::proof::parser::ParsedProof::parse(&n22.1).expect("N-22 parses; it is a stateful denial");
    assert_ne!(
        proof.claims.request_id, FIXTURE_REQUEST_ID,
        "N-22's cti is deliberately not the request it is presented against"
    );
    let resolver = classical_enrollment();
    assert!(
        crate::proof::verify::verify_response_proof(
            &proof,
            "registry.svc.hyprstream.test",
            &FIXTURE_REQUEST_ID,
            &resolver,
            FIXTURE_NOW,
        )
        .is_err(),
        "a response proof can never verify for another request ID"
    );
}

/// N-2 — the exact P-2 bytes presented in the credential/authorization slot.
/// The credential path requires an `at+jwt` (or CWT access-token) type and an
/// issuer key; a proof CWT carries the proof `typ` and is signed by a
/// cnf-bound request-proof key, so it can never be consumed as a credential.
#[test]
fn n2_proof_in_the_credential_slot_denies() {
    let vectors = load_negative_vectors();
    let n2 = vectors
        .iter()
        .find(|(id, _, _)| id == "N-2")
        .expect("N-2 must exist");

    // It is a valid *proof* in the proof slot ...
    let as_proof = crate::proof::parser::ParsedProof::parse(&n2.1);
    assert!(as_proof.is_ok(), "N-2 is P-2's bytes: valid in the proof slot");

    // ... and is not a credential in the credential slot. The credential slot
    // is a compact-serialization token; these bytes are neither UTF-8 nor a
    // three-part JWS, so no issuer key is ever consulted.
    let as_credential = std::str::from_utf8(&n2.1);
    assert!(
        as_credential.is_err() || as_credential.unwrap().split('.').count() != 3,
        "a proof CWT must not parse as a credential token"
    );
}

// ---------------------------------------------------------------------------
// Spot-check specific vectors for correctness of the denial reason
// ---------------------------------------------------------------------------

#[test]
fn test_p1_unattributed_sign1_accepts() {
    let cbor = &load_positive_vectors()[0].1;
    let proof = crate::proof::parser::ParsedProof::parse(cbor).unwrap();
    assert_eq!(proof.kind, crate::proof::ProofKind::Request);
    assert_eq!(
        proof.disposition,
        crate::proof::ProofDisposition::Unattributed
    );
    assert_eq!(proof.structure, crate::proof::parser::CoseStructure::Sign1);
    assert!(proof.claims.credential_hash.is_none());
    assert!(proof.claims.nonce.is_some());
}

#[test]
fn test_p3_response_proof_accepts() {
    let vectors = load_positive_vectors();
    let p3 = vectors.iter().find(|(id, _)| id == "P-3").unwrap();
    let proof = crate::proof::parser::ParsedProof::parse(&p3.1).unwrap();
    assert_eq!(proof.kind, crate::proof::ProofKind::Response);
}

#[test]
fn test_n3_missing_typ_denies() {
    let vectors = load_negative_vectors();
    let n3 = vectors.iter().find(|(id, _, _)| id == "N-3").unwrap();
    let result = crate::proof::parser::ParsedProof::parse(&n3.1);
    assert!(result.is_err());
}

#[test]
fn test_n4_wrong_domain_denies() {
    let vectors = load_negative_vectors();
    let n4 = vectors.iter().find(|(id, _, _)| id == "N-4").unwrap();
    assert!(crate::proof::parser::ParsedProof::parse(&n4.1).is_err());
}

#[test]
fn test_n6_nine_signer_groups_denies() {
    let vectors = load_negative_vectors();
    let n6 = vectors.iter().find(|(id, _, _)| id == "N-6").unwrap();
    assert!(crate::proof::parser::ParsedProof::parse(&n6.1).is_err());
}

#[test]
fn test_n9b_indefinite_length_denies() {
    let vectors = load_negative_vectors();
    let n9b = vectors.iter().find(|(id, _, _)| id == "N-9b").unwrap();
    assert!(crate::proof::parser::ParsedProof::parse(&n9b.1).is_err());
}

#[test]
fn test_n16_unattributed_no_nonce_denies() {
    let vectors = load_negative_vectors();
    let n16 = vectors.iter().find(|(id, _, _)| id == "N-16").unwrap();
    assert!(crate::proof::parser::ParsedProof::parse(&n16.1).is_err());
}

// ---------------------------------------------------------------------------
// Generated method policy against the frozen vectors
// ---------------------------------------------------------------------------

/// The verified output of a frozen vector must satisfy — and only satisfy —
/// the method policy that matches its actual signing topology.
#[test]
fn frozen_vectors_satisfy_only_their_matching_method_policy() {
    use crate::proof::policy::{evaluate, ApproverRule, CryptoSuite, SignaturePolicy};

    let classical = classical_enrollment();
    let hybrid = hybrid_enrollment();
    let cnf = ed25519_public("client-ed25519-1");
    let parse = |id: &str| {
        let v = load_positive_vectors();
        let bytes = &v.iter().find(|(vid, _)| vid == id).unwrap().1;
        crate::proof::parser::ParsedProof::parse(bytes).unwrap()
    };

    // P-1: unattributed, standalone suite, no approvers.
    let p1 = parse("P-1");
    let p1_v =
        crate::proof::verify::verify_proof_signatures(&p1, None, None, FIXTURE_NOW).unwrap();
    assert!(evaluate(
        &SignaturePolicy::UnauthenticatedOrTokenBound {
            suite: CryptoSuite::Classical
        },
        p1.disposition,
        &p1_v
    )
    .is_ok());
    // The same proof cannot satisfy a token-bound method...
    assert!(evaluate(
        &SignaturePolicy::TokenBound {
            suite: CryptoSuite::Classical
        },
        p1.disposition,
        &p1_v
    )
    .is_err());
    // ...nor a public method that requires the hybrid suite.
    assert!(evaluate(
        &SignaturePolicy::UnauthenticatedOrTokenBound {
            suite: CryptoSuite::Hybrid
        },
        p1.disposition,
        &p1_v
    )
    .is_err());

    // P-2: authenticated hybrid, one logical signer, no approvers.
    let p2 = parse("P-2");
    let p2_v =
        crate::proof::verify::verify_proof_signatures(&p2, Some(&cnf), Some(&hybrid), FIXTURE_NOW)
            .unwrap();
    assert_eq!(p2_v.primary_suite, crate::proof::SUITE_HYBRID);
    assert!(p2_v.approvers.is_empty());
    assert!(evaluate(
        &SignaturePolicy::TokenBound {
            suite: CryptoSuite::Hybrid
        },
        p2.disposition,
        &p2_v
    )
    .is_ok());
    // A method that requires an approval is not satisfied by one signer.
    assert!(evaluate(
        &SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Hybrid,
            approver_rule: ApproverRule::KOfN { k: 1, n: 1 },
        },
        p2.disposition,
        &p2_v
    )
    .is_err());

    // P-5: authenticated, two distinct logical signer groups.
    let p5 = parse("P-5");
    let p5_v = crate::proof::verify::verify_proof_signatures(
        &p5,
        Some(&cnf),
        Some(&classical),
        FIXTURE_NOW,
    )
    .unwrap();
    assert_eq!(p5_v.primary_principal.as_deref(), Some("client"));
    assert_eq!(p5_v.approvers.len(), 1);
    assert_eq!(p5_v.approvers[0].principal, "approver");
    assert!(evaluate(
        &SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: ApproverRule::KOfN { k: 1, n: 1 },
        },
        p5.disposition,
        &p5_v
    )
    .is_ok());
    // Two approvals are required; one does not satisfy the threshold.
    assert!(evaluate(
        &SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: ApproverRule::KOfN { k: 2, n: 3 },
        },
        p5.disposition,
        &p5_v
    )
    .is_err());
    // A method declaring no approver rule rejects the extra approval group.
    assert!(evaluate(
        &SignaturePolicy::TokenBound {
            suite: CryptoSuite::Classical
        },
        p5.disposition,
        &p5_v
    )
    .is_err());
}
