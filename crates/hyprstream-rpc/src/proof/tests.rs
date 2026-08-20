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
            enrollment_policy_id: "test-enrollment-v1".to_owned(),
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
    let mut approver = record(
        "approver",
        crate::proof::SUITE_CLASSICAL,
        vec![ed_component("approver-ed25519-1")],
        SignerRole::Approver,
    );
    // The enrolled approver role is enrollment data; a generated approver rule
    // names it per group, and a group holding a different role denies.
    approver.approver_role = Some("security".to_owned());
    resolver.enrol_approver(approver).unwrap();
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

/// Positive vectors that this C lane's amended parser intentionally no longer
/// accepts as frozen, pending WS-A re-issuing the fixture at the amended wire
/// value.
///
/// P-4 encodes the **pre-amendment** three-field `response_binding`
/// (`{1: schema, 2: combined-mode, 3: kem}`). Gate-2 amendments 3+4 replace it
/// with the orthogonal four-field map
/// (`{1: root_type_id, 2: response_kind, 3: protection_mode, 4: kem}`), which
/// this lane now enforces. The four-field decode correctly rejects the old
/// three-field shape, so P-4-as-frozen no longer parses. WS-A owns
/// `docs/standards/v16/vectors/` and re-issues the deterministic fixture at the
/// amended value; C must not hand-edit A's fixtures. Until that handoff lands,
/// P-4 is a tracked residual blocker, and the amended decode is proven instead
/// by the inline unit tests in `proof::response` (four-field accept, three-field
/// reject, exact −70200 enforcement).
const A_REISSUE_PENDING_POSITIVE_VECTORS: &[&str] = &["P-4"];

/// Every positive vector MUST be accepted by the parser, except those awaiting
/// a WS-A fixture re-issue at an amended Gate-2 wire value (see the constant).
#[test]
fn all_positive_vectors_accept() {
    for (id, cbor) in load_positive_vectors() {
        if A_REISSUE_PENDING_POSITIVE_VECTORS.contains(&id.as_str()) {
            // Assert the reason is exactly the amended-binding mismatch, so this
            // skip cannot silently mask an unrelated regression in P-4.
            let result = crate::proof::parser::ParsedProof::parse(&cbor);
            assert!(
                result.is_err(),
                "{id} is expected to fail against the amended four-field response_binding \
                 until WS-A re-issues it; if it now parses, the fixture was re-issued and this \
                 skip should be removed"
            );
            continue;
        }
        let result = crate::proof::parser::ParsedProof::parse(&cbor);
        assert!(
            result.is_ok(),
            "positive vector {id} should parse: {:?}",
            result.err()
        );
    }
}

/// The vocabulary a parser denial for `deny_class` must use.
///
/// F-E: `is_err()` alone lets a vector deny for an *unintended* structural
/// reason and still pass. Each frozen negative vector declares the rule it is
/// meant to trip; this binds that declared `deny_class` to the parser's own
/// denial vocabulary, so a vector that denies for the wrong reason fails the
/// test. The sets are the literal substrings the parser/claims/plan/cbor-audit
/// code emits for each class (an OR: the error chain must contain at least
/// one). An empty set means the reason is not bound (none here).
fn deny_reason_tokens(deny_class: &str) -> &'static [&'static str] {
    match deny_class {
        "type-confusion" => &["typ/hs_domain mismatch", "hs_domain: missing", "typ: missing"],
        "missing-typ" => &["typ: missing", "missing (label 16)"],
        "domain-separation" => &["typ/hs_domain mismatch", "hs_domain"],
        "component-stripping" => &[
            "signatures but plan expects",
            "plan components matched",
            "not in plan",
        ],
        "parser-cap" => &[
            "exceeds cap",
            "exceeds",
            "at most",
            "must be 1..",
            "must have at most",
        ],
        "closed-claim-set" => &["unknown claim key", "missing"],
        "non-deterministic-encoding" => &["deterministic CBOR"],
        // A crit label declared but absent from its bucket is denied either by
        // the crit-bucket check or, when the absent label is a header the parser
        // reads first (e.g. hs_domain −70100), by that header's own
        // "missing (label …)" — the same violation surfacing earlier.
        "crit-set" => &["crit", "missing (label"],
        "disposition-confusion" => &[
            "credential_hash",
            "response proof must not carry",
            "unattributed",
        ],
        "algorithm" => &["not in profile"],
        "credential-binding" => &["credential_hash", "credential hash"],
        "freshness" => &["server challenge", "Nonce", "challenge"],
        "unprotected-authority" => &["unprotected"],
        "key-set-strictness" => &["key set", "key-set"],
        "plan-mismatch" => &["not in plan", "plan expects", "plan group"],
        _ => &[],
    }
}

/// Every negative vector MUST be denied, and — F-E — for the declared reason.
///
/// Two are denied by the verifier rather than the parser, because they are
/// context-dependent: N-2 is P-2's exact bytes presented in the credential
/// slot, and N-22 is a well-formed response proof answering a different
/// request. Both are covered by dedicated tests below, so nothing is merely
/// skipped.
#[test]
fn all_negative_vectors_deny() {
    let verifier_side = ["N-2", "N-22"];
    // Negative vectors whose *denial reason* (not the denial itself) is masked
    // by the pre-amendment three-field `response_binding` they still carry: the
    // amended claims decode rejects that binding before reaching the vector's
    // intended (later) check. They still deny fail-closed — only the reason
    // binding is deferred until WS-A re-issues them at the four-field value.
    // Tracked as a residual blocker alongside P-4 (see status-mac-v16-c.md).
    let reason_pending_a_reissue = ["N-10f"];
    for (id, cbor, deny_class) in load_negative_vectors() {
        if verifier_side.contains(&id.as_str()) {
            continue;
        }
        let result = crate::proof::parser::ParsedProof::parse(&cbor);
        let err = match result {
            Ok(_) => panic!("negative vector {id} ({deny_class}) should deny, but was accepted"),
            Err(e) => format!("{e:#}"),
        };
        if reason_pending_a_reissue.contains(&id.as_str()) {
            // Still must deny; the reason binding is deferred (see above).
            continue;
        }
        let tokens = deny_reason_tokens(&deny_class);
        if !tokens.is_empty() {
            assert!(
                tokens.iter().any(|t| err.contains(t)),
                "negative vector {id} denied, but the reason does not match its declared \
                 deny_class '{deny_class}'. Expected one of {tokens:?}; got: {err}"
            );
        }
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

/// N-2 at the request-envelope credential slot: the frozen bytes presented in
/// the authorization slot of a real `RequestEnvelope` must be refused when the
/// envelope is decoded, on every transport that carries it.
///
/// Both presentations are covered: the raw COSE object (the shape a CWT
/// credential slot accepts) and a compact-serialization token whose header
/// `typ` names a proof media type. A genuine credential in the same slot is
/// unaffected.
#[test]
fn n2_in_the_request_envelope_credential_slot_denies() {
    use crate::envelope::{Authorization, RequestEnvelope};
    use crate::{FromCapnp, ToCapnp};
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

    let vectors = load_negative_vectors();
    let n2 = &vectors
        .iter()
        .find(|(id, _, _)| id == "N-2")
        .expect("N-2 must exist")
        .1;

    let roundtrip = |auth: Authorization| -> anyhow::Result<Authorization> {
        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3]).with_authorization(auth);
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder =
                message.init_root::<crate::common_capnp::request_envelope::Builder>();
            envelope.write_to(&mut builder);
        }
        let mut bytes = Vec::new();
        capnp::serialize::write_message(&mut bytes, &message)?;
        let reader = capnp::serialize::read_message(
            &mut &bytes[..],
            capnp::message::ReaderOptions::new(),
        )?;
        let decoded = RequestEnvelope::read_from(
            reader.get_root::<crate::common_capnp::request_envelope::Reader>()?,
        )?;
        Ok(decoded.authorization)
    };

    // 1. The raw COSE proof object. Two independent facts hold, and both are
    //    required: the byte-level rule refuses these exact bytes wherever a
    //    credential slot accepts bytes (the CWT credential path), and this
    //    envelope's Text slot cannot carry them intact in the first place.
    assert!(
        crate::proof::is_proof_typed_credential(n2),
        "the credential-slot guard must refuse the exact proof bytes"
    );
    let through_text_slot = String::from_utf8_lossy(n2).into_owned();
    assert_ne!(
        through_text_slot.as_bytes(),
        n2.as_slice(),
        "a Text credential slot cannot deliver a proof CWT intact"
    );

    // 2. The same proof wrapped as a compact token carrying the proof typ.
    for proof_typ in [crate::proof::PROOF_TYP, crate::proof::RESPONSE_PROOF_TYP] {
        let header = format!(r#"{{"alg":"EdDSA","typ":"{proof_typ}","kid":"k1"}}"#);
        let token = format!(
            "{}.{}.{}",
            URL_SAFE_NO_PAD.encode(header),
            URL_SAFE_NO_PAD.encode(n2),
            URL_SAFE_NO_PAD.encode([0u8; 64])
        );
        assert!(
            roundtrip(Authorization::IdJag(token)).is_err(),
            "a {proof_typ}-typed token in the credential slot must be refused"
        );
    }

    // 3. A credential-shaped token in the same slot still decodes: the gate
    //    rejects proof typing, not the slot itself.
    let credential_header = r#"{"alg":"EdDSA","typ":"at+jwt","kid":"k1"}"#;
    let credential = format!(
        "{}.{}.{}",
        URL_SAFE_NO_PAD.encode(credential_header),
        URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#),
        URL_SAFE_NO_PAD.encode([0u8; 64])
    );
    assert!(
        roundtrip(Authorization::IdJag(credential)).is_ok(),
        "an ordinary credential must still be accepted"
    );
}

/// N-2 — the exact P-2 bytes presented in the credential/authorization slot.
///
/// The real credential-slot parser must reject them. Two independent rules do
/// so, and both are exercised here against the frozen vector rather than
/// asserted in prose:
///
/// 1. **Encoding.** The credential slot carries a compact-serialization token.
///    The proof is binary CBOR, so the protected-header parse fails outright.
/// 2. **Type.** Even shaped as a compact token, a header whose `typ` is a
///    proof media type is not in any credential slot's closed allowed-type
///    list, so no issuer key is ever consulted.
#[test]
fn n2_proof_in_the_credential_slot_denies() {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

    let vectors = load_negative_vectors();
    let n2 = vectors
        .iter()
        .find(|(id, _, _)| id == "N-2")
        .expect("N-2 must exist");

    // It is a valid *proof* in the proof slot ...
    assert!(
        crate::proof::parser::ParsedProof::parse(&n2.1).is_ok(),
        "N-2 is P-2's bytes: valid in the proof slot"
    );

    // ... and the real credential-slot parser rejects the same bytes.
    let as_token = String::from_utf8_lossy(&n2.1);
    assert!(
        crate::auth::jwt::parse_protected_header(&as_token).is_err(),
        "a proof CWT must not parse as a credential protected header"
    );

    // The type rule holds independently of the encoding rule: a
    // compact-shaped token whose typ is the proof media type is rejected by
    // the same closed allowed-type dispatch the credential slot uses, for
    // both the request-proof and response-proof media types.
    let credential_slot_types = [
        crate::auth::RFC9068_ACCESS_TOKEN_TYPES[0],
        crate::auth::RFC9068_ACCESS_TOKEN_TYPES[1],
        "wit+jwt",
    ];
    for proof_typ in [
        crate::proof::PROOF_TYP,
        crate::proof::RESPONSE_PROOF_TYP,
    ] {
        assert!(
            !credential_slot_types.contains(&proof_typ),
            "a proof media type must never be an accepted credential type"
        );
        let header = format!(
            r#"{{"alg":"ML-DSA-65-Ed25519","typ":"{proof_typ}","kid":"k1"}}"#
        );
        let token = format!(
            "{}.{}.{}",
            URL_SAFE_NO_PAD.encode(header),
            URL_SAFE_NO_PAD.encode(&n2.1),
            URL_SAFE_NO_PAD.encode([0u8; 64])
        );
        // The header itself parses — this is not an encoding rejection ...
        assert!(
            crate::auth::jwt::parse_protected_header(&token).is_ok(),
            "the constructed token must be well-formed, so the type rule is what denies"
        );
        // ... and the credential dispatch still refuses it on type alone.
        assert!(
            crate::auth::parse_composite_dispatch(&token, &credential_slot_types).is_err(),
            "a proof-typed token must be rejected by the credential slot"
        );
    }
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
    use crate::proof::policy::{
        evaluate, AllowedApproverGroup, ApproverRule, CryptoSuite, SignaturePolicy,
    };

    // P-5's approver occupies signed logical signer group 2 under the
    // standalone suite; the fixture enrols it with the "security" role.
    let allowed_group_2 = || AllowedApproverGroup {
        group_id: 2,
        suite: CryptoSuite::Classical,
        role: "security".to_owned(),
    };

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
            approver_rule: ApproverRule::KOfN {
                k: 1,
                groups: vec![allowed_group_2()],
            },
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
            approver_rule: ApproverRule::KOfN {
                k: 1,
                groups: vec![allowed_group_2()],
            },
        },
        p5.disposition,
        &p5_v
    )
    .is_ok());
    // Two approvals are required; one does not satisfy the threshold.
    assert!(evaluate(
        &SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: ApproverRule::KOfN {
                k: 2,
                groups: vec![allowed_group_2()],
            },
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
