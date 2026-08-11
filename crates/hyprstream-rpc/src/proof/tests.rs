#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Full frozen-vector acceptance suite — iterates every positive and negative
//! vector from the gate-2 artifacts and asserts each declared `expect` result.

use hex;

// ---------------------------------------------------------------------------
// Vector loading
// ---------------------------------------------------------------------------

fn load_positive_vectors() -> Vec<(String, Vec<u8>)> {
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

/// Every negative vector MUST be denied by the parser.
///
/// Exceptions: N-2 is P-2 presented in the wrong slot (the parser cannot
/// detect this without credential context), and N-22 is a response proof
/// whose cti mismatch is a stateful verifier obligation. These two are
/// tracked as verifier-side checks, not parser rules.
#[test]
fn all_negative_vectors_deny() {
    let skip_parser = ["N-2", "N-22"]; // verifier-side, not parser-side
    for (id, cbor, deny_class) in load_negative_vectors() {
        if skip_parser.contains(&id.as_str()) {
            continue;
        }
        let result = crate::proof::parser::ParsedProof::parse(&cbor);
        assert!(
            result.is_err(),
            "negative vector {id} ({deny_class}) should deny, but was accepted"
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
