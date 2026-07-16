use ed25519_dalek::SigningKey;
use hyprstream_rpc::envelope::{Authorization, RequestEnvelope, SignedEnvelope};
use rand::rngs::OsRng;

fn make_signed(envelope: RequestEnvelope, signing_key: &SigningKey) -> SignedEnvelope {
    // Classical so the raw Ed25519 `sig` field is deterministic (ML-DSA-65
    // signatures are randomized/hedged). Composite behavior is tested in the
    // envelope `cose::`/`crypto::` suites.
    SignedEnvelope::new_signed(envelope, signing_key)
}

#[test]
fn test_envelope_serialization_deterministic() {
    let envelope1 = RequestEnvelope {
        request_id: 123,
        payload: vec![1, 2, 3, 4, 5],
        nonce: [42u8; 16],
        iat: 1234567890,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: None,
        service_domain: None,
    };

    let envelope2 = envelope1.clone();

    assert_eq!(envelope1.request_id, envelope2.request_id);
    assert_eq!(envelope1.payload, envelope2.payload);

    // CRITICAL: Must produce IDENTICAL bytes
    let bytes1 = envelope1.to_bytes();
    let bytes2 = envelope2.to_bytes();

    assert_eq!(
        bytes1, bytes2,
        "Canonicalization MUST be deterministic - same data MUST produce identical bytes"
    );
}

#[test]
fn test_envelope_signature_verification_stable() {
    let mut csprng = OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();

    let envelope = RequestEnvelope {
        request_id: 456,
        payload: vec![9, 8, 7],
        nonce: [99u8; 16],
        iat: 9876543210,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: None,
        service_domain: None,
    };

    let signed1 = make_signed(envelope.clone(), &signing_key);
    let signed2 = make_signed(envelope.clone(), &signing_key);

    // Ed25519 signatures MUST be identical (deterministic serialization)
    assert_eq!(
        signed1.sig, signed2.sig,
        "Identical envelopes MUST produce identical Ed25519 signatures"
    );

    assert!(
        signed1.verify_signature_only(&verifying_key).is_ok(),
        "First signature must verify"
    );
    assert!(
        signed2.verify_signature_only(&verifying_key).is_ok(),
        "Second signature must verify"
    );

    let envelope_bytes1 = signed1.envelope.to_bytes();
    let envelope_bytes2 = signed2.envelope.to_bytes();

    assert_eq!(
        envelope_bytes1, envelope_bytes2,
        "Envelope bytes must be identical for cross-verification"
    );

    // Verify with switched envelope/signature (Ed25519 cross-verification)
    let mixed = SignedEnvelope {
        envelope: signed2.envelope.clone(),
        sig: signed1.sig,
        cnf: signed1.cnf,
        encrypted_envelope: None,
        client_ephemeral_public: None,
        cose: signed1.cose.clone(),
        policy: signed1.policy,
        pq_kem_ciphertext: None,
    };

    assert!(
        mixed.verify_signature_only(&verifying_key).is_ok(),
        "Cross-verification must work due to deterministic serialization"
    );
}

#[test]
fn test_envelope_canonical_form() {
    let envelope = RequestEnvelope {
        request_id: 789,
        payload: vec![],
        nonce: [0u8; 16],
        iat: 1111111111,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: None,
        service_domain: None,
    };

    let bytes = envelope.to_bytes();

    let bytes_again = envelope.to_bytes();
    assert_eq!(
        bytes, bytes_again,
        "Multiple serializations must be identical"
    );

    assert!(
        bytes.len() < 200,
        "Canonical form should be compact, got {} bytes",
        bytes.len()
    );
}

#[test]
fn test_envelope_with_authorization_deterministic() {
    let envelope = RequestEnvelope {
        request_id: 999,
        payload: vec![10, 20, 30],
        nonce: [123u8; 16],
        iat: 5555555555,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: None,
        service_domain: None,
    };

    let bytes1 = envelope.to_bytes();
    let bytes2 = envelope.to_bytes();
    let bytes3 = envelope.to_bytes();

    assert_eq!(bytes1, bytes2, "First and second serialization must match");
    assert_eq!(bytes2, bytes3, "Second and third serialization must match");
}

#[test]
fn test_envelope_different_data_different_bytes() {
    use hyprstream_rpc::crypto::hybrid_kem::{RecipientPublic, SuiteId};

    let envelope1 = RequestEnvelope {
        request_id: 100,
        payload: vec![1, 2, 3],
        nonce: [1u8; 16],
        iat: 1000000000,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: Some(RecipientPublic {
            suite_id: SuiteId::HyKemX25519MlKem768,
            eks: vec![vec![0x11; 32], vec![0x22; 1184]],
        }),
        service_domain: Some("canonical-a".to_owned()),
    };

    let envelope2 = RequestEnvelope {
        request_id: 200,
        payload: vec![4, 5, 6],
        nonce: [2u8; 16],
        iat: 2000000000,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: Some(RecipientPublic {
            suite_id: SuiteId::HyKemX25519MlKem768,
            eks: vec![vec![0x33; 32], vec![0x44; 1184]],
        }),
        service_domain: Some("canonical-b".to_owned()),
    };

    let bytes1 = envelope1.to_bytes();
    let bytes2 = envelope2.to_bytes();

    assert_ne!(
        bytes1, bytes2,
        "Different envelopes must produce different bytes"
    );
}

#[test]
fn test_populated_response_recipient_changes_canonical_bytes() {
    use hyprstream_rpc::crypto::hybrid_kem::{RecipientPublic, SuiteId};

    let recipient = |x25519: u8, mlkem: u8| RecipientPublic {
        suite_id: SuiteId::HyKemX25519MlKem768,
        eks: vec![vec![x25519; 32], vec![mlkem; 1184]],
    };
    let first = RequestEnvelope {
        request_id: 300,
        payload: vec![7, 8, 9],
        nonce: [3u8; 16],
        iat: 3000000000,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        response_kem_recipient: Some(recipient(0x55, 0x66)),
        service_domain: Some("canonical-service".to_owned()),
    };
    let first_again = first.to_bytes();
    assert_eq!(first_again, first.to_bytes());

    let mut second = first.clone();
    second.response_kem_recipient = Some(recipient(0x77, 0x88));
    assert_ne!(
        first.to_bytes(),
        second.to_bytes(),
        "changing only a populated response recipient must change canonical bytes"
    );
}
