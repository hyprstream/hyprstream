use hyprstream_rpc::envelope::{Authorization, RequestEnvelope, SignedEnvelope};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

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
    };

    let signed1 = SignedEnvelope::new_signed(envelope.clone(), &signing_key);
    let signed2 = SignedEnvelope::new_signed(envelope.clone(), &signing_key);

    // Signatures MUST be identical (deterministic serialization)
    assert_eq!(
        signed1.sig, signed2.sig,
        "Identical envelopes MUST produce identical signatures"
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

    // Verify with switched envelope/signature
    let mixed = SignedEnvelope {
        envelope: signed2.envelope.clone(),
        sig: signed1.sig,
        cnf: signed1.cnf,
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
    };

    let bytes = envelope.to_bytes();

    let bytes_again = envelope.to_bytes();
    assert_eq!(bytes, bytes_again, "Multiple serializations must be identical");

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
    };

    let bytes1 = envelope.to_bytes();
    let bytes2 = envelope.to_bytes();
    let bytes3 = envelope.to_bytes();

    assert_eq!(bytes1, bytes2, "First and second serialization must match");
    assert_eq!(bytes2, bytes3, "Second and third serialization must match");
}

#[test]
fn test_envelope_different_data_different_bytes() {
    let envelope1 = RequestEnvelope {
        request_id: 100,
        payload: vec![1, 2, 3],
        nonce: [1u8; 16],
        iat: 1000000000,
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
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
    };

    let bytes1 = envelope1.to_bytes();
    let bytes2 = envelope2.to_bytes();

    assert_ne!(
        bytes1, bytes2,
        "Different envelopes must produce different bytes"
    );
}
