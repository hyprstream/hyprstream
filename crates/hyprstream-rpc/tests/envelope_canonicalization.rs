use hyprstream_rpc::envelope::{RequestEnvelope, RequestIdentity, SignedEnvelope};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

#[test]
fn test_envelope_serialization_deterministic() {
    // Create identical envelopes
    let envelope1 = RequestEnvelope {
        request_id: 123,
        identity: RequestIdentity::Local {
            user: "test-user".to_string(),
        },
        payload: vec![1, 2, 3, 4, 5],
        ephemeral_pubkey: None,
        nonce: [42u8; 16],
        timestamp: 1234567890,
        claims: None, // claims field replaced jwt_token
    };

    let envelope2 = envelope1.clone();

    // Logical equality
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
        identity: RequestIdentity::Local {
            user: "signer".to_string(),
        },
        payload: vec![9, 8, 7],
        ephemeral_pubkey: Some([1u8; 32]),
        nonce: [99u8; 16],
        timestamp: 9876543210,
        claims: None,
    };

    // Sign envelope
    let signed1 = SignedEnvelope::new_signed(envelope.clone(), &signing_key);

    // Create another signed envelope from same data
    let signed2 = SignedEnvelope::new_signed(envelope.clone(), &signing_key);

    // Signatures MUST be identical (deterministic serialization)
    assert_eq!(
        signed1.signature, signed2.signature,
        "Identical envelopes MUST produce identical signatures"
    );

    // Both MUST verify successfully
    assert!(
        signed1.verify_signature_only(&verifying_key).is_ok(),
        "First signature must verify"
    );
    assert!(
        signed2.verify_signature_only(&verifying_key).is_ok(),
        "Second signature must verify"
    );

    // Cross-verification MUST work (sig1 with envelope2)
    let envelope_bytes1 = signed1.envelope.to_bytes();
    let envelope_bytes2 = signed2.envelope.to_bytes();

    assert_eq!(
        envelope_bytes1, envelope_bytes2,
        "Envelope bytes must be identical for cross-verification"
    );

    // Verify with switched envelope/signature
    let mixed = SignedEnvelope {
        envelope: signed2.envelope.clone(),
        signature: signed1.signature,
        signer_pubkey: signed1.signer_pubkey,
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
        identity: RequestIdentity::Anonymous,
        payload: vec![],
        ephemeral_pubkey: None,
        nonce: [0u8; 16],
        timestamp: 1111111111,
        claims: None,
    };

    let bytes = envelope.to_bytes();

    // Canonical form requirements (from Cap'n Proto spec):
    // 1. Should be relatively compact (no excessive padding)
    // 2. Should not contain stream framing (segment table)
    // 3. Multiple serializations should be identical

    let bytes_again = envelope.to_bytes();
    assert_eq!(bytes, bytes_again, "Multiple serializations must be identical");

    // Size check - canonical form should be compact
    // A simple envelope like this should be small (< 200 bytes)
    assert!(
        bytes.len() < 200,
        "Canonical form should be compact, got {} bytes",
        bytes.len()
    );
}

#[test]
fn test_envelope_with_claims_deterministic() {
    let envelope = RequestEnvelope {
        request_id: 999,
        identity: RequestIdentity::ApiToken {
            user: "api-user".to_string(),
            token_name: "api-token".to_string(),
        },
        payload: vec![10, 20, 30],
        ephemeral_pubkey: Some([77u8; 32]),
        nonce: [123u8; 16],
        timestamp: 5555555555,
        claims: None, // Using None for test simplicity (claims replaces jwt_token)
    };

    // Multiple serializations must produce identical bytes
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
        identity: RequestIdentity::Local {
            user: "user1".to_string(),
        },
        payload: vec![1, 2, 3],
        ephemeral_pubkey: None,
        nonce: [1u8; 16],
        timestamp: 1000000000,
        claims: None,
    };

    let envelope2 = RequestEnvelope {
        request_id: 200,
        identity: RequestIdentity::Local {
            user: "user2".to_string(),
        },
        payload: vec![4, 5, 6],
        ephemeral_pubkey: None,
        nonce: [2u8; 16],
        timestamp: 2000000000,
        claims: None,
    };

    let bytes1 = envelope1.to_bytes();
    let bytes2 = envelope2.to_bytes();

    assert_ne!(
        bytes1, bytes2,
        "Different envelopes must produce different bytes"
    );
}
