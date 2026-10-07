//! Offline fixtures only. test-rsa.pem is disposable public test material.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use ed25519_dalek::{Signer, SigningKey};
use hyprstream_rpc::crypto::pq::{ml_dsa_sign, ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
use jsonwebtoken::{EncodingKey, Header};
use serde_json::json;

const NOW: u64 = 1_800_000_000;
const CODE: &str = "TEST-CODE-NOT-LIVE";

#[test]
fn reserved_host_client_cannot_downgrade_by_stripping_generation() {
    let mut claims = hyprstream_rpc::auth::Claims::new("alice".into(), 1, 2);
    claims.iss = HOST.into();
    claims.client_id = Some(CLIENT.into());
    assert!(claims.is_reserved_federate_staging_credential());
    claims.client_id = None;
    assert!(!claims.is_reserved_federate_staging_credential());
    claims.client_id = Some("other-client".into());
    assert!(!claims.is_reserved_federate_staging_credential());
    claims.iss = "https://other.example".into();
    claims.client_id = Some(CLIENT.into());
    assert!(!claims.is_reserved_federate_staging_credential());
}
const MODULUS: &str = "C97781230C20F14C2EB47FDD3AFBB4821EAC87003FF1A8240D1E840EE743E3EAF7A5F204503E15CFB42751AEDB97D76BE41DF559834957D283E4AC097283D18D19A12DCE56434FCA31BFC9635E721798A3D6AACC55FC8CAD559F74D66549EFC160AAD48DA9A2F860CCEA515DFEEEFA930E10B8BA5414CB26034B7352E53C7AAD5C4CE24528615705C737B7D4B554B1B4CE1A52F2D3439B42F67324B4F007691E46BE826EC6DC7CC65DD58FBF183B86DF537889EDBA5866BFD911A2AA27181C9EBD14C6DBACA28E3A37153BCD0018071101E6EA5A900812D194E6220A672EDB9FCA2C24C5486DCF981C3A2C9DC20B95E3D14875F9AF20993751BB8CBB6020678D";
fn bytes(hex: &str) -> Vec<u8> {
    hex::decode(hex).unwrap()
}
fn jwks() -> Value {
    json!({"keys":[{"kty":"RSA","kid":"fixture","alg":"RS256","use":"sig","n":B64.encode(bytes(MODULUS)),"e":"AQAB"}]})
}
fn verifier() -> Verifier {
    let mut verifier = Verifier::fixed_staging().unwrap();
    verifier.responses = Some(parking_lot::Mutex::new(vec![Ok(serde_json::to_vec(
        &jwks(),
    )
    .unwrap())]));
    verifier
}
fn commitment() -> Commitment {
    let ed = SigningKey::from_bytes(&[7; 32]);
    let pq = ml_dsa_sk_from_seed(&[8; 32]);
    Commitment::new(
        &B64.encode([1; 32]),
        &B64.encode([2; 32]),
        &B64.encode(sha(B64.encode([3; 32]).as_bytes())),
        &B64.encode(ed.verifying_key().as_bytes()),
        &B64.encode(ml_dsa_sk_to_vk_bytes(&pq)),
    )
    .unwrap()
}
fn claims() -> Value {
    json!({"iss":ISSUER,"aud":CLIENT,"sub":"opaque-sub","jti":"source-id","iat":NOW,"exp":NOW+300,"nonce":commitment().nonce(),"c_hash":B64.encode(&sha(CODE.as_bytes())[..16]),"federated_claims":{"connector_id":"atproto","user_id":"did:plc:abcdefghijklmnopqrstuvwx"}})
}
fn token(v: &Value) -> String {
    let mut h = Header::new(Algorithm::RS256);
    h.kid = Some("fixture".into());
    jsonwebtoken::encode(&h, v, &encoding_key()).unwrap()
}
fn raw_token(header: &str, claims: &str) -> String {
    let msg = format!("{}.{}", B64.encode(header), B64.encode(claims));
    let sig = crypto::sign(msg.as_bytes(), &encoding_key(), Algorithm::RS256).unwrap();
    format!("{msg}.{sig}")
}

fn encoding_key() -> EncodingKey {
    let pem: String = include_str!("test-rsa.pem")
        .lines()
        .filter(|s| !s.starts_with('-'))
        .collect();
    EncodingKey::from_rsa_der(
        &base64::engine::general_purpose::STANDARD
            .decode(pem)
            .unwrap(),
    )
}
fn binding() -> AuthorityBinding {
    AuthorityBinding {
        account: "test-local-account".into(),
        subject: "test-rpc-subject".into(),
        tenant: "test-tenant".into(),
        requested: "query:registry:*".into(),
        granted: "query:registry:*".into(),
        revision: "test-grant-revision".into(),
    }
}
async fn source(v: &Verifier, t: &str) -> Source {
    v.source(t, CODE, commitment(), NOW, None).await.unwrap()
}

#[tokio::test]
async fn valid_source_and_both_signatures_only() {
    let v = verifier();
    let t = token(&claims());
    let c = source(&v, &t)
        .await
        .challenge(binding(), [9; 32], NOW)
        .unwrap();
    let ed = SigningKey::from_bytes(&[7; 32])
        .sign(c.transcript())
        .to_bytes();
    let pq = ml_dsa_sign(&ml_dsa_sk_from_seed(&[8; 32]), c.transcript());
    let evidence = c
        .verify(source(&v, &t).await, &[9; 32], &ed, &pq, NOW + 1)
        .unwrap();
    assert_eq!(evidence.source.subject, "opaque-sub");
    assert_eq!(evidence.source.atproto_did, "did:plc:abcdefghijklmnopqrstuvwx");
    assert_eq!(evidence.source.jti, "source-id");
    assert_eq!(evidence.binding.tenant, "test-tenant");
    for bad in 0..6 {
        let c = source(&v, &t)
            .await
            .challenge(binding(), [9; 32], NOW)
            .unwrap();
        let mut es = ed;
        let mut ps = pq.clone();
        let mut id = [9; 32];
        match bad {
            0 => es[0] ^= 1,
            1 => ps[0] ^= 1,
            2 => id[0] ^= 1,
            3 => {
                ps.pop();
            }
            4 => {
                es = SigningKey::from_bytes(&[10; 32])
                    .sign(c.transcript())
                    .to_bytes();
            }
            _ => ps = ml_dsa_sign(&ml_dsa_sk_from_seed(&[11; 32]), c.transcript()),
        }
        assert!(c
            .verify(source(&v, &t).await, &id, &es, &ps, NOW + 1)
            .is_err());
    }
}

#[tokio::test]
async fn claims_fail_closed() {
    let v = verifier();
    let cases = vec![
        ("iss", json!("https://login.federate.to/dex")),
        ("aud", json!("other")),
        ("aud", json!([CLIENT, "other"])),
        ("aud", json!([CLIENT, CLIENT])),
        ("azp", json!("other")),
        ("sub", json!("")),
        ("federated_claims", Value::Null),
        ("jti", Value::Null),
        ("iat", json!(NOW + 31)),
        ("iat", json!(NOW - 121)),
        ("iat", json!(-1)),
        ("iat", json!(NOW as f64)),
        ("exp", json!(NOW)),
        ("exp", json!(NOW + 301)),
        ("nbf", json!(NOW + 31)),
        ("nbf", json!(-1)),
        ("nonce", json!("hsn1.wrong")),
        ("c_hash", json!("invalid")),
    ];
    for (k, value) in cases {
        let mut c = claims();
        c[k] = value;
        assert!(
            v.source(&token(&c), CODE, commitment(), NOW, None)
                .await
                .is_err(),
            "{k}"
        );
    }
    let mut c = claims();
    c["aud"] = json!([CLIENT]);
    c["azp"] = json!(CLIENT);
    assert!(v
        .source(&token(&c), CODE, commitment(), NOW, None)
        .await
        .is_ok());
    let t = token(&claims());
    assert!(v
        .source(&t, "wrong code", commitment(), NOW, None)
        .await
        .is_err());
    let mut c = commitment();
    c.pkce = B64.encode([5; 32]);
    assert!(v.source(&t, CODE, c, NOW, None).await.is_err());
    let mut c = commitment();
    c.state = B64.encode([5; 32]);
    assert!(v.source(&t, CODE, c, NOW, None).await.is_err());
    let mut c = commitment();
    c.pq = ml_dsa_sk_to_vk_bytes(&ml_dsa_sk_from_seed(&[5; 32]));
    assert!(v.source(&t, CODE, c, NOW, None).await.is_err());
}

#[tokio::test]
async fn federated_identity_claim_requires_exact_atproto_connector_and_did() {
    let v = verifier();
    for claim in [
        json!({"connector_id":"github","user_id":"did:plc:abcdefghijklmnopqrstuvwx"}),
        json!({"connector_id":"atproto","user_id":"alice.example"}),
        json!({"connector_id":"atproto","user_id":"did:plc:short"}),
    ] {
        let mut c = claims();
        c["federated_claims"] = claim;
        assert!(v.source(&token(&c), CODE, commitment(), NOW, None).await.is_err());
    }
    let mut c = claims();
    c["federated_claims"] = json!({"connector_id":"atproto","user_id":"did:web:users.example:alice"});
    assert!(v.source(&token(&c), CODE, commitment(), NOW, None).await.is_ok());
    c["federated_claims"] = json!({"connector_id":"atproto","user_id":"did:plc:abcdefghijklmnopqrstuvwx","future_claim":"ignored"});
    assert!(v.source(&token(&c), CODE, commitment(), NOW, None).await.is_ok());
}

#[tokio::test]
async fn at_hash_optional_host_input_but_checked_when_available() {
    let v = verifier();
    let mut c = claims();
    c["at_hash"] = json!(B64.encode(&sha(b"access")[..16]));
    let t = token(&c);
    assert!(v
        .source(&t, CODE, commitment(), NOW, Some("access"))
        .await
        .is_ok());
    assert!(v
        .source(&t, CODE, commitment(), NOW, Some("wrong"))
        .await
        .is_err());
    assert!(v.source(&t, CODE, commitment(), NOW, None).await.is_ok());
    c["at_hash"] = json!("invalid");
    assert!(v
        .source(&token(&c), CODE, commitment(), NOW, None)
        .await
        .is_err());
}

#[tokio::test]
async fn compact_jws_headers_duplicates_sizes_and_signature() {
    let v = verifier();
    let c = claims().to_string();
    for h in [
        r#"{"alg":"none","kid":"fixture"}"#,
        r#"{"alg":"HS256","kid":"fixture"}"#,
        r#"{"alg":"RS256","kid":"fixture","jku":"https://evil/keys"}"#,
        r#"{"alg":"RS256","kid":"fixture","jwk":{}}"#,
        r#"{"alg":"RS256","kid":"fixture","x5u":"https://evil"}"#,
        r#"{"alg":"RS256","kid":"fixture","crit":["b64"]}"#,
        r#"{"alg":"RS256","kid":"fixture","b64":false}"#,
        r#"{"alg":"RS256","kid":"fixture","typ":"at+jwt"}"#,
        r#"{"alg":"RS256","alg":"RS256","kid":"fixture"}"#,
    ] {
        assert!(v
            .source(&raw_token(h, &c), CODE, commitment(), NOW, None)
            .await
            .is_err());
    }
    let h = r#"{"alg":"RS256","kid":"fixture"}"#;
    for extra in [
        r#", "sub":"another"}"#,
        r#", "extra":{"x":1,"x":2}}"#,
        r#", "\u0073ub":"another"}"#,
    ] {
        let duplicate = format!("{}{extra}", &c[..c.len() - 1]);
        assert!(v
            .source(&raw_token(h, &duplicate), CODE, commitment(), NOW, None)
            .await
            .is_err());
    }
    assert!(v
        .source(&raw_token(h, &c), CODE, commitment(), NOW, None)
        .await
        .is_ok());
    for t in [
        "x".repeat(MAX_JWS + 1),
        "a.b.c.d".into(),
        format!("{}=", token(&claims())),
    ] {
        assert!(v.source(&t, CODE, commitment(), NOW, None).await.is_err());
    }
    let t = token(&claims());
    let mut parts: Vec<_> = t.split('.').map(str::to_owned).collect();
    let mut sig = decode(&parts[2], 512).unwrap();
    sig[0] ^= 1;
    parts[2] = B64.encode(sig);
    assert!(v
        .source(&parts.join("."), CODE, commitment(), NOW, None)
        .await
        .is_err());
    assert_eq!(
        Verifier::default()
            .source(&t, CODE, commitment(), NOW, None)
            .await
            .err(),
        Some(Error::Disabled)
    );
}

#[tokio::test]
async fn jwks_refresh_failure_unknown_kid_and_expiry() {
    let v = verifier();
    assert!(v.key("missing").await.is_err());
    assert!(v.responses.as_ref().unwrap().lock().is_empty());
    assert!(v.key("fixture").await.is_ok());
    v.keys.lock().await.as_mut().unwrap().fetched = Instant::now() - KEY_TTL;
    assert_eq!(v.key("fixture").await.err(), Some(Error::Unavailable));
    let mut v = verifier();
    v.responses = Some(parking_lot::Mutex::new(vec![Err(Error::Unavailable)]));
    assert_eq!(v.key("fixture").await.err(), Some(Error::Unavailable));
    let mut v = verifier();
    v.responses = Some(parking_lot::Mutex::new(vec![Ok(vec![b' '; MAX_JWKS + 1])]));
    assert_eq!(v.key("fixture").await.err(), Some(Error::Unavailable));
}

#[test]
fn reject_unsafe_jwks() {
    assert!(parse_keys(&serde_json::to_vec(&jwks()).unwrap()).is_ok());
    for (k, value) in [
        ("d", json!("private")),
        ("jku", json!("https://evil")),
        ("x5u", json!("https://evil")),
        ("kty", json!("oct")),
        ("alg", json!("HS256")),
        ("use", json!("enc")),
        ("key_ops", json!(["sign"])),
        ("n", json!("AQAB")),
        ("e", json!("Ag")),
    ] {
        let mut j = jwks();
        j["keys"][0][k] = value;
        assert!(parse_keys(&serde_json::to_vec(&j).unwrap()).is_err(), "{k}");
    }
    let j = jwks();
    let duplicate = json!({"keys":[j["keys"][0],j["keys"][0]]});
    assert!(parse_keys(&serde_json::to_vec(&duplicate).unwrap()).is_err());
    assert!(parse_keys(br#"{"keys":[],"keys":[]}"#).is_err());
    assert!(parse_keys(&vec![b' '; MAX_JWKS + 1]).is_err());
}

#[tokio::test]
async fn challenge_binding_expiry_and_source_replay_mismatch() {
    let v = verifier();
    let t = token(&claims());
    for bad in 0..4 {
        let c = source(&v, &t)
            .await
            .challenge(binding(), [9; 32], NOW)
            .unwrap();
        let ed = SigningKey::from_bytes(&[7; 32])
            .sign(c.transcript())
            .to_bytes();
        let pq = ml_dsa_sign(&ml_dsa_sk_from_seed(&[8; 32]), c.transcript());
        let mut replacement = claims();
        if bad == 0 {
            replacement["jti"] = json!("different");
        }
        let mut s = source(&v, &token(&replacement)).await;
        if bad == 1 {
            s.code_hash = [0; 32];
        }
        let time = if bad == 2 {
            NOW + 60
        } else if bad == 3 {
            NOW - 1
        } else {
            NOW + 1
        };
        assert!(c.verify(s, &[9; 32], &ed, &pq, time).is_err());
    }
    let c = source(&v, &t)
        .await
        .challenge(binding(), [9; 32], NOW)
        .unwrap();
    let ed = SigningKey::from_bytes(&[7; 32])
        .sign(c.transcript())
        .to_bytes();
    let pq = ml_dsa_sign(&ml_dsa_sk_from_seed(&[8; 32]), c.transcript());
    let mut b = binding();
    b.tenant = "other-tenant".into();
    let different = source(&v, &t).await.challenge(b, [9; 32], NOW).unwrap();
    assert!(different
        .verify(source(&v, &t).await, &[9; 32], &ed, &pq, NOW)
        .is_err());
    let mut b = binding();
    b.granted = "write:registry:*".into();
    assert!(source(&v, &t).await.challenge(b, [9; 32], NOW).is_err());
}

#[tokio::test]
async fn authority_binding_limits_match_browser_transcript_profile() {
    let v = verifier();
    let t = token(&claims());
    let mut b = binding();
    b.account = "a".repeat(MAX_AUTHORITY_TEXT);
    b.subject = "s".repeat(MAX_AUTHORITY_TEXT);
    b.tenant = "t".repeat(MAX_AUTHORITY_TEXT);
    b.revision = "r".repeat(MAX_AUTHORITY_TEXT);
    b.requested = "q".repeat(MAX_SCOPE_BYTES);
    b.granted = b.requested.clone();
    let challenge = source(&v, &t).await.challenge(b, [9; 32], NOW).unwrap();
    assert!(challenge.transcript().len() <= MAX_TRANSCRIPT);

    // The A2 browser counts the encoded transcript field bytes, not Rust or
    // JavaScript character count; a two-byte UTF-8 value at exactly 256 bytes
    // is admitted, while the next character exceeds the shared bound.
    let mut b = binding();
    b.account = "é".repeat(MAX_AUTHORITY_TEXT / 2);
    assert!(source(&v, &t).await.challenge(b, [9; 32], NOW).is_ok());
    let mut b = binding();
    b.account = "é".repeat(MAX_AUTHORITY_TEXT / 2 + 1);
    assert!(source(&v, &t).await.challenge(b, [9; 32], NOW).is_err());

    for field in ["account", "subject", "tenant", "revision"] {
        let mut b = binding();
        match field {
            "account" => b.account = "a".repeat(MAX_AUTHORITY_TEXT + 1),
            "subject" => b.subject = "s".repeat(MAX_AUTHORITY_TEXT + 1),
            "tenant" => b.tenant = "t".repeat(MAX_AUTHORITY_TEXT + 1),
            "revision" => b.revision = "r".repeat(MAX_AUTHORITY_TEXT + 1),
            _ => unreachable!(),
        }
        assert!(
            source(&v, &t).await.challenge(b, [9; 32], NOW).is_err(),
            "{field} byte limit"
        );

        for control in ['\0', '\n', '\u{7f}'] {
            let mut b = binding();
            let value = format!("before{control}after");
            match field {
                "account" => b.account = value,
                "subject" => b.subject = value,
                "tenant" => b.tenant = value,
                "revision" => b.revision = value,
                _ => unreachable!(),
            }
            assert!(
                source(&v, &t).await.challenge(b, [9; 32], NOW).is_err(),
                "{field} control U+{:04X}",
                control as u32
            );
        }
    }

    let mut b = binding();
    b.requested = "q".repeat(MAX_SCOPE_BYTES + 1);
    b.granted = b.requested.clone();
    assert!(
        source(&v, &t).await.challenge(b, [9; 32], NOW).is_err(),
        "scope byte limit"
    );

    for scope in [
        "query:registry:*\n",
        "query:registry:*\u{7f}",
        "query\"scope",
        "query\\scope",
        "scope  with-gap",
        "z a",
    ] {
        let mut b = binding();
        b.requested = scope.into();
        b.granted = scope.into();
        assert!(
            source(&v, &t).await.challenge(b, [9; 32], NOW).is_err(),
            "scope {scope:?}"
        );
    }
}

#[test]
fn published_cross_language_framing_vectors() {
    let v: Value = serde_json::from_str(include_str!("framing-vectors.json")).unwrap();
    let fields: Vec<Vec<u8>> = v["login_field_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| bytes(x.as_str().unwrap()))
        .collect();
    // Patterned PQ data is only an encoding fixture, not a signing identity.
    let c = Commitment {
        random: fields[6].clone().try_into().unwrap(),
        state: B64.encode((32..64).collect::<Vec<u8>>()),
        pkce: String::from_utf8(fields[10].clone()).unwrap(),
        ed: fields[7].clone().try_into().unwrap(),
        pq: fields[8].clone(),
    };
    assert_eq!(
        hex::encode(c.preimage()),
        v["nonce_preimage_hex"].as_str().unwrap()
    );
    assert_eq!(c.nonce(), v["nonce"].as_str().unwrap());
    for (actual, expected) in c
        .kids()
        .iter()
        .zip(v["component_kids_hex"].as_array().unwrap())
    {
        assert_eq!(hex::encode(actual), expected.as_str().unwrap());
    }
    let s = Source {
        subject: "unused".into(),
        atproto_did: "did:plc:abcdefghijklmnopqrstuvwx".into(),
        jti: "unused".into(),
        issued_at: NOW,
        expires_at: NOW + 300,
        token_hash: sha(b"TEST-ID-TOKEN-NOT-A-JWT"),
        code_hash: sha(CODE.as_bytes()),
        commitment: c,
    };
    let c = s
        .challenge(
            binding(),
            (96..128).collect::<Vec<u8>>().try_into().unwrap(),
            NOW,
        )
        .unwrap();
    assert_eq!(
        hex::encode(c.transcript()),
        v["possession_transcript_hex"].as_str().unwrap()
    );
    assert_eq!(
        hex::encode(sha(c.transcript())),
        v["possession_transcript_sha256"].as_str().unwrap()
    );
    assert_eq!(
        B64.encode(&sha(CODE.as_bytes())[..16]),
        v["c_hash"].as_str().unwrap()
    );
    assert_ne!(frame("d", &[b"ab", b"c"]), frame("d", &[b"a", b"bc"]));
}

#[tokio::test]
async fn browser_library_signatures_match_native_verifier() {
    let vector: Value = serde_json::from_str(include_str!("signature-vector.json")).unwrap();
    let v = verifier();
    let t = vector["token"].as_str().unwrap();
    let source = source(&v, t).await;
    assert_eq!(
        B64.encode(source.commitment.ed),
        vector["ed_public"].as_str().unwrap()
    );
    assert_eq!(
        B64.encode(&source.commitment.pq),
        vector["pq_public"].as_str().unwrap()
    );
    assert_eq!(source.commitment.nonce(), vector["nonce"].as_str().unwrap());
    let c = source.challenge(binding(), [9; 32], NOW).unwrap();
    assert_eq!(
        B64.encode(c.transcript()),
        vector["transcript"].as_str().unwrap()
    );
    let es = decode(vector["ed_signature"].as_str().unwrap(), 64).unwrap();
    let ps = decode(vector["pq_signature"].as_str().unwrap(), 3309).unwrap();
    c.verify(self::source(&v, t).await, &[9; 32], &es, &ps, NOW)
        .unwrap();
}
