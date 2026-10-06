//! Authenticated request-proof CWT production for the frozen v16 WNS profile.
//!
//! This module emits only the primary authenticated Ed25519 + ML-DSA-65
//! profile.  It takes already-owned proof-dedicated signer material and exact
//! credential/body inputs; it neither mints credentials nor consults mesh,
//! envelope, or enrollment key stores.  Callers must arrange the separately
//! enrolled proof keys and the credential's `cnf` binding before transmission.

use anyhow::{bail, Result};
use ciborium::value::Value as CborValue;
use ed25519_dalek::Signer as _;
use rand::RngCore as _;
use sha2::{Digest as _, Sha256};
use crate::transport_traits::Signer;

use super::{
    recipient_binding::FederateRecipientBinding,
    response::{KemRecipient, ProtectionMode, ResponseBinding, ResponseKind},
    CredentialHash, RequestId, ALG_ED25519, ALG_ML_DSA_65, CLAIM_CAPNP_BODY_BYTES,
    CLAIM_CAPNP_SCHEMA_ID, CLAIM_CREDENTIAL_HASH, CLAIM_FEDERATE_RECIPIENT_BINDING,
    CLAIM_RESPONSE_BINDING, COSE_HEADER_ALG,
    COSE_HEADER_CRIT, COSE_HEADER_KID, COSE_HEADER_TYP, CWT_CLAIM_AUD, CWT_CLAIM_CTI,
    CWT_CLAIM_EXP, CWT_CLAIM_IAT, HEADER_HS_DOMAIN, HEADER_HS_LOGICAL_SIGNER_GROUP,
    HEADER_HS_SIGNATURE_PLAN, MAX_BODY_BYTES, MAX_KID_BYTES, PROOF_TYP, REQUEST_PROOF_DOMAIN,
    SUITE_HYBRID,
};

/// Proof-dedicated private components for the sole supported producer suite.
///
/// These keys are deliberately supplied by the caller.  They must be distinct
/// from mesh, envelope, CA, and node identity keys, and their public halves
/// must already be enrolled as one primary WNS suite by the verifier.
pub struct AuthenticatedHybridProofSigner {
    ed25519: ed25519_dalek::SigningKey,
    ed25519_kid: Vec<u8>,
    ml_dsa_65: crate::crypto::pq::MlDsaSigningKey,
    ml_dsa_65_kid: Vec<u8>,
}

impl AuthenticatedHybridProofSigner {
    /// Constructs a proof signer from already-owned, proof-dedicated keys.
    pub fn new(
        ed25519: ed25519_dalek::SigningKey,
        ed25519_kid: impl Into<Vec<u8>>,
        ml_dsa_65: crate::crypto::pq::MlDsaSigningKey,
        ml_dsa_65_kid: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        let ed25519_kid = ed25519_kid.into();
        let ml_dsa_65_kid = ml_dsa_65_kid.into();
        validate_kid(&ed25519_kid, "Ed25519")?;
        validate_kid(&ml_dsa_65_kid, "ML-DSA-65")?;
        Ok(Self {
            ed25519,
            ed25519_kid,
            ml_dsa_65,
            ml_dsa_65_kid,
        })
    }
}

/// Exact request commitments for an authenticated proof CWT.
pub struct AuthenticatedRequestProofInput<'a> {
    /// Canonical addressed RPC service domain.
    pub service_domain: &'a str,
    /// Exact credential bytes presented with the envelope; hashed as claim
    /// `-70001` without parsing, normalization, or substitution.
    pub credential: &'a [u8],
    /// Unix seconds used for CWT `iat`. W1 freshness is evaluated by the
    /// verifier against its injected clock; the producer records this value
    /// without consulting an ambient clock or imposing an `exp`/`iat` rule.
    pub issued_at: u64,
    /// Unix seconds used for CWT `exp`. W1 freshness is evaluated by the
    /// verifier against its injected clock; the producer records this value
    /// without consulting an ambient clock or imposing an issued-lifetime cap.
    pub expires_at: u64,
    /// Exact Cap'n Proto root schema ID for `capnp_schema_id`.
    pub capnp_schema_id: u64,
    /// Exact canonical Cap'n Proto request bytes for `capnp_body_bytes`.
    pub capnp_body: &'a [u8],
    /// Commitment to response delivery, or `None` for cleartext unary.
    pub response_binding: Option<&'a ResponseBinding>,
    /// Optional recipient-pair commitment for deferred Federate requests.
    /// Generic request proofs omit this source-slice extension.
    pub federate_recipient_binding: Option<&'a FederateRecipientBinding>,
}

/// Produces a deterministic authenticated request proof in the sole supported
/// WNS profile: one logical group with Ed25519 followed by ML-DSA-65.
///
/// The result is an untagged, deterministic `COSE_Sign`, parseable and
/// cryptographically verifiable by [`super::parser::ParsedProof`] and
/// [`super::verify::verify_proof_signatures`].  Unsupported plans and
/// classical-only emission are intentionally unavailable.
pub fn build_authenticated_hybrid_request_proof(
    input: &AuthenticatedRequestProofInput<'_>,
    signer: &AuthenticatedHybridProofSigner,
) -> Result<Vec<u8>> {
    // The public producer never accepts caller-selected replay identifiers.
    let mut request_id: RequestId = [0; super::REQUEST_ID_SIZE];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    build_authenticated_hybrid_request_proof_with_id(input, signer, request_id)
}

// Private seam for a reproducible signed test vector. Production always enters
// through the public builder above and generates its own request ID.
fn build_authenticated_hybrid_request_proof_with_id(
    input: &AuthenticatedRequestProofInput<'_>,
    signer: &AuthenticatedHybridProofSigner,
    request_id: RequestId,
) -> Result<Vec<u8>> {
    let (body_protected, ed_protected, pq_protected, payload) =
        prepare_hybrid_proof(input, &signer.ed25519_kid, &signer.ml_dsa_65_kid, request_id)?;
    let ed_tbs = signature_structure(&body_protected, &ed_protected, &payload)?;
    let pq_tbs = signature_structure(&body_protected, &pq_protected, &payload)?;
    let ed_signature = signer.ed25519.sign(&ed_tbs).to_bytes().to_vec();
    let pq_signature = crate::crypto::pq::ml_dsa_sign(&signer.ml_dsa_65, &pq_tbs);
    finish_hybrid_proof(body_protected, ed_protected, pq_protected, payload, ed_signature, pq_signature)
}

/// Produce the same v16 COSE_Sign bytes with a browser-backed signer. Its
/// private proof keys may remain in an isolated JS vault; only the exact
/// Sig_structure bytes cross the callback boundary. A fresh CWT request ID is
/// generated here for every call.
pub async fn build_authenticated_hybrid_request_proof_with_signer<S: Signer>(
    input: &AuthenticatedRequestProofInput<'_>,
    signer: &S,
    ed25519_kid: &[u8],
    ml_dsa_65_kid: &[u8],
) -> Result<Vec<u8>> {
    let mut request_id = [0; super::REQUEST_ID_SIZE];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    build_authenticated_hybrid_request_proof_with_signer_and_id(
        input, signer, ed25519_kid, ml_dsa_65_kid, request_id,
    ).await
}

async fn build_authenticated_hybrid_request_proof_with_signer_and_id<S: Signer>(
    input: &AuthenticatedRequestProofInput<'_>,
    signer: &S,
    ed25519_kid: &[u8],
    ml_dsa_65_kid: &[u8],
    request_id: RequestId,
) -> Result<Vec<u8>> {
    let pq_public = signer.pq_pubkey().ok_or_else(|| anyhow::anyhow!(
        "request proof: browser signer has no ML-DSA-65 key"
    ))?;
    let pq_key = crate::crypto::pq::ml_dsa_vk_from_bytes(&pq_public)?;
    let ed_key = ed25519_dalek::VerifyingKey::from_bytes(&signer.pubkey())?;
    let (body_protected, ed_protected, pq_protected, payload) =
        prepare_hybrid_proof(input, ed25519_kid, ml_dsa_65_kid, request_id)?;
    let ed_tbs = signature_structure(&body_protected, &ed_protected, &payload)?;
    let pq_tbs = signature_structure(&body_protected, &pq_protected, &payload)?;
    let ed_signature = signer.sign(&ed_tbs).await?;
    ed_key.verify_strict(&ed_tbs, &ed25519_dalek::Signature::from_bytes(&ed_signature))?;
    let pq_signature = signer.pq_sign(&pq_tbs).await?.ok_or_else(|| anyhow::anyhow!(
        "request proof: browser signer omitted ML-DSA-65 signature"
    ))?;
    crate::crypto::pq::ml_dsa_verify(&pq_key, &pq_tbs, &pq_signature)?;
    finish_hybrid_proof(
        body_protected, ed_protected, pq_protected, payload,
        ed_signature.to_vec(), pq_signature,
    )
}

fn prepare_hybrid_proof(
    input: &AuthenticatedRequestProofInput<'_>,
    ed25519_kid: &[u8],
    ml_dsa_65_kid: &[u8],
    request_id: RequestId,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    validate_kid(ed25519_kid, "Ed25519")?;
    validate_kid(ml_dsa_65_kid, "ML-DSA-65")?;
    crate::envelope::validate_service_domain(input.service_domain)?;
    if input.credential.is_empty() {
        bail!("request proof: credential bytes must not be empty");
    }
    if input.capnp_body.len() > MAX_BODY_BYTES {
        bail!(
            "request proof: capnp body {} exceeds {} bytes",
            input.capnp_body.len(),
            MAX_BODY_BYTES
        );
    }
    let credential_hash: CredentialHash = Sha256::digest(input.credential).into();
    let payload = encode_claims(input, credential_hash, request_id)?;
    let body_protected = encode_body_protected(ed25519_kid, ml_dsa_65_kid)?;
    let ed_protected = encode_signature_protected(ALG_ED25519, ed25519_kid)?;
    let pq_protected = encode_signature_protected(ALG_ML_DSA_65, ml_dsa_65_kid)?;
    Ok((body_protected, ed_protected, pq_protected, payload))
}

fn finish_hybrid_proof(
    body_protected: Vec<u8>,
    ed_protected: Vec<u8>,
    pq_protected: Vec<u8>,
    payload: Vec<u8>,
    ed_signature: Vec<u8>,
    pq_signature: Vec<u8>,
) -> Result<Vec<u8>> {
    let cose = CborValue::Array(vec![
        CborValue::Bytes(body_protected),
        CborValue::Map(vec![]),
        CborValue::Bytes(payload),
        CborValue::Array(vec![
            signature_entry(ed_protected, ed_signature),
            signature_entry(pq_protected, pq_signature),
        ]),
    ]);
    encode(&cose)
}

fn validate_kid(kid: &[u8], name: &str) -> Result<()> {
    if kid.is_empty() || kid.len() > MAX_KID_BYTES {
        bail!("request proof: {name} kid must be 1..{MAX_KID_BYTES} bytes");
    }
    Ok(())
}

fn encode_body_protected(ed_kid: &[u8], pq_kid: &[u8]) -> Result<Vec<u8>> {
    // Key order is RFC 8949 deterministic bytewise order: 2, 16, -70100, -70101.
    encode(&CborValue::Map(vec![
        (
            CborValue::Integer(COSE_HEADER_CRIT.into()),
            CborValue::Array(vec![
                CborValue::Integer(HEADER_HS_SIGNATURE_PLAN.into()),
                CborValue::Integer(HEADER_HS_DOMAIN.into()),
            ]),
        ),
        (
            CborValue::Integer(COSE_HEADER_TYP.into()),
            CborValue::Text(PROOF_TYP.into()),
        ),
        (
            CborValue::Integer(HEADER_HS_DOMAIN.into()),
            CborValue::Text(REQUEST_PROOF_DOMAIN.into()),
        ),
        (
            CborValue::Integer(HEADER_HS_SIGNATURE_PLAN.into()),
            signature_plan(ed_kid, pq_kid),
        ),
    ]))
}

fn signature_plan(ed_kid: &[u8], pq_kid: &[u8]) -> CborValue {
    CborValue::Array(vec![CborValue::Map(vec![
        (CborValue::Integer(1.into()), CborValue::Integer(1.into())),
        (
            CborValue::Integer(2.into()),
            CborValue::Text(SUITE_HYBRID.into()),
        ),
        (
            CborValue::Integer(3.into()),
            CborValue::Array(vec![
                component(ALG_ED25519, ed_kid),
                component(ALG_ML_DSA_65, pq_kid),
            ]),
        ),
    ])])
}

fn component(alg: i64, kid: &[u8]) -> CborValue {
    CborValue::Map(vec![
        (CborValue::Integer(1.into()), CborValue::Integer(alg.into())),
        (CborValue::Integer(2.into()), CborValue::Bytes(kid.to_vec())),
    ])
}

fn encode_signature_protected(alg: i64, kid: &[u8]) -> Result<Vec<u8>> {
    // Key order is 1, 2, 4, -70102.
    encode(&CborValue::Map(vec![
        (
            CborValue::Integer(COSE_HEADER_ALG.into()),
            CborValue::Integer(alg.into()),
        ),
        (
            CborValue::Integer(COSE_HEADER_CRIT.into()),
            CborValue::Array(vec![CborValue::Integer(
                HEADER_HS_LOGICAL_SIGNER_GROUP.into(),
            )]),
        ),
        (
            CborValue::Integer(COSE_HEADER_KID.into()),
            CborValue::Bytes(kid.to_vec()),
        ),
        (
            CborValue::Integer(HEADER_HS_LOGICAL_SIGNER_GROUP.into()),
            CborValue::Integer(1.into()),
        ),
    ]))
}

fn encode_claims(
    input: &AuthenticatedRequestProofInput<'_>,
    credential_hash: CredentialHash,
    request_id: RequestId,
) -> Result<Vec<u8>> {
    // Key order is 3, 4, 6, 7, -70001, -70002, -70003, -70004, -70009.
    let mut claims = vec![
        (
            CborValue::Integer(CWT_CLAIM_AUD.into()),
            CborValue::Text(input.service_domain.into()),
        ),
        (
            CborValue::Integer(CWT_CLAIM_EXP.into()),
            CborValue::Integer(input.expires_at.into()),
        ),
        (
            CborValue::Integer(CWT_CLAIM_IAT.into()),
            CborValue::Integer(input.issued_at.into()),
        ),
        (
            CborValue::Integer(CWT_CLAIM_CTI.into()),
            CborValue::Bytes(request_id.to_vec()),
        ),
        (
            CborValue::Integer(CLAIM_CREDENTIAL_HASH.into()),
            CborValue::Bytes(credential_hash.to_vec()),
        ),
        (
            CborValue::Integer(CLAIM_CAPNP_SCHEMA_ID.into()),
            CborValue::Integer(input.capnp_schema_id.into()),
        ),
        (
            CborValue::Integer(CLAIM_CAPNP_BODY_BYTES.into()),
            CborValue::Bytes(input.capnp_body.to_vec()),
        ),
        (
            CborValue::Integer(CLAIM_RESPONSE_BINDING.into()),
            response_binding_value(input.response_binding)?,
        ),
    ];
    if let Some(binding) = input.federate_recipient_binding {
        claims.push((
            CborValue::Integer(CLAIM_FEDERATE_RECIPIENT_BINDING.into()),
            federate_recipient_binding_value(binding),
        ));
    }
    encode(&CborValue::Map(claims))
}

fn federate_recipient_binding_value(binding: &FederateRecipientBinding) -> CborValue {
    let commitment_value = |value: &Option<[u8; 32]>| match value {
        Some(value) => CborValue::Bytes(value.to_vec()),
        None => CborValue::Null,
    };
    CborValue::Map(vec![
        (CborValue::Integer(1.into()), commitment_value(&binding.response)),
        (CborValue::Integer(2.into()), commitment_value(&binding.stream)),
        (
            CborValue::Integer(3.into()),
            commitment_value(&binding.client_dh_public),
        ),
    ])
}

fn response_binding_value(binding: Option<&ResponseBinding>) -> Result<CborValue> {
    let Some(binding) = binding else {
        return Ok(CborValue::Null);
    };
    let response_kind = match binding.response_kind {
        ResponseKind::Unary => 1,
        ResponseKind::StreamSetup => 2,
    };
    let protection_mode = match binding.protection_mode {
        ProtectionMode::Cleartext => 1,
        ProtectionMode::Encrypted => 2,
    };
    let recipient = match &binding.kem_recipient {
        None => CborValue::Null,
        Some(KemRecipient {
            alg,
            encapsulation_key,
            kid,
        }) => CborValue::Map(vec![
            (
                CborValue::Integer(1.into()),
                CborValue::Integer((*alg).into()),
            ),
            (
                CborValue::Integer(2.into()),
                CborValue::Bytes(encapsulation_key.clone()),
            ),
            (CborValue::Integer(3.into()), CborValue::Bytes(kid.clone())),
        ]),
    };
    let value = CborValue::Map(vec![
        (
            CborValue::Integer(1.into()),
            CborValue::Integer(binding.root_type_id.into()),
        ),
        (
            CborValue::Integer(2.into()),
            CborValue::Integer(response_kind.into()),
        ),
        (
            CborValue::Integer(3.into()),
            CborValue::Integer(protection_mode.into()),
        ),
        (CborValue::Integer(4.into()), recipient),
    ]);
    // Reuse the frozen response-binding validator rather than duplicating its
    // cross-field rules while emitting a map the parser would reject.
    ResponseBinding::decode(&value)?;
    Ok(value)
}

fn signature_structure(
    body_protected: &[u8],
    signer_protected: &[u8],
    payload: &[u8],
) -> Result<Vec<u8>> {
    // RFC 9052 Sig_structure for COSE_Sign; external_aad is explicitly empty
    // in the frozen profile.
    encode(&CborValue::Array(vec![
        CborValue::Text("Signature".into()),
        CborValue::Bytes(body_protected.to_vec()),
        CborValue::Bytes(signer_protected.to_vec()),
        CborValue::Bytes(Vec::new()),
        CborValue::Bytes(payload.to_vec()),
    ]))
}

fn signature_entry(protected: Vec<u8>, signature: Vec<u8>) -> CborValue {
    CborValue::Array(vec![
        CborValue::Bytes(protected),
        CborValue::Map(vec![]),
        CborValue::Bytes(signature),
    ])
}

fn encode(value: &CborValue) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(value, &mut out)
        .map_err(|e| anyhow::anyhow!("request proof: CBOR encode failed: {e}"))?;
    super::cbor_audit::audit_deterministic(&out)?;
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::proof::{
        enrollment::{
            ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole,
            SignerSuiteRecord,
        },
        parser::ParsedProof,
        verify::verify_proof_signatures,
        ProofDisposition,
    };

    const NOW: u64 = 1_800_000_000;
    const SCHEMA: u64 = 0xd4d0_f2a1_b3c5_8e67;

    fn fixture() -> (
        AuthenticatedHybridProofSigner,
        InMemoryEnrollmentResolver,
        ed25519_dalek::VerifyingKey,
    ) {
        let ed = ed25519_dalek::SigningKey::from_bytes(&[9; 32]);
        let pq = crate::crypto::pq::ml_dsa_sk_from_seed(&[10; 32]);
        let cnf = ed.verifying_key();
        let signer = AuthenticatedHybridProofSigner::new(
            ed,
            b"proof-fixture-ed".to_vec(),
            pq,
            b"proof-fixture-pq".to_vec(),
        )
        .unwrap();
        let mut resolver = InMemoryEnrollmentResolver::new();
        resolver
            .enrol_primary(
                &cnf,
                SignerSuiteRecord {
                    principal: "fixture-direct-self".into(),
                    suite_id: SUITE_HYBRID.into(),
                    components: vec![
                        EnrolledComponent::new(
                            b"proof-fixture-ed".to_vec(),
                            ComponentKey::Ed25519(cnf),
                        ),
                        EnrolledComponent::new(
                            b"proof-fixture-pq".to_vec(),
                            ComponentKey::MlDsa65(Box::new(crate::crypto::pq::ml_dsa_sk_to_vk(
                                &signer.ml_dsa_65,
                            ))),
                        ),
                    ],
                    epoch: 7,
                    role: SignerRole::Primary,
                    approver_role: None,
                    enrollment_policy_id: "fixture-proof-only".into(),
                    not_after: NOW + 600,
                    revoked: false,
                },
            )
            .unwrap();
        (signer, resolver, cnf)
    }

    fn input<'a>(body: &'a [u8]) -> AuthenticatedRequestProofInput<'a> {
        AuthenticatedRequestProofInput {
            service_domain: "registry.svc.hyprstream.test",
            credential: b"fixture.sender.constrained.jwt",
            issued_at: NOW - 5,
            expires_at: NOW + 30,
            capnp_schema_id: SCHEMA,
            capnp_body: body,
            response_binding: None,
            federate_recipient_binding: None,
        }
    }

    fn mutate_payload(
        bytes: &[u8],
        change: impl FnOnce(&mut Vec<(CborValue, CborValue)>),
    ) -> Vec<u8> {
        let mut outer: CborValue =
            ciborium::de::from_reader(&mut std::io::Cursor::new(bytes)).unwrap();
        let CborValue::Array(ref mut fields) = outer else {
            panic!("COSE is array");
        };
        let CborValue::Bytes(ref mut payload) = fields[2] else {
            panic!("payload is bytes");
        };
        let mut claims: CborValue =
            ciborium::de::from_reader(&mut std::io::Cursor::new(&*payload)).unwrap();
        let CborValue::Map(ref mut map) = claims else {
            panic!("claims map");
        };
        change(map);
        *payload = encode(&claims).unwrap();
        encode(&outer).unwrap()
    }

    #[test]
    fn producer_round_trips_through_real_parser_verifier_and_envelope_slot() {
        let (signer, resolver, cnf) = fixture();
        let body = b"canonical-capnp-request";
        let bytes = build_authenticated_hybrid_request_proof(&input(body), &signer).unwrap();
        let proof = ParsedProof::parse(&bytes).unwrap();
        assert_eq!(proof.disposition, ProofDisposition::Authenticated);
        assert_eq!(proof.claims.aud, "registry.svc.hyprstream.test");
        assert_eq!(proof.claims.capnp_schema_id, SCHEMA);
        assert_eq!(proof.claims.capnp_body_bytes, body);
        assert_eq!(proof.plan.groups.len(), 1);
        assert_eq!(proof.plan.groups[0].suite_id, SUITE_HYBRID);
        assert_eq!(
            proof.plan.groups[0]
                .components
                .iter()
                .map(|component| component.alg)
                .collect::<Vec<_>>(),
            vec![ALG_ED25519, ALG_ML_DSA_65],
            "WNS component order is frozen: Ed25519 then ML-DSA-65"
        );
        assert_eq!(
            proof.claims.credential_hash,
            Some(Sha256::digest(input(body).credential).into())
        );
        verify_proof_signatures(&proof, Some(&cnf), Some(&resolver), NOW).unwrap();

        let envelope = crate::envelope::RequestEnvelope::new(body.to_vec())
            .with_service_domain("registry.svc.hyprstream.test")
            .unwrap()
            .with_proof_cwt(bytes.clone());
        assert_eq!(envelope.proof_cwt.as_deref(), Some(bytes.as_slice()));
    }

    #[test]
    fn federate_recipient_extension_is_rejected_by_generic_profile_parser() {
        let (signer, _, _) = fixture();
        let suite = crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768;
        let response =
            crate::crypto::hybrid_kem::generate_recipient(suite).unwrap();
        let stream =
            crate::crypto::hybrid_kem::generate_recipient(suite).unwrap();
        let binding = FederateRecipientBinding::from_recipients(
            Some(&response.public()),
            Some(&stream.public()),
            Some([0x33; 32]),
        )
        .unwrap();
        let mut request = input(b"federate-request");
        request.federate_recipient_binding = Some(&binding);
        let bytes = build_authenticated_hybrid_request_proof(&request, &signer).unwrap();

        assert!(
            ParsedProof::parse(&bytes).is_err(),
            "generic v16 parsing must keep the frozen closed claim set"
        );
        assert!(
            ParsedProof::parse_deferred_federate_request(&bytes).is_ok(),
            "only the deferred Federate parser accepts -70009"
        );
    }

    #[test]
    fn federate_signed_proof_vector_is_stable_and_profile_isolated() {
        use crate::crypto::hybrid_kem::{SuiteId, recipient_from_seeds};

        let (signer, resolver, cnf) = fixture();
        let suite = SuiteId::HyKemX25519MlKem768;
        let response = recipient_from_seeds(suite, &[&[7; 32], &[9; 64]])
            .unwrap()
            .public();
        let stream = recipient_from_seeds(suite, &[&[8; 32], &[10; 64]])
            .unwrap()
            .public();
        let binding = FederateRecipientBinding::from_recipients(
            Some(&response),
            Some(&stream),
            Some([0x5a; 32]),
        )
        .unwrap();
        let mut request = input(b"federate-proof-vector-v1");
        request.federate_recipient_binding = Some(&binding);
        let bytes = build_authenticated_hybrid_request_proof_with_id(
            &request,
            &signer,
            [0x3c; super::super::REQUEST_ID_SIZE],
        )
        .unwrap();
        let browser_style_signer = crate::signer::LocalSigner::new(signer.ed25519.clone())
            .with_pq_key(crate::crypto::pq::ml_dsa_sk_from_seed(&[10; 32]));
        let browser_style_bytes = futures::executor::block_on(
            build_authenticated_hybrid_request_proof_with_signer_and_id(
                &request,
                &browser_style_signer,
                b"proof-fixture-ed",
                b"proof-fixture-pq",
                [0x3c; super::super::REQUEST_ID_SIZE],
            ),
        )
        .unwrap();
        assert_eq!(browser_style_bytes, bytes, "external browser signer must emit the native COSE bytes");
        let proof = ParsedProof::parse_deferred_federate_request(&bytes).unwrap();
        assert_eq!(proof.claims.federate_recipient_binding, Some(binding));
        assert!(ParsedProof::parse(&bytes).is_err());
        verify_proof_signatures(&proof, Some(&cnf), Some(&resolver), NOW).unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(&bytes)),
            "186b33dad6d4b41ed750b0d05e408dcaa78e708c33abac7bf4c2778bcf6826d6"
        );

        let omitted = mutate_payload(&bytes, |claims| {
            claims.retain(|(key, _)| {
                *key != CborValue::Integer(CLAIM_FEDERATE_RECIPIENT_BINDING.into())
            });
        });
        assert!(ParsedProof::parse_deferred_federate_request(&omitted).is_err());

        let substituted = mutate_payload(&bytes, |claims| {
            let (_, CborValue::Map(fields)) = claims
                .iter_mut()
                .find(|(key, _)| *key == CborValue::Integer(CLAIM_FEDERATE_RECIPIENT_BINDING.into()))
                .unwrap()
            else {
                panic!("Federate recipient map missing")
            };
            fields[0].1 = CborValue::Bytes(vec![0xff; 32]);
        });
        let changed = ParsedProof::parse_deferred_federate_request(&substituted).unwrap();
        assert!(!changed
            .claims
            .federate_recipient_binding
            .as_ref()
            .unwrap()
            .matches(Some(&response), Some(&stream), Some([0x5a; 32])));
        assert!(verify_proof_signatures(&changed, Some(&cnf), Some(&resolver), NOW).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn external_signer_rejects_wrong_callback_signature() {
        struct WrongSigner(crate::signer::LocalSigner);

        #[async_trait::async_trait]
        impl crate::transport_traits::Signer for WrongSigner {
            fn pubkey(&self) -> [u8; 32] {
                self.0.pubkey()
            }

            async fn sign(&self, _message: &[u8]) -> Result<[u8; 64]> {
                Ok([0; 64])
            }

            fn pq_pubkey(&self) -> Option<Vec<u8>> {
                self.0.pq_pubkey()
            }

            async fn pq_sign(&self, message: &[u8]) -> Result<Option<Vec<u8>>> {
                self.0.pq_sign(message).await
            }
        }

        let signer = WrongSigner(crate::signer::LocalSigner::new(
            ed25519_dalek::SigningKey::from_bytes(&[9; 32]),
        ).with_pq_key(crate::crypto::pq::ml_dsa_sk_from_seed(&[10; 32])));
        let binding = FederateRecipientBinding {
            response: None,
            stream: None,
            client_dh_public: None,
        };
        let mut request = input(b"callback-body");
        request.federate_recipient_binding = Some(&binding);
        let result = futures::executor::block_on(
            build_authenticated_hybrid_request_proof_with_signer_and_id(
                &request, &signer, b"proof-fixture-ed", b"proof-fixture-pq",
                [0x3c; super::super::REQUEST_ID_SIZE],
            ),
        );
        assert!(result.is_err(), "a callback that signs other bytes must fail closed");
    }

    #[test]
    fn causal_mutations_are_rejected_by_parser_or_real_verifier() {
        let (signer, resolver, cnf) = fixture();
        let bytes = build_authenticated_hybrid_request_proof(&input(b"body"), &signer).unwrap();

        let altered_body = mutate_payload(&bytes, |map| {
            map.iter_mut()
                .find(|(k, _)| *k == CborValue::Integer(CLAIM_CAPNP_BODY_BYTES.into()))
                .unwrap()
                .1 = CborValue::Bytes(b"other".to_vec());
        });
        let altered_credential = mutate_payload(&bytes, |map| {
            map.iter_mut()
                .find(|(k, _)| *k == CborValue::Integer(CLAIM_CREDENTIAL_HASH.into()))
                .unwrap()
                .1 = CborValue::Bytes([0x42; 32].to_vec());
        });
        let altered_audience = mutate_payload(&bytes, |map| {
            map.iter_mut()
                .find(|(k, _)| *k == CborValue::Integer(CWT_CLAIM_AUD.into()))
                .unwrap()
                .1 = CborValue::Text("policy.svc.hyprstream.test".into());
        });
        for altered in [altered_body, altered_credential, altered_audience] {
            let proof = ParsedProof::parse(&altered).unwrap();
            assert!(verify_proof_signatures(&proof, Some(&cnf), Some(&resolver), NOW).is_err());
        }

        let mut stripped: CborValue =
            ciborium::de::from_reader(&mut std::io::Cursor::new(&bytes)).unwrap();
        let CborValue::Array(ref mut fields) = stripped else {
            panic!("COSE array");
        };
        let CborValue::Array(ref mut signatures) = fields[3] else {
            panic!("signatures");
        };
        signatures.pop();
        assert!(ParsedProof::parse(&encode(&stripped).unwrap()).is_err());

        let proof = ParsedProof::parse(&bytes).unwrap();
        let wrong_key = ed25519_dalek::SigningKey::from_bytes(&[99; 32]).verifying_key();
        assert!(verify_proof_signatures(&proof, Some(&wrong_key), Some(&resolver), NOW).is_err());
        assert!(verify_proof_signatures(&proof, Some(&cnf), Some(&resolver), NOW + 31).is_err());
    }

    #[test]
    fn producer_rejects_oversize_but_leaves_w1_freshness_to_verifier() {
        let (signer, _, _) = fixture();
        let oversized = vec![0; MAX_BODY_BYTES + 1];
        assert!(build_authenticated_hybrid_request_proof(&input(&oversized), &signer).is_err());
    }

    #[test]
    fn producer_parser_verifier_preserve_injected_clock_w1_boundaries() {
        let cases = [
            (NOW - 30, NOW + 300, None),
            (NOW + 30, NOW + 1, None),
            (NOW, NOW, Some("proof expired:")),
            (
                NOW - 31,
                NOW + 30,
                Some("proof iat out of verifier-clock skew:"),
            ),
            (NOW, NOW + 301, Some("proof over-lifetime:")),
        ];

        for (issued_at, expires_at, expected_error) in cases {
            let (signer, resolver, cnf) = fixture();
            let mut request = input(b"w1-boundary-body");
            request.issued_at = issued_at;
            request.expires_at = expires_at;
            let bytes = build_authenticated_hybrid_request_proof(&request, &signer)
                .expect("producer accepts caller-supplied timestamps");
            let proof = ParsedProof::parse(&bytes).expect("parser accepts producer output");
            let result = verify_proof_signatures(&proof, Some(&cnf), Some(&resolver), NOW);

            match expected_error {
                None => {
                    result.expect("verifier accepts W1 boundary");
                }
                Some(expected) => {
                    let error = result
                        .expect_err("verifier rejects W1 negative")
                        .to_string();
                    assert!(
                        error.contains(expected),
                        "expected verifier reason {expected:?}, got {error:?}"
                    );
                }
            }
        }
    }
}
