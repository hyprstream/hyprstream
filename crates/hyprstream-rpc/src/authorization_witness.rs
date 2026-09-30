//! Purpose-bound hybrid holder evidence for read-only Policy queries.
//! This is deliberately not a credential, envelope signature, or v16 proof.

use crate::envelope::{RequestEnvelope, MAX_TIMESTAMP_AGE_MS};
#[cfg(not(target_arch = "wasm32"))]
use crate::envelope::MAX_MEDIATED_EVIDENCE_BYTES;
use crate::transport_traits::Signer;
#[cfg(not(target_arch = "wasm32"))]
use crate::{FromCapnp, ToCapnp};
use anyhow::{anyhow, ensure, Result};
use sha2::{Digest, Sha256};

pub const MAX_WITNESS_BYTES: usize = 16 * 1024;
const AAD: &[u8] = b"hyprstream:mediated-authorization-witness:v1";
const PURPOSE: &[u8] = b"authorization-query\0v1\0";
const PURPOSE_V2: &[u8] = b"authorization-query\0v2\0";
#[cfg(not(target_arch = "wasm32"))]
const EVIDENCE_MAGIC: &[u8; 8] = b"HAQW1\0\0\0";
#[cfg(not(target_arch = "wasm32"))]
const EVIDENCE_V2_MAGIC: &[u8; 8] = b"HAQW2\0\0\0";
const WITNESS_V2_MAGIC: &[u8; 8] = b"HAQS2\0\0\0";

// Fixed-width integers/hashes, and an explicitly length-prefixed target, give
// one unambiguous transcript. The witness itself is excluded to avoid cycles.
fn transcript(request: &RequestEnvelope, signer: &[u8; 32], expires: i64) -> Result<Vec<u8>> {
    transcript_with_operation(request, signer, expires, None)
}

// The v2 operation is derived from the request leaf before the holder signs.
// It is never accepted from the mediator, and lets compact evidence retain the
// method binding without copying an arbitrarily large signed body.
fn transcript_with_operation(
    request: &RequestEnvelope,
    signer: &[u8; 32],
    expires: i64,
    operation: Option<&str>,
) -> Result<Vec<u8>> {
    ensure!(
        request.delegation_token.is_none(),
        "nested delegated witness denied"
    );
    let target = request
        .service_domain
        .as_deref()
        .ok_or_else(|| anyhow!("witness target missing"))?;
    crate::envelope::validate_service_domain(target)?;
    let credential = request
        .jwt_token()
        .ok_or_else(|| anyhow!("witness credential missing"))?;
    transcript_from_parts(
        target,
        credential,
        Sha256::digest(&request.payload).into(),
        request.request_id,
        request.nonce,
        request.iat,
        signer,
        expires,
        operation,
    )
}

#[allow(clippy::too_many_arguments)]
fn transcript_from_parts(
    target: &str,
    credential: &str,
    payload_hash: [u8; 32],
    request_id: u64,
    nonce: [u8; 16],
    iat: i64,
    signer: &[u8; 32],
    expires: i64,
    operation: Option<&str>,
) -> Result<Vec<u8>> {
    crate::envelope::validate_service_domain(target)?;
    let target_len = u16::try_from(target.len())?;
    let operation_len = operation.map_or(0, str::len);
    let mut bytes = Vec::with_capacity(160 + target.len() + operation_len);
    bytes.extend_from_slice(if operation.is_some() { PURPOSE_V2 } else { PURPOSE });
    bytes.extend_from_slice(signer);
    bytes.extend_from_slice(&target_len.to_be_bytes());
    bytes.extend_from_slice(target.as_bytes());
    bytes.extend_from_slice(&Sha256::digest(credential.as_bytes()));
    bytes.extend_from_slice(&payload_hash);
    bytes.extend_from_slice(&request_id.to_be_bytes());
    bytes.extend_from_slice(&nonce);
    bytes.extend_from_slice(&iat.to_be_bytes());
    bytes.extend_from_slice(&expires.to_be_bytes());
    if let Some(operation) = operation {
        let operation_len = u16::try_from(operation.len())?;
        bytes.extend_from_slice(&operation_len.to_be_bytes());
        bytes.extend_from_slice(operation.as_bytes());
    }
    Ok(bytes)
}

/// Sign after final payload mutation and before transport sealing. No new
/// authority is issued; only Policy's boolean-query handler understands AAD.
pub async fn sign<S: Signer>(
    request: &RequestEnvelope,
    signer: &S,
) -> Result<crate::sensitive::SensitiveBytes> {
    use crate::crypto::cose_sign::{
        assemble_composite_nested_pq_bound, inner_tbs_pq_bound, outer_tbs,
    };
    let expires = request
        .iat
        .checked_add(MAX_TIMESTAMP_AGE_MS)
        .ok_or_else(|| anyhow!("witness expiry overflow"))?;
    let ed = signer.pubkey();
    let pq = signer
        .pq_pubkey()
        .ok_or_else(|| anyhow!("hybrid witness signer required"))?;
    #[cfg(not(target_arch = "wasm32"))]
    let operation = request
        .service_domain
        .as_deref()
        .and_then(|service| crate::proof::policy::mediated_operation_for_request(service, &request.payload).ok());
    #[cfg(target_arch = "wasm32")]
    let operation: Option<String> = None;
    let payload = transcript_with_operation(request, &ed, expires, operation.as_deref())?;
    let ed_sig = signer
        .sign(&inner_tbs_pq_bound(ed.to_vec(), &pq, &payload, AAD))
        .await?
        .to_vec();
    let pq_sig = signer
        .pq_sign(&outer_tbs(pq.clone(), &payload, &ed_sig, AAD))
        .await?
        .ok_or_else(|| anyhow!("hybrid witness signature required"))?;
    let cose = assemble_composite_nested_pq_bound((ed.to_vec(), ed_sig), (pq, pq_sig))?;
    let mut witness = if operation.is_some() { WITNESS_V2_MAGIC.to_vec() } else { Vec::new() };
    witness.extend_from_slice(&expires.to_be_bytes());
    witness.extend_from_slice(&cose);
    ensure!(
        witness.len() <= MAX_WITNESS_BYTES,
        "authorization witness too large"
    );
    Ok(witness.into())
}

/// Retain a decrypted request only when it carries independently verifiable
/// holder evidence. This packages bytes; Policy still verifies every binding.
#[cfg(not(target_arch = "wasm32"))]
pub fn package(request: &RequestEnvelope, signer: &[u8; 32]) -> Option<Vec<u8>> {
    let witness = request.authorization_witness.as_ref()?;
    if witness.starts_with(WITNESS_V2_MAGIC) {
        return package_v2(request, signer);
    }
    if !crate::envelope::mediated_request_fits(request) {
        return None;
    }
    let mut message = capnp::message::Builder::new_default();
    let mut root =
        message.init_root::<crate::common_capnp::authorization_query_evidence::Builder>();
    root.set_signer(signer);
    request.write_to(&mut root.init_request());
    let mut bytes = EVIDENCE_MAGIC.to_vec();
    bytes.extend_from_slice(&capnp::serialize::write_message_to_words(&message));
    (bytes.len() <= MAX_MEDIATED_EVIDENCE_BYTES).then_some(bytes)
}

/// Compact v2 evidence retains the holder-signed body digest and derived
/// method action, rather than retaining the body itself. This is required for
/// valid large requests while keeping the mediation frame bounded.
#[cfg(not(target_arch = "wasm32"))]
fn package_v2(request: &RequestEnvelope, signer: &[u8; 32]) -> Option<Vec<u8>> {
    let target = request.service_domain.as_deref()?;
    let operation = crate::proof::policy::mediated_operation_for_request(target, &request.payload).ok()?;
    package_v2_for_operation(request, signer, &operation)
}

#[cfg(not(target_arch = "wasm32"))]
fn package_v2_for_operation(
    request: &RequestEnvelope,
    signer: &[u8; 32],
    operation: &str,
) -> Option<Vec<u8>> {
    let target = request.service_domain.as_deref()?;
    crate::envelope::validate_service_domain(target).ok()?;
    let credential = request.jwt_token()?;
    let witness = request.authorization_witness.as_ref()?;
    if request.delegation_token.is_some()
        || !witness.starts_with(WITNESS_V2_MAGIC)
        || target.len() > u16::MAX as usize
        || credential.len() > u32::MAX as usize
        || operation.len() > u16::MAX as usize
        || witness.len() > u16::MAX as usize
    {
        return None;
    }

    let mut bytes = Vec::with_capacity(
        EVIDENCE_V2_MAGIC.len() + signer.len() + 2 + target.len() + 4 + credential.len()
            + 32 + 8 + 16 + 8 + 2 + operation.len() + 2 + witness.len(),
    );
    bytes.extend_from_slice(EVIDENCE_V2_MAGIC);
    bytes.extend_from_slice(signer);
    bytes.extend_from_slice(&(target.len() as u16).to_be_bytes());
    bytes.extend_from_slice(target.as_bytes());
    bytes.extend_from_slice(&(credential.len() as u32).to_be_bytes());
    bytes.extend_from_slice(credential.as_bytes());
    bytes.extend_from_slice(&Sha256::digest(&request.payload));
    bytes.extend_from_slice(&request.request_id.to_be_bytes());
    bytes.extend_from_slice(&request.nonce);
    bytes.extend_from_slice(&request.iat.to_be_bytes());
    bytes.extend_from_slice(&(operation.len() as u16).to_be_bytes());
    bytes.extend_from_slice(operation.as_bytes());
    bytes.extend_from_slice(&(witness.len() as u16).to_be_bytes());
    bytes.extend_from_slice(witness);
    (bytes.len() <= MAX_MEDIATED_EVIDENCE_BYTES).then_some(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn is_witness_evidence(bytes: &[u8]) -> bool {
    bytes.starts_with(EVIDENCE_MAGIC) || bytes.starts_with(EVIDENCE_V2_MAGIC)
}

/// Verify the hybrid holder and all committed fields. Does not verify the
/// credential, decide policy, grant local provenance, or consume replay state.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct VerifiedWitness {
    pub(crate) request: RequestEnvelope,
    pub(crate) signer: [u8; 32],
    /// A v2 action is holder-signed and derived from the original request
    /// leaf. A v1 witness retains the original body and derives it on verify.
    pub(crate) operation: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn verify(bytes: &[u8], mediator: &str) -> Result<VerifiedWitness> {
    if bytes.starts_with(EVIDENCE_V2_MAGIC) {
        return verify_v2(bytes, mediator);
    }
    verify_v1(bytes, mediator)
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_v1(bytes: &[u8], mediator: &str) -> Result<VerifiedWitness> {
    let store = crate::envelope::global_pq_store();
    verify_with_store(bytes, mediator, store.as_deref())
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_with_store(
    bytes: &[u8],
    mediator: &str,
    store: Option<&dyn crate::envelope::PqTrustStore>,
) -> Result<VerifiedWitness> {
    ensure!(
        bytes.len() <= MAX_MEDIATED_EVIDENCE_BYTES,
        "mediated evidence too large"
    );
    let mut remaining = bytes
        .strip_prefix(EVIDENCE_MAGIC)
        .ok_or_else(|| anyhow!("wrong witness evidence type"))?;
    let mut options = capnp::message::ReaderOptions::new();
    options.traversal_limit_in_words(Some(MAX_MEDIATED_EVIDENCE_BYTES / 8));
    options.nesting_limit(16);
    let message = capnp::serialize::read_message_from_flat_slice(&mut remaining, options)?;
    ensure!(remaining.is_empty(), "trailing witness evidence");
    let root = message.get_root::<crate::common_capnp::authorization_query_evidence::Reader>()?;
    let signer: [u8; 32] = root.get_signer()?.try_into()?;
    let request = RequestEnvelope::read_from(root.get_request()?)?;
    ensure!(
        request.service_domain.as_deref() == Some(mediator),
        "witness target mismatch"
    );
    let witness = request
        .authorization_witness
        .as_ref()
        .ok_or_else(|| anyhow!("authorization witness missing"))?;
    ensure!(
        witness.len() > 8 && witness.len() <= MAX_WITNESS_BYTES,
        "invalid authorization witness size"
    );
    let expires = i64::from_be_bytes(witness[..8].try_into()?);
    let now = crate::envelope::current_timestamp();
    ensure!(
        expires
            == request
                .iat
                .checked_add(MAX_TIMESTAMP_AGE_MS)
                .ok_or_else(|| anyhow!("witness expiry overflow"))?
            && now <= expires
            && i128::from(request.iat) - i128::from(now) <= 30_000,
        "authorization witness expired or outside freshness window"
    );
    let payload = transcript(&request, &signer, expires)?;
    let store = store.ok_or_else(|| anyhow!("witness PQ anchors missing"))?;
    let pq = store
        .ml_dsa_key_for(&signer)
        .ok_or_else(|| anyhow!("witness holder PQ anchor missing"))?;
    let ed = ed25519_dalek::VerifyingKey::from_bytes(&signer)?;
    crate::crypto::cose_sign::verify_composite(&witness[8..], &ed, Some(&pq), &payload, AAD, true)?;
    Ok(VerifiedWitness { request, signer, operation: None })
}

#[cfg(not(target_arch = "wasm32"))]
fn take_bytes<'a>(remaining: &mut &'a [u8], length: usize) -> Result<&'a [u8]> {
    ensure!(length <= remaining.len(), "truncated compact witness evidence");
    let (value, rest) = remaining.split_at(length);
    *remaining = rest;
    Ok(value)
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_v2(bytes: &[u8], mediator: &str) -> Result<VerifiedWitness> {
    let store = crate::envelope::global_pq_store()
        .ok_or_else(|| anyhow!("witness PQ anchors missing"))?;
    verify_v2_with_store(bytes, mediator, Some(store.as_ref()))
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_v2_with_store(
    bytes: &[u8],
    mediator: &str,
    store: Option<&dyn crate::envelope::PqTrustStore>,
) -> Result<VerifiedWitness> {
    ensure!(bytes.len() <= MAX_MEDIATED_EVIDENCE_BYTES, "mediated evidence too large");
    let mut remaining = bytes
        .strip_prefix(EVIDENCE_V2_MAGIC)
        .ok_or_else(|| anyhow!("wrong compact witness evidence type"))?;
    let signer: [u8; 32] = take_bytes(&mut remaining, 32)?.try_into()?;
    let target_len = u16::from_be_bytes(take_bytes(&mut remaining, 2)?.try_into()?) as usize;
    let target = std::str::from_utf8(take_bytes(&mut remaining, target_len)?)?;
    crate::envelope::validate_service_domain(target)?;
    ensure!(target == mediator, "witness target mismatch");
    let credential_len = u32::from_be_bytes(take_bytes(&mut remaining, 4)?.try_into()?) as usize;
    let credential = std::str::from_utf8(take_bytes(&mut remaining, credential_len)?)?;
    ensure!(!credential.is_empty(), "witness credential missing");
    let payload_hash: [u8; 32] = take_bytes(&mut remaining, 32)?.try_into()?;
    let request_id = u64::from_be_bytes(take_bytes(&mut remaining, 8)?.try_into()?);
    let nonce: [u8; 16] = take_bytes(&mut remaining, 16)?.try_into()?;
    let iat = i64::from_be_bytes(take_bytes(&mut remaining, 8)?.try_into()?);
    let operation_len = u16::from_be_bytes(take_bytes(&mut remaining, 2)?.try_into()?) as usize;
    let operation = std::str::from_utf8(take_bytes(&mut remaining, operation_len)?)?;
    ensure!(!operation.is_empty() && operation.len() <= 128, "invalid mediated operation");
    let witness_len = u16::from_be_bytes(take_bytes(&mut remaining, 2)?.try_into()?) as usize;
    let witness = take_bytes(&mut remaining, witness_len)?;
    ensure!(remaining.is_empty(), "trailing compact witness evidence");
    ensure!(
        witness.starts_with(WITNESS_V2_MAGIC)
            && witness.len() > WITNESS_V2_MAGIC.len() + 8
            && witness.len() <= MAX_WITNESS_BYTES,
        "invalid compact authorization witness size"
    );
    let expires = i64::from_be_bytes(
        witness[WITNESS_V2_MAGIC.len()..WITNESS_V2_MAGIC.len() + 8].try_into()?,
    );
    let now = crate::envelope::current_timestamp();
    ensure!(
        expires == iat.checked_add(MAX_TIMESTAMP_AGE_MS).ok_or_else(|| anyhow!("witness expiry overflow"))?
            && now <= expires
            && i128::from(iat) - i128::from(now) <= 30_000,
        "authorization witness expired or outside freshness window"
    );
    let payload = transcript_from_parts(
        target, credential, payload_hash, request_id, nonce, iat, &signer, expires, Some(operation),
    )?;
    let store = store.ok_or_else(|| anyhow!("witness PQ anchors missing"))?;
    let pq = store
        .ml_dsa_key_for(&signer)
        .ok_or_else(|| anyhow!("witness holder PQ anchor missing"))?;
    let ed = ed25519_dalek::VerifyingKey::from_bytes(&signer)?;
    crate::crypto::cose_sign::verify_composite(
        &witness[WITNESS_V2_MAGIC.len() + 8..], &ed, Some(&pq), &payload, AAD, true,
    )?;
    let mut request = RequestEnvelope::new(Vec::new())
        .with_jwt_token(credential.to_owned())
        .with_service_domain(target.to_owned())?;
    request.request_id = request_id;
    request.nonce = nonce;
    request.iat = iat;
    request.authorization_witness = Some(witness.to_vec().into());
    Ok(VerifiedWitness { request, signer, operation: Some(operation.to_owned()) })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::envelope::{KeyedPqTrustStore, SignedEnvelope};
    use crate::node_identity::derive_mesh_mldsa_key;
    use crate::signer::LocalSigner;

    #[tokio::test]
    async fn witness_binds_holder_target_body_credential_freshness_and_protocol() -> Result<()> {
        let key = ed25519_dalek::SigningKey::from_bytes(&[0x91; 32]);
        let other = ed25519_dalek::SigningKey::from_bytes(&[0x92; 32]);
        let signer = LocalSigner::new(key.clone());
        let mut store = KeyedPqTrustStore::new();
        for key in [&key, &other] {
            let pq = crate::crypto::pq::ml_dsa_vk_from_bytes(
                &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(key)),
            )?;
            store.bind(key.verifying_key().to_bytes(), &pq);
        }
        let mut request = RequestEnvelope::new(b"exact dispatched body".to_vec())
            .with_jwt_token("credential-placeholder".to_owned())
            .with_service_domain("registry")?;
        request.authorization_witness = Some(sign(&request, &signer).await?);
        let encode = |request: &RequestEnvelope, signer: &[u8; 32]| {
            let mut message = capnp::message::Builder::new_default();
            let mut root =
                message.init_root::<crate::common_capnp::authorization_query_evidence::Builder>();
            root.set_signer(signer);
            request.write_to(&mut root.init_request());
            let mut bytes = EVIDENCE_MAGIC.to_vec();
            bytes.extend_from_slice(&capnp::serialize::write_message_to_words(&message));
            bytes
        };
        let holder = signer.pubkey();
        let bytes =
            package(&request, &holder).ok_or_else(|| anyhow!("fixture evidence missing"))?;
        assert_eq!(
            verify_with_store(&bytes, "registry", Some(&store))?
                .request
                .payload,
            request.payload
        );
        assert!(verify_with_store(&bytes, "model", Some(&store)).is_err());
        assert!(verify_with_store(&bytes, "registry", None).is_err());
        assert!(verify_with_store(&bytes, "registry", Some(&KeyedPqTrustStore::new())).is_err());
        assert!(verify_with_store(
            &encode(&request, &other.verifying_key().to_bytes()),
            "registry",
            Some(&store)
        )
        .is_err());

        let mut changed = request.clone();
        changed.payload.push(0);
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request
            .clone()
            .with_jwt_token("another-credential".to_owned());
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request.clone().with_service_domain("model")?;
        // Even a mediator whose identity matches the changed target cannot
        // repurpose the original Registry witness.
        assert!(verify_with_store(&encode(&changed, &holder), "model", Some(&store)).is_err());
        changed = request.clone();
        changed.nonce[0] ^= 1;
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request.clone();
        changed.request_id += 1;
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request.clone();
        let mut corrupted = changed
            .authorization_witness
            .as_ref()
            .ok_or_else(|| anyhow!("fixture witness missing"))?
            .to_vec();
        let last = corrupted
            .last_mut()
            .ok_or_else(|| anyhow!("empty witness"))?;
        *last ^= 1;
        changed.authorization_witness = Some(corrupted.into());
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request.clone();
        changed.iat += 60_000;
        changed.authorization_witness = Some(sign(&changed, &signer).await?);
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request.clone();
        changed.iat -= MAX_TIMESTAMP_AGE_MS + 60_000;
        changed.authorization_witness = Some(sign(&changed, &signer).await?);
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        changed = request.clone();
        changed.authorization_witness = None;
        assert!(package(&changed, &holder).is_none());
        assert!(verify_with_store(&encode(&changed, &holder), "registry", Some(&store)).is_err());
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[0; 8]);
        assert!(verify_with_store(&trailing, "registry", Some(&store)).is_err());
        assert!(verify_with_store(
            &vec![0; MAX_MEDIATED_EVIDENCE_BYTES + 1],
            "registry",
            Some(&store)
        )
        .is_err());

        // The purpose-separated signature cannot be transplanted into an
        // ordinary RPC envelope or parsed as a v16 dispatch proof/credential.
        let witness = request
            .authorization_witness
            .as_ref()
            .ok_or_else(|| anyhow!("fixture witness missing"))?;
        let mut envelope =
            SignedEnvelope::new_signed_hybrid(request.clone(), &key, &derive_mesh_mldsa_key(&key));
        envelope.cose = witness[8..].to_vec();
        assert!(envelope
            .verify_signature_only_with(
                &key.verifying_key(),
                Some(&store),
                crate::crypto::CryptoPolicy::Hybrid
            )
            .is_err());
        assert!(crate::proof::parser::ParsedProof::parse(witness).is_err());
        assert!(crate::auth::decode_unverified(&String::from_utf8_lossy(witness)).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn compact_witness_keeps_large_body_holder_and_operation_bindings() -> Result<()> {
        use crate::crypto::cose_sign::{assemble_composite_nested_pq_bound, inner_tbs_pq_bound, outer_tbs};

        let key = ed25519_dalek::SigningKey::from_bytes(&[0x93; 32]);
        let signer = LocalSigner::new(key.clone());
        let mut store = KeyedPqTrustStore::new();
        let pq = crate::crypto::pq::ml_dsa_vk_from_bytes(
            &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(&key)),
        )?;
        store.bind(key.verifying_key().to_bytes(), &pq);

        let mut request = RequestEnvelope::new(vec![0xA5; MAX_MEDIATED_EVIDENCE_BYTES + 1])
            .with_jwt_token("credential-placeholder".to_owned())
            .with_service_domain("registry")?;
        let operation = "query";
        let expires = request.iat.checked_add(MAX_TIMESTAMP_AGE_MS)
            .ok_or_else(|| anyhow!("fixture expiry overflow"))?;
        let ed = signer.pubkey();
        let pq = signer.pq_pubkey().ok_or_else(|| anyhow!("fixture PQ key missing"))?;
        let payload = transcript_with_operation(&request, &ed, expires, Some(operation))?;
        let ed_sig = signer.sign(&inner_tbs_pq_bound(ed.to_vec(), &pq, &payload, AAD)).await?.to_vec();
        let pq_sig = signer.pq_sign(&outer_tbs(pq.clone(), &payload, &ed_sig, AAD)).await?
            .ok_or_else(|| anyhow!("fixture PQ signature missing"))?;
        let cose = assemble_composite_nested_pq_bound((ed.to_vec(), ed_sig), (pq, pq_sig))?;
        let mut witness = WITNESS_V2_MAGIC.to_vec();
        witness.extend_from_slice(&expires.to_be_bytes());
        witness.extend_from_slice(&cose);
        request.authorization_witness = Some(witness.into());

        let evidence = package_v2_for_operation(&request, &ed, operation)
            .ok_or_else(|| anyhow!("large compact witness was not retained"))?;
        assert!(evidence.len() <= MAX_MEDIATED_EVIDENCE_BYTES);
        let verified = verify_v2_with_store(&evidence, "registry", Some(&store))?;
        assert_eq!(verified.operation.as_deref(), Some(operation));
        assert_eq!(verified.request.payload, Vec::<u8>::new(), "the large body is never copied into mediation evidence");
        assert_eq!(verified.request.request_id, request.request_id);

        let payload_hash_offset = EVIDENCE_V2_MAGIC.len() + 32 + 2 + "registry".len()
            + 4 + "credential-placeholder".len();
        let mut changed_hash = evidence.clone();
        changed_hash[payload_hash_offset] ^= 1;
        assert!(verify_v2_with_store(&changed_hash, "registry", Some(&store)).is_err());
        let operation_offset = payload_hash_offset + 32 + 8 + 16 + 8 + 2;
        let mut changed_operation = evidence;
        changed_operation[operation_offset] = b'w';
        assert!(verify_v2_with_store(&changed_operation, "registry", Some(&store)).is_err());
        Ok(())
    }
}
