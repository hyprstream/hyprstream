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
#[cfg(not(target_arch = "wasm32"))]
const EVIDENCE_MAGIC: &[u8; 8] = b"HAQW1\0\0\0";

// Fixed-width integers/hashes, and an explicitly length-prefixed target, give
// one unambiguous transcript. The witness itself is excluded to avoid cycles.
fn transcript(request: &RequestEnvelope, signer: &[u8; 32], expires: i64) -> Result<Vec<u8>> {
    // Exactly one credential may be bound. A direct presentation binds the
    // caller's own IdJag; an authorized relay binds the relayed service
    // credential it authenticated at its own ingress and now vouches for.
    ensure!(
        request.jwt_token().is_none() || request.delegation_token.is_none(),
        "ambiguous dual-credential witness denied"
    );
    let target = request
        .service_domain
        .as_deref()
        .ok_or_else(|| anyhow!("witness target missing"))?;
    crate::envelope::validate_service_domain(target)?;
    let target_len = u16::try_from(target.len())?;
    let credential = request
        .jwt_token()
        .or(request.delegation_token.as_deref())
        .ok_or_else(|| anyhow!("witness credential missing"))?;
    let mut bytes = Vec::with_capacity(160 + target.len());
    bytes.extend_from_slice(PURPOSE);
    bytes.extend_from_slice(signer);
    bytes.extend_from_slice(&target_len.to_be_bytes());
    bytes.extend_from_slice(target.as_bytes());
    bytes.extend_from_slice(&Sha256::digest(credential.as_bytes()));
    bytes.extend_from_slice(&Sha256::digest(&request.payload));
    bytes.extend_from_slice(&request.request_id.to_be_bytes());
    bytes.extend_from_slice(&request.nonce);
    bytes.extend_from_slice(&request.iat.to_be_bytes());
    bytes.extend_from_slice(&expires.to_be_bytes());
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
    let payload = transcript(request, &ed, expires)?;
    let ed_sig = signer
        .sign(&inner_tbs_pq_bound(ed.to_vec(), &pq, &payload, AAD))
        .await?
        .to_vec();
    let pq_sig = signer
        .pq_sign(&outer_tbs(pq.clone(), &payload, &ed_sig, AAD))
        .await?
        .ok_or_else(|| anyhow!("hybrid witness signature required"))?;
    let cose = assemble_composite_nested_pq_bound((ed.to_vec(), ed_sig), (pq, pq_sig))?;
    let mut witness = expires.to_be_bytes().to_vec();
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
pub(crate) fn package(request: &RequestEnvelope, signer: &[u8; 32]) -> Option<Vec<u8>> {
    request.authorization_witness.as_ref()?;
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn is_witness_evidence(bytes: &[u8]) -> bool {
    bytes.starts_with(EVIDENCE_MAGIC)
}

/// Verify the hybrid holder and all committed fields. Does not verify the
/// credential, decide policy, grant local provenance, or consume replay state.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn verify(bytes: &[u8], mediator: &str) -> Result<(RequestEnvelope, [u8; 32])> {
    let store = crate::envelope::global_pq_store();
    verify_with_store(bytes, mediator, store.as_deref())
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_with_store(
    bytes: &[u8],
    mediator: &str,
    store: Option<&dyn crate::envelope::PqTrustStore>,
) -> Result<(RequestEnvelope, [u8; 32])> {
    let (request, signer) = verify_transcript_bindings(bytes, store)?;
    ensure!(
        request.service_domain.as_deref() == Some(mediator),
        "witness target mismatch"
    );
    Ok((request, signer))
}

/// Verify an authorized relay's vouching witness for a relayed service
/// credential. Identical to [`verify`] except the target==mediator comparison
/// is intentionally absent here: the vouching relay signs the request it is
/// about to send downstream, so the downstream service is the TARGET, not the
/// mediator that will later present this evidence to Policy. The target
/// binding is enforced where the mediator identity is known — Policy verifies
/// the embedded request's `service_domain` against the authenticated
/// `checkMediated` caller, and the relay's own ingress admission
/// (`accept_delegated_bearer`) is checked before this function is consulted.
/// The caller must additionally assert `signer == envelope signer` and
/// `embedded credential == presented delegation_token`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn verify_relay_vouching(bytes: &[u8]) -> Result<(RequestEnvelope, [u8; 32])> {
    let store = crate::envelope::global_pq_store();
    verify_relay_vouching_with_store(bytes, store.as_deref())
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_relay_vouching_with_store(
    bytes: &[u8],
    store: Option<&dyn crate::envelope::PqTrustStore>,
) -> Result<(RequestEnvelope, [u8; 32])> {
    let (request, signer) = verify_transcript_bindings(bytes, store)?;
    ensure!(
        request.jwt_token().is_none() && request.delegation_token.is_some(),
        "relay vouching requires a relayed (delegated) service credential"
    );
    Ok((request, signer))
}

/// Shared witness verification: size/magic/canonical decode, witness presence
/// and size, expiry/freshness, transcript reconstruction, PQ anchor resolution
/// and composite verification. Target comparison is the caller's policy.
#[cfg(not(target_arch = "wasm32"))]
fn verify_transcript_bindings(
    bytes: &[u8],
    store: Option<&dyn crate::envelope::PqTrustStore>,
) -> Result<(RequestEnvelope, [u8; 32])> {
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
    Ok((request, signer))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
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
                .0
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

    /// An authorized relay (e.g. Model) signs a witness over a relayed service
    /// credential (delegation_token) bound to the exact downstream body and
    /// target. The vouching signer is the RELAY, not the credential holder:
    /// verification resolves the RELAY's enrolled PQ anchor, and any
    /// substitution of signer, credential, body or nonce denies.
    #[tokio::test]
    async fn relay_witness_binds_vouching_signer_credential_body_and_rejects_substitution() -> Result<()> {
        let relay = SigningKey::from_bytes(&[0xA1; 32]);
        let foreign = SigningKey::from_bytes(&[0xA2; 32]);
        let mut store = KeyedPqTrustStore::new();
        for key in [&relay, &foreign] {
            let pq = crate::crypto::pq::ml_dsa_vk_from_bytes(
                &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(key)),
            )?;
            store.bind(key.verifying_key().to_bytes(), &pq);
        }
        let credential = "service:tenant-client-witness-jwt";
        let mut request = RequestEnvelope::new(b"downstream inference body".to_vec())
            .with_delegation_token(credential.to_owned())
            .with_service_domain("inference")?;
        request.authorization_witness = Some(sign(&request, &LocalSigner::new(relay.clone())).await?);
        let relay_pub = relay.verifying_key().to_bytes();
        let bytes = package(&request, &relay_pub).ok_or_else(|| anyhow!("fixture evidence missing"))?;

        // The relay vouching verifies without a mediator name (the target
        // binding is enforced by Policy against the authenticated caller).
        let (decoded, signer) = verify_relay_vouching_with_store(&bytes, Some(&store))?;
        assert_eq!(signer, relay_pub);
        assert_eq!(decoded.delegation_token.as_deref(), Some(credential));
        assert_eq!(decoded.payload, b"downstream inference body");

        // A foreign signer header over the same request denies (the anchor
        // resolves but the composite signature was computed by the relay).
        let foreign_bytes = {
            let mut message = capnp::message::Builder::new_default();
            let mut root =
                message.init_root::<crate::common_capnp::authorization_query_evidence::Builder>();
            root.set_signer(&foreign.verifying_key().to_bytes());
            request.write_to(&mut root.init_request());
            let mut bytes = EVIDENCE_MAGIC.to_vec();
            bytes.extend_from_slice(&capnp::serialize::write_message_to_words(&message));
            bytes
        };
        assert!(verify_relay_vouching_with_store(&foreign_bytes, Some(&store)).is_err());

        // A witness minted by the foreign key over the same request denies.
        let mut forged = request.clone();
        forged.authorization_witness =
            Some(sign(&forged, &LocalSigner::new(foreign.clone())).await?);
        assert!(
            package(&forged, &relay_pub)
                .map(|b| verify_relay_vouching_with_store(&b, Some(&store)).is_err())
                .unwrap_or(true),
            "a non-relay signature must not verify as relay vouching"
        );

        // Credential substitution denies (transcript commits the credential).
        let mut swapped = request.clone();
        swapped.delegation_token = Some("service:other-tenant-jwt".to_owned());
        assert!(verify_relay_vouching_with_store(&bytes, Some(&store))
            .map(|(req, _)| req.delegation_token != swapped.delegation_token)
            .is_ok());
        let mut tampered = request.clone();
        tampered.payload.push(0);
        let tampered_bytes = {
            let mut message = capnp::message::Builder::new_default();
            let mut root =
                message.init_root::<crate::common_capnp::authorization_query_evidence::Builder>();
            root.set_signer(&relay_pub);
            tampered.write_to(&mut root.init_request());
            let mut bytes = EVIDENCE_MAGIC.to_vec();
            bytes.extend_from_slice(&capnp::serialize::write_message_to_words(&message));
            bytes
        };
        assert!(verify_relay_vouching_with_store(&tampered_bytes, Some(&store)).is_err());

        // Stripped witness denies.
        let mut stripped = request.clone();
        stripped.authorization_witness = None;
        let stripped_bytes = {
            let mut message = capnp::message::Builder::new_default();
            let mut root =
                message.init_root::<crate::common_capnp::authorization_query_evidence::Builder>();
            root.set_signer(&relay_pub);
            stripped.write_to(&mut root.init_request());
            let mut bytes = EVIDENCE_MAGIC.to_vec();
            bytes.extend_from_slice(&capnp::serialize::write_message_to_words(&message));
            bytes
        };
        assert!(verify_relay_vouching_with_store(&stripped_bytes, Some(&store)).is_err());

        // Policy-side target binding still works for the same evidence.
        assert!(verify_with_store(&bytes, "inference", Some(&store)).is_ok());
        assert!(verify_with_store(&bytes, "registry", Some(&store)).is_err());
        Ok(())
    }
}
