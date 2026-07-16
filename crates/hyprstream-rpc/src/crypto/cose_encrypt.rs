//! Canonical COSE_Encrypt0 sealing over Hyprstream's pinned hybrid-KEM suite.
//!
//! This module implements the reusable confidentiality substrate for #553.  The
//! outer object and its authenticated-data construction are COSE_Encrypt0 from
//! RFC 9052, and the content algorithm is the registered ChaCha20/Poly1305 COSE
//! algorithm (24).  Key establishment is Hyprstream's suite-identified N-KEM
//! combiner in [`crate::crypto::hybrid_kem`].
//!
//! This is deliberately **not** called HPKE or COSE-HPKE.  Hyprstream's N-KEM
//! combiner is not an HPKE ciphersuite, and its composite encapsulation material
//! is not the `enc` / `ek` value defined by `draft-ietf-cose-hpke`.  The private
//! labels below define a project-local profile and make no draft-interoperability
//! claim.  PQUIP / RFC 9794 supplies transition terminology, not wire semantics.
//!
//! New consumers should use [`seal`], [`open`], [`seal_prederived`], and
//! [`open_prederived`] with an owning [`NonceLedger`].  The legacy wrappers at
//! the bottom remain only for the already-landed envelope path, whose replay
//! lifecycle is owned by `SignedEnvelope`; they must not be used for new stream
//! or group lifecycles.

use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use coset::cbor::value::Value as CborValue;
use coset::{
    iana::{self, EnumI64},
    CborSerializable, CoseEncrypt0, CoseEncrypt0Builder, Header, HeaderBuilder, Label,
    RegisteredLabelWithPrivate,
};
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::crypto::hybrid_kem::{
    self, HybridKemMaterial, RecipientKeypair, RecipientPublic, SuiteId,
};

/// Maximum plaintext accepted by this substrate (16 MiB).
pub const MAX_PLAINTEXT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum caller-supplied external AAD / transcript bytes (64 KiB).
pub const MAX_EXTERNAL_AAD_BYTES: usize = 64 * 1024;
/// Maximum canonical COSE object accepted before parsing or KEM work.
pub const MAX_COSE_BYTES: usize = MAX_PLAINTEXT_BYTES + 16 * 1024;
/// Maximum recipient or key identifier length.
pub const MAX_IDENTIFIER_BYTES: usize = 256;
/// Maximum number of nonce/replay entries retained by one ledger.
pub const MAX_LEDGER_ENTRIES: usize = 1_048_576;

const ALG_CHACHA20POLY1305: i64 = iana::Algorithm::ChaCha20Poly1305 as i64;

// Project-local private-use labels.  They intentionally do not reuse the
// provisional `ek` label or algorithm assignments from draft-ietf-cose-hpke.
const HDR_SUITE_ID: i64 = -65538;
const HDR_KEM_MATERIAL: i64 = -65539;
const HDR_RECIPIENT_ID: i64 = -65540;
const HDR_NONCE_DOMAIN: i64 = -65541;
const HDR_EPOCH: i64 = -65542;
const HDR_SEQUENCE: i64 = -65543;

const KDF_KEY_LABEL: &[u8] = b"hyprstream cose-encrypt0 base key v2";
const KDF_CONTEXT_LABEL: &[u8] = b"hyprstream cose-encrypt0 context key v2";
const COMPAT_RECIPIENT_LABEL: &[u8] = b"hyprstream cose-encrypt0 compat recipient v2";
const COMPAT_KEY_LABEL: &[u8] = b"hyprstream cose-encrypt0 compat key v2";
const COMPAT_DOMAIN_LABEL: &[u8] = b"hyprstream cose-encrypt0 compat domain v2";

/// Authenticated identity and nonce coordinates for one sealed object.
///
/// `nonce_domain` is a lifecycle-generated, collision-resistant identifier for
/// one direction/track/key domain.  The owner must not reuse it when restarting
/// a lifecycle with the same key.  Epoch installation and retirement policy is
/// intentionally outside #553 and belongs to #554/#555.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealContext {
    /// Canonical recipient identity (for example, an accepted service DID).
    pub recipient_id: Vec<u8>,
    /// Selected recipient key identifier.
    pub key_id: Vec<u8>,
    /// Collision-resistant lifecycle/direction/track nonce domain.
    pub nonce_domain: [u8; 32],
    /// Installed epoch.  It is encoded as a u32 in the 96-bit nonce.
    pub epoch: u32,
    /// Monotonic sequence within the epoch.
    pub sequence: u64,
}

impl SealContext {
    fn validate(&self) -> Result<()> {
        validate_identifier("recipient id", &self.recipient_id)?;
        validate_identifier("key id", &self.key_id)?;
        if self.nonce_domain == [0; 32] {
            bail!("nonce domain must not be all zero");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct NonceUse {
    suite: u16,
    recipient_id: Vec<u8>,
    key_id: Vec<u8>,
    nonce_domain: [u8; 32],
    epoch: u32,
    sequence: u64,
}

impl NonceUse {
    fn new(suite: SuiteId, context: &SealContext) -> Self {
        Self {
            suite: suite.as_u16(),
            recipient_id: context.recipient_id.clone(),
            key_id: context.key_id.clone(),
            nonce_domain: context.nonce_domain,
            epoch: context.epoch,
            sequence: context.sequence,
        }
    }
}

/// Bounded owning state for outbound nonce-use and inbound replay rejection.
///
/// A ledger is directional: do not share the same instance between a sender and
/// receiver.  `seal*` burns a coordinate before cryptographic work, so even an
/// error cannot lead to an uncertain nonce being reused.  `open*` records a
/// coordinate only after successful authentication, preventing invalid packets
/// from consuming the replay budget.
#[derive(Debug)]
pub struct NonceLedger {
    capacity: usize,
    seen: HashSet<NonceUse>,
}

impl NonceLedger {
    /// Construct a ledger with a finite non-zero capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > MAX_LEDGER_ENTRIES {
            bail!("nonce ledger capacity must be in 1..={MAX_LEDGER_ENTRIES}");
        }
        Ok(Self {
            capacity,
            seen: HashSet::with_capacity(capacity.min(4096)),
        })
    }

    /// Number of retained nonce/replay coordinates.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether no coordinates are retained.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Retire one epoch after the owning lifecycle has made it unacceptable.
    ///
    /// Removing replay state is safe only after the lifecycle rejects that epoch
    /// before calling `open`; this method does not itself implement grace policy.
    pub fn retire_epoch(
        &mut self,
        suite: SuiteId,
        recipient_id: &[u8],
        key_id: &[u8],
        nonce_domain: &[u8; 32],
        epoch: u32,
    ) {
        self.seen.retain(|entry| {
            !(entry.suite == suite.as_u16()
                && entry.recipient_id == recipient_id
                && entry.key_id == key_id
                && &entry.nonce_domain == nonce_domain
                && entry.epoch == epoch)
        });
    }

    fn ensure_unseen(&self, usage: &NonceUse) -> Result<()> {
        if self.seen.contains(usage) {
            bail!("nonce coordinate already used or ciphertext replayed");
        }
        Ok(())
    }

    fn record(&mut self, usage: NonceUse) -> Result<()> {
        self.ensure_unseen(&usage)?;
        if self.seen.len() >= self.capacity {
            bail!("nonce ledger capacity exhausted; rotate/retire at lifecycle boundary");
        }
        self.seen.insert(usage);
        Ok(())
    }
}

/// Derive a separated base content key from a hybrid combiner secret.
pub fn derive_aead_key(hybrid_secret: &[u8; 32]) -> Zeroizing<[u8; 32]> {
    let hk = Hkdf::<Sha256>::new(None, hybrid_secret);
    let mut key = [0u8; 32];
    // SHA-256's HKDF output limit is far above 32 bytes.
    if hk.expand(KDF_KEY_LABEL, &mut key).is_err() {
        unreachable!("HKDF-SHA256 cannot reject a 32-byte output");
    }
    Zeroizing::new(key)
}

fn validate_identifier(name: &str, value: &[u8]) -> Result<()> {
    if value.is_empty() || value.len() > MAX_IDENTIFIER_BYTES {
        bail!("{name} length must be in 1..={MAX_IDENTIFIER_BYTES}");
    }
    Ok(())
}

fn validate_inputs(context: &SealContext, plaintext_len: usize, external_aad: &[u8]) -> Result<()> {
    context.validate()?;
    if plaintext_len > MAX_PLAINTEXT_BYTES {
        bail!("plaintext exceeds {MAX_PLAINTEXT_BYTES}-byte limit");
    }
    if external_aad.len() > MAX_EXTERNAL_AAD_BYTES {
        bail!("external AAD exceeds {MAX_EXTERNAL_AAD_BYTES}-byte limit");
    }
    Ok(())
}

fn derive_context_key(
    base_key: &[u8; 32],
    suite: SuiteId,
    context: &SealContext,
) -> Result<Zeroizing<[u8; 32]>> {
    context.validate()?;
    let mut info = Vec::with_capacity(
        KDF_CONTEXT_LABEL.len() + context.recipient_id.len() + context.key_id.len() + 80,
    );
    append_lp(&mut info, KDF_CONTEXT_LABEL)?;
    info.extend_from_slice(&suite.as_u16().to_be_bytes());
    append_lp(&mut info, &context.recipient_id)?;
    append_lp(&mut info, &context.key_id)?;
    info.extend_from_slice(&context.nonce_domain);
    info.extend_from_slice(&context.epoch.to_be_bytes());

    let hk =
        Hkdf::<Sha256>::from_prk(base_key).map_err(|_| anyhow::anyhow!("invalid HKDF base key"))?;
    let mut key = [0u8; 32];
    hk.expand(&info, &mut key)
        .map_err(|_| anyhow::anyhow!("COSE context KDF info too long"))?;
    Ok(Zeroizing::new(key))
}

fn append_lp(out: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    let len = u32::try_from(value.len()).context("context field too long")?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn derive_nonce(context: &SealContext) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[..4].copy_from_slice(&context.epoch.to_be_bytes());
    nonce[4..].copy_from_slice(&context.sequence.to_be_bytes());
    nonce
}

fn aead_seal(key: &[u8; 32], nonce: &[u8; 12], plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    ChaCha20Poly1305::new(Key::from_slice(key))
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("ChaCha20Poly1305 encrypt failed"))
}

fn aead_open(key: &[u8; 32], nonce: &[u8; 12], ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    ChaCha20Poly1305::new(Key::from_slice(key))
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("ChaCha20Poly1305 decrypt/auth failed"))
}

fn private_label(value: i64) -> RegisteredLabelWithPrivate<iana::HeaderParameter> {
    RegisteredLabelWithPrivate::PrivateUse(value)
}

fn critical_headers() -> Vec<RegisteredLabelWithPrivate<iana::HeaderParameter>> {
    vec![
        private_label(HDR_SUITE_ID),
        private_label(HDR_RECIPIENT_ID),
        private_label(HDR_NONCE_DOMAIN),
        private_label(HDR_EPOCH),
        private_label(HDR_SEQUENCE),
    ]
}

fn build_protected(suite: SuiteId, context: &SealContext) -> Header {
    HeaderBuilder::new()
        .algorithm(iana::Algorithm::ChaCha20Poly1305)
        .key_id(context.key_id.clone())
        .add_critical_label(private_label(HDR_SUITE_ID))
        .add_critical_label(private_label(HDR_RECIPIENT_ID))
        .add_critical_label(private_label(HDR_NONCE_DOMAIN))
        .add_critical_label(private_label(HDR_EPOCH))
        .add_critical_label(private_label(HDR_SEQUENCE))
        .value(HDR_SUITE_ID, CborValue::Integer(suite.as_u16().into()))
        .value(
            HDR_RECIPIENT_ID,
            CborValue::Bytes(context.recipient_id.clone()),
        )
        .value(
            HDR_NONCE_DOMAIN,
            CborValue::Bytes(context.nonce_domain.to_vec()),
        )
        .value(HDR_EPOCH, CborValue::Integer(context.epoch.into()))
        .value(HDR_SEQUENCE, CborValue::Integer(context.sequence.into()))
        .build()
}

fn empty_except_rest(header: &Header) -> bool {
    header.alg.is_none()
        && header.crit.is_empty()
        && header.content_type.is_none()
        && header.key_id.is_empty()
        && header.iv.is_empty()
        && header.partial_iv.is_empty()
        && header.counter_signatures.is_empty()
}

fn require_exact_protected(
    enc: &CoseEncrypt0,
    expected_suite: SuiteId,
    expected: &SealContext,
) -> Result<()> {
    let header = &enc.protected.header;
    match header.alg.as_ref() {
        Some(coset::Algorithm::Assigned(a)) if a.to_i64() == ALG_CHACHA20POLY1305 => {}
        Some(coset::Algorithm::Assigned(a)) => {
            bail!(
                "COSE_Encrypt0 algorithm {} is not ChaCha20Poly1305 (24)",
                a.to_i64()
            )
        }
        Some(coset::Algorithm::PrivateUse(value)) => {
            bail!("private-use COSE algorithm {value} is not accepted")
        }
        Some(coset::Algorithm::Text(value)) => bail!("text COSE algorithm {value} is unsupported"),
        None => bail!("COSE_Encrypt0 missing protected algorithm"),
    }
    if header.crit != critical_headers() {
        bail!("COSE_Encrypt0 critical-header set/order is not canonical");
    }
    if header.key_id != expected.key_id {
        bail!("COSE_Encrypt0 recipient key id mismatch");
    }
    if header.content_type.is_some()
        || !header.iv.is_empty()
        || !header.partial_iv.is_empty()
        || !header.counter_signatures.is_empty()
    {
        bail!("COSE_Encrypt0 contains unsupported protected headers");
    }
    if header.rest.len() != 5 {
        bail!("COSE_Encrypt0 protected private-header set is not exact");
    }

    require_int(
        header,
        HDR_SUITE_ID,
        i128::from(expected_suite.as_u16()),
        "suite",
    )?;
    require_bytes(
        header,
        HDR_RECIPIENT_ID,
        &expected.recipient_id,
        "recipient id",
    )?;
    require_bytes(
        header,
        HDR_NONCE_DOMAIN,
        &expected.nonce_domain,
        "nonce domain",
    )?;
    require_int(header, HDR_EPOCH, i128::from(expected.epoch), "epoch")?;
    require_int(
        header,
        HDR_SEQUENCE,
        i128::from(expected.sequence),
        "sequence",
    )?;
    Ok(())
}

fn require_int(header: &Header, label: i64, expected: i128, name: &str) -> Result<()> {
    match unique_header_value(header, label)? {
        CborValue::Integer(value) if i128::from(*value) == expected => Ok(()),
        CborValue::Integer(value) => {
            let actual = i128::from(*value);
            bail!("COSE_Encrypt0 {name} mismatch: {actual} != {expected}")
        }
        _ => bail!("COSE_Encrypt0 {name} header has wrong type"),
    }
}

fn require_bytes(header: &Header, label: i64, expected: &[u8], name: &str) -> Result<()> {
    match unique_header_value(header, label)? {
        CborValue::Bytes(value) if value == expected => Ok(()),
        CborValue::Bytes(_) => bail!("COSE_Encrypt0 {name} mismatch"),
        _ => bail!("COSE_Encrypt0 {name} header has wrong type"),
    }
}

fn unique_header_value(header: &Header, label: i64) -> Result<&CborValue> {
    let mut values = header.rest.iter().filter_map(|(candidate, value)| {
        if candidate == &Label::Int(label) {
            Some(value)
        } else {
            None
        }
    });
    let value = values
        .next()
        .ok_or_else(|| anyhow::anyhow!("COSE_Encrypt0 missing private header {label}"))?;
    if values.next().is_some() {
        bail!("COSE_Encrypt0 duplicate private header {label}");
    }
    Ok(value)
}

fn require_material(enc: &CoseEncrypt0) -> Result<HybridKemMaterial> {
    if !empty_except_rest(&enc.unprotected) || enc.unprotected.rest.len() != 1 {
        bail!("COSE_Encrypt0 unprotected header set is not exact");
    }
    match unique_header_value(&enc.unprotected, HDR_KEM_MATERIAL)? {
        CborValue::Bytes(value) => HybridKemMaterial::decode(value),
        _ => bail!("COSE_Encrypt0 hybrid-KEM material has wrong type"),
    }
}

fn require_no_unprotected(enc: &CoseEncrypt0) -> Result<()> {
    if !enc.unprotected.is_empty() {
        bail!("pre-derived-key COSE_Encrypt0 must have no unprotected headers");
    }
    Ok(())
}

fn parse_canonical(cose_bytes: &[u8]) -> Result<CoseEncrypt0> {
    if cose_bytes.len() > MAX_COSE_BYTES {
        bail!("COSE_Encrypt0 exceeds {MAX_COSE_BYTES}-byte limit");
    }
    let enc = CoseEncrypt0::from_slice(cose_bytes)
        .map_err(|error| anyhow::anyhow!("malformed COSE_Encrypt0: {error}"))?;
    let ciphertext_len = enc
        .ciphertext
        .as_ref()
        .context("COSE_Encrypt0 has no ciphertext")?
        .len();
    if !(16..=MAX_PLAINTEXT_BYTES + 16).contains(&ciphertext_len) {
        bail!("COSE_Encrypt0 ciphertext length is outside bounded AEAD limits");
    }

    let mut normalized = enc.clone();
    normalized.protected.original_data = None;
    let canonical = normalized
        .to_vec()
        .map_err(|error| anyhow::anyhow!("cannot canonicalize COSE_Encrypt0: {error}"))?;
    if canonical != cose_bytes {
        bail!("COSE_Encrypt0 is not deterministic/canonical CBOR");
    }
    Ok(enc)
}

fn seal_core(
    base_key: &[u8; 32],
    suite: SuiteId,
    context: &SealContext,
    material: Option<&HybridKemMaterial>,
    plaintext: &[u8],
    external_aad: &[u8],
) -> Result<Vec<u8>> {
    validate_inputs(context, plaintext.len(), external_aad)?;
    let context_key = derive_context_key(base_key, suite, context)?;
    let nonce = derive_nonce(context);
    let mut builder = CoseEncrypt0Builder::new().protected(build_protected(suite, context));
    if let Some(material) = material {
        let unprotected = HeaderBuilder::new()
            .value(HDR_KEM_MATERIAL, CborValue::Bytes(material.encode()))
            .build();
        builder = builder.unprotected(unprotected);
    }
    let enc = builder
        .try_create_ciphertext(plaintext, external_aad, |pt, cose_aad| {
            aead_seal(&context_key, &nonce, pt, cose_aad)
        })?
        .build();
    let bytes = enc
        .to_vec()
        .map_err(|error| anyhow::anyhow!("serialize COSE_Encrypt0: {error}"))?;
    if bytes.len() > MAX_COSE_BYTES {
        bail!("serialized COSE_Encrypt0 exceeds {MAX_COSE_BYTES}-byte limit");
    }
    Ok(bytes)
}

fn open_core(
    base_key: &[u8; 32],
    expected_suite: SuiteId,
    expected: &SealContext,
    enc: &CoseEncrypt0,
    external_aad: &[u8],
) -> Result<Vec<u8>> {
    validate_inputs(expected, 0, external_aad)?;
    require_exact_protected(enc, expected_suite, expected)?;
    let context_key = derive_context_key(base_key, expected_suite, expected)?;
    let nonce = derive_nonce(expected);
    enc.decrypt_ciphertext(
        external_aad,
        || anyhow::anyhow!("COSE_Encrypt0 has no ciphertext"),
        |ciphertext, cose_aad| aead_open(&context_key, &nonce, ciphertext, cose_aad),
    )
}

/// Seal to a hybrid recipient with bounded canonical COSE and nonce-use state.
pub fn seal(
    recipient: &RecipientPublic,
    context: &SealContext,
    plaintext: &[u8],
    external_aad: &[u8],
    nonce_ledger: &mut NonceLedger,
) -> Result<Vec<u8>> {
    validate_inputs(context, plaintext.len(), external_aad)?;
    nonce_ledger.record(NonceUse::new(recipient.suite_id, context))?;
    let (material, secret) = hybrid_kem::encapsulate_to(recipient)?;
    let base_key = derive_aead_key(&secret);
    seal_core(
        &base_key,
        recipient.suite_id,
        context,
        Some(&material),
        plaintext,
        external_aad,
    )
}

/// Open from a hybrid recipient and record replay state after authentication.
pub fn open(
    recipient: &RecipientKeypair,
    expected: &SealContext,
    cose_bytes: &[u8],
    external_aad: &[u8],
    replay_ledger: &mut NonceLedger,
) -> Result<Vec<u8>> {
    validate_inputs(expected, 0, external_aad)?;
    let usage = NonceUse::new(recipient.suite_id, expected);
    replay_ledger.ensure_unseen(&usage)?;
    let enc = parse_canonical(cose_bytes)?;
    require_exact_protected(&enc, recipient.suite_id, expected)?;
    let material = require_material(&enc)?;
    if material.suite_id != recipient.suite_id {
        bail!("hybrid-KEM material suite does not match recipient suite");
    }
    let secret = hybrid_kem::decapsulate(recipient, &material)?;
    let base_key = derive_aead_key(&secret);
    let plaintext = open_core(&base_key, recipient.suite_id, expected, &enc, external_aad)?;
    replay_ledger.record(usage)?;
    Ok(plaintext)
}

/// Seal with a pre-derived base key and explicit lifecycle-owned nonce state.
///
/// The base key must be independently bound to the same pinned suite.  This is
/// the narrow seam intended for #554/#555 after their handshake/epoch commit.
pub fn seal_prederived(
    base_key: &[u8; 32],
    suite: SuiteId,
    context: &SealContext,
    plaintext: &[u8],
    external_aad: &[u8],
    nonce_ledger: &mut NonceLedger,
) -> Result<Vec<u8>> {
    validate_inputs(context, plaintext.len(), external_aad)?;
    nonce_ledger.record(NonceUse::new(suite, context))?;
    seal_core(base_key, suite, context, None, plaintext, external_aad)
}

/// Open with a pre-derived base key and explicit replay state.
pub fn open_prederived(
    base_key: &[u8; 32],
    expected_suite: SuiteId,
    expected: &SealContext,
    cose_bytes: &[u8],
    external_aad: &[u8],
    replay_ledger: &mut NonceLedger,
) -> Result<Vec<u8>> {
    validate_inputs(expected, 0, external_aad)?;
    let usage = NonceUse::new(expected_suite, expected);
    replay_ledger.ensure_unseen(&usage)?;
    let enc = parse_canonical(cose_bytes)?;
    require_no_unprotected(&enc)?;
    let plaintext = open_core(base_key, expected_suite, expected, &enc, external_aad)?;
    replay_ledger.record(usage)?;
    Ok(plaintext)
}

fn compatibility_context(
    recipient: &RecipientPublic,
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<SealContext> {
    let epoch = u32::try_from(epoch).context("epoch exceeds u32 nonce domain")?;
    let encoded = recipient.encode();
    let recipient_id = digest_parts(COMPAT_RECIPIENT_LABEL, &[&encoded]).to_vec();
    let key_id = digest_parts(COMPAT_KEY_LABEL, &[&encoded]).to_vec();
    let nonce_domain = digest_parts(COMPAT_DOMAIN_LABEL, &[external_aad]);
    Ok(SealContext {
        recipient_id,
        key_id,
        nonce_domain,
        epoch,
        sequence,
    })
}

fn compatibility_key_context(
    base_key: &[u8; 32],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<SealContext> {
    let epoch = u32::try_from(epoch).context("epoch exceeds u32 nonce domain")?;
    Ok(SealContext {
        recipient_id: digest_parts(COMPAT_RECIPIENT_LABEL, &[base_key]).to_vec(),
        key_id: digest_parts(COMPAT_KEY_LABEL, &[base_key]).to_vec(),
        nonce_domain: digest_parts(COMPAT_DOMAIN_LABEL, &[external_aad]),
        epoch,
        sequence,
    })
}

fn digest_parts(label: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(label);
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part);
    }
    hash.finalize().into()
}

/// Compatibility wrapper for the existing encrypted-envelope lifecycle.
///
/// New code must use [`seal`] with an owning [`NonceLedger`].
pub fn seal_to_recipient(
    recipient: &RecipientPublic,
    plaintext: &[u8],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<Vec<u8>> {
    let context = compatibility_context(recipient, external_aad, epoch, sequence)?;
    let (material, secret) = hybrid_kem::encapsulate_to(recipient)?;
    let base_key = derive_aead_key(&secret);
    seal_core(
        &base_key,
        recipient.suite_id,
        &context,
        Some(&material),
        plaintext,
        external_aad,
    )
}

/// Compatibility wrapper for the existing encrypted-envelope lifecycle.
///
/// New code must use [`open`] with an owning [`NonceLedger`].
pub fn open_from_recipient(
    recipient: &RecipientKeypair,
    cose_bytes: &[u8],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<Vec<u8>> {
    let context = compatibility_context(&recipient.public(), external_aad, epoch, sequence)?;
    let enc = parse_canonical(cose_bytes)?;
    require_exact_protected(&enc, recipient.suite_id, &context)?;
    let material = require_material(&enc)?;
    let secret = hybrid_kem::decapsulate(recipient, &material)?;
    let base_key = derive_aead_key(&secret);
    open_core(&base_key, recipient.suite_id, &context, &enc, external_aad)
}

/// Compatibility wrapper for pre-derived key callers.
///
/// New code must use [`seal_prederived`] with an owning [`NonceLedger`].
pub fn seal_with_key(
    base_key: &[u8; 32],
    suite: SuiteId,
    plaintext: &[u8],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<Vec<u8>> {
    let context = compatibility_key_context(base_key, external_aad, epoch, sequence)?;
    seal_core(base_key, suite, &context, None, plaintext, external_aad)
}

/// Compatibility wrapper for pre-derived key callers.
///
/// New code must use [`open_prederived`] with an owning [`NonceLedger`].
pub fn open_with_key(
    base_key: &[u8; 32],
    expected_suite: SuiteId,
    cose_bytes: &[u8],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<Vec<u8>> {
    let context = compatibility_key_context(base_key, external_aad, epoch, sequence)?;
    let enc = parse_canonical(cose_bytes)?;
    require_no_unprotected(&enc)?;
    open_core(base_key, expected_suite, &context, &enc, external_aad)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::hybrid_kem::{generate_recipient, recipient_from_seeds, KemId, KemShare};

    const SUITE: SuiteId = SuiteId::HyKemX25519MlKem768;
    const AAD: &[u8] = b"canonical transcript bytes";

    fn context(sequence: u64) -> SealContext {
        SealContext {
            recipient_id: b"did:at9p:example/service/inference".to_vec(),
            key_id: b"did:at9p:example#mesh-kem-7".to_vec(),
            nonce_domain: [0x5a; 32],
            epoch: 3,
            sequence,
        }
    }

    fn ledger() -> NonceLedger {
        NonceLedger::new(32).unwrap()
    }

    fn material(enc: &CoseEncrypt0) -> HybridKemMaterial {
        require_material(enc).unwrap()
    }

    fn replace_material(enc: &mut CoseEncrypt0, value: &HybridKemMaterial) {
        let (_, header_value) = enc
            .unprotected
            .rest
            .iter_mut()
            .find(|(label, _)| label == &Label::Int(HDR_KEM_MATERIAL))
            .unwrap();
        *header_value = CborValue::Bytes(value.encode());
    }

    #[test]
    fn recipient_roundtrip_and_replay_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(9);
        let mut outbound = ledger();
        let sealed = seal(&recipient.public(), &ctx, b"secret", AAD, &mut outbound).unwrap();
        let mut inbound = ledger();
        assert_eq!(
            open(&recipient, &ctx, &sealed, AAD, &mut inbound).unwrap(),
            b"secret"
        );
        assert!(open(&recipient, &ctx, &sealed, AAD, &mut inbound).is_err());
    }

    #[test]
    fn outbound_nonce_reuse_is_burned() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(4);
        let mut outbound = ledger();
        seal(&recipient.public(), &ctx, b"first", AAD, &mut outbound).unwrap();
        assert!(seal(&recipient.public(), &ctx, b"second", AAD, &mut outbound).is_err());
    }

    #[test]
    fn wrong_context_fields_and_aad_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(7);
        let sealed = seal(&recipient.public(), &ctx, b"secret", AAD, &mut ledger()).unwrap();
        let mutations = [
            {
                let mut value = ctx.clone();
                value.recipient_id.push(b'x');
                value
            },
            {
                let mut value = ctx.clone();
                value.key_id.push(b'x');
                value
            },
            {
                let mut value = ctx.clone();
                value.nonce_domain[0] ^= 1;
                value
            },
            {
                let mut value = ctx.clone();
                value.epoch += 1;
                value
            },
            {
                let mut value = ctx.clone();
                value.sequence += 1;
                value
            },
        ];
        for mutation in mutations {
            assert!(open(&recipient, &mutation, &sealed, AAD, &mut ledger()).is_err());
        }
        assert!(open(
            &recipient,
            &ctx,
            &sealed,
            b"wrong transcript",
            &mut ledger()
        )
        .is_err());
    }

    #[test]
    fn wrong_recipient_and_cartesian_component_keys_rejected() {
        let first = generate_recipient(SUITE).unwrap();
        let second = generate_recipient(SUITE).unwrap();
        let ctx = context(1);
        let sealed = seal(&first.public(), &ctx, b"secret", AAD, &mut ledger()).unwrap();
        assert!(open(&second, &ctx, &sealed, AAD, &mut ledger()).is_err());

        let x_seed = [0x11; 32];
        let pq_seed_a = [0x22; 64];
        let pq_seed_b = [0x33; 64];
        let intended = recipient_from_seeds(SUITE, &[&x_seed, &pq_seed_a]).unwrap();
        let cartesian = recipient_from_seeds(SUITE, &[&x_seed, &pq_seed_b]).unwrap();
        let sealed = seal(
            &intended.public(),
            &context(2),
            b"secret",
            AAD,
            &mut ledger(),
        )
        .unwrap();
        assert!(open(&cartesian, &context(2), &sealed, AAD, &mut ledger()).is_err());
    }

    #[test]
    fn missing_reordered_and_cross_encapsulation_pq_shares_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx_a = context(10);
        let ctx_b = context(11);
        let sealed_a = seal(&recipient.public(), &ctx_a, b"a", AAD, &mut ledger()).unwrap();
        let sealed_b = seal(&recipient.public(), &ctx_b, b"b", AAD, &mut ledger()).unwrap();
        let enc_a = CoseEncrypt0::from_slice(&sealed_a).unwrap();
        let enc_b = CoseEncrypt0::from_slice(&sealed_b).unwrap();

        let mut missing = material(&enc_a);
        missing.shares.pop();
        let mut mutated = enc_a.clone();
        replace_material(&mut mutated, &missing);
        mutated.protected.original_data = None;
        assert!(open(
            &recipient,
            &ctx_a,
            &mutated.to_vec().unwrap(),
            AAD,
            &mut ledger()
        )
        .is_err());

        let original = material(&enc_a);
        let reordered = HybridKemMaterial {
            suite_id: SUITE,
            shares: vec![original.shares[1].clone(), original.shares[0].clone()],
        };
        let mut mutated = enc_a.clone();
        replace_material(&mut mutated, &reordered);
        mutated.protected.original_data = None;
        assert!(open(
            &recipient,
            &ctx_a,
            &mutated.to_vec().unwrap(),
            AAD,
            &mut ledger()
        )
        .is_err());

        let mut crossed = original;
        let other = material(&enc_b);
        assert_eq!(crossed.shares[1].kem_id, KemId::MlKem768);
        crossed.shares[1] = KemShare {
            kem_id: KemId::MlKem768,
            bytes: other.shares[1].bytes.clone(),
        };
        let mut mutated = enc_a;
        replace_material(&mut mutated, &crossed);
        mutated.protected.original_data = None;
        assert!(open(
            &recipient,
            &ctx_a,
            &mutated.to_vec().unwrap(),
            AAD,
            &mut ledger()
        )
        .is_err());
    }

    #[test]
    fn noncanonical_and_duplicate_headers_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(12);
        let sealed = seal(&recipient.public(), &ctx, b"secret", AAD, &mut ledger()).unwrap();
        assert_eq!(sealed[0], 0x83);
        let mut noncanonical = vec![0x98, 0x03];
        noncanonical.extend_from_slice(&sealed[1..]);
        assert!(open(&recipient, &ctx, &noncanonical, AAD, &mut ledger()).is_err());

        let mut top: CborValue = ciborium::de::from_reader(sealed.as_slice()).unwrap();
        let array = match &mut top {
            CborValue::Array(value) => value,
            _ => panic!("COSE_Encrypt0 must be an array"),
        };
        let protected = match &mut array[0] {
            CborValue::Bytes(value) => value,
            _ => panic!("protected header must be bytes"),
        };
        assert!((0xa0..=0xb7).contains(&protected[0]));
        protected[0] += 1;
        let duplicate = CborValue::Map(vec![(
            CborValue::Integer(HDR_SUITE_ID.into()),
            CborValue::Integer(SUITE.as_u16().into()),
        )]);
        let mut encoded_duplicate = Vec::new();
        ciborium::ser::into_writer(&duplicate, &mut encoded_duplicate).unwrap();
        protected.extend_from_slice(&encoded_duplicate[1..]);
        let mut duplicate_wire = Vec::new();
        ciborium::ser::into_writer(&top, &mut duplicate_wire).unwrap();
        assert!(open(&recipient, &ctx, &duplicate_wire, AAD, &mut ledger()).is_err());
    }

    #[test]
    fn truncation_unknown_headers_and_tamper_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(13);
        let sealed = seal(&recipient.public(), &ctx, b"secret", AAD, &mut ledger()).unwrap();
        for cut in [0, 1, sealed.len() / 2, sealed.len() - 1] {
            assert!(open(&recipient, &ctx, &sealed[..cut], AAD, &mut ledger()).is_err());
        }
        let mut tampered = sealed.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x80;
        assert!(open(&recipient, &ctx, &tampered, AAD, &mut ledger()).is_err());

        let mut unknown = CoseEncrypt0::from_slice(&sealed).unwrap();
        unknown
            .protected
            .header
            .rest
            .push((Label::Int(-65560), CborValue::Integer(1.into())));
        unknown.protected.original_data = None;
        assert!(open(
            &recipient,
            &ctx,
            &unknown.to_vec().unwrap(),
            AAD,
            &mut ledger()
        )
        .is_err());
    }

    #[test]
    fn wrong_suite_and_algorithm_headers_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(14);
        let sealed = seal(&recipient.public(), &ctx, b"secret", AAD, &mut ledger()).unwrap();

        let mut wrong_suite = CoseEncrypt0::from_slice(&sealed).unwrap();
        let (_, suite_value) = wrong_suite
            .protected
            .header
            .rest
            .iter_mut()
            .find(|(label, _)| label == &Label::Int(HDR_SUITE_ID))
            .unwrap();
        *suite_value = CborValue::Integer(2.into());
        wrong_suite.protected.original_data = None;
        assert!(open(
            &recipient,
            &ctx,
            &wrong_suite.to_vec().unwrap(),
            AAD,
            &mut ledger()
        )
        .is_err());

        let mut wrong_algorithm = CoseEncrypt0::from_slice(&sealed).unwrap();
        wrong_algorithm.protected.header.alg =
            Some(coset::Algorithm::Assigned(iana::Algorithm::A256GCM));
        wrong_algorithm.protected.original_data = None;
        assert!(open(
            &recipient,
            &ctx,
            &wrong_algorithm.to_vec().unwrap(),
            AAD,
            &mut ledger()
        )
        .is_err());
    }

    #[test]
    fn bounded_inputs_and_ledger_exhaustion_fail_closed() {
        let recipient = generate_recipient(SUITE).unwrap();
        let mut one = NonceLedger::new(1).unwrap();
        seal(&recipient.public(), &context(0), b"a", AAD, &mut one).unwrap();
        assert!(seal(&recipient.public(), &context(1), b"b", AAD, &mut one).is_err());
        assert!(NonceLedger::new(0).is_err());
        assert!(seal(
            &recipient.public(),
            &context(2),
            b"x",
            &vec![0; MAX_EXTERNAL_AAD_BYTES + 1],
            &mut ledger()
        )
        .is_err());
        assert!(seal(
            &recipient.public(),
            &context(3),
            &vec![0; MAX_PLAINTEXT_BYTES + 1],
            AAD,
            &mut ledger()
        )
        .is_err());
        assert!(open(
            &recipient,
            &context(4),
            &vec![0; MAX_COSE_BYTES + 1],
            AAD,
            &mut ledger()
        )
        .is_err());
    }

    #[test]
    fn prederived_roundtrip_context_key_and_epoch_retirement() {
        let key = [0x42; 32];
        let ctx = context(21);
        let mut outbound = ledger();
        let sealed = seal_prederived(&key, SUITE, &ctx, b"block", AAD, &mut outbound).unwrap();
        let mut inbound = ledger();
        assert_eq!(
            open_prederived(&key, SUITE, &ctx, &sealed, AAD, &mut inbound).unwrap(),
            b"block"
        );
        inbound.retire_epoch(
            SUITE,
            &ctx.recipient_id,
            &ctx.key_id,
            &ctx.nonce_domain,
            ctx.epoch,
        );
        assert!(inbound.is_empty());
    }

    #[test]
    fn rfc8439_chacha20poly1305_vector() {
        // RFC 8439 section 2.8.2.  This proves the registered content algorithm;
        // it is not represented as a Hyprstream/COSE interoperability vector.
        let key = hex::decode("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
            .unwrap();
        let nonce = hex::decode("070000004041424344454647").unwrap();
        let aad = hex::decode("50515253c0c1c2c3c4c5c6c7").unwrap();
        let plaintext = hex::decode(concat!(
            "4c616469657320616e642047656e746c656d656e206f662074686520636c61737320",
            "6f66202739393a204966204920636f756c64206f6666657220796f75206f6e6c7920",
            "6f6e652074697020666f7220746865206675747572652c2073756e73637265656e20",
            "776f756c642062652069742e"
        ))
        .unwrap();
        let expected = hex::decode(concat!(
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6",
            "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36",
            "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc",
            "3ff4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd060",
            "0691"
        ))
        .unwrap();
        let mut key_array = [0; 32];
        key_array.copy_from_slice(&key);
        let mut nonce_array = [0; 12];
        nonce_array.copy_from_slice(&nonce);
        assert_eq!(
            aead_seal(&key_array, &nonce_array, &plaintext, &aad).unwrap(),
            expected
        );
    }

    #[test]
    fn compatibility_wrappers_remain_canonical() {
        let recipient = generate_recipient(SUITE).unwrap();
        let sealed = seal_to_recipient(&recipient.public(), b"legacy", AAD, 0, 0).unwrap();
        assert_eq!(
            open_from_recipient(&recipient, &sealed, AAD, 0, 0).unwrap(),
            b"legacy"
        );
        let key = [7; 32];
        let sealed = seal_with_key(&key, SUITE, b"legacy", AAD, 0, 1).unwrap();
        assert_eq!(
            open_with_key(&key, SUITE, &sealed, AAD, 0, 1).unwrap(),
            b"legacy"
        );
    }
}
