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
//! New consumers should use [`seal`] and [`seal_prederived`] with an owning
//! [`OutboundNonceLedger`], and [`open`] and [`open_prederived`] with an owning
//! [`InboundReplayLedger`].  The fresh-encapsulation envelope wrappers at the
//! bottom remain only for the already-landed signed envelope path, whose replay
//! lifecycle is owned by `SignedEnvelope`.

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

// These structural limits are deliberately much tighter than the 16 MiB wire
// cap.  This profile has a three-item outer array, at most eight protected
// headers, one unprotected header, and a five-item `crit` array.  The modest
// headroom permits future private headers without allowing a small input to
// amplify into a large generic `ciborium::Value` tree before schema validation.
const MAX_CBOR_DEPTH: usize = 8;
const MAX_CBOR_CONTAINER_ITEMS: usize = 32;
const MAX_CBOR_TOTAL_ITEMS: usize = 64;
const MAX_PROTECTED_HEADER_BYTES: usize = 4 * 1024;
const MAX_HEADER_VALUE_BYTES: usize = 4 * 1024;
const MAX_HEADER_TEXT_BYTES: usize = MAX_IDENTIFIER_BYTES;

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

/// Irreversible bounded state for outbound nonce use.
///
/// One ledger owns the complete lifetime of the base key supplied to `seal*`.
/// It may be dropped only after that base key has been irreversibly destroyed or
/// rotated out; constructing a replacement ledger while retaining the key would
/// discard its nonce-safety history.  Burns are never evicted or retired, and
/// capacity exhaustion therefore requires base-key rotation.  A coordinate is
/// burned before cryptographic work, so even an error cannot make an uncertain
/// nonce reusable.
#[derive(Debug)]
pub struct OutboundNonceLedger {
    capacity: usize,
    seen: HashSet<NonceUse>,
}

impl OutboundNonceLedger {
    /// Construct outbound state with a finite non-zero lifetime capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > MAX_LEDGER_ENTRIES {
            bail!("outbound nonce ledger capacity must be in 1..={MAX_LEDGER_ENTRIES}");
        }
        Ok(Self {
            capacity,
            seen: HashSet::with_capacity(capacity.min(4096)),
        })
    }

    /// Number of irreversibly burned nonce coordinates.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether no coordinates have been burned.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
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
            bail!("outbound nonce ledger capacity exhausted; destroy and rotate the base key");
        }
        self.seen.insert(usage);
        Ok(())
    }
}

/// Bounded state for authenticated inbound replay rejection.
///
/// Unlike outbound burns, replay entries may be retired after the public
/// lifecycle owner has made the corresponding epoch unacceptable before
/// calling `open*`.  Retirement does not and cannot affect outbound state.
#[derive(Debug)]
pub struct InboundReplayLedger {
    capacity: usize,
    seen: HashSet<NonceUse>,
}

impl InboundReplayLedger {
    /// Construct inbound replay state with a finite non-zero capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > MAX_LEDGER_ENTRIES {
            bail!("inbound replay ledger capacity must be in 1..={MAX_LEDGER_ENTRIES}");
        }
        Ok(Self {
            capacity,
            seen: HashSet::with_capacity(capacity.min(4096)),
        })
    }

    /// Number of retained authenticated replay coordinates.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether no replay coordinates are retained.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Retire an inbound epoch after the owner has made it unacceptable.
    ///
    /// This is replay-retention management only; epoch install, accept, grace,
    /// and retirement policy remain with #554/#555.
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
            bail!("ciphertext replayed");
        }
        Ok(())
    }

    fn record(&mut self, usage: NonceUse) -> Result<()> {
        self.ensure_unseen(&usage)?;
        if self.seen.len() >= self.capacity {
            bail!("inbound replay ledger capacity exhausted; retire an unacceptable epoch");
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

/// Allocation-free structural validation of the small CBOR subset used by the
/// COSE profile.  In addition to bounding work, this independently enforces RFC
/// 8949 core deterministic argument encodings and map-key order (encoded-key
/// bytes compared lexicographically).  It intentionally performs no COSE
/// semantics; `coset` remains the sole semantic decoder below.
struct CborPreflight<'a> {
    input: &'a [u8],
    offset: usize,
    items: usize,
    max_byte_string: usize,
}

impl<'a> CborPreflight<'a> {
    fn new(input: &'a [u8], max_byte_string: usize) -> Self {
        Self {
            input,
            offset: 0,
            items: 0,
            max_byte_string,
        }
    }

    fn finish(self) -> Result<()> {
        if self.offset != self.input.len() {
            bail!("CBOR contains trailing data");
        }
        Ok(())
    }

    fn read_byte(&mut self) -> Result<u8> {
        let byte = *self.input.get(self.offset).context("truncated CBOR item")?;
        self.offset += 1;
        Ok(byte)
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.offset.checked_add(N).context("CBOR length overflow")?;
        let bytes = self
            .input
            .get(self.offset..end)
            .context("truncated CBOR argument")?;
        self.offset = end;
        bytes.try_into().context("invalid CBOR argument width")
    }

    fn head(&mut self) -> Result<(u8, u64)> {
        self.items = self.items.checked_add(1).context("CBOR item overflow")?;
        if self.items > MAX_CBOR_TOTAL_ITEMS {
            bail!("CBOR contains more than {MAX_CBOR_TOTAL_ITEMS} items");
        }
        let initial = self.read_byte()?;
        let major = initial >> 5;
        let additional = initial & 0x1f;
        let argument = match additional {
            value @ 0..=23 => u64::from(value),
            24 => {
                let value = u64::from(self.read_byte()?);
                if value < 24 {
                    bail!("CBOR argument is not shortest-form encoded");
                }
                value
            }
            25 => {
                let value = u64::from(u16::from_be_bytes(self.read_exact()?));
                if value <= u64::from(u8::MAX) {
                    bail!("CBOR argument is not shortest-form encoded");
                }
                value
            }
            26 => {
                let value = u64::from(u32::from_be_bytes(self.read_exact()?));
                if value <= u64::from(u16::MAX) {
                    bail!("CBOR argument is not shortest-form encoded");
                }
                value
            }
            27 => {
                let value = u64::from_be_bytes(self.read_exact()?);
                if value <= u64::from(u32::MAX) {
                    bail!("CBOR argument is not shortest-form encoded");
                }
                value
            }
            31 => bail!("indefinite-length CBOR is forbidden"),
            _ => bail!("reserved CBOR additional-information value"),
        };
        Ok((major, argument))
    }

    fn length(argument: u64) -> Result<usize> {
        usize::try_from(argument).context("CBOR length does not fit address space")
    }

    fn bytes(&mut self, expected_major: u8, limit: usize) -> Result<&'a [u8]> {
        let (major, argument) = self.head()?;
        if major != expected_major {
            bail!("unexpected CBOR major type {major}, expected {expected_major}");
        }
        let length = Self::length(argument)?;
        if length > limit {
            bail!("CBOR string length {length} exceeds {limit}-byte limit");
        }
        let end = self
            .offset
            .checked_add(length)
            .context("CBOR string length overflow")?;
        let bytes = self
            .input
            .get(self.offset..end)
            .context("truncated CBOR string")?;
        self.offset = end;
        Ok(bytes)
    }

    fn container(&mut self, expected_major: u8) -> Result<usize> {
        let (major, argument) = self.head()?;
        if major != expected_major {
            bail!("unexpected CBOR major type {major}, expected {expected_major}");
        }
        let length = Self::length(argument)?;
        if length > MAX_CBOR_CONTAINER_ITEMS {
            bail!("CBOR container has {length} entries; limit is {MAX_CBOR_CONTAINER_ITEMS}");
        }
        Ok(length)
    }

    fn item(&mut self, depth: usize) -> Result<()> {
        if depth > MAX_CBOR_DEPTH {
            bail!("CBOR nesting exceeds depth {MAX_CBOR_DEPTH}");
        }
        let start = self.offset;
        let (major, argument) = self.head()?;
        match major {
            0 | 1 => {}
            2 | 3 => {
                let length = Self::length(argument)?;
                let limit = if major == 2 {
                    self.max_byte_string
                } else {
                    MAX_HEADER_TEXT_BYTES
                };
                if length > limit {
                    bail!("CBOR string length {length} exceeds {limit}-byte limit");
                }
                self.offset = self
                    .offset
                    .checked_add(length)
                    .context("CBOR string length overflow")?;
                if self.offset > self.input.len() {
                    bail!("truncated CBOR string");
                }
            }
            4 => {
                let length = Self::length(argument)?;
                if length > MAX_CBOR_CONTAINER_ITEMS {
                    bail!("CBOR array has {length} entries; limit is {MAX_CBOR_CONTAINER_ITEMS}");
                }
                for _ in 0..length {
                    self.item(depth + 1)?;
                }
            }
            5 => {
                let length = Self::length(argument)?;
                if length > MAX_CBOR_CONTAINER_ITEMS {
                    bail!("CBOR map has {length} entries; limit is {MAX_CBOR_CONTAINER_ITEMS}");
                }
                let mut previous_key: Option<(usize, usize)> = None;
                for _ in 0..length {
                    let key_start = self.offset;
                    self.item(depth + 1)?;
                    let key_end = self.offset;
                    if let Some((previous_start, previous_end)) = previous_key {
                        let previous = &self.input[previous_start..previous_end];
                        let current = &self.input[key_start..key_end];
                        if previous >= current {
                            bail!("CBOR map keys are duplicate or not in deterministic order");
                        }
                    }
                    previous_key = Some((key_start, key_end));
                    self.item(depth + 1)?;
                }
            }
            // Tags, floats, simple values, and break are not used anywhere in
            // this fixed COSE profile.  Rejecting them keeps the preflight small
            // and prevents semantic-decoder discrepancies.
            6 | 7 => bail!("unsupported CBOR type in COSE_Encrypt0 profile"),
            _ => unreachable!("CBOR major type is three bits"),
        }
        if self.offset <= start {
            bail!("CBOR scanner made no progress");
        }
        Ok(())
    }
}

fn preflight_cose_encrypt0(cose_bytes: &[u8]) -> Result<()> {
    let mut outer = CborPreflight::new(cose_bytes, MAX_HEADER_VALUE_BYTES);
    let top_level_items = outer.container(4)?;
    if top_level_items != 3 {
        bail!("COSE_Encrypt0 must be a definite three-element array");
    }

    let protected = outer.bytes(2, MAX_PROTECTED_HEADER_BYTES)?;
    let mut protected_scanner = CborPreflight::new(protected, MAX_HEADER_VALUE_BYTES);
    let protected_entries = protected_scanner.container(5)?;
    if protected_entries > 16 {
        bail!("protected header map has too many entries");
    }
    // Rewind and scan the entire map so key ordering and nested values are
    // checked by the same generic deterministic scanner.
    protected_scanner.offset = 0;
    protected_scanner.items = 0;
    protected_scanner.item(0)?;
    protected_scanner.finish()?;

    let unprotected_start = outer.offset;
    let unprotected_entries = outer.container(5)?;
    if unprotected_entries > 16 {
        bail!("unprotected header map has too many entries");
    }
    outer.offset = unprotected_start;
    outer.items -= 1;
    outer.item(0)?;

    let ciphertext = outer.bytes(2, MAX_PLAINTEXT_BYTES + 16)?;
    if ciphertext.len() < 16 {
        bail!("COSE_Encrypt0 ciphertext is shorter than its AEAD tag");
    }
    outer.finish()
}

fn parse_canonical(cose_bytes: &[u8]) -> Result<CoseEncrypt0> {
    if cose_bytes.len() > MAX_COSE_BYTES {
        bail!("COSE_Encrypt0 exceeds {MAX_COSE_BYTES}-byte limit");
    }
    preflight_cose_encrypt0(cose_bytes)
        .context("COSE_Encrypt0 failed bounded deterministic CBOR preflight")?;
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
    nonce_ledger: &mut OutboundNonceLedger,
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
    replay_ledger: &mut InboundReplayLedger,
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
    nonce_ledger: &mut OutboundNonceLedger,
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
    replay_ledger: &mut InboundReplayLedger,
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
    if external_aad.len() > MAX_EXTERNAL_AAD_BYTES {
        bail!("external AAD exceeds {MAX_EXTERNAL_AAD_BYTES}-byte limit");
    }
    recipient.validate()?;
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
/// New code must use [`seal`] with an owning [`OutboundNonceLedger`].
pub fn seal_to_recipient(
    recipient: &RecipientPublic,
    plaintext: &[u8],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<Vec<u8>> {
    let context = compatibility_context(recipient, external_aad, epoch, sequence)?;
    validate_inputs(&context, plaintext.len(), external_aad)?;
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
/// New code must use [`open`] with an owning [`InboundReplayLedger`].
pub fn open_from_recipient(
    recipient: &RecipientKeypair,
    cose_bytes: &[u8],
    external_aad: &[u8],
    epoch: u64,
    sequence: u64,
) -> Result<Vec<u8>> {
    let context = compatibility_context(&recipient.public(), external_aad, epoch, sequence)?;
    validate_inputs(&context, 0, external_aad)?;
    let enc = parse_canonical(cose_bytes)?;
    require_exact_protected(&enc, recipient.suite_id, &context)?;
    let material = require_material(&enc)?;
    let secret = hybrid_kem::decapsulate(recipient, &material)?;
    let base_key = derive_aead_key(&secret);
    open_core(&base_key, recipient.suite_id, &context, &enc, external_aad)
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

    fn outbound_ledger() -> OutboundNonceLedger {
        OutboundNonceLedger::new(32).unwrap()
    }

    fn inbound_ledger() -> InboundReplayLedger {
        InboundReplayLedger::new(32).unwrap()
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

    fn encode_value(value: &CborValue) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(value, &mut bytes).unwrap();
        bytes
    }

    fn mutate_protected_map(
        sealed: &[u8],
        mutation: impl FnOnce(&mut Vec<(CborValue, CborValue)>),
    ) -> Vec<u8> {
        let mut top: CborValue = ciborium::de::from_reader(sealed).unwrap();
        let top_array = match &mut top {
            CborValue::Array(items) => items,
            _ => panic!("COSE_Encrypt0 must be an array"),
        };
        let protected_bytes = match &mut top_array[0] {
            CborValue::Bytes(bytes) => bytes,
            _ => panic!("protected header must be a byte string"),
        };
        let mut protected: CborValue =
            ciborium::de::from_reader(protected_bytes.as_slice()).unwrap();
        let map = match &mut protected {
            CborValue::Map(entries) => entries,
            _ => panic!("protected header must contain a map"),
        };
        mutation(map);
        *protected_bytes = encode_value(&protected);
        encode_value(&top)
    }

    fn replace_once(bytes: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) {
        let offset = bytes
            .windows(needle.len())
            .position(|candidate| candidate == needle)
            .expect("mutation target must exist");
        bytes.splice(offset..offset + needle.len(), replacement.iter().copied());
    }

    #[test]
    fn recipient_roundtrip_and_replay_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(9);
        let mut outbound = outbound_ledger();
        let sealed = seal(&recipient.public(), &ctx, b"secret", AAD, &mut outbound).unwrap();
        let mut inbound = inbound_ledger();
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
        let mut outbound = outbound_ledger();
        seal(&recipient.public(), &ctx, b"first", AAD, &mut outbound).unwrap();
        assert!(seal(&recipient.public(), &ctx, b"second", AAD, &mut outbound).is_err());
    }

    #[test]
    fn wrong_context_fields_and_aad_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(7);
        let sealed = seal(
            &recipient.public(),
            &ctx,
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();
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
            assert!(open(&recipient, &mutation, &sealed, AAD, &mut inbound_ledger()).is_err());
        }
        assert!(open(
            &recipient,
            &ctx,
            &sealed,
            b"wrong transcript",
            &mut inbound_ledger()
        )
        .is_err());
    }

    #[test]
    fn wrong_recipient_and_cartesian_component_keys_rejected() {
        let first = generate_recipient(SUITE).unwrap();
        let second = generate_recipient(SUITE).unwrap();
        let ctx = context(1);
        let sealed = seal(
            &first.public(),
            &ctx,
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();
        assert!(open(&second, &ctx, &sealed, AAD, &mut inbound_ledger()).is_err());

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
            &mut outbound_ledger(),
        )
        .unwrap();
        assert!(open(&cartesian, &context(2), &sealed, AAD, &mut inbound_ledger()).is_err());
    }

    #[test]
    fn missing_reordered_and_cross_encapsulation_pq_shares_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx_a = context(10);
        let ctx_b = context(11);
        let sealed_a = seal(
            &recipient.public(),
            &ctx_a,
            b"a",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();
        let sealed_b = seal(
            &recipient.public(),
            &ctx_b,
            b"b",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();
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
            &mut inbound_ledger()
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
            &mut inbound_ledger()
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
            &mut inbound_ledger()
        )
        .is_err());
    }

    #[test]
    fn noncanonical_and_duplicate_headers_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(12);
        let sealed = seal(
            &recipient.public(),
            &ctx,
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();
        assert_eq!(sealed[0], 0x83);
        let mut noncanonical = vec![0x98, 0x03];
        noncanonical.extend_from_slice(&sealed[1..]);
        assert!(open(&recipient, &ctx, &noncanonical, AAD, &mut inbound_ledger()).is_err());

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
        assert!(open(
            &recipient,
            &ctx,
            &duplicate_wire,
            AAD,
            &mut inbound_ledger()
        )
        .is_err());
    }

    #[test]
    fn reordered_protected_and_unprotected_maps_are_rejected_by_preflight() {
        let recipient = generate_recipient(SUITE).unwrap();
        let sealed = seal(
            &recipient.public(),
            &context(121),
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();

        let reordered_protected = mutate_protected_map(&sealed, |entries| {
            let first = entries
                .iter()
                .position(|(key, _)| key == &CborValue::Integer(HDR_SUITE_ID.into()))
                .unwrap();
            let second = entries
                .iter()
                .position(|(key, _)| key == &CborValue::Integer(HDR_RECIPIENT_ID.into()))
                .unwrap();
            entries.swap(first, second);
        });
        assert!(parse_canonical(&reordered_protected).is_err());

        let mut top: CborValue = ciborium::de::from_reader(sealed.as_slice()).unwrap();
        let unprotected = match &mut top {
            CborValue::Array(items) => match &mut items[1] {
                CborValue::Map(entries) => entries,
                _ => panic!("unprotected header must be a map"),
            },
            _ => panic!("COSE_Encrypt0 must be an array"),
        };
        unprotected.push((
            CborValue::Integer((-65560).into()),
            CborValue::Integer(1.into()),
        ));
        // Both private labels have equal-length encodings; -65539 sorts before
        // -65560 by its encoded argument.  Swap them to retain a valid map with
        // a mutation that only the independent ordering check must reject.
        unprotected.swap(0, 1);
        assert!(parse_canonical(&encode_value(&top)).is_err());
    }

    #[test]
    fn nonminimal_header_keys_and_values_are_rejected_by_preflight() {
        let recipient = generate_recipient(SUITE).unwrap();
        let sealed = seal(
            &recipient.public(),
            &context(122),
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();

        // -65538 is major type 1 with argument 65537.  Eight argument bytes are
        // valid CBOR but not the shortest deterministic representation.
        let mut protected_key = mutate_protected_map(&sealed, |_| {});
        let mut top: CborValue = ciborium::de::from_reader(protected_key.as_slice()).unwrap();
        let protected = match &mut top {
            CborValue::Array(items) => match &mut items[0] {
                CborValue::Bytes(bytes) => bytes,
                _ => panic!("protected header must be bytes"),
            },
            _ => panic!("COSE_Encrypt0 must be an array"),
        };
        replace_once(
            protected,
            &[0x3a, 0x00, 0x01, 0x00, 0x01],
            &[0x3b, 0, 0, 0, 0, 0, 1, 0, 1],
        );
        protected_key = encode_value(&top);
        assert!(parse_canonical(&protected_key).is_err());

        // The suite value 1 follows that key.  Encoding it as 0x18 0x01 is
        // likewise a valid but non-shortest integer representation.
        let mut top: CborValue = ciborium::de::from_reader(sealed.as_slice()).unwrap();
        let protected = match &mut top {
            CborValue::Array(items) => match &mut items[0] {
                CborValue::Bytes(bytes) => bytes,
                _ => panic!("protected header must be bytes"),
            },
            _ => panic!("COSE_Encrypt0 must be an array"),
        };
        replace_once(
            protected,
            &[0x3a, 0x00, 0x01, 0x00, 0x01, 0x01],
            &[0x3a, 0x00, 0x01, 0x00, 0x01, 0x18, 0x01],
        );
        assert!(parse_canonical(&encode_value(&top)).is_err());

        // The unprotected private key -65539 is also rejected when widened.
        let mut unprotected_key = sealed.clone();
        replace_once(
            &mut unprotected_key,
            &[0x3a, 0x00, 0x01, 0x00, 0x02],
            &[0x3b, 0, 0, 0, 0, 0, 1, 0, 2],
        );
        assert!(parse_canonical(&unprotected_key).is_err());

        // The hybrid material is a 16-bit-length byte string in the valid
        // object.  Widen only that length argument to 32 bits.
        let mut unprotected_value = sealed.clone();
        let key = [0x3a, 0x00, 0x01, 0x00, 0x02];
        let key_offset = unprotected_value
            .windows(key.len())
            .position(|candidate| candidate == key)
            .unwrap();
        let value_offset = key_offset + key.len();
        assert_eq!(unprotected_value[value_offset], 0x59);
        let high = unprotected_value[value_offset + 1];
        let low = unprotected_value[value_offset + 2];
        unprotected_value.splice(
            value_offset..value_offset + 3,
            [0x5a, 0x00, 0x00, high, low],
        );
        assert!(parse_canonical(&unprotected_value).is_err());
    }

    #[test]
    fn hostile_cbor_is_rejected_before_value_tree_amplification() {
        let ciphertext = [0u8; 16];
        let finish = |mut prefix: Vec<u8>| {
            prefix.push(0x50);
            prefix.extend_from_slice(&ciphertext);
            prefix
        };

        // A 12-byte prefix claims more than four billion array members.  The
        // preflight rejects the declared count without iterating or allocating.
        let amplified = finish(vec![
            0x83, 0x41, 0xa0, 0xa1, 0x00, 0x9a, 0xff, 0xff, 0xff, 0xff,
        ]);
        assert!(preflight_cose_encrypt0(&amplified).is_err());

        let oversized_container = finish(vec![0x83, 0x41, 0xa0, 0xa1, 0x00, 0x98, 0x21]);
        assert!(preflight_cose_encrypt0(&oversized_container).is_err());

        let indefinite = finish(vec![0x83, 0x41, 0xa0, 0xbf, 0xff]);
        assert!(preflight_cose_encrypt0(&indefinite).is_err());

        let mut deep = vec![0x83, 0x41, 0xa0, 0xa1, 0x00];
        deep.extend(std::iter::repeat_n(0x81, MAX_CBOR_DEPTH + 1));
        deep.push(0x00);
        assert!(preflight_cose_encrypt0(&finish(deep)).is_err());

        // A header byte string over the 4 KiB schema limit is rejected from its
        // five-byte declaration alone, before looking for or allocating bytes.
        let oversized_string = finish(vec![
            0x83, 0x41, 0xa0, 0xa1, 0x00, 0x5a, 0x00, 0x00, 0x10, 0x01,
        ]);
        assert!(preflight_cose_encrypt0(&oversized_string).is_err());

        let mut trailing = finish(vec![0x83, 0x41, 0xa0, 0xa0]);
        trailing.push(0x00);
        assert!(preflight_cose_encrypt0(&trailing).is_err());
    }

    #[test]
    fn deterministic_map_order_is_core_bytewise_lexicographic() {
        let finish = |mut prefix: Vec<u8>| {
            prefix.push(0x50);
            prefix.extend_from_slice(&[0u8; 16]);
            prefix
        };
        // RFC 8949 section 4.2.1 compares the deterministic key encodings
        // bytewise, so 100 (0x18 0x64) sorts before -1 (0x20).  The reverse is
        // the optional length-first order from section 4.2.3, which this profile
        // deliberately does not implement.
        let canonical = finish(vec![0x83, 0x41, 0xa0, 0xa2, 0x18, 0x64, 0x00, 0x20, 0x00]);
        assert!(preflight_cose_encrypt0(&canonical).is_ok());

        let reversed = finish(vec![0x83, 0x41, 0xa0, 0xa2, 0x20, 0x00, 0x18, 0x64, 0x00]);
        assert!(preflight_cose_encrypt0(&reversed).is_err());
    }

    #[test]
    fn focused_cose_gate_covers_every_hybrid_component_secret() {
        assert!(hybrid_kem::combiner_is_sensitive_to_every_component_for_cose());
    }

    #[test]
    fn truncation_unknown_headers_and_tamper_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(13);
        let sealed = seal(
            &recipient.public(),
            &ctx,
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();
        for cut in [0, 1, sealed.len() / 2, sealed.len() - 1] {
            assert!(open(&recipient, &ctx, &sealed[..cut], AAD, &mut inbound_ledger()).is_err());
        }
        let mut tampered = sealed.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x80;
        assert!(open(&recipient, &ctx, &tampered, AAD, &mut inbound_ledger()).is_err());

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
            &mut inbound_ledger()
        )
        .is_err());
    }

    #[test]
    fn wrong_suite_and_algorithm_headers_rejected() {
        let recipient = generate_recipient(SUITE).unwrap();
        let ctx = context(14);
        let sealed = seal(
            &recipient.public(),
            &ctx,
            b"secret",
            AAD,
            &mut outbound_ledger(),
        )
        .unwrap();

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
            &mut inbound_ledger()
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
            &mut inbound_ledger()
        )
        .is_err());
    }

    #[test]
    fn bounded_inputs_and_ledger_exhaustion_fail_closed() {
        let recipient = generate_recipient(SUITE).unwrap();
        let mut one = OutboundNonceLedger::new(1).unwrap();
        seal(&recipient.public(), &context(0), b"a", AAD, &mut one).unwrap();
        assert!(seal(&recipient.public(), &context(1), b"b", AAD, &mut one).is_err());
        assert!(OutboundNonceLedger::new(0).is_err());
        assert!(InboundReplayLedger::new(0).is_err());
        assert!(seal(
            &recipient.public(),
            &context(2),
            b"x",
            &vec![0; MAX_EXTERNAL_AAD_BYTES + 1],
            &mut outbound_ledger()
        )
        .is_err());
        assert!(seal(
            &recipient.public(),
            &context(3),
            &vec![0; MAX_PLAINTEXT_BYTES + 1],
            AAD,
            &mut outbound_ledger()
        )
        .is_err());
        assert!(open(
            &recipient,
            &context(4),
            &vec![0; MAX_COSE_BYTES + 1],
            AAD,
            &mut inbound_ledger()
        )
        .is_err());
    }

    #[test]
    fn prederived_outbound_burn_survives_inbound_epoch_retirement() {
        let key = [0x42; 32];
        let ctx = context(21);
        let mut outbound = outbound_ledger();
        let sealed = seal_prederived(&key, SUITE, &ctx, b"block", AAD, &mut outbound).unwrap();
        let mut inbound = inbound_ledger();
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
        assert!(seal_prederived(&key, SUITE, &ctx, b"again", AAD, &mut outbound).is_err());
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
    }

    #[test]
    fn compatibility_wrappers_validate_before_encoding_or_crypto() {
        let recipient = generate_recipient(SUITE).unwrap();
        let oversized_aad = vec![0u8; MAX_EXTERNAL_AAD_BYTES + 1];
        assert!(seal_to_recipient(&recipient.public(), b"legacy", &oversized_aad, 0, 0).is_err());
        assert!(open_from_recipient(&recipient, &[], &oversized_aad, 0, 0).is_err());

        let mut malformed = recipient.public();
        malformed.eks.push(vec![0u8; 32]);
        assert!(seal_to_recipient(&malformed, b"legacy", AAD, 0, 0).is_err());
    }
}
