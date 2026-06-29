//! Hybrid multi-KEM key agreement ("HyKEM") — #551 / S0 of epic #550.
//!
//! A component-pluggable hybrid Key Encapsulation Mechanism whose combined shared
//! secret is **secure as long as at least one component KEM remains unbroken**
//! (the standard "secure-if-one-holds" combiner property). This is the
//! confidentiality counterpart to the hybrid EdDSA + ML-DSA-65 *signature* stack
//! in [`crate::crypto::cose_sign`].
//!
//! # Construction
//!
//! A [`SuiteId`] fixes an **ordered** list of component KEMs. Encapsulation runs
//! every component against the recipient's per-component encapsulation key,
//! collects each `(ciphertext, shared_secret)`, and folds them into a single
//! 32-byte secret with a **KMAC256 transcript-binding combiner** (NIST SP
//! 800-185), generalizing the X-Wing combiner
//! (`SHA3-256(ss_M ‖ ss_X ‖ ct_X ‖ pk_X ‖ label)`,
//! draft-connolly-cfrg-xwing-kem) to N components:
//!
//! ```text
//! K  = KMAC256_DOMAIN_KEY                       (fixed 32-byte domain key)
//! S  = "HyKEM-v1-combiner" ‖ 0x00 ‖ suite_id    (KMAC customization string)
//! X  = be16(suite_id) ‖ be32(n)
//!      ‖  Σ_i lp(ss_i)                           (ALL component shared secrets)
//!      ‖  Σ_i lp(ct_i)                           (ALL component ciphertexts)
//!      ‖  Σ_i lp(ek_i)                           (ALL recipient encap keys)
//! secret = KMAC256(K, X, L=256 bits, S)
//! ```
//! where `lp(x) = be32(len(x)) ‖ x` and the component order is the suite's.
//!
//! ## Why secure-if-one-holds
//!
//! KMAC256 is a PRF keyed by a fixed domain key. **Every** component shared
//! secret is an input block, so the output is pseudorandom to an adversary who is
//! missing *any single* `ss_i` — breaking N-1 components (learning their secrets)
//! does not let them predict the output while one `ss_i` stays unknown. Including
//! every `ct_i` and recipient `ek_i` binds the full transcript (X-Wing binds the
//! X25519 `ct`/`pk` for exactly this reason), so a component's ciphertext cannot
//! be swapped without changing the key.
//!
//! ## Why suite-bound (extensibility without collisions)
//!
//! The `suite_id` is folded into **both** the KMAC customization string `S` and
//! the message `X`. A future suite that appends a third (e.g. non-lattice) leg is
//! a **new `SuiteId`** with a distinct id, so its combined secret is in a
//! separate domain and can never collide with the 2-KEM suite's — the trait and
//! combiner never name a concrete KEM, so adding a leg is additive (epic #550 §3).
//! Suite selection is a pinned policy decision; there is no in-band negotiation.
//!
//! # Forward secrecy
//!
//! The X25519 leg is modeled as DHKEM (RFC 9180 §4.1): each encapsulation uses a
//! fresh **ephemeral** X25519 key, so that leg is forward-secret. The ML-KEM leg
//! encapsulates to the recipient's (possibly long-lived) key; forward secrecy of
//! the *suite* therefore rests on the ephemeral X25519 leg unless the ML-KEM key
//! is itself ephemeral/rotated (handled by the caller — streams use ephemeral
//! ML-KEM keys per #554; the envelope uses rotated `#mesh-kem` prekeys per #552).
//!
//! # Wasm
//!
//! All components are pure Rust (`x25519-dalek`, `ml-kem`, `tiny-keccak`) and
//! compile to `wasm32-unknown-unknown`.
//!
//! # Scope (S0)
//!
//! This module is the primitive + Rust wire types only. The Cap'n Proto schema
//! wiring of [`HybridKemMaterial`] into `RequestEnvelope` / `StreamInfo` /
//! `SignedEnvelope` is deferred to S3 (#554) and S4 (#555).

use anyhow::{bail, Result};
use rand::rngs::OsRng;
use rand::RngCore;
use tiny_keccak::{Hasher, Kmac};
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};
use zeroize::Zeroizing;

use crate::crypto::pq;

// ============================================================================
// Component / suite identifiers
// ============================================================================

/// Stable wire identifier for a single KEM component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum KemId {
    /// X25519 modeled as DHKEM (RFC 9180 §4.1). `ek`=32B, `ct`=32B, `ss`=32B.
    X25519 = 0x0001,
    /// ML-KEM-768 (FIPS 203). `ek`=1184B, `ct`=1088B, `ss`=32B.
    MlKem768 = 0x0002,
}

impl KemId {
    /// Stable u16 wire id.
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Parse from the u16 wire id.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0001 => Some(Self::X25519),
            0x0002 => Some(Self::MlKem768),
            _ => None,
        }
    }

    /// The component implementation for this id.
    fn component(self) -> &'static dyn KemComponent {
        match self {
            Self::X25519 => &X25519Kem,
            Self::MlKem768 => &MlKem768Kem,
        }
    }
}

/// A pinned hybrid-KEM suite: an ordered set of component KEMs plus a stable id
/// that is cryptographically bound into the combiner.
///
/// Adding a suite (e.g. a future `…-McEliece6960119`) is a single new variant —
/// no change to [`KemComponent`] or the combiner. Suite choice is policy-pinned
/// (epic #550 principle: crypto-agility = pinned suite, no in-band downgrade).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SuiteId {
    /// X25519 + ML-KEM-768 — the 2-KEM baseline (#551).
    HyKemX25519MlKem768,
}

impl SuiteId {
    /// Stable string id, bound into the KMAC customization string `S`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HyKemX25519MlKem768 => "HyKEM-X25519-MLKEM768",
        }
    }

    /// Stable u16 wire id, bound into the combiner message `X` and the
    /// [`HybridKemMaterial`] wire encoding.
    pub fn as_u16(self) -> u16 {
        match self {
            Self::HyKemX25519MlKem768 => 0x0001,
        }
    }

    /// Parse from the u16 wire id.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0001 => Some(Self::HyKemX25519MlKem768),
            _ => None,
        }
    }

    /// The **ordered** component list. Order is part of the suite definition and
    /// is covered by the combiner — never reorder without minting a new suite id.
    pub fn components(self) -> &'static [KemId] {
        match self {
            Self::HyKemX25519MlKem768 => &[KemId::X25519, KemId::MlKem768],
        }
    }
}

// ============================================================================
// Component trait + implementations (object-safe, byte-oriented)
// ============================================================================

/// One KEM component of a hybrid suite.
///
/// Object-safe and byte-oriented: the combiner iterates `&[&dyn KemComponent]`,
/// and components have heterogeneous key/ciphertext types, so all material
/// crosses the trait boundary as bytes. Shared secrets are wrapped in
/// [`Zeroizing`].
pub trait KemComponent {
    /// Stable identifier for this component.
    fn kem_id(&self) -> KemId;
    /// Recipient encapsulation-key length in bytes.
    fn ek_len(&self) -> usize;
    /// Ciphertext length in bytes.
    fn ct_len(&self) -> usize;
    /// Shared-secret length in bytes.
    fn ss_len(&self) -> usize;

    /// Generate a recipient keypair, returning `(decap_key_bytes, encap_key_bytes)`.
    fn generate_recipient_keypair(&self) -> Result<(Zeroizing<Vec<u8>>, Vec<u8>)>;

    /// Encapsulate to `recipient_ek`, returning `(ciphertext, shared_secret)`.
    fn encapsulate(&self, recipient_ek: &[u8]) -> Result<(Vec<u8>, Zeroizing<Vec<u8>>)>;

    /// Decapsulate `ct` with `decap_key`, returning the shared secret.
    fn decapsulate(&self, decap_key: &[u8], ct: &[u8]) -> Result<Zeroizing<Vec<u8>>>;
}

fn arr32(b: &[u8]) -> Result<[u8; 32]> {
    if b.len() != 32 {
        bail!("expected 32 bytes, got {}", b.len());
    }
    let mut a = [0u8; 32];
    a.copy_from_slice(b);
    Ok(a)
}

fn arr64(b: &[u8]) -> Result<[u8; 64]> {
    if b.len() != 64 {
        bail!("expected 64-byte ML-KEM seed, got {}", b.len());
    }
    let mut a = [0u8; 64];
    a.copy_from_slice(b);
    Ok(a)
}

/// X25519 as a DHKEM (RFC 9180 §4.1): `Encap` picks a fresh ephemeral key, the
/// ciphertext is the ephemeral public key, and the shared secret is the X25519
/// DH output. The ephemeral key gives this leg forward secrecy.
pub struct X25519Kem;

impl KemComponent for X25519Kem {
    fn kem_id(&self) -> KemId {
        KemId::X25519
    }
    fn ek_len(&self) -> usize {
        32
    }
    fn ct_len(&self) -> usize {
        32
    }
    fn ss_len(&self) -> usize {
        32
    }

    fn generate_recipient_keypair(&self) -> Result<(Zeroizing<Vec<u8>>, Vec<u8>)> {
        let sk = StaticSecret::random_from_rng(OsRng);
        let pk = PublicKey::from(&sk);
        Ok((
            Zeroizing::new(sk.to_bytes().to_vec()),
            pk.to_bytes().to_vec(),
        ))
    }

    fn encapsulate(&self, recipient_ek: &[u8]) -> Result<(Vec<u8>, Zeroizing<Vec<u8>>)> {
        let recipient_pk = PublicKey::from(arr32(recipient_ek)?);
        let eph = EphemeralSecret::random_from_rng(OsRng);
        let eph_pub = PublicKey::from(&eph);
        let ss = eph.diffie_hellman(&recipient_pk);
        // Reject the all-zero shared secret from a low-order recipient key. The
        // combiner mixes in the other legs, but failing closed here is cleaner.
        if !ss.was_contributory() {
            bail!("X25519 KEM: non-contributory shared secret (low-order point)");
        }
        Ok((
            eph_pub.to_bytes().to_vec(),
            Zeroizing::new(ss.to_bytes().to_vec()),
        ))
    }

    fn decapsulate(&self, decap_key: &[u8], ct: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
        let sk = StaticSecret::from(arr32(decap_key)?);
        let eph_pub = PublicKey::from(arr32(ct)?);
        let ss = sk.diffie_hellman(&eph_pub);
        if !ss.was_contributory() {
            bail!("X25519 KEM: non-contributory shared secret (low-order point)");
        }
        Ok(Zeroizing::new(ss.to_bytes().to_vec()))
    }
}

/// ML-KEM-768 (FIPS 203) component. The recipient decapsulation key is carried
/// as its 64-byte FIPS 203 seed (`d ‖ z`).
pub struct MlKem768Kem;

impl KemComponent for MlKem768Kem {
    fn kem_id(&self) -> KemId {
        KemId::MlKem768
    }
    fn ek_len(&self) -> usize {
        1184
    }
    fn ct_len(&self) -> usize {
        1088
    }
    fn ss_len(&self) -> usize {
        32
    }

    fn generate_recipient_keypair(&self) -> Result<(Zeroizing<Vec<u8>>, Vec<u8>)> {
        let mut seed = [0u8; 64];
        OsRng.fill_bytes(&mut seed);
        let dk = pq::ml_kem_decaps_from_seed(&seed);
        let ek_bytes = pq::ml_kem_ek_of_dk(&dk);
        Ok((Zeroizing::new(seed.to_vec()), ek_bytes))
    }

    fn encapsulate(&self, recipient_ek: &[u8]) -> Result<(Vec<u8>, Zeroizing<Vec<u8>>)> {
        let ek = pq::ml_kem_ek_from_bytes(recipient_ek)?;
        let (ct, ss) = pq::ml_kem_encapsulate(&ek);
        Ok((ct, Zeroizing::new(ss.to_vec())))
    }

    fn decapsulate(&self, decap_key: &[u8], ct: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
        let seed = arr64(decap_key)?;
        let dk = pq::ml_kem_decaps_from_seed(&seed);
        let ss = pq::ml_kem_decapsulate(&dk, ct)?;
        Ok(Zeroizing::new(ss.to_vec()))
    }
}

// ============================================================================
// KMAC256 transcript-binding combiner
// ============================================================================

/// Fixed 32-byte KMAC256 domain key. A constant (not a secret) — the secrecy
/// comes from the component shared secrets in the message, per SP 800-185 KMAC.
const KMAC_DOMAIN_KEY: &[u8; 32] = b"hyprstream HyKEM v1 combiner key";
/// KMAC256 customization-string prefix (the suite id is appended).
const KMAC_CUSTOM_PREFIX: &[u8] = b"HyKEM-v1-combiner";

fn update_lp(k: &mut Kmac, b: &[u8]) {
    k.update(&(b.len() as u32).to_be_bytes());
    k.update(b);
}

/// The combiner, parameterized by the suite's string + u16 id so tests can prove
/// domain separation without a second registered suite.
fn combine_inner(
    suite_str: &str,
    suite_u16: u16,
    shared_secrets: &[Zeroizing<Vec<u8>>],
    ciphertexts: &[Vec<u8>],
    recipient_eks: &[&[u8]],
) -> Zeroizing<[u8; 32]> {
    let mut custom = Vec::with_capacity(KMAC_CUSTOM_PREFIX.len() + 1 + suite_str.len());
    custom.extend_from_slice(KMAC_CUSTOM_PREFIX);
    custom.push(0x00);
    custom.extend_from_slice(suite_str.as_bytes());

    let mut k = Kmac::v256(KMAC_DOMAIN_KEY, &custom);
    k.update(&suite_u16.to_be_bytes());
    k.update(&(shared_secrets.len() as u32).to_be_bytes());
    for ss in shared_secrets {
        update_lp(&mut k, ss);
    }
    for ct in ciphertexts {
        update_lp(&mut k, ct);
    }
    for ek in recipient_eks {
        update_lp(&mut k, ek);
    }

    let mut out = [0u8; 32];
    k.finalize(&mut out);
    Zeroizing::new(out)
}

fn combine(
    suite: SuiteId,
    shared_secrets: &[Zeroizing<Vec<u8>>],
    ciphertexts: &[Vec<u8>],
    recipient_eks: &[&[u8]],
) -> Zeroizing<[u8; 32]> {
    combine_inner(
        suite.as_str(),
        suite.as_u16(),
        shared_secrets,
        ciphertexts,
        recipient_eks,
    )
}

// ============================================================================
// Suite-identified wire material + recipient keys
// ============================================================================

/// One component's encapsulated ciphertext, tagged with its [`KemId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KemShare {
    pub kem_id: KemId,
    pub bytes: Vec<u8>,
}

/// The full set of per-component ciphertexts for a hybrid encapsulation, tagged
/// with the suite. Replaces the legacy single fixed-ML-KEM-ciphertext field; the
/// Cap'n Proto field wiring lands in S3/S4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridKemMaterial {
    pub suite_id: SuiteId,
    pub shares: Vec<KemShare>,
}

impl HybridKemMaterial {
    /// Canonical length-prefixed encoding:
    /// `be16(suite_id) ‖ be16(n) ‖ Σ_i (be16(kem_id) ‖ be32(len) ‖ bytes)`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.suite_id.as_u16().to_be_bytes());
        out.extend_from_slice(&(self.shares.len() as u16).to_be_bytes());
        for s in &self.shares {
            out.extend_from_slice(&s.kem_id.as_u16().to_be_bytes());
            out.extend_from_slice(&(s.bytes.len() as u32).to_be_bytes());
            out.extend_from_slice(&s.bytes);
        }
        out
    }

    /// Parse and validate the canonical encoding. Rejects unknown suites/kem-ids,
    /// truncation, trailing bytes, and any share that does not match the suite's
    /// component at that position (fixed order).
    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut r = Reader { buf, pos: 0 };
        let suite_id = SuiteId::from_u16(r.u16()?)
            .ok_or_else(|| anyhow::anyhow!("unknown hybrid-KEM suite id"))?;
        let n = r.u16()? as usize;
        let comps = suite_id.components();
        if n != comps.len() {
            bail!(
                "share count {n} does not match suite {} ({} components)",
                suite_id.as_str(),
                comps.len()
            );
        }
        let mut shares = Vec::with_capacity(n);
        for (i, &want) in comps.iter().enumerate() {
            let kem_id =
                KemId::from_u16(r.u16()?).ok_or_else(|| anyhow::anyhow!("unknown kem id"))?;
            if kem_id != want {
                bail!(
                    "share {i} kem id {:?} does not match suite component {:?}",
                    kem_id,
                    want
                );
            }
            let len = r.u32()? as usize;
            let bytes = r.take(len)?.to_vec();
            if bytes.len() != want.component().ct_len() {
                bail!(
                    "share {i} ({:?}) ciphertext length {} != expected {}",
                    want,
                    bytes.len(),
                    want.component().ct_len()
                );
            }
            shares.push(KemShare { kem_id, bytes });
        }
        if r.pos != buf.len() {
            bail!("trailing bytes after hybrid-KEM material");
        }
        Ok(Self { suite_id, shares })
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| anyhow::anyhow!("hybrid-KEM material truncated"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// A recipient's hybrid keypair: per-component decapsulation keys (secret) and
/// encapsulation keys (public), in suite-component order.
pub struct RecipientKeypair {
    pub suite_id: SuiteId,
    dks: Vec<Zeroizing<Vec<u8>>>,
    eks: Vec<Vec<u8>>,
}

/// A recipient's public hybrid material: per-component encapsulation keys in
/// suite order. This is what an encapsulator needs (published via `#mesh-kem`
/// for the envelope, or sent in the stream handshake — wiring in S3/S4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipientPublic {
    pub suite_id: SuiteId,
    pub eks: Vec<Vec<u8>>,
}

impl RecipientKeypair {
    /// The public half (encapsulation keys), for handing to encapsulators.
    pub fn public(&self) -> RecipientPublic {
        RecipientPublic {
            suite_id: self.suite_id,
            eks: self.eks.clone(),
        }
    }

    /// The per-component decapsulation-key bytes, suite order (e.g. to persist /
    /// publish via `#mesh-kem`).
    pub fn decap_keys(&self) -> &[Zeroizing<Vec<u8>>] {
        &self.dks
    }
}

impl RecipientPublic {
    /// Canonical length-prefixed encoding of the recipient's per-component
    /// encapsulation keys, mirroring [`HybridKemMaterial::encode`]:
    /// `be16(suite_id) ‖ be16(n) ‖ Σ_i (be16(kem_id) ‖ be32(len) ‖ ek_bytes)`.
    ///
    /// This is what the stream client sends as its ephemeral public material
    /// (`RequestEnvelope.clientKemPublic`, S3 #554), and what `#mesh-kem`
    /// publishes for the envelope path (S4 #555).
    pub fn encode(&self) -> Vec<u8> {
        let comps = self.suite_id.components();
        let mut out = Vec::new();
        out.extend_from_slice(&self.suite_id.as_u16().to_be_bytes());
        out.extend_from_slice(&(self.eks.len() as u16).to_be_bytes());
        for (i, ek) in self.eks.iter().enumerate() {
            // Component order is the suite's; tag each ek with its kem id so the
            // decoder can fail closed on a reordered / wrong-suite encoding.
            out.extend_from_slice(&comps[i].as_u16().to_be_bytes());
            out.extend_from_slice(&(ek.len() as u32).to_be_bytes());
            out.extend_from_slice(ek);
        }
        out
    }

    /// Parse + validate the canonical encoding. Rejects unknown suites/kem-ids,
    /// wrong component count or order, wrong ek length, truncation, or trailing
    /// bytes — fail-closed against malformed client public material.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut r = Reader { buf, pos: 0 };
        let suite_id = SuiteId::from_u16(r.u16()?)
            .ok_or_else(|| anyhow::anyhow!("unknown hybrid-KEM suite id"))?;
        let n = r.u16()? as usize;
        let comps = suite_id.components();
        if n != comps.len() {
            bail!(
                "ek count {n} does not match suite {} ({} components)",
                suite_id.as_str(),
                comps.len()
            );
        }
        let mut eks = Vec::with_capacity(n);
        for (i, &want) in comps.iter().enumerate() {
            let kem_id =
                KemId::from_u16(r.u16()?).ok_or_else(|| anyhow::anyhow!("unknown kem id"))?;
            if kem_id != want {
                bail!(
                    "ek {i} kem id {:?} does not match suite component {:?}",
                    kem_id,
                    want
                );
            }
            let len = r.u32()? as usize;
            let bytes = r.take(len)?.to_vec();
            if bytes.len() != want.component().ek_len() {
                bail!(
                    "ek {i} ({:?}) length {} != expected {}",
                    want,
                    bytes.len(),
                    want.component().ek_len()
                );
            }
            eks.push(bytes);
        }
        if r.pos != buf.len() {
            bail!("trailing bytes after recipient public material");
        }
        Ok(Self { suite_id, eks })
    }
}

// ============================================================================
// High-level hybrid KEM API
// ============================================================================

/// Generate a fresh recipient keypair for `suite` (one component keypair each).
pub fn generate_recipient(suite: SuiteId) -> Result<RecipientKeypair> {
    let mut dks = Vec::new();
    let mut eks = Vec::new();
    for &kem in suite.components() {
        let (dk, ek) = kem.component().generate_recipient_keypair()?;
        dks.push(dk);
        eks.push(ek);
    }
    Ok(RecipientKeypair {
        suite_id: suite,
        dks,
        eks,
    })
}

/// Encapsulate to a recipient's public material, returning the wire
/// [`HybridKemMaterial`] (the per-component ciphertexts) and the combined 32-byte
/// shared secret.
pub fn encapsulate_to(
    recipient: &RecipientPublic,
) -> Result<(HybridKemMaterial, Zeroizing<[u8; 32]>)> {
    let suite = recipient.suite_id;
    let comps = suite.components();
    if recipient.eks.len() != comps.len() {
        bail!(
            "recipient has {} encap keys, suite {} needs {}",
            recipient.eks.len(),
            suite.as_str(),
            comps.len()
        );
    }

    let mut shares = Vec::with_capacity(comps.len());
    let mut sss = Vec::with_capacity(comps.len());
    let mut cts = Vec::with_capacity(comps.len());
    let mut eks: Vec<&[u8]> = Vec::with_capacity(comps.len());

    for (i, &kem) in comps.iter().enumerate() {
        let ek = &recipient.eks[i];
        if ek.len() != kem.component().ek_len() {
            bail!(
                "recipient component {:?} encap key length {} != expected {}",
                kem,
                ek.len(),
                kem.component().ek_len()
            );
        }
        let (ct, ss) = kem.component().encapsulate(ek)?;
        cts.push(ct.clone());
        shares.push(KemShare {
            kem_id: kem,
            bytes: ct,
        });
        sss.push(ss);
        eks.push(ek.as_slice());
    }

    let secret = combine(suite, &sss, &cts, &eks);
    Ok((
        HybridKemMaterial {
            suite_id: suite,
            shares,
        },
        secret,
    ))
}

/// Decapsulate hybrid material with a recipient keypair, recovering the same
/// combined 32-byte shared secret. Fails closed on suite/share mismatch.
pub fn decapsulate(
    recipient: &RecipientKeypair,
    material: &HybridKemMaterial,
) -> Result<Zeroizing<[u8; 32]>> {
    let suite = recipient.suite_id;
    if material.suite_id != suite {
        bail!(
            "material suite {} != recipient suite {}",
            material.suite_id.as_str(),
            suite.as_str()
        );
    }
    let comps = suite.components();
    if material.shares.len() != comps.len() || recipient.dks.len() != comps.len() {
        bail!("share/key count does not match suite {}", suite.as_str());
    }

    let mut sss = Vec::with_capacity(comps.len());
    let mut cts = Vec::with_capacity(comps.len());
    let mut eks: Vec<&[u8]> = Vec::with_capacity(comps.len());

    for (i, &kem) in comps.iter().enumerate() {
        let share = &material.shares[i];
        if share.kem_id != kem {
            bail!(
                "share {i} kem {:?} != suite component {:?}",
                share.kem_id,
                kem
            );
        }
        let ss = kem
            .component()
            .decapsulate(&recipient.dks[i], &share.bytes)?;
        sss.push(ss);
        cts.push(share.bytes.clone());
        eks.push(recipient.eks[i].as_slice());
    }

    Ok(combine(suite, &sss, &cts, &eks))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ---- per-component roundtrips ----

    fn component_roundtrip(c: &dyn KemComponent) {
        let (dk, ek) = c.generate_recipient_keypair().unwrap();
        assert_eq!(ek.len(), c.ek_len());
        let (ct, ss_enc) = c.encapsulate(&ek).unwrap();
        assert_eq!(ct.len(), c.ct_len());
        assert_eq!(ss_enc.len(), c.ss_len());
        let ss_dec = c.decapsulate(&dk, &ct).unwrap();
        assert_eq!(
            &*ss_enc, &*ss_dec,
            "component encap/decap shared secret must agree"
        );
    }

    #[test]
    fn x25519_component_roundtrip() {
        component_roundtrip(&X25519Kem);
    }

    #[test]
    fn mlkem768_component_roundtrip() {
        component_roundtrip(&MlKem768Kem);
    }

    #[test]
    fn mlkem_dk_seed_roundtrip() {
        // The decap key serializes to a 64-byte seed and back.
        let (dk, _ek) = crate::crypto::pq::ml_kem_generate_keypair();
        let seed = crate::crypto::pq::ml_kem_dk_to_seed(&dk).expect("generated key is seed-backed");
        let dk2 = crate::crypto::pq::ml_kem_decaps_from_seed(&seed);
        assert_eq!(
            crate::crypto::pq::ml_kem_ek_of_dk(&dk),
            crate::crypto::pq::ml_kem_ek_of_dk(&dk2)
        );
    }

    // ---- full suite roundtrip ----

    #[test]
    fn suite_roundtrip_secrets_agree() {
        let suite = SuiteId::HyKemX25519MlKem768;
        let recipient = generate_recipient(suite).unwrap();
        let (material, ss_enc) = encapsulate_to(&recipient.public()).unwrap();
        assert_eq!(material.suite_id, suite);
        assert_eq!(material.shares.len(), 2);
        let ss_dec = decapsulate(&recipient, &material).unwrap();
        assert_eq!(
            *ss_enc, *ss_dec,
            "suite encap/decap shared secret must agree"
        );
    }

    #[test]
    fn distinct_encapsulations_differ() {
        // Fresh ephemeral X25519 each time → different secret & material.
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let (m1, s1) = encapsulate_to(&recipient.public()).unwrap();
        let (m2, s2) = encapsulate_to(&recipient.public()).unwrap();
        assert_ne!(*s1, *s2);
        assert_ne!(m1, m2);
    }

    // ---- secure-if-one-holds (combiner) ----

    fn z(v: &[u8]) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(v.to_vec())
    }

    #[test]
    fn combiner_changes_when_any_single_ss_changes() {
        // The "secure-if-one-holds" surface: perturbing ANY single component's
        // shared secret (others fixed) changes the combined key — so an adversary
        // missing one ss cannot predict the output.
        let suite = SuiteId::HyKemX25519MlKem768;
        let cts = vec![vec![9u8; 32], vec![8u8; 1088]];
        let ek_refs: Vec<&[u8]> = vec![&[7u8; 32], &[6u8; 1184]];

        let base = combine(suite, &[z(&[1u8; 32]), z(&[2u8; 32])], &cts, &ek_refs);
        // change ONLY component 0's ss
        let only_first = combine(suite, &[z(&[0xff; 32]), z(&[2u8; 32])], &cts, &ek_refs);
        // change ONLY component 1's ss
        let only_second = combine(suite, &[z(&[1u8; 32]), z(&[0xff; 32])], &cts, &ek_refs);

        assert_ne!(
            *base, *only_first,
            "changing component 0 ss must change the key"
        );
        assert_ne!(
            *base, *only_second,
            "changing component 1 ss must change the key"
        );
    }

    #[test]
    fn combiner_is_deterministic() {
        let suite = SuiteId::HyKemX25519MlKem768;
        let sss = vec![z(&[3u8; 32]), z(&[4u8; 32])];
        let cts = vec![vec![1u8; 32], vec![2u8; 1088]];
        let ek_refs: Vec<&[u8]> = vec![&[5u8; 32], &[6u8; 1184]];
        let a = combine(suite, &sss, &cts, &ek_refs);
        let b = combine(suite, &sss, &cts, &ek_refs);
        assert_eq!(*a, *b);
    }

    #[test]
    fn combiner_binds_ciphertexts_and_eks() {
        let suite = SuiteId::HyKemX25519MlKem768;
        let sss = vec![z(&[3u8; 32]), z(&[4u8; 32])];
        let cts = vec![vec![1u8; 32], vec![2u8; 1088]];
        let cts2 = vec![vec![1u8; 32], vec![0u8; 1088]]; // one ct byte differs
        let ek_refs: Vec<&[u8]> = vec![&[5u8; 32], &[6u8; 1184]];
        let ek_refs2: Vec<&[u8]> = vec![&[5u8; 32], &[7u8; 1184]]; // one ek differs
        let base = combine(suite, &sss, &cts, &ek_refs);
        assert_ne!(
            *base,
            *combine(suite, &sss, &cts2, &ek_refs),
            "ct must be bound"
        );
        assert_ne!(
            *base,
            *combine(suite, &sss, &cts, &ek_refs2),
            "ek must be bound"
        );
    }

    #[test]
    fn suite_domain_separation() {
        // Same secrets/cts/eks under two different suite ids → different keys.
        let sss = vec![z(&[3u8; 32]), z(&[4u8; 32])];
        let cts = vec![vec![1u8; 32], vec![2u8; 1088]];
        let ek_refs: Vec<&[u8]> = vec![&[5u8; 32], &[6u8; 1184]];
        let a = combine_inner("HyKEM-X25519-MLKEM768", 0x0001, &sss, &cts, &ek_refs);
        let b = combine_inner(
            "HyKEM-X25519-MLKEM768-McEliece6960119",
            0x0002,
            &sss,
            &cts,
            &ek_refs,
        );
        assert_ne!(
            *a, *b,
            "different suite id must domain-separate the combined key"
        );
    }

    // ---- tamper / failure ----

    #[test]
    fn tampered_ciphertext_changes_or_fails() {
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let (mut material, ss_enc) = encapsulate_to(&recipient.public()).unwrap();
        // flip a byte in the ML-KEM ciphertext (component 1)
        material.shares[1].bytes[0] ^= 0xff;
        // ML-KEM implicit rejection yields a *different* ss (no error), so the
        // combined secret must differ from the encapsulator's.
        let ss_dec = decapsulate(&recipient, &material).unwrap();
        assert_ne!(*ss_enc, *ss_dec);
    }

    #[test]
    fn wrong_recipient_fails_to_agree() {
        let r1 = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let r2 = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let (material, ss_enc) = encapsulate_to(&r1.public()).unwrap();
        let ss_dec = decapsulate(&r2, &material).unwrap();
        assert_ne!(*ss_enc, *ss_dec);
    }

    // ---- wire encode/decode ----

    #[test]
    fn material_encode_decode_roundtrip() {
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let (material, _) = encapsulate_to(&recipient.public()).unwrap();
        let enc = material.encode();
        let dec = HybridKemMaterial::decode(&enc).unwrap();
        assert_eq!(material, dec);
    }

    #[test]
    fn decode_rejects_malformed() {
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let (material, _) = encapsulate_to(&recipient.public()).unwrap();
        let enc = material.encode();
        // truncated
        assert!(HybridKemMaterial::decode(&enc[..enc.len() - 1]).is_err());
        // trailing byte
        let mut extra = enc.clone();
        extra.push(0);
        assert!(HybridKemMaterial::decode(&extra).is_err());
        // unknown suite id
        let mut bad_suite = enc.clone();
        bad_suite[0] = 0xff;
        bad_suite[1] = 0xff;
        assert!(HybridKemMaterial::decode(&bad_suite).is_err());
        // wrong ML-KEM ct length (shrink the declared len → trailing/῾len mismatch)
        let mut wrong_len = enc.clone();
        // component 0 share starts at offset 4 (suite u16 + count u16); its len u32
        // is at offset 6..10. Corrupt the second component instead is complex; just
        // assert a single-byte flip in the count breaks it.
        wrong_len[3] = 0x05;
        assert!(HybridKemMaterial::decode(&wrong_len).is_err());
    }

    // ---- KAT regression pin ----
    //
    // Pins the combiner output over fixed inputs so a future refactor cannot
    // silently change the construction. Value computed by this implementation.
    #[test]
    fn combiner_kat() {
        let sss = vec![z(&[0xaa; 32]), z(&[0xbb; 32])];
        let cts = vec![vec![0xcc; 32], vec![0xdd; 1088]];
        let ek_refs: Vec<&[u8]> = vec![&[0xee; 32], &[0x11; 1184]];
        let out = combine(SuiteId::HyKemX25519MlKem768, &sss, &cts, &ek_refs);
        let got = hex::encode(*out);
        // Pinned output of the KMAC256 combiner over the fixed inputs above. A
        // change here means the construction (KMAC key/customization, field order,
        // or length-prefixing) was altered — review before updating.
        const KAT: &str = "00f1efc789785e571aa6e985a9f2567cca39dfc7dc272b514380b212524ce031";
        assert_eq!(got, KAT, "combiner KAT changed — construction altered!");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod recipient_public_codec_tests {
    use super::*;

    #[test]
    fn recipient_public_roundtrip() {
        let kp = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let pubmat = kp.public();
        let enc = pubmat.encode();
        let dec = RecipientPublic::decode(&enc).unwrap();
        assert_eq!(dec, pubmat);
        assert_eq!(dec.encode(), enc, "encoding is canonical");
        // The decoded public actually works for encapsulation.
        let (material, _secret) = encapsulate_to(&dec).unwrap();
        assert_eq!(material.suite_id, SuiteId::HyKemX25519MlKem768);
    }

    #[test]
    fn recipient_public_rejects_malformed() {
        assert!(RecipientPublic::decode(&[]).is_err());
        assert!(RecipientPublic::decode(b"garbage bytes here").is_err());
        let kp = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        // Truncation.
        let mut enc = kp.public().encode();
        enc.truncate(enc.len() - 1);
        assert!(RecipientPublic::decode(&enc).is_err());
        // Trailing bytes.
        let mut enc2 = kp.public().encode();
        enc2.push(0xff);
        assert!(RecipientPublic::decode(&enc2).is_err());
    }

    #[test]
    fn recipient_public_rejects_wrong_ek_length() {
        let kp = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let mut bad = kp.public();
        bad.eks[0] = vec![0u8; 31]; // wrong X25519 ek length (must be 32)
        assert!(RecipientPublic::decode(&bad.encode()).is_err());
    }
}
